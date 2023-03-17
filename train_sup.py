import os
import gc
import time
import json
import datetime
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from omni_res.config import LazyConfig, instantiate
from omni_res.datasets.dataloader import build_train_loader, build_test_loader
from omni_res.scheduler.build import build_lr_scheduler
from omni_res.utils.model_ema import EMA
from omni_res.utils.logger import create_logger
from omni_res.utils.env import seed_everything
from omni_res.utils.metric import AverageMeter
from omni_res.utils.distributed import reduce_meters, is_main_process, cleanup_distributed
from omni_res.utils.checkpoint import save_checkpoint_sup, load_checkpoint_sup, auto_resume_helper
from eval_sup import validate
from process_omni import *

def f(model, image, ref, scalar=None):
    if scalar is not None:
        with torch.cuda.amp.autocast():
            mask_logit, mask_feat = model(image, ref)
    else:
        mask_logit, mask_feat = model(image, ref)
    return mask_logit, mask_feat

def loss_fn(seg_pred, seg_label):
    loss_seg = nn.BCELoss(reduction='sum')(seg_pred, seg_label) / len(seg_pred)
    return loss_seg

def train_one_epoch(cfg, model, optimizer, scheduler, data_loader, scalar, epoch, rank, ema=None):
    model.train()
    data_loader.sampler.set_epoch(epoch)
    
    num_iters = len(data_loader)
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')

    meters = [batch_time, data_time, losses]
    meters_dict = {meter.name: meter for meter in meters}
    
    start = time.time()
    end = time.time()

    loader_l_iter = iter(data_loader)
    for idx in range(len(loader_l_iter)):
        data_time.update(time.time() - end)

        # labeled_data
        (ref_l, img_l, seg_label, det_label, _ , _) = loader_l_iter.next()
        ref_l = ref_l.cuda(non_blocking=True)
        img_l = img_l.cuda(non_blocking=True)
        seg_label = seg_label.cuda(non_blocking=True)
        det_label = det_label.cuda(non_blocking=True)

        # supervised loss
        seg_logit, _ = f(model, img_l, ref_l, scalar=scalar)
        loss = loss_fn(seg_logit, seg_label.squeeze(1))
        
        optimizer.zero_grad()
        if scalar is not None:
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            if cfg.train.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.train.clip_grad_norm
                )
            scalar.update()
        else:
            loss.backward()
            if cfg.train.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)
            optimizer.step()
        scheduler.step()
        
        if ema is not None:
            ema.update_params()
        
        losses.update(loss.item(), len(img_l))
        reduce_meters(meters_dict, rank, cfg)
        
        if idx % cfg.train.log_period == 0 or idx==len(data_loader)-1:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_iters - idx)
            logger.info(
                f'Train: [{epoch}/{cfg.train.epochs-1}][{idx}/{num_iters-1}]  '
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.7f}  '
                f'Time {batch_time.val:.4f} ({batch_time.avg_reduce:.4f})  '
                f'Loss {losses.avg_reduce:.4f}  '
                f'Mem {memory_used:.0f}MB')

        del ref_l, img_l, seg_label, det_label, loss, seg_logit

        batch_time.update(time.time() - end)
        end = time.time()
    
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def main(cfg):
    global best_seg_acc
    best_seg_acc = 0.

    # build training dataset and dataloader
    cfg.dataset.split = "train"
    cfg.dataset.size = len(json.load(open(cfg.dataset.sup_ann_path[cfg.dataset.dataset], 'r'))['train'])*cfg.dataset.times

    train_set = instantiate(cfg.dataset)
    train_loader = build_train_loader(
        cfg,
        train_set,
        shuffle=True,
        drop_last=True,
    )
    
    # build validation dataset and dataloader
    cfg.dataset.split = "val"
    val_set = instantiate(cfg.dataset)
    val_loader = build_test_loader(
        cfg, 
        val_set,
        shuffle=False,
        drop_last=False,
    )

    # build model
    cfg.model.language_encoder.pretrained_emb = train_set.pretrained_emb
    cfg.model.language_encoder.token_size = train_set.token_size
    model = instantiate(cfg.model)

    # build optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    cfg.optim.params = params
    optimizer = instantiate(cfg.optim)

    # model ema
    ema = None

    torch.cuda.set_device(dist.get_rank())
    if cfg.train.sync_bn.enabled:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Converted model to use Synchronized BatchNorm.")
    model = DistributedDataParallel(model.cuda(), device_ids=[dist.get_rank()], find_unused_parameters=True, broadcast_buffers=True)
    model_without_ddp = model.module

    if is_main_process():
        total_params = sum([param.nelement() for param in model.parameters()])
        trainable_params = sum([param.nelement() for param in model.parameters() if param.requires_grad])
        logger.info(str(model))
        logger.info("Number of all params: %.2fM" % (total_params / 1e6))
        logger.info("Number of trainable params: %.2fM" % (trainable_params / 1e6))

    scheduler = build_lr_scheduler(cfg, optimizer, len(train_loader))

    start_epoch = 0

    if cfg.train.auto_resume.enabled:
        resume_file = auto_resume_helper(cfg.train.output_dir)
        if resume_file:
            if cfg.train.resume_path:
                logger.warning(f"auto-resume changing resume file from {cfg.train.resume_path} to {resume_file}")
            cfg.train.resume_path = resume_file
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {cfg.train.output_dir}, ignoring auto resume')

    if cfg.train.resume_path:
        start_epoch = load_checkpoint_sup(cfg, model_without_ddp, optimizer, scheduler, logger)

    if os.path.isfile(cfg.train.vl_pretrain_weight):
        checkpoint = torch.load(cfg.train.vl_pretrain_weight, map_location=lambda storage, loc: storage.cuda())
        logger.warning("loading pretrained weight for finetuning, ignoring resume training, reset start epoch to 0")
        msg = model_without_ddp.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(msg)
        start_epoch = 0
        logger.info("==> loaded checkpoint from {}\n".format(cfg.train.vl_pretrain_weight) +
                    "==> epoch: {} lr: {} ".format(checkpoint['epoch'], checkpoint['lr']))

    if cfg.train.amp.enabled:
        assert torch.__version__ >= '1.6.0', "Automatic Mixed Precision training only supported in PyTorch-1.6.0 or higher"
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    for epoch in range(start_epoch, cfg.train.epochs):
        if cfg.train.ema.enabled and ema is None:
            ema = EMA(model, cfg.train.ema.alpha, cfg.train.ema.buffer_ema)
        train_one_epoch(cfg, model, optimizer, scheduler, train_loader, scalar, epoch, dist.get_rank(), ema)
        clean_cache()
        mask_ap = validate(cfg, model, val_loader, epoch, val_set.ix_to_token, logger, dist.get_rank(), ema=ema)
        clean_cache()

        # save checkpoints
        if epoch % cfg.train.save_period == 0 or epoch == (cfg.train.epochs - 1):
            logger.info(f"saving checkpoints......")
            if is_main_process():
                if ema is not None:
                    ema.apply_shadow()
                # periodically save checkpoint
                save_checkpoint_sup(cfg, epoch, model_without_ddp, optimizer, scheduler, logger)
                # save best checkpoint
                if mask_ap > best_seg_acc:
                    save_checkpoint_sup(cfg, epoch, model_without_ddp, optimizer, scheduler, logger, seg_best=True)
                    best_seg_acc = mask_ap
                if ema is not None:
                    ema.restore()
            logger.info(f"checkpoints saved !!!")

    cleanup_distributed()

def clean_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SimREC")
    parser.add_argument('--config', type=str, required=True, default='./config/simrec_refcoco_scratch.py')
    parser.add_argument(
        "opts",
        help="""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    cfg = LazyConfig.load(args.config)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    # Environments setting
    seed_everything(cfg.train.seed)

    # Distributed setting
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend=cfg.train.ddp.backend,
        init_method=cfg.train.ddp.init_method, 
        world_size=world_size, 
        rank=rank
    )
    torch.distributed.barrier()

    # Path setting
    output_dir = cfg.train.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=cfg.train.output_dir, dist_rank=dist.get_rank())

    if is_main_process():
        path = os.path.join(cfg.train.output_dir, "config.yaml")
        LazyConfig.save(cfg, path)

    main(cfg)
