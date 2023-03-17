import collections
import os
import time
import json
import datetime
import argparse
import gc
import numpy as np
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
from omni_res.utils.checkpoint import save_checkpoint, load_checkpoint, auto_resume_helper
from eval_omni import validate
from process_omni import *
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


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

def compute_unsupervised_loss(seg_logit_student, seg_pred_teacher, seg_logit_teacher, zero_th, low_th, high_th):
    negative_index_th = torch.logical_and(seg_logit_teacher<low_th, seg_logit_teacher>zero_th)
    positive_index_th = torch.logical_and(seg_logit_teacher>high_th, seg_pred_teacher>0.99)
    index_th = torch.logical_or(negative_index_th, positive_index_th)

    non_positive_index = seg_pred_teacher.sum(dim=(1,2))==0
    negative_index_th[non_positive_index] = False
    positive_index_th[non_positive_index] = False
    index_th[non_positive_index] = False

    negative_ratio = negative_index_th.sum()/(seg_logit_teacher.shape[0]*seg_logit_teacher.shape[1]*seg_logit_teacher.shape[2])
    positive_ratio = positive_index_th.sum()/(seg_logit_teacher.shape[0]*seg_logit_teacher.shape[1]*seg_logit_teacher.shape[2])

    loss_seg = nn.BCELoss(reduction='sum')(seg_logit_student[index_th], seg_pred_teacher[index_th]) / len(seg_logit_student)

    if torch.isnan(loss_seg):
        loss_seg = torch.Tensor([0]).cuda()
    return loss_seg, negative_ratio, positive_ratio

def train_one_epoch(cfg, model, model_teacher, optimizer, scheduler, data_loader, omni_loader, scalar, epoch, rank, ema=None):
    start_unsup_epoch = cfg.train.burning_in_epochs
    start_semi_step = start_unsup_epoch * len(data_loader)                # 设置开始半监督的step
    # sigmoid_thold = [0.35, 0.4, 0.5, 0.6, 0.7, 0.75]                    # 设置验证性能时的像素置信度阈值
    zero_logit_thold = cfg.train.data.logit_thold[0]
    low_logit_thold, high_logit_thold = cfg.train.data.logit_thold[1], cfg.train.data.logit_thold[2]

    omni_label = cfg.train.data.omni_label
    process_method = ''
    if process_method == 'box_logit':
        high_logit_tholds = [0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]
        th_decay_epochs = [8, 7, 6, 6, 6, 6, 6]
        assert sum(th_decay_epochs) == cfg.train.epochs-start_unsup_epoch,\
         f"thold decay epochs {sum(th_decay_epochs)}, unsup epochs {cfg.train.epochs-start_unsup_epoch}, imbalance!"
        th_decay_epochs = [start_unsup_epoch+sum(th_decay_epochs[:i+1]) for i in range(len(th_decay_epochs))]
        for h in range(len(high_logit_tholds)):
            if epoch<th_decay_epochs[h]:
                high_logit_thold = high_logit_tholds[h]
                break
    logger.info('high logit thold: ' + str(high_logit_thold))

    model_teacher.cuda()
    model.train()
    data_loader.sampler.set_epoch(epoch)
    omni_loader.sampler.set_epoch(epoch)
    
    num_iters = len(data_loader)
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    losses_l = AverageMeter('LossSup', ':.4f')
    losses_u = AverageMeter('LossUnsup', ':.4f')

    ppr = AverageMeter('Postive Pesudo Ratio', ':.3f')
    npr = AverageMeter('Negative Pesudo Ratio', ':.3f')

    meters = [batch_time, data_time, losses, losses_l, losses_u, ppr, npr]
    
    if omni_label == 'point':
        mask_box_iou = AverageMeter('Box mIoU', ':6.2f')
        mask_box_iou_5 = AverageMeter('BoxIoU@0.5', ':6.2f')
        mask_box_iou_6 = AverageMeter('BoxIoU@0.6', ':6.2f')
        mask_box_iou_7 = AverageMeter('BoxIoU@0.7', ':6.2f')
        meters.append(mask_box_iou)
        meters.append(mask_box_iou_5)
        meters.append(mask_box_iou_6)
        meters.append(mask_box_iou_7)
    elif omni_label == 'box':
        mask_box_ratio = AverageMeter('Mask/Box Ratio', ':6.2f')
        logit_in_box = AverageMeter('Logit in Box', ':6.2f')
        meters.append(mask_box_ratio)
        meters.append(logit_in_box)
    elif omni_label == 'point_distance':
        point_dist = AverageMeter('LossDist', ':6.2f')
        meters.append(point_dist)

    meters_dict = {meter.name: meter for meter in meters}
    
    start = time.time()
    end = time.time()

    loader_l_iter = iter(data_loader)
    loader_u_iter = iter(omni_loader)
    assert len(loader_l_iter) == len(loader_u_iter), f"labeled data {len(loader_l_iter)} unlabeled data {len(loader_u_iter)}, imbalance!"

    for idx in range(len(loader_l_iter)):
        data_time.update(time.time() - end)
        step = len(loader_l_iter)*epoch + idx

        # labeled_data
        (ref_l, img_l, seg_label, det_label, _ , _) = loader_l_iter.next()
        ref_l = ref_l.cuda(non_blocking=True)
        img_l = img_l.cuda(non_blocking=True)
        seg_label = seg_label.cuda(non_blocking=True)
        det_label = det_label.cuda(non_blocking=True)

        # supervised loss
        seg_logit, _ = f(model, img_l, ref_l, scalar=scalar)
        loss = loss_fn(seg_logit, seg_label.squeeze(1))
        
        if step >= start_semi_step:
            if step==start_semi_step:
                model_teacher.load_state_dict(model.module.state_dict())

            # unlabeled_data
            (images, boxes, refs, ref_u, img_u, ref_u_q, img_u_q, seg_label_q, det_label_q, info_img) = loader_u_iter.next()
            ref_u = ref_u.cuda(non_blocking=True)
            img_u = img_u.cuda(non_blocking=True)
            ref_u_q = ref_u_q.cuda(non_blocking=True)
            img_u_q = img_u_q.cuda(non_blocking=True)

            # unsupervised loss
            seg_logit_u, _ = f(model, img_u_q, ref_u_q, scalar=scalar)

            # generate pseudo labels
            with torch.no_grad():
                model_teacher.eval()
                seg_logit_teacher, _ = f(model_teacher, img_u, ref_u, scalar=scalar)
                seg_logit_teacher = seg_logit_teacher.detach()

            # generate positive pixels for pesudo labels
            seg_pesudo_label = torch.zeros((seg_logit_teacher.shape[0], seg_logit_teacher.shape[1], seg_logit_teacher.shape[2])).float().cuda(non_blocking=True)
            seg_pesudo_label[seg_logit_teacher>high_logit_thold] = 1.0

            if omni_label == 'box':
                pn_ratio_avg, logit_avg = process_box_labels(images, boxes, refs, seg_pesudo_label, seg_logit_teacher, det_label_q, seg_label_q, info_img, logger)
            elif omni_label == 'point':
                mask_box_ious, mask_box_ious_5, mask_box_ious_6, mask_box_ious_7 = process_point_labels(seg_pesudo_label, det_label_q, seg_label_q)
            elif omni_label == 'point_distance':
                point_y, point_x, mean_y, mean_x = process_point_distance(seg_pesudo_label, det_label_q, seg_logit_u)
                loss_d = ((point_y.to(mean_y.device)-mean_y)**2 + (point_x.to(mean_x.device)-mean_x)**2).mean()/2  

            loss_u, negative_ratio, positive_ratio = compute_unsupervised_loss(seg_logit_u, seg_pesudo_label, seg_logit_teacher, zero_logit_thold, low_logit_thold, high_logit_thold)
            loss_u = cfg.train.omni_weight * loss_u
            loss += loss_u
            if omni_label == 'point_distance':
                loss += loss_d

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
        
        if step>=start_semi_step:
            ema_decay = 0.9996
            with torch.no_grad():
                student_state_dict = model.module.state_dict()
                teacher_state_dict = model_teacher.state_dict()
                new_state_dict = collections.OrderedDict()
                for state_dict_key in student_state_dict.keys():
                    new_state_dict[state_dict_key] = ema_decay*teacher_state_dict[state_dict_key] + (1-ema_decay)*student_state_dict[state_dict_key]
                model_teacher.load_state_dict(new_state_dict)
        
        losses.update(loss.item(), len(img_l))
        if step>=start_semi_step:
            if omni_label == 'point_distance':
                losses_l.update(loss.item()-loss_u.item()-loss_d.item(), len(img_l))
            else:
                losses_l.update(loss.item()-loss_u.item(), len(img_l))
            losses_u.update(loss_u.item(), len(img_u))
            # losses_p.update(loss_proj_term.item(), len(img_u))
            npr.update(negative_ratio.item()), len(img_u)
            ppr.update(positive_ratio.item()), len(img_u)

            if omni_label == 'point':
                mask_box_iou.update(mask_box_ious.item(), len(img_u))
                mask_box_iou_5.update(mask_box_ious_5.item(), len(img_u))
                mask_box_iou_6.update(mask_box_ious_6.item(), len(img_u))
                mask_box_iou_7.update(mask_box_ious_7.item(), len(img_u))
            elif omni_label == 'box':
                mask_box_ratio.update(pn_ratio_avg.item(), len(img_u))
                logit_in_box.update(logit_avg.item(), len(img_u))
            elif omni_label == 'point_distance':
                point_dist.update(loss_d.item(), len(img_u))
            '''
            for i in range(len(sigmoid_thold)):
                mask_ap[i].update(seg_iou_per_batch[i]/len(img_u), len(img_u))
            if logit_num_per_batch!=0: 
                mask_pesudo.update(seg_logit_per_batch/logit_num_per_batch, logit_num_per_batch)
            '''
        
        reduce_meters(meters_dict, rank, cfg)
        
        if idx % cfg.train.log_period == 0 or idx==len(data_loader)-1:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_iters - idx)
            if step < start_semi_step:
                logger.info(
                    f'Train: [{epoch}/{cfg.train.epochs-1}][{idx}/{num_iters-1}]  '
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.7f}  '
                    f'Time {batch_time.val:.4f} ({batch_time.avg_reduce:.4f})  '
                    f'Loss {losses.avg_reduce:.4f}  '
                    f'Mem {memory_used:.0f}MB')
            else:
                # mask_str = [f'MaskIou({sigmoid_thold[i]}) {mask_ap[i].avg:.2f}  ' for i in range(len(sigmoid_thold))]
                if omni_label == 'point':
                    point_log = f'BoxIoU {mask_box_iou.avg_reduce:.2f}  BoxIoU@0.5 {mask_box_iou_5.avg_reduce:.2f}  BoxIoU@0.6 {mask_box_iou_6.avg_reduce:.2f}  BoxIoU@0.7 {mask_box_iou_7.avg_reduce:.2f}  '
                else:
                    point_log = ''
                
                if omni_label == 'box':
                    box_log = f'Mask/Box Ratio {mask_box_ratio.avg_reduce:.2f}  Logit in Box {logit_in_box.avg_reduce:.2f}  '
                else:
                    box_log = ''
                
                if omni_label == 'point_distance':
                    pointd_log = f'LossDist {point_dist.avg_reduce:.2f}  '
                else:
                    pointd_log = ''
                
                logger.info(
                    f'Train: [{epoch}/{cfg.train.epochs-1}][{idx}/{num_iters-1}]  '
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.7f}  '
                    f'Time {batch_time.val:.4f} ({batch_time.avg_reduce:.4f})  '
                    f'Loss {losses.avg_reduce:.4f}  '
                    f'LossSup {losses_l.avg_reduce:.4f}  '
                    f'LossUnsup {losses_u.avg_reduce:.4f}  '
                    # f'LossProj {losses_p.avg:.4f}  '
                    +point_log+box_log+pointd_log+
                    # f'PesudoIoU {mask_pesudo.avg:.2f}  '
                    # +''.join(mask_str)+
                    f'Positive Pesudo Ratio {ppr.avg_reduce:.3f}  '
                    f'Negative Pesudo Ratio {npr.avg_reduce:.3f}  '
                    f'Mem {memory_used:.0f}MB')
        
        del ref_l, img_l, seg_label, det_label, loss, seg_logit
        if step>=start_semi_step:
            del ref_u, img_u, ref_u_q, img_u_q, seg_label_q, det_label_q, \
                loss_u, seg_logit_u, seg_logit_teacher

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % 5000 == 0 and idx != 0:
            yield False
            model.train()
    
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    yield True


def main(cfg):
    global best_seg_acc
    best_seg_acc = 0.

    # build training dataset and dataloader
    cfg.dataset.split = "train"
    
    datasets = cfg.dataset.dataset.split(',')
    if len(datasets)==1:
        cfg.dataset.size = (len(json.load(open(cfg.dataset.sup_ann_path[cfg.dataset.dataset], 'r'))['train'])+len(json.load(open(cfg.dataset.omni_ann_path[cfg.dataset.dataset], 'r'))['train']))
        train_set = instantiate(cfg.dataset)
        train_loader = build_train_loader(
            cfg,
            train_set,
            shuffle=True,
            drop_last=True,
        )

        cfg.dataset.label = "omni"
        omni_set = instantiate(cfg.dataset)
        omni_loader = build_train_loader(
            cfg, 
            omni_set, 
            shuffle=True,
            drop_last=True,
        )

        # build validation dataset and dataloader
        cfg.dataset.label = None
        cfg.dataset.split = "val"
        val_set = instantiate(cfg.dataset)
        val_loader = build_test_loader(
            cfg, 
            val_set,
            shuffle=False,
            drop_last=False,
        )
    else:
        cfg.dataset.size = len(json.load(open(cfg.dataset.sup_ann_path[datasets[0]], 'r'))['train']) \
            +len(json.load(open(cfg.dataset.omni_ann_path[datasets[1]], 'r'))['train']) \
            +len(json.load(open(cfg.dataset.omni_ann_path[datasets[1]], 'r'))['val'])
        
        cfg.dataset.dataset = 'refcoco_merge'
        train_set = instantiate(cfg.dataset)
        train_loader = build_train_loader(
            cfg,
            train_set,
            shuffle=True,
            drop_last=True,
        )

        cfg.dataset.dataset = 'vg'
        cfg.dataset.label = "omni"
        cfg.dataset.split = "train+val"
        omni_set = instantiate(cfg.dataset)
        omni_loader = build_train_loader(
            cfg, 
            omni_set, 
            shuffle=True,
            drop_last=True,
        )
    
        # build validation dataset and dataloader
        cfg.dataset.label = None
        cfg.dataset.split = "val"
        cfg.dataset.dataset = 'refcoco_merge'
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

    # Teacher model
    model_teacher = instantiate(cfg.model)
    for p in model_teacher.parameters():
        p.requires_grad = False

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
        start_epoch = load_checkpoint(cfg, model_without_ddp, optimizer, scheduler, logger, model_teacher)

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

    gc.collect()
    for epoch in range(start_epoch, cfg.train.epochs):
        if cfg.train.ema.enabled and ema is None:
            ema = EMA(model, cfg.train.ema.alpha, cfg.train.ema.buffer_ema)
        iter_train = train_one_epoch(cfg, model, model_teacher, optimizer, scheduler, train_loader, omni_loader, scalar, epoch, dist.get_rank(), ema)
        flag = next(iter_train)
        mask_ap = validate(cfg, model, val_loader, epoch, val_set.ix_to_token, logger, dist.get_rank(), ema=ema)
        # if is_main_process():
        #     save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger, model_teacher, save_last=False)
        while not flag:
            flag = next(iter_train)
            mask_ap = validate(cfg, model, val_loader, epoch, val_set.ix_to_token, logger, dist.get_rank(), ema=ema)
            # if is_main_process():
            #     save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger, model_teacher, save_last=False)
        mask_ap = validate(cfg, model, val_loader, epoch, val_set.ix_to_token, logger, dist.get_rank(), ema=ema)
        # save checkpoints
        if epoch % cfg.train.save_period == 0 or epoch == (cfg.train.epochs - 1):
            logger.info(f"saving checkpoints......")
            if is_main_process():
                if ema is not None:
                    ema.apply_shadow()
                # periodically save checkpoint
                save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger, model_teacher)
                # save best checkpoint
                if mask_ap > best_seg_acc:
                    save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger, model_teacher, seg_best=True)
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
