import os
import time
import json
import argparse
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from omni_res.config import instantiate, LazyConfig
from omni_res.datasets.dataloader import build_test_loader
from omni_res.models.utils import mask_processing, mask_iou
from omni_res.utils.env import seed_everything
from omni_res.utils.logger import create_logger
from omni_res.utils.metric import AverageMeter
from omni_res.utils.distributed import is_main_process, reduce_meters


def validate(cfg, model, data_loader, epoch, ix_to_token, logger, rank, prefix='Val', ema=None):
    if ema is not None:
        ema.apply_shadow()
    model.eval()

    sigmoid_threshold = (0.4, 0.5, 0.6, 0.7)
    accuracy_threshold = (0.5, 0.6, 0.7, 0.8, 0.9)

    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')

    oiou_log = [AverageMeter('oIoU@'+str(i), ':.2f') for i in sigmoid_threshold]
    miou_log = [AverageMeter('mIoU@'+str(i), ':.2f') for i in sigmoid_threshold]
    acc_log = [AverageMeter('Acc@'+str(i), ':.2f') for i in accuracy_threshold]
    
    meters = [batch_time, data_time, *oiou_log, *miou_log, *acc_log]
    meters_dict = {meter.name: meter for meter in meters}
    
    cum_I, cum_U = [0]*len(sigmoid_threshold), [0]*len(sigmoid_threshold)
    acc = [0]*len(accuracy_threshold)
    num = 0

    with torch.no_grad():
        end = time.time()
        for idx, (ref_iter, image_iter, _ , _ , mask_id, info_iter) in enumerate(data_loader):
            ref_iter = ref_iter.cuda(non_blocking=True)
            image_iter = image_iter.cuda(non_blocking=True)

            seg_logit, _ = model(image_iter, ref_iter)
            seg_pred_list =  [(seg_logit>i).float().cpu().numpy() for i in sigmoid_threshold]

            # predictions to ground-truth
            seg_iou_list = [[] for i in range(len(sigmoid_threshold))]
            for i in range(len(seg_pred_list[0])):
                if cfg.dataset.dataset=='refcoco_merge':
                    mask_gt = np.load(os.path.join(os.path.join(cfg.dataset.mask_path[cfg.dataset.dataset], 'refcoco'),'%d.npy' % mask_id[i]))
                else:
                    mask_gt = np.load(os.path.join(cfg.dataset.mask_path[cfg.dataset.dataset],'%d.npy' % mask_id[i]))
                
                for j in range(len(sigmoid_threshold)):
                    mask_pred = mask_processing(seg_pred_list[j][i], info_iter[i]).astype(np.uint8)
                    single_seg_iou, single_seg_ap, I, U = mask_iou(mask_gt, mask_pred, accuracy_threshold)
                    seg_iou_list[j].append(single_seg_iou)

                    cum_I[j] += I
                    cum_U[j] += U

                    if sigmoid_threshold[j]==0.5:
                        for a in range(len(accuracy_threshold)):
                            acc[a] += single_seg_ap[accuracy_threshold[a]]
                num += 1
            seg_iou_list = [np.array(i).astype(np.float32) for i in seg_iou_list]

            for i in range(len(sigmoid_threshold)):
                oiou_log[i].update(cum_I[i]/cum_U[i]*100., seg_iou_list[0].shape[0])
                miou_log[i].update(seg_iou_list[i].mean()*100., seg_iou_list[0].shape[0])
            
            for i in range(len(accuracy_threshold)):
                acc_log[i].update(acc[i]/num*100., seg_iou_list[0].shape[0])
            
            reduce_meters(meters_dict, rank, cfg)

            if (idx % cfg.train.log_period == 0 or idx==(len(data_loader)-1)):
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                # mask_str = [f'MaskIou({sigmoid_threshold[i]}) {mask_ap[i].avg:.2f}  ' for i in range(len(sigmoid_threshold))]
                logger.info(
                    f'Evaluation on {prefix}: [{idx}/{len(data_loader)-1}]  '
                    f'Time {batch_time.val:.3f} ({batch_time.avg_reduce:.3f})  '
                    # +''.join(mask_str)+
                    f'Mem {memory_used:.0f}MB')
            
            del ref_iter, image_iter, mask_id, info_iter, seg_logit, seg_pred_list, seg_iou_list
            
            batch_time.update(time.time() - end)
            end = time.time()

        mask_str = [f' * oIoU({sigmoid_threshold[i]}) {oiou_log[i].avg_reduce:.2f}' for i in range(len(sigmoid_threshold))] + ['\n']
        mask_str += [f' * mIoU({sigmoid_threshold[i]}) {miou_log[i].avg_reduce:.2f}' for i in range(len(sigmoid_threshold))] + ['\n']
        mask_str += [f' * Acc({accuracy_threshold[i]}) {acc_log[i].avg_reduce:.2f}' for i in range(len(accuracy_threshold))] + ['\n']
        logger.info(''.join(mask_str))

    if ema is not None:
        ema.restore()
    return acc_log[0].avg_reduce


def main(cfg):
    global best_seg_acc
    best_seg_acc = 0.

    loaders = []
    prefixs = ['val']
    
    datasets = cfg.dataset.dataset.split(',')
    if len(datasets)==1:
        # build training dataset and dataloader
        cfg.dataset.split = "train"
        cfg.dataset.size = len(json.load(open(cfg.dataset.sup_ann_path[cfg.dataset], 'r'))['train'])
        train_set = instantiate(cfg.dataset)

        # build single or multi-datasets for validation
        
        cfg.dataset.split = "val"
        val_set = instantiate(cfg.dataset)
        val_loader = build_test_loader(cfg, val_set, shuffle=False, drop_last=False)
        loaders.append(val_loader)
        
        if cfg.dataset.dataset == 'refcoco' or cfg.dataset.dataset == 'refcoco+':
            cfg.dataset.split = "testA"
            testA_dataset = instantiate(cfg.dataset)
            testA_loader = build_test_loader(cfg, testA_dataset, shuffle=False, drop_last=False)

            cfg.dataset.split = "testB"
            testB_dataset = instantiate(cfg.dataset)
            testB_loader = build_test_loader(cfg, testB_dataset, shuffle=False, drop_last=False)
            prefixs.extend(['testA','testB'])
            loaders.extend([testA_loader,testB_loader])
        else:
            cfg.dataset.split = "test"
            test_dataset=instantiate(cfg.dataset)
            test_loader=build_test_loader(cfg, test_dataset, shuffle=False, drop_last=False)
            prefixs.append('test')
            loaders.append(test_loader)
    else:
        cfg.dataset.size = len(json.load(open(cfg.dataset.sup_ann_path[datasets[0]], 'r'))['train'])
        cfg.dataset.dataset = 'refcoco_merge'
        train_set = instantiate(cfg.dataset)
    
        # build validation dataset and dataloader
        
        def set_tokenizer(src, target):
            target.token_to_ix = src.token_to_ix
            target.ix_to_token = src.ix_to_token
            target.pretrained_emb = src.pretrained_emb
            target.max_token = src.max_token
            target.token_size = src.token_size

        cfg.dataset.dataset = 'refcoco'

        cfg.dataset.split = "val"
        val_set = instantiate(cfg.dataset)
        set_tokenizer(train_set, val_set)
        val_loader = build_test_loader(cfg, val_set, shuffle=False, drop_last=False)

        cfg.dataset.split = "testA"
        testA_dataset = instantiate(cfg.dataset)
        set_tokenizer(train_set, testA_dataset)
        testA_loader = build_test_loader(cfg, testA_dataset, shuffle=False, drop_last=False)

        cfg.dataset.split = "testB"
        testB_dataset = instantiate(cfg.dataset)
        set_tokenizer(train_set, testB_dataset)
        testB_loader = build_test_loader(cfg, testB_dataset, shuffle=False, drop_last=False)

        prefixs.extend(['testA', 'testB'])
        loaders.extend([testA_loader, testB_loader])
        # else:
        #     cfg.dataset.split = "test"
        #     test_dataset=instantiate(cfg.dataset)
        #     test_loader=build_test_loader(cfg, test_dataset, shuffle=False, drop_last=False)
        #     prefixs.append('test')
        #     loaders.append(test_loader)
        
    # build model
    cfg.model.language_encoder.pretrained_emb = train_set.pretrained_emb
    cfg.model.language_encoder.token_size = train_set.token_size
    model = instantiate(cfg.model)

    # build optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    cfg.optim.params = params
    optimizer = instantiate(cfg.optim)

    torch.cuda.set_device(dist.get_rank())
    model = DistributedDataParallel(model.cuda(), device_ids=[dist.get_rank()], find_unused_parameters=True)
    model_without_ddp = model.module

    if is_main_process():
        total_params = sum([param.nelement() for param in model.parameters()])
        trainable_params = sum([param.nelement() for param in model.parameters() if param.requires_grad])
        logger.info(str(model))
        logger.info("Number of all params: %.2fM" % (total_params / 1e6))
        logger.info("Number of trainable params: %.2fM" % (trainable_params / 1e6))

    checkpoint = torch.load(cfg.train.resume_path, map_location=lambda storage, loc: storage.cuda() )
    model_without_ddp.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for data_loader, prefix in zip(loaders, prefixs):
        validate(cfg=cfg, model=model, data_loader=data_loader, epoch=0, ix_to_token=val_set.ix_to_token, logger=logger, rank=dist.get_rank(), prefix=prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SimREC")
    parser.add_argument('--config', type=str, required=True, default='./config/simrec_refcoco_scratch.py')
    parser.add_argument('--eval-weights', type=str, required=True, default='')
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

    # Refine cfg for evaluation
    cfg.train.resume_path = args.eval_weights
    logger.info(f"Running evaluation from specific checkpoint {cfg.train.resume_path}......")

    if is_main_process():
        path = os.path.join(cfg.train.output_dir, "config.yaml")
        LazyConfig.save(cfg, path)

    main(cfg)
