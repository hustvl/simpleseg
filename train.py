
import argparse
import datetime
import logging
import os
from tabnanny import check
import numpy as np
import torch
import time
from torch.distributed.distributed_c10d import get_rank
import torch.nn as nn
from torchvision.transforms.functional import resize
import torch.nn.functional as F

from apex.parallel import DistributedDataParallel

# NOTE: new
from simpleseg.config import get_cfg_defaults

from simpleseg.evaluation.evaluation import ConfusionMatrix
from simpleseg.utils.utils import get_world_size, save_checkpoint, load_checkpoint, is_main_process, mkdir, reduce_dict, output_evaluation_results

from simpleseg.datasets import build_data_loader
from simpleseg.datasets.dataset_meta import DATASET_META
from simpleseg.models import build_segmentor
from simpleseg.optimizer import build_optimizer, build_poly_lr_scheduler, PolyLR
from simpleseg.utils.metric_logger import MetricLogger
from simpleseg.utils.logger import setup_logger


def _init_dist_envs(local_rank):
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://')


def train_segmentor(cfg, data_loader, test_loader, model, optimizer, scheduler, writter, output_dir, arguments):
    logger = logging.getLogger("simpleseg.trainer")
    start_iter = arguments['start_iter']
    iters_per_epoch = arguments['iters_per_epoch']
    scheduler_epoch = cfg.SOLVER.SCHEDULER_EPOCH
    max_iters = cfg.SOLVER.MAX_ITERS
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    test_period = cfg.SOLVER.TEST_PERIOD
    meters = MetricLogger(" ")
    best_mIoU = 0
    best_iteration = 0
    # train_loss_meter = AverageMeter()
    model.train()
    # epoch = 0
    start_time = time.time()
    for iteration, data in enumerate(data_loader, start_iter):
        iteration = iteration + 1
        data_time = time.time() - start_time
        # step lr

        images, targets, _ = data
        images = images.cuda()
        targets = targets.cuda()
        # print(torch.unique(targets))
        loss_dict = model(images, targets)
        # print(pred.shape)
        # loss_dict = criterion(pred, targets)
        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        torch.cuda.empty_cache()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - start_time
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iters - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if iteration % 20 == 0:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: [{iter}/{max_iter}]",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    max_iter=max_iters,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if iteration % checkpoint_period == 0:
            if is_main_process():
                save_path = os.path.join(
                    output_dir, 'model_iter_{}.pth'.format(iteration))
                save_checkpoint(
                    save_path, model, iteration,
                    optimizer=optimizer,
                    scheduler=scheduler)
                logger.info("save checkpoint to: {}".format(save_path))

        if iteration % test_period == 0 and test_loader is not None:

            del images
            del targets
            del losses
            torch.cuda.empty_cache()

            # if is_main_process():
            logger.info("testing")
            # ? distributed
            metrics = test_segmenter(
                cfg, test_loader, model, writter, arguments, True)
            if is_main_process():
                logger.info(
                    "evaluation results for iteration {}".format(iteration))
                output_evaluation_results(
                    metrics, dataset_name=cfg.DATASET.NAME, logger=logger)
                miou = metrics["mIoU"]
                if best_mIoU < miou:
                    best_mIoU = miou
                    # best_iteration = iteration
                    extra = dict()
                    extra['mIoU'] = miou
                    extra['aAcc'] = metrics['class_accuracy']
                    extra['mAcc'] = metrics['mAcc']
                    extra['iu'] = metrics['iu']
                    save_path = os.path.join(
                        output_dir, 'model_iter_best.pth')
                    save_checkpoint(
                        save_path, model, iteration,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        extra=extra)
                    logger.info("save checkpoint to: {}".format(save_path))
            model.train()
        if not scheduler_epoch:
            scheduler.step()
        else:
            if iteration % iters_per_epoch == 0:
                scheduler.step()
        start_time = time.time()
    logger.info("finished training")
    logger.info("best mIoU: {:.2f}".format(best_mIoU*100))


def test_segmenter(cfg, data_loader, model, writter, arguments, is_distributed=True):
    logger = logging.getLogger("simpleseg.trainer")
    num_classes = cfg.DATASET.NUM_CLASSES
    model.eval()
    # hist = 0
    conf_matrix = ConfusionMatrix(num_classes)
    for idx, data in enumerate(data_loader):
        # data[0], data[1] -> images, targets,
        #
        images, targets, img_meta = data
        images = images.cuda()
        # targets = targets.cuda()

        with torch.no_grad():
            pred = model(images)

        assert not (img_meta[0]["pad"] and pred.size(
            0) > 1), "only support batch size=1 for pad and resize."
        if img_meta[0]["pad"]:
            img_size = img_meta[0]["img_size"]
            # remove padding
            pred = pred[:, :, :img_size[0], :img_size[1]]
            # resize to original size
            pred = F.interpolate(
                pred, size=img_meta[0]["ori_size"], mode='bilinear', align_corners=False)

        pred = pred.detach().argmax(dim=1).cpu()

        if idx % 20 == 0:
            if is_main_process():
                logger.info(
                    "[testing] {cur}/{all}".format(cur=idx+1, all=len(data_loader)))

        conf_matrix.update(pred.flatten(), targets.cpu().flatten())
        torch.cuda.empty_cache()

    # merge between devices
    if is_distributed:
        conf_matrix.synchronize_from_all()
    eval_results = conf_matrix.summarize()
    return eval_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument(
        '--cfg',
        type=str,
        required=True
    )
    parser.add_argument(
        '--local_rank',
        default=0,
        type=int,
        help='parameter used by apex library'
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg = get_cfg_defaults()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enable = True

    # if get_world_size() > 0

    _init_dist_envs(args.local_rank)
    print("initialized distributed environment, rank:", args.local_rank)

    # output dir
    if is_main_process():
        mkdir(cfg.OUTPUT_DIR)
    # _init_log(cfg.OUTPUT_DIR)
    logger = setup_logger("simpleseg", cfg.OUTPUT_DIR, get_rank())
    logger.info(cfg)
    # extra arguments
    arguments = dict()
    # count iters per epoch
    arguments['iters_per_epoch'] = 0
    # decay w.r.t epochs not iters
    arguments['scheuler_epoch'] = not cfg.SOLVER.ITERATION_BASED

    # dataset information
    dataset_meta = DATASET_META[cfg.DATASET.NAME]
    ignore_index = dataset_meta['ignore_index']
    num_classes = dataset_meta['num_classes']

    # build loss
    # criterion = build_segmentation_loss(cfg, ignore_index)

    # build model
    model = build_segmentor(cfg, num_classes, ignore_index)
    # print(type(model))
    if is_main_process():
        logger.info(model)
    # optimizer and scheduler
    max_iters = cfg.SOLVER.MAX_ITERS
    iters_per_epoch = dataset_meta['train_samples'] // cfg.SOLVER.BATCH_SIZE
    arguments['iters_per_epoch'] = iters_per_epoch
    if cfg.SOLVER.ITERATION_BASED and max_iters == 0:
        max_iterations = iters_per_epoch * cfg.SOLVER.MAX_EPOCHS
        # update max iters in config
        cfg.defrost()
        cfg.DATASET.NUM_CLASSES = num_classes
        cfg.SOLVER.MAX_ITERS = max_iterations
        cfg.freeze()

    scheduler_iterations = cfg.SOLVER.MAX_ITERS

    max_epochs = cfg.SOLVER.MAX_EPOCHS
    if max_epochs == 0:
        max_epochs = (scheduler_iterations +
                      iters_per_epoch - 1) // iters_per_epoch
    if cfg.SOLVER.SCHEDULER_EPOCH:
        scheduler_iterations = max_epochs

    # assert arguments['iters_per_epoch'] != 0 and not cfg.SOLVER.ITERATION_BASED
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = build_optimizer(cfg, model)
    # print(optimizer)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, [5000, 5000], 0.1)
    # scheduler = PolyLR(optimizer, cfg.SOLVER.POWER, scheduler_iterations)
    scheduler = build_poly_lr_scheduler(cfg, optimizer, scheduler_iterations)

    # load checkpoint
    checkpoint_path = cfg.MODEL.WEIGHTS
    # print(checkpoint_path)
    if len(checkpoint_path) > 0 and os.path.exists(checkpoint_path):
        
        logger.info("loading checkpoints from: {}".format(checkpoint_path))
        data = load_checkpoint(checkpoint_path)
        state_dict = model.state_dict()
        new_state_dict={k:v if v.size()==state_dict[k].size()  else  state_dict[k] for k,v in zip(state_dict.keys(), data['model'].values())}
        model.load_state_dict(new_state_dict)
        if 'iteration' in data:
            start_iter = data['iteration']
        else:
            start_iter = 0
        if 'optimizer' in data:
            optimizer.load_state_dict(data['optimizer'])
        if 'scheduler' in data:
            scheduler.load_state_dict(data['scheduler'])
    else:
        start_iter = 0
    arguments['start_iter'] = start_iter
    # build dataloaders
    train_loader = build_data_loader(
        cfg, is_distributed=True, is_train=True, start_iter=start_iter)
    val_loader = build_data_loader(cfg, is_distributed=True, is_train=False)
    # print(len(train_loader))
    # wrap into distributed parallel model
    model = DistributedDataParallel(model)
    torch.cuda.empty_cache()
    # logger.info("training ")
    # load state dict
    logger.info("Start training.")
    train_segmentor(
        cfg=cfg,
        data_loader=train_loader,
        test_loader=val_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        writter=None,
        output_dir=cfg.OUTPUT_DIR,
        arguments=arguments
    )
