
import argparse
import datetime
import logging
import os
import numpy as np
import torch
import time
# from torch.distributed.distributed_c10d import get_rank
import torch.nn as nn
import torch.nn.functional as F

from apex.parallel import DistributedDataParallel

# NOTE: new
from simpleseg.config import get_cfg_defaults

from simpleseg.evaluation.evaluation import ConfusionMatrix
from simpleseg.utils.utils import load_checkpoint, is_main_process, mkdir, output_evaluation_results

from simpleseg.datasets import build_data_loader
from simpleseg.datasets.dataset_meta import DATASET_META
from simpleseg.models import build_segmentor
from simpleseg.utils.logger import setup_logger
from simpleseg.utils.utils import get_rank


def _init_dist_envs(local_rank):
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://')


def test_segmenter(cfg, data_loader, model, is_distributed=True, output_dir=None):
    logger = logging.getLogger("simpleseg.test")
    num_classes = cfg.DATASET.NUM_CLASSES
    model.eval()
    # hist = 0
    # measure speed
    warmup = 20
    latency = 0.0
    latency_counter = 0
    # measure accuracy
    conf_matrix = ConfusionMatrix(num_classes)
    for idx, data in enumerate(data_loader):
        images, targets, img_meta = data

        images = images.cuda()

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            pred = model(images)
        torch.cuda.synchronize()
        end = time.perf_counter() - start
        if idx >= warmup:
            latency += end
            latency_counter += 1

        assert not (img_meta[0]["pad"] and pred.size(
            0) > 1), "only support batch size=1 for pad and resize."
        if img_meta[0]["pad"]:
            img_size = img_meta[0]["img_size"]
            # remove padding
            pred = pred[:, :, :img_size[0], :img_size[1]]
            # resize to original size
            pred = F.interpolate(
                pred, size=img_meta[0]["ori_size"], mode='bilinear', align_corners=False)

        pred = pred.argmax(dim=1).cpu()

        if idx % 20 == 0:
            if is_main_process():
                logger.info(
                    "[testing] {cur}/{all}".format(cur=idx+1, all=len(data_loader)))
        # print(targets.shape, pred.shape)
        if targets.sum() > 0:
            conf_matrix.update(pred.flatten(), targets.cpu().flatten())
        # output to folders
        # print(img_meta)
        for i in range(pred.size(0)):
            prediction = pred[i]
            data_loader.dataset.output_format(
                prediction, img_meta[i], output_dir)

    time_per_image = 1.0 * latency / latency_counter
    fps = 1 / time_per_image

    # merge between devices
    if is_distributed:
        conf_matrix.synchronize_from_all()
    eval_results = conf_matrix.summarize()
    output_evaluation_results(eval_results, cfg.DATASET.NAME, logger)

    logger.info("inference speed: {:.3f} ms per image {:.2f} FPS".format(
        time_per_image * 1000, fps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument(
        '--cfg',
        type=str,
        required=True
    )
    parser.add_argument(
        '--ckpt',
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
    # change SyncBN to BN
    cfg.MODEL.NORM = "BN"
    cfg.freeze()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enable = True

    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        _init_dist_envs(args.local_rank)
        print("initialized distributed environment, rank:", args.local_rank)

    # output dir
    output_dir = os.path.join(cfg.OUTPUT_DIR, 'test')
    if is_main_process():
        mkdir(output_dir)
        mkdir(os.path.join(output_dir, 'pred'))
        mkdir(os.path.join(output_dir, 'vis'))
    logger = setup_logger(
        "simpleseg", cfg.OUTPUT_DIR, get_rank(), filename='test_log.txt')
    logger.info(cfg)

    dataset_meta = DATASET_META[cfg.DATASET.NAME]
    ignore_index = dataset_meta['ignore_index']
    num_classes = dataset_meta['num_classes']

    # build model
    model = build_segmentor(cfg, num_classes, ignore_index)
    # print(type(model))
    if is_main_process():
        logger.info(model)

    checkpoint_path = args.ckpt
    assert os.path.exists(
        checkpoint_path), "checkpoint file {} doesn't exist!".format(checkpoint_path)
    data = load_checkpoint(checkpoint_path)
    model.load_state_dict(data['model'])

    test_loader = build_data_loader(
        cfg, is_distributed=distributed, is_train=False)
    # print(len(train_loader))
    # wrap into distributed parallel model
    if distributed:
        model = DistributedDataParallel(model)
    torch.cuda.empty_cache()
    # logger.info("training ")
    # load state dict
    logger.info("Start testing.")
    test_segmenter(
        cfg=cfg,
        data_loader=test_loader,
        model=model,
        is_distributed=distributed,
        output_dir=output_dir
    )
