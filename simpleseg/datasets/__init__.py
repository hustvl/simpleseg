from posixpath import join
import torch
from torch.distributed.distributed_c10d import get_rank
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
import torchvision.transforms as standard_transforms

from simpleseg.datasets.cityscapes import Cityscapes
from simpleseg.datasets.camvid import CamVid
from simpleseg.datasets.ade20k import ADE20K
import simpleseg.datasets.transforms.joint_transforms as joint_transforms
import simpleseg.datasets.transforms.transforms as extended_transforms
from simpleseg.datasets.dataset_meta import DATASET_META
from simpleseg.utils.utils import get_world_size
from simpleseg.datasets.sampler import DistributedSampler, IterationBasedBatchSampler


def collate_fn(inputs):
    transposed_data = list(zip(*inputs))
    images = torch.stack(transposed_data[0], 0)
    targets = torch.stack(transposed_data[1], 0)
    img_metas = transposed_data[2]
    return images, targets, img_metas


def build_train_joint_transform(cfg, ignore_index):
    if cfg.DATASET.TRANSFORMS.CROP_CATE_RATIO < 1.0:
        random_crop_op = joint_transforms.RandomCropV2
    else:
        random_crop_op = joint_transforms.RandomCropFix

    transform_list = [
        joint_transforms.RandomResizeFix(
            scale_min=cfg.DATASET.TRANSFORMS.SCALES[0],
            scale_max=cfg.DATASET.TRANSFORMS.SCALES[1],
        ),
        random_crop_op(
            size=cfg.DATASET.TRANSFORMS.CROP_SIZE,
            pad_if_needed=True,
            label_fill=ignore_index,
            cat_max_ratio=cfg.DATASET.TRANSFORMS.CROP_CATE_RATIO
        ),
        joint_transforms.RandomHorizontallyFlip()]
    return joint_transforms.Compose(transform_list)


def build_train_transform(cfg):
    transform_list = []
    color_aug = cfg.DATASET.TRANSFORMS.COLOR_AUG
    transform_list += [
        extended_transforms.ColorJitter(
            brightness=color_aug[0],
            contrast=color_aug[1],
            saturation=color_aug[2],
            hue=color_aug[3])
    ]
    blur = cfg.DATASET.TRANSFORMS.BLUR
    if blur == "bblur":
        transform_list += [extended_transforms.RandomBilateralBlur()]
    elif blur == "gblur":
        transform_list += [extended_transforms.RandomGaussianBlur()]

    transform_list += [
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD)
    ]
    return standard_transforms.Compose(transform_list)


def build_test_transform(cfg):
    transform_list = []
    # if cfg.TEST.INPUT_MODE == "resize":
    #     transform_list += [
    #         extended_transforms.ResizePad(cfg.TEST.SIZE[0], divisible_size=32)
    #     ]

    transform_list += [
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD)
    ]
    return standard_transforms.Compose(transform_list)


def build_data_loader(cfg, is_distributed=True, is_train=True, start_iter=0):
    """
    Setup Data Loaders[Currently supports Cityscapes, Mapillary and ADE20kin]
    input: argument passed by the user
    return:  training data loader, validation data loader loader,  train_set
    """
    dataset_name = cfg.DATASET.NAME
    data_path = cfg.DATASET.PATH
    num_gpus = get_world_size()
    batch_size = cfg.SOLVER.BATCH_SIZE // num_gpus
    if not is_train:
        batch_size = cfg.TEST.BATCH_SIZE // num_gpus
    # print(batch_size)
    shuffle = is_train
    num_workers = cfg.DATASET.NUM_WORKERS
    num_workers = num_workers if is_train else num_workers // 2
    # choose split
    split = cfg.DATASET.TRAIN if is_train else cfg.DATASET.TEST
    ignore_index = DATASET_META[dataset_name]['ignore_index']
    if is_train:
        joint_transform = build_train_joint_transform(cfg, ignore_index)
        input_transform = build_train_transform(cfg)
        target_transform = extended_transforms.MaskToTensor()
    else:
        joint_transform = None
        input_transform = build_test_transform(cfg)
        target_transform = extended_transforms.MaskToTensor()

    # FIXME:
    # * maxSkip
    # * class_uniform_pct
    if dataset_name == "cityscapes":
        dataset = Cityscapes(
            data_path,
            split,
            joint_transform=joint_transform,
            input_transform=input_transform,
            target_transform=target_transform)
    elif dataset_name == 'camvid':
        dataset = CamVid(
            data_path,
            split,
            joint_transform=joint_transform,
            input_transform=input_transform,
            target_transform=target_transform)
    elif dataset_name == 'ade20k':
        dataset = ADE20K(
            data_path,
            split,
            joint_transform=joint_transform,
            input_transform=input_transform,
            target_transform=target_transform)
    else:
        raise NotImplementedError()
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = None
    if is_train:
        # print("dist rank: {}, samples:{}".format(get_rank(), len(sampler)))
        batch_sampler = BatchSampler(
            sampler, batch_size=batch_size, drop_last=True)
        # print("batch rank: {}, samples:{}".format(
        #     get_rank(), len(batch_sampler)))

        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iterations=cfg.SOLVER.MAX_ITERS, start_iter=start_iter)
        # print("iterative batch rank: {}, samples:{}".format(
        #     get_rank(), len(batch_sampler)))

    else:
        sampler = SequentialSampler(dataset) if sampler is None else sampler
        batch_sampler = BatchSampler(
            sampler, batch_size=batch_size, drop_last=False)
    # print(len(dataset))
    data_loader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn)
    return data_loader
