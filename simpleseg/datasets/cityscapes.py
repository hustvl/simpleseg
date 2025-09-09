"""
Cityscapes Dataset Loader
"""
import logging
# import json
import os
import shutil
import numpy as np
from PIL import Image
import torch
from torch.utils import data

import torchvision.transforms as transforms
# import simpleseg.datasets.uniform as uniform
import simpleseg.datasets.cityscapes_labels as cityscapes_labels
# import simpleseg.datasets.edge_utils as edge_utils
logger = logging.getLogger(__name__)
# from config import cfg
# print(__name__)
trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid
trainid_to_id = cityscapes_labels.trainId2label


def make_cityscapes_dataset(data_dir, split, mode='fine'):
    images = []
    target = []

    def _get_target(filename):
        return filename.replace('_leftImg8bit.png', '_gtFine_labelIds.png')

    images_dir = os.path.join(data_dir, 'leftImg8bit', split)
    target_dir = os.path.join(data_dir, 'gtFine', split)

    for city in os.listdir(images_dir):
        img_dir = os.path.join(images_dir, city)
        tgt_dir = os.path.join(target_dir, city)
        for filename in os.listdir(img_dir):
            images.append(
                os.path.join(img_dir, filename)
            )
            target.append(
                os.path.join(tgt_dir, _get_target(filename))
            )

    return images, target


class Cityscapes(data.Dataset):

    NUM_CLASSES = 19
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self, root, split, mode='fine',
                 joint_transform=None, input_transform=None, target_transform=None):
        super().__init__()
        self.mode = mode
        if isinstance(split, str):
            split = [split]
        self.split = split
        self.joint_transform = joint_transform
        self.input_transform = input_transform
        self.target_transform = target_transform

        self.images = []
        self.masks = []
        for s in split:
            assert s in ['train', 'val', 'test']
            s_images, s_masks = make_cityscapes_dataset(root, s)
            self.images += s_images
            self.masks += s_masks
            logger.info("loading {} images from {}".format(len(s_images), s))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert('RGB')
        w, h = img.size
        if self.split[0] != "test":
            # convert label
            mask = np.array(Image.open(self.masks[index]))
            # print(np.unique(mask))
            # NOTE: -1 for license plate
            mask[mask < 0] = 0
            mask_ = mask.copy()
            for k, v in id_to_trainid.items():
                mask_[mask == k] = v
            mask = Image.fromarray(mask_, mode='P')
        else:
            mask = Image.fromarray(np.zeros_like(img, dtype=np.uint8))
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.input_transform is not None:
            img = self.input_transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        img_size = img.shape[-2:]
        # print(img.shape, mask.shape, torch.unique(mask))
        image_meta = {"name": self.images[index],
                      "ori_size": (h, w), "img_size": img_size, "pad": False}
        return img, mask, image_meta

    def output_format(self, prediction, img_meta, output_dir):
        name = img_meta["name"]
        basename = os.path.basename(name)
        pred_dir = os.path.join(output_dir, 'pred')
        vis_dir = os.path.join(output_dir, 'vis')
        assert prediction.dim() == 2, "only support HxW prediction"
        # prediction is (HxW)
        palette = torch.tensor(self.PALETTE, dtype=torch.uint8)
        palette_name = basename.replace('.png', '_pred.png')
        palette_results = palette[prediction].data.numpy()
        # convert trainid to id
        pred_results = prediction.clone()
        for k, v in trainid_to_id.items():
            pred_results[prediction == k] = v.id
        pred_results = pred_results.data.numpy()

        palette_img = Image.fromarray(palette_results.astype(np.uint8))
        result_img = Image.fromarray(
            pred_results.astype(np.uint8)).convert('P')
        palette_img.save(os.path.join(vis_dir, palette_name))
        # mv original image to the folder
        shutil.copyfile(name, os.path.join(vis_dir, basename))
        # save results
        result_img.save(os.path.join(pred_dir, basename))
        # .convert('P')
