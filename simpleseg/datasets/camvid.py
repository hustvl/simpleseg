"""
Camvid Dataset Loader
"""

import os
import logging
import shutil
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision.datasets.folder import is_image_file

logger = logging.getLogger(__name__)


def make_camvid_dataset(root, split):
    data_dir = os.path.join(root, split)
    images = []
    for root, _, fnames in sorted(os.walk(data_dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    targets = [x.replace(split, split + '_labels').replace('.', '_L.')
               for x in images]
    return images, targets


class CamVid(data.Dataset):

    NUM_CLASSES = 11
    CLASSES = ('Sky', 'Building', 'Column-Pole', 'Road', 'Sidewalk', 'Tree',
               'Sign-Symbol', 'Fence', 'Car', 'Pedestrain', 'Bicyclist')

    PALETTE = [(128, 128, 128), (128, 0, 0), (192, 192, 128), (128, 64, 128),
               (0, 0, 192), (128, 128, 0), (192, 128, 128), (64, 64, 128),
               (64, 0, 128), (64, 64, 0), (0, 128, 192)]

    def __init__(
            self, root, split,
            joint_transform=None, input_transform=None, target_transform=None):
        super().__init__()
        self.root = root
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
            s_images, s_masks = make_camvid_dataset(root, s)
            self.images += s_images
            self.masks += s_masks
            logger.info("loading {} images from {}".format(len(s_images), s))

    def convert_label(self, mask):
        mask = np.array(mask)
        mask_ind = np.full(mask.shape[:2], 255, dtype='uint8')
        for i, color in enumerate(CamVid.PALETTE):
            mask_ind[np.all(mask == color, axis=2)] = i
        return Image.fromarray(mask_ind, mode='P')

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert('RGB')
        w, h = img.size
        mask = self.convert_label(Image.open(self.masks[index]))

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

        palette_img = Image.fromarray(palette_results.astype(np.uint8))
        result_img = Image.fromarray(
            prediction.data.numpy().astype(np.uint8)).convert('P')
        palette_img.save(os.path.join(vis_dir, palette_name))
        # mv original image to the folder
        shutil.copyfile(name, os.path.join(vis_dir, basename))
        # save results
        result_img.save(os.path.join(pred_dir, basename))
        # .convert('P')
