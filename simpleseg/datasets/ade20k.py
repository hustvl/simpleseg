import os
import logging
import shutil
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision.datasets.folder import is_image_file
from simpleseg.datasets.transforms.transforms import resize_pad

logger = logging.getLogger(__name__)


def make_ade20k_dataset(root, split):
    # assert split in ["training", "validation"]
    annotation_dir = os.path.join(root, "annotations", split)
    image_dir = os.path.join(root, "images", split)

    image_names = [x for x in os.listdir(image_dir) if ".jpg" in x]
    images = [os.path.join(image_dir, x) for x in image_names]
    targets = [os.path.join(annotation_dir, x.replace(
        '.jpg', '.png')) for x in image_names]
    return images, targets


class ADE20K(data.Dataset):

    NUM_CLASSES = 150
    CLASSES = (
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
               [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
               [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
               [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
               [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
               [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
               [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
               [102, 255, 0], [92, 0, 255]]

    def __init__(self, root, split,
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
            assert s in ['training', 'validation']
            s_images, s_masks = make_ade20k_dataset(root, s)
            self.images += s_images
            self.masks += s_masks
            logger.info("loading {} images from {}".format(len(s_images), s))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert('RGB')
        w, h = img.size
        mask = np.array(Image.open(self.masks[index]).convert('L'))
        # reduce zero label
        # mask = mask - 1
        # mask[mask == -1] = 255
        mask[mask == 0] = 255
        mask = mask - 1
        mask[mask == 254] = 255
        mask = Image.fromarray(mask, mode='L')
        # FIXME: support testing set!
        # mask = self.convert_label(Image.open(self.masks[index]))

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        img_size = (img.size[1], img.size[0])

        # FIXME: resize and  padding
        # size = 512
        # padding 32
        pad = False
        if self.split[0] == "validation":
            img, img_size = resize_pad(img, 512)
            pad = True

        if self.input_transform is not None:
            img = self.input_transform(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        image_meta = {"name": self.images[index],
                      "ori_size": (h, w), "img_size": img_size, "pad": pad}
        return img, mask, image_meta

    def output_format(self, prediction, img_meta, output_dir):
        name = img_meta["name"]
        basename = os.path.basename(name)
        pred_dir = os.path.join(output_dir, 'pred')
        vis_dir = os.path.join(output_dir, 'vis')
        assert prediction.dim() == 2, "only support HxW prediction"
        # prediction is (HxW)
        palette = torch.tensor(self.PALETTE, dtype=torch.uint8)
        palette_name = basename.replace('.jpg', '_pred.png')
        palette_results = palette[prediction].data.numpy()

        palette_img = Image.fromarray(palette_results.astype(np.uint8))
        result_img = Image.fromarray(
            prediction.data.numpy().astype(np.uint8)).convert('P')
        palette_img.save(os.path.join(vis_dir, palette_name))
        # mv original image to the folder
        shutil.copyfile(name, os.path.join(vis_dir, basename))
        # save results
        # only support png for P mode
        result_img.save(os.path.join(
            pred_dir, basename.replace(".jpg", ".png")))
