from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = "./ouptut"

_C.MODEL = CN()
_C.MODEL.NORM = "SyncBN"
_C.MODEL.WEIGHTS = ""
# ./pretrained_models/df2_imagenet.pth
_C.MODEL.PRETRAINED = "./pretrained_models/resnet18-deep-inplane128.pth"

_C.MODEL.BACKBONE = "resnet18"
_C.MODEL.BACKBONE_OUTPUTS = [0, 1, 2, 3]
_C.MODEL.RESNET = CN()
# dilations and strides for C4 and C5
_C.MODEL.RESNET.DILATIONS = (1, 1)
# output_stride = 8, (1, 1)
# output_stride = 16, (-1, 1)
# output_stride = 32, (-1, -1)
_C.MODEL.RESNET.STRIDES = (-1, -1)

# Swin-Lite
_C.MODEL.SWINLITE = CN()
_C.MODEL.SWINLITE.EMBED_DIM = 64
_C.MODEL.SWINLITE.DEPTH = [2, 2, 2, 2]
_C.MODEL.SWINLITE.HEADS = [2, 4, 8, 16]
_C.MODEL.SWINLITE.WINDOW_SIZE = 7

# ResT-Lite
_C.MODEL.RESTLITE = CN()
_C.MODEL.RESTLITE.EMBED_DIM = [64, 128, 256, 512]
_C.MODEL.RESTLITE.DEPTH = [2, 2, 2, 2]
_C.MODEL.RESTLITE.HEADS = [1, 2, 4, 8]
_C.MODEL.RESTLITE.BATCH_NORM = "SyncBN"

# PVT
_C.MODEL.PVT = CN()
_C.MODEL.PVT.EMBED_DIM = [32, 64, 160, 256]
_C.MODEL.PVT.HEADS = [1, 2, 5, 8]
_C.MODEL.PVT.DEPTH = [2, 2, 2, 2]
# DFNet
_C.MODEL.DFNET = CN()
_C.MODEL.DFNET.STRIDE = 32

# CLANet
_C.MODEL.CLANET = CN()
_C.MODEL.CLANET.HEAD = "CLAHead"
_C.MODEL.CLANET.CONTEXT = ""
_C.MODEL.CLANET.NUM_CHANNELS = 128
_C.MODEL.CLANET.FUSE_3X3 = False
_C.MODEL.CLANET.OUTPUT_3X3 = False
_C.MODEL.CLANET.ATTN_TYPE = "CC"

_C.MODEL.SFNET = CN()
_C.MODEL.SFNET.NUM_CHANNELS = 128

_C.MODEL.NAME = ""

_C.DATASET = CN()
_C.DATASET.NUM_CLASSES = 19
_C.DATASET.NUM_WORKERS = 4
_C.DATASET.NAME = "cityscapes"
_C.DATASET.PATH = "data/cityscapes/"
# split for training: train, trainval
_C.DATASET.TRAIN = ("train",)
# split for testing: val, test
_C.DATASET.TEST = "val"

_C.DATASET.MEAN = [0.485, 0.456, 0.406]
_C.DATASET.STD = [0.229, 0.224, 0.225]
# Border Relaxtion
_C.DATASET.TRANSFORMS = CN()
# blur: ("none", "gblur", "bblur")
_C.DATASET.TRANSFORMS.BLUR = "none"
# resize and crop > transforms
_C.DATASET.TRANSFORMS.CROP_CATE_RATIO = 1.0
_C.DATASET.TRANSFORMS.SCALES = (0.5, 2.0)
_C.DATASET.TRANSFORMS.CROP_SIZE = (720, 720)
# color augment (brightness, contrast, saturation, hue)
_C.DATASET.TRANSFORMS.COLOR_AUG = (0.25, 0.25, 0.25, 0.25)

_C.LOSS = CN()
_C.LOSS.NAME = ""
#
_C.LOSS.AUX_WEIGHT = 0.4
_C.LOSS.USE_WEIGHT = False
_C.LOSS.SIZE_AVERAGE = True


_C.SOLVER = CN()
_C.SOLVER.SCHEDULER = "poly"
_C.SOLVER.ITERATION_BASED = True
_C.SOLVER.MAX_EPOCHS = 0
# batchsize per GPU
_C.SOLVER.BATCH_SIZE = 2
_C.SOLVER.OPTIMIZER = "sgd"
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.MAX_ITERS = 0
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.AMSGRAD = False
_C.SOLVER.POWER = 0.9
_C.SOLVER.SCHEDULER_EPOCH = False

_C.SOLVER.WARMUP_FACTOR = 0.001
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 5000
_C.SOLVER.TEST_PERIOD = 5000

_C.TEST = CN()
_C.TEST.MODE = ""
_C.TEST.BATCH_SIZE = 4
# ["origin", "resize"]
_C.TEST.INPUT_MODE = "origin"
_C.TEST.SIZE = (512, 2048)

# _C.TEST = CN()
# # ["whole", "pooling", "sliding"]
# _C.TEST.MODE = ""


def get_cfg_defaults():
    return _C.clone()
