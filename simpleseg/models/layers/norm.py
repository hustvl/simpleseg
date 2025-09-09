import torch
from torch.functional import norm
import torch.nn as nn
from apex.parallel import SyncBatchNorm


def get_norm(norm_type, channels):

    if norm_type == "BN":
        return nn.BatchNorm2d(channels)
    elif norm_type == "SyncBN":
        return SyncBatchNorm(channels)
    else:
        raise NotImplementedError()
