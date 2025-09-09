import torch
import torch.nn as nn
import torch.nn.functional as F

from simpleseg.models.layers.norm import get_norm
from simpleseg.models.backbone.resnet import build_resnet
from simpleseg.models.backbone.dfnet import build_dfnet
from simpleseg.models.layers.operators import PyramidPoolingModule
from simpleseg.models.losses.loss import build_segmentation_loss
from .clanet import AuxilaryHead
from .segmentor import SEGMENTOR


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, norm="BN"):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        get_norm(norm, out_planes),
        nn.ReLU(inplace=True),
    )


@SEGMENTOR.register()
class ResPSPNet(nn.Module):
    def __init__(self, cfg, num_classes, ignore_index):
        super(ResPSPNet, self).__init__()
        # num_channels = cfg.MODEL.SFNET.NUM_CHANNELS  # 128 /
        norm = cfg.MODEL.NORM
        self.backbone = build_resnet(cfg)
        channels = self.backbone.output_channels
        # see: https://github.com/hszhao/semseg/blob/4f274c3f276778228bc14a4565822d46359f0cc8/model/pspnet.py#L62
        self.ppm = PyramidPoolingModule(
            channels, channels, channels // 4, norm=norm)
        self.cls = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=1, bias=False),
            get_norm(norm, channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, num_classes, 1))

        # aux head
        num_channels = 256
        num_aux_outputs = 1
        dsn = []
        for _ in range(num_aux_outputs):
            dsn.append(AuxilaryHead(
                num_channels, num_channels, num_classes, norm))
        self.dsn = nn.ModuleList(dsn)
        self.criterion = build_segmentation_loss(cfg, ignore_index)

    def forward_train(self, out, aux_outputs, targets):
        dsn_out = [dsn(aux_output)
                   for dsn, aux_output in zip(self.dsn, aux_outputs)]

        loss_dict = self.criterion((out, dsn_out), targets)
        return loss_dict

    def forward(self, x, targets=None):
        x_size = x.size()  # 800
        _, _, aux, x = self.backbone(x)
        x = self.ppm(x)
        x = self.cls(x)
        # print(x_size, x.shape, aux.shape)
        out = F.interpolate(
            x, x_size[2:], mode='bilinear', align_corners=False)
        if self.training:
            return self.forward_train(out, [aux], targets)
        else:
            return out



@SEGMENTOR.register()
class DFPSPNet(nn.Module):
    def __init__(self, cfg, num_classes, ignore_index):
        super(ResPSPNet, self).__init__()
        # num_channels = cfg.MODEL.SFNET.NUM_CHANNELS  # 128 /
        norm = cfg.MODEL.NORM
        self.backbone = build_dfnet(cfg)
        channels = self.backbone.output_channels
        # see: https://github.com/hszhao/semseg/blob/4f274c3f276778228bc14a4565822d46359f0cc8/model/pspnet.py#L62
        self.ppm = PyramidPoolingModule(
            channels, channels, channels // 4, norm=norm)
        self.cls = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=1, bias=False),
            get_norm(norm, channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, num_classes, 1))

        # aux head
        num_channels = 256
        num_aux_outputs = 1
        dsn = []
        for _ in range(num_aux_outputs):
            dsn.append(AuxilaryHead(
                num_channels, num_channels, num_classes, norm))
        self.dsn = nn.ModuleList(dsn)
        self.criterion = build_segmentation_loss(cfg, ignore_index)

    def forward_train(self, out, aux_outputs, targets):
        dsn_out = [dsn(aux_output)
                   for dsn, aux_output in zip(self.dsn, aux_outputs)]

        loss_dict = self.criterion((out, dsn_out), targets)
        return loss_dict

    def forward(self, x, targets=None):
        x_size = x.size()  # 800
        _, aux, x = self.backbone(x)
        x = self.ppm(x)
        x = self.cls(x)
        # print(x_size, x.shape, aux.shape)
        out = F.interpolate(
            x, x_size[2:], mode='bilinear', align_corners=False)
        if self.training:
            return self.forward_train(out, [aux], targets)
        else:
            return out
