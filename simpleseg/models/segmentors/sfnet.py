import torch
import torch.nn as nn
import torch.nn.functional as F

from simpleseg.models.layers.norm import get_norm
from simpleseg.models.backbone.resnet import build_resnet
from simpleseg.models.backbone.dfnet import build_dfnet
from simpleseg.models.layers.operators import AlignedModule, PyramidPoolingModule
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


class UperNetAlignHead(nn.Module):

    def __init__(self, inplane, num_classes, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256, norm="BN"):
        super(UperNetAlignHead, self).__init__()

        self.ppm = PyramidPoolingModule(
            inplane, out_features=fpn_dim, mid_features=fpn_dim, norm=norm)
        # self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    get_norm(norm, fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for _ in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1, norm=norm),
            ))

            self.fpn_out_align.append(
                AlignedModule(inplane=fpn_dim, outplane=fpn_dim//2)
            )

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(
                len(fpn_inplanes) * fpn_dim, fpn_dim, 1, norm=norm),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        )

    def forward(self, conv_out):
        psp_out = self.ppm(conv_out[-1])

        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
            out.append(f)

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        # different from the original paper which adopts FAM to upsample multi-scale features for fusion
        # but same as the official implementation which adopts the bilinear interpolation
        # see: https://github.com/lxtGH/SFSegNets/blob/ce97d2e7dfd7cf3f3d2af7b0c31e23e51df6b182/network/sfnet_resnet.py#L103
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x, out


@SEGMENTOR.register()
class ResSFNet(nn.Module):
    def __init__(self, cfg, num_classes, ignore_index):
        super(ResSFNet, self).__init__()
        num_channels = cfg.MODEL.SFNET.NUM_CHANNELS  # 128 /
        norm = cfg.MODEL.NORM
        self.backbone = build_resnet(cfg)
        channels = self.backbone.output_channels
        in_channels = [channels // 8, channels // 4, channels // 2, channels]
        self.seg_head = UperNetAlignHead(
            channels, num_classes, in_channels, num_channels, norm)

        num_aux_outputs = 3
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
        x = self.backbone(x)
        x, aux_outputs = self.seg_head(x)
        out = F.interpolate(
            x, x_size[2:], mode='bilinear', align_corners=False)
        if self.training:
            return self.forward_train(out, aux_outputs, targets)
        else:
            return out


@SEGMENTOR.register()
class DFSFNet(nn.Module):
    def __init__(self, cfg, num_classes, ignore_index):
        super(DFSFNet, self).__init__()
        num_channels = cfg.MODEL.SFNET.NUM_CHANNELS  # 64 /
        norm = cfg.MODEL.NORM
        self.backbone = build_dfnet(cfg)
        channels = self.backbone.output_channels
        in_channels = [channels // 4, channels // 2, channels]
        self.seg_head = UperNetAlignHead(
            channels, num_classes, in_channels, num_channels, norm)

        num_aux_outputs = 2
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
        x = self.backbone(x)
        x, aux_outputs = self.seg_head(x)
        out = F.interpolate(
            x, x_size[2:], mode='bilinear', align_corners=False)
        if self.training:
            return self.forward_train(out, aux_outputs, targets)
        else:
            return out
