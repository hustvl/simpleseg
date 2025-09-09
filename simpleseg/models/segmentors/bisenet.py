import torch
import torch.nn as nn
import torch.nn.functional as F
from simpleseg.models.backbone.dfnet import build_dfnet

from simpleseg.models.layers.norm import get_norm
from simpleseg.models.backbone.resnet import build_resnet
from simpleseg.models.losses.loss import build_segmentation_loss
from .segmentor import SEGMENTOR


def conv_bn_relu(in_planes, out_planes, ksize, stride=1, norm="BN"):
    padding = ksize // 2
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                  padding=padding, stride=stride, bias=False),
        get_norm(norm, out_planes),
        nn.ReLU(inplace=True),
    )


class AttentionRefinement(nn.Module):
    def __init__(self, in_planes, out_planes, norm):
        super(AttentionRefinement, self).__init__()
        self.conv_3x3 = conv_bn_relu(in_planes, out_planes, 3, norm=norm)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_planes, out_planes, 1, padding=0, bias=False),
            get_norm(norm, out_planes),
            nn.Sigmoid()
        )

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.attention(fm)
        fm = fm * fm_se
        return fm


class FeatureFusion(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=1, norm="BN"):
        super(FeatureFusion, self).__init__()
        self.conv_1x1 = conv_bn_relu(in_planes, out_planes, 1, norm=norm)
        mid_channels = out_planes // reduction
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_planes, mid_channels, 1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, out_planes, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.attention(fm)
        output = fm + fm * fm_se
        return output


class SpatialPath(nn.Module):
    def __init__(self, in_planes, out_planes, norm="BN"):
        super(SpatialPath, self).__init__()
        inner_channel = 64
        self.conv_7x7 = conv_bn_relu(in_planes, inner_channel, 7, 2, norm=norm)
        self.conv_3x3_1 = conv_bn_relu(
            inner_channel, inner_channel, 3, 2, norm=norm)
        self.conv_3x3_2 = conv_bn_relu(
            inner_channel, inner_channel, 3, 2, norm=norm)
        self.conv_1x1 = conv_bn_relu(
            inner_channel, out_planes, 1, 1, norm=norm)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)
        return output


class AuxilaryHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes=19, norm="BN"):
        super(AuxilaryHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            get_norm(norm, channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, num_classes, kernel_size=1, bias=True))

    def forward(self, x):
        out = self.layers(x)
        return out


@SEGMENTOR.register()
class BiSeNet(nn.Module):

    def __init__(self, cfg, num_classes, ignore_index):
        super().__init__()

        norm = cfg.MODEL.NORM

        self.context_path = build_resnet(cfg)
        self.spatial_path = SpatialPath(3, 128, norm=norm)
        conv_channel = 128

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv_bn_relu(512, conv_channel, 1, norm=norm)
        )

        self.arms = nn.ModuleList([
            AttentionRefinement(512, conv_channel, norm),
            AttentionRefinement(256, conv_channel, norm)
        ])
        self.refines = nn.ModuleList([
            conv_bn_relu(conv_channel, conv_channel, 3, 1, norm),
            conv_bn_relu(conv_channel, conv_channel, 3, 1, norm),
        ])

        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2, 1, norm)

        self.output = nn.Sequential(
            nn.Conv2d(conv_channel * 2, 64, 3, padding=1, bias=False),
            get_norm(norm, 64),
            nn.ReLU(True),
            nn.Conv2d(64, num_classes, 1)
        )

        num_aux_outputs = 2
        dsn = []
        for _ in range(num_aux_outputs):
            dsn.append(AuxilaryHead(
                conv_channel, 256, num_classes, norm))
        self.dsn = nn.ModuleList(dsn)
        self.criterion = build_segmentation_loss(cfg, ignore_index)

    def forward_train(self, out, aux_outputs, targets):
        dsn_out = [dsn(aux_output)
                   for dsn, aux_output in zip(self.dsn, aux_outputs)]

        loss_dict = self.criterion((out, dsn_out), targets)
        return loss_dict

    def forward(self, x, targets=None):
        x_size = x.size()
        spatial_out = self.spatial_path(x)
        context_out = self.context_path(x)

        global_context = self.global_context(context_out[-1])
        global_context = F.interpolate(
            global_context, size=context_out[-1].size()[2:], mode='bilinear', align_corners=False)

        prev_feature = global_context
        out_features = []
        # context_out.reverse()
        for i, (arm, refine) in enumerate(zip(self.arms, self.refines)):
            feature = arm(context_out[-i-1])
            # print(feature.shape, prev_feature.shape)
            feature = feature + prev_feature
            prev_feature = F.interpolate(
                feature, size=context_out[-i-2].shape[2:], mode='bilinear', align_corners=False)
            prev_feature = refine(prev_feature)
            out_features.append(prev_feature)

        context_out = prev_feature

        concate_fm = self.ffm(spatial_out, context_out)

        out = self.output(concate_fm)
        out = F.interpolate(
            out, x_size[2:], mode='bilinear', align_corners=False)

        if self.training:
            return self.forward_train(out, out_features, targets)
        else:
            return out


@SEGMENTOR.register()
class DFBiSeNet(nn.Module):

    def __init__(self, cfg, num_classes, ignore_index):
        super().__init__()

        norm = cfg.MODEL.NORM

        self.context_path = build_dfnet(cfg)
        self.spatial_path = SpatialPath(3, 128, norm=norm)
        conv_channel = 128

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv_bn_relu(512, conv_channel, 1, norm=norm)
        )

        self.arms = nn.ModuleList([
            AttentionRefinement(512, conv_channel, norm),
            AttentionRefinement(256, conv_channel, norm)
        ])
        self.refines = nn.ModuleList([
            conv_bn_relu(conv_channel, conv_channel, 3, 1, norm),
            conv_bn_relu(conv_channel, conv_channel, 3, 1, norm),
        ])

        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2, 1, norm)

        self.output = nn.Sequential(
            nn.Conv2d(conv_channel * 2, 64, 3, padding=1, bias=False),
            get_norm(norm, 64),
            nn.ReLU(True),
            nn.Conv2d(64, num_classes, 1)
        )

        num_aux_outputs = 2
        dsn = []
        for _ in range(num_aux_outputs):
            dsn.append(AuxilaryHead(
                conv_channel, 256, num_classes, norm))
        self.dsn = nn.ModuleList(dsn)
        self.criterion = build_segmentation_loss(cfg, ignore_index)

    def forward_train(self, out, aux_outputs, targets):
        dsn_out = [dsn(aux_output)
                   for dsn, aux_output in zip(self.dsn, aux_outputs)]

        loss_dict = self.criterion((out, dsn_out), targets)
        return loss_dict

    def forward(self, x, targets=None):
        x_size = x.size()
        spatial_out = self.spatial_path(x)
        context_out = self.context_path(x)

        global_context = self.global_context(context_out[-1])
        global_context = F.interpolate(
            global_context, size=context_out[-1].size()[2:], mode='bilinear', align_corners=False)

        prev_feature = global_context
        out_features = []
        # context_out.reverse()
        for i, (arm, refine) in enumerate(zip(self.arms, self.refines)):
            feature = arm(context_out[-i-1])
            # print(feature.shape, prev_feature.shape)
            feature = feature + prev_feature
            prev_feature = F.interpolate(
                feature, size=context_out[-i-2].shape[2:], mode='bilinear', align_corners=False)
            prev_feature = refine(prev_feature)
            out_features.append(prev_feature)

        context_out = prev_feature

        concate_fm = self.ffm(spatial_out, context_out)

        out = self.output(concate_fm)
        out = F.interpolate(
            out, x_size[2:], mode='bilinear', align_corners=False)

        if self.training:
            return self.forward_train(out, out_features, targets)
        else:
            return out
