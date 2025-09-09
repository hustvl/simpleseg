import torch
import torch.nn as nn
import torch.nn.functional as F

from simpleseg.models.layers.norm import get_norm
from simpleseg.models.backbone.resnet import build_resnet
from simpleseg.models.backbone.dfnet import build_dfnet
from simpleseg.models.backbone.swin_transformer import build_swin_lite
from simpleseg.models.backbone.rest import build_rest_lite
from simpleseg.models.backbone.pvt import build_pvt_tiny
from simpleseg.models.layers.cross_layer_attention import AlginCrossLayerFusion, CCCrossLayerAttention, NLCrossLayerAttention, BilinearCrossLayerFusion, CAFARECrossLayerFusion
from simpleseg.models.losses.loss import build_segmentation_loss
from .segmentor import SEGMENTOR
from simpleseg.models.layers.cross_layer_attention import CCCrossLayerAttentionB, CCCrossLayerAttentionC, CCCrossLayerAttentionV, SparseCrossLayerAttention

CROSS_LAYER_ATTENTION = {
    "CC": CCCrossLayerAttention,
    "NL": NLCrossLayerAttention,
    "BL": BilinearCrossLayerFusion,
    "CC-B": CCCrossLayerAttentionB,
    "CC-C": CCCrossLayerAttentionC,
    "Align": AlginCrossLayerFusion,
    "VIS": CCCrossLayerAttentionV,
    "Sparse": SparseCrossLayerAttention
}


class GAP(nn.Module):

    def __init__(self, channels, norm):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            get_norm(norm, channels),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            get_norm(norm, channels),
            nn.ReLU(True)
        )
        # self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2]:
            for k in m.modules():
                if isinstance(k, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(k.weight)
                    if k.bias is not None:
                        k.bias.data.zero_()

    def forward(self, features):
        global_features = torch.mean(features, [2, 3], keepdim=True)
        out = self.conv1(global_features)
        out = self.conv2(out + features)
        return out


class AuxilaryHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes=19, norm="BN"):
        super(AuxilaryHead, self).__init__()
        # print(in_channels, channels)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            get_norm(norm, channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(channels, num_classes, kernel_size=1, bias=True))

        # self._init_weights()

    def _init_weights(self):
        for k in self.layers.modules():
            if isinstance(k, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(k.weight)
                if k.bias is not None:
                    k.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        return out


class PSPModule(nn.Module):

    def __init__(self, features, out_features, sizes=(1, 2, 3, 6), norm="BN"):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(features, out_features, size, norm) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, features,
                      kernel_size=1, padding=0, dilation=1, bias=False),
            get_norm(norm, features),
            nn.ReLU(True),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size, norm):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = get_norm(norm, out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(
            stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class CLAHead(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes, context='', norm="BN", with_3x3=False, attn_type="CC"):
        super().__init__()
        self.fpn_in = nn.ModuleList()
        # BUG: support
        if not isinstance(in_channels, list):
            in_channels = [
                in_channels // 8, in_channels // 4, in_channels // 2, in_channels]
        # print(in_channels)
        for in_channel in in_channels:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, out_channels, 1, bias=False),
                    get_norm(norm, out_channels),
                    nn.ReLU(True)
                )
            )
        if context == 'ppm':
            self.ppm = PSPModule(out_channels, out_channels // 4, norm=norm)
        elif context == 'gap':
            self.ppm = GAP(out_channels, norm)
        else:
            self.ppm = lambda x: x

        fuse = [
            nn.Conv2d(out_channels*3, out_channels, kernel_size=1, bias=False),
            get_norm(norm, out_channels),
            nn.ReLU(True)
        ]
        if with_3x3:
            fuse += [
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                get_norm(norm, out_channels),
                nn.ReLU(True),
            ]

        self.merge = nn.Sequential(*fuse)
        cross_layer_attention = CROSS_LAYER_ATTENTION[attn_type]
        self.cross_layer_attentions = nn.ModuleList([
            cross_layer_attention(out_channels, norm),
            cross_layer_attention(out_channels, norm)
        ])
        output = [
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1, bias=False),
            get_norm(norm, out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, num_classes, 1)
        ]
        self.output = nn.Sequential(*output)

    def forward(self, features):
        p2 = self.fpn_in[0](features[0])
        p3 = self.fpn_in[1](features[1])
        p4 = self.fpn_in[2](features[2])
        p5 = self.fpn_in[3](features[3])
        # p5
        p5 = self.ppm(p5)
        # add name for visualize
        p4 = self.cross_layer_attentions[0](p3, p4)
        p5 = self.cross_layer_attentions[1](p3, p5)
        out_features = [p3, p4, p5]
        features = torch.cat(out_features, dim=1)
        features = self.merge(features)
        features = F.interpolate(
            features, p2.size()[2:], mode='bilinear', align_corners=False)
        output = self.output(torch.cat([features, p2], dim=1))
        # output features for dsn
        return output, out_features[1:]


class CLAHeadS8(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes, context='', norm="BN", with_3x3=False, attn_type="CC"):
        super().__init__()
        self.fpn_in = nn.ModuleList()
        in_channels = [in_channels // 4, in_channels // 2, in_channels]
        for in_channel in in_channels:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, out_channels, 1, bias=False),
                    get_norm(norm, out_channels),
                    nn.ReLU(True)
                )
            )
        if context == 'ppm':
            self.ppm = PSPModule(out_channels, out_channels // 4, norm=norm)
        elif context == 'gap':
            self.ppm = GAP(out_channels, norm)
        else:
            self.ppm = lambda x: x

        fuse = [
            nn.Conv2d(out_channels*3, out_channels, kernel_size=1, bias=False),
            get_norm(norm, out_channels),
            nn.ReLU(True)
        ]
        if with_3x3:
            fuse += [
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                get_norm(norm, out_channels),
                nn.ReLU(True),
            ]

        self.merge = nn.Sequential(*fuse)
        cross_layer_attention = CROSS_LAYER_ATTENTION[attn_type]
        self.cross_layer_attentions = nn.ModuleList([
            cross_layer_attention(out_channels, norm),
            cross_layer_attention(out_channels, norm)
        ])
        self.output = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, features):
        # p2 = self.fpn_in[0](features[0])
        p3 = self.fpn_in[0](features[1])
        p4 = self.fpn_in[1](features[2])
        p5 = self.fpn_in[2](features[3])
        # p5
        p5 = self.ppm(p5)
        p4 = self.cross_layer_attentions[0](p3, p4)
        p5 = self.cross_layer_attentions[1](p3, p5)
        out_features = [p3, p4, p5]
        features = torch.cat(out_features, dim=1)
        features = self.merge(features)
        output = self.output(features)
        # features = F.interpolate(
        #     features, p2.size()[2:], mode='bilinear', align_corners=False)
        # output = self.output(torch.cat([features, p2], dim=1))
        # output features for dsn
        return output, []  # out_features[1:]


class CLAHeadCARAFE(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes, context='', norm="BN", with_3x3=False, attn_type="CC"):
        super().__init__()
        self.fpn_in = nn.ModuleList()
        in_channels = [
            in_channels // 8, in_channels // 4, in_channels // 2, in_channels]
        for in_channel in in_channels:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, out_channels, 1, bias=False),
                    get_norm(norm, out_channels),
                    nn.ReLU(True)
                )
            )
        if context == 'ppm':
            self.ppm = PSPModule(out_channels, out_channels // 4, norm=norm)
        elif context == 'gap':
            self.ppm = GAP(out_channels, norm)
        else:
            self.ppm = lambda x: x

        fuse = [
            nn.Conv2d(out_channels*3, out_channels, kernel_size=1, bias=False),
            get_norm(norm, out_channels),
            nn.ReLU(True)
        ]
        if with_3x3:
            fuse += [
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                get_norm(norm, out_channels),
                nn.ReLU(True),
            ]

        self.merge = nn.Sequential(*fuse)
        # cross_layer_attention = CROSS_LAYER_ATTENTION[attn_type]
        self.cross_layer_attentions = nn.ModuleList([
            CAFARECrossLayerFusion(out_channels, norm, 2),
            CAFARECrossLayerFusion(out_channels, norm, 4)
        ])
        output = [
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1, bias=False),
            get_norm(norm, out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, num_classes, 1)
        ]
        self.output = nn.Sequential(*output)

    def forward(self, features):
        p2 = self.fpn_in[0](features[0])
        p3 = self.fpn_in[1](features[1])
        p4 = self.fpn_in[2](features[2])
        p5 = self.fpn_in[3](features[3])
        # p5
        p5 = self.ppm(p5)
        p4 = self.cross_layer_attentions[0](p3, p4)
        p5 = self.cross_layer_attentions[1](p3, p5)
        # print(p4.shape, p3.shape, p5.shape)
        out_features = [p3, p4, p5]
        features = torch.cat(out_features, dim=1)
        features = self.merge(features)
        features = F.interpolate(
            features, p2.size()[2:], mode='bilinear', align_corners=False)
        output = self.output(torch.cat([features, p2], dim=1))
        # output features for dsn
        return output, out_features[1:]


class CLAGAPHead(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes, norm="BN", with_3x3=False, attn_type="CC"):
        super().__init__()
        self.fpn_in = nn.ModuleList()
        in_channels = [
            in_channels // 8, in_channels // 4, in_channels // 2, in_channels]
        for in_channel in in_channels:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, out_channels, 1, bias=False),
                    get_norm(norm, out_channels),
                    nn.ReLU(True)
                )
            )
        self.gaps = nn.ModuleList([
            GAP(out_channels, norm),
            GAP(out_channels, norm)
        ])

        fuse = [
            nn.Conv2d(out_channels*3, out_channels, kernel_size=1, bias=False),
            get_norm(norm, out_channels),
            nn.ReLU(True)
        ]
        if with_3x3:
            fuse += [
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                get_norm(norm, out_channels),
                nn.ReLU(True),
            ]

        self.merge = nn.Sequential(*fuse)
        cross_layer_attention = CROSS_LAYER_ATTENTION[attn_type]
        self.cross_layer_attentions = nn.ModuleList([
            cross_layer_attention(out_channels, norm),
            cross_layer_attention(out_channels, norm)
        ])
        output = [
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1, bias=False),
            get_norm(norm, out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, num_classes, 1)
        ]
        self.output = nn.Sequential(*output)

    def forward(self, features):
        p2 = self.fpn_in[0](features[0])
        p3 = self.fpn_in[1](features[1])
        p4 = self.fpn_in[2](features[2])
        p5 = self.fpn_in[3](features[3])
        # p5
        p5 = self.gaps[0](p5)
        p4 = self.gaps[1](p4)
        p4 = self.cross_layer_attentions[0](p3, p4)
        p5 = self.cross_layer_attentions[1](p3, p5)
        out_features = [p3, p4, p5]
        features = torch.cat(out_features, dim=1)
        features = self.merge(features)
        features = F.interpolate(
            features, p2.size()[2:], mode='bilinear', align_corners=False)
        output = self.output(torch.cat([features, p2], dim=1))
        # output features for dsn
        return output, out_features[1:]


class CLALightHead(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes, context='', norm="BN", attn_type="CC"):
        super().__init__()
        self.fpn_in = nn.ModuleList()
        in_channels = [
            in_channels // 4, in_channels // 2, in_channels]
        for in_channel in in_channels:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, out_channels, 1, bias=False),
                    get_norm(norm, out_channels),
                    nn.ReLU(True)
                )
            )
        if context == 'ppm':
            self.ppm = PSPModule(out_channels, out_channels // 4, norm=norm)
        elif context == 'gap':
            self.ppm = GAP(out_channels, norm)
        else:
            self.ppm = lambda x: x

        # self.merge = nn.Sequential(
        #     nn.Conv2d(out_channels*3, out_channels, kernel_size=1, bias=False),
        #     get_norm(norm, out_channels),
        #     nn.ReLU(True)
        # )
        cross_layer_attention = CROSS_LAYER_ATTENTION[attn_type]
        self.cross_layer_attentions = nn.ModuleList([
            cross_layer_attention(out_channels, norm),
            cross_layer_attention(out_channels, norm)
        ])

        self.output = nn.Sequential(
            nn.Conv2d(out_channels*3, out_channels, kernel_size=1, bias=False),
            get_norm(norm, out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            get_norm(norm, out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, num_classes, 1)
        )

    def forward(self, features):
        # for k in features:
        #     print(k.shape)
        # p2 = self.fpn_in[0](features[0])
        p3 = self.fpn_in[0](features[0])
        p4 = self.fpn_in[1](features[1])
        p5 = self.fpn_in[2](features[2])
        # p5
        p5 = self.ppm(p5)
        p4 = self.cross_layer_attentions[0](p3, p4)
        p5 = self.cross_layer_attentions[1](p3, p5)
        out_features = [p3, p4, p5]
        features = torch.cat(out_features, dim=1)
        # features = self.merge(features)
        # features = F.interpolate(
        #     features, p2.size()[2:], mode='bilinear', align_corners=False)
        output = self.output(features)
        # output features for dsn
        return output, out_features[1:]


def build_cla_head(cfg, in_channels, out_channels, num_classes, context, norm, with_3x3=False):
    name = cfg.MODEL.CLANET.HEAD
    attn_type = cfg.MODEL.CLANET.ATTN_TYPE
    if name == "CLAHead":
        return CLAHead(in_channels, out_channels, num_classes, context, norm, with_3x3, attn_type=attn_type)
    elif name == "CLALightHead":
        return CLALightHead(in_channels, out_channels, num_classes, context, norm, attn_type=attn_type)
    elif name == "CLAGAPHead":
        return CLAGAPHead(in_channels, out_channels, num_classes, norm, with_3x3, attn_type=attn_type)
    elif name == "CLAHeadCARAFE":
        return CLAHeadCARAFE(in_channels, out_channels, num_classes, context, norm, with_3x3, attn_type=attn_type)
    elif name == "CLAHeadS8":
        return CLAHeadS8(in_channels, out_channels, num_classes, context, norm, with_3x3, attn_type=attn_type)


@SEGMENTOR.register()
class ResCLANet(nn.Module):

    def __init__(self, cfg, num_classes, ignore_index):
        super().__init__()
        context = cfg.MODEL.CLANET.CONTEXT
        num_channels = cfg.MODEL.CLANET.NUM_CHANNELS
        with_3x3 = cfg.MODEL.CLANET.FUSE_3X3
        norm = cfg.MODEL.NORM
        self.backbone = build_resnet(cfg)
        channels = self.backbone.output_channels
        self.seg_head = build_cla_head(
            cfg, channels, num_channels, num_classes, context, norm, with_3x3=with_3x3)
        num_aux_outputs = 2
        dsn = []
        for _ in range(num_aux_outputs):
            dsn.append(
                AuxilaryHead(num_channels, num_channels, num_classes, norm))
        self.dsn = nn.ModuleList(dsn)
        self.criterion = build_segmentation_loss(cfg, ignore_index)

    def forward_train(self, out, aux_outputs, targets):
        dsn_out = [dsn(aux_output)
                   for dsn, aux_output in zip(self.dsn, aux_outputs)]

        loss_dict = self.criterion((out, dsn_out), targets)
        return loss_dict

    def forward(self, x, targets=None):
        x_size = x.size()
        x = self.backbone(x)
        x, aux_outputs = self.seg_head(x)
        out = F.interpolate(
            x, x_size[2:], mode='bilinear', align_corners=False)
        if self.training:
            return self.forward_train(out, aux_outputs, targets)
        else:
            return out


@SEGMENTOR.register()
class SwinCLANet(nn.Module):

    def __init__(self, cfg, num_classes, ignore_index):
        super().__init__()
        context = cfg.MODEL.CLANET.CONTEXT
        num_channels = cfg.MODEL.CLANET.NUM_CHANNELS
        with_3x3 = cfg.MODEL.CLANET.FUSE_3X3
        norm = cfg.MODEL.NORM
        self.backbone = build_swin_lite(cfg)
        channels = 512
        # channels = self.backbone.output_channels
        self.seg_head = build_cla_head(
            cfg, channels, num_channels, num_classes, context, norm, with_3x3=with_3x3)
        num_aux_outputs = 2
        dsn = []
        for _ in range(num_aux_outputs):
            dsn.append(
                AuxilaryHead(num_channels, num_channels, num_classes, norm))
        self.dsn = nn.ModuleList(dsn)
        self.criterion = build_segmentation_loss(cfg, ignore_index)

    def forward_train(self, out, aux_outputs, targets):
        dsn_out = [dsn(aux_output)
                   for dsn, aux_output in zip(self.dsn, aux_outputs)]

        loss_dict = self.criterion((out, dsn_out), targets)
        return loss_dict

    def forward(self, x, targets=None):
        x_size = x.size()
        x = self.backbone(x)
        x, aux_outputs = self.seg_head(x)
        out = F.interpolate(
            x, x_size[2:], mode='bilinear', align_corners=False)
        if self.training:
            return self.forward_train(out, aux_outputs, targets)
        else:
            return out


@SEGMENTOR.register()
class PVTCLANet(nn.Module):

    def __init__(self, cfg, num_classes, ignore_index):
        super().__init__()
        context = cfg.MODEL.CLANET.CONTEXT
        num_channels = cfg.MODEL.CLANET.NUM_CHANNELS
        with_3x3 = cfg.MODEL.CLANET.FUSE_3X3
        norm = cfg.MODEL.NORM
        self.backbone = build_pvt_tiny(cfg)
        channels = cfg.MODEL.PVT.EMBED_DIM
        # channels = 512
        self.seg_head = build_cla_head(
            cfg, channels, num_channels, num_classes, context, norm, with_3x3=with_3x3)
        num_aux_outputs = 2
        dsn = []
        for _ in range(num_aux_outputs):
            dsn.append(
                AuxilaryHead(num_channels, num_channels, num_classes, norm))
        self.dsn = nn.ModuleList(dsn)
        self.criterion = build_segmentation_loss(cfg, ignore_index)

    def forward_train(self, out, aux_outputs, targets):
        dsn_out = [dsn(aux_output)
                   for dsn, aux_output in zip(self.dsn, aux_outputs)]

        loss_dict = self.criterion((out, dsn_out), targets)
        return loss_dict

    def forward(self, x, targets=None):
        x_size = x.size()
        x = self.backbone(x)
        x, aux_outputs = self.seg_head(x)
        out = F.interpolate(
            x, x_size[2:], mode='bilinear', align_corners=False)
        if self.training:
            return self.forward_train(out, aux_outputs, targets)
        else:
            return out


@SEGMENTOR.register()
class ResTCLANet(nn.Module):

    def __init__(self, cfg, num_classes, ignore_index):
        super().__init__()
        context = cfg.MODEL.CLANET.CONTEXT
        num_channels = cfg.MODEL.CLANET.NUM_CHANNELS
        with_3x3 = cfg.MODEL.CLANET.FUSE_3X3
        norm = cfg.MODEL.NORM
        self.backbone = build_rest_lite(cfg)
        channels = 512
        # channels = self.backbone.output_channels
        self.seg_head = build_cla_head(
            cfg, channels, num_channels, num_classes, context, norm, with_3x3=with_3x3)
        num_aux_outputs = 2
        dsn = []
        for _ in range(num_aux_outputs):
            dsn.append(
                AuxilaryHead(num_channels, num_channels, num_classes, norm))
        self.dsn = nn.ModuleList(dsn)
        self.criterion = build_segmentation_loss(cfg, ignore_index)

    def forward_train(self, out, aux_outputs, targets):
        dsn_out = [dsn(aux_output)
                   for dsn, aux_output in zip(self.dsn, aux_outputs)]

        loss_dict = self.criterion((out, dsn_out), targets)
        return loss_dict

    def forward(self, x, targets=None):
        x_size = x.size()
        x = self.backbone(x)
        x, aux_outputs = self.seg_head(x)
        out = F.interpolate(
            x, x_size[2:], mode='bilinear', align_corners=False)
        if self.training:
            return self.forward_train(out, aux_outputs, targets)
        else:
            return out


@SEGMENTOR.register()
class DFCLANet(nn.Module):

    def __init__(self, cfg, num_classes, ignore_index):
        super().__init__()
        context = cfg.MODEL.CLANET.CONTEXT
        num_channels = cfg.MODEL.CLANET.NUM_CHANNELS
        norm = cfg.MODEL.NORM
        self.backbone = build_dfnet(cfg)
        channels = self.backbone.output_channels
        self.seg_head = build_cla_head(
            cfg, channels, num_channels, num_classes, context, norm)
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
        x_size = x.size()
        x = self.backbone(x)
        x, aux_outputs = self.seg_head(x)
        out = F.interpolate(
            x, x_size[2:], mode='bilinear', align_corners=False)
        if self.training:
            return self.forward_train(out, aux_outputs, targets)
        else:
            return out
