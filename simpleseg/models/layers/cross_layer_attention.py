import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from .cc_attention_op import CrissCrossAttentionOps
from .operators import AlignedModule
from .norm import get_norm
from .sparse_attention import MultiScaleAttention


class SparseCrossLayerAttention(nn.Module):

    def __init__(self, channels, norm="BN"):
        super(SparseCrossLayerAttention, self).__init__()
        # 128, tune
        in_channels = channels // 2
        inter_channels = channels // 16
        self.attention = MultiScaleAttention(in_channels, inter_channels, in_channels, num_points=32)
        self.conv1 = nn.Conv2d(2 * channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, in_channels, kernel_size=1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+channels, channels,
                      kernel_size=1, dilation=1, bias=False),
            get_norm(norm, channels),
            nn.ReLU(True)
        )
        self.counter = 1

    def forward(self, low_feature, high_feature):
        # global counter
        out_size = low_feature.size()[2:]
        high_feature = F.interpolate(
            high_feature, out_size, mode='bilinear', align_corners=False)
        query_feature = self.conv1(
            torch.cat([high_feature, low_feature], dim=1))
        value_feature = self.conv2(high_feature)
        value_feature = self.attention(query_feature, value_feature)
        # for _ in range(2):
        #     value_feature = self.attention(
        #         query_feature, value_feature, value_feature)
        output = self.bottleneck(
            torch.cat([value_feature, high_feature], dim=1))
        return output



class CCCrossLayerAttention(nn.Module):

    def __init__(self, channels, norm="BN"):
        super(CCCrossLayerAttention, self).__init__()
        # 128, tune
        in_channels = channels // 2
        inter_channels = channels // 16
        self.attention = CrissCrossAttentionOps(in_channels, inter_channels)
        self.conv1 = nn.Conv2d(2 * channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, in_channels, kernel_size=1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+channels, channels,
                      kernel_size=1, dilation=1, bias=False),
            get_norm(norm, channels),
            nn.ReLU(True)
        )
        self.counter = 1

    def forward(self, low_feature, high_feature):
        # global counter
        out_size = low_feature.size()[2:]
        high_feature = F.interpolate(
            high_feature, out_size, mode='bilinear', align_corners=False)
        query_feature = self.conv1(
            torch.cat([high_feature, low_feature], dim=1))
        value_feature = self.conv2(high_feature)
        for _ in range(2):
            value_feature = self.attention(
                query_feature, value_feature, value_feature)
        output = self.bottleneck(
            torch.cat([value_feature, high_feature], dim=1))
        return output


class FastCCCrossLayerAttention(nn.Module):

    def __init__(self, channels, norm="BN"):
        super(CCCrossLayerAttention, self).__init__()
        # 128, tune
        in_channels = channels // 2
        inter_channels = channels // 16
        self.attention = CrissCrossAttentionOps(in_channels, inter_channels)
        self.conv1 = nn.Conv2d(2 * channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, in_channels, kernel_size=1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+channels, channels,
                      kernel_size=1, dilation=1, bias=False),
            get_norm(norm, channels),
            nn.ReLU(True)
        )
        self.counter = 1

    def forward(self, low_feature, high_feature):
        # global counter
        out_size = low_feature.size()[2:]
        high_feature = F.interpolate(
            high_feature, out_size, mode='bilinear', align_corners=False)
        query_feature = self.conv1(
            torch.cat([high_feature, low_feature], dim=1))
        value_feature = self.conv2(high_feature)
        value_feature = self.attention(
            query_feature, value_feature, value_feature)
        output = self.bottleneck(
            torch.cat([value_feature, high_feature], dim=1))
        return output


class CCCrossLayerAttentionV(nn.Module):

    def __init__(self, channels, norm="BN"):
        super(CCCrossLayerAttentionV, self).__init__()
        # 128, tune
        in_channels = channels // 2
        inter_channels = channels // 16
        self.attention = CrissCrossAttentionOps(in_channels, inter_channels)
        self.conv1 = nn.Conv2d(2 * channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, in_channels, kernel_size=1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+channels, channels,
                      kernel_size=1, dilation=1, bias=False),
            get_norm(norm, channels),
            nn.ReLU(True)
        )
        self.counter = 1

    def forward(self, low_feature, high_feature, name="p4"):
        # global counter
        out_size = low_feature.size()[2:]
        high_feature = F.interpolate(
            high_feature, out_size, mode='bilinear', align_corners=False)
        query_feature = self.conv1(
            torch.cat([high_feature, low_feature], dim=1))
        value_feature = self.conv2(high_feature)
        weights = []
        for _ in range(2):
            value_feature, weight = self.attention(
                query_feature, value_feature, value_feature, return_weight=True)
            weights.append(weight)

        weights = torch.stack(weights, dim=1)
        torch.save(weights, 'weights/{}_{}.pth'.format(name, self.counter))
        self.counter += 1

        output = self.bottleneck(
            torch.cat([value_feature, high_feature], dim=1))
        return output


class CCCrossLayerAttentionB(nn.Module):

    def __init__(self, channels, norm="BN"):
        super(CCCrossLayerAttentionB, self).__init__()
        # 128, tune
        in_channels = channels // 2
        inter_channels = channels // 16
        self.attention = CrissCrossAttentionOps(in_channels, inter_channels)
        self.conv1 = nn.Conv2d(channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, in_channels, kernel_size=1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+channels, channels,
                      kernel_size=1, dilation=1, bias=False),
            get_norm(norm, channels),
            nn.ReLU(True)
        )
        self.counter = 1

    def forward(self, low_feature, high_feature):
        # global counter
        out_size = low_feature.size()[2:]
        high_feature = F.interpolate(
            high_feature, out_size, mode='bilinear', align_corners=False)
        query_feature = self.conv1(low_feature)
        value_feature = self.conv2(high_feature)
        for _ in range(2):
            value_feature = self.attention(
                query_feature, value_feature, value_feature)
        output = self.bottleneck(
            torch.cat([value_feature, high_feature], dim=1))
        return output


class CCCrossLayerAttentionC(nn.Module):

    def __init__(self, channels, norm="BN"):
        super(CCCrossLayerAttentionC, self).__init__()
        # 128, tune
        in_channels = channels // 2
        inter_channels = channels // 16
        self.attention = CrissCrossAttentionOps(in_channels, inter_channels)
        self.conv1 = nn.Conv2d(channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, in_channels, kernel_size=1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+channels, channels,
                      kernel_size=1, dilation=1, bias=False),
            get_norm(norm, channels),
            nn.ReLU(True)
        )
        self.counter = 1

    def forward(self, low_feature, high_feature):
        # global counter
        out_size = low_feature.size()[2:]
        high_feature = F.interpolate(
            high_feature, out_size, mode='bilinear', align_corners=False)
        query_feature = self.conv1(high_feature)
        value_feature = self.conv2(high_feature)
        for _ in range(2):
            value_feature = self.attention(
                query_feature, value_feature, value_feature)
        output = self.bottleneck(
            torch.cat([value_feature, high_feature], dim=1))
        return output


class SelfAttention2D(nn.Module):

    def __init__(self, in_channels, inter_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=inter_channels, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=inter_channels, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1
        )
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = inter_channels ** -0.5

    def forward(self, query, key, value):
        B, _, H, W = query.size()
        q = self.query_conv(query).view(
            B, -1, H * W).transpose(1, 2)  # BxNxC, N=HxW
        k = self.key_conv(key).view(B, -1, H * W)  # BxCxN
        v = self.value_conv(value).view(B, -1, H * W)
        weight = self.softmax((q @ k) * self.scale)
        output = weight @ v.transpose(1, 2)
        output = output.contiguous().view(B, -1, H, W)
        return output * self.gamma + value


class NLCrossLayerAttention(nn.Module):
    # non-local
    def __init__(self, channels, norm="BN"):
        super(NLCrossLayerAttention, self).__init__()
        # 128, tune
        in_channels = channels // 2
        inter_channels = channels // 16
        self.attention = SelfAttention2D(in_channels, inter_channels)
        self.conv1 = nn.Conv2d(2 * channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, in_channels, kernel_size=1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+channels, channels,
                      kernel_size=1, dilation=1, bias=False),
            get_norm(norm, channels),
            nn.ReLU(True)
        )
        # self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, low_feature, high_feature):
        # global counter
        out_size = low_feature.size()[2:]
        high_feature = F.interpolate(
            high_feature, out_size, mode='bilinear', align_corners=False)
        query_feature = self.conv1(
            torch.cat([high_feature, low_feature], dim=1))
        value_feature = self.conv2(high_feature)
        value_feature = self.attention(
            query_feature, value_feature, value_feature)
        # value_feature = self.attention(
        #     query_feature, value_feature, value_feature)
        output = self.bottleneck(
            torch.cat([value_feature, high_feature], dim=1))
        return output


class BilinearCrossLayerFusion(nn.Module):
    # bilinear
    def __init__(self, channels, norm="BN"):
        super(BilinearCrossLayerFusion, self).__init__()
        # 128, tune
        in_channels = channels // 2
        self.conv2 = nn.Conv2d(channels, in_channels, kernel_size=1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+channels, channels,
                      kernel_size=1, dilation=1, bias=False),
            get_norm(norm, channels),
            nn.ReLU(True)
        )

    def forward(self, low_feature, high_feature):
        # global counter
        out_size = low_feature.size()[2:]
        high_feature = F.interpolate(
            high_feature, out_size, mode='bilinear', align_corners=False)
        value_feature = self.conv2(high_feature)
        output = self.bottleneck(
            torch.cat([value_feature, high_feature], dim=1))
        return output


class AlginCrossLayerFusion(nn.Module):
    # Guided Upsample Module / Align
    # bilinear
    def __init__(self, channels, norm="BN"):
        super(AlginCrossLayerFusion, self).__init__()
        # 128, tune
        # in_channels = channels // 2
        self.align = AlignedModule(channels, channels)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels, channels,
                      kernel_size=1, dilation=1, bias=False),
            get_norm(norm, channels),
            nn.ReLU(True)
        )

    def forward(self, low_feature, high_feature):
        high_feature = self.align((low_feature, high_feature))
        output = self.bottleneck(high_feature + low_feature)
        return output


# class PixelShuffleCrossLayerFusion(nn.Module):
#     def __init__(self, channels, norm="BN", ratio=2):
#         super(PixelShuffleCrossLayerFusion, self).__init__()
#         from mmcv.ops import CARAFEPack
#         # 128, tune
#         # in_channels = channels // 2
#         # inter_channels = channels // 16
#         self.up_conv = nn.Conv2d
#         # self.carafe = CARAFEPack(channels, ratio)
#         # self.conv1 = nn.Conv2d(2 * channels, in_channels, kernel_size=1)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(channels+channels, channels,
#                       kernel_size=1, dilation=1, bias=False),
#             get_norm(norm, channels),
#             nn.ReLU(True)
#         )

#     def forward(self, low_feature, high_feature):
#         # global counter
#         out_size = low_feature.size()[2:]
#         high_feature = self.carafe(high_feature)
#         value_feature = self.conv2(high_feature)
#         output = self.bottleneck(
#             torch.cat([value_feature, high_feature], dim=1))
#         return output


class CAFARECrossLayerFusion(nn.Module):
    #
    def __init__(self, channels, norm="BN", ratio=2):
        super(CAFARECrossLayerFusion, self).__init__()
        from mmcv.ops import CARAFEPack
        # 128, tune
        # in_channels = channels // 2
        # inter_channels = channels // 16
        self.carafe = CARAFEPack(channels, ratio)
        # self.conv1 = nn.Conv2d(2 * channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels+channels, channels,
                      kernel_size=1, dilation=1, bias=False),
            get_norm(norm, channels),
            nn.ReLU(True)
        )

    def forward(self, low_feature, high_feature):
        # global counter
        # out_size = low_feature.size()[2:]
        # print(high_feature.size())
        high_feature = self.carafe(high_feature)
        #  print(high_feature.size())
        value_feature = self.conv2(high_feature)
        output = self.bottleneck(
            torch.cat([value_feature, high_feature], dim=1))
        return output
