import torch
import torch.nn as nn

from simpleseg.models.layers.norm import get_norm
from .resnet import BasicBlock

expansion = BasicBlock.expansion


def build_dfnet(cfg):
    norm = cfg.MODEL.NORM
    stride = cfg.MODEL.DFNET.STRIDE
    pretrained = cfg.MODEL.PRETRAINED
    model = DFNetV2(norm, stride)
    if len(pretrained) > 0:
        model.load_state_dict(torch.load(
            pretrained, map_location='cpu'), strict=False)

    return model


class DFNetV2(nn.Module):

    def __init__(self, norm="BN", stride=32):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2, bias=False),
            get_norm(norm, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            get_norm(norm, 64),
            nn.ReLU(inplace=True)
        )
        self.inplanes = 64
        self.stage2_1 = self._make_layer(64, 2, stride=2, norm=norm)
        self.stage2_2 = self._make_layer(128, 1, stride=1, norm=norm)
        self.stage3_1 = self._make_layer(128, 10, stride=2, norm=norm)
        self.stage3_2 = self._make_layer(256, 1, stride=1, norm=norm)
        self.stage4_1 = self._make_layer(256, 4, stride=2, norm=norm)
        if stride == 32:
            self.stage4_2 = self._make_layer(512, 2, stride=1, norm=norm)
        elif stride == 64:
            self.stage4_2 = self._make_layer(512, 2, stride=2, norm=norm)
        self.output_channels = 512

    def _make_layer(self, planes, blocks, stride=1, norm="BN"):
        downsample = None
        outplanes = planes * expansion
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                get_norm(norm, outplanes),
            )
        layers = []
        layers.append(
            BasicBlock(self.inplanes, planes, stride, norm=norm, downsample=downsample))
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, norm=norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)  # 4x32
        x = self.stage2_1(x)  # 8x64
        x3 = self.stage2_2(x)  # 8x64
        x4 = self.stage3_1(x3)  # 16x128
        x4 = self.stage3_2(x4)  # 16x128
        x5 = self.stage4_1(x4)  # 32x256
        x5 = self.stage4_2(x5)  # 32x256
        return x3, x4, x5
