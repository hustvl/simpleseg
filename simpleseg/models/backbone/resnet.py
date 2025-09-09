import torch
import torch.nn as nn

from simpleseg.models.layers.norm import get_norm
from simpleseg.utils.utils import get_rank

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    Basic Block for Resnet
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm="BN", downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = get_norm(norm, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = get_norm(norm, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck Layer for Resnet
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, norm="BN", downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = get_norm(norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = get_norm(norm, planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = get_norm(norm, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Resnet
    """

    def __init__(self, block, layers, norm="BN"):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            conv3x3(3, 64, stride=2),
            get_norm(norm, 64),
            nn.ReLU(inplace=True),
            conv3x3(64, 64),
            get_norm(norm, 64),
            nn.ReLU(inplace=True),
            conv3x3(64, 128))
        self.bn1 = get_norm(norm, 128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm=norm)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, norm=norm)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, norm=norm)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, norm=norm)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.output_channels = block.expansion * 512

    def _make_layer(self, block, planes, blocks, stride=1, norm="BN"):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                get_norm(norm, planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, norm, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm=norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


def build_resnet(cfg):
    name = cfg.MODEL.BACKBONE
    strides = cfg.MODEL.RESNET.STRIDES
    dialtions = cfg.MODEL.RESNET.DILATIONS
    pretrained = cfg.MODEL.PRETRAINED
    norm = cfg.MODEL.NORM

    if name == "resnet18":
        model = ResNet(BasicBlock, [2, 2, 2, 2], norm=norm)
    elif name == "resnet50":
        model = ResNet(Bottleneck, [3, 4, 6, 3], norm=norm)
    elif name == "resnet101":
        model = ResNet(Bottleneck, [3, 4, 23, 3], norm=norm)
    else:
        raise NotImplementedError()
    if len(pretrained) > 0:
        state_dict = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

    d, s = dialtions[0], strides[0]
    for n, m in model.layer3.named_modules():
        # for basic block
        if 'conv1' in n and s == 1:
            m.stride = 1
        if 'conv2' in n:
            m.dilation, m.padding = (d, d), (d, d)
            if s == 1:
                # first block is 2, defualt = 1
                m.stride = 1
        elif 'downsample.0' in n:
            if s == 1:
                # default = 2
                m.stride = 1
    # C5
    d, s = dialtions[1], strides[1]
    for n, m in model.layer4.named_modules():
        # for basic block
        if 'conv1' in n and s == 1:
            m.stride = 1
        if 'conv2' in n:
            m.dilation, m.padding = (d, d), (d, d)
            if s == 1:
                # first block is 2, defualt = 1
                m.stride = 1
        elif 'downsample.0' in n:
            if s == 1:
                # default = 2
                m.stride = 1
    return model
