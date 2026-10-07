import torch
from torch import nn
from typing import List, Dict, Optional, Tuple

from .base import BaseBackbone
from .registry import register_backbone

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return self.relu(out + identity)

class Bottleneck(nn.Module):
    """ Bloc ResNet Bottleneck standard """
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        return self.relu(out + identity)

@register_backbone("resnet")
class ResNetBackbone(BaseBackbone):

    def __init__(self, block="basic", layers=(2, 2, 2, 2), stem="cifar", **kwargs):
        super().__init__()
        block_cls = {"basic": BasicBlock, "bottleneck": Bottleneck}[block]
        self.in_planes = 64

        if stem == "cifar":        # petites images (32x32)
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.maxpool = nn.Identity()
        elif stem == "imagenet":   # grandes images (>= 128px)
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.maxpool = nn.MaxPool2d(3, 2, 1)
        else:
            raise ValueError(f"stem inconnu : {stem}")
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        widths, strides = (64, 128, 256, 512), (1, 2, 2, 2)
        self.stages = nn.Sequential(*[
            self._make_layer(block_cls, w, n, s)
            for w, n, s in zip(widths, layers, strides)
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self._feature_dim = 512 * block_cls.expansion
        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        layers += [block(self.in_planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.stages(x)
        return self.pool(x).flatten(1)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim