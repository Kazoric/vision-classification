import torch
import torch.nn as nn

from .base import BaseBackbone
from .registry import register_backbone


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_batchnorm=True):
        super().__init__()
        layers = []

        # Depthwise convolution
        layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.ReLU(inplace=True))

        # Pointwise convolution
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


@register_backbone("mobilenet")
class MobileNetBackbone(BaseBackbone):
    def __init__(self, use_batchnorm=True, width_multiplier=1.0, small_input=False):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_batchnorm = use_batchnorm
        self.width_multiplier = width_multiplier
        self.small_input = small_input

        def c(ch):  # Apply width multiplier and ensure at least 1
            return max(1, int(ch * width_multiplier))

        # Architecture configuration (channel, stride)
        self.cfg = [
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            *[(512, 1) for _ in range(5)],
            (1024, 2),
            (1024, 1)
        ]

        # Initial conv layer
        if small_input:
            first_channels = c(16)
            first_stride = 1
        else:
            first_channels = c(32)
            first_stride = 2

        layers = [
            nn.Conv2d(3, first_channels, kernel_size=3, stride=first_stride, padding=1, bias=False)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(first_channels))
        layers.append(nn.ReLU(inplace=True))

        in_channels = first_channels

        # Depthwise separable layers
        for out_channels, stride in self.cfg:
            layers.append(DepthwiseSeparableConv(in_channels, c(out_channels), stride, use_batchnorm))
            in_channels = c(out_channels)

        self.features = nn.Sequential(*layers)

        self._feature_dim = c(1024)
        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim