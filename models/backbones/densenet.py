import torch
import torch.nn as nn

from .base import BaseBackbone
from .registry import register_backbone


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout=0.1):
        super().__init__()
        inter_channels = 4 * growth_rate  # Bottleneck

        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False),

            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),

            nn.Dropout(dropout)
        )

    def forward(self, x):
        new_features = self.layer(x)
        return torch.cat([x, new_features], 1)  # Concatenate along channel axis


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                DenseLayer(in_channels + i * growth_rate, growth_rate, dropout)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, compression=0.5):
        super().__init__()
        out_channels = int(in_channels * compression)
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)


@register_backbone("densenet")
class DenseNetBackbone(BaseBackbone):
    def __init__(self, block_config, growth_rate=32, compression=0.5, dropout=0.1, small_input=False):
        super().__init__()

        self.growth_rate = growth_rate

        if small_input:
            self.init_channels = 16
            self.features = nn.Sequential(
                nn.Conv2d(3, self.init_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.init_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.init_channels = 64
            self.features = nn.Sequential(
                nn.Conv2d(3, self.init_channels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(self.init_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        num_channels = self.init_channels
        self.blocks = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_channels, growth_rate, dropout)
            self.blocks.append(block)
            num_channels += num_layers * growth_rate

            if i != len(block_config) - 1:  # no transition after last block
                trans = TransitionLayer(num_channels, compression)
                self.blocks.append(trans)
                num_channels = int(num_channels * compression)

        self.final_bn = nn.BatchNorm2d(num_channels)

        self._feature_dim = num_channels
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_bn(x)
        x = torch.relu(x)
        return x

    @property
    def feature_dim(self) -> int:
        return self._feature_dim