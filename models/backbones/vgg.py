import torch
import torch.nn as nn

from .base import BaseBackbone
from .registry import register_backbone


@register_backbone("vgg")
class VGGBackbone(BaseBackbone):
    VGG_CONFIGS = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 
                  512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }

    def __init__(self, variant='VGG16', use_batchnorm=True, **kwargs):
        super(VGGBackbone, self).__init__()
        self.variant = variant.upper()
        self.use_batchnorm = use_batchnorm

        if self.variant not in self.VGG_CONFIGS:
            raise ValueError(f"Unknown VGG variant '{variant}'. Choose from {list(self.VGG_CONFIGS.keys())}.")

        cfg = self.VGG_CONFIGS[self.variant]
        self.layers = self._make_layers(cfg, batch_norm=self.use_batchnorm)

        self._feature_dim = 512
        self._init_weights()

    def forward(self, x):
        x = self.layers(x)
        return x

    def _make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

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