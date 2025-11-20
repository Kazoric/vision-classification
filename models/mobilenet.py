import torch
import torch.nn as nn

from core.model_base import Model


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


class MobileNetArchitecture(nn.Module):
    def __init__(self, num_classes=100, use_batchnorm=True, width_multiplier=1.0, small_input=False):
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

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c(1024), num_classes)

        self.to(self.device)
        self.apply(self.reset_weights)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def reset_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)


class MobileNetModel(Model):
    def __init__(self, use_batchnorm=True, width_multiplier=1.0, image_size=(224,224), **kwargs):
        self.name = 'MobileNet'
        self.width_multiplier = width_multiplier
        self.use_batchnorm = use_batchnorm
        self.image_size = image_size
        self.small_input = self.image_size[0] <= 64
        super().__init__(**kwargs)

    def build_model(self):
        # Tu peux configurer les paramètres du MobileNet ici
        return MobileNetArchitecture(
            num_classes=self.num_classes,
            width_multiplier=self.width_multiplier,
            use_batchnorm=self.use_batchnorm,
            small_input=self.small_input
        )
    
    def get_model_specific_params(self):
        return {
            "num_classes": self.num_classes,
            "image_size": self.image_size,
            "width_multiplier": self.width_multiplier,
            "use_batchnorm": self.use_batchnorm,
        }
    
    def get_target_layer(self):
        last_block = self.model.features[-1]
        
        for layer in reversed(last_block.block):
            if isinstance(layer, nn.Conv2d):
                return layer
            
        return last_block