import torch
import torch.nn as nn

from core.model_base import Model


class VGGArchitecture(nn.Module):
    VGG_CONFIGS = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 
                  512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }

    def __init__(self, num_classes=100, variant='VGG16', use_batchnorm=True, **kwargs):
        super(VGGArchitecture, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.variant = variant.upper()
        self.use_batchnorm = use_batchnorm

        if self.variant not in self.VGG_CONFIGS:
            raise ValueError(f"Unknown VGG variant '{variant}'. Choose from {list(self.VGG_CONFIGS.keys())}.")

        cfg = self.VGG_CONFIGS[self.variant]
        self.features = self._make_layers(cfg, batch_norm=self.use_batchnorm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Optionally (7, 7) for ImageNet

        self.classifier = nn.Sequential(
            # nn.Linear(512, 512),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(512, 512),
            # nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(512, num_classes)
        )

        self.to(self.device)
        self.apply(self.reset_weights)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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

    def reset_weights(self, m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)


class VGGModel(Model):
    def __init__(self, variant='VGG16', use_batchnorm=True, **kwargs):
        self.name = 'VGG'
        self.variant = variant
        self.use_batchnorm = use_batchnorm
        super().__init__(**kwargs)

    def build_model(self):
        print(f"Building {self.variant}")
        # Tu peux configurer les paramètres du VGG ici
        return VGGArchitecture(
            num_classes=self.num_classes,
            variant=self.variant,
            use_batchnorm=self.use_batchnorm
        )
    
    def get_model_specific_params(self):
        return {
            "num_classes": self.num_classes,
            "variant": self.variant,
            "use_batchnorm": self.use_batchnorm,
        }