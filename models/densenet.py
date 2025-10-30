import torch
import torch.nn as nn

from core.model_base import Model


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout=0.1):
        super(DenseLayer, self).__init__()
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
        super(DenseBlock, self).__init__()
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
        super(TransitionLayer, self).__init__()
        out_channels = int(in_channels * compression)
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)


class DenseNetArchitecture(nn.Module):
    def __init__(self, num_class, block_config, growth_rate=32, compression=0.5, dropout=0.1, small_input=False):
        super(DenseNetArchitecture, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_class = num_class
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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_class)

        self.to(self.device)
        self.apply(self.reset_weights)

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

    def forward(self, x):
        x = self.features(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_bn(x)
        x = torch.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DenseNetModel(Model):
    def __init__(self, block_config=[6, 12, 24, 16], growth_rate=32, compression=0.5, dropout=0.1, image_size=(224, 224), **kwargs):
        self.name = 'DenseNet'
        self.block_config = block_config
        self.growth_rate = growth_rate
        self.compression = compression
        self.dropout = dropout
        self.small_input = image_size[0] <= 64
        super().__init__(**kwargs)

    def build_model(self):
        print(f"Building DenseNet with block config: {self.block_config}, growth_rate: {self.growth_rate}")
        return DenseNetArchitecture(
            num_class=self.num_classes,
            block_config=self.block_config,
            growth_rate=self.growth_rate,
            compression=self.compression,
            dropout=self.dropout,
            small_input=self.small_input
        )
    
    def get_model_specific_params(self):
        return {
            "num_classes": self.num_classes,
            "block_config": self.block_config,
            "growth_rate": self.growth_rate,
            "compression": self.compression,
            "dropout": self.dropout,
        }
