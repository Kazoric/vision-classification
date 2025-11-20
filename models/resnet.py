import torch
import torch.nn as nn

from core.model_base import Model


class BasicBlock(nn.Module):
    expansion = 1 

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, dropout=0.1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.dropout = nn.Dropout(dropout)

        self.a = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        identity = x

        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        # add identity
        x = self.dropout(x)
        x += self.a * identity
        x = torch.relu(x)

        return x
    
class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, dropout=0.1):
        super(BottleneckBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.dropout = nn.Dropout(dropout)

        self.a = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        identity = x

        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        # add identity
        x = self.dropout(x)
        x += self.a * identity
        x = torch.relu(x)
        
        return x

    
class ResNetArchitecture(nn.Module):
    def __init__(self, num_class, layer_list, block=BottleneckBlock, small_input=False, dropout=0.1, **param):
        super(ResNetArchitecture, self).__init__()
        self.block = block
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Generic parameters
        self.num_class = num_class

        # Model hyperparameters
        self.layer_list = layer_list

        if small_input:
            self.in_channels = 16
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.max_pool = nn.Identity()
        else:
            self.in_channels = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.batch_norm1 = nn.BatchNorm2d(self.in_channels)

        self.layers = nn.ModuleList()
        planes = 64 if not small_input else 16

        for i, num_blocks in enumerate(self.layer_list):
            stride = 1 if i == 0 else 2
            layer = self._make_layer(self.block, num_blocks, planes, stride=stride, dropout=dropout)
            self.layers.append(layer)
            planes *= 2
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, self.num_class)

        self.apply(self.reset_weights)
        self.to(self.device)
    
    def reset_weights(self, m):
        """
        Initialize weights and bias for all conv layers.
        """
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #nn.init.constant_(m.bias, 0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        
    def forward(self, x, mask=None):
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, block, blocks, planes, stride=1, dropout=0.1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*block.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*block.expansion)
            )
            
        layers.append(block(self.in_channels, planes, i_downsample=ii_downsample, stride=stride, dropout=dropout))
        self.in_channels = planes*block.expansion
        
        for i in range(blocks-1):
            layers.append(block(self.in_channels, planes, dropout=dropout))
            
        return nn.Sequential(*layers)
    


class ResNetModel(Model):
    def __init__(self, layer_list=[3, 4, 6, 3], block='Bottleneck', image_size=(224, 224), dropout=0.1, **kwargs):
        self.name = 'ResNet'
        self.layer_list = layer_list
        self.block_str = block
        self.image_size = image_size
        self.small_input = self.image_size[0] <= 64
        self.dropout = dropout

        if block == 'Bottleneck':
            self.block = BottleneckBlock
        elif block == 'Basic':
            self.block = BasicBlock
        else:
            raise ValueError(f"Unknown block type: {block}")
        
        super().__init__(**kwargs)

    def build_model(self):
        print(f"Building ResNet with {len(self.layer_list)} stages: {self.layer_list}")
        return ResNetArchitecture(
            num_class=self.num_classes,
            layer_list=self.layer_list,
            block=self.block,
            small_input=self.small_input,
            dropout=self.dropout
        )
    
    def get_model_specific_params(self):
        return {
            "num_classes": self.num_classes,
            "image_size": self.image_size,
            "layer_list": self.layer_list,
            "block": self.block_str,
            "dropout": self.dropout
        }
    
    def get_target_layer(self):
        return self.model.layers[-1][-1]