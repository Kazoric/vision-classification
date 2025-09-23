import torch
import torch.nn as nn

from model_base import Model


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, dropout=0.1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        # self.relu = nn.ReLU()
        self.a = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x.clone()
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        #x += self.dropout(self.a*identity)
        x += identity
        x = torch.relu(x)
        
        return x

    
class ResNetArchitecture(nn.Module):
    def __init__(self, num_class, layer_list, **param):
        super(ResNetArchitecture, self).__init__()
        ResBlock = Bottleneck
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Generic parameters
        self.num_class = num_class

        # Model hyperparameters
        self.layer_list = layer_list

        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, self.layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, self.layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, self.layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, self.layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, self.num_class)

        self.to(self.device)

        self.apply(self.reset_weights)
    
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

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)
    


class ResNetModel(Model):
    def __init__(self, num_class=100, layer_list=[3, 4, 6, 3], **kwargs):
        self.num_class = num_class
        self.layer_list = layer_list
        super().__init__(**kwargs)

    def build_model(self):
        # Tu peux configurer les paramètres du ResNet ici
        return ResNetArchitecture(
            num_class=self.num_class,
            layer_list=[3, 4, 6, 3],
        )