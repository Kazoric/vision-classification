import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import time

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data_loader import get_custom_cifar100_dataloaders, get_custom_stl10_dataloaders
from models.resnet import ResNetModel
from models.wide_resnet import WideResNetModel
from models.vgg import VGGModel
from models.mobilenet import MobileNetModel
from models.densenet import DenseNetModel

# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     X = dict[b'data']
#     Y = dict[b'fine_labels']
#     X = X.reshape(-1, 3, 32, 32).transpose(0,2,3,1)  # NCHW -> NHWC
#     return X, Y
    

train_loader, test_loader = get_custom_cifar100_dataloaders(
    data_path='./cifar-100-python',
    batch_size=2048
)

# train_loader, test_loader = get_custom_stl10_dataloaders(
#     data_path='./stl10_binary',
#     batch_size=64
# )

# model = ResNetModel(num_class=20, layer_list=[1,2,3,1], block='Bottleneck')
model = WideResNetModel(num_class=20, layer_list=[1,2,1], block='Bottleneck', widen_factor=2)
# model = VGGModel(num_class=100, variant='VGG11')
# model = MobileNetModel(num_class=100)
# model = DenseNetModel(
#     num_class=20, 
#     block_config=[6, 12, 24, 16],  # DenseNet-121
#     growth_rate=32,
#     compression=0.5
# )

training_start = time.time()
model.train(train_loader, test_loader, epochs=50)
training_stop = time.time()
training_time = training_stop - training_start
print(f"Training time: {training_time}")

model.plot_acc_loss()