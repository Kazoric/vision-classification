import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import time

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataloader import get_custom_cifar100_dataloaders
from resnet import ResNetModel
from vgg import VGGModel

# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     X = dict[b'data']
#     Y = dict[b'fine_labels']
#     X = X.reshape(-1, 3, 32, 32).transpose(0,2,3,1)  # NCHW -> NHWC
#     return X, Y
    

train_loader, test_loader = get_custom_cifar100_dataloaders(
    data_path='./cifar-100-python',
    batch_size=512
)

# model = ResNetModel(num_class=100, layer_list=[3], block='Bottleneck')
model = VGGModel(num_class=100, variant='VGG11')

training_start = time.time()
model.train(train_loader, test_loader, epochs=10)
training_stop = time.time()
training_time = training_stop - training_start
print(f"Training time: {training_time}")

model.plot_acc_loss()