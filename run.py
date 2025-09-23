from resnet import ResNetModel

import numpy as np
import pickle
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     X = dict[b'data']
#     Y = dict[b'fine_labels']
#     X = X.reshape(-1, 3, 32, 32).transpose(0,2,3,1)  # NCHW -> NHWC
#     return X, Y

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = unpickle('./cifar-100-python/train')

class CIFAR100CustomDataset(Dataset):
    def __init__(self, data_path, train=True, transform=None):
        file = 'train' if train else 'test'
        file_path = os.path.join(data_path, file)
        data_dict = unpickle(file_path)

        self.data = data_dict[b'data']
        self.labels = data_dict[b'fine_labels']
        self.transform = transform

        # Convertir les données en images (N, 3, 32, 32)
        self.data = self.data.reshape(-1, 3, 32, 32).astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index].transpose(1, 2, 0)  # Convertir en HWC pour PIL
        label = self.labels[index]

        # Convertir en PIL.Image pour les transforms de torchvision
        from PIL import Image
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label
    

def get_custom_cifar100_dataloaders(data_path, batch_size=64, num_workers=0):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2762))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2762))
    ])

    train_dataset = CIFAR100CustomDataset(data_path, train=True, transform=transform_train)
    test_dataset = CIFAR100CustomDataset(data_path, train=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
    

train_loader, test_loader = get_custom_cifar100_dataloaders(
    data_path='./cifar-100-python',
    batch_size=512
)

model = ResNetModel(num_class=100)

model.train(train_loader, test_loader, epochs=10)