import numpy as np
import os
import pickle
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    mean = 0.
    std = 0.
    nb_samples = 0.

    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)  # (B, C, H*W)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std

class CIFAR100CustomDataset(Dataset):
    def __init__(self, data_path, train=True, transform=None):
        file = 'train' if train else 'test'
        file_path = os.path.join(data_path, file)
        data_dict = unpickle(file_path)

        self.data = data_dict[b'data']
        self.labels = data_dict[b'coarse_labels']
        self.transform = transform

        # Convertir les données en images (N, 3, 32, 32)
        self.data = self.data.reshape(-1, 3, 32, 32).astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index].transpose(1, 2, 0)  # Convertir en HWC pour PIL
        label = self.labels[index]

        # Convertir en PIL.Image pour les transforms de torchvision
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



def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        # images = np.transpose(images, (0, 3, 2, 1))
        return images
    

class STL10CustomDataset(Dataset):
    def __init__(self, file_path, train=True, transform=None):
        file = 'train' if train else 'test'
        data_path = os.path.join(file_path, f'{file}_X.bin')
        labels_path = os.path.join(file_path, f'{file}_y.bin')
        self.data = read_all_images(data_path)
        self.labels = read_labels(labels_path)

        self.transform = transform

        # Convertir les données en images (N, 3, 96, 96)
        self.data = self.data.reshape(-1, 3, 96, 96).astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index].transpose(1, 2, 0)  # Convertir en HWC pour PIL
        label = self.labels[index] - 1

        # Convertir en PIL.Image pour les transforms de torchvision
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label
    

def get_custom_stl10_dataloaders(data_path, batch_size=64, num_workers=0):
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

    train_dataset = STL10CustomDataset(data_path, train=True, transform=transform_train)
    test_dataset = STL10CustomDataset(data_path, train=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader