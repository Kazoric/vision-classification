# import numpy as np
# import os
# import pickle
# from PIL import Image
# import torch
# from tqdm import tqdm

# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms


# def compute_mean_std(dataset):
#     """
#     Calcule la moyenne et l'écart-type (par canal) d'un dataset Torch.
#     Le dataset doit retourner des PIL Images (avant ToTensor).
#     """
#     loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     n_samples = 0

#     print("Calcul des moyennes et écarts-types...")
#     for images, _ in tqdm(loader):
#         # images = torch.stack([transforms.ToTensor()(img) for img in images])  # (B, C, H, W)
#         n_samples += images.size(0)
#         mean += images.mean(dim=[0, 2, 3]) * images.size(0)
#         std += images.std(dim=[0, 2, 3]) * images.size(0)

#     mean /= n_samples
#     std /= n_samples

#     return mean.tolist(), std.tolist()

# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict

# class CIFAR100CustomDataset(Dataset):
#     def __init__(self, data_path, train=True, transform=None):
#         file = 'train' if train else 'test'
#         file_path = os.path.join(data_path, file)
#         data_dict = unpickle(file_path)

#         self.data = data_dict[b'data']
#         self.labels = data_dict[b'coarse_labels']
#         self.transform = transform

#         # Convertir les données en images (N, 3, 32, 32)
#         self.data = self.data.reshape(-1, 3, 32, 32).astype(np.uint8)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         image = self.data[index].transpose(1, 2, 0)  # Convertir en HWC pour PIL
#         label = self.labels[index]

#         # Convertir en PIL.Image pour les transforms de torchvision
#         image = Image.fromarray(image)

#         if self.transform:
#             image = self.transform(image)

#         return image, label
    

# def get_custom_cifar100_dataloaders(data_path, batch_size=64, num_workers=0):
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(15),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5071, 0.4865, 0.4409),
#                              (0.2673, 0.2564, 0.2762))
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5071, 0.4865, 0.4409),
#                              (0.2673, 0.2564, 0.2762))
#     ])

#     train_dataset = CIFAR100CustomDataset(data_path, train=True, transform=transform_train)
#     test_dataset = CIFAR100CustomDataset(data_path, train=False, transform=transform_test)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     return train_loader, test_loader

# # def get_custom_cifar100_dataloaders(data_path, batch_size=64, num_workers=0):
# #     transform_temp = transforms.Compose([
# #         transforms.ToTensor()
# #     ])
# #     train_dataset_raw = CIFAR100CustomDataset(data_path, train=True, transform=transform_temp)
# #     mean, std = compute_mean_std(train_dataset_raw)
# #     print(f"Mean: {mean}")
# #     print(f"Std: {std}")

# #     transform_train = transforms.Compose([
# #         transforms.RandomCrop(32, padding=4),
# #         transforms.RandomHorizontalFlip(),
# #         transforms.RandomRotation(15),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean,
# #                              std)
# #     ])

# #     transform_test = transforms.Compose([
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean,
# #                              std)
# #     ])

# #     train_dataset = CIFAR100CustomDataset(data_path, train=True, transform=transform_train)
# #     test_dataset = CIFAR100CustomDataset(data_path, train=False, transform=transform_test)

# #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# #     return train_loader, test_loader



# def read_labels(path_to_labels):
#     """
#     :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
#     :return: an array containing the labels
#     """
#     with open(path_to_labels, 'rb') as f:
#         labels = np.fromfile(f, dtype=np.uint8)
#         return labels


# def read_all_images(path_to_data):
#     """
#     :param path_to_data: the file containing the binary images from the STL-10 dataset
#     :return: an array containing all the images
#     """

#     with open(path_to_data, 'rb') as f:
#         # read whole file in uint8 chunks
#         everything = np.fromfile(f, dtype=np.uint8)

#         # We force the data into 3x96x96 chunks, since the
#         # images are stored in "column-major order", meaning
#         # that "the first 96*96 values are the red channel,
#         # the next 96*96 are green, and the last are blue."
#         # The -1 is since the size of the pictures depends
#         # on the input file, and this way numpy determines
#         # the size on its own.

#         images = np.reshape(everything, (-1, 3, 96, 96))

#         # Now transpose the images into a standard image format
#         # readable by, for example, matplotlib.imshow
#         # You might want to comment this line or reverse the shuffle
#         # if you will use a learning algorithm like CNN, since they like
#         # their channels separated.
#         # images = np.transpose(images, (0, 3, 2, 1))
#         return images
    

# class STL10CustomDataset(Dataset):
#     def __init__(self, file_path, train=True, transform=None):
#         file = 'train' if train else 'test'
#         data_path = os.path.join(file_path, f'{file}_X.bin')
#         labels_path = os.path.join(file_path, f'{file}_y.bin')
#         self.data = read_all_images(data_path)
#         self.labels = read_labels(labels_path)

#         self.transform = transform

#         # Convertir les données en images (N, 3, 96, 96)
#         self.data = self.data.reshape(-1, 3, 96, 96).astype(np.uint8)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         image = self.data[index].transpose(1, 2, 0)  # Convertir en HWC pour PIL
#         label = self.labels[index] - 1

#         # Convertir en PIL.Image pour les transforms de torchvision
#         image = Image.fromarray(image)

#         if self.transform:
#             image = self.transform(image)

#         return image, label
    

# def get_custom_stl10_dataloaders(data_path, batch_size=64, num_workers=0):
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5071, 0.4865, 0.4409),
#                              (0.2673, 0.2564, 0.2762))
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5071, 0.4865, 0.4409),
#                              (0.2673, 0.2564, 0.2762))
#     ])

#     train_dataset = STL10CustomDataset(data_path, train=True, transform=transform_train)
#     test_dataset = STL10CustomDataset(data_path, train=False, transform=transform_test)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     return train_loader, test_loader


import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(image_size):
    """
    Définit les transformations standard pour l'entraînement et la validation.
    """
    # Note: On utilise souvent des transformations d'augmentation plus légères 
    # ou spécifiques pour CIFAR (ex: RandomCrop), mais on garde ici un standard
    # qui inclut un redimensionnement si image_size est > taille native.
    train_transforms = transforms.Compose([
        # transforms.Resize(image_size),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet Norms
    ])
    
    val_transforms = transforms.Compose([
        # transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transforms, val_transforms


def get_torchvision_dataset(
    dataset_name: str, 
    root_dir: str = './data', 
    batch_size: int = 64, 
    num_workers: int = 0, 
    image_size: int = 224
):
    """
    Charge un dataset standard de torchvision (ex: CIFAR10, CIFAR100).
    Télécharge le dataset si nécessaire (s'il n'existe pas déjà dans root_dir).
    """
    dataset_name = dataset_name.upper()
    
    # Mapping des noms de datasets aux classes de torchvision
    TORCHVISION_DATASETS = {
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100,
        # Ajoutez d'autres datasets ici, ex: 'MNIST': datasets.MNIST
    }

    if dataset_name not in TORCHVISION_DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' non supporté. Supportés: {list(TORCHVISION_DATASETS.keys())}")

    dataset_class = TORCHVISION_DATASETS[dataset_name]
    train_transform, val_transform = get_transforms(image_size)

    # Créer le répertoire de données s'il n'existe pas
    os.makedirs(root_dir, exist_ok=True)
    
    # 1. Chargement du jeu d'entraînement
    # L'argument `download=True` assure le téléchargement uniquement si le dataset n'est pas trouvé.
    try:
        train_set = dataset_class(
            root=root_dir, 
            train=True, 
            download=True,  # Télécharge s'il n'existe pas
            transform=train_transform
        )
        # 2. Chargement du jeu de validation
        val_set = dataset_class(
            root=root_dir, 
            train=False, 
            download=True, # Télécharge s'il n'existe pas
            transform=val_transform
        )
        print(f"Dataset '{dataset_name}' chargé (téléchargé si nécessaire) à partir de '{root_dir}'.")

    except Exception as e:
        print(f"Erreur lors du chargement/téléchargement du dataset {dataset_name}: {e}")
        raise

    # 3. Création des DataLoaders
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True # Optimisation PyTorch
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader