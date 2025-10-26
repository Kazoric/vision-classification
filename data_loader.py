import os, math
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional, List, Dict, Any
import inspect


def supports_download(dataset_class):
    sig = inspect.signature(dataset_class.__init__)
    return 'download' in sig.parameters

# -------------------------------------------------
# Utility to compute per‑channel mean & std
# -------------------------------------------------
def compute_mean_std(
    dataset: torch.utils.data.Dataset,
    root_dir: str,
    batch_size: int,
    image_size: Tuple[int, int]
) -> Tuple[List[float], List[float]]:
    """
    Computes the per‑channel mean and standard deviation of a dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        A dataset that returns ``(image, target)`` pairs where *image* is
        a ``torch.Tensor`` of shape ``[C, H, W]`` (typically 3‑channel RGB).

    Returns
    -------
    mean : tuple[float, float, float]
        Mean for each channel (R, G, B).
    std  : tuple[float, float, float]
        Standard deviation for each channel (R, G, B).
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    if dataset == datasets.ImageFolder:
        temp_train_set = dataset(root=root_dir, transform=transform)
    else:
        temp_train_set = dataset(root=root_dir, download=True, transform=transform)
    
    loader = DataLoader(temp_train_set, batch_size=batch_size, shuffle=False, num_workers=0)

    n_pixels = 0
    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        pixels = b * h * w
        n_pixels += pixels

        sum_ += images.sum(dim=[0, 2, 3])
        sum_sq += (images ** 2).sum(dim=[0, 2, 3])

    mean = sum_ / n_pixels
    std = (sum_sq / n_pixels - mean ** 2).sqrt()

    return mean.tolist(), std.tolist()

# -------------------------------------------------
# Updated get_transforms that accepts mean/std
# -------------------------------------------------
def get_transforms(
    image_size: Tuple[int, int] = (224,224),
    resize: bool = False,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns a pair of ``Compose`` objects for training & validation.
    If *mean* and *std* are ``None`` the ImageNet defaults are used.
    """
    if mean is None:
        mean = (0.485, 0.456, 0.406)
    if std is None:
        std = (0.229, 0.224, 0.225)

    train_transform_list = []
    val_transform_list = []
    if resize:
        train_transform_list.append(transforms.Resize(image_size))
        val_transform_list.append(transforms.Resize(image_size))

    train_transform_list.extend([
        # transforms.Resize(image_size),   # Uncomment if you need resizing
        transforms.RandomCrop(image_size, padding=image_size[0]//8),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform_list.extend([
        # transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(train_transform_list), transforms.Compose(val_transform_list)

# -------------------------------------------------
# Loading the dataset
# -------------------------------------------------
def get_torchvision_dataset(
    dataset_name: str,
    root_dir: str = './data',
    batch_size: int = 64,
    num_workers: int = 0,
    use_computed_stats: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Loads a standard torchvision dataset (e.g. CIFAR10, CIFAR100).
    If *use_computed_stats* is ``True`` the function will compute
    the mean & std on the training split and use those for
    ``transforms.Normalize``.
    """
    dataset_name = dataset_name.upper()

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Supported: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset_name]
    dataset_class = config['class']
    resize = config['resize']
    # if resize:
    image_size = config['image_size']
    root_dir = os.path.join(root_dir, dataset_name)
    os.makedirs(root_dir, exist_ok=True)

    # Create a temporary loader to compute stats if requested
    if use_computed_stats:
        mean, std = compute_mean_std(dataset_class, root_dir, batch_size, image_size)
        print(f"[Stats] {dataset_name} mean: {mean}, std: {std}")
    else:
        mean, std = None, None

    # Build the actual transforms
    train_transform, val_transform = get_transforms(image_size, resize, mean=mean, std=std)

    try:
        if supports_download(dataset_class):
            train_set = dataset_class(
                root=root_dir,
                download=True,
                transform=train_transform,
                **config['train_args']
            )
            val_set = dataset_class(
                root=root_dir,
                download=True,
                transform=val_transform,
                **config['val_args']
            )
        else:
            # Dataset type ImageFolder ou custom local
            train_root = os.path.join(root_dir, 'train')
            val_root = os.path.join(root_dir, 'val')

            train_set = dataset_class(root=train_root, transform=train_transform)
            val_set = dataset_class(root=val_root, transform=val_transform)
        print(f"Dataset '{dataset_name}' loaded (downloaded if necessary) from '{root_dir}'.")

    except Exception as e:
        print(f"Error loading/downloading dataset {dataset_name}: {e}")
        raise

    # Create the DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


DATASET_CONFIGS = {
    'CIFAR10': {
        'class': datasets.CIFAR10,
        'train_args': {'train': True},
        'val_args': {'train': False},
        'resize': False,
        'image_size': (32, 32)
    },
    'CIFAR100': {
        'class': datasets.CIFAR100,
        'train_args': {'train': True},
        'val_args': {'train': False},
        'resize': False,
        'image_size': (32, 32)
    },
    'IMAGENETTE': {
        'class': datasets.Imagenette,
        'train_args': {'split': 'train'},
        'val_args': {'split': 'val'},
        'resize': True,
        'image_size': (160, 160)
    },
    'IMAGENET': {
        'class': datasets.ImageNet,
        'train_args': {'split': 'train'},
        'val_args': {'split': 'val'},
        'resize': True,
        'image_size': (224, 224)
    },
    'EXAMPLE': {
        'class': datasets.ImageFolder,
        'train_args': {'split': 'train'},
        'val_args': {'split': 'val'},
        'resize': True,
        'image_size': (160, 160)
    }
}