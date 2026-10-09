import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, Optional, List
import inspect
import json

from core.ssl.augmentations import TwoViewTransform


def supports_download(dataset_class):
    sig = inspect.signature(dataset_class.__init__)
    return 'download' in sig.parameters

# -------------------------------------------------
# Utility to compute per‑channel mean & std
# -------------------------------------------------
def compute_mean_std(
    dataset_class: torch.utils.data.Dataset,
    root_dir: str,
    batch_size: int = 64,
    image_size: Tuple[int, int] = (224, 224),
    **dataset_kwargs
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

    if supports_download(dataset_class):
        dataset = dataset_class(root=root_dir, download=True, transform=transform, **dataset_kwargs)
    else:
        dataset = dataset_class(root=root_dir, transform=transform, **dataset_kwargs)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

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

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=image_size[0] // 8),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, val_transform

def get_dataset_class(dataset_name: str):
    """Try to fetch a torchvision dataset class by name."""
    dataset_name = dataset_name.lower()
    for name in dir(datasets):
        cls = getattr(datasets, name)
        if inspect.isclass(cls) and name.lower() == dataset_name:
            return cls
    raise ValueError(f"Dataset '{dataset_name}' not found in torchvision.datasets.")


def _resolve_dataset(dataset_name: str, root_dir: str):
    dataset_name = dataset_name.upper()
    try:
        dataset_class = get_dataset_class(dataset_name)
    except ValueError:
        print(f"[Info] Using ImageFolder for custom dataset '{dataset_name}'.")
        dataset_class = datasets.ImageFolder
    root_dir = os.path.join(root_dir, dataset_name)
    os.makedirs(root_dir, exist_ok=True)
    return dataset_class, root_dir, dataset_name


def _resolve_stats(dataset_class, root_dir, dataset_name, batch_size, image_size,
                   use_computed_stats, **dataset_kwargs):
    """Returns (mean, std): computed/cached values or default ImageNet values."""
    if not use_computed_stats:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    stats_path = os.path.join(root_dir, "stats.json")
    if os.path.exists(stats_path):
        print(f"[Stats] Loading existing stats from {stats_path}")
        with open(stats_path, "r") as f:
            stats = json.load(f)
        mean, std = stats["mean"], stats["std"]
    else:
        # For ImageFolder, compute statistics using only the training set
        stats_root = root_dir if supports_download(dataset_class) else os.path.join(root_dir, "train")
        mean, std = compute_mean_std(dataset_class, stats_root, batch_size, image_size, **dataset_kwargs)
        with open(stats_path, "w") as f:
            json.dump({"mean": mean, "std": std}, f, indent=4)
    print(f"[Stats] {dataset_name} mean={mean}, std={std}")
    return mean, std


def _make_dataset(dataset_class, root_dir, split: str, transform, **dataset_kwargs):
    """split : 'train' or 'val'."""
    if not supports_download(dataset_class):             # ImageFolder : root/train, root/valid
        folder = "train" if split == "train" else "valid"
        return dataset_class(os.path.join(root_dir, folder), transform=transform)

    params = inspect.signature(dataset_class.__init__).parameters
    kw = dict(dataset_kwargs)
    if "train" in params:                                # CIFAR10, CIFAR100, etc.
        kw["train"] = split == "train"
    elif "split" in params:                              # Imagenette, etc. (STL10 uses 'test' instead of 'val')
        kw["split"] = "train" if split == "train" else "val"
    return dataset_class(root=root_dir, download=True, transform=transform, **kw)


def _get_targets(dataset) -> np.ndarray:
    """Returns the labels of a torchvision dataset (attributes vary by class)."""
    for attr in ("targets", "labels", "_labels"):
        if hasattr(dataset, attr):
            return np.asarray(getattr(dataset, attr))
    for attr in ("samples", "_samples"):               # ImageFolder, Imagenette, etc.
        if hasattr(dataset, attr):
            return np.asarray([s[1] for s in getattr(dataset, attr)])
    raise AttributeError(f"Impossible de trouver les labels de {type(dataset).__name__}")


def _stratified_subset(dataset, fraction: float, seed: int):
    """
    Keeps `fraction` of the images from each class. For the same seed, the subsets
    are nested (the 1% subset is included in the 10% subset), ensuring fair comparisons.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"train_fraction must be in ]0, 1], received {fraction}")
    if fraction == 1.0:
        return dataset

    targets = _get_targets(dataset)
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(targets):
        cls_idx = rng.permutation(np.where(targets == c)[0])
        idx += cls_idx[:max(1, round(fraction * len(cls_idx)))].tolist()
    idx.sort()

    subset = torch.utils.data.Subset(dataset, idx)
    subset.classes = getattr(dataset, "classes", None)
    subset.targets = targets[idx].tolist()
    counts = np.bincount(targets[idx])
    print(f"[DATA] train_fraction={fraction}: {len(idx)} labeled images "
          f"(per class: min={counts.min()}, max={counts.max()})")
    return subset


def _loader_kwargs(num_workers: int, prefetch_factor: int = 4,
                   persistent_workers: bool = True) -> dict:
    kw = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:                       # These options cause errors when num_workers=0
        kw.update(persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
    return kw


def get_torchvision_dataset(
    dataset_name: str,
    root_dir: str = './data',
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    use_computed_stats: bool = False,
    train_fraction: float = 1.0,       # Fraction of training labels to use (stratified)
    seed: int = 42,                    # Random seed for subset sampling
    persistent_workers: bool = True,   # Set to False when running many consecutive experiments
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    dataset_class, root_dir, dataset_name = _resolve_dataset(dataset_name, root_dir)
    mean, std = _resolve_stats(dataset_class, root_dir, dataset_name, batch_size,
                               image_size, use_computed_stats, **dataset_kwargs)
    train_tf, val_tf = get_transforms(image_size, mean, std)

    train_set = _make_dataset(dataset_class, root_dir, "train", train_tf, **dataset_kwargs)
    train_set = _stratified_subset(train_set, train_fraction, seed)
    val_set = _make_dataset(dataset_class, root_dir, "val", val_tf, **dataset_kwargs)
    print(f"Loaded dataset '{dataset_name}' from '{root_dir}'.")

    # With small datasets, the last batch may be very small, making BatchNorm unstable; discard it.
    # No change when train_fraction=1.0: the baseline remains identical.
    drop_last = train_fraction < 1.0 and len(train_set) > batch_size

    kw = _loader_kwargs(num_workers, persistent_workers=persistent_workers)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=drop_last, **kw)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader


def get_ssl_dataloaders(
    dataset_name: str,
    aug_cfg,
    root_dir: str = './data',
    batch_size: int = 256,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (32, 32),
    use_computed_stats: bool = False,
    limit: Optional[int] = None,        # Random training subset (smoke-test mode)
    knn_fraction: float = 1.0,          # Fraction of labels to use in the kNN bank (stratified)
    seed: int = 42,
    **dataset_kwargs
):
    dataset_class, root_dir, dataset_name = _resolve_dataset(dataset_name, root_dir)
    mean, std = _resolve_stats(dataset_class, root_dir, dataset_name, batch_size,
                               image_size, use_computed_stats, **dataset_kwargs)
    _, eval_tf = get_transforms(image_size, mean, std)
    two_view = TwoViewTransform(aug_cfg, image_size, mean, std)

    ssl_set = _make_dataset(dataset_class, root_dir, "train", two_view, **dataset_kwargs)
    knn_train_set = _make_dataset(dataset_class, root_dir, "train", eval_tf, **dataset_kwargs)
    val_set = _make_dataset(dataset_class, root_dir, "val", eval_tf, **dataset_kwargs)

    if limit and knn_fraction < 1.0:
        raise ValueError("limit and knn_fraction cannot be used together; choose one")

    if limit:      # Smoke-test mode: use the same subset for pretraining and the kNN bank
        idx = np.random.default_rng(seed).permutation(len(ssl_set))[:limit].tolist()
        ssl_set = torch.utils.data.Subset(ssl_set, idx)
        knn_train_set = torch.utils.data.Subset(knn_train_set, idx)

    # The kNN bank is restricted to the labeled images defined by the protocol (pretraining uses all images).
    knn_train_set = _stratified_subset(knn_train_set, knn_fraction, seed)

    kw = _loader_kwargs(num_workers)
    ssl_loader = DataLoader(ssl_set, batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    knn_train_loader = DataLoader(knn_train_set, batch_size=512, shuffle=False, **kw)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, **kw)

    print(f"[DATA] ssl_train={len(ssl_set)} | knn_bank={len(knn_train_set)} | val={len(val_set)}")
    return ssl_loader, knn_train_loader, val_loader, getattr(val_set, "classes", None)