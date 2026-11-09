import inspect
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from core.ssl.augmentations import TwoViewTransform

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Noms de split essayés, dans l'ordre, pour les datasets torchvision qui ont un paramètre `split`
SPLIT_CANDIDATES = {
    "train": ("train",),
    "val": ("val", "test", "valid"),
    "ssl": ("unlabeled", "train"),
}

# -------------------------------------------------
# Registre des datasets particuliers
# -------------------------------------------------
_REGISTRY: Dict[str, Callable] = {}


def register_dataset(name: str):
    """
    Enregistre un builder `fn(root_dir, split, transform, **kwargs) -> Dataset` pour un dataset donné.
    `split` vaut 'train', 'val' ou 'ssl' ; le split 'ssl' doit renvoyer des paires (image, -1).
    """
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name.upper()] = fn
        return fn
    return decorator


class UnlabeledImageFolder(torch.utils.data.Dataset):
    """Images sans labels (dossier plat ou arborescence quelconque), pour le pré-entraînement SSL."""

    def __init__(self, root: str, transform=None):
        self.paths = sorted(str(p) for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.paths:
            raise FileNotFoundError(f"Aucune image trouvée dans '{root}'")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, -1                                     # pas de label


def supports_download(dataset_class):
    sig = inspect.signature(dataset_class.__init__)
    return 'download' in sig.parameters

# -------------------------------------------------
# Utility to compute per‑channel mean & std
# -------------------------------------------------
def compute_mean_std(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[List[float], List[float]]:
    """
    Computes the per‑channel mean and standard deviation of a dataset.

    The dataset must already return ``(image, target)`` pairs where *image* is a ``torch.Tensor``
    of shape ``[C, H, W]`` (3 channels, values in [0, 1], i.e. resized + ``ToTensor`` only).

    Returns
    -------
    mean, std : lists of 3 floats (R, G, B).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n_pixels = 0
    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        n_pixels += b * h * w
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
    """
    Returns (source, root_dir, dataset_name). `source` is a registered builder, a torchvision
    dataset class or ImageFolder (priority in this order).
    """
    name = dataset_name.upper()
    if name in _REGISTRY:
        source = _REGISTRY[name]
    else:
        try:
            source = get_dataset_class(name)
        except ValueError:
            print(f"[Info] Using ImageFolder for custom dataset '{name}'.")
            source = datasets.ImageFolder
    root_dir = os.path.join(root_dir, name)
    os.makedirs(root_dir, exist_ok=True)
    return source, root_dir, name


# -------------------------------------------------
# Construction d'un split logique ('train', 'val', 'ssl')
# -------------------------------------------------
def _build_folder(source, root_dir, split: str, transform):
    """ImageFolder-like : root/train, root/valid et, pour 'ssl', root/unlabeled s'il existe (sinon root/train)."""
    if split == "ssl":
        unlabeled = os.path.join(root_dir, "unlabeled")
        if os.path.isdir(unlabeled):
            return UnlabeledImageFolder(unlabeled, transform)
    folder = "valid" if split == "val" else "train"
    return source(os.path.join(root_dir, folder), transform=transform)


def _build_torchvision(dataset_class, root_dir, split: str, transform, **kwargs):
    """Dataset torchvision : flag `train`, ou paramètre `split` dont le nom est deviné (SPLIT_CANDIDATES)."""
    params = inspect.signature(dataset_class.__init__).parameters
    if "train" in params:                                  # CIFAR10, CIFAR100, MNIST...
        return dataset_class(root=root_dir, train=(split != "val"), download=True,
                             transform=transform, **kwargs)
    if "split" in params:                                  # STL10, Imagenette...
        last_error = None
        for name in SPLIT_CANDIDATES[split]:
            try:
                return dataset_class(root=root_dir, split=name, download=True, transform=transform, **kwargs)
            except (ValueError, AssertionError) as e:      # nom de split invalide pour ce dataset
                last_error = e
        raise ValueError(f"{dataset_class.__name__} : aucun split valide pour '{split}' parmi "
                         f"{SPLIT_CANDIDATES[split]} ({last_error}). "
                         f"Enregistre un builder avec @register_dataset.")
    return dataset_class(root=root_dir, download=True, transform=transform, **kwargs)   # pas de notion de split


def _make_dataset(source, root_dir, split: str, transform, **dataset_kwargs):
    """split logique : 'train' (labellisé), 'val' (évaluation) ou 'ssl' (pré-entraînement)."""
    if split not in ("train", "val", "ssl"):
        raise ValueError(f"split inconnu : '{split}' (attendu : 'train', 'val' ou 'ssl')")
    if not inspect.isclass(source):                        # builder enregistré
        return source(root_dir, split, transform, **dataset_kwargs)
    if not supports_download(source):                      # ImageFolder-like
        return _build_folder(source, root_dir, split, transform)
    return _build_torchvision(source, root_dir, split, transform, **dataset_kwargs)


def _resolve_stats(source, root_dir, dataset_name, batch_size, image_size,
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
        # Statistics are always computed on the labeled 'train' split, whatever the usage.
        tf = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
        train_set = _make_dataset(source, root_dir, "train", tf, **dataset_kwargs)
        mean, std = compute_mean_std(train_set, batch_size)
        with open(stats_path, "w") as f:
            json.dump({"mean": mean, "std": std}, f, indent=4)
    print(f"[Stats] {dataset_name} mean={mean}, std={std}")
    return mean, std


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


# -------------------------------------------------
# API publique
# -------------------------------------------------
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
    """Supervised loaders (train, val) for any dataset: torchvision, ImageFolder or registered."""
    source, root_dir, dataset_name = _resolve_dataset(dataset_name, root_dir)
    mean, std = _resolve_stats(source, root_dir, dataset_name, batch_size,
                               image_size, use_computed_stats, **dataset_kwargs)
    train_tf, val_tf = get_transforms(image_size, mean, std)

    train_set = _make_dataset(source, root_dir, "train", train_tf, **dataset_kwargs)
    train_set = _stratified_subset(train_set, train_fraction, seed)
    val_set = _make_dataset(source, root_dir, "val", val_tf, **dataset_kwargs)
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
    """
    Returns (ssl_loader, knn_train_loader, val_loader, class_names):
      - ssl_loader       : 'ssl' split, two augmented views per image, labels ignored
      - knn_train_loader : labeled 'train' split without augmentation (kNN bank)
      - val_loader       : 'val' split without augmentation
    """
    source, root_dir, dataset_name = _resolve_dataset(dataset_name, root_dir)
    mean, std = _resolve_stats(source, root_dir, dataset_name, batch_size,
                               image_size, use_computed_stats, **dataset_kwargs)
    _, eval_tf = get_transforms(image_size, mean, std)
    two_view = TwoViewTransform(aug_cfg, image_size, mean, std)

    ssl_set = _make_dataset(source, root_dir, "ssl", two_view, **dataset_kwargs)
    knn_train_set = _make_dataset(source, root_dir, "train", eval_tf, **dataset_kwargs)
    val_set = _make_dataset(source, root_dir, "val", eval_tf, **dataset_kwargs)

    # True when the SSL pool is the labeled train split itself (no dedicated unlabeled pool).
    same_pool = type(ssl_set) is type(knn_train_set) and len(ssl_set) == len(knn_train_set)

    if limit and knn_fraction < 1.0:
        raise ValueError("limit and knn_fraction cannot be used together; choose one")

    if limit:
        idx = np.random.default_rng(seed).permutation(len(ssl_set))[:limit].tolist()
        ssl_set = torch.utils.data.Subset(ssl_set, idx)
        if same_pool:                                   # same subset for the kNN bank
            knn_train_set = torch.utils.data.Subset(knn_train_set, idx)

    # The kNN bank is restricted to the labeled images defined by the protocol (pretraining uses all images).
    knn_train_set = _stratified_subset(knn_train_set, knn_fraction, seed)

    kw = _loader_kwargs(num_workers)
    ssl_loader = DataLoader(ssl_set, batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    knn_train_loader = DataLoader(knn_train_set, batch_size=512, shuffle=False, **kw)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, **kw)

    print(f"[DATA] ssl_train={len(ssl_set)} | knn_bank={len(knn_train_set)} | val={len(val_set)}")
    return ssl_loader, knn_train_loader, val_loader, getattr(val_set, "classes", None)


# =================================================
# Datasets particuliers (builders enregistrés)
# =================================================
class STL10Unlabeled(torch.utils.data.Dataset):
    """STL-10 'unlabeled' (100 000 images 96x96) lu par memmap : pas de copie de 2,8 Go par worker."""

    def __init__(self, root_dir: str, transform=None):
        path = os.path.join(root_dir, "stl10_binary", "unlabeled_X.bin")
        if not os.path.exists(path):                       # premier lancement : déclenche le téléchargement
            datasets.STL10(root=root_dir, split="train", download=True)
        self.path, self.transform = path, transform
        self.n = os.path.getsize(path) // (3 * 96 * 96)
        self._data = None                                  # ouvert dans chaque worker, jamais picklé

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._data is None:
            self._data = np.memmap(self.path, dtype=np.uint8, mode="r", shape=(self.n, 3, 96, 96))
        img = Image.fromarray(np.ascontiguousarray(np.transpose(self._data[i], (2, 1, 0))))
        if self.transform is not None:
            img = self.transform(img)
        return img, -1                                     # pas de label


@register_dataset("STL10")
def _build_stl10(root_dir, split, transform, **kwargs):
    """Optimisation mémoire : le split non labellisé est lu par memmap. Sans ce builder, STL-10 fonctionne quand
    même via torchvision (split 'unlabeled' détecté automatiquement) mais chaque worker en copie 2,8 Go."""
    if split == "ssl":
        return STL10Unlabeled(root_dir, transform)
    return _build_torchvision(datasets.STL10, root_dir, split, transform, **kwargs)