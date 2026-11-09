# Vision - Image Classification & Self-Supervised Learning

This project provides a modular and extensible image classification framework using PyTorch. On top of a classic supervised pipeline, this branch adds a **self-supervised learning (SSL) module** based on **BYOL**, together with a complete protocol to measure what SSL brings when labels are scarce: kNN monitoring, linear probe, fine-tuning, and a supervised baseline on the same labelled images.

## 🚀 Features

### Supervised pipeline
- **Modular training pipeline** with a clean separation of responsibilities:
  - `Trainer`: training loop, validation, metrics, scheduler, best-model tracking
  - `CheckpointManager`: saves/loads checkpoints
  - `Predictor`: single-batch or multi-batch inference
  - `Visualizer`: plots loss, metrics and confusion matrices
- **Config-driven experiments**: one `Config` dataclass (experiment, data, model, training, optimizer, scheduler, metrics, ssl) built from a plain dictionary or YAML.
- **Metrics registry** (`@register_metric`): Top-K accuracy, macro F1, precision, recall, implemented in pure PyTorch (no `sklearn`). Metrics are referenced by name in the config and `num_classes` is injected automatically.
- **Backbone registry** (`@register_backbone`) on an abstract `BaseBackbone`. Backbones return **feature maps** `[B, C, H, W]`, so they can be reused for other tasks; global pooling is done by the classifier head and by the SSL methods.

**Self-supervised learning**
- **BYOL** from scratch: online and EMA target networks, projector and predictor, symmetrised loss, bf16 mixed precision, two-view augmentations configurable per view.
- **Diagnostics**: a weighted kNN monitor (with an epoch-0 floor for the untrained network) and a collapse monitor (standard deviation of the normalised embeddings, about `1/sqrt(d)` when healthy).
- A small `SSLMethod` interface to add other methods.


### Evaluation protocol
For each fraction of labels (1 %, 10 %, 100 %), four setups are compared on exactly the same labelled images:

| | Starting backbone | What is trained |
|---|---|---|
| **A. Supervised** | random initialisation | whole network |
| **B. Linear probe** | pre-trained BYOL, frozen | linear head only |
| **C. Fine-tuning** | pre-trained BYOL | whole network |
| **D. Random probe** | random initialisation, frozen | linear head only (floor) |

Labelled subsets are **stratified and nested** for a given seed (the 1 % subset is contained in the 10 % subset), so comparisons are made at equal data.

## 📊 Results

### CIFAR-10

Backbone: ResNet-18 backbone (BasicBlock, CIFAR stem, 512-d features), operating on 32×32 images. BYOL is pre-trained on the 50,000 training images **without labels**; 200 epochs; the labelled subsets are drawn from the same images.

| Labels | A. Supervised | B. Fine-tuning BYOL | C. Probe BYOL | D. Probe random | C − A |
|---|---|---|---|---|---|
| 1 % (500 images) | 41.2 | 80.7 | 71.9 | 27.8 | +30.8 |
| 10 % (5,000 images) | 76.7 | 83.8 | 86.9 | 31.2 | +10.2 |
| 100 % (50,000 images) | 93.6 | 84.7 | 94.6 | 35.9 | +1.0 |

### STL-10
 
Backbone: ResNet-18 backbone (BasicBlock, small stem, 512-d features), operating on 64×64 images. BYOL is pre-trained on the 100,000 unlabelled images; 100 epochs; labelled subsets come from the 5,000-image training split; evaluation on the 8,000-image test split.
 
| Labels | A. Supervised | B. Fine-tuning BYOL | C. Probe BYOL | D. Probe random | C − A |
|---|---|---|---|---|---|
| 1 % (50 images) | 25.7 | 64.8 | 40.3 | 17.1 | +14.7 |
| 10 % (500 images) | 45.0 | 70.4 | 72.4 | 16.6 | +27.4 |
| 100 % (5,000 images) | 68.3 | 77.9 | 87.3 | 20.4 | +19.0 |


### What these numbers show
- **SSL helps most when labelled data is scarce in absolute terms.** On CIFAR-10, the advantage shrinks to about +1 point with 50,000 labels. On STL-10, even "100 %" is only 5,000 labelled images, and BYOL still adds +19 points.
- **With very few labels, a frozen probe beats full fine-tuning** (CIFAR-10 at 1 %: 80.7 vs 71.9; STL-10 at 1 %: 64.8 vs 40.3): updating all parameters on a handful of images degrades the pre-trained features. With more labels, fine-tuning takes over (STL-10: 87.3 vs 77.9 at 100 %).
- A frozen **random** backbone gives the floor: about 17-20 % on STL-10 and 28-36 % on CIFAR-10.



## 🧠 Backbones

Interface: forward(x) returns a feature map [B, C, H, W], and feature_dim is C.

| Backbone | Options |
|---|---|
| ResNet | `block` (`basic`, `bottleneck`), `layers`, `stem`; presets `resnet18` to `resnet152` |
| ConvNeXt | `depths`, `dims`, `stem`; presets `convnext_atto` to `convnext_huge` |
| ViT | `image_size`, `patch_size`, `embed_dim`, `depth`, `num_heads`, `mlp_ratio`, `dropout` |
| VGG | `variant` (`VGG11`, `VGG13`, `VGG16`, `VGG19`), `use_batchnorm` |
| MobileNet | `use_batchnorm`, `width_multiplier`, `small_input` |
| DenseNet | `block_config`, `growth_rate`, `compression`, `dropout`, `small_input` |


Adding a backbone:

```python
from models.backbones.base import BaseBackbone
from models.backbones.registry import register_backbone

@register_backbone("my_backbone")
class MyBackbone(BaseBackbone):
    ...
    @property
    def feature_dim(self) -> int: ...
    def forward(self, x): ...      # returns [B, C, H, W]
```

Then reference it with `"backbone": {"name": "my_backbone", ...}` and import its module in `models/backbones/__init__.py` so that the decorator runs.

## 🛠️ Installation

```bash
git clone https://github.com/Kazoric/vision-classification.git
cd vision-classification
git checkout ssl

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install requirements
pip install -r requirements.txt
```

## 💻 Usage

Experiments are described by a configuration dictionary (or YAML) loaded with `Config.from_dict`. Scripts and notebooks are thin wrappers around the same building blocks.

### 1. Self-supervised pre-training (`run_ssl.py`)

Pre-trains BYOL on the full training set without labels, saves the backbone.

```python
config = Config.from_dict({
    "experiment": {"dataset_name": "STL10", "seed": 42},
    "data": {"image_size": [32, 32]},
    "model": {"num_classes": 10,
              "backbone": {"name": "resnet", "block": "basic", "layers": [2, 2, 2, 2], "stem": "small"}},
    "training": {"lr": 0.05, "batch_size": 256, "epochs": 200, "warm_up_epochs": 10},
    "optimizer": {"type": "AdamW", "params": {"weight_decay": 0.0001, "betas": [0.9, 0.999]}},
    "scheduler": {"type": "CosineAnnealingLR", "params": {"T_max": 190}},
    "ssl": {
        "method": "byol",
        "params": {"projection_dim": 256, "hidden_dim": 1024, "tau_base": 0.99},
        "augmentation": {"crop_scale": [0.2, 1.0], "color_jitter": [0.4, 0.4, 0.2, 0.1],
                         "color_jitter_prob": 0.8, "grayscale_prob": 0.2,
                         "blur_probs": [0.0, 0.0], "solarize_probs": [0.0, 0.2]},
        "monitor": {"knn_every": 10, "knn_k": 20, "knn_temperature": 0.1},
        "mixed_precision": True,
    },
})
```

Outputs are written to `experiments/<run_id>/`: `backbone_last.pth` (the weights to transfer), `backbone_best.pth` (best kNN), `ssl_curves.png` (loss, collapse monitor, kNN) and `meta.json`.

Use `SMOKE = True` in the script for a quick plumbing test (2 epochs on 5,000 images).


### 2. Supervised training (`run.py` and `run_training.ipynb`)

Main entry point for classic supervised training: load a dataset, pick a backbone and hyper-parameters, train, validate and save checkpoints.

The backbone configuration is handled in the model section:

- pretrained_backbone: path to the pretrained backbone checkpoint (e.g. from a previous self-supervised learning run). Set to null if training from scratch.
- freeze_backbone: controls whether the backbone is frozen during training.

When the backbone is frozen, it stays in `eval` mode so that BatchNorm statistics do not drift, and only the head is passed to the optimizer.

```python
"model": {
    "num_classes": 10,
    "backbone": {"name": "resnet", "block": "basic", "layers": [2, 2, 2, 2], "stem": "cifar"},
    "pretrained_backbone": "experiments/<ssl_run_id>/backbone_last.pth",
    "freeze_backbone": True,       # True = linear probe, False = fine-tuning
},
```

The three additional parameters are optional and can be added to get_torchvision_dataset to customize dataset loading:

- train_fraction: controls the fraction of the training dataset used. For example, 0.1 uses 10% of the available training data.
- seed: sets the random seed to ensure reproducibility when splitting the dataset.
- persistent_workers: keeps data loader workers alive between epochs, avoiding the overhead of restarting them at each epoch (useful when num_workers > 0).


```python
train_loader, val_loader = get_torchvision_dataset(
        dataset_name=config.experiment.dataset_name, 
        root_dir='./data', 
        batch_size=config.training.batch_size,
        image_size=config.data.image_size,
        use_computed_stats=True,
        train_fraction=0.1,          # Fraction of the training dataset to use (e.g. 10%)
        seed=42,                     # Random seed for reproducible dataset splitting
        persistent_workers=True      # Keep data loader workers alive between epochs
    )
```

### 4. Inference & visualization (`run_inference.ipynb`)

- Load a saved `.pth` checkpoint and run inference on new images.
- Generate Grad-CAM heatmaps to visualise the regions that contributed most to the decision.

## 📂 Project Structure

```bash
.
├── data/                       # Datasets (not tracked by Git)
├── models/
│   ├── backbones/
│   │   ├── base.py             # BaseBackbone (abstract contract)
│   │   ├── registry.py         # @register_backbone / build_backbone
│   │   ├── vgg.py              # VGGBackbone
│   │   ├── mobilenet.py        # MobileNetBackbone
│   │   ├── densenet.py         # DenseNetBackbone
│   │   ├── vit.py              # ViTBackbone
│   │   ├── resnet.py           # ResNetBackbone
│   │   └── convnext.py         # ConvNeXtBackbone
│   └── classifier.py           # Classifier = backbone + linear head (freeze / load helpers)
├── core/
│   ├── config.py               # Config dataclasses (incl. the SSL section)
│   ├── model_base.py           # Model: classifier + loss + run identity
│   ├── optim.py                # Optimizer and scheduler factories
│   ├── trainer.py              # Supervised training loop
│   ├── metrics.py              # Metrics and registry
│   ├── checkpoint.py           # CheckpointManager
│   ├── predictor.py            # Inference logic
│   ├── visualizer.py           # Plots
│   ├── utils.py                # Seeding
│   └── ssl/
│       ├── augmentations.py    # Two-view transform
│       ├── methods.py          # SSLMethod interface, BYOL, method registry
│       ├── ssl_trainer.py      # Pre-training loop (no labels, EMA, kNN monitor)
│       ├── evaluation.py       # Feature extraction and weighted kNN
│       └── plots.py            # SSL training curves
├── data_loader.py              # Supervised and SSL data loaders, label fractions, normalisation stats
├── run_ssl.py                  # SSL pre-training + linear probe
├── run_all.py                  # A / C sweeps over label fractions
├── run_training.ipynb          # Supervised training walkthrough
├── run_inference.ipynb         # Inference walkthrough
├── run.py                      # Developer sandbox
└── README.md
```

## 🗂️ Dataset Management

The data interface is shared by the supervised and the SSL loaders.

### Supported datasets

- **Torchvision datasets** (e.g. `CIFAR10`, `CIFAR100`, `STL-10`, `Imagenette`), downloaded automatically to the data directory.
- **Custom datasets** in the [PyTorch ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) format (`train/` and `valid/` folders with one sub-folder per class).

```bash
data/
└── dataset/
    ├── train/
    │   ├── class_1/
    │   └── class_2/
    └── valid/
        ├── class_1/
        └── class_2/
```

### Normalisation

With `use_computed_stats=True`, per-channel mean and standard deviation are computed once on the training split and cached in `stats.json`. **Use the same setting for SSL pre-training and for every downstream run**, otherwise the backbone sees differently normalised inputs and the evaluation underestimates the learned features.

### SSL loaders

`get_ssl_dataloaders` returns three loaders:
- the pre-training loader (training images, two augmented views, labels ignored);
- a kNN bank (training images without augmentation);
- the validation loader.

SSL pre-training always uses the **whole** training set, since it does not need labels. The `train_fraction` option of the supervised loader restricts only the labelled data of the downstream runs.


## 📚 References

- Grill et al., *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning*, NeurIPS 2020.
- He et al., *Deep Residual Learning for Image Recognition* (ResNet), CVPR 2016.
- Liu et al., *A ConvNet for the 2020s* (ConvNeXt), CVPR 2022.
- Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (ViT), ICLR 2021.
- Simonyan and Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition* (VGG), ICLR 2015.
- Howard et al., *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications* (MobileNet), 2017.
- Huang et al., *Densely Connected Convolutional Networks* (DenseNet), CVPR 2017.