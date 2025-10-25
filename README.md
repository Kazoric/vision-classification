# Vision - Image Classification Module

This project provides a modular and extensible image classification framework using PyTorch. It supports multiple popular architectures and separates core training logic into reusable components such as training, prediction, checkpointing, and visualization.

## 🚀 Features

- **Modular training pipeline** with clean separation of responsibilities:
  - `Trainer`: handles training loop, validation, metrics, scheduler
  - `CheckpointManager`: saves/loads checkpoints automatically
  - `Predictor`: for single-batch or multi-batch inference
  - `Visualizer`: for plotting loss and metrics over epochs
- Built-in support for metrics:
  - **Top-K Accuracy**, **F1 Score**, **Precision**, **Recall**
  - Implemented using **pure PyTorch** (no `sklearn`)
- Configurable schedulers and optimizers via `run.py`
- Resume training from checkpoints automatically
- Loss and metrics plotted after training
- Easily extensible to new models

## 🧠 Supported Architectures

- VGG
- DenseNet
- MobileNet
- ResNet
- Wide ResNet
- ViT

Each model file defines:
- `Architecture`: the PyTorch `nn.Module` class
- `Model`: a subclass of `ModelBase` that builds the architecture

## 🗂️ Dataset Management

This module provides a **flexible and unified data loading interface**, compatible with both standard torchvision datasets and your own custom datasets.

### Supported Datasets

- **Torchvision datasets** (e.g., `CIFAR10`, `CIFAR100`, `ImageNet`, `Imagenette`). These datasets are **automatically downloaded** to the specified directory to avoid repeated downloads.

- **Custom datasets** following the [PyTorch ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) format. Simply organize your images into the standard folder structure, which is illustrated below, and the module will handle them automatically.

```bash
data/
└── dataset/
    ├── train/
    │   ├── class_1/
    │   │   ├── class_1_001.jpg
    │   │   ├── class_1_002.jpg
    │   │   └── ...
    │   └── class_2/
    │       ├── class_2_001.jpg
    │       ├── class_2_002.jpg
    │       └── ...
    └── val/
        ├── class_1/
        │   ├── class_1_001.jpg
        │   ├── class_1_002.jpg
        │   └── ...
        └── class_2/
            ├── class_2_001.jpg
            ├── class_2_002.jpg
            └── ...
```

## 📂 Project Structure

```bash
.
├── data/                      # Datasets (not tracked by Git)
│   └── dataset/               # Your train/val folders or Torchvision downloads
├── models/
│   ├── resnet.py               # ResNetArchitecture and ResNetModel
│   ├── vgg.py                  # VGGArchitecture and VGGModel
│   ├── densenet.py             # DenseNetArchitecture and DenseNetModel
│   ├── mobilenet.py            # MobileNetArchitecture and MobileNetModel
│   ├── wide_resnet.py          # WideResNetArchitecture and WideResNetModel
│   └── vit.py                  # ViTArchitecture and ViTModel
│
├── core/
│   ├── model_base.py           # Base Model class (handles trainer, predictor, checkpoint)
│   ├── trainer.py              # Trainer class (training loop, metrics, validation)
│   ├── predictor.py            # Inference logic
│   ├── checkpoint.py           # CheckpointManager (save/load checkpoints)
│   ├── visualizer.py           # Visualization of loss and metrics
│   └── metrics.py              # Torch implementations of F1, Precision, Recall, Accuracy
│
├── data_loader.py              # Loaders for CIFAR-10 and CIFAR-100
├── run.py                      # Main script to launch training/evaluation
└── README.md                   # This file
