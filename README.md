# Vision - Image Classification Module

This project provides a modular and extensible image classification framework using PyTorch. It supports multiple popular architectures and separates core training logic into reusable components such as training, prediction, checkpointing, and visualization.

## 🚀 Features

- **Modular training pipeline** with clean separation of responsibilities:
  - `Trainer`: handles training loop, validation, metrics, scheduler
  - `CheckpointManager`: saves/loads checkpoints automatically
  - `Predictor`: for single-batch or multi-batch inference
  - `Visualizer`: for plotting loss and metrics over epochs
- Built-in support for metrics:
  - **Accuracy**, **F1 Score**, **Precision**, **Recall**
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

Each model file defines:
- `Architecture`: the PyTorch `nn.Module` class
- `Model`: a subclass of `ModelBase` that builds the architecture

## 📂 Project Structure

```bash
.
├── models/
│   ├── resnet.py               # ResNetArchitecture and ResNetModel
│   ├── vgg.py                  # VGGArchitecture and VGGModel
│   ├── densenet.py             # DenseNetArchitecture and DenseNetModel
│   ├── mobilenet.py            # MobileNetArchitecture and MobileNetModel
│   └── wide_resnet.py          # WideResNetArchitecture and WideResNetModel
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
