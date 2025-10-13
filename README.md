# Vision - Image Classification Module

This project provides a Python module for image classification featuring various popular pretrained and custom models. It includes multiple architectures and a modular design with a shared base class.

## 🚀 Features

- Implementation of several image classification models:
  - VGG
  - DenseNet
  - MobileNet
  - ResNet
  - Wide ResNet
- A `model_base` class inherited by all models to ensure easy extensibility and maintenance.
- Two `data_loader` classes to handle CIFAR10 and CIFAR100 dataset loading.
- Modular and extensible design to easily integrate new models or datasets.

## 📂 Project Structure

```bash
.
├── models/
│   ├── model_base.py           # Base class for models
│   ├── vgg.py                  # VGG model
│   ├── densenet.py             # DenseNet model
│   ├── mobilenet.py            # MobileNet model
│   ├── resnet.py               # ResNet model
│   └── wide_resnet.py          # Wide ResNet model
├── data_loader.py              # CIFAR100 and CIFAR10 data loader
├── run.py                      # Script to execute training/evaluation
└── README.md
