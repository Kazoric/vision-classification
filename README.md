# Vision - Image Classification Module

This project provides a Python module for image classification featuring various popular and custom models. It includes multiple architectures and a modular design with a shared base class.

## 🚀 Features

- Implementation of several image classification models:
  - VGG
  - DenseNet
  - MobileNet
  - ResNet
  - Wide ResNet
- A `Model` class in `model_base.py` that provides:
  - Training
  - Evaluation
  - Checkpoint saving and loading
  - Prediction
  - Accuracy and loss plotting
- Each model file defines:
  - `Architecture`: the PyTorch model architecture class
  - `Model`: the class declaring the model, including a `build_model` method to instantiate the architecture
- Two `data_loader` classes to handle CIFAR10 and CIFAR100 dataset loading
- Modular and extensible design to easily add new models or datasets

## 📂 Project Structure

```bash
.
├── models/
│   ├── model_base.py           # Contains the Model class (train, eval, checkpoint, predict, plot)
│   ├── vgg.py                  # Defines VGGArchitecture and VGGModel
│   ├── densenet.py             # Defines DenseNetArchitecture and DenseNetModel
│   ├── mobilenet.py            # Defines MobileNetArchitecture and MobileNetModel
│   ├── resnet.py               # Defines ResNetArchitecture and ResNetModel for ResNet
│   └── wide_resnet.py          # Defines WideResNetArchitecture and WideResNetModel for Wide ResNet
├── data_loader.py              # CIFAR100 and CIFAR10 data loader
├── run.py                      # Script to execute training/evaluation
└── README.md
