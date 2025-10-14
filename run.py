# import numpy as np
# import pickle
# import os
# import matplotlib.pyplot as plt
# import time

# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms

# from data_loader import get_custom_cifar100_dataloaders, get_custom_stl10_dataloaders
# from models.resnet import ResNetModel
# from models.wide_resnet import WideResNetModel
# from models.vgg import VGGModel
# from models.mobilenet import MobileNetModel
# from models.densenet import DenseNetModel

# # def unpickle(file):
# #     with open(file, 'rb') as fo:
# #         dict = pickle.load(fo, encoding='bytes')
# #     X = dict[b'data']
# #     Y = dict[b'fine_labels']
# #     X = X.reshape(-1, 3, 32, 32).transpose(0,2,3,1)  # NCHW -> NHWC
# #     return X, Y
    

# train_loader, test_loader = get_custom_cifar100_dataloaders(
#     data_path='./cifar-100-python',
#     batch_size=2048
# )

# # train_loader, test_loader = get_custom_stl10_dataloaders(
# #     data_path='./stl10_binary',
# #     batch_size=64
# # )

# # model = ResNetModel(num_class=20, layer_list=[1,2,3,1], block='Bottleneck')
# model = WideResNetModel(num_class=20, layer_list=[1,2,1], block='Bottleneck', widen_factor=2, save=True)
# # model = VGGModel(num_class=100, variant='VGG11')
# # model = MobileNetModel(num_class=100)
# # model = DenseNetModel(
# #     num_class=20, 
# #     block_config=[6, 12, 24, 16],  # DenseNet-121
# #     growth_rate=32,
# #     compression=0.5
# # )

# # model.load_checkpoint()

# training_start = time.time()
# model.train(train_loader, test_loader, epochs=5)
# training_stop = time.time()
# training_time = training_stop - training_start
# print(f"Training time: {training_time}")

# model.plot_acc_loss()


# # visualizer = Visualizer()
# # visualizer.plot_metrics(
# #     train_acc=trainer.train_acc,
# #     val_acc=trainer.valid_acc,
# #     train_loss=trainer.train_loss,
# #     val_loss=trainer.valid_loss,
# #     save_path="results/curves.png"
# # )



# run.py

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

from models.resnet import ResNetModel
from data_loader import get_custom_cifar100_dataloaders, get_custom_stl10_dataloaders
from core.visualizer import Visualizer

def main():
    # 🔧 Hyperparamètres
    optimizer = optim.SGD
    optimizer_params = {"momentum": 0.9, "weight_decay": 5e-4}
    batch_size = 2048
    learning_rate = 0.001
    scheduler = StepLR
    scheduler_params = {"step_size": 10, "gamma": 0.1}
    num_epochs = 25
    model_name = "resnet_cifar10"
    metrics=["F1", "Accuracy", "Precision", "Recall"]
    resume = False  # True pour charger un checkpoint s’il existe

    # 📦 Données
    train_loader, val_loader = get_custom_cifar100_dataloaders(
        data_path='./cifar-100-python',
        batch_size=batch_size
    )

    # 🧠 Modèle
    model = ResNetModel(lr=learning_rate, model_name=model_name, save=True,
                        # optimizer_cls=optimizer,
                        # optimizer_params=optimizer_params,
                        # scheduler_cls = scheduler,
                        # scheduler_params = scheduler_params,
                        metrics=metrics,
                        num_classes=20, layer_list=[1,2,3,1], block='Bottleneck',
                        checkpoint_dir="./checkpoints/exp2")

    # ♻️ Chargement du checkpoint (optionnel)
    if resume:
        model.load_checkpoint()

    # 🚀 Entraînement
    model.train(train_loader, val_loader, epochs=num_epochs)

    # 📈 Visualisation
    visualizer = Visualizer()
    # visualizer.plot_metrics(
    #     train_acc=model.trainer.train_acc,
    #     val_acc=model.trainer.valid_acc,
    #     train_loss=model.trainer.train_loss,
    #     val_loss=model.trainer.valid_loss,
    #     save_path="plot_loss/training_curves.png"
    # )
    visualizer.plot_metrics(model.trainer)

    # 🔍 Prédiction d'exemple
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    outputs = model.predict(images[:4])
    print(f"🎯 Predicted classes: {outputs.tolist()}")
    print(f"✅ Ground truth:     {labels[:4].tolist()}")

if __name__ == "__main__":
    main()
