# run.py

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
import time

from models.resnet import ResNetModel
# from data_loader import get_custom_cifar100_dataloaders, get_custom_stl10_dataloaders
from data_loader import get_torchvision_dataset
from core.visualizer import Visualizer
from core.metrics import confusion_matrix_torch, plot_confusion_matrix

def main():
    # 🔧 Hyperparamètres
    optimizer = optim.SGD
    optimizer_params = {"momentum": 0.9, "weight_decay": 5e-4}
    batch_size = 2048
    learning_rate = 0.001
    scheduler = StepLR
    scheduler_params = {"step_size": 10, "gamma": 0.1}
    num_epochs = 25
    num_classes = 10
    model_name = "resnet_cifar10"
    metrics=["F1", "Accuracy", "Precision", "Recall"]
    resume = False  # True pour charger un checkpoint s’il existe

    # 📦 Données
    # train_loader, val_loader = get_custom_cifar100_dataloaders(
    #     data_path='./cifar-100-python',
    #     batch_size=batch_size
    # )
    train_loader, val_loader = get_torchvision_dataset(
        dataset_name='CIFAR10', 
        root_dir='./data/cifar', 
        batch_size=2048,
        image_size=224
    )
    assert num_classes == len(train_loader.dataset.classes), \
        f"Configuration error: you set num_classes={num_classes}, but the dataset actually contains {len(train_loader.dataset.classes)} classes."


    # 🧠 Modèle
    model = ResNetModel(lr=learning_rate, model_name=model_name, save=True,
                        # optimizer_cls=optimizer,
                        # optimizer_params=optimizer_params,
                        # scheduler_cls = scheduler,
                        # scheduler_params = scheduler_params,
                        metrics=metrics,
                        num_classes=num_classes, 
                        layer_list=[1,1,1,1], block='Bottleneck'
                        )

    # ♻️ Chargement du checkpoint (optionnel)
    if resume:
        model.load_checkpoint()

    # 🚀 Entraînement
    start_time = time.time()
    model.train(train_loader, val_loader, epochs=num_epochs)
    end_time = time.time() - start_time
    print(f"Training took {end_time:.2f} seconds")

    # 📈 Visualisation
    visualizer = Visualizer()
    visualizer.plot_metrics(model.trainer, model.run_id)

    # 🔍 Prédiction d'exemple
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    outputs = model.predict(images[:4])
    print(f"🎯 Predicted classes: {outputs.tolist()}")
    print(f"✅ Ground truth:     {labels[:4].tolist()}")

    labels, outputs = model.predict_on_loader(val_loader)
    print(labels.device)
    print(outputs.device)
    cm = confusion_matrix_torch(labels, outputs.cpu(), num_classes=num_classes)
    plot_confusion_matrix(cm, train_loader.dataset.classes)

    model.save_hyperparams(
        optimizer_name=optimizer.__name__,
        optimizer_params=optimizer_params,
        scheduler_name=scheduler.__name__,
        scheduler_params=scheduler_params,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )


if __name__ == "__main__":
    main()
    print()
