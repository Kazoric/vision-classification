# run.py

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
import argparse
from threading import Thread
from tensorboard import program
import webbrowser
import time

from models.resnet import ResNetModel
from data_loader import get_custom_cifar100_dataloaders, get_custom_stl10_dataloaders
from core.visualizer import Visualizer

def launch_tensorboard(log_dir="runs", port=6006):
    tb = program.TensorBoard()
    tb.configure(argv=[None, f"--logdir={log_dir}", f"--port={port}"])
    url = tb.launch()
    print(f"🔍 TensorBoard lancé à l'adresse : {url}")
    # Optionnel : ouvre automatiquement dans le navigateur
    time.sleep(2)
    webbrowser.open(url)

def main():
    # 🔧 Hyperparamètres
    optimizer = optim.SGD
    optimizer_params = {"momentum": 0.9, "weight_decay": 5e-4}
    batch_size = 2048
    learning_rate = 0.001
    scheduler = StepLR
    scheduler_params = {"step_size": 10, "gamma": 0.1}
    num_epochs = 10
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
                        num_classes=20, layer_list=[1,2,3,1], block='Bottleneck'
                        )

    # ♻️ Chargement du checkpoint (optionnel)
    if resume:
        model.load_checkpoint()

    # 🚀 Entraînement
    model.train(train_loader, val_loader, epochs=num_epochs)
    model.close_logger()

    # 📈 Visualisation
    visualizer = Visualizer()
    visualizer.plot_metrics(model.trainer, model.run_id)

    # 🔍 Prédiction d'exemple
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    outputs = model.predict(images[:4])
    print(f"🎯 Predicted classes: {outputs.tolist()}")
    print(f"✅ Ground truth:     {labels[:4].tolist()}")

    model.save_hyperparams(
        optimizer_name=optimizer.__name__,
        optimizer_params=optimizer_params,
        scheduler_name=scheduler.__name__,
        scheduler_params=scheduler_params,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    # 🚦 Lancer TensorBoard si demandé
    if args.tensorboard:
        # Lance dans un thread pour ne pas bloquer la fin du script
        tb_thread = Thread(target=launch_tensorboard, args=("runs",))
        tb_thread.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script with optional TensorBoard launch")
    parser.add_argument('--tensorboard', action='store_true', help="Lancer TensorBoard après l'entraînement")
    args = parser.parse_args()
    main()
    print()
