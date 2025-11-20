import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import time

from models.resnet import ResNetModel
from models.densenet import DenseNetModel
from models.mobilenet import MobileNetModel
from models.vgg import VGGModel
from models.wide_resnet import WideResNetModel
from models.vit import ViTModel
from models.convnext import ConvNeXtModel
from data_loader import get_torchvision_dataset
from core.visualizer import Visualizer
from core.metrics import topk_accuracy_torch, f1_score_torch, precision_score_torch, recall_score_torch, confusion_matrix_torch, plot_confusion_matrix

def main():
    # 🔧 Hyperparameters
    # optimizer = optim.SGD
    # optimizer_params = {"momentum": 0.9, "weight_decay": 5e-4}
    optimizer = optim.AdamW
    optimizer_params = {"weight_decay": 5e-4}
    batch_size = 512
    learning_rate = 0.001
    # scheduler = StepLR
    # scheduler_params = {"step_size": 10, "gamma": 0.1}
    scheduler = CosineAnnealingLR
    scheduler_params = {"T_max": 5}
    num_epochs = 5
    num_classes = 10
    label_smoothing = 0.1
    dataset_name = 'cifar10'
    image_size=(32,32)
    resume = False  # True to load a checkpoint if it exists
    if resume:
        run_id = 'vit_cifar10_2025-10-30_16-00-46'
    else:
        run_id = None
    metrics = {
        "Top-1 Accuracy": (topk_accuracy_torch, {"k": 1}),
        "Top-5 Accuracy": (topk_accuracy_torch, {"k": 5}),
        "F1": (f1_score_torch, {"num_classes": num_classes}),
        "Precision": (precision_score_torch, {"num_classes": num_classes}),
        "Recall": (recall_score_torch, {"num_classes": num_classes}),
    }

    # 📦 Data
    train_loader, val_loader = get_torchvision_dataset(
        dataset_name=dataset_name, 
        root_dir='./data', 
        batch_size=batch_size,
        image_size=image_size,
        use_computed_stats=True
    )
    assert num_classes == len(train_loader.dataset.classes), \
        f"Configuration error: you set num_classes={num_classes}, but the dataset actually contains {len(train_loader.dataset.classes)} classes."


    # 🧠 Model
    # model = ResNetModel(
    #     lr=learning_rate, dataset_name=dataset_name, save=True,
    #     run_id=run_id, # needed to resume
    #     optimizer_cls=optimizer,
    #     optimizer_params=optimizer_params,
    #     scheduler_cls = scheduler,
    #     scheduler_params = scheduler_params,
    #     metrics=metrics,
    #     num_classes=num_classes,
    #     label_smoothing=label_smoothing,
    #     layer_list=[2,2,2,2], block='Basic', dropout=0.0,
    #     image_size=(32,32)
    # )
    
    # model = ViTModel(
    #     lr=learning_rate, dataset_name=dataset_name, save=True,
    #     run_id=run_id, # needed to resume
    #     optimizer_cls=optimizer,
    #     optimizer_params=optimizer_params,
    #     scheduler_cls = scheduler,
    #     scheduler_params = scheduler_params,
    #     metrics=metrics,
    #     num_classes=num_classes,
    #     image_size=image_size,
    #     patch_size = 4,
    #     depth = 4,
    #     label_smoothing=label_smoothing
    # )

    # model = ConvNeXtModel(
    #     lr=learning_rate, dataset_name=dataset_name, save=True,
    #     run_id=run_id, # needed to resume
    #     optimizer_cls=optimizer,
    #     optimizer_params=optimizer_params,
    #     scheduler_cls = scheduler,
    #     scheduler_params = scheduler_params,
    #     metrics=metrics,
    #     num_classes=num_classes,
    #     image_size=image_size,
    #     label_smoothing=label_smoothing
    # )

    # model = MobileNetModel(
    #     lr=learning_rate, dataset_name=dataset_name, save=True,
    #     run_id=run_id, # needed to resume
    #     optimizer_cls=optimizer,
    #     optimizer_params=optimizer_params,
    #     scheduler_cls = scheduler,
    #     scheduler_params = scheduler_params,
    #     metrics=metrics,
    #     num_classes=num_classes,
    #     label_smoothing=label_smoothing,
    #     image_size=image_size
    # )

    # model = DenseNetModel(
    #     lr=learning_rate, dataset_name=dataset_name, save=True,
    #     run_id=run_id, # needed to resume
    #     optimizer_cls=optimizer,
    #     optimizer_params=optimizer_params,
    #     scheduler_cls = scheduler,
    #     scheduler_params = scheduler_params,
    #     metrics=metrics,
    #     num_classes=num_classes,
    #     label_smoothing=label_smoothing,
    #     block_config = [6, 12, 24],
    #     growth_rate = 12, 
    #     image_size=(32,32)
    # )

    # model = WideResNetModel(
    #     lr=learning_rate, model_name=model_name, dataset_name=dataset_name, save=True,
    #     run_id=run_id, # needed to resume
    #     # optimizer_cls=optimizer,
    #     # optimizer_params=optimizer_params,
    #     scheduler_cls = scheduler,
    #     scheduler_params = scheduler_params,
    #     metrics=metrics,
    #     num_classes=num_classes,
    #     # label_smoothing=label_smoothing,
    #     layer_list=[4, 4, 4], block='Basic', widen_factor=4, image_size=(32,32)
    # )

    model = VGGModel(
        lr=learning_rate, dataset_name=dataset_name, save=True,
        run_id=run_id, # needed to resume
        optimizer_cls=optimizer,
        optimizer_params=optimizer_params,
        scheduler_cls = scheduler,
        scheduler_params = scheduler_params,
        metrics=metrics,
        num_classes=num_classes,
        label_smoothing=label_smoothing,
        image_size=image_size
    )

    # ♻️ Loading a checkpoint (optional)
    if resume:
        model.load_checkpoint()

    # 🚀 Training
    start_time = time.time()
    model.train(train_loader, val_loader, epochs=num_epochs)
    end_time = time.time() - start_time
    print(f"Training took {end_time:.2f} seconds\n")

    # 🔍 Prediction
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    outputs = model.predict(images[:4])
    print(f"Predicted classes: {outputs.tolist()}")
    print(f"Ground truth:     {labels[:4].tolist()}")

    labels, outputs = model.predict_on_loader(val_loader)
    cm = confusion_matrix_torch(labels, outputs, num_classes=num_classes)
    # plot_confusion_matrix(cm, train_loader.dataset.classes)

    # 📈 Visualization
    visualizer = Visualizer()
    visualizer.plot_metrics(model.trainer, model.run_id)
    visualizer.plot_confusion_matrix(cm, train_loader.dataset.classes, model.run_id)

    model.save_hyperparams(
        batch_size=batch_size,
        num_epochs=num_epochs,
    )


if __name__ == "__main__":
    main()
    print()
