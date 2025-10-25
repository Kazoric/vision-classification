import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
import time

from models.resnet import ResNetModel
from models.wide_resnet import WideResNetModel
from models.vit import ViTModel
from data_loader import get_torchvision_dataset
from core.visualizer import Visualizer
from core.metrics import topk_accuracy_torch, f1_score_torch, precision_score_torch, recall_score_torch, confusion_matrix_torch, plot_confusion_matrix

def main():
    # 🔧 Hyperparameters
    optimizer = optim.SGD
    optimizer_params = {"momentum": 0.9, "weight_decay": 5e-4}
    batch_size = 128
    learning_rate = 0.001
    scheduler = StepLR
    scheduler_params = {"step_size": 10, "gamma": 0.1}
    num_epochs = 10
    num_classes = 2
    label_smoothing = 0.1
    model_name = "ResNet"
    dataset_name = 'example'
    resume = False  # True to load a checkpoint if it exists
    if resume:
        run_id = 'ResNet_Imagenette_2025-10-19_17-28-16'
    else:
        run_id = None
    metrics = {
        "Top-1 Accuracy": (topk_accuracy_torch, {"k": 1}),
        #"Top-5 Accuracy": (topk_accuracy_torch, {"k": 5}),
        "F1": (f1_score_torch, {"num_classes": num_classes}),
        "Precision": (precision_score_torch, {"num_classes": num_classes}),
        "Recall": (recall_score_torch, {"num_classes": num_classes}),
    }

    # 📦 Data
    train_loader, val_loader = get_torchvision_dataset(
        dataset_name=dataset_name, 
        root_dir='./data', 
        batch_size=batch_size,
        use_computed_stats=True
    )
    assert num_classes == len(train_loader.dataset.classes), \
        f"Configuration error: you set num_classes={num_classes}, but the dataset actually contains {len(train_loader.dataset.classes)} classes."


    # 🧠 Model
    model = ResNetModel(
        lr=learning_rate, model_name=model_name, dataset_name=dataset_name, save=True,
        run_id=run_id, # needed to resume
        # optimizer_cls=optimizer,
        # optimizer_params=optimizer_params,
        scheduler_cls = scheduler,
        scheduler_params = scheduler_params,
        metrics=metrics,
        num_classes=num_classes,
        # label_smoothing=label_smoothing,
        layer_list=[2,2,2,2], block='Bottleneck'
    )
    
    # model = ViTModel(
    #     lr=learning_rate, model_name=model_name, dataset_name=dataset_name, save=True,
    #     run_id=run_id, # needed to resume
    #     # optimizer_cls=optimizer,
    #     # optimizer_params=optimizer_params,
    #     # scheduler_cls = scheduler,
    #     # scheduler_params = scheduler_params,
    #     metrics=metrics,
    #     num_classes=num_classes,
    #     image_size=train_loader.dataset[0][0].shape[-1],
    #     patch_size = 4,
    #     depth = 4,
    #     # label_smoothing=label_smoothing
    # )

    # ♻️ Loading a checkpoint (optional)
    if resume:
        model.load_checkpoint()

    # 🚀 Training
    start_time = time.time()
    model.train(train_loader, val_loader, epochs=num_epochs)
    end_time = time.time() - start_time
    print(f"Training took {end_time:.2f} seconds\n")

    # 📈 Visualization
    visualizer = Visualizer()
    visualizer.plot_metrics(model.trainer, model.run_id)

    # 🔍 Prediction
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    outputs = model.predict(images[:4])
    print(f"Predicted classes: {outputs.tolist()}")
    print(f"Ground truth:     {labels[:4].tolist()}")

    labels, outputs = model.predict_on_loader(val_loader)
    cm = confusion_matrix_torch(labels, outputs, num_classes=num_classes)
    plot_confusion_matrix(cm, train_loader.dataset.classes)

    model.save_hyperparams(
        batch_size=batch_size,
        num_epochs=num_epochs,
    )


if __name__ == "__main__":
    main()
    print()
