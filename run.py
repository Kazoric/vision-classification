import os
import json
import time
import torch

# Core modules imports
from core.config import Config
from core.trainer import Trainer
from core.predictor import Predictor
from core.checkpoint import CheckpointManager
from core.visualizer import Visualizer
from core.model_base import Model
from core.metrics import confusion_matrix_torch
from core.utils import generate_run_id, set_seed

# Data and Model imports
from data_loader import get_torchvision_dataset

yaml_config = {
    "experiment": {
        "dataset_name": "cifar10",
        "save_checkpoints": False,
        "seed": 42
    },
    "data": {
        "image_size": [32, 32],
    },
    "model": {
        "num_classes": 10,
        "backbone": {"name": "resnet18", "stem": "cifar"},
    },
    "training": {"lr": 0.01, "batch_size": 256, "epochs": 5, "warm_up_epochs": 0, "label_smoothing": 0.1},
    "optimizer": {"type": "AdamW", "params": {"weight_decay": 0.0001, "betas": [0.9, 0.999]}},
    "scheduler": {"type": "CosineAnnealingLR", "params": {"T_max": 10}},   # epochs - warm_up
    'metrics': {
        'monitor_metric': 'Top_1_Accuracy',
        'monitor_mode': 'max',
        'metric_functions': {
            'Top_1_Accuracy': ['topk_accuracy_torch', {'k': 1}],
            'Top-5 Accuracy': ['topk_accuracy_torch', {'k': 5}],
            'Precision': ['precision_score_torch', {'num_classes': 10}],
            'Recall': ['recall_score_torch', {'num_classes': 10}],
            'F1': ['f1_score_torch', {'num_classes': 10}]
        }
    }
}

if __name__ == "__main__":
    config = Config.from_dict(yaml_config)

    # Set run_id
    if config.experiment.run_id is None:
        run_id = generate_run_id(
            model_name=config.model.backbone['name'],
            dataset_name=config.experiment.dataset_name,
        )
        config.experiment.run_id = run_id

    set_seed(config.experiment.seed)

    # Determine the computation device (CUDA if available, otherwise CPU).
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging initial setup information.
    print(f"  Device        : {device}")
    print(f"  Num classes   : {config.model.num_classes}")
    train_loader, val_loader = get_torchvision_dataset(
        dataset_name=config.experiment.dataset_name, 
        root_dir='./data', 
        batch_size=config.training.batch_size,
        image_size=config.data.image_size,
        use_computed_stats=True
    )
    assert config.model.num_classes == len(train_loader.dataset.classes), \
        f"Configuration error: you set num_classes={config.model.num_classes}, but the dataset actually contains {len(train_loader.dataset.classes)} classes."
    model = Model(config)

    # Log model specifics like name and total parameters.
    print(f"  Model        : {model.name}")
    print(f"  Parameters    : {sum(p.numel() for p in model.parameters()):,}")

    # Setup the CheckpointManager to save and load training state.
    checkpoint = None
    if config.experiment.save_checkpoints:
        checkpoint = CheckpointManager(
            run_id=config.experiment.run_id,
            model_name=config.model.backbone['name'],
        )
    
    trainer = Trainer.from_config(
        model=model,
        config=config,
        device=device,
        metrics_config=config.metrics if config.metrics.metric_functions else None,
        on_best_model=checkpoint.save_best if checkpoint else None,
        on_epoch_end=checkpoint.save_last if checkpoint else None,
    )

    # Log the unique run identifier.
    print(f"  Run ID        : {run_id}")

    # Initialize the Predictor and Visualizer components.
    predictor = Predictor(model=model, device=device)
    visualizer = Visualizer()
    RESUME = False
    if RESUME and checkpoint.exists():
        # Load training state if RESUME is True and a checkpoint exists.
        state = checkpoint.load(load_optimizer=True)
        trainer.resume_from(
            epoch=state["epoch"],
            best_metric_value=state["best_metric_value"],
        )
    print(f"\n=== [STEP 5] Training ({config.training.epochs} epochs) ===")
    start_time = time.time()
    # Execute the training loop on both training and validation data.
    trainer.train(train_loader, val_loader, epochs=config.training.epochs)
    end_time = time.time() - start_time
    print(f"Training took {end_time:.2f} seconds\n")
    y, logits = trainer.collect_predictions(val_loader)
    cm = confusion_matrix_torch(y, logits, config.model.num_classes)
    visualizer.plot_confusion_matrix(cm, train_loader.dataset.classes, model_name=model.name, run_id=model.run_id)
    visualizer.plot_metrics(trainer, model.run_id)
    trainer.save_hyperparams()