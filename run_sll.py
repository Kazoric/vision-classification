import os
import gc
import time
from datetime import datetime

import torch

# Core module imports
from core.config import Config
from core.trainer import Trainer
from core.checkpoint import CheckpointManager
from core.visualizer import Visualizer
from core.model_base import Model
from core.metrics import confusion_matrix_torch
from core.utils import set_seed

# SSL imports
from models.backbones import build_backbone
from core.ssl.methods import build_ssl_method
from core.ssl.evaluation import make_knn_monitor
from core.ssl.ssl_trainer import SSLTrainer
from core.ssl.plots import plot_ssl_curves

# Data imports
from data_loader import get_ssl_dataloaders, get_torchvision_dataset

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
SMOKE = False                  # True: smoke test (2 SSL epochs on 5,000 images, 3 probe epochs)
SEED = 42
DATASET = "cifar10"
IMAGE_SIZE = (32, 32)
USE_STATS = True               # Use the same normalization for SSL and the probe.
BACKBONE = {"name": "resnet", "block": "basic", "layers": [2, 2, 2, 2], "stem": "cifar"}
NUM_CLASSES = 10

SSL_EPOCHS = 2 if SMOKE else 200
SSL_WARMUP = 0 if SMOKE else 10
LIMIT = 5000 if SMOKE else None

PROBE_EPOCHS = 3 if SMOKE else 50
PROBE_BATCH = 256
TRAIN_FRACTION = 0.1           # Fraction of labels used for the probe (0.1 / 0.01: low-label protocol)
RUN_RANDOM_CONTROL = True      # Evaluate an untrained backbone as a baseline to beat.

device = "cuda" if torch.cuda.is_available() else "cpu"


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------
# Phase 1: SSL pretraining on the ENTIRE training set (without labels)
# ---------------------------------------------------------------
def run_ssl():
    config = Config.from_dict({
        "experiment": {"dataset_name": DATASET, "seed": SEED},
        "data": {"image_size": list(IMAGE_SIZE)},
        "model": {"num_classes": NUM_CLASSES, "backbone": BACKBONE},
        "training": {"lr": 0.05, "batch_size": 256, "epochs": SSL_EPOCHS, "warm_up_epochs": SSL_WARMUP},
        "optimizer": {"type": "SGD", "params": {"momentum": 0.9, "weight_decay": 5.0e-4}},
        "scheduler": {"type": "CosineAnnealingLR", "params": {"T_max": SSL_EPOCHS - SSL_WARMUP}},
        "ssl": {
            "method": "byol",
            "params": {"projection_dim": 256, "hidden_dim": 1024, "tau_base": 0.99},
            "augmentation": {
                "crop_scale": [0.2, 1.0],
                "color_jitter": [0.4, 0.4, 0.2, 0.1],
                "color_jitter_prob": 0.8,
                "grayscale_prob": 0.2,
                "blur_probs": [0.0, 0.0],
                "solarize_probs": [0.0, 0.2],
            },
            "monitor": {"knn_every": 1 if SMOKE else 10, "knn_k": 20, "knn_temperature": 0.1},
            "mixed_precision": True,
        },
    })
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    print(f"\n=== PHASE 1: SSL ({config.ssl.method}, {SSL_EPOCHS} epochs) | device={device} ===")

    ssl_loader, knn_train_loader, val_loader, class_names = get_ssl_dataloaders(
        dataset_name=config.experiment.dataset_name,
        aug_cfg=config.ssl.augmentation,
        root_dir="./data",
        batch_size=config.training.batch_size,
        image_size=IMAGE_SIZE,
        use_computed_stats=USE_STATS,
        limit=LIMIT,
        seed=SEED,
    )
    assert class_names is None or config.model.num_classes == len(class_names), \
        f"num_classes={config.model.num_classes}, but the dataset has {len(class_names)} classes"

    backbone_params = dict(config.model.backbone)
    backbone_name = backbone_params.pop("name")
    backbone = build_backbone(backbone_name, **backbone_params)
    method = build_ssl_method(config.ssl.method, backbone, **config.ssl.params).to(device)
    print(f"  Backbone   : {backbone_name} ({backbone.feature_dim}-d)")
    print(f"  Parameters : {sum(p.numel() for p in method.parameters() if p.requires_grad):,} trainable")

    run_id = (f"{config.ssl.method}_{backbone_name}_{DATASET}_"
              f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    run_dir = os.path.join("experiments", run_id)
    os.makedirs(run_dir, exist_ok=True)

    mon = config.ssl.monitor
    monitor = make_knn_monitor(knn_train_loader, val_loader, device,
                               config.model.num_classes, mon.knn_k, mon.knn_temperature)
    trainer = SSLTrainer.from_config(method, config, device, monitor_fn=monitor, run_id=run_id)
    checkpoint = CheckpointManager(model=method, optimizer=trainer.optimizer, run_id=run_id)

    def on_best(epoch: int, knn_acc: float) -> None:
        checkpoint.save(epoch, 1 - knn_acc)
        torch.save(method.backbone.state_dict(), os.path.join(run_dir, "backbone_best.pth"))

    if config.experiment.save_checkpoints:
        trainer.on_best = on_best

    start = time.time()
    trainer.train(ssl_loader, epochs=config.training.epochs)
    print(f"Pretraining time: {time.time() - start:.0f} s")

    torch.save(method.backbone.state_dict(), os.path.join(run_dir, "backbone_last.pth"))
    plot_ssl_curves(trainer, config.ssl.params["projection_dim"], os.path.join(run_dir, "ssl_curves.png"))
    final_knn = trainer.knn_history[-1][1]
    trainer.save_hyperparams(extra_results={"final_knn": final_knn})

    if torch.cuda.is_available():
        print(f"Peak VRAM: allocated {torch.cuda.max_memory_allocated()/1e9:.1f} GB | "
              f"reserved {torch.cuda.max_memory_reserved()/1e9:.1f} GB")
    return run_dir, final_knn


# ---------------------------------------------------------------
# Phase 2: Linear probe (frozen backbone, linear head only)
# ---------------------------------------------------------------
def run_probe(backbone_path: str, run_id: str) -> dict:
    config = Config.from_dict({
        "experiment": {"dataset_name": DATASET, "run_id": run_id, "seed": SEED},
        "data": {"image_size": list(IMAGE_SIZE)},
        "model": {
            "num_classes": NUM_CLASSES,
            "backbone": BACKBONE,
            "pretrained_backbone": backbone_path,
            "freeze_backbone": True,
        },
        "training": {"lr": 0.1, "batch_size": PROBE_BATCH, "epochs": PROBE_EPOCHS,
                     "warm_up_epochs": 0, "label_smoothing": 0.0},
        "optimizer": {"type": "SGD", "params": {"momentum": 0.9, "weight_decay": 0.0}},
        "scheduler": {"type": "CosineAnnealingLR", "params": {"T_max": PROBE_EPOCHS}},
        "metrics": {
            "monitor_metric": "Top-1 Accuracy",
            "Top-1 Accuracy": ["topk_accuracy_torch", {"k": 1}],
            "Top-5 Accuracy": ("topk_accuracy_torch", {"k": 5}),
            "F1": ("f1_score_torch", {"num_classes": NUM_CLASSES}),
        },
    })
    set_seed(SEED)
    print(f"\n=== PHASE 2: Linear probe ({run_id}) ===")

    train_loader, val_loader = get_torchvision_dataset(
        dataset_name=DATASET, root_dir="./data", batch_size=config.training.batch_size,
        image_size=IMAGE_SIZE, use_computed_stats=USE_STATS,
        train_fraction=TRAIN_FRACTION, seed=SEED, persistent_workers=True,
    )
    model = Model(config)

    # Safety check: only the head should be trainable.
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all(n.startswith("net.head") for n in trainable), f"Backbone is not frozen: {trainable}"
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    trainer = Trainer.from_config(model=model, config=config, device=device,
                                  metrics_config=config.metrics)
    checkpoint = CheckpointManager(model=model, optimizer=trainer.optimizer, run_id=trainer.run_id)
    if config.experiment.save_checkpoints:
        trainer.on_best_model = checkpoint.save

    trainer.train(train_loader, val_loader, epochs=config.training.epochs)

    y, logits = trainer.collect_predictions(val_loader)
    cm = confusion_matrix_torch(y, logits, config.model.num_classes)
    Visualizer().plot_confusion_matrix(cm, train_loader.dataset.classes, run_id=model.run_id)
    trainer.save_hyperparams()

    return {
        "last": trainer.valid_metrics["Top-1 Accuracy"][-1],
        "best": trainer.get_final_metrics()["val_metrics"]["Top-1 Accuracy"],
    }


# ---------------------------------------------------------------
# Randomly initialized backbone (control)
# ---------------------------------------------------------------
def make_random_backbone(path: str) -> str:
    set_seed(SEED)
    params = dict(BACKBONE)
    name = params.pop("name")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(build_backbone(name, **params).state_dict(), path)
    return path


# ---------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()

    ssl_dir, final_knn = run_ssl()
    free_memory()

    results = {"Frozen BYOL": run_probe(os.path.join(ssl_dir, "backbone_last.pth"),
                                        run_id="linearprobe_" + os.path.basename(ssl_dir))}
    free_memory()

    if RUN_RANDOM_CONTROL:
        rand_path = make_random_backbone("experiments/random_backbone.pth")
        results["Frozen random"] = run_probe(rand_path, run_id="linearprobe_random_" +
                                             datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        free_memory()

    print("\n" + "=" * 52)
    print(f"Summary (labels used for the probe: {TRAIN_FRACTION:.0%}) | total duration {time.time() - t0:.0f} s")
    print(f"  Final SSL kNN accuracy      : {final_knn:.4f}")
    for name, r in results.items():
        print(f"  Linear probe {name:<16}: last epoch {r['last']:.4f} | best {r['best']:.4f}")
    print("=" * 52)