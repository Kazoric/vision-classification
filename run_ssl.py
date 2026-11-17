import os
import gc
import time
from datetime import datetime

import torch

# Core module imports
from core.config import Config
from core.checkpoint import CheckpointManager
from core.utils import set_seed, generate_run_id

# SSL imports
from models.backbones import build_backbone
from core.ssl.methods import build_ssl_method
from core.ssl.evaluation import make_knn_monitor
from core.ssl.ssl_trainer import SSLTrainer
from core.ssl.plots import plot_ssl_curves
from core.ssl.methods import SSL_METHODS

# import core.ssl.methods.fcmae

# Data imports
from data_loader import get_ssl_dataloaders

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
SMOKE = True                  # True: smoke test (2 SSL epochs on 5,000 images, 3 probe epochs)
SEED = 42
DATASET = "STL10"
IMAGE_SIZE = (96, 96)
USE_STATS = True
BACKBONE = {"name": "resnet18", "stem": "small"}
NUM_CLASSES = 10

SSL_EPOCHS = 2 if SMOKE else 100
SSL_WARMUP = 0 if SMOKE else 10
LIMIT = 5000 if SMOKE else None

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------
# Phase 1: SSL pretraining on the ENTIRE training set (without labels)
# ---------------------------------------------------------------
def run_ssl():
    config = Config.from_dict({
        "experiment": {"dataset_name": DATASET, "seed": SEED},
        "data": {"image_size": list(IMAGE_SIZE)},
        "model": {"num_classes": NUM_CLASSES, "backbone": BACKBONE},
        "training": {
            "lr": 0.001, "batch_size": 128, "epochs": SSL_EPOCHS, "warm_up_epochs": SSL_WARMUP,
            "mixed_precision": True,
            "channels_last": True
        },
        "optimizer": {"type": "AdamW", "params": {"weight_decay": 0.0001, "betas": [0.9, 0.999]}},
        "scheduler": {"type": "CosineAnnealingLR", "params": {"T_max": SSL_EPOCHS - SSL_WARMUP}},
        "ssl": {
            "method": "simsiam",
            # "params": {"projection_dim": 256, "hidden_dim": 1024, "tau_base": 0.99},
            # "params": {"mask_ratio": 0.6, "decoder_dim": 512, "decoder_depth": 1, "norm_pix_loss": True},
            "params": {"projection_dim": 256, "hidden_dim": 1024},
            "augmentation": {
                "crop_scale": [0.2, 1.0],
                "color_jitter": [0.4, 0.4, 0.2, 0.1],
                "color_jitter_prob": 0.8,
                "grayscale_prob": 0.2,
                "blur_probs": [0.0, 0.0],
                "solarize_probs": [0.0, 0.2],
            },
            "monitor": {"knn_every": 1 if SMOKE else 10, "knn_k": 20, "knn_temperature": 0.1},
            # "mixed_precision": True,
            # "channels_last": True
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
        num_views=SSL_METHODS[config.ssl.method].num_views
    )
    assert class_names is None or config.model.num_classes == len(class_names), \
        f"num_classes={config.model.num_classes}, but the dataset has {len(class_names)} classes"

    backbone_params = dict(config.model.backbone)
    backbone_name = backbone_params.pop("name")
    backbone = build_backbone(backbone_name, **backbone_params)
    method = build_ssl_method(config.ssl.method, backbone, **config.ssl.params).to(device)
    print(f"  Backbone   : {backbone_name} ({backbone.feature_dim}-d)")
    print(f"  Parameters : {sum(p.numel() for p in method.parameters() if p.requires_grad):,} trainable")

    # Set run_id
    if config.experiment.run_id is None:
        exp_name = f"{config.ssl.method}_{backbone_name}"
        run_id = generate_run_id(
            model_name=exp_name,
            dataset_name=config.experiment.dataset_name,
        )
        config.experiment.run_id = run_id

    run_dir = os.path.join("experiments", config.experiment.run_id)
    os.makedirs(run_dir, exist_ok=True)

    mon = config.ssl.monitor
    monitor = make_knn_monitor(knn_train_loader, val_loader, device,
                               config.model.num_classes, mon.knn_k, mon.knn_temperature)
    checkpoint = None
    if config.experiment.save_checkpoints:
        checkpoint = CheckpointManager(
            run_id=config.experiment.run_id,
            model_name=config.model.backbone['name'],
        )
    trainer = SSLTrainer.from_config(method, config, device, monitor_fn=monitor, save_function=checkpoint.save_backbone if checkpoint else None)

    start = time.time()
    trainer.train(ssl_loader, epochs=config.training.epochs)
    print(f"Pretraining time: {time.time() - start:.0f} s")

    plot_ssl_curves(trainer, config.ssl.params["projection_dim"], os.path.join(run_dir, "ssl_curves.png"))
    final_knn = trainer.knn_history[-1][1]
    trainer.save_hyperparams(extra_results={"final_knn": final_knn})

    if torch.cuda.is_available():
        print(f"Peak VRAM: allocated {torch.cuda.max_memory_allocated()/1e9:.1f} GB | "
              f"reserved {torch.cuda.max_memory_reserved()/1e9:.1f} GB")
    return run_dir, final_knn


# ---------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()

    ssl_dir, final_knn = run_ssl()

    print("\n" + "=" * 52)
    print(f"  Final SSL kNN accuracy      : {final_knn:.4f}")
    print("=" * 52)