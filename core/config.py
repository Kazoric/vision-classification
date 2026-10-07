from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ExperimentConfig:
    dataset_name: str
    run_id: Optional[str] = None
    save_checkpoints: bool = True
    seed: int = 42


@dataclass
class DataConfig:
    image_size: int = 32
    # Valeurs CIFAR-10 par défaut : à recalculer pour un dataset industriel
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    train_fraction: float = 1.0        # 0.01 / 0.1 : sous-ensemble stratifié labellisé


@dataclass
class ModelConfig:
    num_classes: int
    # Fabrique : {"name": "resnet", "block": "basic", "layers": [2,2,2,2], "stem": "cifar"}
    backbone: Dict[str, Any] = field(default_factory=lambda: {"name": "resnet"})
    hidden_dim: Optional[int] = None
    dropout: float = 0.0
    pretrained_backbone: Optional[str] = None   # chemin d'un backbone SSL
    freeze_backbone: bool = False                # True = linear probe

    def __post_init__(self):
        if "name" not in self.backbone:
            raise ValueError("model.backbone doit contenir une clé 'name'")
        if self.freeze_backbone and not self.pretrained_backbone:
            raise ValueError("freeze_backbone=True n'a de sens qu'avec pretrained_backbone")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(**d)


@dataclass
class TrainingConfig:
    lr: float
    batch_size: int
    epochs: int
    warm_up_epochs: int = 0
    label_smoothing: float = 0.0     # supervisé uniquement


@dataclass
class OptimizerConfig:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    type: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsConfig:
    monitor_metric: str = "Top-1 Accuracy"
    monitor_mode: str = "max"
    configs: Dict[str, Any] = field(
        default_factory=lambda: {"Top-1 Accuracy": ("topk_accuracy_torch", {"k": 1})}
    )


# ----------------------------------------------------------------------
# SSL
# ----------------------------------------------------------------------

@dataclass
class SSLAugmentationConfig:
    crop_scale: Tuple[float, float] = (0.2, 1.0)       # 0.08 pour ImageNet, 0.2 pour CIFAR
    flip_prob: float = 0.5
    color_jitter: Tuple[float, float, float, float] = (0.4, 0.4, 0.2, 0.1)  # b, c, s, h
    color_jitter_prob: float = 0.8
    grayscale_prob: float = 0.2
    blur_probs: Tuple[float, float] = (0.0, 0.0)       # (vue 1, vue 2) ; (1.0, 0.1) pour ImageNet
    solarize_probs: Tuple[float, float] = (0.0, 0.2)   # (vue 1, vue 2)


@dataclass
class SSLMonitorConfig:
    knn_every: int = 10          # epochs entre deux évaluations kNN (0 = désactivé)
    knn_k: int = 20
    knn_temperature: float = 0.1


@dataclass
class SSLConfig:
    method: str = "byol"                                   # "byol" | "simclr" | ...
    # Paramètres propres à la méthode (fabrique, comme backbone)
    #   byol   : {"projection_dim": 256, "hidden_dim": 2048, "tau_base": 0.99}
    #   simclr : {"projection_dim": 128, "hidden_dim": 2048, "temperature": 0.5}
    params: Dict[str, Any] = field(default_factory=dict)
    augmentation: SSLAugmentationConfig = field(default_factory=SSLAugmentationConfig)
    monitor: SSLMonitorConfig = field(default_factory=SSLMonitorConfig)
    mixed_precision: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SSLConfig":
        return cls(
            method=d.get("method", "byol"),
            params=d.get("params", {}),
            augmentation=SSLAugmentationConfig(**d.get("augmentation", {})),
            monitor=SSLMonitorConfig(**d.get("monitor", {})),
            mixed_precision=d.get("mixed_precision", True),
        )


# ----------------------------------------------------------------------
# Config maîtresse
# ----------------------------------------------------------------------

@dataclass
class Config:
    experiment: ExperimentConfig
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    data: DataConfig = field(default_factory=DataConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    ssl: Optional[SSLConfig] = None          # présent uniquement pour le pré-entraînement SSL

    def __post_init__(self):
        if self.training.warm_up_epochs >= self.training.epochs:
            raise ValueError("warm_up_epochs doit être < epochs")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        metrics_raw = d.get("metrics", {}).copy()
        monitor_metric = metrics_raw.pop("monitor_metric", "Top-1 Accuracy")
        monitor_mode = metrics_raw.pop("monitor_mode", "max")

        metrics = MetricsConfig(monitor_metric=monitor_metric, monitor_mode=monitor_mode)
        if metrics_raw:                       # sinon on garde la métrique par défaut
            metrics.configs = metrics_raw

        return cls(
            experiment=ExperimentConfig(**d["experiment"]),
            model=ModelConfig.from_dict(d["model"]),
            training=TrainingConfig(**d["training"]),
            optimizer=OptimizerConfig(**d["optimizer"]),
            scheduler=SchedulerConfig(**d.get("scheduler", {})),
            data=DataConfig(**d.get("data", {})),
            metrics=metrics,
            ssl=SSLConfig.from_dict(d["ssl"]) if "ssl" in d else None,
        )