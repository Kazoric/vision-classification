import os
import json
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn

from models.backbones import build_backbone
from core.config import Config
from core.optim import build_optimizer, build_scheduler
from models.classifier import Classifier


class Model(nn.Module):
    """
    Modèle de classification d'images : Classifier (backbone + tête) + loss,
    optimiseur, scheduler et gestion du run, le tout piloté par la config.
    """

    def __init__(self, config: Config, device: Optional[str] = None) -> None:
        super().__init__()
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        m = config.model
        backbone_params = dict(m.backbone)
        backbone_name = backbone_params.pop("name")
        self.name = backbone_name
        self.num_classes = m.num_classes
        self.dataset_name = config.experiment.dataset_name
        self.lr = config.training.lr

        # --- Run ID ---
        run_id = config.experiment.run_id
        if run_id is None:
            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_id = f"{self.name}_{self.dataset_name}_{date}"
        self.run_id = run_id

        # --- Architecture ---
        backbone = build_backbone(backbone_name, **backbone_params)
        self.net = Classifier(backbone, m.num_classes,
                              hidden_dim=m.hidden_dim, dropout=m.dropout)

        # Poids SSL puis gel éventuel : AVANT la création de l'optimiseur
        if m.pretrained_backbone:
            self.net.load_backbone(m.pretrained_backbone)
        if m.freeze_backbone:
            self.net.freeze_backbone()

        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
        self.to(self.device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)

    def save_backbone(self, path: str) -> None:
        """Exporte le backbone seul (réutilisable pour le SSL ou un autre Classifier)."""
        torch.save(self.net.backbone.state_dict(), path)

    def save_hyperparams(self, extra_results: Optional[Dict] = None) -> None:
        meta = asdict(self.config)
        meta["experiment"]["run_id"] = self.run_id

        if extra_results is not None:
            meta["results"] = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "best_validation_results": extra_results,
            }

        path = os.path.join("experiments", self.run_id, "meta.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Configuration saved: {path}")