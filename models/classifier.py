import torch
from torch import nn
from typing import List, Dict, Optional, Tuple

from .backbones.base import BaseBackbone

class Classifier(nn.Module):
    def __init__(self, backbone: BaseBackbone, num_classes: int,
                 hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        d = backbone.feature_dim
        layers = [nn.Dropout(dropout)] if dropout > 0 else []
        if hidden_dim:
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            d = hidden_dim
        layers.append(nn.Linear(d, num_classes))
        self.head = nn.Sequential(*layers)
        self._backbone_frozen = False

    def forward(self, x):
        return self.head(self.backbone(x))

    # --- utilitaires pour le SSL ---
    def load_backbone(self, path: str):
        self.backbone.load_state_dict(torch.load(path, map_location="cpu"))

    def freeze_backbone(self, frozen: bool = True):
        self._backbone_frozen = frozen
        for p in self.backbone.parameters():
            p.requires_grad = not frozen

    def train(self, mode: bool = True):
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self