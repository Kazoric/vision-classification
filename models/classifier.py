import torch
from torch import nn

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
        self.pool = nn.AdaptiveAvgPool2d(1)
        self._backbone_frozen = False

    def forward(self, x):
        x = self.backbone(x)
        if x.ndim == 4:
            # CNN: (B, C, H, W) -> (B, C)
            x = self.pool(x).flatten(1)
        elif x.ndim == 3:
            # Transformer: (B, N, D) -> (B, D)
            x = x[:, 0]
        return self.head(x)

    # --- SSL utilities ---
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