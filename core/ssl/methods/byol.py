import copy
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SSLMethod
from core.ssl.pooling import global_pool
from .registry import register_ssl_method

def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                         nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim))

def _byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Computes 2 - 2 cos(p, z) in float32, even under autocast."""
    p = F.normalize(p.float(), dim=-1)
    z = F.normalize(z.float(), dim=-1)
    return (2 - 2 * (p * z).sum(dim=-1)).mean()

@register_ssl_method("byol")
class BYOL(SSLMethod):
    num_views = 2
    def __init__(self, backbone, projection_dim: int = 256,
                 hidden_dim: int = 1024, tau_base: float = 0.99):
        super().__init__()
        d = backbone.feature_dim
        # Online network
        self.backbone = backbone
        self.projector = _mlp(d, hidden_dim, projection_dim)
        self.predictor = _mlp(projection_dim, hidden_dim, projection_dim)
        # Target network: exact copy, without gradients, updated via EMA
        self.target_backbone = copy.deepcopy(backbone)
        self.target_projector = copy.deepcopy(self.projector)
        for p in itertools.chain(self.target_backbone.parameters(),
                                 self.target_projector.parameters()):
            p.requires_grad = False
        self.tau_base = tau_base
        self.tau = tau_base

    def _online(self, x):
        return self.projector(global_pool(self.backbone(x)))

    def _target(self, x):
        return self.target_projector(global_pool(self.target_backbone(x)))

    def forward(self, v1, v2):
        z1, z2 = self._online(v1), self._online(v2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        with torch.no_grad():
            t1, t2 = self._target(v1), self._target(v2)
        loss = 0.5 * (_byol_loss(p1, t2) + _byol_loss(p2, t1))
        return {"loss": loss, "z": z1.detach()}

    @torch.no_grad()
    def after_step(self, progress: float) -> None:
        # tau increases from tau_base to 1 following a cosine schedule
        self.tau = 1 - (1 - self.tau_base) * (math.cos(math.pi * progress) + 1) / 2
        online = itertools.chain(self.backbone.parameters(), self.projector.parameters())
        target = itertools.chain(self.target_backbone.parameters(),
                                 self.target_projector.parameters())
        for o, t in zip(online, target):
            t.mul_(self.tau).add_(o.detach(), alpha=1 - self.tau)