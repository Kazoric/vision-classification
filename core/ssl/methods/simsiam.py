import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SSLMethod
from core.ssl.pooling import global_pool
from .registry import register_ssl_method

def _projector(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
                         nn.Linear(hidden_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
                         nn.Linear(hidden_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim, affine=False))

def _predictor(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
                         nn.Linear(hidden_dim, out_dim))

def _sim(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # z = z.detach()

    # p = F.normalize(p.float(), dim=1)
    # z = F.normalize(z.float(), dim=1)

    # return -(p * z).sum(dim=1).mean()
    return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

@register_ssl_method("simsiam")
class SimSiam(SSLMethod):
    num_views = 2
    def __init__(self, backbone, projection_dim: int = 256, hidden_dim: int = 1024):
        super().__init__()
        d = backbone.feature_dim
        # Online network
        self.backbone = backbone
        self.projector = _projector(d, hidden_dim, projection_dim)
        self.predictor = _predictor(projection_dim, hidden_dim, projection_dim)

    def _online(self, x):
        return self.projector(global_pool(self.backbone(x)))

    def forward(self, v1, v2):
        z1, z2 = self._online(v1), self._online(v2)
        p1, p2 = self.predictor(z1), self.predictor(z2)

        loss = 0.5 * (_sim(p1, z2) + _sim(p2, z1))
        return {"loss": loss, "z": z1.detach()}