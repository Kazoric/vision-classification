from typing import Dict

import torch
import torch.nn as nn

class SSLMethod(nn.Module):
    """Common interface: self.backbone (network to transfer), forward(v1, v2), after_step."""
    backbone: nn.Module

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns {"loss": ..., "z": detached embeddings for collapse monitoring}."""
        raise NotImplementedError

    def after_step(self, progress: float) -> None:
        """Called after each optimizer.step(); progress is in [0, 1]."""