from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    """
    Abstract base class for all feature extraction backbones (e.g., ResNet).
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        pass