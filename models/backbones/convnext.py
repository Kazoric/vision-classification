import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any

from .base import BaseBackbone
from .registry import register_backbone

# --- ConvNeXt Utilities ---

class LayerNorm(nn.Module):
    """
    LayerNorm supporting channels_last (default) or channels_first formats.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unknown data format: {self.data_format}")
        self.normalized_shape = (normalized_shape, )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True) -> torch.Tensor:
    """Stochastic Depth (DropPath)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class ConvNeXtBlock(nn.Module):
    """ Standard ConvNeXt Block """
    def __init__(self, dim: int, drop_path: float = 0., layer_scale_init_value: float = 1e-6):
        super().__init__()
        # 1. Depthwise conv (7x7)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # 2. LayerNorm
        self.norm = LayerNorm(dim, eps=1e-6)
        
        # 3. Pointwise convs (Inverted MLP: dim -> 4*dim -> dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # 4. Layer Scale & Stochastic Depth
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        
        # Switch to channels_last format for LN and Linear layers
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma * x
            
        # Switch back to channels_first format
        x = x.permute(0, 3, 1, 2)

        return identity + self.drop_path(x)


@register_backbone("convnext")
class ConvNeXtBackbone(BaseBackbone):

    def __init__(
        self,
        depths: Tuple[int, ...] = (3, 3, 9, 3),
        dims: Tuple[int, ...] = (96, 192, 384, 768),
        stem: str = "cifar",
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        **kwargs
    ):
        super().__init__()
        self.depths = depths
        self.dims = dims

        # 1. Stem
        if stem == "cifar":        # Small images (32x32)
            self.stem = nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=3, stride=1, padding=1),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
        elif stem == "imagenet":   # Large images (>= 128px)
            self.stem = nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
        else:
            raise ValueError(f"Unknown stem type: {stem}")

        # 2. Stochastic depth rates distributed across blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur_dp = 0

        # 3. Stages
        stages_list = []
        in_dim = dims[0]
        for i, (out_dim, depth) in enumerate(zip(dims, depths)):
            stage_dp_rates = dp_rates[cur_dp : cur_dp + depth]
            cur_dp += depth
            
            stage = self._make_stage(in_dim=in_dim, out_dim=out_dim, depth=depth, dp_rates=stage_dp_rates, is_first=(i == 0), layer_scale_init_value=layer_scale_init_value)
            stages_list.append(stage)
            in_dim = out_dim

        self.stages = nn.Sequential(*stages_list)

        # 4. Final normalization & pooling
        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        
        self._feature_dim = dims[-1]
        self._init_weights()

    def _make_stage(self, in_dim: int, out_dim: int, depth: int, dp_rates: List[float], is_first: bool, layer_scale_init_value: float) -> nn.Sequential:
        layers = []
        # Downsampling between stages (except for the first stage handled by the stem)
        if not is_first:
            layers.append(
                nn.Sequential(
                    LayerNorm(in_dim, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
                )
            )

        # ConvNeXt blocks
        for j in range(depth):
            layers.append(ConvNeXtBlock(dim=out_dim, drop_path=dp_rates[j], layer_scale_init_value=layer_scale_init_value))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


@register_backbone("convnext_atto")
def convnextv2_atto(**kwargs):
    model = ConvNeXtBackbone(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

@register_backbone("convnext_femto")
def convnextv2_femto(**kwargs):
    model = ConvNeXtBackbone(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

@register_backbone("convnext_pico")
def convnext_pico(**kwargs):
    model = ConvNeXtBackbone(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

@register_backbone("convnext_nano")
def convnextv2_nano(**kwargs):
    model = ConvNeXtBackbone(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

@register_backbone("convnext_tiny")
def convnextv2_tiny(**kwargs):
    model = ConvNeXtBackbone(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

@register_backbone("convnext_small")
def convnextv2_small(**kwargs):
    model = ConvNeXtBackbone(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

@register_backbone("convnext_base")
def convnextv2_base(**kwargs):
    model = ConvNeXtBackbone(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

@register_backbone("convnext_large")
def convnextv2_large(**kwargs):
    model = ConvNeXtBackbone(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

@register_backbone("convnext_huge")
def convnextv2_huge(**kwargs):
    model = ConvNeXtBackbone(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model

# @register_backbone("convnext_tiny")
# def convnext_tiny(**kwargs) -> ConvNeXtBackbone:
#     return ConvNeXtBackbone(
#         depths=(3, 3, 9, 3),
#         dims=(96, 192, 384, 768),
#         **kwargs,
#     )

# @register_backbone("convnext_small")
# def convnext_small(**kwargs) -> ConvNeXtBackbone:
#     return ConvNeXtBackbone(
#         depths=(3, 3, 27, 3),
#         dims=(96, 192, 384, 768),
#         **kwargs,
#     )

# @register_backbone("convnext_base")
# def convnext_base(**kwargs) -> ConvNeXtBackbone:
#     return ConvNeXtBackbone(
#         depths=(3, 3, 27, 3),
#         dims=(128, 256, 512, 1024),
#         **kwargs,
#     )

# @register_backbone("convnext_large")
# def convnext_large(**kwargs) -> ConvNeXtBackbone:
#     return ConvNeXtBackbone(
#         depths=(3, 3, 27, 3),
#         dims=(192, 384, 768, 1536),
#         **kwargs,
#     )