import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any

from .base import BaseBackbone
from .registry import register_backbone

# --- ConvNeXtV2 Utilities ---

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

class GRN(nn.Module):
    """ 
    GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

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


class ConvNeXtV2Block(nn.Module):
    """ ConvNeXt V2 Block """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # 1. Depthwise conv (7x7)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        
        # 2. Norm (LayerNorm)
        self.norm = LayerNorm(dim, eps=1e-6)
        
        # 3. Pointwise convs (Inverted MLP: dim -> 4*dim -> dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # 4. Stochastic Depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, keep=None):
        input = x
        x = self.dwconv(x)

        if keep is not None:
            x = x * keep
        
        # Switch to channels_last format for LN and Linear layers
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        
        x = self.pwconv1(x)
        x = self.act(x)
        if keep is not None:
            x = x * keep.permute(0, 2, 3, 1)
        x = self.grn(x)
        x = self.pwconv2(x)
            
        # Switch back to channels_first format
        x = x.permute(0, 3, 1, 2) 
        if keep is not None:
            x = x * keep

        x = input + self.drop_path(x)
        return x


class _MaskedBlocks(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
 
    def forward(self, x, keep=None):
        for b in self.blocks:
            x = b(x, keep)
        return x

@register_backbone("convnextv2")
class ConvNeXtV2Backbone(nn.Module):
    def __init__(self, depths=(2, 2, 6, 2), dims=(64, 128, 256, 512), stem="imagenet"):
        super().__init__()
        self.stem_stride = {"cifar": 1, "small": 2, "imagenet": 4}[stem]
        k = self.stem_stride
        self.stem = nn.Sequential(nn.Conv2d(3, dims[0], k, k) if k > 1 else nn.Conv2d(3, dims[0], 3, 1, 1),
                                  LayerNorm(dims[0], data_format="channels_first"))
        self.downsamples = nn.ModuleList([nn.Identity()] + [
            nn.Sequential(LayerNorm(dims[i - 1], data_format="channels_first"), nn.Conv2d(dims[i - 1], dims[i], 2, 2))
            for i in range(1, len(dims))])
        self.stages = nn.ModuleList([_MaskedBlocks([ConvNeXtV2Block(d) for _ in range(n)]) for n, d in zip(depths, dims)])
        self.norm = LayerNorm(dims[-1], data_format="channels_first")
        self._dims = dims
        self.apply(self._init)
 
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
 
    @property
    def feature_dim(self):
        return self._dims[-1]
 
    @property
    def total_stride(self):
        return self.stem_stride * 2 ** (len(self._dims) - 1)
 
    def make_block(self, dim):
        return ConvNeXtV2Block(dim)
 
    def forward(self, x, mask=None):
        """mask : [B, 1, h, w], 1 = masked."""
        n = len(self.stages)
        keep_last = None if mask is None else 1.0 - mask
        keep_of = lambda i: None if keep_last is None else keep_last.repeat_interleave(2 ** (n - 1 - i), 2).repeat_interleave(2 ** (n - 1 - i), 3).to(x.dtype)
        if mask is not None:
            keep0 = keep_of(0)
            x = x * keep0.repeat_interleave(self.stem_stride, 2).repeat_interleave(self.stem_stride, 3)  # pixels masqués = 0
        x = self.stem(x)
        for i, (down, stage) in enumerate(zip(self.downsamples, self.stages)):
            keep = keep_of(i)
            x = down(x)
            if keep is not None:
                x = x * keep
            x = stage(x, keep)
        x = self.norm(x)
        return x


@register_backbone("convnextv2_atto")
def convnextv2_atto(**kwargs):
    model = ConvNeXtV2Backbone(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

@register_backbone("convnextv2_femto")
def convnextv2_femto(**kwargs):
    model = ConvNeXtV2Backbone(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

@register_backbone("convnextv2_pico")
def convnextv2_pico(**kwargs):
    model = ConvNeXtV2Backbone(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

@register_backbone("convnextv2_nano")
def convnextv2_nano(**kwargs):
    model = ConvNeXtV2Backbone(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

@register_backbone("convnextv2_tiny")
def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2Backbone(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

@register_backbone("convnextv2_small")
def convnextv2_small(**kwargs):
    model = ConvNeXtV2Backbone(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

@register_backbone("convnextv2_base")
def convnextv2_base(**kwargs):
    model = ConvNeXtV2Backbone(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

@register_backbone("convnextv2_large")
def convnextv2_large(**kwargs):
    model = ConvNeXtV2Backbone(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

@register_backbone("convnextv2_huge")
def convnextv2_huge(**kwargs):
    model = ConvNeXtV2Backbone(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model