import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any

from .base import BaseBackbone
from .registry import register_backbone


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self‑attention
        x_res = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        x = x_res + attn_output

        # MLP
        x_res = x
        x = self.norm2(x)
        x = self.mlp(x)
        return x_res + x

@register_backbone("vit")
class ViTArchitecture(nn.Module):
    """Vision Transformer architecture."""

    def __init__(self, image_size: int, patch_size: int, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()

        # Calculate number of patches
        assert image_size[0] % patch_size == 0, "Image size must be divisible by patch size."
        self.num_patches = (image_size[0] // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self._feature_dim = embed_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B = x.shape[0]
        # Patchify and flatten
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

        # Prepare class token and position embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoder
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls_token_final = x[:, 0]  # (B, embed_dim)
        return cls_token_final

    @property
    def feature_dim(self) -> int:
        return self._feature_dim