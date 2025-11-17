import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any

from core.model_base import Model


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


class ViTArchitecture(nn.Module):
    """Vision Transformer architecture."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Calculate number of patches
        assert image_size[0] % patch_size == 0, "Image size must be divisible by patch size."
        self.num_patches = (image_size[0] // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer encoder
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Init weights
        self.apply(self._init_weights)
        self.to(self.device)

    @staticmethod
    def _init_weights(m: nn.Module):
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

        # Classification head on class token
        cls_token_final = x[:, 0]  # (B, embed_dim)
        logits = self.head(cls_token_final)
        return logits


class ViTModel(Model):
    """Model wrapper for Vision Transformer."""

    def __init__(
        self,
        num_classes: int = 100,
        image_size: Tuple[int, int] = (32, 32),
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        **kwargs,
    ):
        self.name = "ViT"
        self.params = {
            "num_classes": num_classes,
            "image_size": image_size,
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "dropout": dropout,
        }
        super().__init__(num_classes=num_classes, **kwargs)

    def build_model(self):
        print(
            f"Building ViT with {self.params['depth']} layers, "
            f"patch size {self.params['patch_size']}x{self.params['patch_size']}."
        )
        return ViTArchitecture(
            image_size=self.params["image_size"],
            patch_size=self.params["patch_size"],
            num_classes=self.num_classes,
            embed_dim=self.params["embed_dim"],
            depth=self.params["depth"],
            num_heads=self.params["num_heads"],
            mlp_ratio=self.params["mlp_ratio"],
            dropout=self.params["dropout"],
        )

    def get_model_specific_params(self):
        return self.params