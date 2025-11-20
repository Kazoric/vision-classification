import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any

from core.model_base import Model

# --- ConvNeXt Utilities ---

class LayerNorm(nn.Module):
    """
    LayerNorm that supports two data formats: 
    channels_last (default) or channels_first. 
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Stochastic Depth (DropPath) function."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """DropPath module wrapper."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block. 
    Equivalent to TransformerEncoderBlock but with convolutions.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # 1. Depthwise conv (7x7) -> simulates the large attention window of ViT
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        
        # 2. Norm (LayerNorm)
        self.norm = LayerNorm(dim, eps=1e-6)
        
        # 3. Pointwise convs (Inverted MLP: dim -> 4*dim -> dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # 4. Layer Scale and Stochastic Depth
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        
        # Permute to switch from (N, C, H, W) to (N, H, W, C) for LayerNorm and Linear layers
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma * x
            
        # Return to (N, C, H, W) format
        x = x.permute(0, 3, 1, 2) 

        x = input + self.drop_path(x)
        return x


class ConvNeXtArchitecture(nn.Module):
    """ConvNeXt Architecture."""

    def __init__(
        self,
        num_classes: int,
        depths: List[int],
        dims: List[int],
        drop_path_rate: float = 0.0,
        small_input: bool = False,
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.downsample_layers = nn.ModuleList() # Stem + 3 downsampling layers
        
        # 1. Stem (Patchify similar to ViT but with Conv)
        if small_input:
            # CONFIGURATION CIFAR (32x32 -> 32x32)
            stem = nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=3, stride=2, padding=1),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
        else:
            # CONFIGURATION IMAGENET (224x224 -> 56x56)
            stem = nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
        self.downsample_layers.append(stem)
        
        # 2. Downsampling layers between stages
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 3. ConvNeXt stages (sequence of blocks)
        self.stages = nn.ModuleList() 
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j]) 
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # 4. Final Norm & Head
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # Final norm
        self.head = nn.Linear(dims[-1], num_classes)

        # Init weights
        self.apply(self._init_weights)
        self.to(self.device)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Iterate through the 4 stages (pyramidal architecture)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        # Global Average Pooling (N, C, H, W) -> (N, C)
        # ConvNeXt performs pooling *before* the last LayerNorm
        x = x.mean([-2, -1]) 
        
        x = self.norm(x)
        x = self.head(x)
        return x


class ConvNeXtModel(Model):
    """Model wrapper for ConvNeXt."""

    def __init__(
        self,
        num_classes: int = 100,
        image_size: Tuple[int, int] = (32, 32),
        # Architecture hyperparameters
        embed_dim: int = 96,               # Initial embedding dimension (Stage 0)
        depths: List[int] = [3, 3, 9, 3],  # Number of blocks per stage (Default: ConvNeXt-Tiny)
        drop_path_rate: float = 0.0,       # Stochastic depth rate
        **kwargs,
    ):
        self.name = "ConvNeXt"
        self.small_input = image_size[0] <= 64
        
        # 1. SAVING CONFIGURATION
        # We store exactly what is passed to __init__ in self.params.
        # This dictionary is the "recipe" to recreate the model instance later.
        self.params = {
            "num_classes": num_classes,
            "image_size": image_size,
            "embed_dim": embed_dim,
            "depths": depths,             # Explicitly stored
            "drop_path_rate": drop_path_rate,
        }

        # 2. DERIVED ATTRIBUTES
        # These are calculated based on the inputs but not stored in self.params
        # to avoid arguments mismatch during model reloading.
        
        # Standard ConvNeXt width expansion: [dim, dim*2, dim*4, dim*8]
        # e.g., [96, 192, 384, 768]
        self.dims = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
        
        # Initialize base class
        super().__init__(num_classes=num_classes, **kwargs)

    def build_model(self):
        print(
            f"Building ConvNeXt with depths {self.params['depths']} " 
            f"and dims {self.dims}."
        )
        
        # Pass the stored params and the calculated dims to the architecture
        return ConvNeXtArchitecture(
            num_classes=self.params["num_classes"],
            small_input=self.small_input,
            depths=self.params["depths"],        # Uses the list passed in __init__
            dims=self.dims,                      # Uses the calculated dimensions
            drop_path_rate=self.params["drop_path_rate"],
        )

    def get_model_specific_params(self):
        return self.params
    
    def get_target_layer(self):
        # On vise le dernier bloc du dernier stage (stage 3)
        # self.model est l'instance de ConvNeXtArchitecture
        # stages[3] est le dernier stage
        # [-1] est le dernier bloc ConvNeXtBlock de ce stage
        return self.model.stages[-1][-1]