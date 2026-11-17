import torch
import torch.nn as nn

from .base import SSLMethod
from .registry import register_ssl_method


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """[B, 3, H, W] -> [B, h*w, p*p*3]"""
    B, C, H, W = x.shape
    h, w = H // p, W // p
    x = x.reshape(B, C, h, p, w, p)
    return torch.einsum("nchpwq->nhwpqc", x).reshape(B, h * w, p * p * C)


@register_ssl_method("fcmae")
class FCMAE(SSLMethod):
    num_views = 1

    def __init__(self, backbone, mask_ratio: float = 0.6, decoder_dim: int = 512,
                 decoder_depth: int = 1, norm_pix_loss: bool = True):
        super().__init__()
        for attr in ("total_stride", "feature_dim", "make_block"):
            if not hasattr(backbone, attr):
                raise AttributeError(f"FCMAE : le backbone doit définir '{attr}' (voir fcmae.py).")
        self.backbone = backbone
        self.patch = backbone.total_stride
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.proj = nn.Conv2d(backbone.feature_dim, decoder_dim, kernel_size=1)
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_dim, 1, 1))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder = nn.Sequential(*[backbone.make_block(decoder_dim) for _ in range(decoder_depth)])
        self.pred = nn.Conv2d(decoder_dim, 3 * self.patch ** 2, kernel_size=1)

    def random_mask(self, B: int, h: int, w: int, device) -> torch.Tensor:
        """[B, 1, h, w], 1 = masqué. Même nombre de patchs masqués par image."""
        L = h * w
        len_keep = int(L * (1 - self.mask_ratio))
        rank = torch.rand(B, L, device=device).argsort(1).argsort(1)
        return (rank >= len_keep).float().view(B, 1, h, w)

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        p = self.patch
        if H % p or W % p:
            raise ValueError(f"Taille d'image {H}x{W} non multiple du patch {p} (stride total du backbone).")
        h, w = H // p, W // p
        mask = self.random_mask(B, h, w, x.device)

        feats = self.backbone(x, mask=mask)                       # [B, C, h, w]
        z = self.proj(feats)
        z = z * (1 - mask) + self.mask_token.to(z.dtype) * mask   # jeton appris aux positions masquées
        pred = self.pred(self.decoder(z))                         # [B, 3p², h, w]
        pred = pred.permute(0, 2, 3, 1).reshape(B, h * w, -1).float()

        target = patchify(x.float(), p)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
        m = mask.flatten(1)
        loss = (((pred - target) ** 2).mean(dim=-1) * m).sum() / m.sum()
        return {"loss": loss, "z": None}