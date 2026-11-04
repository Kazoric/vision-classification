import torch


def global_pool(f: torch.Tensor) -> torch.Tensor:
    """Carte de features [B, C, H, W] -> vecteur [B, C] (moyenne spatiale, équivalent à
    AdaptiveAvgPool2d(1) + flatten). Un tenseur déjà plat [B, D] est renvoyé tel quel."""
    if f.ndim == 4:
        return f.mean(dim=(2, 3))
    if f.ndim != 2:
        raise ValueError(f"Features inattendues : {tuple(f.shape)} (attendu [B, C, H, W] ou [B, D])")
    return f