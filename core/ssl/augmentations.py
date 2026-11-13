from torchvision import transforms as T


def _build_view(aug, image_size, mean, std, blur_p=0.0, solarize_p=0.0, strong=True):
    h, w = image_size
    ops = [T.RandomResizedCrop((h, w), scale=tuple(aug.crop_scale), interpolation=T.InterpolationMode.BICUBIC),
           T.RandomHorizontalFlip(aug.flip_prob)]
    if strong:                                   # BYOL : couleur, gris, flou, solarisation
        ops += [T.RandomApply([T.ColorJitter(*aug.color_jitter)], p=aug.color_jitter_prob),
                T.RandomGrayscale(aug.grayscale_prob)]
        if blur_p > 0:
            k = max(3, int(0.1 * min(h, w)) // 2 * 2 + 1)
            ops.append(T.RandomApply([T.GaussianBlur(k, sigma=(0.1, 2.0))], p=blur_p))
        if solarize_p > 0:
            ops.append(T.RandomSolarize(threshold=128, p=solarize_p))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


class MultiViewTransform:
    """Image PIL -> tuple de num_views tenseurs.
    2 vues : BYOL (augmentations fortes, asymétriques). 1 vue : MIM / FCMAE (recadrage + flip seulement,
    la couleur ferait partie de la cible à reconstruire)."""
 
    def __init__(self, aug, image_size, mean, std, num_views: int = 2):
        if num_views == 1:
            self.views = [_build_view(aug, image_size, mean, std, strong=False)]
        elif num_views == 2:
            self.views = [_build_view(aug, image_size, mean, std, aug.blur_probs[i], aug.solarize_probs[i])
                          for i in range(2)]
        else:
            raise ValueError("num_views doit valoir 1 ou 2")
 
    def __call__(self, img):
        return tuple(v(img) for v in self.views)