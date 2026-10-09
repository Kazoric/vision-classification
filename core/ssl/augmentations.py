from torchvision import transforms as T


def _hw(image_size):
    return (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)


def _build_view(aug, image_size, mean, std, blur_p: float, solarize_p: float) -> T.Compose:
    h, w = _hw(image_size)
    ops = [
        T.RandomResizedCrop((h, w), scale=tuple(aug.crop_scale),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(aug.flip_prob),
        T.RandomApply([T.ColorJitter(*aug.color_jitter)], p=aug.color_jitter_prob),
        T.RandomGrayscale(aug.grayscale_prob),
    ]
    if blur_p > 0:
        k = max(3, int(0.1 * min(h, w)) // 2 * 2 + 1)
        ops.append(T.RandomApply([T.GaussianBlur(k, sigma=(0.1, 2.0))], p=blur_p))
    if solarize_p > 0:
        ops.append(T.RandomSolarize(threshold=128, p=solarize_p))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


class TwoViewTransform:
    def __init__(self, aug, image_size, mean, std):
        self.view1 = _build_view(aug, image_size, mean, std, aug.blur_probs[0], aug.solarize_probs[0])
        self.view2 = _build_view(aug, image_size, mean, std, aug.blur_probs[1], aug.solarize_probs[1])

    def __call__(self, img):
        return self.view1(img), self.view2(img)