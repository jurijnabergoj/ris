import random

from torchvision import transforms


class CutPasteColony:
    """Paste random crops from the image onto itself to simulate denser colony growth."""

    def __init__(self, n_patches=5, patch_scale=(0.03, 0.10), p=0.5):
        self.n_patches = n_patches
        self.patch_scale = patch_scale
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        result = img.copy()
        for _ in range(self.n_patches):
            scale = random.uniform(*self.patch_scale)
            pw = max(1, int(w * scale))
            ph = max(1, int(h * scale))
            sx = random.randint(0, w - pw)
            sy = random.randint(0, h - ph)
            patch = img.crop((sx, sy, sx + pw, sy + ph))
            tx = random.randint(0, w - pw)
            ty = random.randint(0, h - ph)
            result.paste(patch, (tx, ty))
        return result


class DataAugmentation:
    def __init__(self, augment_cfg):
        # Transforms operate on PIL Images
        self.augmentation = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=augment_cfg["hf_prob"]),
                transforms.RandomVerticalFlip(p=augment_cfg["vf_prob"]),
                transforms.RandomApply(
                    [CutPasteColony(
                        n_patches=augment_cfg.get("cpp_n_patches", 5),
                        patch_scale=(
                            augment_cfg.get("cpp_scale_min", 0.03),
                            augment_cfg.get("cpp_scale_max", 0.10),
                        ),
                        p=1.0,
                    )],
                    p=augment_cfg.get("cpp_p", 0.5),
                ),
                transforms.RandomRotation(degrees=augment_cfg["rot_deg"]),
                transforms.ColorJitter(
                    brightness=augment_cfg["cj_brightness"],
                    contrast=augment_cfg["cj_contrast"],
                    saturation=augment_cfg["cj_saturation"],
                    hue=augment_cfg["cj_hue"],
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(augment_cfg["gb_kernel_size"])],
                    p=augment_cfg["gb_p"],
                ),
            ]
        )

    def __call__(self, image):
        return self.augmentation(image)
