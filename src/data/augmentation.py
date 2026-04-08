from torchvision import transforms


class DataAugmentation:
    def __init__(self, augment_cfg):
        # Transforms operate on PIL Images
        self.augmentation = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=augment_cfg["hf_prob"]),
                transforms.RandomVerticalFlip(p=augment_cfg["vf_prob"]),
                transforms.RandomResizedCrop(
                    size=augment_cfg["rrc_size"],
                    scale=(augment_cfg["rrc_scale_min"], augment_cfg["rrc_scale_max"]),
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
