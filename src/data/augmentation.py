from torchvision import transforms


class DataAugmentation:
    def __init__(self, augment_cfg):
        # Transforms operate on PIL Images
        self.augmentation = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=augment_cfg["hf_prob"]),
                transforms.RandomVerticalFlip(p=augment_cfg["vf_prob"]),
                transforms.RandomRotation(degrees=augment_cfg["rot_deg"]),
                transforms.RandomAffine(
                    degrees=augment_cfg["raf_degrees"],
                    scale=(augment_cfg["raf_scale_x"], augment_cfg["raf_scale_y"]),
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(augment_cfg["gb_kernel_size"])],
                    p=augment_cfg["gb_p"],
                ),
            ]
        )

    def __call__(self, image):
        return self.augmentation(image)
