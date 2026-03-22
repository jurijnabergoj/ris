from torchvision import transforms


class DataAugmentation:
    def __init__(self, hf_prob=0.5, vf_prob=0.5):
        # Transforms operate on PIL Images
        self.augmentation = transforms.Compose(
            [
                # transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=hf_prob),
                transforms.RandomVerticalFlip(p=vf_prob),
                transforms.RandomRotation(degrees=360),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
                ),
            ]
        )

    def __call__(self, image):
        return self.augmentation(image)
