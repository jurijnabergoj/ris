from torchvision import transforms


class DataAugmentation:
    def __init__(self, hf_prob=0.5, vf_prob=0.5):
        # All transforms operate on PIL Images — ToTensor/Normalize happen in DataTransform
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=hf_prob),
            transforms.RandomVerticalFlip(p=vf_prob),
            transforms.RandomRotation(degrees=360),  # Petri dishes have full rotational symmetry
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        ])

    def __call__(self, image):
        return self.augmentation(image)