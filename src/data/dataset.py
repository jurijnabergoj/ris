from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image


class RisDataset(Dataset):
    def __init__(self, data_dir, csv_path, transform=None, augment=None, files=None):
        """
        Args:
            data_dir: directory containing image files
            csv_path: path to CSV with columns IME_SLIKE, OZNAKA
            transform: DataTransform instance (Resize → ToTensor → Normalize)
            augment:   DataAugmentation instance (PIL → PIL); pass None for validation
            files:     optional explicit list of filenames to use (for train/val splits)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment

        df = pd.read_csv(csv_path)
        self.filename_to_label = dict(zip(df["IME_SLIKE"], df["OZNAKA"]))

        self.classes = sorted(df["OZNAKA"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        labeled_filenames = set(self.filename_to_label.keys())
        if files is not None:
            self.images = sorted(f for f in files if f in labeled_filenames)
        else:
            self.images = sorted(f for f in os.listdir(data_dir) if f in labeled_filenames)
        self.labels = [self.class_to_idx[self.filename_to_label[f]] for f in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename = self.images[index]
        image = Image.open(os.path.join(self.data_dir, filename)).convert("RGB")
        label = self.class_to_idx[self.filename_to_label[filename]]

        # Augmentation runs on PIL Image, before ToTensor
        if self.augment:
            image = self.augment(image)
        # Transform: Resize → ToTensor → Normalize
        if self.transform:
            image = self.transform(image)

        return image, label
    

class TestDataset(Dataset):
    """Returns raw PIL images so augment → transform runs in the correct order during TTA."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = sorted(f for f in os.listdir(data_dir) if f.endswith(".png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename = self.images[index]
        image = Image.open(os.path.join(self.data_dir, filename)).convert("RGB")
        return image, filename