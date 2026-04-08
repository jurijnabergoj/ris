import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

FEATURE_COLS = [
    "blob_count",
    "mean_area",
    "std_area",
    "max_area",
    "min_area",
    "total_area",
]


def _load_features(features_path) -> dict[str, torch.Tensor]:
    """Load colony features csv -> {filename: feature_tensor}."""
    df = pd.read_csv(features_path)
    return {
        row["filename"]: torch.tensor(
            row[FEATURE_COLS].values.astype(np.float32), dtype=torch.float32
        )
        for _, row in df.iterrows()
    }


class RisDataset(Dataset):
    """Dataset for training/validation. Returns transformed tensors ready for model input."""

    def __init__(
        self,
        data_dir,
        csv_path,
        transform=None,
        augment=None,
        files=None,
        features_path=None,
    ):
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
            self.images = sorted(
                f for f in os.listdir(data_dir) if f in labeled_filenames
            )
        self.labels = [
            self.class_to_idx[self.filename_to_label[f]] for f in self.images
        ]

        self.feats_dict = _load_features(features_path) if features_path else {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename = self.images[index]
        image = Image.open(os.path.join(self.data_dir, filename)).convert("RGB")
        label = self.labels[index]

        if self.augment:  # PIL -> PIL
            image = self.augment(image)
        if self.transform:  # PIL -> tensor
            image = self.transform(image)

        if self.feats_dict:
            feats = self.feats_dict[filename]
        else:
            feats = torch.zeros(0, dtype=torch.float32)

        return image, feats, label


class TestDataset(Dataset):
    """Returns raw PIL images so augment → transform runs in the correct order during TTA."""

    def __init__(self, data_dir, features_path=None):
        self.data_dir = data_dir
        self.images = sorted(f for f in os.listdir(data_dir) if f.endswith(".png"))
        self.feats_dict = _load_features(features_path) if features_path else {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename = self.images[index]
        image = Image.open(os.path.join(self.data_dir, filename)).convert("RGB")
        feats = self.feats_dict.get(filename, torch.zeros(0, dtype=torch.float32))
        return image, feats, filename
