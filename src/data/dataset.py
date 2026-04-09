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


def _load_features(
    features_path, norm_stats: dict | None = None
) -> tuple[dict[str, torch.Tensor], dict]:
    """Load colony features and normalize.

    If norm_stats is None, computes mean/std from this CSV (use for training data).
    If norm_stats is provided, applies those stats (use for test data so scale matches training).
    Returns (feats_dict, norm_stats).
    """
    df = pd.read_csv(features_path)
    feat_df = df[FEATURE_COLS].astype(np.float32)

    if norm_stats is None:
        mean = feat_df.mean()
        std = feat_df.std().replace(0, 1)
        norm_stats = {"mean": mean, "std": std}
    else:
        mean = norm_stats["mean"]
        std = norm_stats["std"]

    feat_df = (feat_df - mean) / std
    feat_df = feat_df.clip(-3.0, 3.0)  # cap extreme OOD values (e.g. late-stage colonies)

    feats_dict = {
        row["filename"]: torch.tensor(feat_df.loc[i].values, dtype=torch.float32)
        for i, row in df.iterrows()
    }
    return feats_dict, norm_stats


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

        if features_path:
            self.feats_dict, self.norm_stats = _load_features(features_path)
        else:
            self.feats_dict, self.norm_stats = {}, None

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

    def __init__(self, data_dir, features_path=None, norm_stats=None):
        self.data_dir = data_dir
        self.images = sorted(f for f in os.listdir(data_dir) if f.endswith(".png"))
        if features_path:
            self.feats_dict, _ = _load_features(features_path, norm_stats=norm_stats)
        else:
            self.feats_dict = {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename = self.images[index]
        image = Image.open(os.path.join(self.data_dir, filename)).convert("RGB")
        feats = self.feats_dict.get(filename, torch.zeros(0, dtype=torch.float32))
        return image, feats, filename
