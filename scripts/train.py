import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from configs.utils import load_config
from src.data.augmentation import DataAugmentation
from src.data.dataset import RisDataset
from src.data.transform import DataTransform
from src.models.classifier import make_model
from src.training.trainer import train

PROJECT_ROOT = Path(__file__).parent.parent


def compute_class_weights(dataset: RisDataset) -> torch.Tensor:
    """Inverse-frequency weights to counter class imbalance."""
    counts = np.bincount(dataset.labels, minlength=len(dataset.classes))
    weights = 1.0 / counts.astype(float)
    weights /= weights.sum()
    return torch.tensor(weights, dtype=torch.float)


def prepare_data(cfg):
    d = cfg["data"]
    data_dir = d["data_dir"]
    csv_path = d["csv_path"]

    transform = DataTransform(d["transform"]["height"], d["transform"]["width"])
    augment = DataAugmentation(d["augment"]["hf_prob"], d["augment"]["vf_prob"])

    # Build a plain dataset (no augment) just to get the full file list + labels for splitting
    base_dataset = RisDataset(data_dir, csv_path)

    train_idx, val_idx = train_test_split(
        np.arange(len(base_dataset)),
        test_size=d["test_size"],
        random_state=42,
        shuffle=True,
        stratify=base_dataset.labels,
    )

    train_files = [base_dataset.images[i] for i in train_idx]
    val_files = [base_dataset.images[i] for i in val_idx]

    # Two separate dataset instances: train gets augmentation, val does not
    train_dataset = RisDataset(data_dir, csv_path, transform=transform, augment=augment, files=train_files)
    val_dataset = RisDataset(data_dir, csv_path, transform=transform, augment=None, files=val_files)

    num_workers = d.get("num_workers", 0)
    train_loader = DataLoader(train_dataset, batch_size=d["batch_size"], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=d["batch_size"], shuffle=False, num_workers=num_workers)

    class_weights = compute_class_weights(base_dataset)

    return train_loader, val_loader, class_weights, len(base_dataset.classes)


if __name__ == "__main__":
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")

    os.makedirs(os.path.dirname(config["train"]["checkpoint_path"]), exist_ok=True)

    train_loader, val_loader, class_weights, num_classes = prepare_data(config)

    model = make_model(config, num_classes=num_classes)
    train(config, model, train_loader, val_loader, class_weights=class_weights)
