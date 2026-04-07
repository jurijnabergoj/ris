import argparse
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from configs.utils import load_config, resolve_cfg_paths
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = resolve_cfg_paths(load_config(PROJECT_ROOT / args.config))

    data_dir = config["data"]["data_dir"]
    csv_path = config["data"]["csv_path"]
    checkpoint_base = config["train"]["checkpoint_path"]
    batch_size = config["data"]["batch_size"]

    num_workers = config["data"].get("num_workers", 0)
    n_splits = config["train"].get("n_splits", 5)

    os.makedirs(os.path.dirname(checkpoint_base), exist_ok=True)

    transform = DataTransform(
        config["data"]["transform"]["height"], config["data"]["transform"]["width"]
    )
    augment = DataAugmentation(config["data"]["augment"])

    base_dataset = RisDataset(data_dir, csv_path)
    num_classes = len(base_dataset.classes)
    class_weights = compute_class_weights(base_dataset)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_val_accs = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(base_dataset.images, base_dataset.labels)
    ):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{n_splits}")
        print(f"{'='*50}")

        train_files = [base_dataset.images[i] for i in train_idx]
        val_files = [base_dataset.images[i] for i in val_idx]

        train_dataset = RisDataset(
            data_dir, csv_path, transform=transform, augment=augment, files=train_files
        )
        val_dataset = RisDataset(
            data_dir, csv_path, transform=transform, augment=None, files=val_files
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Fresh model for each fold so we dont carry weights from the previous fold
        model = make_model(config, num_classes=num_classes)

        fold_checkpoint = checkpoint_base.replace(".pth", f"_fold{fold + 1}.pth")
        best_val_acc = train(
            config,
            model,
            train_loader,
            val_loader,
            class_weights=class_weights,
            checkpoint_path=fold_checkpoint,
        )
        fold_val_accs.append(best_val_acc)

    fold_val_accs = np.array(fold_val_accs)
    print(f"\n{'='*50}")
    print(f"K-FOLD CROSS-VALIDATION RESULTS ({n_splits} folds)")
    print(f"{'='*50}")
    for i, acc in enumerate(fold_val_accs):
        print(f"  Fold {i + 1}: {acc:.4f}")
    print(f"  Mean:  {fold_val_accs.mean():.4f}")
    print(f"  Std:   {fold_val_accs.std():.4f}")
    print(f"\n  Accuracy:  {fold_val_accs.mean():.4f} +- {fold_val_accs.std():.4f}")
