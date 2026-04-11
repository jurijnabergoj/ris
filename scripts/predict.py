from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.utils import load_config, resolve_cfg_paths
from src.data.dataset import TestDataset, RisDataset
from src.models.classifier import make_model
from src.inference.inference import inference, load_fold_models, pil_collate

PROJECT_ROOT = Path(__file__).parent.parent


def load_data(cfg):
    d = cfg["data"]
    test_features_path = d.get("test_features_path", None)
    train_features_path = d.get("features_path", None)

    # Load training dataset to get the idx → label string mapping and norm stats
    base_dataset = RisDataset(
        d["data_dir"], d["csv_path"], features_path=train_features_path
    )
    idx_to_class = {v: k for k, v in base_dataset.class_to_idx.items()}
    num_classes = len(base_dataset.classes)

    # Pass train norm_stats to test so features are on the same scale the model was trained with
    test_dataset = TestDataset(
        d["test_dir"],
        features_path=test_features_path,
        norm_stats=base_dataset.norm_stats,
    )
    # num_workers=0 because PIL images can't be shared across worker processes
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=pil_collate
    )

    return test_loader, idx_to_class, num_classes


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = resolve_cfg_paths(
        load_config(PROJECT_ROOT / "configs" / "efficient_net_b2.yaml")
    )
    test_loader, idx_to_class, num_classes = load_data(config)

    arch_configs = [
        PROJECT_ROOT / "configs" / "efficient_net_b2.yaml",  # EfficientNet-B2
        # PROJECT_ROOT / "configs" / "efficient_net_b4.yaml",  # EfficientNet-B4
        PROJECT_ROOT / "configs" / "convnext_tiny.yaml",  # ConvNeXt-Tiny
        # PROJECT_ROOT / "configs" / "vit_b_16.yaml",  # ViT-b-16
    ]
    submission_path = PROJECT_ROOT / "Jur.txt"

    all_fold_models = []
    for config_path in arch_configs:
        cfg = resolve_cfg_paths(load_config(config_path))
        model = make_model(cfg, num_classes=num_classes)
        checkpoint_base = cfg["train"]["checkpoint_path"]
        fold1_path = checkpoint_base.replace(".pth", "_fold1.pth")

        if not Path(fold1_path).exists():
            print(
                f"Skipping {cfg['model']['arch']}: no checkpoints found at {fold1_path}"
            )
            continue
        val_acc_threshold = cfg.get("predict", {}).get("val_acc_threshold", 0.0)
        print(f"\nLoading {cfg['model']['arch']} checkpoints (threshold={val_acc_threshold:.2f})...")
        all_fold_models.extend(load_fold_models(cfg, model, device, val_acc_threshold=val_acc_threshold))

    print(f"\nEnsemble size: {len(all_fold_models)} models total")
    predictions = inference(
        config, None, test_loader, idx_to_class, fold_models=all_fold_models
    )

    with open(submission_path, "w") as f:
        f.write("IME_SLIKE,OZNAKA\n")
        for pred in predictions:
            f.write(f"{pred['image_filename']},{pred['predicted_label']}\n")

    print(f"Wrote {len(predictions)} predictions to {submission_path}")
