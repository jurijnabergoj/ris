import copy
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.augmentation import DataAugmentation
from src.data.transform import DataTransform


def pil_collate(batch):
    """Custom collate for TestDataset which returns (pil_image, feats, filename)."""
    images, feats, filenames = zip(*batch)
    return list(images), torch.stack(feats), list(filenames)


def _run_pass(
    n_tta, pil_image, extra_features, model, softmax_fn, device, augment, transform
):
    """Run n_tta augmented forward passes over a single PIL image."""
    probs = []
    feats = (
        extra_features.unsqueeze(0).to(device) if extra_features.numel() > 0 else None
    )
    for _ in range(n_tta):
        tensor = transform(augment(pil_image)).unsqueeze(0).to(device)
        probs.append(softmax_fn(model(tensor, feats)))
    return probs


def _test_phase(n_tta, fold_models, test_loader, softmax_fn, device, class_mappings):
    predictions = []
    with torch.no_grad():
        for images, feats, filenames in tqdm(test_loader, leave=False):
            pil_image = images[0]
            extra_features = feats[0]  # (F,)
            filename = filenames[0]

            fold_probs = []
            for m, augment, transform in fold_models:
                probs = _run_pass(
                    n_tta,
                    pil_image,
                    extra_features,
                    m,
                    softmax_fn,
                    device,
                    augment,
                    transform,
                )
                fold_probs.append(torch.stack(probs, dim=0).mean(dim=0))

            avg_prob = torch.stack(fold_probs, dim=0).mean(dim=0)
            predicted_label = class_mappings[avg_prob.argmax(dim=1).item()]
            predictions.append(
                {"image_filename": filename, "predicted_label": predicted_label}
            )
            print(
                f"Predicted: {predicted_label} for {filename}, probability: {avg_prob.max().item():.4f}"
            )

    return predictions


def load_fold_models(cfg, model, device, val_acc_threshold=0.0):
    """Load fold checkpoints, optionally filtering by best val accuracy.

    Checkpoints may be saved as either a bare state_dict (old format) or a dict
    with keys 'state_dict' and 'best_val_acc' (new format). Both are handled.
    Folds whose best_val_acc < val_acc_threshold are skipped.
    """
    n_splits = cfg["train"].get("n_splits", 5)
    checkpoint_base = cfg["train"]["checkpoint_path"]
    augment = DataAugmentation(cfg["data"]["augment"])
    transform = DataTransform(
        cfg["data"]["transform"]["height"], cfg["data"]["transform"]["width"]
    )
    fold_models = []

    for fold in range(n_splits):
        fold_path = checkpoint_base.replace(".pth", f"_fold{fold + 1}.pth")
        ckpt = torch.load(fold_path, map_location=device, weights_only=True)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            best_val_acc = ckpt.get("best_val_acc", 1.0)
        else:
            state_dict = ckpt
            best_val_acc = 1.0  # old checkpoint — assume it passed (no info stored)

        if best_val_acc < val_acc_threshold:
            print(
                f"Skipping fold {fold + 1} (val_acc={best_val_acc:.3f} < threshold {val_acc_threshold:.3f}): {fold_path}"
            )
            continue

        m = copy.deepcopy(model)
        m.load_state_dict(state_dict)
        m.to(device).eval()
        fold_models.append((m, augment, transform))
        print(f"Loaded fold {fold + 1} checkpoint (val_acc={best_val_acc:.3f}): {fold_path}")
    return fold_models


def inference(cfg, model, test_loader, class_mappings, fold_models=None):
    """
    Run inference. If fold_models is provided, use them directly (multi-arch ensemble).
    Each entry in fold_models is a (model, augment, transform) tuple.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_tta = cfg["test"]["n_tta"]
    softmax_fn = nn.Softmax(dim=1)

    if fold_models is None:
        fold_models = load_fold_models(cfg, model, device)

    return _test_phase(
        n_tta, fold_models, test_loader, softmax_fn, device, class_mappings
    )
