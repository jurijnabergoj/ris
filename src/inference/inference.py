import copy
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.augmentation import DataAugmentation
from src.data.transform import DataTransform


def pil_collate(batch):
    """Custom collate that keeps PIL images as a list instead of stacking into a tensor."""
    images, filenames = zip(*batch)
    return list(images), list(filenames)


def _run_pass(n_tta, pil_image, model, softmax_fn, device, augment, transform):
    """Run n_tta augmented forward passes over a single PIL image."""
    probs = []
    for _ in range(n_tta):
        tensor = transform(augment(pil_image)).unsqueeze(0).to(device)
        probs.append(softmax_fn(model(tensor)))
    return probs


def _test_phase(
    n_tta,
    fold_models,
    test_loader,
    softmax_fn,
    device,
    augment,
    transform,
    class_mappings,
):
    predictions = []

    with torch.no_grad():
        for images, filenames in tqdm(test_loader, leave=False):
            pil_image = images[0]
            filename = filenames[0]

            fold_probs = []
            for m in fold_models:
                probs = _run_pass(
                    n_tta, pil_image, m, softmax_fn, device, augment, transform
                )
                # Average over TTA passes for fold: [n_tta, 1, C] → [1, C]
                fold_probs.append(torch.stack(probs, dim=0).mean(dim=0))

            # Average over folds: [n_folds, 1, C] → [1, C]
            avg_prob = torch.stack(fold_probs, dim=0).mean(dim=0)
            predicted_label = class_mappings[avg_prob.argmax(dim=1).item()]
            predictions.append(
                {"image_filename": filename, "predicted_label": predicted_label}
            )

            print(
                f"Predicted: {predicted_label} for {filename}, probability: {avg_prob.max().item():.4f}"
            )

    return predictions


def load_fold_models(cfg, model, device):
    """Load all fold checkpoints. Returns a list of eval-mode models."""
    n_splits = cfg["train"].get("n_splits", 5)
    checkpoint_base = cfg["train"]["checkpoint_path"]
    fold_models = []

    for fold in range(n_splits):
        fold_path = checkpoint_base.replace(".pth", f"_fold{fold + 1}.pth")
        m = copy.deepcopy(model)
        m.load_state_dict(torch.load(fold_path, map_location=device, weights_only=True))
        m.to(device).eval()
        fold_models.append(m)
        print(f"Loaded fold {fold + 1} checkpoint: {fold_path}")
    return fold_models


def inference(cfg, model, test_loader, class_mappings, fold_models=None):
    """
    Run inference. If fold_models is provided, use them directly (multi-arch ensemble).
    Otherwise load fold checkpoints from cfg.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_tta = cfg["test"]["n_tta"]
    augment = DataAugmentation(
        hf_prob=cfg["data"]["augment"]["hf_prob"],
        vf_prob=cfg["data"]["augment"]["vf_prob"],
    )
    transform = DataTransform(
        cfg["data"]["transform"]["height"], cfg["data"]["transform"]["width"]
    )
    softmax_fn = nn.Softmax(dim=1)

    if fold_models is None:
        fold_models = load_fold_models(cfg, model, device)

    return _test_phase(
        n_tta,
        fold_models,
        test_loader,
        softmax_fn,
        device,
        augment,
        transform,
        class_mappings,
    )
