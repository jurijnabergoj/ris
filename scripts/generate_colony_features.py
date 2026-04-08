import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/d/hpc/home/jn16867/cso/ext/GECO2")

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from sam2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2.build_sam import build_sam2
import sam2.sam2 as _sam2_pkg

PROJECT_ROOT = Path(__file__).parent.parent
FEATURE_KEYS = [
    "blob_count",
    "mean_area",
    "std_area",
    "max_area",
    "min_area",
    "total_area",
]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def build_mask_generator(
    sam_checkpoint: Path, device: str
) -> SAM2AutomaticMaskGenerator:
    sam2_configs_dir = str(
        Path(_sam2_pkg.__file__).resolve().parent.parent / "sam2_configs"
    )
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=sam2_configs_dir, job_name="sam2")
    model = build_sam2(
        "sam2_hiera_b+.yaml", ckpt_path=str(sam_checkpoint), device=device
    )
    return SAM2AutomaticMaskGenerator(
        model,
        points_per_side=64,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.90,
        min_mask_region_area=10,
    )


def generate_features(
    img_path: Path, mask_generator: SAM2AutomaticMaskGenerator
) -> dict:
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    dish_area = img_rgb.shape[0] * img_rgb.shape[1]

    masks = mask_generator.generate(img_rgb)
    colonies = [m for m in masks if 0.0000001 < m["area"] / dish_area < 0.01]
    areas = np.array([m["area"] for m in colonies], dtype=float)

    if len(areas) == 0:
        return {k: 0.0 for k in FEATURE_KEYS}

    return {
        "blob_count": len(colonies),
        "mean_area": float(areas.mean()),
        "std_area": float(areas.std()),
        "max_area": float(areas.max()),
        "min_area": float(areas.min()),
        "total_area": float(areas.sum()),
    }


def display_detected_blobs(img_path: Path, mask_generator: SAM2AutomaticMaskGenerator):
    from matplotlib import pyplot as plt

    img = Image.open(img_path).convert("RGB")
    img_rgb = np.array(img)
    dish_area = img_rgb.shape[0] * img_rgb.shape[1]

    masks = mask_generator.generate(img_rgb)
    colonies = [m for m in masks if 0.0000001 < m["area"] / dish_area < 0.01]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(img)
    overlay = np.zeros((*img_rgb.shape[:2], 4), dtype=float)
    colors = plt.cm.tab20.colors

    for i, m in enumerate(colonies):
        mask = m["segmentation"]
        color = colors[i % len(colors)]
        overlay[mask, :3] = color
        overlay[mask, 3] = 0.45
        rows_m, cols_m = np.where(mask)
        # rows = y axis, cols = x axis
        axes[1].plot(
            cols_m.mean(),
            rows_m.mean(),
            "x",
            color=color,
            markersize=6,
            markeredgewidth=1.5,
        )

    axes[1].imshow(overlay)
    axes[1].set_title(f"{len(colonies)} colonies — {img_path.name}")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SAM2 colony features for a dataset directory."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Directory of images (e.g. data/ris2026-krog1-ucni)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: data/<dataset_name>_colony_features.csv)",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=Path("/d/hpc/projects/FRI/jn16867/checkpoints/sam2_hiera_base_plus.pt"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset_dir = (
        args.dataset if args.dataset.is_absolute() else PROJECT_ROOT / args.dataset
    )
    output_path = (
        args.output or PROJECT_ROOT / "data" / f"{dataset_dir.name}_colony_features.csv"
    )

    print(f"Dataset:    {dataset_dir}")
    print(f"Output:     {output_path}")
    print(f"Checkpoint: {args.sam_checkpoint}")
    print(f"Device:     {args.device}\n")

    mask_generator = build_mask_generator(args.sam_checkpoint, args.device)

    img_paths = sorted(
        p for p in dataset_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    print(f"Found {len(img_paths)} images\n")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename"] + FEATURE_KEYS)
        writer.writeheader()

        for img_path in img_paths:
            features = generate_features(img_path, mask_generator)
            writer.writerow({"filename": img_path.name, **features})
            print(
                f"  {img_path.name}: {features['blob_count']} colonies, "
                f"mean_area={features['mean_area']:.0f}px"
            )

    print(f"\nDone. Written to {output_path}")
