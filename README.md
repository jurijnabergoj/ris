# RIS 2026 Round 1 — Bacterial Colony Classification

Team: **Jur**

Task: 9-class classification of bacterial colony images (85 training, 26 test images).

## Method overview

- Two-phase transfer learning: EfficientNet-B2 and ConvNeXt-Tiny pretrained on ImageNet
- 10-fold stratified cross-validation → 20 models total
- Mixup augmentation with soft label targets during phase 2
- Test-time augmentation (TTA, n=8): random crops, flips, rotations averaged per image
- Class-imbalance handling: inverse-frequency weighted cross-entropy loss + label smoothing
- 20-model ensemble: predictions averaged across all folds and both architectures

## Project structure

```
ris/
├── configs/
│   ├── default.yaml       # EfficientNet-B2 config
│   ├── convnext.yaml      # ConvNeXt-Tiny config
│   └── utils.py
├── src/
│   ├── data/
│   │   ├── dataset.py     # RisDataset, TestDataset
│   │   ├── augmentation.py
│   │   └── transform.py
│   ├── models/
│   │   └── classifier.py  # RisClassifier (EfficientNet-B2 / ConvNeXt-Tiny)
│   ├── training/
│   │   └── trainer.py     # Two-phase training loop, Mixup
│   └── inference/
│       └── inference.py   # TTA inference, ensemble
├── scripts/
│   ├── train.py           # K-fold training
│   ├── predict.py         # Ensemble prediction → Jur.txt
│   └── crop_images.py     # One-time image preprocessing
├── slurm/                 # SLURM job scripts (ARNES HPC)
├── data/
│   ├── ris2026-krog1-ucni/    # Training images
│   ├── ris2026-krog1-testni/  # Test images
│   └── ucni_set.csv           # Training labels
├── checkpoints_mixed/     # Trained model weights
└── requirements.txt
```

## Environment setup

Python 3.10+ with CUDA 12.1 recommended. Install dependencies:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Or with conda:

```bash
conda create -n ris python=3.10
conda activate ris
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data preparation

Place the competition images in:
- `data/ris2026-krog1-ucni/` — training images
- `data/ris2026-krog1-testni/` — test images
- `data/ucni_set.csv` — training labels CSV

Run the preprocessing script once:

```bash
python scripts/crop_images.py
```

This crops 4 training images (`75c8bd04.png`, `3075a94c.png`, `8501bff5.png`, `2283929d.png`) and
1 test image (`b40ccdbd.png`) to their left half. These images contain two side-by-side petri
dishes; only the labelled dish is kept.

> **Note:** If using the provided `checkpoints_mixed/` weights, the training images are already
> in their cropped state — do not run `crop_images.py` again on already-cropped images.

## Reproducing predictions from pre-trained checkpoints

With the `checkpoints_mixed/` directory in place, run prediction directly:

```bash
python scripts/predict.py
```

Output is written to `Jur.txt`.

## Reproducing from scratch (full retraining)

Train EfficientNet-B2 (saves to `checkpoints_mixed/`):

```bash
python scripts/train.py --config configs/default.yaml
```

Train ConvNeXt-Tiny:

```bash
python scripts/train.py --config configs/convnext.yaml
```

Then predict:

```bash
python scripts/predict.py
```

On ARNES HPC with SLURM, use the provided job scripts in `slurm/`. For example:

```bash
sbatch slurm/train.slurm
```

## Reproducibility notes

- Random seed is fixed (`random_state=42`) for the StratifiedKFold split
- PyTorch/CUDA operations are not fully deterministic by default which means minor numerical differences between runs are expected
- Pretrained weights (EfficientNet-B2, ConvNeXt-Tiny) are downloaded from torchvision on first
  run and cached in `~/.cache/torch/`
