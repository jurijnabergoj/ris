# RIS 2026 Bacterial Colony Hyperspectral Classification

Team: **Jur**

---

## Round 2

Task: 8-class classification of hyperspectral bacterial colony images (659 training, 167 test).  
Input: 3D `.npy` arrays `(H, W, 184)`, wavelengths 452–949 nm, background pixels = -1.

### Method

- Per-image 99th-percentile normalization to handle saturated sensor pixels
- Feature engineering: mean spectrum + std spectrum + gradients + L2-shape-normalized variants (1483-dim richshape features)
- 4-model soft ensemble: 5-seed XGB bag + 5-seed LightGBM bag + linear SVM + MLP
- Sample weighting: 49 consistently-misclassified images down-weighted (possibly label noise)
- 5-fold stratified CV accuracy: **92.1%**

### Environment

```bash
conda activate ris
# dependencies: numpy, scikit-learn, xgboost, lightgbm, scipy, pandas
```

### Run inference

```bash
python scripts/predict_round2.py \
    --test_dir /d/hpc/projects/FRI/jn16867/ris/ris2026_krog2_testni \
    --model_dir models/round2 \
    --output Jur.txt
```

Requires saved models in `models/round2/`:
```
xgb_models.pkl
lgbm_models.pkl
svm_final.pkl
mlp_final.pkl
label_encoder.pkl
config.pkl
```

### Retrain from scratch

Open `notebooks/round2_eda.ipynb` and run all cells in order. The final cell trains all models and saves them to `models/round2/`. Configured with:


### Project structure (Round 2)

```
ris/
├── notebooks/
│   └── round2_eda.ipynb       # EDA, feature engineering, model development
├── scripts/
│   └── predict_round2.py      # Inference script
├── models/
│   └── round2/                # Saved model files
└── README.md
```

---

## Round 1

Task: 9-class classification of bacterial colony RGB images (85 training, 26 test).

### Method

- Two-phase transfer learning: EfficientNet-B2 and ConvNeXt-Tiny pretrained on ImageNet
- 10-fold stratified cross-validation → 20 models total
- Mixup augmentation with soft label targets during phase 2
- Test-time augmentation (TTA, n=8): random crops, flips, rotations averaged per image
- Class-imbalance handling: inverse-frequency weighted cross-entropy loss + label smoothing
- 20-model ensemble: predictions averaged across all folds and both architectures

### Run inference (Round 1)

With pre-trained checkpoints:

```bash
python scripts/predict.py
```

Retrain from scratch:

```bash
python scripts/train.py --config configs/default.yaml   # EfficientNet-B2
python scripts/train.py --config configs/convnext.yaml  # ConvNeXt-Tiny
python scripts/predict.py
```

### Project structure (Round 1)

```
ris/
├── configs/
│   ├── default.yaml       # EfficientNet-B2 config
│   └── convnext.yaml      # ConvNeXt-Tiny config
├── src/
│   ├── data/              # dataset.py, augmentation.py, transform.py
│   ├── models/            # classifier.py
│   ├── training/          # trainer.py
│   └── inference/         # inference.py
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── crop_images.py     # one-time preprocessing (do not re-run on already-cropped images)
├── slurm/                 # ARNES HPC job scripts
├── data/
│   ├── ris2026-krog1-ucni/
│   ├── ris2026-krog1-testni/
│   └── ucni_set.csv
├── checkpoints/
└── requirements.txt
```

### Environment setup (Round 1)

```bash
conda create -n ris python=3.10
conda activate ris
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
