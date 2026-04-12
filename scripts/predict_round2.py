import argparse
import pickle
from pathlib import Path
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

"""
This script runs inference for round 2 test set.

Usage:
    python scripts/predict_round2.py \
        --test_dir /path/to/test/images \
        --model_dir models/round2 \
        --output Jur.txt

Requires models saved from hsi_eda.ipynb (Final Model cell):
    models/round2/xgb_models.pkl
    models/round2/lgbm_models.pkl
    models/round2/svm_final.pkl
    models/round2/mlp_final.pkl
    models/round2/label_encoder.pkl
    models/round2/config.pkl
"""


def extract_record(fpath: Path) -> dict:
    arr = np.load(fpath)
    H, W, C = arr.shape
    mask = arr[..., 0] != -1
    colony_px = int(mask.sum())

    if colony_px == 0:
        mean_spec = np.zeros(C)
        std_spec = np.zeros(C)
    else:
        raw = arr[mask].astype(float)
        pct = np.percentile(raw[raw >= 0], 99)
        if pct < 1e-8:
            pct = 1.0
        std_spec = raw.std(axis=0)
        colony_vals = np.clip(raw, 0, pct)
        mean_spec = colony_vals.mean(axis=0) / pct

    return {
        "filename": fpath.name,
        "H": H,
        "W": W,
        "colony_px": colony_px,
        "fill_frac": colony_px / (H * W),
        "mean_spec": mean_spec,
        "std_spec": std_spec,
    }


def build_features(records):
    rich, shape = [], []
    for r in records:
        ms = r["mean_spec"]
        ss = r["std_spec"]
        spatial = np.array(
            [
                np.log1p(r["colony_px"]),
                r["fill_frac"],
                r["H"] / (r["W"] + 1e-8),
                np.log1p(r["H"] * r["W"]),
            ]
        )

        # rich features
        g1 = np.gradient(ms)
        g2 = np.gradient(g1)
        rich.append(np.concatenate([ms, ss, g1, g2, spatial]))

        # shape-normalised features
        norm = np.linalg.norm(ms) + 1e-8
        ms_s = ms / norm
        ss_s = ss / norm
        g1_s = np.gradient(ms_s)
        g2_s = np.gradient(g1_s)
        level = np.array([norm, ms.mean(), ms.max()])
        shape.append(np.concatenate([ms_s, ss_s, g1_s, g2_s, level, spatial]))

    return np.hstack([np.array(rich), np.array(shape)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--model_dir", default="models/round2")
    parser.add_argument("--output", default="Jur.txt")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    model_dir = Path(args.model_dir)

    with open(model_dir / "xgb_models.pkl", "rb") as f:
        xgb_models = pickle.load(f)
    with open(model_dir / "lgbm_models.pkl", "rb") as f:
        lgbm_models = pickle.load(f)
    with open(model_dir / "svm_final.pkl", "rb") as f:
        svm = pickle.load(f)
    with open(model_dir / "mlp_final.pkl", "rb") as f:
        mlp = pickle.load(f)
    with open(model_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open(model_dir / "config.pkl", "rb") as f:
        cfg = pickle.load(f)

    weights = cfg["weights"]
    lgbm_class_order = cfg.get("lgbm_class_order")
    label_order = le.classes_

    test_files = sorted(f for f in test_dir.glob("*.npy") if f.name != "lam.npy")

    records = []
    for fpath in test_files:
        r = extract_record(fpath)
        records.append(r)

    X_test = build_features(records)

    def reorder(proba, from_classes, to_classes):
        idx = [list(from_classes).index(c) for c in to_classes]
        return proba[:, idx]

    xgb_proba = np.mean([m.predict_proba(X_test) for m in xgb_models], axis=0)

    lgbm_raw = np.mean([m.predict_proba(X_test) for m in lgbm_models], axis=0)
    lgbm_proba = (
        reorder(lgbm_raw, lgbm_class_order, label_order)
        if lgbm_class_order is not None
        else lgbm_raw
    )

    svm_proba = reorder(svm.predict_proba(X_test), svm.classes_, label_order)
    mlp_proba = reorder(mlp.predict_proba(X_test), mlp.classes_, label_order)

    proba_ens = (
        weights["xgb"] * xgb_proba
        + weights["lgbm"] * lgbm_proba
        + weights["svm"] * svm_proba
        + weights["mlp"] * mlp_proba
    )
    y_pred = label_order[proba_ens.argmax(axis=1)]

    with open(args.output, "w") as f:
        for r, label in zip(records, y_pred):
            f.write(f"{r['filename']},{label}\n")


if __name__ == "__main__":
    main()
