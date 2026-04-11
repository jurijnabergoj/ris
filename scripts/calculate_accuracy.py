import argparse
import os
from datetime import datetime

import pandas as pd


def calculate_accuracy(predictions_path, ground_truth_path):
    preds = pd.read_csv(predictions_path)
    gt = pd.read_csv(ground_truth_path)

    merged = preds.merge(gt, on="IME_SLIKE", suffixes=("_pred", "_true"))
    if len(merged) != len(gt):
        missing = set(gt["IME_SLIKE"]) - set(preds["IME_SLIKE"])
        print(f"WARNING: {len(missing)} test images missing from predictions: {missing}")

    correct = (merged["OZNAKA_pred"] == merged["OZNAKA_true"]).sum()
    total = len(merged)
    accuracy = correct / total

    wrong = merged[merged["OZNAKA_pred"] != merged["OZNAKA_true"]][
        ["IME_SLIKE", "OZNAKA_true", "OZNAKA_pred"]
    ]

    return accuracy, correct, total, wrong


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="Jur.txt")
    parser.add_argument("--ground-truth", default="data/testni_set.csv")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--log", default="logs/pipeline/results.log")
    args = parser.parse_args()

    accuracy, correct, total, wrong = calculate_accuracy(
        args.predictions, args.ground_truth
    )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*50}")
    print(f"Run: {run_id}")
    print(f"Accuracy: {accuracy:.4f}  ({correct}/{total} correct)")
    print(f"{'='*50}")
    print(f"\nMisclassified ({total - correct}):")
    for _, row in wrong.iterrows():
        print(f"  {row['IME_SLIKE']}: true={row['OZNAKA_true']} pred={row['OZNAKA_pred']}")

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "a") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Run:      {run_id}\n")
        f.write(f"Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Accuracy: {accuracy:.4f}  ({correct}/{total} correct)\n")
        f.write(f"Misclassified ({total - correct}):\n")
        for _, row in wrong.iterrows():
            f.write(f"  {row['IME_SLIKE']}: {row['OZNAKA_true']} -> {row['OZNAKA_pred']}\n")

    print(f"\nAppended to {args.log}")


if __name__ == "__main__":
    main()
