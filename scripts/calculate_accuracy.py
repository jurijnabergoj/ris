import pandas as pd


def load_labels(path):
    df = pd.read_csv(path)
    df = df.sort_values(by=["IME_SLIKE"], ascending=False)
    labels = df["OZNAKA"].tolist()

    return labels


def get_accuracy(gt_labels, pred_labels):
    sum = 0
    total = len(gt_labels)

    for i in range(total):
        if gt_labels[i] == pred_labels[i]:
            sum = sum + 1

    return sum / total


gt_labels = load_labels("resitve_krog1_2026.csv")
pred_labels = load_labels("Jur.txt")

accuracy = get_accuracy(gt_labels, pred_labels)
print(f"Accuracy: {accuracy}")
