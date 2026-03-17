import pandas as pd


if __name__ == "__main__":
    print("This script is for checking the accuracy of your model on the test set. It should not be used for training or generating predictions for submission.")
    gt_df = pd.read_csv("/d/hpc/home/jn16867/ris/data/ucni_set.csv")    
    pred_df = pd.read_csv("/d/hpc/home/jn16867/ris/TeamName.csv")    
    filename_to_label = dict(zip(pred_df["IMAGE NAME"], pred_df["LABEL"]))
    correct = 0
    
    for filename, label in filename_to_label.items():
        gt_label = gt_df.loc[gt_df["IME_SLIKE"] == filename, "OZNAKA"].values[0]
        if label == gt_label:
            correct += 1
        else:
            print(f"Mismatch at {filename} predicted as {label}, but ground truth is {gt_label}.")

    accuracy = correct / len(filename_to_label)
    print(f"Accuracy: {accuracy:.4f}")