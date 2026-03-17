from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.utils import load_config
from src.data.dataset import TestDataset, RisDataset
from src.models.classifier import make_model
from src.inference.inference import inference, pil_collate

PROJECT_ROOT = Path(__file__).parent.parent


def prepare_data(cfg):
    d = cfg["data"]

    # Load training dataset only to get the class mapping (idx → label string)
    base_dataset = RisDataset(d["data_dir"], d["csv_path"])
    idx_to_class = {v: k for k, v in base_dataset.class_to_idx.items()}
    num_classes = len(base_dataset.classes)

    test_dataset = TestDataset(d["test_dir"])
    # num_workers=0: PIL images can't be shared across worker processes
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=pil_collate)

    return test_loader, idx_to_class, num_classes


if __name__ == "__main__":
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")

    test_loader, idx_to_class, num_classes = prepare_data(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config, num_classes=num_classes)
    model.load_state_dict(torch.load(config["train"]["checkpoint_path"], map_location=device))
    model.eval()

    predictions = inference(config, model, test_loader, idx_to_class)

    submission_path = PROJECT_ROOT / "TeamName.csv"
    with open(submission_path, "w") as f:
        f.write("IMAGE NAME,LABEL\n")
        for pred in predictions:
            f.write(f"{pred['image_filename']},{pred['predicted_label']}\n")

    print(f"Wrote {len(predictions)} predictions to {submission_path}")
