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
    """Run n_tta augmented forward passes over a single PIL image.

    Correct order: augment(PIL) → ToTensor → Normalize → model
    This ensures ColorJitter and RandomRotation operate on [0,1] pixel values.

    Returns:
        list of n_tta softmax tensors, each shape [1, num_classes]
    """
    probs = []
    for _ in range(n_tta):
        tensor = transform(augment(pil_image)).unsqueeze(0).to(device)  # [1, C, H, W]
        logits = model(tensor)
        probs.append(softmax_fn(logits))
    return probs


def _test_phase(n_tta, model, test_loader, softmax_fn, device, augment, transform, class_mappings):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, filenames in tqdm(test_loader, leave=False):
            # images is a list of PIL images (batch_size=1, so one image per iteration)
            pil_image = images[0]
            filename = filenames[0]

            probs = _run_pass(n_tta, pil_image, model, softmax_fn, device, augment, transform)
            probs_tensor = torch.stack(probs, dim=0)   # [n_tta, 1, num_classes]
            avg_prob = probs_tensor.mean(dim=0)         # [1, num_classes]
            predicted_index = avg_prob.argmax(dim=1)    # [1]
            predicted_label = class_mappings[predicted_index.item()]
            predictions.append({"image_filename": filename, "predicted_label": predicted_label})

    return predictions


def inference(cfg, model, test_loader, class_mappings):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    d = cfg["data"]
    augment = DataAugmentation(
        hf_prob=d["augment"]["hf_prob"],
        vf_prob=d["augment"]["vf_prob"],
    )
    transform = DataTransform(
        d["transform"]["height"],
        d["transform"]["width"],
    )
    softmax_fn = nn.Softmax(dim=1)
    n_tta = cfg["test"]["n_tta"]

    return _test_phase(n_tta, model, test_loader, softmax_fn, device, augment, transform, class_mappings)
