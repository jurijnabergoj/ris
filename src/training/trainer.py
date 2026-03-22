import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def _mixup_batch(images, labels, num_classes, label_smoothing, alpha=0.4):
    """Apply Mixup to a batch. Returns mixed images and soft label targets."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    lam = max(lam, 1.0 - lam)  # keep lam >= 0.5 so the primary label dominates
    perm = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[perm]

    eps = label_smoothing
    y_a = F.one_hot(labels, num_classes).float()
    y_b = F.one_hot(labels[perm], num_classes).float()

    soft = lam * y_a + (1.0 - lam) * y_b
    soft = soft * (1.0 - eps) + eps / num_classes
    return mixed, soft


def _soft_cross_entropy(logits, soft_targets):
    """Cross-entropy loss for soft (non-one-hot) label targets."""
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def _run_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    training: bool,
    num_classes: int = 0,
    label_smoothing: float = 0.0,
    mixup: bool = False,
):
    """One full pass over a dataloader. Returns (avg_loss, accuracy)."""
    model.train() if training else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.no_grad() if not training else torch.enable_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if training and mixup and images.size(0) > 1:
                mixed, soft_targets = _mixup_batch(
                    images, labels, num_classes, label_smoothing
                )
                logits = model(mixed)
                loss = _soft_cross_entropy(logits, soft_targets)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
            else:
                logits = model(images)
                loss = loss_fn(logits, labels)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def _train_phase(
    tag,
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience,
    checkpoint_path,
    initial_best_val_acc=0.0,
    num_classes=0,
    label_smoothing=0.0,
    mixup=False,
):
    """Training loop for one phase. Saves best checkpoint."""
    best_val_acc = initial_best_val_acc
    patience_counter = 0
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        train_loss, train_acc = _run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            training=True,
            num_classes=num_classes,
            label_smoothing=label_smoothing,
            mixup=mixup,
        )
        val_loss, val_acc = _run_epoch(
            model,
            val_loader,
            loss_fn,
            None,
            device,
            training=False,
        )
        scheduler.step()

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        print(
            f"[{tag}] Epoch {epoch + 1}/{num_epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Best val acc {best_val_acc:.3f} — checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping triggered at epoch {epoch + 1}")
                break

    return results


def train(
    cfg, model, train_loader, val_loader, class_weights=None, checkpoint_path=None
):
    """
    Two-phase training for one fold:
      Phase 1 — freeze backbone, train head only (higher LR, few epochs)
      Phase 2 — unfreeze all, fine-tune entire network (lower LR, more epochs)

    Returns best validation accuracy achieved across both phases.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on {device}")

    label_smoothing = cfg["train"].get("label_smoothing", 0.1)
    weight = class_weights.to(device) if class_weights is not None else None
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    checkpoint_path = checkpoint_path or cfg["train"]["checkpoint_path"]
    mixup = cfg["train"].get("mixup", True)
    patience = cfg["train"].get("patience", 10)
    weight_decay = cfg["train"].get("weight_decay", 1e-4)
    phase1_lr = cfg["train"].get("phase1_lr", 1e-3)
    phase1_epochs = cfg["train"].get("phase1_epochs", 10)
    phase2_lr = cfg["train"].get("phase2_lr", 1e-4)
    phase2_epochs = cfg["train"].get("phase2_epochs", 20)

    num_classes = model.num_classes

    # --- Phase 1: head only ---
    print("\n=== Phase 1: Training head only ===")
    model.freeze_backbone()
    optimizer1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=phase1_lr,
        weight_decay=weight_decay,
    )
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, phase1_epochs)

    phase1_results = _train_phase(
        "phase1",
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer1,
        scheduler1,
        device,
        phase1_epochs,
        patience,
        checkpoint_path,
        mixup=False,
    )

    # Start phase 2 from the best phase 1 checkpoint
    phase1_best_val_acc = (
        max(phase1_results["val_acc"]) if phase1_results["val_acc"] else 0.0
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )

    # --- Phase 2: full fine-tune with Mixup ---
    print("\n=== Phase 2: Fine-tuning entire network ===")
    model.unfreeze()
    optimizer2 = torch.optim.AdamW(
        model.parameters(),
        lr=phase2_lr,
        weight_decay=weight_decay,
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, phase2_epochs)

    phase2_results = _train_phase(
        "phase2",
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer2,
        scheduler2,
        device,
        phase2_epochs,
        patience,
        checkpoint_path,
        initial_best_val_acc=phase1_best_val_acc,
        num_classes=num_classes,
        label_smoothing=label_smoothing,
        mixup=mixup,
    )

    best_val_acc = (
        max(phase2_results["val_acc"])
        if phase2_results["val_acc"]
        else phase1_best_val_acc
    )
    print(f"\nTraining complete. Best val acc: {best_val_acc:.3f}")
    print(f"Checkpoint: {checkpoint_path}")

    return best_val_acc
