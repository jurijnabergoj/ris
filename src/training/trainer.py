import torch
import torch.nn as nn
from tqdm import tqdm


def _run_epoch(model, loader, loss_fn, optimizer, device, training: bool):
    """One full pass over a dataloader. Returns (avg_loss, accuracy)."""
    model.train() if training else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.no_grad() if not training else torch.enable_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def _train_phase(
    tag, model, train_loader, val_loader, loss_fn,
    optimizer, scheduler, device,
    num_epochs, patience, checkpoint_path,
):
    """Training loop for one phase. Saves best checkpoint; supports early stopping."""
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        train_loss, train_acc = _run_epoch(
            model, train_loader, loss_fn, optimizer, device, training=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, loss_fn, None, device, training=False
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

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

    return history


def train(cfg, model, train_loader, val_loader, class_weights=None):
    """
    Two-phase training:
      Phase 1 — freeze backbone, train head only (higher LR, few epochs)
      Phase 2 — unfreeze all, fine-tune entire network (lower LR, more epochs)

    Args:
        cfg:           config dict (see configs/default.yaml)
        model:         RisClassifier instance
        train_loader:  DataLoader for training split
        val_loader:    DataLoader for validation split
        class_weights: optional 1-D tensor of per-class weights for imbalance

    Returns:
        Combined training history dict with keys train_loss, train_acc, val_loss, val_acc.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on {device}")

    weight = class_weights.to(device) if class_weights is not None else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    checkpoint_path = cfg["train"]["checkpoint_path"]
    patience = cfg["train"].get("patience", 10)
    t = cfg["train"]

    # --- Phase 1: head only ---
    print("\n=== Phase 1: Training head only ===")
    model.freeze_backbone()
    optimizer1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=t["phase1_lr"],
        weight_decay=t["weight_decay"],
    )
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, t["phase1_epochs"])

    history1 = _train_phase(
        "phase1", model, train_loader, val_loader, loss_fn,
        optimizer1, scheduler1, device,
        t["phase1_epochs"], patience, checkpoint_path,
    )

    # Start phase 2 from the best phase 1 checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # --- Phase 2: full fine-tune ---
    print("\n=== Phase 2: Fine-tuning entire network ===")
    model.unfreeze()
    optimizer2 = torch.optim.AdamW(
        model.parameters(),
        lr=t["phase2_lr"],
        weight_decay=t["weight_decay"],
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, t["phase2_epochs"])

    history2 = _train_phase(
        "phase2", model, train_loader, val_loader, loss_fn,
        optimizer2, scheduler2, device,
        t["phase2_epochs"], patience, checkpoint_path,
    )

    best_val_acc = max(history2["val_acc"]) if history2["val_acc"] else max(history1["val_acc"])
    print(f"\nTraining complete. Best val acc: {best_val_acc:.3f}")
    print(f"Checkpoint: {checkpoint_path}")

    # Concatenate both phases into one history for plotting
    return {k: history1[k] + history2[k] for k in history1}
