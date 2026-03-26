from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fruit_classification.utils.artifacts import save_confusion_matrix, save_history_csv, save_json
from fruit_classification.utils.metrics import compute_classification_metrics, summarize_probabilities


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer_name = config["training"].get("optimizer", "adamw").lower()
    lr = float(config["training"]["lr"])
    weight_decay = float(config["training"].get("weight_decay", 1e-4))

    if optimizer_name == "sgd":
        return torch.optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> Any:
    scheduler_name = config["training"].get("scheduler", "cosine").lower()
    epochs = int(config["training"]["epochs"])
    if scheduler_name == "none":
        return None
    if scheduler_name == "step":
        step_size = int(config["training"].get("step_size", max(1, epochs // 3)))
        gamma = float(config["training"].get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def _move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["image"].to(device), batch["label"].to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for batch in tqdm(loader, desc="train", leave=False):
        images, labels = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return {"loss": running_loss / max(total, 1), "accuracy": correct / max(total, 1)}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    model.eval()
    running_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    probs: list[np.ndarray] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        images, labels = _move_batch(batch, device)
        logits = model(images)
        loss = criterion(logits, labels)
        prob = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        probs.append(prob.cpu().numpy())

    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    metrics["loss"] = running_loss / max(total, 1)
    if probs:
        metrics.update(summarize_probabilities(np.concatenate(probs, axis=0)))
    return metrics


def run_training(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    config: dict[str, Any],
    class_names: list[str],
    run_dir: str | Path,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["training"].get("label_smoothing", 0.0)))
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    epochs = int(config["training"]["epochs"])
    early_stopping = int(config["training"].get("early_stopping_patience", 0))
    monitor_metric = config["training"].get("monitor_metric", "macro_f1")

    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = -1
    best_checkpoint_path = run_dir / "best_model.pt"
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_metrics = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_metrics = evaluate(model, dataloaders["val"], criterion, device, class_names)
        if scheduler is not None:
            scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "elapsed_sec": time.time() - start_time,
            }
        )

        epoch_log = history[-1]
        print(
            "[Epoch {epoch:03d}/{epochs:03d}] "
            "train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            "val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} val_macro_f1={val_macro_f1:.4f} | "
            "lr={lr:.6f} time={elapsed_sec:.1f}s".format(
                epoch=epoch,
                epochs=epochs,
                **epoch_log,
            )
        )

        current_score = float(val_metrics[monitor_metric])
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "class_names": class_names,
                    "best_score": best_score,
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if early_stopping > 0 and epochs_without_improvement >= early_stopping:
            print(
                f"Early stopping at epoch {epoch}: no improvement in {monitor_metric} "
                f"for {epochs_without_improvement} epoch(s)."
            )
            break

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, dataloaders["test"], criterion, device, class_names)

    save_history_csv(history, run_dir / "history.csv")
    save_json(
        {
            "best_epoch": best_epoch,
            "best_score": best_score,
            "monitor_metric": monitor_metric,
            "test_metrics": test_metrics,
        },
        run_dir / "summary.json",
    )
    save_json(test_metrics, run_dir / "test_metrics.json")
    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names,
        run_dir / "confusion_matrix_test.png",
        title="Test Confusion Matrix",
    )

    return {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "monitor_metric": monitor_metric,
        "history": history,
        "test_metrics": test_metrics,
        "best_checkpoint_path": str(best_checkpoint_path),
    }
