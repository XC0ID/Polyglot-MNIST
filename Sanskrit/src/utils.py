"""
utils.py — Training helpers, metric logging, and visualisation utilities.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")           # headless backend — safe on any server
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# ── Training loop helpers ────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Runs a single training epoch.

    Returns:
        avg_loss  (float)
        accuracy  (float, 0-1)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)               # forward pass
        loss = criterion(logits, labels)     # compute loss
        loss.backward()                      # backpropagation
        optimizer.step()                     # parameter update

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluates model on a DataLoader.

    Returns:
        avg_loss  (float)
        accuracy  (float, 0-1)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ── Metrics & checkpointing ──────────────────────────────────────────────

class MetricsTracker:
    """
    Lightweight tracker that records per-epoch train/val metrics and
    serialises them to JSON for later plotting.
    """

    def __init__(self) -> None:
        self.history: dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "epoch_time": [],
        }

    def update(self, train_loss, train_acc, val_loss, val_acc, epoch_time) -> None:
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["epoch_time"].append(epoch_time)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "MetricsTracker":
        tracker = cls()
        with open(path) as f:
            tracker.history = json.load(f)
        return tracker


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    path: str | Path,
    extra: dict[str, Any] | None = None,
) -> None:
    """Saves model state dict + training metadata to a .pt file."""
    payload = {
        "epoch": epoch,
        "val_acc": val_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: nn.Module, device: torch.device):
    """Loads weights into model in-place; returns the checkpoint dict."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


# ── Visualisation ────────────────────────────────────────────────────────

def plot_training_curves(tracker: MetricsTracker, save_path: str | Path) -> None:
    """Saves a 2-panel figure: loss curve + accuracy curve."""
    h = tracker.history
    epochs = range(1, len(h["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History — Sanskrit MNIST CNN", fontsize=13, fontweight="bold")

    # Loss
    ax1.plot(epochs, h["train_loss"], label="Train loss", color="#4C72B0")
    ax1.plot(epochs, h["val_loss"],   label="Val loss",   color="#DD8452", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in h["train_acc"]], label="Train acc", color="#4C72B0")
    ax2.plot(epochs, [a * 100 for a in h["val_acc"]],   label="Val acc",   color="#DD8452", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ↳ Training curves saved to {save_path}")


@torch.no_grad()
def plot_confusion_matrix(
    model: nn.Module,
    loader,
    label_map: dict,
    device: torch.device,
    save_path: str | Path,
) -> None:
    """Saves a heatmap of the per-class confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    class_names = [label_map[i]["roman"] for i in range(len(label_map))]

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title("Confusion Matrix — Sanskrit MNIST (test set)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0,  fontsize=6)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ↳ Confusion matrix saved to {save_path}")


@torch.no_grad()
def plot_sample_predictions(
    model: nn.Module,
    loader,
    label_map: dict,
    device: torch.device,
    save_path: str | Path,
    n: int = 32,
) -> None:
    """Saves a grid of n sample images with true and predicted labels."""
    model.eval()
    images, labels = next(iter(loader))
    images_dev = images[:n].to(device)
    preds = model(images_dev).argmax(dim=1).cpu().numpy()
    labels = labels[:n].numpy()

    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.2))
    fig.suptitle("Sample Predictions — Sanskrit MNIST", fontsize=12, fontweight="bold")

    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.axis("off")
            continue
        img = images[idx].squeeze().numpy()
        img = (img * 0.5) + 0.5         # de-normalise to [0, 1]
        ax.imshow(img, cmap="gray")
        true_lbl = label_map[labels[idx]]["roman"]
        pred_lbl = label_map[preds[idx]]["roman"]
        color = "green" if labels[idx] == preds[idx] else "red"
        ax.set_title(f"T:{true_lbl}\nP:{pred_lbl}", fontsize=7, color=color)
        ax.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ↳ Sample predictions saved to {save_path}")


def get_device() -> torch.device:
    """Returns CUDA if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def set_seed(seed: int = 42) -> None:
    """Fixes all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
