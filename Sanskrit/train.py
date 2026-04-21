#!/usr/bin/env python3
"""
train.py — Entry point for training the Sanskrit MNIST CNN.

Usage:
    python train.py                          # default settings
    python train.py --epochs 30 --lr 5e-4   # custom hyper-params
    python train.py --help                   # show all options

Outputs saved to:
    data/processed/metrics.json
    data/processed/best_model.pt
    data/processed/training_curves.png
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import get_dataloaders
from src.model import get_model, count_parameters
from src.utils import (
    MetricsTracker,
    evaluate,
    get_device,
    plot_training_curves,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Sanskrit MNIST CNN")
    p.add_argument("--data-dir",    default="data/raw",  help="Path to raw dataset")
    p.add_argument("--out-dir",     default="data/processed", help="Output directory")
    p.add_argument("--epochs",      type=int,   default=25,   help="Number of epochs")
    p.add_argument("--batch-size",  type=int,   default=64,   help="Batch size")
    p.add_argument("--lr",          type=float, default=1e-3, help="Learning rate (Adam)")
    p.add_argument("--dropout",     type=float, default=0.5,  help="Dropout probability")
    p.add_argument("--val-split",   type=float, default=0.15, help="Validation fraction")
    p.add_argument("--test-split",  type=float, default=0.15, help="Test fraction")
    p.add_argument("--num-workers", type=int,   default=2,    help="DataLoader workers")
    p.add_argument("--seed",        type=int,   default=42,   help="Random seed")
    p.add_argument("--patience",    type=int,   default=7,    help="Early-stopping patience (epochs without val improvement)")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_device()

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, label_map = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    num_classes = len(label_map)
    model = get_model(num_classes=num_classes, dropout=args.dropout).to(device)
    print(f"Model: SanskritCNN | classes: {num_classes} | params: {count_parameters(model):,}")

    # ── Optimiser & Loss ──────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──────────────────────────────────────────────────────
    tracker = MetricsTracker()
    best_val_acc = 0.0
    no_improve_count = 0
    total_train_time = 0.0

    print("\n" + "─" * 70)
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'Time':>6}")
    print("─" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Forward + backward on training set
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - t0
        total_train_time += epoch_time

        tracker.update(train_loss, train_acc, val_loss, val_acc, epoch_time)
        scheduler.step(val_acc)

        print(
            f"{epoch:>5}  {train_loss:>10.4f}  {train_acc*100:>8.2f}%  "
            f"{val_loss:>8.4f}  {val_acc*100:>6.2f}%  {epoch_time:>5.1f}s"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                path=out_dir / "best_model.pt",
                extra={"args": vars(args), "num_classes": num_classes},
            )
            print(f"  ★  New best val accuracy: {val_acc*100:.2f}% — checkpoint saved")
        else:
            no_improve_count += 1
            if no_improve_count >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (patience={args.patience}).")
                break

    print("─" * 70)

    # ── Final evaluation on test set ──────────────────────────────────────
    print("\nLoading best model for test evaluation …")
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*40}")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Total time    : {total_train_time:.1f}s ({total_train_time/60:.1f} min)")
    print(f"{'='*40}\n")

    # ── Save metrics & plots ──────────────────────────────────────────────
    tracker.save(out_dir / "metrics.json")
    plot_training_curves(tracker, out_dir / "training_curves.png")

    # Append final test metrics to checkpoint
    ckpt_data = torch.load(out_dir / "best_model.pt", map_location="cpu")
    ckpt_data["test_acc"] = test_acc
    ckpt_data["test_loss"] = test_loss
    ckpt_data["total_train_time_s"] = total_train_time
    torch.save(ckpt_data, out_dir / "best_model.pt")

    print("All outputs saved to:", out_dir)


if __name__ == "__main__":
    main()
