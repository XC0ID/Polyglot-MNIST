import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import get_dataloaders
from src.model import get_model
from src.utils import (
    evaluate,
    get_device,
    load_checkpoint,
    plot_confusion_matrix,
    plot_sample_predictions,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Sanskrit MNIST model")
    p.add_argument("--checkpoint", default="data/processed/best_model.pt")
    p.add_argument("--data-dir",   default="data/raw")
    p.add_argument("--out-dir",    default="data/processed")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    _, _, test_loader, label_map = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    # ── Load model ─────────────────────────────────────────────────────────
    num_classes = len(label_map)
    model = get_model(num_classes=num_classes).to(device)
    ckpt = load_checkpoint(args.checkpoint, model, device)
    print(f"Loaded checkpoint  →  epoch {ckpt['epoch']} | val acc {ckpt['val_acc']*100:.2f}%")

    # ── Accuracy / loss ────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest accuracy : {test_acc*100:.2f}%")
    print(f"Test loss     : {test_loss:.4f}")

    # ── Per-class report ───────────────────────────────────────────────────
    y_true, y_pred = collect_predictions(model, test_loader, device)
    target_names = [
        f"{label_map[i]['roman']} ({label_map[i]['char']})"
        for i in range(num_classes)
    ]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
    print("\n" + report)

    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  ↳ Classification report saved to {report_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_confusion_matrix(
        model, test_loader, label_map, device,
        save_path=out_dir / "confusion_matrix.png",
    )
    plot_sample_predictions(
        model, test_loader, label_map, device,
        save_path=out_dir / "sample_predictions.png",
        n=32,
    )


if __name__ == "__main__":
    main()
