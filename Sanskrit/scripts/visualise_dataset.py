#!/usr/bin/env python3
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import load_label_map


DATA_DIR  = Path("data/raw")
ASSET_DIR = Path("assets")
ASSET_DIR.mkdir(parents=True, exist_ok=True)


def class_distribution():
    label_map = load_label_map(DATA_DIR / "label_map.csv")
    images_dir = DATA_DIR / "images"

    counts = {}
    for folder in sorted(images_dir.iterdir()):
        if not folder.is_dir():
            continue
        label_id = int(folder.name.split("_")[0])
        counts[label_id] = len(list(folder.glob("*.png")))

    df = pd.DataFrame(
        [(label_map[k]["roman"], label_map[k]["char"], label_map[k]["category"], v)
         for k, v in counts.items()],
        columns=["roman", "char", "category", "count"],
    )

    # Color by category
    cat_color = {"vowel": "#4C72B0", "consonant": "#DD8452", "numeral": "#59A14F"}
    colors = df["category"].map(cat_color).tolist()

    fig, ax = plt.subplots(figsize=(20, 5))
    bars = ax.bar(df["roman"], df["count"], color=colors, width=0.7)
    ax.set_xlabel("Character (Roman transliteration)", fontsize=11)
    ax.set_ylabel("Image count", fontsize=11)
    ax.set_title("Sanskrit MNIST — Class Distribution", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["roman"], rotation=90, fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=cat.capitalize())
                       for cat, c in cat_color.items()]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    out = ASSET_DIR / "eda_class_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def sample_grid():
    label_map = load_label_map(DATA_DIR / "label_map.csv")
    images_dir = DATA_DIR / "images"
    folders = sorted(images_dir.iterdir())

    # Show 4 samples per class for first 20 classes
    n_classes = min(20, len(folders))
    n_samples = 4
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(n_samples * 1.5, n_classes * 1.5))
    fig.suptitle("Sanskrit MNIST — Sample Images (first 20 classes)", fontsize=12, fontweight="bold")

    for row, folder in enumerate(folders[:n_classes]):
        label_id = int(folder.name.split("_")[0])
        imgs = sorted(folder.glob("*.png"))[:n_samples]
        for col, img_path in enumerate(imgs):
            ax = axes[row][col]
            ax.imshow(Image.open(img_path), cmap="gray")
            if col == 0:
                lbl = label_map[label_id]
                ax.set_ylabel(f"{lbl['char']}\n{lbl['roman']}", fontsize=8, rotation=0,
                              labelpad=30, va="center")
            ax.axis("off")

    plt.tight_layout()
    out = ASSET_DIR / "eda_sample_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Generating EDA plots …")
    class_distribution()
    sample_grid()
    print("Done.")
