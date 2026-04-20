"""
dataset.py — Sanskrit MNIST dataset loader and preprocessor.

Loads images from the folder structure:
    data/raw/images/<class_folder>/<image>.png

Splits into train / val / test and returns PyTorch DataLoader objects.
"""

import os
import random
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


# ── Label map ──────────────────────────────────────────────────────────────

def load_label_map(csv_path: str | Path) -> dict[int, dict]:
    """
    Returns a dict mapping integer label → {char, roman, category}.
    Example: {0: {'char': 'अ', 'roman': 'a', 'category': 'vowel'}, ...}
    """
    df = pd.read_csv(csv_path)
    return {
        int(row["label"]): {
            "char": row["char"],
            "roman": row["roman"],
            "category": row["category"],
        }
        for _, row in df.iterrows()
    }


# ── Dataset ─────────────────────────────────────────────────────────────────

class SanskritMNIST(Dataset):
    """
    Reads Sanskrit MNIST images from disk.

    Args:
        images_dir : path to the images/ folder containing class sub-folders.
        label_map  : dict returned by load_label_map().
        transform  : torchvision transforms applied to each image.
    """

    def __init__(
        self,
        images_dir: str | Path,
        label_map: dict,
        transform=None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.label_map = label_map
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []   # (path, label)

        # Each sub-folder is named "<label_id>_<roman>" e.g. "00_a"
        for folder in sorted(self.images_dir.iterdir()):
            if not folder.is_dir():
                continue
            label_id = int(folder.name.split("_")[0])
            for img_path in sorted(folder.glob("*.png")):
                self.samples.append((img_path, label_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        # Load as grayscale (mode "L") → already 32×32 for this dataset
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Transforms ──────────────────────────────────────────────────────────────

def get_transforms(augment: bool = False):
    """
    Returns a transform pipeline.

    Training pipeline optionally adds light augmentation to improve
    generalisation without distorting the glyphs.

    Both pipelines:
        • Convert PIL → Tensor  (values 0–255 → 0.0–1.0)
        • Normalise  (mean=0.5, std=0.5) → range [-1, 1]
    """
    base = [
        transforms.ToTensor(),                      # H×W→1×H×W, [0,1]
        transforms.Normalize((0.5,), (0.5,)),       # → [-1, 1]
    ]
    if augment:
        aug = [
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)


# ── DataLoader factory ────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str | Path = "data/raw",
    batch_size: int = 64,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Build train / val / test DataLoaders from the raw dataset.

    Returns:
        train_loader, val_loader, test_loader, label_map
    """
    data_dir = Path(data_dir)
    label_map = load_label_map(data_dir / "label_map.csv")

    # Full dataset (without augmentation — we'll wrap train subset)
    full_dataset = SanskritMNIST(
        images_dir=data_dir / "images",
        label_map=label_map,
        transform=get_transforms(augment=False),
    )

    # Reproducible split
    total = len(full_dataset)
    n_test = int(total * test_split)
    n_val = int(total * val_split)
    n_train = total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Re-wrap train subset with augmentation
    train_ds.dataset = SanskritMNIST(
        images_dir=data_dir / "images",
        label_map=label_map,
        transform=get_transforms(augment=True),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(
        f"Dataset split — train: {n_train} | val: {n_val} | test: {n_test} "
        f"| classes: {len(label_map)}"
    )
    return train_loader, val_loader, test_loader, label_map
