#!/usr/bin/env python3

import argparse
import shutil
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Prepare Sanskrit MNIST data")
    p.add_argument(
        "--src",
        default=None,
        help="Path to the extracted 'Sanskrit Mnist' folder. "
             "If omitted, the script looks for data/raw/Sanskrit_Mnist.zip and extracts it.",
    )
    p.add_argument("--dst", default="data/raw", help="Destination directory")
    p.add_argument("--zip", default="data/raw/Sanskrit_Mnist.zip")
    return p.parse_args()


def main():
    args = parse_args()
    dst = Path(args.dst)

    # ── Find / extract source ──────────────────────────────────────────────
    if args.src:
        src = Path(args.src)
    else:
        zip_path = Path(args.zip)
        if not zip_path.exists():
            print(
                f"ERROR: Zip file not found at '{zip_path}'.\n"
                "Please download the dataset and place it there, or pass --src."
            )
            sys.exit(1)

        print(f"Extracting {zip_path} …")
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dst)

        # The zip contains a top-level "Sanskrit Mnist/" folder
        src = dst / "Sanskrit Mnist"

    if not src.exists():
        print(f"ERROR: Source folder '{src}' not found.")
        sys.exit(1)

    # ── Copy images/ and label_map.csv ────────────────────────────────────
    images_dst = dst / "images"
    label_dst  = dst / "label_map.csv"

    if images_dst.exists():
        print(f"'{images_dst}' already exists — skipping image copy.")
    else:
        print(f"Copying images …")
        shutil.copytree(src / "images", images_dst)

    if label_dst.exists():
        print(f"'{label_dst}' already exists — skipping CSV copy.")
    else:
        shutil.copy(src / "label_map.csv", label_dst)

    # ── Verify ─────────────────────────────────────────────────────────────
    n_classes = len(list(images_dst.iterdir()))
    n_images  = len(list(images_dst.rglob("*.png")))
    print(f"\n✓ Data ready at '{dst}'")
    print(f"  Classes : {n_classes}")
    print(f"  Images  : {n_images}")


if __name__ == "__main__":
    main()
