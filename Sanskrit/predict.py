import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import load_label_map
from src.model import get_model
from src.utils import get_device, load_checkpoint


# Inference transform — same normalisation used during training
_INFER_TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def parse_args():
    p = argparse.ArgumentParser(description="Sanskrit MNIST single-image predictor")
    p.add_argument("--image",      default="data/raw/images/00_a/00_0000.png", help="Path to the input image")
    p.add_argument("--checkpoint", default="data/processed/best_model.pt")
    p.add_argument("--label-map",  default="data/raw/label_map.csv")
    p.add_argument("--top-k",      type=int, default=3, help="Show top-K predictions")
    return p.parse_args()


def predict(image_path: str, model, label_map: dict, device, top_k: int = 3):
    """
    Runs inference on a single image file.

    Returns a list of (label_dict, probability) tuples sorted by confidence.
    """
    img = Image.open(image_path)
    tensor = _INFER_TRANSFORM(img).unsqueeze(0).to(device)   # add batch dim

    model.eval()
    with torch.no_grad():
        logits = model(tensor)                          # shape: (1, num_classes)
        probs = F.softmax(logits, dim=1).squeeze(0)    # shape: (num_classes,)

    top_k_vals, top_k_idxs = probs.topk(top_k)

    results = []
    for prob, idx in zip(top_k_vals.cpu().tolist(), top_k_idxs.cpu().tolist()):
        results.append((label_map[idx], prob))

    return results


def main():
    args = parse_args()
    device = get_device()

    label_map = load_label_map(args.label_map)
    num_classes = len(label_map)

    model = get_model(num_classes=num_classes).to(device)
    ckpt = load_checkpoint(args.checkpoint, model, device)
    print(f"Model loaded (epoch {ckpt['epoch']}, val acc {ckpt['val_acc']*100:.2f}%)\n")

    results = predict(args.image, model, label_map, device, top_k=args.top_k)

    print(f"Image : {args.image}")
    print(f"{'Rank':<5}  {'Char':<6}  {'Roman':<10}  {'Category':<12}  {'Prob':>7}")
    print("─" * 46)
    for rank, (lbl, prob) in enumerate(results, 1):
        print(
            f"{rank:<5}  {lbl['char']:<6}  {lbl['roman']:<10}  "
            f"{lbl['category']:<12}  {prob*100:>6.2f}%"
        )


if __name__ == "__main__":
    main()
