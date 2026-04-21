# 🕉️ Sanskrit MNIST — Devanagari Character Recognition

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3.1-orange?logo=pytorch)](https://pytorch.org)
[![License MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset: 62 classes](https://img.shields.io/badge/Classes-62-purple)]()
[![Images: 31k](https://img.shields.io/badge/Images-31%2C000-teal)]()

A complete deep learning pipeline for classifying **62 Devanagari characters** — vowels, consonants, and Sanskrit numerals — using a compact CNN trained on 32×32 grayscale images.

---

## 📖 Table of Contents

- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Inference](#-inference)
- [Results](#-results)
- [Docker](#-docker)
- [Why Sanskrit MNIST?](#-why-sanskrit-mnist)
- [Known Limitations](#-known-limitations)
- [References](#-references)

---

## 📊 Dataset

| Property       | Value                         |
|---------------|-------------------------------|
| Total images   | 31,000                       |
| Classes        | 62                           |
| Images/class   | 500                          |
| Image size     | 32 × 32 px (grayscale)       |
| Format         | PNG                          |
| Split          | 70% train / 15% val / 15% test |

### Class categories

| Category    | Count | Examples                                      |
|-------------|-------|-----------------------------------------------|
| Vowels      | 15    | अ (a), आ (aa), इ (i), ई (ii), उ (u) …       |
| Consonants  | 37    | क (ka), ख (kha), ग (ga), … क्ष (ksha) …    |
| Numerals    | 10    | ० (0) – ९ (9)                               |

The folder structure mirrors the label IDs:

```
data/raw/
├── label_map.csv          # label → char / roman / category
└── images/
    ├── 00_a/              # Class 0 — अ
    │   ├── 00_0000.png
    │   └── ...            # 500 images
    ├── 01_aa/             # Class 1 — आ
    ├── ...
    └── 61_9/              # Class 61 — ९
```

### Getting the dataset

1. Download `Sanskrit_Mnist.zip`
2. Place it at `data/raw/Sanskrit_Mnist.zip`
3. Run the prep script:

```bash
python scripts/prepare_data.py
```

---

## 🧠 Model Architecture

**SanskritCNN** — a 3-block convolutional network designed for small (32×32) glyph images.

```
Input: 1 × 32 × 32

Conv Block 1:  Conv2d(1→32, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)   → 32×16×16
Conv Block 2:  Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → 64×8×8
Conv Block 3:  Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2×2) → 128×4×4

Flatten:  128×4×4 = 2,048

FC1:  Linear(2048 → 512) → ReLU → Dropout(0.5)
FC2:  Linear(512  → 62)

Output: 62-class logits
```

| Setting          | Value               |
|-----------------|---------------------|
| Optimizer        | Adam (lr = 1e-3)   |
| LR scheduler     | ReduceLROnPlateau (factor=0.5, patience=3) |
| Loss function    | CrossEntropyLoss   |
| Batch size       | 64                 |
| Epochs (default) | 25                 |
| Early stopping   | patience = 7 epochs |
| Dropout          | 0.5               |
| Trainable params | ~1.2 M             |
| Input norm       | mean=0.5, std=0.5 → range [-1, 1] |

### Data augmentation (training only)

- Random rotation ±10°
- Random affine translation ±10%

---

## 📁 Project Structure

```
sanskrit-mnist/
├── src/
│   ├── __init__.py        # Package exports
│   ├── dataset.py         # SanskritMNIST Dataset + DataLoader factory
│   ├── model.py           # SanskritCNN architecture
│   └── utils.py           # Train loop, metrics, plots, checkpointing
│
├── scripts/
│   ├── prepare_data.py    # Unzips & organises raw data
│   └── visualise_dataset.py  # EDA plots (class distribution, sample grid)
│
├── notebooks/
│   └── exploration.ipynb  # Step-by-step walkthrough (data → train → eval)
│
├── data/
│   ├── raw/               # Place dataset here (not committed to git)
│   │   ├── label_map.csv
│   │   └── images/
│   └── processed/         # Training outputs (checkpoints, plots, reports)
│
├── assets/                # EDA plots committed for the README
│
├── train.py               # ← Training entry point
├── evaluate.py            # ← Evaluation & confusion matrix
├── predict.py             # ← Single-image inference
│
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .gitattributes
└── README.md
```

---

## ⚡ Quick Start

### 1 — Clone & set up environment

```bash
git clone https://github.com/<your-username>/sanskrit-mnist.git
cd sanskrit-mnist

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Prepare data

```bash
# Copy Sanskrit_Mnist.zip into data/raw/, then:
python scripts/prepare_data.py
```

### 3 — (Optional) Explore the dataset

```bash
python scripts/visualise_dataset.py
# → assets/eda_class_distribution.png
# → assets/eda_sample_grid.png
```

### 4 — Train

```bash
python train.py
```

### 5 — Evaluate

```bash
python evaluate.py
```

### 6 — Predict on a new image

```bash
python predict.py --image data/raw/images/00_a/00_0000.png --top-k 3
```

---

## 🏋️ Training

```
python train.py [OPTIONS]

Options:
  --data-dir    PATH    Path to raw data dir            [default: data/raw]
  --out-dir     PATH    Output dir for checkpoints      [default: data/processed]
  --epochs      INT     Max training epochs             [default: 25]
  --batch-size  INT     Batch size                      [default: 64]
  --lr          FLOAT   Adam learning rate              [default: 0.001]
  --dropout     FLOAT   Dropout probability             [default: 0.5]
  --val-split   FLOAT   Validation fraction             [default: 0.15]
  --test-split  FLOAT   Test fraction                   [default: 0.15]
  --num-workers INT     DataLoader workers              [default: 2]
  --seed        INT     Random seed                     [default: 42]
  --patience    INT     Early stopping patience         [default: 7]
```

Training outputs in `data/processed/`:

| File                    | Description                          |
|------------------------|--------------------------------------|
| `best_model.pt`         | Best checkpoint (by val accuracy)   |
| `metrics.json`          | Per-epoch loss & accuracy history   |
| `training_curves.png`   | Loss + accuracy plots               |

---

## 📈 Evaluation

```
python evaluate.py [OPTIONS]

Options:
  --checkpoint  PATH    Model checkpoint to evaluate    [default: data/processed/best_model.pt]
  --data-dir    PATH    Raw data directory              [default: data/raw]
  --out-dir     PATH    Output directory                [default: data/processed]
  --batch-size  INT     Batch size                      [default: 64]
  --seed        INT     Random seed                     [default: 42]
```

Evaluation outputs in `data/processed/`:

| File                          | Description                          |
|------------------------------|--------------------------------------|
| `confusion_matrix.png`        | 62×62 heatmap                       |
| `sample_predictions.png`      | 32-image grid with true/pred labels |
| `classification_report.txt`   | Per-class precision/recall/F1       |

---

## 🔍 Inference

```
python predict.py --image PATH [OPTIONS]

Options:
  --image       PATH    Input image (required)
  --checkpoint  PATH    Model checkpoint                [default: data/processed/best_model.pt]
  --label-map   PATH    label_map.csv path              [default: data/raw/label_map.csv]
  --top-k       INT     Show top-K predictions          [default: 3]
```

Example output:

```
Image : data/raw/images/15_ka/15_0042.png
Rank   Char    Roman       Category      Prob
──────────────────────────────────────────────
1      क       ka          consonant    97.43%
2      ख       kha         consonant     1.24%
3      ग       ga          consonant     0.61%
```

---

## 📊 Results

> Results will be updated after training. Re-run `python train.py && python evaluate.py` and paste values below.

| Metric          | Value   |
|----------------|---------|
| Test accuracy   | —       |
| Test loss       | —       |
| Training time   | —       |
| Epochs trained  | —       |
| Hardware        | —       |

*Expected accuracy: ~95–97% on this dataset with the default settings.*

### Comparison table (Polyglot-MNIST context)

| Language    | Framework     | Test Acc | Train Time | Params  |
|-------------|--------------|----------|------------|---------|
| Python      | PyTorch 2.3  | —        | —          | ~1.2 M  |

---

## 🐳 Docker

**Build:**

```bash
docker build -t sanskrit-mnist .
```

**Train:**

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  sanskrit-mnist python train.py --epochs 25
```

**Evaluate:**

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  sanskrit-mnist python evaluate.py
```

**GPU support:** Replace the base image in `Dockerfile` with:
```
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
```

---

## 🕉️ Why Sanskrit MNIST?

Sanskrit (written in Devanagari script) is one of the world's oldest languages, with a phonetically precise alphabet of 46+ characters (plus numerals and conjuncts). Unlike MNIST's 10 digit classes, Sanskrit MNIST presents a substantially harder 62-class problem with visually similar glyphs — making it an excellent benchmark for studying:

- Fine-grained visual recognition
- Script recognition in low-resource settings
- Transfer learning from standard MNIST models

The dataset serves as a drop-in replacement for the original MNIST in the [Polyglot-MNIST](https://github.com/<your-username>/polyglot-mnist) repository, allowing direct accuracy and speed comparisons across implementations in different programming languages.

---

## ⚠️ Known Limitations

- **Dataset size:** At 500 images/class, the dataset is relatively small. Results may vary on out-of-distribution handwriting styles.
- **Conjunct consonants:** Characters like क्ष (ksha), ज्ञ (jna), and त्र (tra) are visually complex and may have lower per-class accuracy.
- **No test-time augmentation (TTA):** Adding TTA could push accuracy higher.
- **CPU training:** Training is optimised for readability; GPU acceleration is supported but no mixed-precision (AMP) is used by default.
- **Single model:** No ensemble — a simple average of 3–5 models typically adds 1–2% accuracy.

---

## 📚 References

- [Devanagari Handwritten Character Dataset — UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [LeCun et al. — Gradient-Based Learning Applied to Document Recognition (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- [Batch Normalisation — Ioffe & Szegedy (2015)](https://arxiv.org/abs/1502.03167)

---

## 📄 License

MIT © 2024 — see [LICENSE](LICENSE) for details.
