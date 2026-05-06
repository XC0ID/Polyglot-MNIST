# 🇪🇸 Spanish MNIST Dataset

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Classes-80-purple)](data/raw/spanish_mnist.csv)

A handwritten-style character recognition dataset for the **Spanish language** — extending classic MNIST with digits, uppercase/lowercase letters, accented characters (á, é, í, ó, ú, ü), and the unique Spanish characters **ñ, ¡, ¿**.

---

## 📦 Dataset Overview

| Category | Characters | Classes |
|---|---|---|
| Digits | 0–9 | 10 |
| Uppercase | A–Z | 26 |
| Accented Uppercase | Á É Í Ó Ú Ü Ñ | 7 |
| Lowercase | a–z | 26 |
| Accented Lowercase | á é í ó ú ü ñ | 7 |
| Special Spanish | ¡ ¿ « » | 4 |
| **Total** | | **80 classes** |

- **Image size:** 28 × 28 pixels  
- **Color mode:** Grayscale  
- **Samples per class:** 10 (expandable via generator)  
- **Total images:** 800 PNG files  
- **Format:** PNG + CSV metadata  

---

## 🗂️ Project Structure

```
spanish-mnist/
├── data/
│   ├── raw/
│   │   ├── images/              # 800 PNG images (28x28, grayscale)
│   │   │   ├── digit_0/
│   │   │   ├── upper_A/
│   │   │   ├── lower_a/
│   │   │   ├── upper_Á/
│   │   │   └── ...
│   │   └── spanish_mnist.csv    # Full metadata CSV
│   └── processed/               # Train/val/test splits (auto-generated)
│
├── datasets/
│   └── spanish_mnist_dataset.py # PyTorch Dataset class
│
├── models/
│   ├── cnn_model.py             # CNN classifier
│   └── lenet_model.py           # LeNet-5 variant
│
├── utils/
│   ├── generate_dataset.py      # Dataset generator (expand samples)
│   ├── visualize.py             # Visualization utilities
│   └── metrics.py               # Evaluation helpers
│
├── scripts/
│   ├── prepare_data.py          # Train/val/test split
│   └── train.py                 # Training script
│
├── tests/
│   ├── test_dataset.py
│   └── test_model.py
│
├── notebooks/
│   └── 01_exploration.ipynb     # EDA notebook
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/spanish-mnist.git
cd spanish-mnist
pip install -r requirements.txt
```

### 2. Prepare Data (train/val/test split)

```bash
python scripts/prepare_data.py --split 0.7 0.15 0.15
```

### 3. Generate More Samples (optional)

```bash
python utils/generate_dataset.py --samples_per_class 100 --output data/raw
```

### 4. Train the Model

```bash
python scripts/train.py --model cnn --epochs 30 --batch_size 64
```

### 5. Evaluate

```bash
python scripts/train.py --evaluate --checkpoint checkpoints/best_model.pth
```

---

## 🐍 Python API

```python
from datasets.spanish_mnist_dataset import SpanishMNISTDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = SpanishMNISTDataset(
    csv_file="data/raw/spanish_mnist.csv",
    img_dir="data/raw",
    split="train",
    transform=transform
)

loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

for images, labels in loader:
    print(images.shape)  # torch.Size([64, 1, 28, 28])
    print(labels.shape)  # torch.Size([64])
    break
```

---

## 📊 Sample Images

Each character is rendered with realistic handwriting-style augmentation:
- Random font variation (Serif, Italic)
- Random rotation (±12°)
- Random position jitter
- Gaussian noise
- Slight blur

---

## 🧠 Models Included

| Model | Parameters | Description |
|---|---|---|
| `CNNModel` | ~120K | 3-layer CNN with BatchNorm + Dropout |
| `LeNet5Spanish` | ~60K | Classic LeNet-5 adapted for 80 classes |

---

## 📋 CSV Format

```
class_id, character, category, label, sample_index, filename, unicode, width, height, color_mode
0, 0, digit, digit_0, 0, images/digit_0/digit_0_0000.png, U+0030, 28, 28, grayscale
...
```

---

## 🤝 Contributing

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/new-augmentation`)
3. Commit changes (`git commit -m 'Add new augmentation'`)
4. Push and open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

Inspired by the original [MNIST dataset](http://yann.lecun.com/exdb/mnist/) by Yann LeCun et al. Extended to support the full Spanish character set.
