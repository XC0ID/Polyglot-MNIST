"""
model.py — CNN model for Sanskrit MNIST classification.

Architecture  : SanskritCNN
Input         : 1 × 32 × 32 (grayscale)
Output        : 62-class softmax logits

Architecture summary:
    Conv block 1 : Conv2d(1→32, 3×3) → BN → ReLU → MaxPool(2×2)  →  16×16
    Conv block 2 : Conv2d(32→64, 3×3) → BN → ReLU → MaxPool(2×2) →   8×8
    Conv block 3 : Conv2d(64→128, 3×3) → BN → ReLU → MaxPool(2×2) →  4×4
    Flatten      : 128 × 4 × 4 = 2048
    FC1          : 2048 → 512 → ReLU → Dropout(0.5)
    FC2          : 512  → 62  (logits)

Optimizer : Adam (lr=1e-3)
Loss      : CrossEntropyLoss
"""

import torch
import torch.nn as nn


class SanskritCNN(nn.Module):
    """
    Compact CNN for 32×32 grayscale glyph recognition.

    Designed to be shallow enough to train quickly on CPU while
    achieving strong accuracy on the 62-class Sanskrit MNIST task.
    """

    def __init__(self, num_classes: int = 62, dropout: float = 0.5) -> None:
        super().__init__()

        # ── Convolutional backbone ────────────────────────────────────────
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 1×32×32 → 32×32×32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # → 32×16×16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → 64×16×16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # → 64×8×8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # → 128×8×8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # → 128×4×4
        )

        # ── Classifier head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                                  # 128×4×4 = 2048
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        # Weight initialisation (Kaiming for conv, Xavier for linear)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # Convolutional feature extraction
        x = self.classifier(x)     # Classification head
        return x                   # Raw logits — loss fn applies softmax


def get_model(num_classes: int = 62, dropout: float = 0.5) -> SanskritCNN:
    """Convenience factory; returns a fresh SanskritCNN."""
    return SanskritCNN(num_classes=num_classes, dropout=dropout)


def count_parameters(model: nn.Module) -> int:
    """Returns the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = get_model()
    x = torch.randn(4, 1, 32, 32)           # batch of 4 dummy images
    out = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    print(f"Trainable parameters : {count_parameters(model):,}")
