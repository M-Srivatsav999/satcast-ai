# models.py – CNN + LSTM + Attention (Enhanced Architecture)
"""
Architecture improvements over baseline:
  1. Deeper ResNet-style CNN with skip connections
  2. Bidirectional LSTM for richer temporal encoding
  3. Multi-head self-attention over the time axis
  4. Multi-horizon output head (1h, 3h, 6h, 12h, 24h)
  5. Auxiliary classifier head: rain category (5 classes)
  6. Uncertainty estimation via MC-Dropout at inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CNN_FILTERS, LSTM_UNITS, ATTENTION_HEADS, DROPOUT, IMG_SIZE, FORECAST_HOURS


# ── 1. Residual CNN Block ────────────────────────────────────────────
class ResBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN + skip connection."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


# ── 2. Deep CNN Feature Extractor ───────────────────────────────────
class DeepCNNExtractor(nn.Module):
    """
    4-stage CNN with residual blocks.
    Input:  (B, 3, 128, 128)
    Output: (B, 256) feature vector
    """
    def __init__(self):
        super().__init__()
        f = CNN_FILTERS  # [32, 64, 128, 256]

        self.stage1 = nn.Sequential(
            nn.Conv2d(3,    f[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f[0]), nn.ReLU(),
            ResBlock(f[0]),
        )                                               # → (B, 32, 64, 64)

        self.stage2 = nn.Sequential(
            nn.Conv2d(f[0], f[1], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f[1]), nn.ReLU(),
            ResBlock(f[1]),
        )                                               # → (B, 64, 32, 32)

        self.stage3 = nn.Sequential(
            nn.Conv2d(f[1], f[2], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f[2]), nn.ReLU(),
            ResBlock(f[2]),
        )                                               # → (B, 128, 16, 16)

        self.stage4 = nn.Sequential(
            nn.Conv2d(f[2], f[3], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f[3]), nn.ReLU(),
            ResBlock(f[3]),
        )                                               # → (B, 256, 8, 8)

        self.gap  = nn.AdaptiveAvgPool2d(1)             # → (B, 256, 1, 1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f[3], 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )                                               # → (B, 256)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        return self.proj(x)


# ── 3. Temporal Attention ────────────────────────────────────────────
class TemporalAttention(nn.Module):
    """
    Multi-head self-attention over the sequence dimension.
    Lets the model weight 'which past frame matters most'.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=DROPOUT, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):              # x: (B, T, d_model)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


# ── 4. Main CNN-LSTM-Attention Model ────────────────────────────────
class RainfallNet(nn.Module):
    """
    Full model:
      CNN per frame → Bidirectional LSTM → Temporal Attention → Multi-head output

    Outputs:
      • regression:   (B, 5)  – mm/h at 1h, 3h, 6h, 12h, 24h horizons
      • category:     (B, 5)  – rain severity class logits (5 classes)
      • uncertainty:  sampled via MC-Dropout at inference
    """
    def __init__(self):
        super().__init__()
        self.cnn      = DeepCNNExtractor()
        self.lstm     = nn.LSTM(
            256, LSTM_UNITS,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT,
        )
        d_lstm = LSTM_UNITS * 2   # bidirectional doubles dim
        self.attention = TemporalAttention(d_lstm, ATTENTION_HEADS)

        # Regression head: predict mm/h for each forecast horizon
        self.reg_head = nn.Sequential(
            nn.Linear(d_lstm, 128), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(128,  64),   nn.GELU(),
            nn.Linear(64,   len(FORECAST_HOURS)),
            nn.Softplus(),          # guarantee non-negative output
        )

        # Classification head: rain severity category
        self.cls_head = nn.Sequential(
            nn.Linear(d_lstm, 64), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(64, 5),      # 5 categories
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, 3, H, W)  – batch of image sequences
        Returns:
            regression: (B, len(FORECAST_HOURS))
            category:   (B, 5)
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats = self.cnn(x_flat)                    # (B*T, 256)
        feats = feats.view(B, T, 256)               # (B, T, 256)

        lstm_out, _ = self.lstm(feats)              # (B, T, 2*LSTM_UNITS)
        attn_out = self.attention(lstm_out)         # (B, T, 2*LSTM_UNITS)
        ctx = attn_out[:, -1, :]                    # last timestep context

        regression = self.reg_head(ctx)             # (B, num_horizons)
        category   = self.cls_head(ctx)             # (B, 5)
        return regression, category

    @torch.no_grad()
    def predict_with_uncertainty(self, x, n_samples: int = 20):
        """
        MC-Dropout inference: run n_samples forward passes with dropout ON.
        Returns mean prediction + std (uncertainty estimate).
        """
        self.train()  # enable dropout
        preds = []
        for _ in range(n_samples):
            reg, _ = self.forward(x)
            preds.append(reg.cpu().numpy())
        self.eval()

        import numpy as np
        preds = np.array(preds)           # (n_samples, B, horizons)
        mean  = preds.mean(axis=0)
        std   = preds.std(axis=0)
        return mean, std
