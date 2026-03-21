# train.py – Enhanced training with combined loss + proper evaluation
"""
Improvements over baseline:
  • Combined MSE + MAE loss (robustness to outliers)
  • Cosine annealing LR scheduler
  • Multi-horizon evaluation (R² per horizon)
  • Early stopping
  • Saves best model by validation loss (not last epoch)
  • Detailed training plots
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from data_preprocessing import get_dataloaders
from models import RainfallNet
from config import DEVICE, NUM_EPOCHS, LEARNING_RATE, FORECAST_HOURS


class CombinedLoss(nn.Module):
    """MSE + 0.5 * MAE – better than pure MSE for skewed rainfall distributions."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        return self.mse(pred, target) + 0.5 * self.mae(pred, target)


def train_model():
    print(f"🚀 Training RainfallNet on {DEVICE}...")
    print(f"   Forecast horizons: {FORECAST_HOURS} hours")

    train_loader, test_loader, scaler = get_dataloaders()
    model = RainfallNet().to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {n_params:,}")

    reg_criterion = CombinedLoss()
    cls_criterion = nn.CrossEntropyLoss()
    optimizer     = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler     = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience, patience_count = 8, 0

    for epoch in range(NUM_EPOCHS):
        # ── Training ────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for imgs, meta, rain_targets in train_loader:
            imgs         = imgs.to(DEVICE)
            rain_targets = rain_targets.to(DEVICE)

            optimizer.zero_grad()
            reg_pred, cls_pred = model(imgs)

            reg_loss = reg_criterion(reg_pred, rain_targets)
            # Derive rain category from scaled target (use 1h horizon)
            t_1h = rain_targets[:, 0].detach().cpu().numpy()
            cats = _targets_to_categories(scaler.inverse_transform(
                np.column_stack([t_1h] + [np.zeros(len(t_1h))] * (len(FORECAST_HOURS)-1))
            )[:, 0])
            cats_t = torch.LongTensor(cats).to(DEVICE)
            cls_loss = cls_criterion(cls_pred, cats_t)

            loss = reg_loss + 0.1 * cls_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── Validation ──────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        preds_all, truths_all = [], []
        with torch.no_grad():
            for imgs, meta, rain_targets in test_loader:
                imgs         = imgs.to(DEVICE)
                rain_targets = rain_targets.to(DEVICE)
                reg_pred, _  = model(imgs)
                val_loss    += reg_criterion(reg_pred, rain_targets).item()
                preds_all.extend(reg_pred.cpu().numpy())
                truths_all.extend(rain_targets.cpu().numpy())

        avg_val = val_loss / len(test_loader)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss  = avg_val
            patience_count = 0
            torch.save(model.state_dict(), "cnn_lstm_model.pth")
        else:
            patience_count += 1

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS}  "
                  f"train={avg_train:.4f}  val={avg_val:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

        if patience_count >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # ── Final Evaluation ────────────────────────────────────────────
    preds_inv  = scaler.inverse_transform(np.array(preds_all))
    truths_inv = scaler.inverse_transform(np.array(truths_all))

    print("\n📊 R² per forecast horizon:")
    for i, hrs in enumerate(FORECAST_HOURS):
        r2  = r2_score(truths_inv[:, i], preds_inv[:, i])
        mae = mean_absolute_error(truths_inv[:, i], preds_inv[:, i])
        print(f"   {hrs:3d}h  R²={r2:.3f}   MAE={mae:.2f} mm/h")

    overall_r2 = r2_score(truths_inv.flatten(), preds_inv.flatten())
    print(f"\n✅ Overall R²: {overall_r2:.3f}")

    # ── Save scaler ─────────────────────────────────────────────────
    joblib.dump(scaler, "scaler.pkl")

    # ── Plots ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Loss curves
    axes[0].plot(train_losses, label="Train", color="#2196F3")
    axes[0].plot(val_losses,   label="Val",   color="#FF5722")
    axes[0].set_title("Loss Curves"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # 1h horizon scatter
    axes[1].scatter(truths_inv[:, 0], preds_inv[:, 0], alpha=0.4, s=10, color="#4CAF50")
    mx = max(truths_inv[:, 0].max(), preds_inv[:, 0].max())
    axes[1].plot([0, mx], [0, mx], 'r--', lw=1.5)
    axes[1].set_title(f"1h Forecast  R²={r2_score(truths_inv[:,0], preds_inv[:,0]):.3f}")
    axes[1].set_xlabel("True (mm/h)"); axes[1].set_ylabel("Predicted")

    # R² per horizon bar
    r2s = [r2_score(truths_inv[:, i], preds_inv[:, i]) for i in range(len(FORECAST_HOURS))]
    axes[2].bar([f"{h}h" for h in FORECAST_HOURS], r2s, color="#9C27B0", alpha=0.8)
    axes[2].set_ylim(0, 1); axes[2].set_title("R² by Horizon")
    axes[2].axhline(0.7, color='red', linestyle='--', alpha=0.5, label='Baseline 0.7')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("💾 Saved: cnn_lstm_model.pth | scaler.pkl | training_results.png")
    return overall_r2


def _targets_to_categories(rain_values: np.ndarray) -> list[int]:
    """Map mm/h values to category indices (0-4)."""
    cats = []
    for v in rain_values:
        if v < 1:   cats.append(0)
        elif v < 5: cats.append(1)
        elif v < 15: cats.append(2)
        elif v < 35: cats.append(3)
        else:        cats.append(4)
    return cats


if __name__ == "__main__":
    r2 = train_model()
    print(f"\n🎉 DONE! R²={r2:.3f}  →  streamlit run app.py")
