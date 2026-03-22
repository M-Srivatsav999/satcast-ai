# predict.py – Smart prediction engine

import os
import torch
import joblib
import numpy as np
from PIL import Image
from models import RainfallNet
from config import DEVICE, IMG_SIZE, SEQ_LENGTH, FORECAST_HOURS, RAIN_CATEGORIES, SEVERITY_COLORS

_model_cache  = None
_scaler_cache = None


def _load_model_and_scaler():
    global _model_cache, _scaler_cache
    if _model_cache is None:
        model = RainfallNet()
        model.load_state_dict(torch.load("cnn_lstm_model.pth", map_location="cpu"))
        model.eval()
        _model_cache = model
    if _scaler_cache is None:
        _scaler_cache = joblib.load("scaler.pkl")
    return _model_cache, _scaler_cache


def _images_to_tensor(image_inputs):
    imgs = []
    for img in image_inputs[-SEQ_LENGTH:]:
        if isinstance(img, str):
            img = Image.open(img)
        img = img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        imgs.append(arr)
    while len(imgs) < SEQ_LENGTH:
        imgs.append(imgs[-1])
    return torch.FloatTensor(np.array(imgs)).unsqueeze(0)


def _classify_rain(mm_h):
    for category, (lo, hi) in RAIN_CATEGORIES.items():
        if lo <= mm_h < hi:
            return category, SEVERITY_COLORS[category]
    return "Extreme", SEVERITY_COLORS["Extreme"]


def _extract_cloud_signal(image_inputs):
    scores = []
    for img_input in image_inputs:
        if isinstance(img_input, str):
            img = Image.open(img_input).convert("RGB")
        else:
            img = img_input.convert("RGB")

        arr = np.array(img.resize((256, 256))).astype(np.float32) / 255.0
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        brightness = (r + g + b) / 3.0

        cloud_cover  = float(np.mean(brightness > 0.40))
        dense_cloud  = float(np.mean(brightness > 0.60))
        cold_tops    = float(np.mean((r > 0.75) & (g > 0.75) & (b > 0.75)))
        conv_texture = min(float(np.var(brightness)) * 6.0, 1.0)
        clear_sky    = float(np.mean((b > r * 1.15) & (brightness < 0.55)))
        adjusted     = max(0.0, cloud_cover - clear_sky * 0.6)

        score = (
            adjusted     * 0.30 +
            dense_cloud  * 0.25 +
            cold_tops    * 0.25 +
            conv_texture * 0.20
        )
        scores.append(float(np.clip(score, 0.0, 1.0)))

    cloud_score = float(np.mean(scores))

    if cloud_score < 0.08:
        rain_1h = cloud_score * 10.0
    elif cloud_score < 0.18:
        rain_1h = 0.8 + (cloud_score - 0.08) * 20.0
    elif cloud_score < 0.30:
        rain_1h = 2.8 + (cloud_score - 0.18) * 60.0
    elif cloud_score < 0.45:
        rain_1h = 10.0 + (cloud_score - 0.30) * 100.0
    elif cloud_score < 0.65:
        rain_1h = 25.0 + (cloud_score - 0.45) * 100.0
    else:
        rain_1h = 45.0 + (cloud_score - 0.65) * 120.0

    return {
        "cloud_score": round(cloud_score, 3),
        "rain_1h_mmh": round(float(rain_1h), 2),
    }


@torch.no_grad()
def predict_rainfall(image_inputs, n_mc_samples=20, use_physics=False):
    model, scaler = _load_model_and_scaler()
    x = _images_to_tensor(image_inputs).to(DEVICE)

    mean_scaled, std_scaled = model.predict_with_uncertainty(x, n_samples=n_mc_samples)
    mean_inv = scaler.inverse_transform(mean_scaled)[0]
    std_inv  = scaler.inverse_transform(np.clip(std_scaled, 0, None))[0]

    if not use_physics:
        horizons = []
        for i, hrs in enumerate(FORECAST_HOURS):
            mm_h = float(max(0.0, mean_inv[i]))
            unc  = float(max(0.0, std_inv[i]))
            cat, col = _classify_rain(mm_h)
            horizons.append({
                "hours":       hrs,
                "mm_h":        round(mm_h, 1),
                "uncertainty": round(unc, 1),
                "category":    cat,
                "color":       col,
            })
        primary = horizons[0]
        cv = (primary["uncertainty"] / (primary["mm_h"] + 1e-3)) * 100
        confidence = round(max(20.0, min(98.0, 100 - cv * 0.5)), 1)

        return {
            "horizons":       horizons,
            "primary_mm_h":   primary["mm_h"],
            "primary_cat":    primary["category"],
            "primary_color":  primary["color"],
            "confidence_pct": confidence,
            "model_version":  "CNN+BiLSTM+Attention v2.0",
            "cloud_score":    None,
        }

    else:
        physics      = _extract_cloud_signal(image_inputs)
        phys_rain_1h = physics["rain_1h_mmh"]
        cloud_score  = physics["cloud_score"]

        model_1h = max(float(mean_inv[0]), 1e-6)
        if model_1h > 0.5:
            ratios = [max(float(mean_inv[i]), 0) / model_1h for i in range(len(FORECAST_HOURS))]
        else:
            ratios = [1.0, 0.80, 0.62, 0.42, 0.25]

        horizons = []
        for i, hrs in enumerate(FORECAST_HOURS):
            mm_h = round(float(max(0.0, phys_rain_1h * ratios[i])), 1)
            unc  = round(float(max(0.05, mm_h * 0.18)), 1)
            cat, col = _classify_rain(mm_h)
            horizons.append({
                "hours":       hrs,
                "mm_h":        mm_h,
                "uncertainty": unc,
                "category":    cat,
                "color":       col,
            })

        if cloud_score < 0.08:
            confidence = 92.0
        elif cloud_score < 0.18:
            confidence = 70.0
        elif cloud_score < 0.35:
            confidence = 78.0
        else:
            confidence = round(min(97.0, 85.0 + cloud_score * 12), 1)

        primary = horizons[0]
        return {
            "horizons":       horizons,
            "primary_mm_h":   primary["mm_h"],
            "primary_cat":    primary["category"],
            "primary_color":  primary["color"],
            "confidence_pct": confidence,
            "model_version":  "CNN+BiLSTM+Attention v2.0 + Physics Scaling",
            "cloud_score":    cloud_score,
        }