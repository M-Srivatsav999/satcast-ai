# 🛰️ SatCast AI — Real-Time Rainfall Prediction Using Satellite Imagery & Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask)
![NASA](https://img.shields.io/badge/NASA_GIBS-Live_Satellite-0B3D91?style=for-the-badge&logo=nasa)
![Deploy](https://img.shields.io/badge/Deployed-Render.com-46E3B7?style=for-the-badge)

**CNN + Bidirectional LSTM + Multi-Head Attention · 5-Horizon Forecast · MC-Dropout Uncertainty · Zero API Cost**

[🌐 Live Demo](https://satcast-ai.onrender.com) · [🚀 Quick Start](#quick-start) · [👥 Team](#team)

</div>

---

## 📌 Overview

**SatCast AI** is a deep learning-based real-time rainfall prediction system that uses live NASA satellite imagery to forecast rainfall intensity at **5 simultaneous horizons** (1h, 3h, 6h, 12h, 24h) — with zero API cost and sub-second inference.

Unlike traditional Numerical Weather Prediction (NWP) systems that require government-scale supercomputing infrastructure, SatCast AI runs on a standard laptop or free cloud hosting while achieving **R² ≈ 0.75+** on the test set.

```
Input: 5 consecutive satellite frames (128×128×3)
  ↓
ResNet-4 CNN  →  BiLSTM  →  Multi-Head Attention
  ↓
Output: Rainfall mm/h × 5 horizons + Severity Class + Confidence Interval
```

---

## ✨ Features

| Feature | Details |
|---|---|
| 🛰️ **Live NASA Satellite Data** | MODIS Terra/Aqua, GOES-East ABI, GPM IMERG — 7 layers, free, no API key |
| 🧠 **Deep Learning Architecture** | ResNet-4 CNN + Bidirectional LSTM + 4-Head Self-Attention |
| ⚡ **5-Horizon Forecast** | Simultaneous predictions at 1h, 3h, 6h, 12h, 24h |
| 🎯 **Uncertainty Quantification** | Monte Carlo Dropout — 20 stochastic passes → mean ± std |
| ⚠️ **IMD Severity Classification** | No Rain / Light / Moderate / Heavy / Extreme |
| 🌡️ **Live Weather Context** | Open-Meteo API — current + 24h hourly forecast |
| 📊 **Interactive Dashboard** | Chart.js charts, radar animations, atmospheric dark theme |
| 📥 **JSON Report Download** | Full prediction results exportable as JSON |
| 🔄 **Dual Prediction Mode** | Physics scaling for uploads · Pure model for NASA live |
| 0₹ **Zero Cost** | All APIs free · Deployable on Render free tier |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                          │
│  NASA GIBS WMS  ──►  5 × PNG Frames (128×128×3)       │
│  User Upload    ──►  Physics Cloud Signal Extraction   │
└────────────────────────┬────────────────────────────────┘
                         │ Preprocess: Resize → Normalize → Stack
                         ▼
┌─────────────────────────────────────────────────────────┐
│           ResNet-4 CNN  (per frame, shared weights)     │
│  Stage 1: Conv(3→32)   + ResBlock  →  64×64×32         │
│  Stage 2: Conv(32→64)  + ResBlock  →  32×32×64         │
│  Stage 3: Conv(64→128) + ResBlock  →  16×16×128        │
│  Stage 4: Conv(128→256)+ ResBlock  →   8×8×256         │
│  GlobalAvgPool + Linear + LayerNorm  →  256-dim vector  │
└────────────────────────┬────────────────────────────────┘
                         │ 5 frame vectors → sequence
                         ▼
┌─────────────────────────────────────────────────────────┐
│         Bidirectional LSTM  (2 layers, 128 units)       │
│  Forward:  frame₁ → frame₂ → frame₃ → frame₄ → frame₅ │
│  Backward: frame₅ → frame₄ → frame₃ → frame₂ → frame₁ │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│         Multi-Head Self-Attention  (4 heads)            │
│  Learns which frames are most rain-relevant             │
│  Output: 256-dim context vector                         │
└──────────────┬──────────────────────┬───────────────────┘
               ▼                      ▼
    ┌─────────────────┐    ┌──────────────────────┐
    │ Regression Head │    │   Classifier Head    │
    │ → 5 × mm/h      │    │   → 5 class probs    │
    └─────────────────┘    └──────────────────────┘
               │
               ▼
    MC-Dropout: 20 passes → mean ± std (confidence interval)
```

---

## 📊 Model Performance

| Horizon | R² Score | MAE (mm/h) |
|---------|----------|------------|
| 1 hour  | ~0.78    | ~1.2       |
| 3 hours | ~0.75    | ~1.4       |
| 6 hours | ~0.71    | ~1.7       |
| 12 hours| ~0.65    | ~2.1       |
| 24 hours| ~0.58    | ~2.6       |

**vs Baseline:** R² improved from **≈ -0.000 → 0.75+**

---

## 🛰️ Satellite Layers

| Layer | Source | Update Frequency |
|---|---|---|
| True Color (MODIS Terra) | NASA | Daily |
| True Color (MODIS Aqua) | NASA | Daily |
| IR Thermal (GOES-East) | NOAA | Every 10 min |
| GPM Precipitation Rate | NASA GPM | Every 30 min |
| Cloud Top Temperature | NASA | Daily |
| Water Vapor (Mid-Level) | NOAA | Every 10 min |
| Snow & Ice Cover | NASA | Daily |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Internet connection (for NASA GIBS and Open-Meteo APIs)

### Installation

```bash
# Clone the repository
git clone https://github.com/M-Srivatsav999/satcast-ai.git
cd satcast-ai

# Install dependencies
pip install -r requirements.txt

# Train the model (~3 minutes on CPU)
python train.py

# Launch the application
python app.py
```

Open your browser at **http://localhost:5000** 🎉

---

## 📁 Project Structure

```
satcast-ai/
├── app.py                  # Flask server + REST API endpoints
├── models.py               # ResNet-4 CNN + BiLSTM + Attention
├── predict.py              # MC-Dropout inference engine
├── train.py                # Training pipeline
├── data_preprocessing.py   # Physics data + Open-Meteo API
├── satellite_fetch.py      # NASA GIBS WMS client
├── config.py               # Hyperparameters + API config
├── cnn_lstm_model.pth      # Trained model weights
├── scaler.pkl              # Fitted MinMaxScaler
├── requirements.txt
├── Procfile                # gunicorn deployment
├── templates/index.html    # SatCast AI dashboard
└── static/
    ├── css/main.css        # Atmospheric dark theme
    └── js/main.js          # Charts + API calls + animations
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Main dashboard |
| `POST` | `/api/fetch-satellite` | Fetch live NASA GIBS frames |
| `POST` | `/api/predict` | Run rainfall prediction |
| `POST` | `/api/weather` | Fetch live weather data |
| `POST` | `/api/satellite-view` | Multi-layer comparison |
| `GET`  | `/api/locations` | Available regions and layers |

---

## ☁️ Deployment

### Local
```bash
python app.py
# → http://localhost:5000
```

### Render.com (Free Cloud)
```
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
```
Live at: **https://satcast-ai.onrender.com**

> ⚠️ Free tier spins down after 15 min of inactivity. First request takes ~30s to wake up.

---

## 🧪 Tech Stack

| Category | Technology |
|---|---|
| Deep Learning | PyTorch 2.x |
| Web Framework | Flask 3.x |
| Satellite Data | NASA GIBS WMS (free, no key) |
| Weather Data | Open-Meteo API (free, no key) |
| Image Processing | Pillow (PIL) |
| ML Utilities | NumPy · scikit-learn |
| Frontend | Chart.js 4.4 · HTML5 · CSS3 |
| Production Server | gunicorn |
| Cloud Hosting | Render.com |

---

## 👥 Team

**Department of Computer Science and Engineering (AI&ML)**  
**Sphoorthy Engineering College, Hyderabad — Mini Project 2025-26**

| Name | Roll Number | Role |
|---|---|---|
| **M. Srivatsav** | 23N81A66F9 | Project Lead · ML Architecture · NASA GIBS Integration |
| **A. Saisree** | 23N81A66D3 | Data Pipeline · Physics Data · Open-Meteo API |
| **M. Vaishnavi** | 23N81A66E6 | Training Pipeline · Loss Functions · Evaluation |
| **K. Venkatesh** | 23N81A66F7 | Frontend Dashboard · Flask API · Deployment |

**Guide:** Mr. M. Ramesh, Assistant Professor, Dept. of CSE (AI&ML)

---

##  Acknowledgements

- **NASA GIBS** — free open-access satellite imagery via Global Imagery Browse Services
- **Open-Meteo** — free weather forecast API with no registration required
- **PyTorch** — deep learning framework
- **Render.com** — free cloud deployment platform

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

Made with ❤️ by **Team SatCast AI** · Sphoorthy Engineering College · 2025-26

⭐ **Star this repo if you found it useful!**

</div>
