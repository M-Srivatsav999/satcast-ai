# SatCast AI — Flask Web Application

## Run Locally

```bash
# 1. Copy these files to your APP folder alongside models.py, predict.py etc.
# 2. Install dependencies
pip install flask gunicorn

# 3. Train model if not already done
python train.py

# 4. Run Flask server
python app.py

# 5. Open browser
http://localhost:5000
```

## Deploy on Render (Free)

1. Push to GitHub
2. Go to render.com → New Web Service
3. Connect your GitHub repo
4. Settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
5. Deploy ✅

## Deploy on Railway (Free)

1. Go to railway.app
2. New Project → Deploy from GitHub
3. Select your repo → Deploy
4. Railway auto-detects Procfile ✅

## File Structure Required

```
your-repo/
├── app.py                  ← Flask server (this file)
├── models.py               ← Deep learning model
├── predict.py              ← Inference
├── train.py                ← Training
├── config.py               ← Settings
├── data_preprocessing.py   ← Data pipeline
├── satellite_fetch.py      ← NASA GIBS
├── cnn_lstm_model.pth      ← Trained weights
├── scaler.pkl              ← Fitted scaler
├── requirements.txt
├── Procfile
├── templates/
│   └── index.html
└── static/
    ├── css/main.css
    └── js/main.js
```

## Developer

**Mukka Srivatsav and Team**  
Dept. of Computer Science & Engineering  
Mini Project 2025-26
