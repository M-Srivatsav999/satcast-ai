# config.py - Enhanced Configuration
import torch

# ── Model Architecture ──────────────────────────────────────────────
IMG_SIZE = (128, 128)          # Higher res for real satellite imagery
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 0.0005
SEQ_LENGTH = 5                 # 5-frame temporal context (was 3)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CNN_FILTERS = [32, 64, 128, 256]   # Deeper CNN (was 3 layers)
LSTM_UNITS = 128               # Larger LSTM (was 32)
ATTENTION_HEADS = 4            # NEW: Multi-head attention
DROPOUT = 0.3
TRAIN_SPLIT = 0.8

# ── Satellite Data Sources (FREE, no API key) ────────────────────────
# NASA GIBS – real-time MODIS/VIIRS cloud imagery
NASA_GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?"
GIBS_LAYER_CLOUD   = "MODIS_Terra_Cloud_Top_Temp_Day"
GIBS_LAYER_TRUE    = "MODIS_Terra_CorrectedReflectance_TrueColor"
GIBS_LAYER_PRECIP  = "IMERG_Precipitation_Rate"        # GPM near-real-time rain
GIBS_LAYER_IR      = "GOES-East_ABI_Band13_Clean_Infrared"  # IR thermal

# Open-Meteo – free weather API, no key needed
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# ── Prediction Settings ──────────────────────────────────────────────
FORECAST_HOURS = [1, 3, 6, 12, 24]   # Multi-horizon output
RAIN_CATEGORIES = {
    "No Rain":   (0,    1),
    "Light":     (1,    5),
    "Moderate":  (5,    15),
    "Heavy":     (15,   35),
    "Extreme":   (35,   9999),
}
SEVERITY_COLORS = {
    "No Rain":  "#4CAF50",
    "Light":    "#8BC34A",
    "Moderate": "#FFC107",
    "Heavy":    "#FF5722",
    "Extreme":  "#9C27B0",
}
