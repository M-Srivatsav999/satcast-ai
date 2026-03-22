# satellite_fetch.py – Real-time satellite imagery from NASA GIBS

import requests
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime, timedelta, timezone
from config import NASA_GIBS_WMS, IMG_SIZE

LAYERS = {
    "True Color (MODIS Terra)":  "MODIS_Terra_CorrectedReflectance_TrueColor",
    "True Color (MODIS Aqua)":   "MODIS_Aqua_CorrectedReflectance_TrueColor",
    "Cloud Top Temperature":     "MODIS_Terra_Cloud_Top_Temp_Day",
    "IR Thermal (GOES-East)":    "GOES-East_ABI_Band13_Clean_Infrared",
    "GPM Precipitation Rate":    "IMERG_Precipitation_Rate",
    "Water Vapor (GOES-East)":   "GOES-East_ABI_Band09_Mid-level_Water_Vapor",
    "Snow & Ice (MODIS)":        "MODIS_Terra_Snow_Cover_Daily",
}

LOCATIONS = {
    "India (Hyderabad)":     [77.0, 16.5, 80.0, 18.5],
    "India (Mumbai)":        [72.0, 18.5, 74.5, 20.5],
    "India (Chennai)":       [79.0, 12.5, 81.5, 14.5],
    "Bay of Bengal":         [80.0,  8.0, 95.0, 22.0],
    "Arabian Sea":           [55.0,  5.0, 78.0, 25.0],
    "South Asia (Full)":     [60.0,  5.0, 100.0, 40.0],
    "Southeast Asia":        [95.0, -10.0, 140.0, 30.0],
    "Western Pacific":       [120.0, -20.0, 180.0, 40.0],
    "Gulf of Mexico":        [-100.0, 15.0, -75.0, 35.0],
    "Custom (enter coords)": None,
}


def get_available_dates(n_days=7):
    today = datetime.now(timezone.utc)
    return [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]


def fetch_satellite_image(layer, bbox, date, width=512, height=512):
    params = {
        "SERVICE":     "WMS",
        "VERSION":     "1.1.1",
        "REQUEST":     "GetMap",
        "LAYERS":      layer,
        "SRS":         "EPSG:4326",
        "BBOX":        f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "WIDTH":       str(width),
        "HEIGHT":      str(height),
        "FORMAT":      "image/png",
        "TIME":        date,
        "TRANSPARENT": "true",
    }
    try:
        resp = requests.get(NASA_GIBS_WMS, params=params, timeout=15)
        if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
            return Image.open(BytesIO(resp.content)).convert("RGB")
        return None
    except Exception:
        return None


def fetch_sequence(layer, bbox, n_frames=5, base_date=None):
    dates = get_available_dates(n_days=n_frames + 5)
    images, fetched_dates = [], []

    for d in dates:
        if len(images) >= n_frames:
            break
        img = fetch_satellite_image(layer, bbox, d)
        if img is not None:
            images.append(img)
            fetched_dates.append(d)

    while len(images) < n_frames and images:
        images.append(images[-1])
        fetched_dates.append(fetched_dates[-1])

    images.reverse()
    fetched_dates.reverse()
    return images, fetched_dates


def images_to_tensor_array(images):
    target = IMG_SIZE
    arr = []
    for img in images:
        img = img.resize(target, Image.LANCZOS)
        a = np.array(img).astype(np.float32) / 255.0
        a = np.transpose(a, (2, 0, 1))
        arr.append(a)
    return np.array(arr)


def extract_cloud_features(image):
    arr = np.array(image.resize((256, 256))).astype(np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    cloud_cover         = float(np.mean(b + g) / 2)
    cold_cloud_fraction = float(np.mean((r > 0.8) & (g > 0.8) & (b > 0.8)))
    moisture_index      = float(np.mean(b) - np.mean(r))
    brightness          = (r + g + b) / 3
    brightness_variance = float(np.var(brightness))
    ndvi_proxy          = float(np.mean((g - r) / (g + r + 1e-6)))

    return {
        "cloud_cover":          round(cloud_cover, 4),
        "cold_cloud_fraction":  round(cold_cloud_fraction, 4),
        "moisture_index":       round(moisture_index, 4),
        "brightness_variance":  round(brightness_variance, 4),
        "ndvi_proxy":           round(ndvi_proxy, 4),
    }