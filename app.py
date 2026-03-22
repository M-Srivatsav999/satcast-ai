# app.py – SatCast AI Flask Server
# Developer: Mukka Srivatsav and Team | CSE-AIML Mini Project 2025-26

import os, sys, json
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from satellite_fetch import fetch_sequence, LAYERS, LOCATIONS, get_available_dates, fetch_satellite_image, extract_cloud_features
from data_preprocessing import fetch_weather_metadata
from config import FORECAST_HOURS, RAIN_CATEGORIES, SEVERITY_COLORS
from predict import predict_rainfall
from PIL import Image
from io import BytesIO
import base64, io

app = Flask(__name__)

# ── Helper: PIL Image → base64 string ────────────────────────────────
def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ── Helper: base64 → PIL Image ────────────────────────────────────────
def b64_to_pil(b64str: str) -> Image.Image:
    data = base64.b64decode(b64str.split(",")[-1])
    return Image.open(BytesIO(data)).convert("RGB")

# ══════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    locations = list(LOCATIONS.keys())
    layers    = list(LAYERS.keys())
    dates     = get_available_dates(7)
    return render_template("index.html",
                           locations=locations,
                           layers=layers,
                           dates=dates,
                           now=datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"))

@app.route("/api/fetch-satellite", methods=["POST"])
def api_fetch_satellite():
    try:
        data        = request.json
        location    = data.get("location", "India (Hyderabad)")
        layer_name  = data.get("layer", "True Color (MODIS Aqua)")
        n_frames    = int(data.get("n_frames", 5))
        custom_bbox = data.get("bbox", None)

        bbox     = custom_bbox if custom_bbox else LOCATIONS.get(location)
        layer_id = LAYERS.get(layer_name)

        if not bbox or not layer_id:
            return jsonify({"error": "Invalid location or layer"}), 400

        images, dates_fetched = fetch_sequence(layer_id, bbox, n_frames=n_frames)

        if not images:
            return jsonify({"error": "Could not fetch satellite imagery from NASA GIBS"}), 503

        frames = []
        for img, d in zip(images, dates_fetched):
            feats = extract_cloud_features(img)
            frames.append({
                "date":     d,
                "b64":      pil_to_b64(img.resize((384, 384))),
                "cloud":    round(feats["cloud_cover"], 3),
                "cold":     round(feats["cold_cloud_fraction"], 3),
                "moisture": round(feats["moisture_index"], 3),
            })

        return jsonify({"frames": frames, "count": len(frames)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data       = request.json
        mode       = data.get("mode", "upload")
        frames_b64 = data.get("frames", [])
        location   = data.get("location", "India (Hyderabad)")

        if not frames_b64:
            return jsonify({"error": "No image frames provided"}), 400

        pil_images  = [b64_to_pil(f) for f in frames_b64]
        use_physics = (mode == "upload")
        result      = predict_rainfall(pil_images, use_physics=use_physics)

        return jsonify({
            "primary_mm_h":   result["primary_mm_h"],
            "primary_cat":    result["primary_cat"],
            "primary_color":  result["primary_color"],
            "confidence_pct": result["confidence_pct"],
            "horizons":       result["horizons"],
            "cloud_score":    result.get("cloud_score"),
            "model_version":  result["model_version"],
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "location":       location,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/weather", methods=["POST"])
def api_weather():
    try:
        data = request.json
        lat  = float(data.get("lat", 17.4))
        lon  = float(data.get("lon", 78.5))
        wx   = fetch_weather_metadata(lat, lon)
        return jsonify(wx)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/satellite-view", methods=["POST"])
def api_satellite_view():
    try:
        data        = request.json
        layer_names = data.get("layers", ["True Color (MODIS Terra)"])
        location    = data.get("location", "India (Hyderabad)")
        date        = data.get("date", get_available_dates(1)[0])
        custom_bbox = data.get("bbox", None)
        resolution  = int(data.get("resolution", 384))

        bbox = custom_bbox if custom_bbox else LOCATIONS.get(location)
        if not bbox:
            return jsonify({"error": "Invalid location"}), 400

        results = []
        for lname in layer_names:
            lid = LAYERS.get(lname)
            if not lid:
                continue
            img = fetch_satellite_image(lid, bbox, date, width=resolution, height=resolution)
            if img:
                feats = extract_cloud_features(img)
                results.append({
                    "layer":    lname,
                    "b64":      pil_to_b64(img),
                    "cloud":    round(feats["cloud_cover"], 3),
                    "cold":     round(feats["cold_cloud_fraction"], 3),
                    "moisture": round(feats["moisture_index"], 3),
                    "texture":  round(feats["brightness_variance"], 4),
                })
            else:
                results.append({"layer": lname, "error": f"Unavailable for {date}"})

        return jsonify({"layers": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/locations")
def api_locations():
    return jsonify({
        "locations": list(LOCATIONS.keys()),
        "layers":    list(LAYERS.keys()),
        "dates":     get_available_dates(7),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)