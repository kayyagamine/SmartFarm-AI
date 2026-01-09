# /mnt/data/crop_reco_api.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pickle
import requests
import numpy as np
import pandas as pd
import os

router = APIRouter(tags=["Crop Recommendation"])

# ----------------------------
# CONFIG: update path if needed
# ----------------------------
import os
from pathlib import Path
import joblib  # or pickle, whatever you're using

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "crop_reco.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

crop_model = joblib.load(MODEL_PATH)  # or your actual load call

OPENWEATHER_KEY = "988abbce6c327b33eb3033e64cb6f8f8"  # replace if you want

# ----------------------------
# LOAD MODEL BUNDLE
# ----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

# support both naming conventions used earlier
pipeline = bundle.get("pipeline")           # preferred (already includes preprocessor)
preprocessor = bundle.get("preprocessor")  # fallback
clf = bundle.get("clf")
le = bundle.get("label_encoder")
numeric_features = bundle.get("numeric_features", [])
categorical_features = bundle.get("categorical_features", [])
variety_lookup = bundle.get("variety_lookup", {})

if le is None:
    raise RuntimeError("Label encoder not found in model bundle; cannot decode predictions.")

# ----------------------------
# HELPERS: remote data fetchers
# ----------------------------
def safe_get(url, timeout=20):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def get_nasa_climate(lat, lon):
    try:
        url = (
            f"https://power.larc.nasa.gov/api/temporal/climatology/point?"
            f"parameters=T2M,RH2M,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&format=JSON"
        )
        d = safe_get(url)
        if not d:
            return {"Avg_Temp": None, "Rainfall": None, "Humidity": None}
        p = d["properties"]["parameter"]
        return {
            "Avg_Temp": round(p.get("T2M", {}).get("ANN", None), 2) if p.get("T2M") else None,
            "Rainfall": round(p.get("PRECTOTCORR", {}).get("ANN", None), 2) if p.get("PRECTOTCORR") else None,
            "Humidity": round(p.get("RH2M", {}).get("ANN", None), 2) if p.get("RH2M") else None,
        }
    except Exception:
        return {"Avg_Temp": None, "Rainfall": None, "Humidity": None}

def get_openmeteo_soil(lat, lon):
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&hourly=soil_temperature_0cm,soil_moisture_0_to_1cm"
        )
        d = safe_get(url)
        if not d or "hourly" not in d:
            return {"Soil_pH": None, "Organic_Carbon": None}
        temps = d["hourly"].get("soil_temperature_0cm", [])
        moist = d["hourly"].get("soil_moisture_0_to_1cm", [])
        if temps and moist:
            soil_temp = sum(temps) / len(temps)
            soil_moist = sum(moist) / len(moist)
            soil_ph = round(6.5 + (soil_moist - 0.2) * 2, 2)
            org_c = round(10 + (soil_moist * 10), 2)
            return {"Soil_pH": soil_ph, "Organic_Carbon": org_c}
        return {"Soil_pH": None, "Organic_Carbon": None}
    except Exception:
        return {"Soil_pH": None, "Organic_Carbon": None}

def get_elevation(lat, lon):
    try:
        d = safe_get(f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}")
        if not d:
            return None
        return round(d["results"][0].get("elevation", None), 2)
    except Exception:
        return None

def get_live_weather(lat, lon, api_key):
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather?"
            f"lat={lat}&lon={lon}&appid={api_key}&units=metric"
        )
        d = safe_get(url)
        if not d or "main" not in d:
            return None
        return round(d["main"].get("temp", None), 2)
    except Exception:
        return None

# ----------------------------
# Build feature vector for the model
# ----------------------------
def build_row_from_env(env, season: Optional[str] = None):
    # env keys we may have: Avg_Temp, Rainfall, Humidity, Soil_pH, Organic_Carbon, Elevation, Temp_Now
    # Map to model numeric_features and categorical_features
    # Use sensible fallbacks where available
    row = {}
    for f in numeric_features:
        if f.lower() in ["temperature", "temp", "avg_temp", "avg_temp_c"]:
            # prefer Avg_Temp -> Temp_Now -> None
            val = env.get("Avg_Temp") if env.get("Avg_Temp") is not None else env.get("Temp_Now")
            row[f] = float(val) if val is not None else np.nan
        elif f.lower() in ["rainfall", "rain", "precip", "annual_rainfall"]:
            row[f] = float(env.get("Rainfall")) if env.get("Rainfall") is not None else np.nan
        elif f.lower() in ["humidity", "rel_humidity"]:
            row[f] = float(env.get("Humidity")) if env.get("Humidity") is not None else np.nan
        elif f.lower() in ["soil_ph", "ph"]:
            row[f] = float(env.get("Soil_pH")) if env.get("Soil_pH") is not None else np.nan
        elif f.lower() in ["organic_carbon", "org_c", "organiccarbon"]:
            row[f] = float(env.get("Organic_Carbon")) if env.get("Organic_Carbon") is not None else np.nan
        elif f.lower() in ["elevation", "elev"]:
            row[f] = float(env.get("Elevation")) if env.get("Elevation") is not None else np.nan
        else:
            # if the feature exists in env dictionary, use it; otherwise NaN
            if f in env:
                try:
                    row[f] = float(env[f]) if env[f] is not None else np.nan
                except:
                    row[f] = np.nan
            else:
                row[f] = np.nan

    # categorical features
    for c in categorical_features:
        if c.lower() == "season":
            row[c] = season if season is not None else "missing"
        else:
            # no district inference from lat/lon here (could add reverse-geocode if desired)
            row[c] = "unknown"

    return pd.DataFrame([row], columns=(numeric_features + categorical_features))

# ----------------------------
# API request/response models
# ----------------------------
class PredictInput(BaseModel):
    lat: float
    lon: float
    season: Optional[str] = None
    top_n: Optional[int] = 5

@router.post("/predict")
def predict(inp: PredictInput):
    lat = inp.lat; lon = inp.lon; season = inp.season; top_n = int(inp.top_n or 5)

    # 1) fetch environmental data
    nasa = get_nasa_climate(lat, lon)
    soil = get_openmeteo_soil(lat, lon)
    elev = get_elevation(lat, lon)
    temp_now = get_live_weather(lat, lon, OPENWEATHER_KEY)

    env = {
        "Latitude": lat,
        "Longitude": lon,
        "Avg_Temp": nasa.get("Avg_Temp"),
        "Rainfall": nasa.get("Rainfall"),
        "Humidity": nasa.get("Humidity"),
        "Soil_pH": soil.get("Soil_pH"),
        "Organic_Carbon": soil.get("Organic_Carbon"),
        "Elevation": elev,
        "Temp_Now": temp_now,
    }

    # sanity check
    if (env["Avg_Temp"] is None and env["Temp_Now"] is None) or env["Rainfall"] is None:
        # still proceed but warn user
        warning = "Some environment values missing (Avg_Temp or Rainfall). Predictions may be less accurate."
    else:
        warning = None

    # 2) build feature vector compatible with model
    Xrow = build_row_from_env(env, season=season)

    # 3) transform & predict
    try:
        if pipeline is not None:
            probs = pipeline.predict_proba(Xrow)
        else:
            # use preprocessor + clf
            Xproc = preprocessor.transform(Xrow)
            probs = clf.predict_proba(Xproc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    probs = np.asarray(probs)[0]
    idxs = np.argsort(probs)[::-1][:top_n]
    preds = []
    for i in idxs:
        try:
            crop_name = le.inverse_transform([i])[0]
        except Exception:
            # if label encoder uses strings of classes (not indices)
            crop_name = str(i)
        preds.append({
            "crop": crop_name,
            "probability": float(probs[i]),
            "top_varieties": variety_lookup.get(crop_name, [])
        })

    return {
        "input": {"lat": lat, "lon": lon, "season": season},
        "environment": env,
        "warning": warning,
        "predictions": preds
    }

@router.get("/")
def root():
    return {"message": "Crop recommendation API (lat/lon) running."}
