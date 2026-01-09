"""
fertilizer_recommendation_api.py

Full FastAPI application that:
- Loads your 1000-row Karnataka dataset (CSV path auto-filled)
- Accepts POST /recommend with crop_name, disease_name, latitude, longitude
- Matches crop + disease robustly (exact, partial, reverse, fuzzy)
- Returns fertilizer recommendation, NPK, application method, frequency
- Looks up nearby fertilizer stores via Foursquare (keys auto-filled)
- Returns online purchase links

Run with:
    uvicorn fertilizer_recommendation_api:app --reload

NOTE: You asked for the CSV path and Foursquare keys to be auto-filled; both are included below.
      Normally it's better to use environment variables for credentials.
"""

from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
import requests
import logging
import urllib.parse
from difflib import get_close_matches
import os
# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fertilizer_recommendation_api")

# -------------------------
# Config (AUTO-FILLED by user request)
# -------------------------
from pathlib import Path

BASE_DIR = Path(__file__).parent
CSV_PATH = os.getenv("DISEASE_DATASET_PATH")
IPSTACK_API_KEY = os.getenv("IPSTACK_API_KEY")
FOURSQUARE_CLIENT_ID = os.getenv("FOURSQUARE_CLIENT_ID")
FOURSQUARE_CLIENT_SECRET = os.getenv("FOURSQUARE_CLIENT_SECRET")

if not all([IPSTACK_API_KEY, FOURSQUARE_CLIENT_ID, FOURSQUARE_CLIENT_SECRET]):
    raise RuntimeError("Missing required API keys. Check environment variables.")
FOURSQUARE_VERSION = "20230101"
FOURSQUARE_RADIUS = 10000
FOURSQUARE_LIMIT = 5
REQUEST_TIMEOUT = 10.0

# -------------------------
# FastAPI app & router
# -------------------------
app = FastAPI(title="Fertilizer & Pesticide Recommendation API")
router = APIRouter(prefix="/fert", tags=["fertilizer-recommend"])

# -------------------------
# Data (lazy loaded)
# -------------------------
_data: Optional[pd.DataFrame] = None
_unique_diseases: Optional[List[str]] = None


def load_data() -> pd.DataFrame:
    """
    Lazy-load the CSV into a module-level DataFrame.
    """
    global _data, _unique_diseases
    if _data is not None:
        return _data

    try:
        df = pd.read_csv(CSV_PATH)
        # normalize column names
        df.columns = [c.strip() for c in df.columns]

        # ensure expected columns exist (if not, proceed but warn)
        expected_cols = {
            "Plant_Name", "Disease_Name", "Disease_Type", "Symptoms",
            "Fertilizer_Treatment", "NPK_Ratio", "Additional_Treatment",
            "Application_Method", "Frequency"
        }
        missing = expected_cols - set(df.columns)
        if missing:
            logger.warning(f"CSV is missing expected columns: {missing} — continuing with available columns.")

        # create a lowered disease name column for matching
        df["Disease_Name_Lower"] = df["Disease_Name"].astype(str).str.lower().str.strip()
        df["Plant_Name_Lower"] = df["Plant_Name"].astype(str).str.lower().str.strip()

        _data = df
        _unique_diseases = df["Disease_Name"].dropna().astype(str).unique().tolist()
        logger.info(f"Loaded dataset from: {CSV_PATH} ({len(df)} rows, {len(_unique_diseases)} unique diseases)")
    except Exception as e:
        logger.exception(f"Failed to load CSV at {CSV_PATH}: {e}")
        _data = pd.DataFrame()
        _unique_diseases = []
    return _data


# -------------------------
# Request/Response models
# -------------------------
class FertilizerRequest(BaseModel):
    crop_name: str
    disease_name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None


# -------------------------
# Matching logic
# -------------------------
def find_disease_in_crop(crop: str, disease: str) -> (pd.DataFrame, str):
    """
    Search for disease inside a crop.
    Returns (matching_dataframe, status)
    status: "ok", "crop_not_found", "not_in_crop"
    """
    df = load_data()
    if df.empty:
        return pd.DataFrame(), "no_data"

    crop_lower = str(crop).lower().strip()
    disease_lower = str(disease).lower().strip()

    # Filter rows by crop
    crop_rows = df[df["Plant_Name_Lower"] == crop_lower]
    if crop_rows.empty:
        return pd.DataFrame(), "crop_not_found"

    # 1) exact disease match within crop
    exact = crop_rows[crop_rows["Disease_Name_Lower"] == disease_lower]
    if not exact.empty:
        return exact, "ok"

    # 2) partial match (disease query in disease name)
    partial = crop_rows[crop_rows["Disease_Name_Lower"].str.contains(disease_lower, na=False, regex=False)]
    if not partial.empty:
        return partial, "ok"

    # 3) reverse match (stored disease inside query)
    for d in crop_rows["Disease_Name_Lower"].unique():
        if len(d) > 3 and d in disease_lower:
            return crop_rows[crop_rows["Disease_Name_Lower"] == d], "ok"

    # 4) fuzzy match within crop (lower cutoff since names can vary)
    close = get_close_matches(disease_lower, crop_rows["Disease_Name_Lower"].unique().tolist(), n=1, cutoff=0.5)
    if close:
        return crop_rows[crop_rows["Disease_Name_Lower"] == close[0]], "ok"

    # not found inside crop, but crop exists
    return pd.DataFrame(), "not_in_crop"


def suggest_similar_diseases(query: str, n: int = 5) -> List[str]:
    """
    Return up to n fuzzy suggestions across all diseases in the database.
    """
    df = load_data()
    if df.empty:
        return []
    suggestions = get_close_matches(str(query).lower().strip(), df["Disease_Name_Lower"].unique().tolist(), n=n, cutoff=0.3)
    results = []
    for s in suggestions:
        row = df[df["Disease_Name_Lower"] == s]
        if not row.empty:
            results.append(row["Disease_Name"].iloc[0])
    return results


# -------------------------
# Foursquare helper
# -------------------------
def find_nearby_stores(lat: Optional[float], lon: Optional[float], query_text: str = "fertilizer store") -> List[Dict]:
    """
    Call Foursquare V2 venues/search to find nearby fertilizer/agri stores.
    If credentials are not available or lat/lon missing, return [].
    """
    if lat is None or lon is None:
        logger.info("Latitude/Longitude not provided; skipping store lookup.")
        return []

    if not (FOURSQUARE_CLIENT_ID and FOURSQUARE_CLIENT_SECRET):
        logger.warning("Foursquare credentials not set — skipping nearby store lookup.")
        return []

    url = "https://api.foursquare.com/v2/venues/search"
    params = {
        "ll": f"{lat},{lon}",
        "radius": FOURSQUARE_RADIUS,
        "limit": FOURSQUARE_LIMIT,
        "query": query_text,
        "client_id": FOURSQUARE_CLIENT_ID,
        "client_secret": FOURSQUARE_CLIENT_SECRET,
        "v": FOURSQUARE_VERSION
    }
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        js = resp.json()
        venues = js.get("response", {}).get("venues", [])
        stores = []
        for v in venues:
            loc = v.get("location", {})
            address = ", ".join([part for part in loc.get("formattedAddress", []) if part])
            stores.append({
                "name": v.get("name", "Unknown"),
                "address": address,
                "distance_m": loc.get("distance")
            })
        return stores
    except Exception as e:
        logger.exception(f"Error calling Foursquare: {e}")
        return []


# -------------------------
# Endpoint
# -------------------------
@router.post("/recommend")
def get_fertilizer(req: FertilizerRequest):
    """
    Primary endpoint:
    - Validates crop exists
    - Matches disease within the crop (exact, partial, fuzzy)
    - Returns fertilizer + npk + application + frequency
    - Optionally includes nearby stores and online links
    """
    df = load_data()
    if df.empty:
        raise HTTPException(status_code=500, detail="Internal dataset not available. Check CSV path and logs.")

    crop = req.crop_name
    disease = req.disease_name
    lat = req.latitude
    lon = req.longitude

    logger.info(f"Request: crop='{crop}', disease='{disease}', lat={lat}, lon={lon}")

    match_df, status = find_disease_in_crop(crop, disease)

    if status == "no_data":
        raise HTTPException(status_code=500, detail="Dataset load failed (see server logs).")

    if status == "crop_not_found":
        # Offer top diseases sample or help
        available_crops = df["Plant_Name"].dropna().unique().tolist()
        raise HTTPException(status_code=404, detail={
            "error": f"Crop '{crop}' not found in database.",
            "available_crop_sample": available_crops[:20],
            "total_crops": len(available_crops)
        })

    if status == "not_in_crop":
        # Provide diseases available for this crop to help user pick correct one
        available = df[df["Plant_Name_Lower"] == crop.lower().strip()]["Disease_Name"].unique().tolist()
        # Also provide fuzzy suggestions across full DB
        suggestions = suggest_similar_diseases(disease, n=5)
        raise HTTPException(status_code=404, detail={
            "error": f"Disease '{disease}' not found for crop '{crop}'.",
            "available_for_crop": available,
            "suggestions_across_db": suggestions
        })

    # status == ok, match_df contains one or more rows
    row = match_df.iloc[0]
    if len(match_df) > 1:
        logger.info(f"Multiple ({len(match_df)}) matches found. Returning first match; 'matched_rows' will indicate total.")

    fertilizer = row.get("Fertilizer_Treatment", "")
    npk = row.get("NPK_Ratio", "")
    add_treat = row.get("Additional_Treatment", "")
    method = row.get("Application_Method", "")
    frequency = row.get("Frequency", "")
    plant_name = row.get("Plant_Name", "")
    disease_name = row.get("Disease_Name", "")

    # Nearby stores (may be empty if lat/lon missing or API fails)
    stores = find_nearby_stores(lat, lon)

    # Online links (URL-encoded)
    fert_query = urllib.parse.quote_plus(str(fertilizer) or "fertilizer")
    online_links = [
        f"https://www.amazon.in/s?k={fert_query}",
        f"https://www.flipkart.com/search?q={fert_query}",
        f"https://www.indiamart.com/search.mp?ss={fert_query}",
        f"https://krishijagran.com/search/?q={fert_query}",
        f"https://agribegri.com/search.php?search={fert_query}"
    ]

    response = {
        "crop": plant_name,
        "disease": disease_name,
        "fertilizer": fertilizer,
        "NPK_ratio": npk,
        "additional_treatment": add_treat,
        "application_method": method,
        "frequency": frequency,
        "nearby_stores": stores,
        "online_links": online_links,
        "matched_rows": len(match_df)
    }

    return response


# -------------------------
# Include router & health
# -------------------------
app.include_router(router)

@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "fertilizer_recommendation_api", "csv_path": CSV_PATH}


# -------------------------
# If running as script, launch uvicorn (optional)
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fertilizer_recommendation_api:app", host="127.0.0.1", port=8000, reload=True)
