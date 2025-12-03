from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator

# Import project modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.data.fetcher import fetch_current_data
from src.data.processor import process_data

app = FastAPI(title="Weather RPS API", version="1.0.0")

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, specify the frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument Prometheus
Instrumentator().instrument(app).expose(app)

# Load Model
model_path = "models/weather_model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    else:
        print("WARNING: Model not found. API will fail on predict.")

class PredictionRequest(BaseModel):
    latitude: float = 59.91
    longitude: float = 10.75

@app.get("/")
def read_root():
    return {"message": "Weather RPS API is running!"}

@app.post("/predict")
def predict(request: PredictionRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # 1. Fetch Data (Forecast)
    # We need to predict for TOMORROW.
    # fetch_current_data fetches past 7 days + 1 forecast day.
    try:
        df_raw = fetch_current_data(request.latitude, request.longitude)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch weather data: {str(e)}")

    # 2. Process Data
    # We need to process it to get features for the LAST row (which corresponds to "tomorrow" or "today" depending on how we want to predict)
    # Our model was trained to predict 'temperature_max' using LAGGED features from the PREVIOUS day.
    # So if we want to predict Tomorrow's Max Temp, we need Today's data (Temp, Precip, etc.)
    
    # Save to temp file for processor (reusing existing logic)
    temp_raw_path = "data/raw/temp_inference.csv"
    os.makedirs("data/raw", exist_ok=True)
    df_raw.to_csv(temp_raw_path, index=False)
    
    temp_proc_path = "data/processed/temp_inference.csv"
    try:
        df_proc = process_data(temp_raw_path, temp_proc_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process data: {str(e)}")
        
    # Get the last row (most recent data point that has valid features)
    # Note: process_data drops NaNs.
    if df_proc.empty:
        raise HTTPException(status_code=500, detail="Processed data is empty. Not enough history?")
        
    latest_features = df_proc.iloc[[-1]]
    
    features = ['temp_max_lag_1', 'temp_min_lag_1', 'precip_lag_1', 'temp_max_roll_3', 'month', 'day_of_year']
    X = latest_features[features]
    
    # 3. Predict
    prediction = model.predict(X)[0]
    
    # 4. Heuristics (Clothing)
    clothing = suggest_clothing(prediction)
    
    return {
        "predicted_max_temp": prediction,
        "clothing_suggestion": clothing,
        "input_date": str(latest_features['date'].values[0])
    }

def suggest_clothing(temp):
    if temp > 25:
        return "T-shirt, Shorts, Sunglasses"
    elif temp > 15:
        return "Light Jacket, Jeans"
    elif temp > 5:
        return "Coat, Scarf, Sweater"
    else:
        return "Heavy Coat, Gloves, Thermal Wear"
