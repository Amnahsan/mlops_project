import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta
import os

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def fetch_historical_data(latitude=59.91, longitude=10.75, start_date="2022-01-01", end_date=None):
    """
    Fetches historical weather data for training.
    Default location: Oslo, Norway.
    """
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
        "timezone": "auto"
    }
    
    print(f"Fetching historical data from {start_date} to {end_date}...")
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Process daily data
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(2).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(3).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )}
    daily_data["temperature_max"] = daily_temperature_2m_max
    daily_data["temperature_min"] = daily_temperature_2m_min
    daily_data["precipitation"] = daily_precipitation_sum
    daily_data["wind_speed"] = daily_wind_speed_10m_max

    df = pd.DataFrame(data=daily_data)
    
    # Save to raw
    os.makedirs("data/raw", exist_ok=True)
    output_path = f"data/raw/historical_weather_{end_date}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved historical data to {output_path}")
    return df

def fetch_current_data(latitude=59.91, longitude=10.75):
    """
    Fetches recent data (past few days) to run inference for 'tomorrow'.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "past_days": 7,
        "forecast_days": 1, # We only need recent history to predict tomorrow (if using autoregressive)
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
        "timezone": "auto"
    }
    
    print("Fetching current/recent data...")
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(2).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(3).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )}
    daily_data["temperature_max"] = daily_temperature_2m_max
    daily_data["temperature_min"] = daily_temperature_2m_min
    daily_data["precipitation"] = daily_precipitation_sum
    daily_data["wind_speed"] = daily_wind_speed_10m_max

    df = pd.DataFrame(data=daily_data)
    
    os.makedirs("data/raw", exist_ok=True)
    output_path = f"data/raw/current_weather_{datetime.now().strftime('%Y-%m-%d')}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved current data to {output_path}")
    return df

if __name__ == "__main__":
    # Test the fetcher
    fetch_historical_data()
    fetch_current_data()
