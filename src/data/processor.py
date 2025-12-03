import pandas as pd
import numpy as np
import os
from ydata_profiling import ProfileReport

def check_quality(df):
    """
    Performs data quality checks.
    Returns True if passed, False otherwise.
    """
    print("Running Data Quality Checks...")
    
    # Check for nulls
    null_counts = df.isnull().sum()
    if null_counts.any():
        print("WARNING: Null values found!")
        print(null_counts)
        # For now, we might just warn, or fail if critical
        # In a strict pipeline, we would return False
    
    # Check schema
    required_cols = ['date', 'temperature_max', 'temperature_min', 'precipitation', 'wind_speed']
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: Missing columns. Expected {required_cols}")
        return False

    # Check value ranges (Sanity check)
    if (df['temperature_max'] > 60).any() or (df['temperature_max'] < -60).any():
        print("ERROR: Temperature out of realistic range!")
        return False
        
    print("Data Quality Checks Passed.")
    return True

def generate_profiling_report(df, output_path, title="Data Quality Report"):
    """
    Generates a comprehensive data profiling report using ydata-profiling.
    
    Args:
        df: pandas DataFrame to profile
        output_path: Path where the HTML report will be saved
        title: Title for the report
    
    Returns:
        str: Path to the generated report
    """
    print(f"Generating profiling report: {title}...")
    
    try:
        # Create reports directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate profile report
        profile = ProfileReport(
            df,
            title=title,
            explorative=True,
            minimal=False
        )
        
        # Save to HTML
        profile.to_file(output_path)
        print(f"✓ Profiling report saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"WARNING: Failed to generate profiling report: {str(e)}")
        return None

def process_data(input_path, output_path, generate_reports=True):
    """
    Loads raw data, performs feature engineering, and saves processed data.
    Optionally generates profiling reports.
    
    Args:
        input_path: Path to raw data CSV
        output_path: Path where processed data will be saved
        generate_reports: Whether to generate profiling reports (default: True)
    
    Returns:
        tuple: (processed_df, raw_report_path, processed_report_path)
    """
    print(f"Processing {input_path}...")
    df = pd.read_csv(input_path)
    
    # Generate profiling report for raw data
    raw_report_path = None
    if generate_reports:
        raw_report_path = generate_profiling_report(
            df,
            "reports/raw_data_profile.html",
            title="Raw Weather Data - Quality Report"
        )
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Feature Engineering
    # 1. Lags (Yesterday's weather to predict today/tomorrow)
    # We want to predict 'target' (e.g., tomorrow's max temp) using 'features' (today's max temp)
    # But for the dataset, we just create features.
    
    df['temp_max_lag_1'] = df['temperature_max'].shift(1)
    df['temp_min_lag_1'] = df['temperature_min'].shift(1)
    df['precip_lag_1'] = df['precipitation'].shift(1)
    
    # 2. Rolling Means (3-day average)
    df['temp_max_roll_3'] = df['temperature_max'].rolling(window=3).mean()
    
    # 3. Date Features
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Drop initial rows with NaNs due to shifting
    df = df.dropna()
    
    # Generate profiling report for processed data
    processed_report_path = None
    if generate_reports:
        processed_report_path = generate_profiling_report(
            df,
            "reports/processed_data_profile.html",
            title="Processed Weather Data - Quality Report"
        )
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    
    return df, raw_report_path, processed_report_path

if __name__ == "__main__":
    # Test processor on the most recent raw file
    raw_dir = "data/raw"
    if os.path.exists(raw_dir):
        files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
        if files:
            latest_file = max([os.path.join(raw_dir, f) for f in files], key=os.path.getctime)
            df = pd.read_csv(latest_file)
            if check_quality(df):
                process_data(latest_file, "data/processed/train_data.csv")
