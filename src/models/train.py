import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.config.dagshub_config import setup_dagshub
from src.data.processor import process_data

def create_visualizations(model, X_test, y_test, predictions, feature_names):
    """
    Create visualizations for model performance and feature importance.
    
    Returns:
        dict: Paths to saved visualization files
    """
    viz_paths = {}
    os.makedirs("reports/visualizations", exist_ok=True)
    
    # 1. Feature Importance
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    importance_path = "reports/visualizations/feature_importance.png"
    plt.savefig(importance_path, dpi=150, bbox_inches='tight')
    plt.close()
    viz_paths['feature_importance'] = importance_path
    print(f"✓ Feature importance plot saved to {importance_path}")
    
    # 2. Predictions vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.title('Predictions vs Actual Values')
    plt.tight_layout()
    
    pred_path = "reports/visualizations/predictions_vs_actual.png"
    plt.savefig(pred_path, dpi=150, bbox_inches='tight')
    plt.close()
    viz_paths['predictions_vs_actual'] = pred_path
    print(f"✓ Predictions vs actual plot saved to {pred_path}")
    
    # 3. Residuals Distribution
    plt.figure(figsize=(10, 6))
    residuals = y_test - predictions
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.xlabel('Residual (°C)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.tight_layout()
    
    residuals_path = "reports/visualizations/residuals_distribution.png"
    plt.savefig(residuals_path, dpi=150, bbox_inches='tight')
    plt.close()
    viz_paths['residuals'] = residuals_path
    print(f"✓ Residuals distribution plot saved to {residuals_path}")
    
    return viz_paths

def train_model(data_path):
    """
    Train the weather prediction model with full MLflow tracking and artifact logging.
    """
    print("=" * 60)
    print("WEATHER RPS MODEL TRAINING")
    print("=" * 60)
    
    # Setup DagsHub MLflow tracking
    print("\n[1/6] Setting up DagsHub MLflow tracking...")
    setup_dagshub()
    
    # Process data and generate profiling reports
    print("\n[2/6] Processing data and generating profiling reports...")
    raw_data_path = "data/raw"
    if os.path.exists(raw_data_path):
        files = [f for f in os.listdir(raw_data_path) if f.endswith(".csv")]
        if files:
            latest_raw = max([os.path.join(raw_data_path, f) for f in files], key=os.path.getctime)
            df, raw_report_path, processed_report_path = process_data(
                latest_raw, 
                data_path,
                generate_reports=True
            )
            print(f"✓ Data processing complete")
        else:
            print("WARNING: No raw data files found. Skipping profiling report generation.")
            raw_report_path = None
            processed_report_path = None
    else:
        raw_report_path = None
        processed_report_path = None
    
    # Load processed data
    print(f"\n[3/6] Loading processed data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Define features and target
    features = ['temp_max_lag_1', 'temp_min_lag_1', 'precip_lag_1', 'temp_max_roll_3', 'month', 'day_of_year']
    target = 'temperature_max'
    
    X = df[features]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✓ Data split: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # MLflow Tracking
    print("\n[4/6] Starting MLflow experiment...")
    mlflow.set_experiment("weather_rps_experiment")
    
    with mlflow.start_run() as run:
        print(f"✓ MLflow Run ID: {run.info.run_id}")
        
        # Model parameters
        n_estimators = 100
        max_depth = 10
        
        # Train model
        print(f"\n[5/6] Training RandomForest (n_estimators={n_estimators}, max_depth={max_depth})...")
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        print("✓ Model training complete")
        
        # Evaluate
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"\n📊 Model Performance Metrics:")
        print(f"   RMSE: {rmse:.4f}°C")
        print(f"   MAE:  {mae:.4f}°C")
        print(f"   R²:   {r2:.4f}")
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("n_samples", len(df))
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mse", mse)
        
        # Log dataset statistics
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        mlflow.log_metric("target_mean", float(y.mean()))
        mlflow.log_metric("target_std", float(y.std()))
        
        # Create and log visualizations
        print("\n[6/6] Creating visualizations...")
        viz_paths = create_visualizations(model, X_test, y_test, predictions, features)
        
        # Log artifacts
        print("\n📦 Logging artifacts to MLflow...")
        
        # Log visualizations
        for viz_name, viz_path in viz_paths.items():
            if os.path.exists(viz_path):
                mlflow.log_artifact(viz_path, artifact_path="visualizations")
                print(f"   ✓ Logged {viz_name}")
        
        # Log profiling reports
        if raw_report_path and os.path.exists(raw_report_path):
            mlflow.log_artifact(raw_report_path, artifact_path="data_quality")
            print(f"   ✓ Logged raw data profiling report")
        
        if processed_report_path and os.path.exists(processed_report_path):
            mlflow.log_artifact(processed_report_path, artifact_path="data_quality")
            print(f"   ✓ Logged processed data profiling report")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        print(f"   ✓ Logged trained model")
        
        # Save model locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/weather_model.pkl")
        print(f"   ✓ Model saved locally to models/weather_model.pkl")
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        print(f"\n🔗 View your experiment on DagsHub:")
        print(f"   https://dagshub.com/Amnahsan/my-first-repo/experiments")
        print(f"\n📊 Run ID: {run.info.run_id}")
        print("=" * 60)

if __name__ == "__main__":
    # Check if processed data exists
    data_path = "data/processed/train_data.csv"
    if os.path.exists(data_path):
        train_model(data_path)
    else:
        print(f"❌ Data not found at {data_path}")
        print("Please run the data processor first:")
        print("  python src/data/processor.py")
