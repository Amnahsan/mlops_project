from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add project root to path so we can import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.fetcher import fetch_historical_data, fetch_current_data

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'weather_rps_pipeline',
    default_args=default_args,
    description='A simple MLOps pipeline for Weather RPS',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id='fetch_data',
        python_callable=fetch_current_data,
    )

    def run_processor(**context):
        # In a real Airflow setup, we'd pass paths via XCom or use a fixed path strategy
        # For this simple example, we assume fetcher saved to 'data/raw/current_weather_YYYY-MM-DD.csv'
        # and we process it.
        # Let's find the latest file in data/raw
        import os
        raw_dir = "data/raw"
        if not os.path.exists(raw_dir):
            raise FileNotFoundError("No raw data found!")
            
        files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
        if not files:
            raise FileNotFoundError("No CSV files in data/raw!")
            
        latest_file = max([os.path.join(raw_dir, f) for f in files], key=os.path.getctime)
        
        from src.data.processor import check_quality, process_data
        
        df = pd.read_csv(latest_file)
        if not check_quality(df):
            raise ValueError("Data Quality Check Failed!")
            
        process_data(latest_file, "data/processed/latest_processed.csv")

    t2 = PythonOperator(
        task_id='process_data',
        python_callable=run_processor,
    )

    t1 >> t2
