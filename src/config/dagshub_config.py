"""
DagsHub Configuration Module

This module handles the configuration of DagsHub and MLflow tracking.
It loads credentials from environment variables and sets up MLflow to track experiments to DagsHub.
"""

import os
import mlflow
import dagshub
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_dagshub_credentials():
    """
    Retrieve DagsHub credentials from environment variables.
    
    Returns:
        tuple: (repo_owner, repo_name, user_token)
    
    Raises:
        ValueError: If required environment variables are not set
    """
    repo_owner = os.getenv('DAGSHUB_REPO_OWNER', 'Amnahsan')
    repo_name = os.getenv('DAGSHUB_REPO_NAME', 'my-first-repo')
    user_token = os.getenv('DAGSHUB_USER_TOKEN')
    
    if not user_token:
        print("WARNING: DAGSHUB_USER_TOKEN not set. MLflow will track locally.")
        print("To enable DagsHub tracking:")
        print("1. Get your token from: https://dagshub.com/user/settings/tokens")
        print("2. Create a .env file based on .env.example")
        print("3. Set DAGSHUB_USER_TOKEN in the .env file")
        return None, None, None
    
    return repo_owner, repo_name, user_token


def setup_dagshub():
    """
    Configure MLflow to track experiments to DagsHub.
    
    This function:
    1. Loads DagsHub credentials
    2. Initializes DagsHub connection
    3. Sets MLflow tracking URI to DagsHub
    4. Configures authentication
    
    Returns:
        bool: True if DagsHub is configured, False if using local tracking
    """
    repo_owner, repo_name, user_token = get_dagshub_credentials()
    
    if not user_token:
        # Fall back to local MLflow tracking
        print("Using local MLflow tracking (mlruns/ directory)")
        return False
    
    try:
        # Initialize DagsHub
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        
        # Set MLflow tracking URI
        tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set authentication
        os.environ['MLFLOW_TRACKING_USERNAME'] = repo_owner
        os.environ['MLFLOW_TRACKING_PASSWORD'] = user_token
        
        print(f"✓ DagsHub MLflow tracking configured successfully!")
        print(f"  Repository: {repo_owner}/{repo_name}")
        print(f"  Tracking URI: {tracking_uri}")
        print(f"  View experiments at: https://dagshub.com/{repo_owner}/{repo_name}/experiments")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to configure DagsHub: {str(e)}")
        print("Falling back to local MLflow tracking")
        return False


def get_mlflow_tracking_uri():
    """
    Get the current MLflow tracking URI.
    
    Returns:
        str: MLflow tracking URI
    """
    return mlflow.get_tracking_uri()


if __name__ == "__main__":
    # Test the configuration
    print("Testing DagsHub Configuration...")
    print("-" * 50)
    
    success = setup_dagshub()
    
    if success:
        print("\n✓ Configuration test passed!")
        print(f"Current tracking URI: {get_mlflow_tracking_uri()}")
    else:
        print("\n⚠ Using local tracking. Set DAGSHUB_USER_TOKEN to enable DagsHub.")
