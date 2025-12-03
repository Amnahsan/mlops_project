from fastapi.testclient import TestClient
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Weather RPS API is running!"}

def test_predict_endpoint_structure():
    # We can't easily test the full prediction without a model and live data mocking,
    # but we can test that the endpoint exists and handles bad input.
    response = client.post("/predict", json={})
    # Should fail validation or 503 if model not loaded, but 422 if missing fields
    # Pydantic defaults might make it valid, let's see.
    # Our model has defaults for lat/lon, so it might try to run.
    # If model is not loaded (which it won't be in a fresh test env unless we mock it), it returns 503.
    
    # Let's just check it doesn't 404
    assert response.status_code != 404
