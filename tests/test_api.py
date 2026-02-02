"""
Basic API tests for insurance enrollment prediction.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["status"] == "active"


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()


def test_info_endpoint():
    """Test the info endpoint."""
    response = client.get("/info")
    assert response.status_code == 200
    assert "name" in response.json()
    assert "endpoints" in response.json()


def test_predict_endpoint_with_valid_data():
    """Test prediction endpoint with valid employee data."""
    valid_employee = {
        "age": 35,
        "gender": "Male",
        "marital_status": "Married",
        "salary": 75000,
        "employment_type": "Full-time",
        "region": "West",
        "has_dependents": 1,
        "tenure_years": 5.5
    }
    
    response = client.post("/predict", json=valid_employee)
    
    # If model is not loaded, we expect 503
    # If model is loaded, we expect 200
    if response.status_code == 200:
        data = response.json()
        assert "enrolled_probability" in data
        assert "prediction" in data
        assert "confidence" in data
        assert 0 <= data["enrolled_probability"] <= 1
        assert data["prediction"] in [0, 1]
        assert data["confidence"] in ["low", "medium", "high"]
    elif response.status_code == 503:
        assert "Model is not loaded" in response.json()["detail"]


def test_predict_endpoint_with_invalid_age():
    """Test prediction endpoint with invalid age."""
    invalid_employee = {
        "age": 150,  # Invalid: too high
        "gender": "Male",
        "marital_status": "Married",
        "salary": 75000,
        "employment_type": "Full-time",
        "region": "West",
        "has_dependents": 1,
        "tenure_years": 5.5
    }
    
    response = client.post("/predict", json=invalid_employee)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_with_negative_salary():
    """Test prediction endpoint with negative salary."""
    invalid_employee = {
        "age": 35,
        "gender": "Male",
        "marital_status": "Married",
        "salary": -75000,  # Invalid: negative
        "employment_type": "Full-time",
        "region": "West",
        "has_dependents": 1,
        "tenure_years": 5.5
    }
    
    response = client.post("/predict", json=invalid_employee)
    assert response.status_code == 422  # Validation error


def test_batch_predict_endpoint():
    """Test batch prediction endpoint."""
    employees = [
        {
            "age": 35,
            "gender": "Male",
            "marital_status": "Married",
            "salary": 75000,
            "employment_type": "Full-time",
            "region": "West",
            "has_dependents": 1,
            "tenure_years": 5.5
        },
        {
            "age": 45,
            "gender": "Female",
            "marital_status": "Single",
            "salary": 85000,
            "employment_type": "Full-time",
            "region": "East",
            "has_dependents": 0,
            "tenure_years": 10.0
        }
    ]
    
    response = client.post("/batch-predict", json=employees)
    
    if response.status_code == 200:
        data = response.json()
        assert "count" in data
        assert "predictions" in data
        assert data["count"] == 2
    elif response.status_code == 503:
        assert "Model is not loaded" in response.json()["detail"]


def test_batch_predict_empty_list():
    """Test batch prediction with empty list."""
    response = client.post("/batch-predict", json=[])
    assert response.status_code == 400
    assert "Empty employee list" in response.json()["detail"]


if __name__ == "__main__":
    import pytest
    
    print("\nRunning API tests...")
    pytest.main([__file__, "-v"])
