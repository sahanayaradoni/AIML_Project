
from fastapi.testclient import TestClient

from w8d1_mlops.app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Cynaris ML Prediction API is running"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_endpoint():
    response = client.post("/predict", json={"value": 5.0})
    assert response.status_code == 200
    assert response.json() == {
        "input_value": 5.0,
        "prediction": 11.0,
    }


def test_prediction_rejects_invalid_input():
    response = client.post("/predict", json={"value": "invalid"})
    assert response.status_code == 422
