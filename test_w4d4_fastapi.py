"""Tests for the W4D4 FastAPI model serving endpoint."""

from fastapi.testclient import TestClient

from w4d4_fastapi import app


client = TestClient(app)


def test_root_endpoint():
    """Verify that the root endpoint is available."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Iris Model Serving API is running"


def test_health_endpoint():
    """Verify that the health endpoint reports the model as ready."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model"] == "iris_model.joblib"


def test_predict_endpoint():
    """Verify that valid Iris features produce a prediction."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "probabilities" in result
    assert result["prediction"] in [0, 1, 2]
    assert len(result["probabilities"]) == 3


def test_predict_rejects_invalid_input():
    """Verify that invalid feature values are rejected."""
    payload = {
        "sepal_length": -1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422