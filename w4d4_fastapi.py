"""
W4D4: FastAPI Model Serving Endpoint

This module:
1. Loads the serialized Iris Logistic Regression model.
2. Defines a validated prediction request using Pydantic.
3. Exposes a health-check endpoint.
4. Exposes a POST /predict endpoint.
5. Returns the predicted class and prediction probabilities.
"""

from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Path to the model serialized during W4D3.
MODEL_PATH = Path("w4d3_models/iris_model.joblib")

# Load the trained model once when the API starts.
model = joblib.load(MODEL_PATH)

# Create the FastAPI application.
app = FastAPI(
    title="Iris Model Serving API",
    description="FastAPI endpoint for serving the W4D3 Iris Logistic Regression model.",
    version="1.0.0",
)


class IrisRequest(BaseModel):
    """Input schema for Iris prediction."""

    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


@app.get("/")
def root():
    """Return basic API information."""
    return {
        "message": "Iris Model Serving API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    """Check whether the API and model are ready."""
    return {
        "status": "healthy",
        "model": MODEL_PATH.name,
    }


@app.post("/predict")
def predict(request: IrisRequest):
    """Generate an Iris prediction from validated input features."""

    features = np.array(
        [[
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width,
        ]]
    )

    prediction = int(model.predict(features)[0])

    probabilities = model.predict_proba(features)[0]

    return {
        "prediction": prediction,
        "probabilities": {
            str(index): round(float(probability), 6)
            for index, probability in enumerate(probabilities)
        },
    }