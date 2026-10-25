
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Cynaris ML Prediction API",
    description="A simple, container-ready machine learning prediction service.",
    version="1.0.0",
)


class PredictionInput(BaseModel):
    value: float = Field(
        ...,
        description="Numeric input for the prediction model.",
        examples=[5.0],
    )


class PredictionOutput(BaseModel):
    input_value: float
    prediction: float


@app.get("/")
def root():
    return {
        "message": "Cynaris ML Prediction API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    # Baseline model: replace with a trained model when one is available.
    prediction = 2.0 * data.value + 1.0

    return PredictionOutput(
        input_value=data.value,
        prediction=prediction,
    )
