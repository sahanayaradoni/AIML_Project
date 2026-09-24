# W4D4 FastAPI Model Serving Endpoint — Results

## Objective

Serve the serialized W4D3 Iris Logistic Regression model through a FastAPI REST API.

## Implementation

- Loaded the W4D3 serialized model from `w4d3_models/iris_model.joblib`.
- Created a FastAPI application.
- Added a root endpoint at `GET /`.
- Added a model health endpoint at `GET /health`.
- Added a prediction endpoint at `POST /predict`.
- Used Pydantic validation for incoming Iris features.
- Returned the predicted class and class probabilities.
- Enabled interactive API documentation through Swagger UI at `/docs`.

## Prediction Test

Input:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```
