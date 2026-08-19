import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load California Housing dataset
data = fetch_california_housing(as_frame=True)

X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("\nFeatures:")
print(X.columns.tolist())
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
# Create Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")

# Print coefficients
print("\nLinear Regression Coefficients:")
for feature, coefficient in zip(X.columns, linear_model.coef_):
    print(f"{feature}: {coefficient:.6f}")

print(f"Intercept: {linear_model.intercept_:.6f}")
# Make predictions on test data
y_pred = linear_model.predict(X_test)

print("\nPredictions generated successfully!")
print("First 5 predictions:")

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Evaluation")
print("=" * 40)
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# Plot predicted vs actual values
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Predicted vs Actual")

# Perfect prediction reference line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.tight_layout()
plt.savefig("predicted_vs_actual.png")
plt.show()

# Calculate residuals
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(8, 6))

plt.scatter(y_pred, residuals, alpha=0.5)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Linear Regression: Residual Plot")

plt.tight_layout()
plt.savefig("residual_plot.png")
plt.show()
# Create Ridge Regression model
ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])

# Train Ridge model
ridge_model.fit(X_train, y_train)

# Make Ridge predictions
ridge_pred = ridge_model.predict(X_test)

print("\nRidge Regression trained successfully!")
# Create Lasso Regression model
lasso_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Lasso(alpha=0.001, max_iter=10000))
])

# Train Lasso model
lasso_model.fit(X_train, y_train)

# Make Lasso predictions
lasso_pred = lasso_model.predict(X_test)

print("\nLasso Regression trained successfully!")

# Calculate Ridge metrics
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_rmse = np.sqrt(ridge_mse)
ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

# Calculate Lasso metrics
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_rmse = np.sqrt(lasso_mse)
lasso_mae = mean_absolute_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

print("\nRidge Regression Evaluation")
print("=" * 40)
print(f"MSE  : {ridge_mse:.4f}")
print(f"RMSE : {ridge_rmse:.4f}")
print(f"MAE  : {ridge_mae:.4f}")
print(f"R²   : {ridge_r2:.4f}")

print("\nLasso Regression Evaluation")
print("=" * 40)
print(f"MSE  : {lasso_mse:.4f}")
print(f"RMSE : {lasso_rmse:.4f}")
print(f"MAE  : {lasso_mae:.4f}")
print(f"R²   : {lasso_r2:.4f}")

# Create model comparison table
results = [
    {
        "Model": "Linear Regression",
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    },
    {
        "Model": "Ridge",
        "MSE": ridge_mse,
        "RMSE": ridge_rmse,
        "MAE": ridge_mae,
        "R2": ridge_r2
    },
    {
        "Model": "Lasso",
        "MSE": lasso_mse,
        "RMSE": lasso_rmse,
        "MAE": lasso_mae,
        "R2": lasso_r2
    }
]

results_df = pd.DataFrame(results)

print("\nModel Comparison")
print("=" * 70)
print(results_df.to_string(index=False))

results_df.to_csv("linear_regression_results.csv", index=False)

print("\nResults saved to linear_regression_results.csv")