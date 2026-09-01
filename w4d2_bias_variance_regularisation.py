"""
W4D2: Bias-Variance Tradeoff & Regularisation

This module demonstrates:
1. Bias-variance tradeoff
2. Underfitting and overfitting
3. Ridge and Lasso regularisation
4. GridSearchCV for systematic hyperparameter tuning
5. RandomizedSearchCV for efficient hyperparameter tuning
6. MLflow experiment tracking
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

RANDOM_STATE = 42
OUTPUT_DIR = Path("w4d2_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MLFLOW_EXPERIMENT = "W4D2_Bias_Variance_Regularisation"


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------

def create_dataset():
    """Create a reproducible regression dataset."""

    X, y = make_regression(
        n_samples=300,
        n_features=8,
        n_informative=6,
        noise=20,
        random_state=RANDOM_STATE,
    )

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )


# -------------------------------------------------------------------
# Bias-Variance Demonstration
# -------------------------------------------------------------------

def evaluate_polynomial_models(X_train, X_test, y_train, y_test):
    """
    Compare polynomial models with different complexity.

    Low complexity may underfit (high bias).
    Very high complexity may overfit (high variance).
    """

    results = []

    for degree in [1, 2, 5, 10]:

        model = Pipeline(
            [
                ("polynomial", PolynomialFeatures(degree=degree)),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )

        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        train_rmse = np.sqrt(
            mean_squared_error(y_train, train_predictions)
        )
        test_rmse = np.sqrt(
            mean_squared_error(y_test, test_predictions)
        )

        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)

        results.append(
            {
                "degree": degree,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_r2": train_r2,
                "test_r2": test_r2,
            }
        )

    return pd.DataFrame(results)


# -------------------------------------------------------------------
# Ridge Regression
# -------------------------------------------------------------------

def run_ridge_search(X_train, y_train):
    """
    Tune Ridge alpha using GridSearchCV.

    Higher alpha means stronger L2 regularisation.
    """

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )

    param_grid = {
        "ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    return search


# -------------------------------------------------------------------
# Lasso Regression
# -------------------------------------------------------------------

def run_lasso_search(X_train, y_train):
    """
    Tune Lasso alpha using GridSearchCV.

    Higher alpha means stronger L1 regularisation.
    """

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", Lasso(max_iter=10000)),
        ]
    )

    param_grid = {
        "lasso__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    return search


# -------------------------------------------------------------------
# Randomized Search
# -------------------------------------------------------------------

def run_randomized_search(X_train, y_train):
    """
    Use RandomizedSearchCV to sample hyperparameter combinations.

    This is useful when the search space is large.
    """

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )

    param_distributions = {
        "ridge__alpha": np.logspace(-3, 3, 100)
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    return search


# -------------------------------------------------------------------
# Model Evaluation
# -------------------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """Calculate test RMSE and R²."""

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return rmse, r2


# -------------------------------------------------------------------
# Main Workflow
# -------------------------------------------------------------------

def main():
    """Run the complete W4D2 workflow."""

    print("=" * 60)
    print("W4D2: Bias-Variance Tradeoff & Regularisation")
    print("=" * 60)

    X_train, X_test, y_train, y_test = create_dataset()

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Bias-variance analysis
    bias_variance_results = evaluate_polynomial_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nBias-Variance Analysis:")
    print(bias_variance_results.to_string(index=False))

    bias_variance_results.to_csv(
        OUTPUT_DIR / "bias_variance_results.csv",
        index=False,
    )

    # Ridge Grid Search
    ridge_search = run_ridge_search(X_train, y_train)
    ridge_rmse, ridge_r2 = evaluate_model(
        ridge_search.best_estimator_,
        X_test,
        y_test,
    )

    print("\nRidge Grid Search:")
    print(f"Best alpha: {ridge_search.best_params_['ridge__alpha']}")
    print(f"Test RMSE: {ridge_rmse:.4f}")
    print(f"Test R²: {ridge_r2:.4f}")

    # Lasso Grid Search
    lasso_search = run_lasso_search(X_train, y_train)
    lasso_rmse, lasso_r2 = evaluate_model(
        lasso_search.best_estimator_,
        X_test,
        y_test,
    )

    print("\nLasso Grid Search:")
    print(f"Best alpha: {lasso_search.best_params_['lasso__alpha']}")
    print(f"Test RMSE: {lasso_rmse:.4f}")
    print(f"Test R²: {lasso_r2:.4f}")

    # Randomized Search
    randomized_search = run_randomized_search(
        X_train,
        y_train,
    )

    randomized_rmse, randomized_r2 = evaluate_model(
        randomized_search.best_estimator_,
        X_test,
        y_test,
    )

    print("\nRandomized Search:")
    print(
        "Best alpha:",
        randomized_search.best_params_["ridge__alpha"],
    )
    print(f"Test RMSE: {randomized_rmse:.4f}")
    print(f"Test R²: {randomized_r2:.4f}")

    # Save comparison results
    comparison = pd.DataFrame(
        [
            {
                "model": "Ridge GridSearchCV",
                "best_alpha": ridge_search.best_params_[
                    "ridge__alpha"
                ],
                "test_rmse": ridge_rmse,
                "test_r2": ridge_r2,
            },
            {
                "model": "Lasso GridSearchCV",
                "best_alpha": lasso_search.best_params_[
                    "lasso__alpha"
                ],
                "test_rmse": lasso_rmse,
                "test_r2": lasso_r2,
            },
            {
                "model": "Ridge RandomizedSearchCV",
                "best_alpha": randomized_search.best_params_[
                    "ridge__alpha"
                ],
                "test_rmse": randomized_rmse,
                "test_r2": randomized_r2,
            },
        ]
    )

    comparison.to_csv(
        OUTPUT_DIR / "regularisation_comparison.csv",
        index=False,
    )

    print("\nModel Comparison:")
    print(comparison.to_string(index=False))

    # MLflow tracking
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="W4D2_regularisation"):
        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )
        mlflow.log_param(
            "cross_validation_folds",
            5,
        )

        mlflow.log_metric(
            "ridge_test_rmse",
            ridge_rmse,
        )
        mlflow.log_metric(
            "ridge_test_r2",
            ridge_r2,
        )

        mlflow.log_metric(
            "lasso_test_rmse",
            lasso_rmse,
        )
        mlflow.log_metric(
            "lasso_test_r2",
            lasso_r2,
        )

        mlflow.log_metric(
            "randomized_test_rmse",
            randomized_rmse,
        )
        mlflow.log_metric(
            "randomized_test_r2",
            randomized_r2,
        )

        mlflow.log_param(
            "ridge_best_alpha",
            ridge_search.best_params_["ridge__alpha"],
        )

        mlflow.log_param(
            "lasso_best_alpha",
            lasso_search.best_params_["lasso__alpha"],
        )

        mlflow.log_param(
            "randomized_best_alpha",
            randomized_search.best_params_[
                "ridge__alpha"
            ],
        )

        mlflow.log_artifact(
            OUTPUT_DIR / "bias_variance_results.csv"
        )

        mlflow.log_artifact(
            OUTPUT_DIR / "regularisation_comparison.csv"
        )

    print("\nMLflow tracking completed.")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()