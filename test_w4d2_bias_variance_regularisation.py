"""
Tests for W4D2: Bias-Variance Tradeoff & Regularisation.
"""

from pathlib import Path

import pandas as pd

from w4d2_bias_variance_regularisation import (
    create_dataset,
    evaluate_model,
    evaluate_polynomial_models,
    run_lasso_search,
    run_randomized_search,
    run_ridge_search,
)


OUTPUT_DIR = Path("w4d2_outputs")


def test_create_dataset():
    """Verify that the dataset is split correctly."""

    X_train, X_test, y_train, y_test = create_dataset()

    assert len(X_train) == 240
    assert len(X_test) == 60
    assert len(y_train) == 240
    assert len(y_test) == 60


def test_bias_variance_results():
    """Verify bias-variance analysis results."""

    X_train, X_test, y_train, y_test = create_dataset()

    results = evaluate_polynomial_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) == 4

    expected_columns = {
        "degree",
        "train_rmse",
        "test_rmse",
        "train_r2",
        "test_r2",
    }

    assert expected_columns.issubset(results.columns)


def test_ridge_grid_search():
    """Verify Ridge GridSearchCV returns a fitted model."""

    X_train, _, y_train, _ = create_dataset()

    search = run_ridge_search(
        X_train,
        y_train,
    )

    assert search.best_estimator_ is not None
    assert "ridge__alpha" in search.best_params_
    assert search.best_score_ < 0


def test_lasso_grid_search():
    """Verify Lasso GridSearchCV returns a fitted model."""

    X_train, _, y_train, _ = create_dataset()

    search = run_lasso_search(
        X_train,
        y_train,
    )

    assert search.best_estimator_ is not None
    assert "lasso__alpha" in search.best_params_
    assert search.best_score_ < 0


def test_randomized_search():
    """Verify RandomizedSearchCV returns a fitted model."""

    X_train, _, y_train, _ = create_dataset()

    search = run_randomized_search(
        X_train,
        y_train,
    )

    assert search.best_estimator_ is not None
    assert "ridge__alpha" in search.best_params_
    assert search.best_score_ < 0


def test_model_evaluation():
    """Verify model evaluation returns valid metrics."""

    X_train, X_test, y_train, y_test = create_dataset()

    search = run_ridge_search(
        X_train,
        y_train,
    )

    rmse, r2 = evaluate_model(
        search.best_estimator_,
        X_test,
        y_test,
    )

    assert rmse >= 0
    assert -1 <= r2 <= 1


def test_output_files_exist():
    """Verify W4D2 output files were generated."""

    bias_variance_file = (
        OUTPUT_DIR / "bias_variance_results.csv"
    )

    comparison_file = (
        OUTPUT_DIR / "regularisation_comparison.csv"
    )

    assert bias_variance_file.exists()
    assert comparison_file.exists()