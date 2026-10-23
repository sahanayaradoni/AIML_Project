"""
Tests for W4D1 Model Evaluation Metrics.
"""

from pathlib import Path

import numpy as np

from w4d1_model_evaluation import (
    OUTPUT_DIR,
    create_model,
    diagnose_fit,
    load_dataset,
    run_kfold_cv,
    run_stratified_kfold_cv,
    split_data,
)


def test_dataset_shape():
    """Verify that the dataset loads with expected dimensions."""

    X, y = load_dataset()

    assert X.shape == (569, 30)
    assert y.shape == (569,)


def test_dataset_classes():
    """Verify that the dataset contains two target classes."""

    _, y = load_dataset()

    assert len(np.unique(y)) == 2


def test_train_test_split():
    """Verify train/test split sizes."""

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = split_data(X, y)

    assert X_train.shape[0] == 455
    assert X_test.shape[0] == 114
    assert y_train.shape[0] == 455
    assert y_test.shape[0] == 114


def test_model_creation():
    """Verify that the model pipeline can be created."""

    model = create_model()

    assert model is not None
    assert "scaler" in model.named_steps
    assert "classifier" in model.named_steps


def test_kfold_cv():
    """Verify K-Fold CV returns five scores."""

    X, y = load_dataset()

    scores = run_kfold_cv(X, y)

    assert len(scores) == 5
    assert np.all(scores >= 0)
    assert np.all(scores <= 1)


def test_stratified_kfold_cv():
    """Verify Stratified K-Fold CV returns five scores."""

    X, y = load_dataset()

    scores = run_stratified_kfold_cv(X, y)

    assert len(scores) == 5
    assert np.all(scores >= 0)
    assert np.all(scores <= 1)


def test_output_directory():
    """Verify that the W4D1 output directory exists."""

    assert OUTPUT_DIR.exists()
    assert OUTPUT_DIR.is_dir()


def test_confusion_matrix_output():
    """Verify confusion matrix output was generated."""

    output_file = Path(OUTPUT_DIR) / "confusion_matrix.png"

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_learning_curve_output():
    """Verify learning curve output was generated."""

    output_file = Path(OUTPUT_DIR) / "learning_curve.png"

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_fit_diagnosis():
    """Verify model fit diagnosis returns a valid message."""

    train_scores = np.array([0.90, 0.92, 0.94])
    validation_scores = np.array([0.88, 0.90, 0.92])

    diagnosis = diagnose_fit(
        train_scores,
        validation_scores,
    )

    assert isinstance(diagnosis, str)
    assert len(diagnosis) > 0