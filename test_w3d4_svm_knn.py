"""
Tests for W3D4: SVM & KNN
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from w3d4_svm_knn import (
    build_svm_pipeline,
    build_knn_pipeline,
    evaluate_model,
)


RANDOM_STATE = 42


def get_test_data():
    """Load Iris data and create a train/test split."""

    iris = load_iris()

    X = iris.data
    y = iris.target

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def test_svm_pipeline_creation():
    """Check that the SVM pipeline is created correctly."""

    model = build_svm_pipeline()

    assert "scaler" in model.named_steps
    assert "svm" in model.named_steps


def test_knn_pipeline_creation():
    """Check that the KNN pipeline is created correctly."""

    model = build_knn_pipeline()

    assert "scaler" in model.named_steps
    assert "knn" in model.named_steps


def test_svm_training_and_prediction():
    """Check that SVM can train and make predictions."""

    X_train, X_test, y_train, y_test = get_test_data()

    model = build_svm_pipeline()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_knn_training_and_prediction():
    """Check that KNN can train and make predictions."""

    X_train, X_test, y_train, y_test = get_test_data()

    model = build_knn_pipeline()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_svm_accuracy_range():
    """Check that SVM accuracy is between 0 and 1."""

    X_train, X_test, y_train, y_test = get_test_data()

    model = build_svm_pipeline()

    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "Test SVM",
    )

    assert 0 <= metrics["accuracy"] <= 1


def test_knn_accuracy_range():
    """Check that KNN accuracy is between 0 and 1."""

    X_train, X_test, y_train, y_test = get_test_data()

    model = build_knn_pipeline()

    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "Test KNN",
    )

    assert 0 <= metrics["accuracy"] <= 1