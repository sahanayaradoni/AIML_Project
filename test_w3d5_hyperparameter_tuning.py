"""
Tests for W3D5: Hyperparameter Tuning

Tests cover:
1. Dataset loading
2. Train-test split
3. SVM GridSearchCV
4. KNN RandomizedSearchCV
5. Model evaluation
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from w3d5_hyperparameter_tuning import evaluate_model


def test_dataset_loading():
    """Test that the Iris dataset loads correctly."""
    iris = load_iris()

    assert iris.data.shape == (150, 4)
    assert len(iris.target) == 150
    assert len(iris.target_names) == 3


def test_train_test_split():
    """Test the 80/20 train-test split."""
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    assert X_train.shape[0] == 120
    assert X_test.shape[0] == 30
    assert y_train.shape[0] == 120
    assert y_test.shape[0] == 30


def test_svm_grid_search():
    """Test SVM GridSearchCV."""
    iris = load_iris()

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVC()),
        ]
    )

    param_grid = {
        "model__C": [0.1, 1, 10, 100],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    search.fit(iris.data, iris.target)

    assert search.best_estimator_ is not None
    assert search.best_score_ >= 0.90


def test_knn_randomized_search():
    """Test KNN RandomizedSearchCV."""
    iris = load_iris()

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier()),
        ]
    )

    param_distributions = {
        "model__n_neighbors": [3, 5, 7, 9, 11],
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )

    search.fit(iris.data, iris.target)

    assert search.best_estimator_ is not None
    assert search.best_score_ >= 0.90


def test_evaluate_model():
    """Test the model evaluation function."""
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVC()),
        ]
    )

    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "Test SVM",
    )

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1