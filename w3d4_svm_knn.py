"""
W3D4: SVM & KNN - When to Use What

This script:
1. Loads the Iris classification dataset.
2. Splits the data into training and testing sets.
3. Applies StandardScaler using pipelines.
4. Trains SVM and KNN classifiers.
5. Evaluates both models using accuracy, precision, recall and F1-score.
6. Generates confusion matrices and a model comparison plot.
7. Performs hyperparameter tuning using GridSearchCV.
8. Logs model parameters and metrics using MLflow.

Author: Sahana
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20

OUTPUT_DIR = Path("w3d4_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MLFLOW_EXPERIMENT = "W3D4_SVM_KNN"


# ---------------------------------------------------------
# 2. Load Dataset
# ---------------------------------------------------------

def load_dataset():
    """Load the Iris dataset and return features and target."""

    iris = load_iris()

    X = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    y = pd.Series(
        iris.target,
        name="target"
    )

    print("\nDataset loaded successfully.")
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(iris.target_names)}")
    print(f"Classes: {list(iris.target_names)}")

    return X, y, iris.target_names


# ---------------------------------------------------------
# 3. Evaluate Model
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a classification model and return its metrics.
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(f"\n{'=' * 50}")
    print(f"{model_name} Evaluation")
    print(f"{'=' * 50}")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predictions": predictions,
    }


# ---------------------------------------------------------
# 4. Save Confusion Matrix
# ---------------------------------------------------------

def save_confusion_matrix(model, X_test, y_test, class_names, model_name):
    """Generate and save a confusion matrix."""

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot()

    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()

    filename = OUTPUT_DIR / (
        f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    plt.savefig(filename)
    plt.close()

    print(f"Saved confusion matrix: {filename}")


# ---------------------------------------------------------
# 5. Build SVM Pipeline
# ---------------------------------------------------------

def build_svm_pipeline():
    """
    Create an SVM pipeline.

    StandardScaler prevents differences in feature scales
    from negatively affecting the SVM model.
    """

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale"
                )
            ),
        ]
    )


# ---------------------------------------------------------
# 6. Build KNN Pipeline
# ---------------------------------------------------------

def build_knn_pipeline():
    """
    Create a KNN pipeline.

    StandardScaler is important because KNN uses
    distance calculations.
    """

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=5
                )
            ),
        ]
    )


# ---------------------------------------------------------
# 7. Hyperparameter Tuning - SVM
# ---------------------------------------------------------

def tune_svm(X_train, y_train):
    """Tune SVM hyperparameters using GridSearchCV."""

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC()),
        ]
    )

    parameter_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__kernel": ["linear", "rbf"],
        "svm__gamma": ["scale", "auto"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\nSVM Hyperparameter Tuning")
    print("-" * 40)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search


# ---------------------------------------------------------
# 8. Hyperparameter Tuning - KNN
# ---------------------------------------------------------

def tune_knn(X_train, y_train):
    """Tune KNN hyperparameters using GridSearchCV."""

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ]
    )

    parameter_grid = {
        "knn__n_neighbors": [3, 5, 7, 9, 11],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\nKNN Hyperparameter Tuning")
    print("-" * 40)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search


# ---------------------------------------------------------
# 9. Compare Models
# ---------------------------------------------------------

def create_comparison_plot(results):
    """Create a bar chart comparing model performance."""

    model_names = list(results.keys())

    accuracy_values = [
        results[name]["accuracy"]
        for name in model_names
    ]

    precision_values = [
        results[name]["precision"]
        for name in model_names
    ]

    recall_values = [
        results[name]["recall"]
        for name in model_names
    ]

    f1_values = [
        results[name]["f1_score"]
        for name in model_names
    ]

    metrics_df = pd.DataFrame(
        {
            "Accuracy": accuracy_values,
            "Precision": precision_values,
            "Recall": recall_values,
            "F1-Score": f1_values,
        },
        index=model_names
    )

    print("\nModel Comparison")
    print("-" * 50)
    print(metrics_df.round(4))

    ax = metrics_df.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title("SVM vs KNN Performance Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()

    output_file = OUTPUT_DIR / "model_comparison.png"

    plt.savefig(output_file)
    plt.close()

    print(f"\nSaved comparison plot: {output_file}")

    return metrics_df


# ---------------------------------------------------------
# 10. MLflow Logging
# ---------------------------------------------------------

def log_experiment(model_name, model, metrics, grid_search=None):
    """
    Log model parameters and evaluation metrics to MLflow.
    """

    with mlflow.start_run(run_name=f"W3D4_{model_name}"):

        mlflow.log_param("model", model_name)

        if model_name == "SVM":
            svm_model = model.named_steps["svm"]

            mlflow.log_param("kernel", svm_model.kernel)
            mlflow.log_param("C", svm_model.C)
            mlflow.log_param("gamma", svm_model.gamma)

        elif model_name == "KNN":
            knn_model = model.named_steps["knn"]

            mlflow.log_param(
                "n_neighbors",
                knn_model.n_neighbors
            )

            mlflow.log_param(
                "weights",
                knn_model.weights
            )

            mlflow.log_param(
                "metric",
                knn_model.metric
            )

        mlflow.log_metric(
            "accuracy",
            metrics["accuracy"]
        )

        mlflow.log_metric(
            "precision",
            metrics["precision"]
        )

        mlflow.log_metric(
            "recall",
            metrics["recall"]
        )

        mlflow.log_metric(
            "f1_score",
            metrics["f1_score"]
        )

        if grid_search is not None:
            mlflow.log_metric(
                "best_cv_accuracy",
                grid_search.best_score_
            )

        print(f"MLflow run logged successfully for {model_name}.")


# ---------------------------------------------------------
# 11. Main Function
# ---------------------------------------------------------

def main():

    print("\n" + "=" * 60)
    print("W3D4: SVM & KNN - When to Use What")
    print("=" * 60)

    # Load dataset
    X, y, class_names = load_dataset()

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTrain/Test Split")
    print("-" * 40)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # Create MLflow experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # -----------------------------------------------------
    # SVM
    # -----------------------------------------------------

    svm_model = build_svm_pipeline()

    svm_model.fit(
        X_train,
        y_train
    )

    svm_results = evaluate_model(
        svm_model,
        X_test,
        y_test,
        "SVM"
    )

    save_confusion_matrix(
        svm_model,
        X_test,
        y_test,
        class_names,
        "SVM"
    )

    log_experiment(
        "SVM",
        svm_model,
        svm_results
    )

    # -----------------------------------------------------
    # KNN
    # -----------------------------------------------------

    knn_model = build_knn_pipeline()

    knn_model.fit(
        X_train,
        y_train
    )

    knn_results = evaluate_model(
        knn_model,
        X_test,
        y_test,
        "KNN"
    )

    save_confusion_matrix(
        knn_model,
        X_test,
        y_test,
        class_names,
        "KNN"
    )

    log_experiment(
        "KNN",
        knn_model,
        knn_results
    )

    # -----------------------------------------------------
    # Hyperparameter Tuning
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)

    best_svm, svm_grid = tune_svm(
        X_train,
        y_train
    )

    best_knn, knn_grid = tune_knn(
        X_train,
        y_train
    )

    # Evaluate tuned SVM
    tuned_svm_results = evaluate_model(
        best_svm,
        X_test,
        y_test,
        "Tuned SVM"
    )

    # Evaluate tuned KNN
    tuned_knn_results = evaluate_model(
        best_knn,
        X_test,
        y_test,
        "Tuned KNN"
    )

    # Save tuned confusion matrices
    save_confusion_matrix(
        best_svm,
        X_test,
        y_test,
        class_names,
        "Tuned SVM"
    )

    save_confusion_matrix(
        best_knn,
        X_test,
        y_test,
        class_names,
        "Tuned KNN"
    )

    # Log tuned models
    log_experiment(
        "Tuned SVM",
        best_svm,
        tuned_svm_results,
        svm_grid
    )

    log_experiment(
        "Tuned KNN",
        best_knn,
        tuned_knn_results,
        knn_grid
    )

    # -----------------------------------------------------
    # Final Comparison
    # -----------------------------------------------------

    comparison_results = {
        "SVM": svm_results,
        "KNN": knn_results,
        "Tuned SVM": tuned_svm_results,
        "Tuned KNN": tuned_knn_results,
    }

    create_comparison_plot(
        comparison_results
    )

    # -----------------------------------------------------
    # Final Summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print("\nBest SVM Parameters:")
    print(svm_grid.best_params_)

    print("\nBest KNN Parameters:")
    print(knn_grid.best_params_)

    print("\nOutput files saved in:")
    print(OUTPUT_DIR.resolve())

    print("\nW3D4 completed successfully.")


# ---------------------------------------------------------
# Program Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()

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