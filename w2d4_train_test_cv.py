# W2D4: Train/Test Split & Cross-Validation

import random
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)


def main() -> None:
    """Train a classification model and evaluate it using cross-validation."""

    # Load the Iris dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Create a pipeline to prevent scaling data leakage
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate on the test set
    test_accuracy = model.score(X_test, y_test)

    # Use explicit stratified 5-fold cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
    )

    mean_cv_accuracy = np.mean(cv_scores)

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "w2d4_output.png"

    # Create output evidence
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    output_text = (
        "W2D4: Train/Test Split & Cross-Validation\n\n"
        f"Training samples: {len(X_train)}\n"
        f"Testing samples: {len(X_test)}\n"
        f"Test accuracy: {test_accuracy:.4f}\n\n"
        f"CV scores: {np.array2string(cv_scores, precision=4)}\n"
        f"Mean CV accuracy: {mean_cv_accuracy:.4f}\n\n"
        "MLflow tracking completed."
    )

    ax.text(
        0.05,
        0.95,
        output_text,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Set MLflow experiment before starting the run
    mlflow.set_experiment("W2D4_Train_Test_CV")

    # Track experiment with MLflow
    with mlflow.start_run():
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("mean_cv_accuracy", mean_cv_accuracy)

        mlflow.set_tag("seed", 42)

        # Log trained model and output evidence
        mlflow.sklearn.log_model(model, name="logreg_model")
        mlflow.log_artifact(str(output_path))

    # Display results
    print("W2D4: Train/Test Split & Cross-Validation")
    print("------------------------------------------")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"CV scores: {cv_scores}")
    print(f"Mean CV accuracy: {mean_cv_accuracy:.4f}")
    print("\nMLflow tracking completed.")
    print(f"Output evidence saved as: {output_path}")


if __name__ == "__main__":
    main()