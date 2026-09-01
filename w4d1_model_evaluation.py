"""
W4D1: Model Evaluation Metrics — Precision, Recall, AUC

This script demonstrates:
1. Train/Test Split
2. K-Fold Cross-Validation
3. Stratified K-Fold Cross-Validation
4. Precision
5. Recall
6. F1-Score
7. ROC-AUC
8. Confusion Matrix
9. Learning Curves
10. Overfitting vs Underfitting diagnosis
11. MLflow experiment tracking
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = Path("w4d1_outputs")

OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------

def load_dataset():
    """Load the Breast Cancer Wisconsin dataset."""

    data = load_breast_cancer()

    X = data.data
    y = data.target

    print("Dataset loaded successfully.")
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Class distribution: {np.bincount(y)}")

    return X, y


# ---------------------------------------------------------------------
# 2. Create model pipeline
# ---------------------------------------------------------------------

def create_model():
    """Create a reproducible preprocessing and classification pipeline."""

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model


# ---------------------------------------------------------------------
# 3. Train/Test Split
# ---------------------------------------------------------------------

def split_data(X, y):
    """Split data into training and testing sets using stratification."""

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# ---------------------------------------------------------------------
# 4. Calculate evaluation metrics
# ---------------------------------------------------------------------

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train the model and calculate classification metrics."""

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probability)

    print("\n--- Test Set Evaluation ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "predictions": y_pred,
        "probabilities": y_probability,
    }


# ---------------------------------------------------------------------
# 5. Confusion Matrix
# ---------------------------------------------------------------------

def save_confusion_matrix(model, X_test, y_test):
    """Generate and save the confusion matrix."""

    predictions = model.predict(X_test)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Malignant", "Benign"],
    )

    display.ax_.set_title("W4D1 Confusion Matrix")

    output_path = OUTPUT_DIR / "confusion_matrix.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nConfusion matrix saved to: {output_path}")


# ---------------------------------------------------------------------
# 6. K-Fold Cross-Validation
# ---------------------------------------------------------------------

def run_kfold_cv(X, y):
    """Evaluate the model using standard K-Fold cross-validation."""

    model = create_model()

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=kfold,
        scoring="accuracy",
    )

    print("\n--- K-Fold Cross-Validation ---")
    print(f"Fold scores: {np.round(scores, 4)}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")

    return scores


# ---------------------------------------------------------------------
# 7. Stratified K-Fold Cross-Validation
# ---------------------------------------------------------------------

def run_stratified_kfold_cv(X, y):
    """Evaluate the model using Stratified K-Fold cross-validation."""

    model = create_model()

    stratified_kfold = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=stratified_kfold,
        scoring="accuracy",
    )

    print("\n--- Stratified K-Fold Cross-Validation ---")
    print(f"Fold scores: {np.round(scores, 4)}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")

    return scores


# ---------------------------------------------------------------------
# 8. Learning Curve
# ---------------------------------------------------------------------

def generate_learning_curve(X, y):
    """
    Generate a learning curve to diagnose
    overfitting and underfitting.
    """

    model = create_model()

    train_sizes, train_scores, validation_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)

    validation_mean = validation_scores.mean(axis=1)
    validation_std = validation_scores.std(axis=1)

    print("\n--- Learning Curve ---")

    for size, train_score, validation_score in zip(
        train_sizes,
        train_mean,
        validation_mean,
    ):
        print(
            f"Training samples: {size:.0f} | "
            f"Training accuracy: {train_score:.4f} | "
            f"Validation accuracy: {validation_score:.4f}"
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_sizes,
        train_mean,
        marker="o",
        label="Training Accuracy",
    )

    plt.plot(
        train_sizes,
        validation_mean,
        marker="o",
        label="Validation Accuracy",
    )

    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
    )

    plt.fill_between(
        train_sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.15,
    )

    plt.xlabel("Number of Training Samples")
    plt.ylabel("Accuracy")
    plt.title("W4D1 Learning Curve")
    plt.legend()
    plt.grid(True)

    output_path = OUTPUT_DIR / "learning_curve.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Learning curve saved to: {output_path}")

    return train_mean, validation_mean


# ---------------------------------------------------------------------
# 9. Diagnose model fit
# ---------------------------------------------------------------------

def diagnose_fit(train_scores, validation_scores):
    """Provide a simple interpretation of training and validation scores."""

    final_train = train_scores[-1]
    final_validation = validation_scores[-1]

    difference = final_train - final_validation

    print("\n--- Model Fit Diagnosis ---")
    print(f"Final training accuracy: {final_train:.4f}")
    print(f"Final validation accuracy: {final_validation:.4f}")
    print(f"Training-validation gap: {difference:.4f}")

    if final_train < 0.80 and final_validation < 0.80:
        diagnosis = "Possible underfitting."

    elif difference > 0.10:
        diagnosis = "Possible overfitting."

    else:
        diagnosis = "Model appears reasonably well-fitted."

    print(f"Diagnosis: {diagnosis}")

    return diagnosis


# ---------------------------------------------------------------------
# 10. MLflow tracking
# ---------------------------------------------------------------------

def log_with_mlflow(metrics, kfold_scores, stratified_scores):
    """Log model evaluation results to MLflow."""

    mlflow.set_experiment("Cynaris_W4D1_Model_Evaluation")

    with mlflow.start_run(run_name="W4D1_Logistic_Regression"):

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("cv_folds", 5)

        mlflow.log_metric("test_accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1"])
        mlflow.log_metric("roc_auc", metrics["auc"])

        mlflow.log_metric(
            "kfold_mean_accuracy",
            kfold_scores.mean(),
        )

        mlflow.log_metric(
            "kfold_std_accuracy",
            kfold_scores.std(),
        )

        mlflow.log_metric(
            "stratified_kfold_mean_accuracy",
            stratified_scores.mean(),
        )

        mlflow.log_metric(
            "stratified_kfold_std_accuracy",
            stratified_scores.std(),
        )

        mlflow.log_artifact(
            str(OUTPUT_DIR / "confusion_matrix.png")
        )

        mlflow.log_artifact(
            str(OUTPUT_DIR / "learning_curve.png")
        )

        print("\nMLflow tracking completed successfully.")


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main():
    """Run the complete W4D1 model evaluation workflow."""

    print("=" * 70)
    print("W4D1: MODEL EVALUATION METRICS")
    print("=" * 70)

    # Load data
    X, y = load_dataset()

    # Train/test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n--- Train/Test Split ---")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples : {X_test.shape[0]}")

    # Create and train model
    model = create_model()

    # Evaluate model
    metrics = evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Confusion matrix
    save_confusion_matrix(
        model,
        X_test,
        y_test,
    )

    # K-Fold CV
    kfold_scores = run_kfold_cv(X, y)

    # Stratified K-Fold CV
    stratified_scores = run_stratified_kfold_cv(X, y)

    # Learning curve
    train_scores, validation_scores = generate_learning_curve(X, y)

    # Diagnose fit
    diagnosis = diagnose_fit(
        train_scores,
        validation_scores,
    )

    # MLflow
    log_with_mlflow(
        metrics,
        kfold_scores,
        stratified_scores,
    )

    print("\n" + "=" * 70)
    print("W4D1 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nKey Results:")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-Score  : {metrics['f1']:.4f}")
    print(f"ROC-AUC   : {metrics['auc']:.4f}")
    print(f"K-Fold CV : {kfold_scores.mean():.4f}")
    print(
        f"Stratified CV: "
        f"{stratified_scores.mean():.4f}"
    )
    print(f"Fit diagnosis: {diagnosis}")


if __name__ == "__main__":
    main()