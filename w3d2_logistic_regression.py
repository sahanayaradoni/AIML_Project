# W3D2: Logistic Regression & Classification
# Cynaris Internship - Week 3 Day 2

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def main():
    # ---------------------------------------------------------
    # 1. Load Breast Cancer dataset
    # ---------------------------------------------------------
    data = load_breast_cancer(as_frame=True)

    df = data.frame

    print("Dataset shape:", df.shape)
    print("Number of features:", len(data.feature_names))
    print("Classes:", data.target_names)

    # ---------------------------------------------------------
    # 2. Missing-value check
    # ---------------------------------------------------------
    missing_values = df.isnull().sum().sum()

    print("\nTotal missing values:", missing_values)

    assert missing_values == 0, "Missing values detected"

    print("Missing-value check passed.")

    # ---------------------------------------------------------
    # 3. Separate features and target
    # ---------------------------------------------------------
    X = df.drop(columns="target")
    y = df["target"]

    assert "target" not in X.columns

    print("\nFeature shape:", X.shape)
    print("Target distribution:")
    print(y.value_counts())

    # ---------------------------------------------------------
    # 4. Stratified train-test split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    # ---------------------------------------------------------
    # 5. Create preprocessing + Logistic Regression pipeline
    # ---------------------------------------------------------
    model = Pipeline(
        [
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

    # ---------------------------------------------------------
    # 6. Train model
    # ---------------------------------------------------------
    model.fit(X_train, y_train)

    print("\nLogistic Regression model trained successfully!")

    # ---------------------------------------------------------
    # 7. Predictions and probabilities
    # ---------------------------------------------------------
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------
    # 8. Evaluation metrics
    # ---------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probability)

    print("\n--- Model Evaluation ---")
    print("=" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # ---------------------------------------------------------
    # 9. Classification report
    # ---------------------------------------------------------
    print("\n--- Classification Report ---")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=data.target_names,
        )
    )

    # ---------------------------------------------------------
    # 10. Confusion matrix
    # ---------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)

    print("--- Confusion Matrix ---")
    print(cm)

    # ---------------------------------------------------------
    # 11. 5-Fold Stratified Cross-Validation
    # ---------------------------------------------------------
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=skf,
        scoring="roc_auc",
    )

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    print("\n--- 5-Fold Stratified Cross-Validation ---")
    print("ROC-AUC CV scores:", cv_scores)
    print(f"CV mean ROC-AUC: {cv_mean:.4f}")
    print(f"CV std ROC-AUC : {cv_std:.4f}")

    # ---------------------------------------------------------
    # 12. Logistic Regression coefficients
    # ---------------------------------------------------------
    classifier = model.named_steps["classifier"]

    coefficients = pd.DataFrame(
        {
            "Feature": data.feature_names,
            "Coefficient": classifier.coef_[0],
        }
    )

    coefficients["Absolute_Coefficient"] = (
        coefficients["Coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        by="Absolute_Coefficient",
        ascending=False,
    )

    print("\n--- Logistic Regression Coefficients ---")
    print(
        coefficients[
            ["Feature", "Coefficient"]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # 13. Save model
    # ---------------------------------------------------------
    joblib.dump(
        model,
        "w3d2_logistic_regression_model.joblib",
    )

    # ---------------------------------------------------------
    # 14. Save metrics/results
    # ---------------------------------------------------------
    results = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
                "CV Mean ROC-AUC",
                "CV Std ROC-AUC",
            ],
            "Score": [
                accuracy,
                precision,
                recall,
                f1,
                roc_auc,
                cv_mean,
                cv_std,
            ],
        }
    )

    results.to_csv(
        "w3d2_logistic_regression_results.csv",
        index=False,
    )

    coefficients.to_csv(
        "w3d2_logistic_regression_coefficients.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 15. Save confusion matrix plot
    # ---------------------------------------------------------
    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title("Logistic Regression - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.xticks(
        [0, 1],
        data.target_names,
    )

    plt.yticks(
        [0, 1],
        data.target_names,
    )

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        "w3d2_confusion_matrix.png",
        dpi=300,
    )

    plt.close()

    # ---------------------------------------------------------
    # 16. Save probability distribution plot
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.hist(
        y_probability[y_test == 0],
        bins=20,
        alpha=0.6,
        label=data.target_names[0],
    )

    plt.hist(
        y_probability[y_test == 1],
        bins=20,
        alpha=0.6,
        label=data.target_names[1],
    )

    plt.axvline(
        0.5,
        linestyle="--",
        label="Decision Threshold = 0.5",
    )

    plt.title("Predicted Probability Distribution")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Number of Samples")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "w3d2_probability_distribution.png",
        dpi=300,
    )

    plt.close()

    print("\nOutput files created successfully.")
    print("Saved model: w3d2_logistic_regression_model.joblib")


if __name__ == "__main__":
    main()