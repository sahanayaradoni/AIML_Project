import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.over_sampling import SMOTE


# 1. Create an imbalanced dataset

X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    weights=[0.90, 0.10],
    random_state=42
)

print("Original class distribution:")
print(pd.Series(y).value_counts())


# 2. Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining distribution BEFORE SMOTE:")
print(pd.Series(y_train).value_counts())


# 3. Apply SMOTE only to the training data

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nTraining distribution AFTER SMOTE:")
print(pd.Series(y_train_smote).value_counts())


# 4. Set up MLflow experiment

mlflow.set_experiment("W2D3_SMOTE")


# 5. Start MLflow run

with mlflow.start_run():

    # Train Logistic Regression model
    model = LogisticRegression(random_state=42)

    model.fit(
        X_train_smote,
        y_train_smote
    )

    # 6. Predict using the untouched test data

    y_pred = model.predict(X_test)


    # 7. Calculate evaluation metrics

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


    # 8. Display classification results

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


    # 9. Display MLflow metrics

    print("\nMLflow Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


    # 10. Log parameters to MLflow

    mlflow.log_param(
        "model",
        "LogisticRegression"
    )

    mlflow.log_param(
        "smote",
        "SMOTE"
    )

    mlflow.log_param(
        "n_samples",
        1000
    )

    mlflow.log_param(
        "test_size",
        0.2
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_param(
        "smote_random_state",
        42
    )


    # 11. Log metrics to MLflow

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )


    # 12. Save model to MLflow

    mlflow.sklearn.log_model(
        model,
        "model"
    )


    # 13. Save class distribution plots

    before = (
        pd.Series(y_train)
        .value_counts()
        .sort_index()
    )

    after = (
        pd.Series(y_train_smote)
        .value_counts()
        .sort_index()
    )


    # BEFORE SMOTE plot

    plt.figure(figsize=(8, 5))

    plt.bar(
        ["Class 0", "Class 1"],
        before.values
    )

    plt.title(
        "Training Class Distribution BEFORE SMOTE"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.savefig(
        "smote_before.png"
    )

    mlflow.log_artifact(
        "smote_before.png"
    )

    plt.close()


    # AFTER SMOTE plot

    plt.figure(figsize=(8, 5))

    plt.bar(
        ["Class 0", "Class 1"],
        after.values
    )

    plt.title(
        "Training Class Distribution AFTER SMOTE"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.savefig(
        "smote_after.png"
    )

    mlflow.log_artifact(
        "smote_after.png"
    )

    plt.close()


print("\nSMOTE processing completed successfully.")

print(
    "Output files: smote_before.png, smote_after.png"
)

print(
    "MLflow experiment: W2D3_SMOTE"
)