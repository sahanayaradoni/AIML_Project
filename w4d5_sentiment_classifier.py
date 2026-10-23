import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split


OUTPUT_DIR = "w4d5_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Binary sentiment-style text classification dataset
categories = ["rec.sport.baseball", "sci.med"]

data = fetch_20newsgroups(
    subset="all",
    categories=categories,
    remove=("headers", "footers", "quotes"),
    random_state=42,
)

X_text = data.data
y = data.target


# Train/test split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# TF-IDF text features
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# Logistic Regression
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)
logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression ===")
print(
    classification_report(
        y_test,
        logistic_predictions,
        target_names=data.target_names,
    )
)

logistic_accuracy = accuracy_score(y_test, logistic_predictions)
logistic_precision = precision_score(y_test, logistic_predictions)
logistic_recall = recall_score(y_test, logistic_predictions)
logistic_roc_auc = roc_auc_score(y_test, logistic_probabilities)

print(f"Accuracy: {logistic_accuracy:.4f}")
print(f"Precision: {logistic_precision:.4f}")
print(f"Recall: {logistic_recall:.4f}")
print(f"ROC-AUC: {logistic_roc_auc:.4f}")


# Logistic Regression confusion matrix
cm = confusion_matrix(y_test, logistic_predictions)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names,
).plot()

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "logistic_confusion_matrix.png")
)
plt.close()


# Random Forest
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)

print("\n=== Random Forest ===")
print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=data.target_names,
    )
)

print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}")


# ROC-AUC curve for Logistic Regression
RocCurveDisplay.from_predictions(
    y_test,
    logistic_probabilities,
)

plt.title("Logistic Regression - ROC-AUC Curve")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "logistic_roc_auc_curve.png")
)
plt.close()


# Model comparison
comparison = pd.DataFrame(
    {
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [logistic_accuracy, rf_accuracy],
        "Precision": [logistic_precision, rf_precision],
        "Recall": [logistic_recall, rf_recall],
    }
)

print("\n=== Model Comparison ===")
print(comparison.to_string(index=False))

comparison.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False,
)

print("\nW4D5 completed successfully.")
print(f"Outputs saved in: {OUTPUT_DIR}")