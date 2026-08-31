# W3D3: Decision Trees & Random Forests
# Cynaris Internship - Week 3 Day 3

import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend to avoid GUI/Tkinter errors
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])

print("Class distribution:")
print(y.value_counts())


# ---------------------------------------------------------
# 2. Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ---------------------------------------------------------
# 3. Train Original Decision Tree
# ---------------------------------------------------------

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    random_state=42,
)

decision_tree.fit(X_train, y_train)


# ---------------------------------------------------------
# 4. Decision Tree Predictions
# ---------------------------------------------------------

y_pred = decision_tree.predict(X_test)


# ---------------------------------------------------------
# 5. Decision Tree Evaluation
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names,
    )
)


# ---------------------------------------------------------
# 6. Confusion Matrix
# ---------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names,
)

disp.plot()
plt.title("Decision Tree - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "w3d3_decision_tree_confusion_matrix.png",
    dpi=300,
)

plt.close()


# ---------------------------------------------------------
# 7. Visualise Decision Tree
# ---------------------------------------------------------

plt.figure(figsize=(20, 10))

plot_tree(
    decision_tree,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    max_depth=3,
)

plt.title("Decision Tree Visualization (First 3 Levels)")
plt.tight_layout()

plt.savefig(
    "w3d3_decision_tree_visualization.png",
    dpi=300,
)

plt.close()


# ---------------------------------------------------------
# 8. Decision Tree Feature Importance
# ---------------------------------------------------------

feature_importance = pd.DataFrame(
    {
        "feature": data.feature_names,
        "importance": decision_tree.feature_importances_,
    }
).sort_values(
    by="importance",
    ascending=False,
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

feature_importance.to_csv(
    "w3d3_decision_tree_feature_importance.csv",
    index=False,
)


# ---------------------------------------------------------
# 9. Save Original Decision Tree Results
# ---------------------------------------------------------

results = pd.DataFrame(
    {
        "model": ["Decision Tree"],
        "criterion": ["gini"],
        "accuracy": [accuracy],
        "tree_depth": [decision_tree.get_depth()],
        "number_of_leaves": [decision_tree.get_n_leaves()],
    }
)

results.to_csv(
    "w3d3_decision_tree_results.csv",
    index=False,
)

print("\nDecision Tree depth:", decision_tree.get_depth())
print("Number of leaves:", decision_tree.get_n_leaves())

print("\nW3D3 Decision Tree implementation completed successfully.")


# ---------------------------------------------------------
# 10. Overfitting Check
# ---------------------------------------------------------

unrestricted_tree = DecisionTreeClassifier(
    criterion="gini",
    random_state=42,
)

unrestricted_tree.fit(X_train, y_train)

train_accuracy = unrestricted_tree.score(
    X_train,
    y_train,
)

test_accuracy = unrestricted_tree.score(
    X_test,
    y_test,
)

print("\nOverfitting Check:")
print("Training Accuracy:", round(train_accuracy, 4))
print("Testing Accuracy:", round(test_accuracy, 4))
print("Unrestricted Tree Depth:", unrestricted_tree.get_depth())


# ---------------------------------------------------------
# 11. Tuned Decision Tree
# ---------------------------------------------------------

tuned_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
)

tuned_tree.fit(X_train, y_train)

tuned_train_accuracy = tuned_tree.score(
    X_train,
    y_train,
)

tuned_test_accuracy = tuned_tree.score(
    X_test,
    y_test,
)

print("\nTuned Decision Tree:")
print("Training Accuracy:", round(tuned_train_accuracy, 4))
print("Testing Accuracy:", round(tuned_test_accuracy, 4))
print("Tuned Tree Depth:", tuned_tree.get_depth())
print("Tuned Tree Leaves:", tuned_tree.get_n_leaves())


# ---------------------------------------------------------
# 12. Compare Original and Tuned Decision Trees
# ---------------------------------------------------------

comparison = pd.DataFrame(
    {
        "model": [
            "Original Decision Tree",
            "Tuned Decision Tree",
        ],
        "criterion": [
            "gini",
            "gini",
        ],
        "train_accuracy": [
            train_accuracy,
            tuned_train_accuracy,
        ],
        "test_accuracy": [
            test_accuracy,
            tuned_test_accuracy,
        ],
        "tree_depth": [
            unrestricted_tree.get_depth(),
            tuned_tree.get_depth(),
        ],
        "number_of_leaves": [
            unrestricted_tree.get_n_leaves(),
            tuned_tree.get_n_leaves(),
        ],
    }
)

print("\nDecision Tree Comparison:")
print(comparison)

comparison.to_csv(
    "w3d3_decision_tree_tuning_comparison.csv",
    index=False,
)


# ---------------------------------------------------------
# 13. Random Forest Classifier
# ---------------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=5,
    random_state=42,
    n_jobs=-1,
)

random_forest.fit(X_train, y_train)

rf_pred = random_forest.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred,
)

print("\nRandom Forest:")
print("Accuracy:", round(rf_accuracy, 4))

print("\nRandom Forest Classification Report:")
print(
    classification_report(
        y_test,
        rf_pred,
        target_names=data.target_names,
    )
)


# ---------------------------------------------------------
# 14. Random Forest Confusion Matrix
# ---------------------------------------------------------

rf_cm = confusion_matrix(
    y_test,
    rf_pred,
)

print("Random Forest Confusion Matrix:")
print(rf_cm)

rf_disp = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=data.target_names,
)

rf_disp.plot()
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "w3d3_random_forest_confusion_matrix.png",
    dpi=300,
)

plt.close()


# ---------------------------------------------------------
# 15. Random Forest Feature Importance
# ---------------------------------------------------------

rf_feature_importance = pd.DataFrame(
    {
        "feature": data.feature_names,
        "importance": random_forest.feature_importances_,
    }
).sort_values(
    by="importance",
    ascending=False,
)

print("\nTop 10 Random Forest Features:")
print(rf_feature_importance.head(10))

rf_feature_importance.to_csv(
    "w3d3_random_forest_feature_importance.csv",
    index=False,
)


# ---------------------------------------------------------
# 16. Model Comparison
# ---------------------------------------------------------

model_comparison = pd.DataFrame(
    {
        "model": [
            "Decision Tree",
            "Tuned Decision Tree",
            "Random Forest",
        ],
        "test_accuracy": [
            accuracy,
            tuned_test_accuracy,
            rf_accuracy,
        ],
    }
)

print("\nModel Comparison:")
print(model_comparison)

model_comparison.to_csv(
    "w3d3_model_comparison.csv",
    index=False,
)


# ---------------------------------------------------------
# 17. Save Random Forest Results
# ---------------------------------------------------------

rf_results = pd.DataFrame(
    {
        "model": ["Random Forest"],
        "n_estimators": [100],
        "criterion": ["gini"],
        "max_depth": [5],
        "accuracy": [rf_accuracy],
    }
)

rf_results.to_csv(
    "w3d3_random_forest_results.csv",
    index=False,
)


# ---------------------------------------------------------
# 18. Final Summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("W3D3 FINAL SUMMARY")
print("=" * 60)

print(
    "Original Decision Tree Test Accuracy:",
    round(accuracy, 4),
)

print(
    "Tuned Decision Tree Test Accuracy:",
    round(tuned_test_accuracy, 4),
)

print(
    "Random Forest Test Accuracy:",
    round(rf_accuracy, 4),
)

print(
    "Original Tree Depth:",
    unrestricted_tree.get_depth(),
)

print(
    "Tuned Tree Depth:",
    tuned_tree.get_depth(),
)

print(
    "Random Forest Estimators:",
    random_forest.n_estimators,
)

print("\nW3D3 Decision Tree and Random Forest implementation completed successfully.")