"""
W3D5: Hyperparameter Tuning — GridSearchCV & RandomizedSearchCV

This script:
1. Loads the Iris dataset.
2. Splits the data into training and testing sets.
3. Builds SVM and KNN pipelines.
4. Uses GridSearchCV for systematic hyperparameter tuning.
5. Uses RandomizedSearchCV for randomized hyperparameter tuning.
6. Evaluates the best models using accuracy, precision, recall, and F1-score.
7. Logs experiments, parameters, metrics, and models using MLflow.
"""


import mlflow
import mlflow.sklearn

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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset shape:", X.shape)
print("Number of classes:", len(iris.target_names))


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

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ---------------------------------------------------------
# 3. Create ML Pipelines
# ---------------------------------------------------------

# StandardScaler standardizes the features before training.
# Pipeline ensures scaling is performed correctly inside
# cross-validation without data leakage.

svm_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", SVC()),
    ]
)

knn_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier()),
    ]
)


# ---------------------------------------------------------
# 4. Define Hyperparameter Search Spaces
# ---------------------------------------------------------

# Hyperparameters for SVM
svm_param_grid = {
    "model__C": [0.1, 1, 10, 100],
    "model__kernel": ["linear", "rbf"],
    "model__gamma": ["scale", "auto"],
}


# Hyperparameters for KNN
knn_param_grid = {
    "model__n_neighbors": [3, 5, 7, 9, 11],
    "model__weights": ["uniform", "distance"],
    "model__metric": ["euclidean", "manhattan"],
}


# ---------------------------------------------------------
# 5. GridSearchCV — SVM
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("GRID SEARCH — SVM")
print("=" * 60)

svm_grid_search = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=svm_param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

svm_grid_search.fit(X_train, y_train)

print("Best SVM parameters:")
print(svm_grid_search.best_params_)

print("Best SVM CV score:")
print(f"{svm_grid_search.best_score_:.4f}")


# ---------------------------------------------------------
# 6. RandomizedSearchCV — KNN
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RANDOMIZED SEARCH — KNN")
print("=" * 60)

knn_random_search = RandomizedSearchCV(
    estimator=knn_pipeline,
    param_distributions=knn_param_grid,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)

knn_random_search.fit(X_train, y_train)

print("Best KNN parameters:")
print(knn_random_search.best_params_)

print("Best KNN CV score:")
print(f"{knn_random_search.best_score_:.4f}")


# ---------------------------------------------------------
# 7. Evaluation Function
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained classification model.

    Returns:
        Dictionary containing evaluation metrics.
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall   : {recall:.4f}")
    print(f"Test F1 Score : {f1:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


# ---------------------------------------------------------
# 8. Evaluate Tuned Models
# ---------------------------------------------------------

svm_metrics = evaluate_model(
    svm_grid_search.best_estimator_,
    X_test,
    y_test,
    "Best SVM — GridSearchCV",
)

knn_metrics = evaluate_model(
    knn_random_search.best_estimator_,
    X_test,
    y_test,
    "Best KNN — RandomizedSearchCV",
)


# ---------------------------------------------------------
# 9. MLflow Experiment Tracking
# ---------------------------------------------------------

mlflow.set_experiment("W3D5_Hyperparameter_Tuning")


# ---------------------------------------------------------
# 9A. Log SVM Experiment
# ---------------------------------------------------------

with mlflow.start_run(run_name="SVM_GridSearch"):

    mlflow.log_param("model", "SVM")
    mlflow.log_param("search_method", "GridSearchCV")
    mlflow.log_param("cv_folds", 5)

    # Log best SVM hyperparameters
    for parameter, value in svm_grid_search.best_params_.items():
        mlflow.log_param(parameter, value)

    # Log CV score
    mlflow.log_metric(
        "cv_best_accuracy",
        svm_grid_search.best_score_,
    )

    # Log test metrics
    for metric_name, metric_value in svm_metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    # Log best SVM model
    mlflow.sklearn.log_model(
        svm_grid_search.best_estimator_,
        name="best_svm_model",
    )


# ---------------------------------------------------------
# 9B. Log KNN Experiment
# ---------------------------------------------------------

with mlflow.start_run(run_name="KNN_RandomizedSearch"):

    mlflow.log_param("model", "KNN")
    mlflow.log_param("search_method", "RandomizedSearchCV")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("n_iter", 10)

    # Log best KNN hyperparameters
    for parameter, value in knn_random_search.best_params_.items():
        mlflow.log_param(parameter, value)

    # Log CV score
    mlflow.log_metric(
        "cv_best_accuracy",
        knn_random_search.best_score_,
    )

    # Log test metrics
    for metric_name, metric_value in knn_metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    # KNN with KDTree/EuclideanDistance64 requires these
    # types to be explicitly trusted by MLflow/skops.
    mlflow.sklearn.log_model(
        knn_random_search.best_estimator_,
        name="best_knn_model",
        skops_trusted_types=[
            "sklearn.metrics._dist_metrics.EuclideanDistance64",
            "sklearn.neighbors._kd_tree.KDTree",
        ],
    )


# ---------------------------------------------------------
# 10. Compare Final Models
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    f"SVM GridSearchCV Accuracy       : "
    f"{svm_metrics['accuracy']:.4f}"
)

print(
    f"KNN RandomizedSearchCV Accuracy : "
    f"{knn_metrics['accuracy']:.4f}"
)


# Determine the best model
if svm_metrics["accuracy"] > knn_metrics["accuracy"]:

    print("\nBest model: SVM with GridSearchCV")

elif knn_metrics["accuracy"] > svm_metrics["accuracy"]:

    print("\nBest model: KNN with RandomizedSearchCV")

else:

    print("\nBoth models achieved the same test accuracy.")

# ---------------------------------------------------------
# 11. Results Summary
# ---------------------------------------------------------

def print_results_summary(svm_metrics, knn_metrics):
    """Print a concise summary of the final model results."""

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"SVM Test Accuracy : {svm_metrics['accuracy']:.4f}")
    print(f"KNN Test Accuracy : {knn_metrics['accuracy']:.4f}")

    best_accuracy = max(
        svm_metrics["accuracy"],
        knn_metrics["accuracy"],
    )

    print(f"Best Test Accuracy: {best_accuracy:.4f}")


print_results_summary(svm_metrics, knn_metrics)
# ---------------------------------------------------------
# 11. Completion Message
# ---------------------------------------------------------

print("\nW3D5 Hyperparameter Tuning completed successfully!")