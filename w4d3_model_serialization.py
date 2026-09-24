"""
W4D3: Model Serialization using joblib and pickle

This script:
1. Loads the Iris dataset.
2. Trains a Logistic Regression model.
3. Saves the trained model using joblib.
4. Saves the trained model using pickle.
5. Loads both serialized models.
6. Verifies that predictions remain unchanged.
"""

from pathlib import Path
import pickle

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Create a directory for serialized models.
MODEL_DIR = Path("w4d3_models")
MODEL_DIR.mkdir(exist_ok=True)

JOBLIB_PATH = MODEL_DIR / "iris_model.joblib"
PICKLE_PATH = MODEL_DIR / "iris_model.pkl"


def train_model():
    """Train and return a Logistic Regression model."""
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    return model, X_test, y_test


def serialize_with_joblib(model):
    """Save the model using joblib."""
    joblib.dump(model, JOBLIB_PATH)


def serialize_with_pickle(model):
    """Save the model using pickle."""
    with open(PICKLE_PATH, "wb") as file:
        pickle.dump(model, file)


def load_with_joblib():
    """Load the model saved with joblib."""
    return joblib.load(JOBLIB_PATH)


def load_with_pickle():
    """Load the model saved with pickle."""
    with open(PICKLE_PATH, "rb") as file:
        return pickle.load(file)


def main():
    """Train, serialize, deserialize, and verify the model."""
    model, X_test, y_test = train_model()

    original_predictions = model.predict(X_test)
    original_accuracy = accuracy_score(y_test, original_predictions)

    serialize_with_joblib(model)
    serialize_with_pickle(model)

    joblib_model = load_with_joblib()
    pickle_model = load_with_pickle()

    joblib_predictions = joblib_model.predict(X_test)
    pickle_predictions = pickle_model.predict(X_test)

    joblib_accuracy = accuracy_score(y_test, joblib_predictions)
    pickle_accuracy = accuracy_score(y_test, pickle_predictions)

    print("W4D3: Model Serialization")
    print("-" * 40)
    print(f"Original model accuracy : {original_accuracy:.4f}")
    print(f"joblib model accuracy   : {joblib_accuracy:.4f}")
    print(f"pickle model accuracy   : {pickle_accuracy:.4f}")

    print()
    print(f"joblib file created: {JOBLIB_PATH.exists()}")
    print(f"pickle file created: {PICKLE_PATH.exists()}")

    print()
    print(
        "joblib predictions match:",
        (original_predictions == joblib_predictions).all(),
    )
    print(
        "pickle predictions match:",
        (original_predictions == pickle_predictions).all(),
    )


if __name__ == "__main__":
    main()