"""Tests for W4D3 model serialization."""

from pathlib import Path

import joblib
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from w4d3_model_serialization import (
    train_model,
    serialize_with_joblib,
    serialize_with_pickle,
    load_with_joblib,
    load_with_pickle,
    JOBLIB_PATH,
    PICKLE_PATH,
)


def test_joblib_serialization():
    """Verify that the model can be saved and loaded with joblib."""
    model, X_test, _ = train_model()

    serialize_with_joblib(model)
    loaded_model = load_with_joblib()

    assert JOBLIB_PATH.exists()
    assert (model.predict(X_test) == loaded_model.predict(X_test)).all()


def test_pickle_serialization():
    """Verify that the model can be saved and loaded with pickle."""
    model, X_test, _ = train_model()

    serialize_with_pickle(model)
    loaded_model = load_with_pickle()

    assert PICKLE_PATH.exists()
    assert (model.predict(X_test) == loaded_model.predict(X_test)).all()


def test_serialized_models_have_same_predictions():
    """Verify joblib and pickle models produce identical predictions."""
    model, X_test, _ = train_model()

    serialize_with_joblib(model)
    serialize_with_pickle(model)

    joblib_model = load_with_joblib()
    pickle_model = load_with_pickle()

    joblib_predictions = joblib_model.predict(X_test)
    pickle_predictions = pickle_model.predict(X_test)

    assert (joblib_predictions == pickle_predictions).all()