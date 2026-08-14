"""
Tests for W2D5 Titanic preprocessing pipeline.
"""

import os

import pandas as pd


OUTPUT_DIR = "w2d5_outputs"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "titanic_ml_ready.csv"
)


def test_output_file_exists():
    """Verify that the ML-ready output file was created."""
    assert os.path.exists(OUTPUT_FILE)


def test_no_missing_values():
    """Verify that the processed dataset has no missing values."""
    df = pd.read_csv(OUTPUT_FILE)

    assert df.isnull().sum().sum() == 0


def test_row_count_preserved():
    """Verify that preprocessing preserves all Titanic rows."""
    df = pd.read_csv(OUTPUT_FILE)

    assert len(df) == 891


def test_target_column_exists():
    """Verify that the target column is present."""
    df = pd.read_csv(OUTPUT_FILE)

    assert "Survived" in df.columns


def test_features_are_numeric():
    """Verify that all processed features are numeric."""
    df = pd.read_csv(OUTPUT_FILE)

    features = df.drop(columns=["Survived"])

    assert all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in features.dtypes
    )