"""
W1D3: Data Loading, Cleaning & Inspection

This script:
1. Loads student CSV dataset
2. Inspects dataset structure and quality
3. Handles missing values and duplicates
4. Saves cleaned dataset
"""

from pathlib import Path
import logging
import pandas as pd


# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# -----------------------------
# File Paths
# -----------------------------
BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "dataset" / "students.csv"

OUTPUT_PATH = BASE_DIR / "dataset" / "cleaned_students.csv"


# -----------------------------
# Load Dataset
# -----------------------------
def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load CSV file into Pandas DataFrame.

    Args:
        file_path: Location of CSV file.

    Returns:
        Loaded DataFrame.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    logger.info("Dataset loaded successfully")
    return df


# -----------------------------
# Inspect Dataset
# -----------------------------
def inspect_data(df: pd.DataFrame) -> None:
    """
    Display dataset information.
    """

    logger.info(f"Dataset Shape: {df.shape}")

    logger.info(
        f"\nColumn Data Types:\n{df.dtypes}"
    )

    logger.info(
        f"\nFirst 10 Rows:\n{df.head(10)}"
    )

    logger.info(
        f"\nMissing Values:\n{df.isnull().sum()}"
    )

    logger.info(
        f"\nDuplicate Rows: {df.duplicated().sum()}"
    )

    logger.info(
        f"\nStatistics:\n{df.describe()}"
    )


# -----------------------------
# Clean Dataset
# -----------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean missing values and duplicate records.
    """

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Handle missing values
    for column in df.columns:

        if df[column].dtype in ["int64", "float64"]:
            df[column] = df[column].fillna(
                df[column].mean()
            )

        else:
            df[column] = df[column].fillna("Missing")


    logger.info("Data cleaning completed")

    return df


# -----------------------------
# Save Cleaned Dataset
# -----------------------------
def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save cleaned dataframe as CSV.
    """

    df.to_csv(output_path, index=False)

    logger.info(
        f"Cleaned dataset saved at: {output_path}"
    )


# -----------------------------
# Main Pipeline
# -----------------------------
def main():

    df = load_data(DATA_PATH)

    print("\n----- BEFORE CLEANING -----")
    inspect_data(df)

    cleaned_df = clean_data(df)

    print("\n----- AFTER CLEANING -----")
    inspect_data(cleaned_df)

    save_data(cleaned_df, OUTPUT_PATH)


if __name__ == "__main__":
    main()