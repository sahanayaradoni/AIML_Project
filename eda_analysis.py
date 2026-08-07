from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# -------------------------------------------------
# Visualization Settings
# -------------------------------------------------

sns.set_theme(
    style="whitegrid",
    palette="colorblind"
)


# -------------------------------------------------
# File Paths
# -------------------------------------------------

DATA_PATH = Path("dataset/cleaned_students.csv")
OUTPUT_DIR = Path("eda_outputs")

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load CSV dataset using pandas.

    Args:
        path: Location of CSV file.

    Returns:
        Loaded pandas DataFrame.
    """

    df = pd.read_csv(path)

    return df


# -------------------------------------------------
# Data Inspection
# -------------------------------------------------

def inspect_dataset(df: pd.DataFrame) -> None:
    """
    Display dataset information and statistics.
    """

    print("\n===== DATASET SHAPE =====")
    print(df.shape)

    print("\n===== DATA TYPES =====")
    df.info()

    print("\n===== STATISTICAL SUMMARY =====")
    print(df.describe())

    print("\n===== MISSING VALUE PERCENTAGE =====")

    missing_percentage = (
        df.isnull()
        .mean()
        * 100
    )

    print(
        missing_percentage[
            missing_percentage > 0
        ]
    )


# -------------------------------------------------
# Numeric Distribution Plot
# -------------------------------------------------

def plot_numeric_distribution(
        df: pd.DataFrame,
        column: str,
        output_dir: Path
) -> None:
    """
    Create histogram and KDE plot
    for numeric columns.
    """

    plt.figure(
        figsize=(8, 5)
    )

    sns.histplot(
        df[column].dropna(),
        kde=True
    )

    plt.title(
        f"Distribution of {column}"
    )

    plt.xlabel(column)

    plt.ylabel(
        "Count"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir /
        f"{column}_distribution.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# -------------------------------------------------
# Correlation Heatmap
# -------------------------------------------------

def plot_correlation_heatmap(
        df: pd.DataFrame,
        numeric_columns,
        output_dir: Path
) -> None:
    """
    Generate correlation heatmap.
    """

    plt.figure(
        figsize=(8, 6)
    )

    correlation = (
        df[numeric_columns]
        .corr()
    )

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir /
        "correlation_heatmap.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# -------------------------------------------------
# Category Count Plot
# -------------------------------------------------

def plot_category_counts(
        df: pd.DataFrame,
        column: str,
        output_dir: Path
) -> None:
    """
    Create top category count plot.
    """

    plt.figure(
        figsize=(8, 5)
    )

    top_categories = (
        df[column]
        .value_counts()
        .head(10)
    )

    sns.barplot(
        x=top_categories.index,
        y=top_categories.values
    )

    plt.title(
        f"Top 10 {column} Counts"
    )

    plt.xlabel(column)

    plt.ylabel(
        "Count"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        output_dir /
        f"{column}_top10_counts.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# -------------------------------------------------
# EDA Narrative
# -------------------------------------------------

EDA_NARRATIVE = """
# Exploratory Data Analysis Narrative

The dataset contains student performance information including
scores from different subjects. Exploratory Data Analysis was
performed using Pandas, Matplotlib, and Seaborn to understand
dataset structure, quality, and patterns.

The dataset was inspected using shape, info(), describe(),
and missing value analysis. These steps helped identify the
number of records, available features, data types, and data
quality issues.

Numerical features were analyzed using distribution plots to
understand score patterns and identify possible outliers.
Correlation analysis was performed using a heatmap to study
relationships between numerical variables.

Categorical features were visualized using count plots to
understand category frequency and distribution.

The analysis showed that the dataset is suitable for further
machine learning tasks after necessary preprocessing.
Potential improvements before modeling include handling missing
values, checking duplicate records, and treating abnormal values.

Overall, EDA provided useful insights into student performance
patterns and prepared the dataset for future statistical
analysis and machine learning model development.
"""


# -------------------------------------------------
# Main Execution
# -------------------------------------------------

def main():

    df = load_dataset(
        DATA_PATH
    )

    print(
        "===== DATASET LOADED SUCCESSFULLY ====="
    )

    inspect_dataset(
        df
    )


    numeric_columns = (
        df.select_dtypes(
            include="number"
        )
        .columns
    )


    for column in numeric_columns:

        plot_numeric_distribution(
            df,
            column,
            OUTPUT_DIR
        )


    plot_correlation_heatmap(
        df,
        numeric_columns,
        OUTPUT_DIR
    )


    categorical_columns = (
        df.select_dtypes(
            include="object"
        )
        .columns
    )


    for column in categorical_columns:

        plot_category_counts(
            df,
            column,
            OUTPUT_DIR
        )


    with open(
        "EDA_NARRATIVE.md",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            EDA_NARRATIVE
        )


    print("\nEDA completed successfully!")
    print(
        "Plots saved inside eda_outputs folder"
    )
    print(
        "EDA narrative saved as EDA_NARRATIVE.md"
    )


if __name__ == "__main__":
    main()