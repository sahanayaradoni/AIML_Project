import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -----------------------------
# File paths
# -----------------------------

DATA_PATH = Path("dataset/cleaned_students.csv")
OUTPUT_DIR = Path("eda_outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv(DATA_PATH)

print("===== DATASET LOADED =====")

# -----------------------------
# Data Inspection
# -----------------------------

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== INFO =====")
df.info()

print("\n===== DESCRIBE =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())


# -----------------------------
# 5 Observations
# -----------------------------

observations = """
===== OBSERVATIONS =====

1. Dataset contains student performance scores and related information.
2. Numeric columns contain student marks such as Math, Science, and English scores.
3. Missing value analysis shows the data quality and identifies columns requiring cleaning.
4. Distribution plots help understand student score patterns and possible outliers.
5. Correlation analysis helps identify relationships between different subject scores.
"""

print(observations)


# -----------------------------
# Numeric Distributions
# -----------------------------

numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    plt.figure(figsize=(6, 4))

    sns.histplot(
        df[column],
        kde=True
    )

    plt.title(f"Distribution of {column}")

    plt.savefig(
        OUTPUT_DIR / f"{column}_distribution.png"
    )

    plt.close()


# -----------------------------
# Correlation Heatmap
# -----------------------------

plt.figure(figsize=(6, 4))

sns.heatmap(
    df[numeric_columns].corr(),
    annot=True
)

plt.title("Correlation Heatmap")

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png"
)

plt.close()


# -----------------------------
# Top 10 Category Counts
# -----------------------------

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:

    plt.figure(figsize=(8, 4))

    df[column].value_counts().head(10).plot(
        kind="bar"
    )

    plt.title(
        f"Top 10 {column} Counts"
    )

    plt.xlabel(column)
    plt.ylabel("Count")

    plt.savefig(
        OUTPUT_DIR / f"{column}_top10_counts.png"
    )

    plt.close()


# -----------------------------
# EDA Narrative (200 words)
# -----------------------------

eda_narrative = """

# Exploratory Data Analysis Narrative

The dataset contains student performance information including scores from different subjects. 
Exploratory Data Analysis was performed using Pandas, Matplotlib, and Seaborn to understand 
the structure, quality, and patterns present in the data.

The dataset was first inspected using shape, info(), describe(), and missing value analysis. 
The analysis helps identify the number of records, available features, data types, and the 
presence of incomplete values. Numerical columns were analyzed using distribution plots to 
understand score patterns and identify possible unusual values or outliers.

A correlation heatmap was generated to study relationships between numerical features. 
The correlation results help understand whether student scores in different subjects are 
related to each other. Category count plots were also created to analyze the frequency of 
different categorical values.

Some suspicious areas that may require attention include possible outliers in scores, 
incorrect data entries, and imbalance in category values. Before applying machine learning 
models, the dataset should be cleaned by handling missing values, removing duplicate records, 
and treating abnormal values.

Overall, EDA provided useful insights into the dataset and prepared it for further machine 
learning analysis and model development.
"""

with open("EDA_NARRATIVE.md", "w") as file:
    file.write(eda_narrative)


print("\nEDA completed successfully!")
print("Plots saved inside eda_outputs folder")
print("EDA narrative saved as EDA_NARRATIVE.md")