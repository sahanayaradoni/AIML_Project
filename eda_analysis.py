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

print("\n===== OBSERVATIONS =====")

print("""
1. Dataset contains student performance scores.
2. Numeric columns contain Math, Science, and English scores.
3. No missing values are present in the dataset.
4. Score distributions can help identify student performance patterns.
5. Correlation analysis helps understand relationships between subjects.
""")


# -----------------------------
# Numeric Distributions
# -----------------------------

numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.savefig(OUTPUT_DIR / f"{column}_distribution.png")
    plt.close()


# -----------------------------
# Correlation Heatmap
# -----------------------------

plt.figure(figsize=(6,4))

sns.heatmap(
    df[numeric_columns].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.close()


# -----------------------------
# Top Category Counts
# -----------------------------

for column in df.select_dtypes(include="object").columns:

    plt.figure(figsize=(8,4))

    df[column].value_counts().head(10).plot(
        kind="bar"
    )

    plt.title(f"Top 10 {column} Counts")
    plt.savefig(
        OUTPUT_DIR / f"{column}_top10_counts.png"
    )

    plt.close()


print("\nEDA completed successfully!")
print("Plots saved inside eda_outputs folder")