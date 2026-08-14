"""
W2D5 - End-to-End Preprocessing Pipeline
Titanic Dataset

EDA → Missing Value Handling → Encoding → Scaling → Export
"""

import os
import logging
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# 1. CREATE OUTPUT FOLDER AUTOMATICALLY
# =========================================================

DATA_PATH = "titanic.csv"
OUTPUT_DIR = "w2d5_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info("Output folder ready: %s", OUTPUT_DIR)


# =========================================================
# 2. LOAD DATASET WITH ERROR HANDLING
# =========================================================

try:
    df = pd.read_csv(DATA_PATH)

except FileNotFoundError:
    logger.error(
        "Dataset not found: %s. Please make sure titanic.csv "
        "exists in the project folder.",
        DATA_PATH
    )
    raise

except Exception as error:
    logger.error("Error while loading dataset: %s", error)
    raise


logger.info("Dataset loaded successfully: %s", DATA_PATH)

print("\n" + "=" * 60)
print("TITANIC DATASET")
print("=" * 60)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# =========================================================
# 3. EDA
# =========================================================

print("\n" + "=" * 60)
print("EDA")
print("=" * 60)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset description:")
print(df.describe())


# Automatically create EDA report
with open(
    os.path.join(OUTPUT_DIR, "eda_report.txt"),
    "w",
    encoding="utf-8"
) as file:

    file.write("W2D5 TITANIC DATASET - EDA REPORT\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Dataset shape: {df.shape}\n\n")

    file.write("Column names:\n")
    file.write(str(df.columns.tolist()))
    file.write("\n\n")

    file.write("Missing values:\n")
    file.write(str(df.isnull().sum()))
    file.write("\n\n")

    file.write("Statistical summary:\n")
    file.write(str(df.describe()))

logger.info("EDA report created successfully")


# =========================================================
# 4. MISSING VALUE REPORT
# =========================================================

missing_report = df.isnull().sum().reset_index()

missing_report.columns = [
    "Column",
    "Missing_Values"
]

missing_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "missing_values_report.csv"
    ),
    index=False
)

logger.info("Missing value report created successfully")


# =========================================================
# 5. SELECT FEATURES AND TARGET
# =========================================================

target = "Survived"

drop_columns = [
    "PassengerId",
    "Name",
    "Ticket"
]

X = df.drop(
    columns=[target] + drop_columns
)

y = df[target]


# =========================================================
# 6. IDENTIFY COLUMN TYPES
# =========================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# =========================================================
# 7. NUMERICAL PIPELINE
# =========================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# =========================================================
# 8. CATEGORICAL PIPELINE
# =========================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# =========================================================
# 9. COMPLETE PREPROCESSING PIPELINE
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# =========================================================
# 10. TRANSFORM DATA
# =========================================================

try:
    X_processed = preprocessor.fit_transform(X)

except Exception as error:
    logger.error(
        "Error during preprocessing: %s",
        error
    )
    raise


feature_names = preprocessor.get_feature_names_out()

processed_df = pd.DataFrame(
    X_processed,
    columns=feature_names
)

# Add target column
processed_df[target] = y.values

logger.info("Preprocessing completed successfully")


# =========================================================
# 11. EXPORT ML-READY DATASET
# =========================================================

ml_ready_path = os.path.join(
    OUTPUT_DIR,
    "titanic_ml_ready.csv"
)

processed_df.to_csv(
    ml_ready_path,
    index=False
)

logger.info(
    "ML-ready dataset created: %s",
    ml_ready_path
)

print("\nML-ready dataset created:")
print(ml_ready_path)


# =========================================================
# 12. VALIDATION TESTS
# =========================================================

print("\n" + "=" * 60)
print("VALIDATION TESTS")
print("=" * 60)

# Test 1
assert len(processed_df) == len(df)
print("PASS: Row count preserved")

# Test 2
assert processed_df.isnull().sum().sum() == 0
print("PASS: No missing values")

# Test 3
assert target in processed_df.columns
print("PASS: Target column exists")

# Test 4
assert all(
    pd.api.types.is_numeric_dtype(dtype)
    for dtype in processed_df.drop(
        columns=[target]
    ).dtypes
)

print("PASS: All features are numeric")

# Test 5
assert os.path.exists(ml_ready_path)
print("PASS: ML-ready file created")


# =========================================================
# 13. CREATE SUMMARY FILE
# =========================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "preprocessing_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("W2D5 PREPROCESSING SUMMARY\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Original dataset shape: {df.shape}\n")
    file.write(
        f"Processed dataset shape: {processed_df.shape}\n\n"
    )

    file.write(
        "Numerical features:\n"
        + str(numerical_features)
        + "\n\n"
    )

    file.write(
        "Categorical features:\n"
        + str(categorical_features)
        + "\n\n"
    )

    file.write("Preprocessing performed:\n")
    file.write("- Missing values handled\n")
    file.write("- Categorical features encoded\n")
    file.write("- Numerical features scaled\n")
    file.write("- ML-ready dataset exported\n")


logger.info("Preprocessing summary created successfully")


# =========================================================
# 14. FINAL OUTPUT
# =========================================================

print("\n" + "=" * 60)
print("W2D5 PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nCreated output files:")

for file_name in os.listdir(OUTPUT_DIR):
    print("✓", file_name)

print("\nFinal dataset shape:", processed_df.shape)

logger.info("W2D5 pipeline completed successfully")