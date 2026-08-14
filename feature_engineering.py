import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from sklearn.feature_selection import SelectKBest, f_classif

# Load dataset
df = pd.read_csv("indian_dataset.csv")

# Display dataset information
print("===== DATASET =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== SHAPE =====")
print(df.shape)

# Label Encoding
label_encoder = LabelEncoder()

df["Gender_Label"] = label_encoder.fit_transform(df["Gender"])

print("\n===== LABEL ENCODING =====")
print(df[["Gender", "Gender_Label"]])

# One-Hot Encoding
one_hot_encoder = OneHotEncoder(sparse_output=False)

gender_encoded = one_hot_encoder.fit_transform(df[["Gender"]])

one_hot_columns = one_hot_encoder.get_feature_names_out(["Gender"])

one_hot_df = pd.DataFrame(
    gender_encoded,
    columns=one_hot_columns
)

print("\n===== ONE-HOT ENCODING =====")
print(one_hot_df)

# Ordinal Encoding
ordinal_encoder = OrdinalEncoder()

df["City_Ordinal"] = ordinal_encoder.fit_transform(df[["City"]])

print("\n===== ORDINAL ENCODING =====")
print(df[["City", "City_Ordinal"]])

# Encoding Trade-offs
print("\n===== ENCODING TRADE-OFFS =====")
print("LabelEncoder: Simple for binary/target categories, but numerical values may imply an order.")
print("OneHotEncoder: Avoids false ordering, but increases the number of features.")
print("OrdinalEncoder: Useful when categories have a meaningful order; otherwise it may introduce false relationships.")

# Standard Scaling
standard_scaler = StandardScaler()

df["Math_Standard"] = standard_scaler.fit_transform(
    df[["Math_Score"]]
)

print("\n===== STANDARD SCALER =====")
print(df[["Math_Score", "Math_Standard"]])

# Min-Max Scaling
min_max_scaler = MinMaxScaler()

df["Math_MinMax"] = min_max_scaler.fit_transform(
    df[["Math_Score"]]
)

print("\n===== MIN-MAX SCALER =====")
print(df[["Math_Score", "Math_MinMax"]])

# Robust Scaling
robust_scaler = RobustScaler()

df["Math_Robust"] = robust_scaler.fit_transform(
    df[["Math_Score"]]
)

print("\n===== ROBUST SCALER =====")
print(df[["Math_Score", "Math_Robust"]])

# Plot distributions before and after scaling
plt.figure(figsize=(10, 6))

plt.hist(df["Math_Score"], alpha=0.5, label="Original")
plt.hist(df["Math_Standard"], alpha=0.5, label="StandardScaler")
plt.hist(df["Math_MinMax"], alpha=0.5, label="MinMaxScaler")
plt.hist(df["Math_Robust"], alpha=0.5, label="RobustScaler")

plt.title("Math Score Distributions Before and After Scaling")
plt.xlabel("Scaled Values")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()

plt.savefig("scaling_distributions.png")
plt.close()

print("\nScaling distribution plot saved successfully.")

# --------------------------------------------------
# 4. SELECTKBEST - TOP 5 FEATURES
# --------------------------------------------------

from sklearn.feature_selection import SelectKBest, f_regression

# Create features using useful columns
features = pd.get_dummies(
    df[["Math_Score", "Science_Score", "Gender", "City"]],
    drop_first=False
)

# Target variable
target = df["English_Score"]

# Select top 5 features
selector = SelectKBest(score_func=f_regression, k=5)
selector.fit(features, target)

# Get selected feature names
selected_features = features.columns[selector.get_support()]

# Get feature scores
feature_scores = pd.DataFrame({
    "Feature": features.columns,
    "Score": selector.scores_
})

feature_scores = feature_scores.sort_values(
    by="Score",
    ascending=False
)

print("\n===== SELECTKBEST - TOP 5 FEATURES =====")
print(feature_scores.head(5))

# --------------------------------------------------
# 5. TOP 5 FEATURE EXPLANATION
# --------------------------------------------------

print("\n===== TOP 5 FEATURE EXPLANATION =====")

print("1. Math_Score - Strongest relationship with English_Score.")
print("2. Science_Score - Shows a strong relationship with English_Score.")
print("3. Gender_Male - Shows some relationship with English_Score in this dataset.")
print("4. Gender_Female - Shows the same relationship because it is the complementary gender category.")
print("5. City_Delhi - Shows a smaller relationship with English_Score compared with the other selected features.")