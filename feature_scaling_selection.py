import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

# Load dataset
df = pd.read_csv("indian_dataset.csv")

print("Dataset loaded successfully!")
print(df.head())
print("\nColumns:")
print(df.columns)

# -------------------------------
# Categorical Feature Encoding
# -------------------------------

# Label Encoding
label_encoder = LabelEncoder()
df["Gender_Label"] = label_encoder.fit_transform(df["Gender"])

print("\nLabel Encoding - Gender:")
print(df[["Gender", "Gender_Label"]])

# One-Hot Encoding
one_hot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

city_encoded = one_hot_encoder.fit_transform(df[["City"]])
city_columns = one_hot_encoder.get_feature_names_out(["City"])

city_encoded_df = pd.DataFrame(
    city_encoded,
    columns=city_columns
)

print("\nOne-Hot Encoding - City:")
print(city_encoded_df)

# Ordinal Encoding
ordinal_encoder = OrdinalEncoder(
    categories=[[
        "Bangalore",
        "Chennai",
        "Delhi",
        "Hyderabad",
        "Kolkata",
        "Mumbai",
        "Pune"
    ]]
)

df["City_Ordinal"] = ordinal_encoder.fit_transform(df[["City"]])

print("\nOrdinal Encoding - City:")
print(df[["City", "City_Ordinal"]])

# -------------------------------
# Feature Scaling
# -------------------------------

numeric_features = [
    "Math_Score",
    "Science_Score",
    "English_Score"
]

# StandardScaler
standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(
    df[numeric_features]
)

standard_scaled_df = pd.DataFrame(
    standard_scaled,
    columns=[
        "Math_Standard",
        "Science_Standard",
        "English_Standard"
    ]
)

print("\nStandardScaler:")
print(standard_scaled_df)

# MinMaxScaler
minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(
    df[numeric_features]
)

minmax_scaled_df = pd.DataFrame(
    minmax_scaled,
    columns=[
        "Math_MinMax",
        "Science_MinMax",
        "English_MinMax"
    ]
)

print("\nMinMaxScaler:")
print(minmax_scaled_df)

# RobustScaler
robust_scaler = RobustScaler()
robust_scaled = robust_scaler.fit_transform(
    df[numeric_features]
)

robust_scaled_df = pd.DataFrame(
    robust_scaled,
    columns=[
        "Math_Robust",
        "Science_Robust",
        "English_Robust"
    ]
)

print("\nRobustScaler:")
print(robust_scaled_df)

# -------------------------------
# Scaling Distribution Plot
# -------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    df["Math_Score"],
    bins=5,
    alpha=0.6,
    label="Original Math Score"
)

plt.hist(
    standard_scaled_df["Math_Standard"],
    bins=5,
    alpha=0.6,
    label="StandardScaler"
)

plt.title("Math Score Distribution: Before vs After Scaling")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()

plt.savefig(
    "scaling_standard_comparison.png",
    dpi=150
)

plt.close()

print("\nScaling plot saved successfully!")

# -------------------------------
# SelectKBest - Top 5 Features
# -------------------------------

# Use Gender as an independent categorical target
# This avoids using Total_Score as a target derived
# directly from the score features.

target = df["Gender_Label"]

# Combine numerical and encoded categorical features
selection_features = pd.concat(
    [
        df[
            [
                "Math_Score",
                "Science_Score",
                "English_Score",
                "City_Ordinal"
            ]
        ],
        city_encoded_df
    ],
    axis=1
)

# Select top 5 features
selector = SelectKBest(
    score_func=f_classif,
    k=5
)

X_selected = selector.fit_transform(
    selection_features,
    target
)

selected_features = selection_features.columns[
    selector.get_support()
]

print("\nTop 5 Features:")
print(selected_features.tolist())

print("\nFeature Selection Scores:")

feature_scores = pd.DataFrame({
    "Feature": selection_features.columns,
    "Score": selector.scores_
})

print(
    feature_scores
    .sort_values("Score", ascending=False)
    .head(5)
)