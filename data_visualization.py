import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder
os.makedirs("visualization_outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("indian_dataset.csv")

# Display dataset information
print(df.head())
print(df.info())

# Set seaborn style
sns.set_style("whitegrid")

# -------------------------------
# 1. Histogram - Math Score
# -------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(df["Math_Score"], bins=5, kde=True)
plt.title("Distribution of Math Scores")
plt.xlabel("Math Score")
plt.ylabel("Frequency")
plt.savefig("visualization_outputs/histogram_math_score.png")
plt.close()

# -------------------------------
# 2. Box Plot - Science Score
# -------------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(y=df["Science_Score"])
plt.title("Box Plot of Science Scores")
plt.savefig("visualization_outputs/boxplot_science_score.png")
plt.close()

# -------------------------------
# 3. Scatter Plot - Math vs Science
# -------------------------------
plt.figure(figsize=(6, 4))
sns.scatterplot(x="Math_Score", y="Science_Score", data=df)
plt.title("Math Score vs Science Score")
plt.savefig("visualization_outputs/scatter_math_science.png")
plt.close()

# -------------------------------
# 4. Count Plot - Gender
# -------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="Gender", data=df)
plt.title("Gender Distribution")
plt.savefig("visualization_outputs/countplot_gender.png")
plt.close()

# -------------------------------
# 5. Correlation Heatmap
# -------------------------------
plt.figure(figsize=(6, 4))
numeric_df = df[["Student_ID", "Math_Score", "Science_Score", "English_Score"]]
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("visualization_outputs/correlation_heatmap.png")
plt.close()

print("\nData visualization completed successfully!")
print("Plots saved inside visualization_outputs folder.")