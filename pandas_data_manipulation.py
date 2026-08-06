"""
Pandas Data Manipulation Task
-----------------------------
Operations:
- Load CSV dataset
- Display shape, data types, and first 10 rows
- Filter data
- GroupBy operation
- Merge operation
- Pivot table
- Export CSV and Parquet files
"""

import os
import pandas as pd


def main():

    # Load dataset
    file_path = os.path.join(
        os.path.dirname(__file__),
        "indian_dataset.csv"
    )

    df = pd.read_csv(file_path)

    # Display basic information
    print("Dataset Shape:")
    print(df.shape)

    print("\nData Types:")
    print(df.dtypes)

    print("\nFirst 10 Rows:")
    print(df.head(10))


    # Filter operation
    # Filtering students with Math score above 85
    filtered_df = df[df["Math_Score"] > 85]

    print("\nFiltered Data (Math Score > 85):")
    print(filtered_df)


    # GroupBy operation
    # Finding average scores city-wise
    grouped_df = df.groupby("City")[["Math_Score", "Science_Score", "English_Score"]].mean()

    print("\nGroupBy City Average Scores:")
    print(grouped_df)


    # Merge operation
    # Creating another dataframe for demonstration
    city_data = pd.DataFrame({
        "City": ["Bangalore", "Mumbai", "Delhi"],
        "State": ["Karnataka", "Maharashtra", "Delhi"]
    })

    merged_df = pd.merge(df, city_data, on="City", how="left")

    print("\nMerged Data:")
    print(merged_df)


    # Pivot Table operation
    pivot = pd.pivot_table(
        df,
        values="Math_Score",
        index="City",
        columns="Gender",
        aggfunc="mean"
    )

    print("\nPivot Table:")
    print(pivot)


    # Export cleaned dataframe
    cleaned_csv = "cleaned_students.csv"
    cleaned_parquet = "cleaned_students.parquet"

    df.to_csv(cleaned_csv, index=False)

    df.to_parquet(cleaned_parquet, index=False)


    # Compare file sizes
    csv_size = os.path.getsize(cleaned_csv)
    parquet_size = os.path.getsize(cleaned_parquet)

    print("\nFile Sizes:")
    print("CSV Size:", csv_size, "bytes")
    print("Parquet Size:", parquet_size, "bytes")


if __name__ == "__main__":
    main()