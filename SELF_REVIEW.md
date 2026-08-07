# Self Review Checklist

✅ NumPy installed in virtual environment  
✅ Created 1D, 2D, and 3D arrays  
✅ Verified array shapes  
✅ Implemented broadcasting  
✅ Used vectorised operations without Python loops  
✅ Performed matrix multiplication  
✅ Calculated mean, standard deviation, and correlation  
✅ Used real CSV dataset  
✅ Created feature branch  
✅ Completed minimum 2 commits  
✅ Raised Pull Request

# Pandas Data Manipulation - Self Review

## Completed Tasks

- Loaded Indian CSV dataset using Pandas
- Checked shape, data types, and first 10 rows
- Implemented filtering operation
- Implemented groupby operation
- Implemented merge operation
- Implemented pivot table operation
- Exported cleaned DataFrame to CSV and Parquet
- Compared file sizes

## Code Quality

- Used reusable Python structure
- Added comments for each Pandas operation
- Used relative file paths
- Maintained clean project structure

## CIA Review

Completed CIA Mentor Mode review before final commit.

## Output Evidence

- Added terminal output screenshot (pandas_output.png)
- Verified CSV and Parquet export
- Completed CIA Mentor review (2 interactions)

# W1D3 Data Loading, Cleaning & Inspection

## Completed Tasks

✅ Loaded CSV dataset using Pandas  
✅ Inspected shape, columns, data types, and statistics  
✅ Checked missing values and duplicate records  
✅ Cleaned dataset using reusable functions  
✅ Implemented missing value handling  
✅ Saved cleaned dataset successfully  
✅ Added logging and pathlib file handling  
✅ Tested script execution successfully  
✅ W1D3 output evidence generated and verified

## CIA Review

✅ Completed 2 CIA interactions

## W1D4 EDA Narrative

# Exploratory Data Analysis (EDA) Narrative

The dataset contains the scores of 10 students in three subjects: Math, Science, and English. The initial inspection using `df.info()` confirmed that all three columns are numeric (`int64`) and that there are no missing values. The dataset has a shape of 10 rows and 3 columns, making it clean and suitable for analysis.

The statistical summary generated with `df.describe()` shows that the average scores are approximately 82 in Math, 84.2 in Science, and 82.7 in English. The standard deviations indicate a moderate variation in student performance across all subjects. The minimum and maximum values show that the scores range from 65 to 92 in Math, 70 to 95 in Science, and 68 to 94 in English.

No duplicate records or missing values were found, so no additional cleaning was required before analysis. Distribution plots indicate that the scores are reasonably balanced without significant outliers. The correlation heatmap helps identify relationships between subjects and suggests that students who perform well in one subject also tend to perform well in the others.

Overall, the dataset is clean, complete, and analysis-ready. It provides a reliable foundation for further data visualization, statistical analysis, and future machine learning tasks such as student performance prediction and educational data modeling.
