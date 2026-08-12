# Self Review Checklist

## NumPy Fundamentals

✅ NumPy installed in virtual environment  
✅ Created 1D, 2D, and 3D arrays  
✅ Verified array shapes  
✅ Implemented broadcasting operations  
✅ Used vectorised operations without Python loops  
✅ Performed matrix multiplication  
✅ Calculated mean, standard deviation, and correlation  
✅ Used real CSV dataset  
✅ Created feature branch  
✅ Completed minimum 2 commits  
✅ Raised Pull Request

# Pandas Data Manipulation - Self Review

## Completed Tasks

✅ Loaded Indian CSV dataset using Pandas  
✅ Checked shape, data types, and first 10 rows  
✅ Implemented filtering operation  
✅ Implemented groupby operation  
✅ Implemented merge operation  
✅ Implemented pivot table operation  
✅ Exported cleaned DataFrame to CSV and Parquet  
✅ Compared CSV and Parquet file sizes

## Code Quality

✅ Used reusable Python structure  
✅ Added comments for Pandas operations  
✅ Used relative file paths  
✅ Maintained clean project structure

## CIA Review

✅ Completed CIA Mentor Mode review before final commit

## Output Evidence

✅ Added terminal output screenshot (`pandas_output.png`)  
✅ Verified CSV and Parquet export  
✅ Completed CIA Mentor review (2 interactions)

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
✅ Generated and verified W1D3 output evidence

## CIA Review

✅ Completed 2 CIA interactions

# W1D4 Exploratory Data Analysis (EDA)

## Completed Tasks

✅ Loaded cleaned dataset using Pandas  
✅ Performed dataset inspection using shape, info(), and describe()  
✅ Checked missing values and data quality  
✅ Generated numerical distribution plots  
✅ Created correlation heatmap  
✅ Created categorical feature count plots  
✅ Saved visualization outputs inside `eda_outputs` folder  
✅ Improved visualization quality using Seaborn styling  
✅ Implemented reusable plotting functions  
✅ Generated EDA narrative documentation  
✅ Verified EDA script execution successfully

## Code Quality Improvements

✅ Organized code using reusable functions  
✅ Added docstrings for functions  
✅ Used pathlib for file handling  
✅ Avoided repetitive plotting code  
✅ Added better figure saving options with proper resolution

## Output Evidence

✅ Generated EDA visualization files inside `eda_outputs/`  
✅ Created `EDA_NARRATIVE.md`  
✅ Verified distribution plots and correlation heatmap  
✅ Completed CIA Mentor review interactions

# Exploratory Data Analysis (EDA) Narrative

The dataset contains student performance information including scores from Math, Science, and English subjects. Exploratory Data Analysis was performed using Pandas, Matplotlib, and Seaborn to understand the structure, quality, and patterns present in the dataset.

The dataset was initially inspected using shape, info(), describe(), and missing value analysis. These steps helped identify the number of records, available features, data types, and possible data quality issues. The analysis confirmed that the dataset was clean and suitable for further exploration.

Numerical features were analyzed using distribution plots to understand score patterns and identify possible outliers. The correlation heatmap was generated to study relationships between different subject scores and understand how student performance varies across features.

Categorical features were also analyzed using count plots to understand category distribution and frequency. The generated visualizations provided better understanding of the dataset characteristics.

Overall, the dataset is clean and analysis-ready. The EDA process provided useful insights into student performance patterns and prepared the dataset for future machine learning tasks such as prediction models and statistical analysis.

# Git Workflow

✅ Created feature branch: `feat/aiml-W1-eda-sahana`  
✅ Maintained meaningful commit messages  
✅ Completed required commits  
✅ Raised Pull Request for review  
✅ Updated PR after CIA feedback and improvements

## W1D5: Data Visualisation — Matplotlib & Seaborn

✅ Created visualizations using Matplotlib and Seaborn.  
✅ Generated histogram, box plot, scatter plot, count plot, and correlation heatmap.  
✅ Saved visualization outputs as evidence.  
✅ Reviewed code with CIA and received improvement suggestions.  
✅ Committed and pushed changes to Git branch.

# W2D1: Feature Engineering & Encoding

## Completed Tasks

✅ Applied LabelEncoder on categorical data
✅ Applied OneHotEncoder on categorical data
✅ Applied OrdinalEncoder on categorical data
✅ Documented encoding trade-offs
✅ Applied StandardScaler
✅ Applied MinMaxScaler
✅ Applied RobustScaler
✅ Generated scaling distribution plot
✅ Used SelectKBest to identify top 5 features
✅ Documented why the top 5 features were selected
✅ Verified feature engineering script execution successfully

## Output Evidence

✅ Generated `scaling_distributions.png`
✅ Verified encoding and scaling outputs in terminal
✅ Verified SelectKBest top 5 feature results

## CIA Review

✅ Completed CIA Mentor Mode review

# W2D2: Feature Scaling & Selection

## Completed Tasks

✅ Applied LabelEncoder on categorical data  
✅ Applied OneHotEncoder on categorical data  
✅ Applied OrdinalEncoder on categorical data  
✅ Documented encoding trade-offs  
✅ Applied StandardScaler  
✅ Applied MinMaxScaler  
✅ Applied RobustScaler  
✅ Generated scaling distribution plot  
✅ Used SelectKBest to identify top 5 features  
✅ Documented why the top 5 features were selected  
✅ Verified feature engineering script execution successfully

## Output Evidence

✅ Generated `scaling_standard_comparison.png`  
✅ Verified encoding and scaling outputs in terminal  
✅ Verified SelectKBest top 5 feature results  
✅ Removed duplicate/empty `scaling_distributions.png`

## Code Quality

✅ Used reusable Python structure  
✅ Used appropriate preprocessing techniques  
✅ Added clear comments and documentation  
✅ Verified script execution successfully  
✅ Maintained clean project structure

# W2D3: Handling Imbalanced Data — SMOTE

## Completed Tasks

✅ Created an imbalanced binary classification dataset
✅ Split data into training and testing sets
✅ Checked class distribution before SMOTE
✅ Applied SMOTE only to the training data
✅ Balanced the minority class using synthetic samples
✅ Trained Logistic Regression model
✅ Evaluated the model using classification report
✅ Generated confusion matrix
✅ Generated before and after SMOTE visualizations
✅ Verified script execution successfully

## Results

- Training distribution before SMOTE:
  - Class 0: 715
  - Class 1: 85

- Training distribution after SMOTE:
  - Class 0: 715
  - Class 1: 715

- Model accuracy: 95%
- Minority class precision: 70%
- Minority class recall: 90%
- Minority class F1-score: 79%

## Output Evidence

✅ Generated `smote_before.png`
✅ Generated `smote_after.png`
✅ Verified terminal output

## Code Quality

✅ Used reusable Python structure
✅ Added comments explaining each step
✅ Applied SMOTE only after train-test split
✅ Kept test data untouched during SMOTE
✅ Verified successful script execution

## Key Learning

SMOTE (Synthetic Minority Over-sampling Technique) handles class imbalance by generating synthetic samples for the minority class. Applying SMOTE only to the training data helps prevent data leakage.

## CIA Review

✅ Completed CIA Mentor Mode review
