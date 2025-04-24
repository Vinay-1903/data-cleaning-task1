# Data Cleaning & Preprocessing Task

## Objective
This project demonstrates how to clean and prepare raw data for machine learning using Python. The Titanic dataset is used as an example.

## Tools Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Steps Performed
1. Imported the Titanic dataset from an online source.
2. Explored basic dataset information including null values and data types.
3. Handled missing values by imputing median for numerical columns and mode for categorical columns.
4. Dropped columns with excessive missing values (e.g., Cabin).
5. Converted categorical features into numerical using label encoding and one-hot encoding.
6. Normalized and standardized numerical features using StandardScaler.
7. Visualized outliers using boxplots.
8. Removed outliers using the IQR method.
9. Saved the cleaned dataset as `titanic_cleaned.csv`.

## How to Run
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the data cleaning script:
   ```
   python Task-1.py
   ```

## What You'll Learn
- Data cleaning techniques
- Handling missing data
- Encoding categorical variables
- Feature scaling (normalization and standardization)
- Outlier detection and removal
- Importance of preprocessing in machine learning

## Notes
- The dataset is loaded directly from an online source.
- The cleaned dataset is saved locally for further use.

## Interview Questions Covered
- Types of missing data
- Handling categorical variables
- Difference between normalization and standardization
- Outlier detection methods
- Importance of preprocessing in ML
- One-hot encoding vs label encoding
- Handling data imbalance
- Effect of preprocessing on model accuracy

---
This completes the data cleaning and preprocessing task.

