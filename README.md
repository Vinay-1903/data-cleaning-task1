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

Interview Questions & Answers

### 1. What are the different types of missing data?
- **MCAR (Missing Completely at Random)**: No logical pattern behind the missing data.
- **MAR (Missing at Random)**: Missing values are related to other observed variables.
- **MNAR (Missing Not at Random)**: Missing values are related to the missing variable itself.

### 2. How do you handle categorical variables?
- **Label Encoding** for ordinal data (e.g., Low, Medium, High).
- **One-Hot Encoding** for nominal data (e.g., cities, genders).
- **Target Encoding** can be used in some advanced scenarios.

### 3. What is the difference between normalization and standardization?
- **Normalization**: Rescales data to a [0, 1] range.
- **Standardization**: Centers data around the mean with a standard deviation of 1 (Z-score).

### 4. How do you detect outliers?
- **Visual methods**: Boxplots, scatter plots.
- **Statistical methods**: Z-score, IQR (Interquartile Range).

### 5. Why is preprocessing important in ML?
- Removes noise and inconsistencies.
- Converts raw data into a usable format.
- Improves model performance and accuracy.
- Prevents biased training and improves generalization.

### 6. What is one-hot encoding vs label encoding?
- **One-Hot Encoding**: Creates separate binary columns for each category.
- **Label Encoding**: Assigns an integer to each category (useful for ordinal data).

### 7. How do you handle data imbalance?
- **Oversampling**: Duplicate or synthetically generate minority class samples (e.g., SMOTE).
- **Undersampling**: Reduce majority class samples.
- **Class weights**: Adjust model loss function to penalize mistakes on the minority class more.

### 8. Can preprocessing affect model accuracy?
Yes, significantly! Good preprocessing ensures the model learns meaningful patterns and improves performance by:
- Removing noise,
- Scaling features properly,
- Encoding correctly,
- Handling outliers and imbalanced data.


---
This completes the data cleaning and preprocessing task.

