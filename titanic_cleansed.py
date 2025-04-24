import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load Titanic dataset from URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 1. Explore basic info
print("Basic Info:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())

# 2. Handle missing values
# Fill Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)
# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
# Drop Cabin column due to many missing values
df.drop('Cabin', axis=1, inplace=True)

print("\nMissing values after imputation:")
print(df.isnull().sum())

# 3. Convert categorical features into numerical using encoding
# Convert Sex to binary
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
# One-hot encode Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# 4. Normalize/standardize numerical features
num_features = ['Age', 'Fare', 'SibSp', 'Parch']
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# 5. Visualize outliers using boxplots and remove them
plt.figure(figsize=(12,8))
for i, col in enumerate(num_features):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Remove outliers using IQR method
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

df_clean = remove_outliers(df, num_features)

print(f"\nShape before removing outliers: {df.shape}")
print(f"Shape after removing outliers: {df_clean.shape}")

# Save cleaned data to CSV (optional)
df_clean.to_csv("titanic_cleaned.csv", index=False)

print("\nData cleaning and preprocessing completed. Cleaned data saved as 'titanic_cleaned.csv'.")
