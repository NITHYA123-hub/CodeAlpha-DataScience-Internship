

import pandas as pd

# Load Dataset
df = pd.read_csv("data/titanic.csv")

print("="*50)
print("TITANIC DATA CLEANING")
print("="*50)

# Dataset Overview
print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

# Standardize Column Names
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
)

print("\nColumn Names")
print(df.columns)

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Remove Duplicate Rows
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

df.drop_duplicates(inplace=True)

# Fill Missing Numerical Values
num_cols = df.select_dtypes(include=["int64","float64"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill Missing Categorical Values
cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Convert Data Types
df["survived"] = df["survived"].astype(int)
df["pclass"] = df["pclass"].astype(int)

# Remove Outliers using IQR
numeric = ["age","fare"]

for col in numeric:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("\nFinal Shape")
print(df.shape)

print("\nRemaining Missing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

print("\nCleaning Completed Successfully!")


