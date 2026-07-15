
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/cleaned_titanic.csv")

print("=" * 50)
print("TITANIC EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# -----------------------------
# Basic Information
# -----------------------------
print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Age Distribution
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["age"], bins=20, kde=True, color="skyblue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# -----------------------------
# Gender Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="sex", data=df)
plt.title("Gender Distribution")
plt.show()

# -----------------------------
# Passenger Class Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="pclass", data=df)
plt.title("Passenger Class")
plt.show()

# -----------------------------
# Survival Count
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="survived", data=df)
plt.title("Survival Count")
plt.xticks([0,1],["Not Survived","Survived"])
plt.show()

# -----------------------------
# Correlation Heatmap
# -----------------------------
plt.figure(figsize=(8,6))

corr = df.select_dtypes(include=["number"]).corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

print("\nEDA Completed Successfully!")