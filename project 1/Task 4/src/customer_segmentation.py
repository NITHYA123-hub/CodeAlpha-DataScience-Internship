

import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import joblib # type: ignore

from sklearn.cluster import KMeans # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/Mall_Customers.csv")

print("=" * 60)
print("CUSTOMER SEGMENTATION USING K-MEANS")
print("=" * 60)

# ==========================================
# Dataset Overview
# ==========================================

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================
# Feature Selection
# ==========================================

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# ==========================================
# Feature Scaling
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# Train K-Means Model
# ==========================================

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nK-Means Model Trained Successfully!")

# ==========================================
# Cluster Summary
# ==========================================

print("\nCustomers in Each Cluster")
print(df["Cluster"].value_counts())

# ==========================================
# Visualization
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    s=80
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")

plt.colorbar(label="Cluster")

plt.show()

# ==========================================
# Save Model
# ==========================================

joblib.dump(kmeans, "models/kmeans_model.pkl")

print("\nModel Saved Successfully!")
print("Location: models/kmeans_model.pkl")