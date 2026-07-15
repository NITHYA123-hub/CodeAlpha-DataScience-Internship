

import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import joblib # type: ignore

from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("data/superstore.csv")

print("=" * 60)
print("SALES PREDICTION USING MACHINE LEARNING")
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
# Remove Missing Values
# ==========================================

df.dropna(inplace=True)

# ==========================================
# Encode Categorical Columns
# ==========================================

encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

# ==========================================
# Feature Selection
# ==========================================

X = df.drop("Sales", axis=1)
y = df["Sales"]

# ==========================================
# Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Train Model
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Evaluation
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("-" * 30)

print("MAE :", round(mae,2))
print("RMSE:", round(rmse,2))
print("R2 Score:", round(r2,2))

# ==========================================
# Visualization
# ==========================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.show()

# ==========================================
# Save Model
# ==========================================

joblib.dump(model,"models/sales_model.pkl")

print("\nModel Saved Successfully!")
print("Location: models/sales_model.pkl")