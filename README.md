# Task 3 - Sales Prediction using Machine Learning

## 📌 Objective

Build a Machine Learning model to predict product sales using the Superstore dataset. The project demonstrates data preprocessing, feature engineering, model training, evaluation, and model saving.

---

## 📂 Project Structure

```
Task3_SalesPrediction/
│
├── data/
│   └── Superstore.csv
│
├── notebooks/
│   └── SalesPrediction.ipynb
│
├── src/
│   └── sales_prediction.py
│
├── models/
│   └── sales_model.pkl
│
├── requirements.txt
│
└── README.md
```

---

## 📊 Dataset

- **Dataset:** Superstore Sales Dataset
- **Target Variable:** Sales

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## 🚀 Project Workflow

1. Load the dataset
2. Explore the dataset
3. Handle missing values
4. Encode categorical features
5. Select features and target
6. Split the dataset into training and testing sets
7. Train a Random Forest Regressor model
8. Predict sales
9. Evaluate the model using:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R² Score
10. Visualize Actual vs Predicted Sales
11. Save the trained model

---

## 📈 Model Evaluation

The model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📦 Output

- Trained Machine Learning model (`sales_model.pkl`)
- Sales predictions
- Actual vs Predicted Sales visualization

---

## ▶️ How to Run

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Run the Python script:

```bash
python src/sales_prediction.py
```

Or open the notebook:

```
notebooks/SalesPrediction.ipynb
```

and run all cells.

---

## 👩‍💻 Author

Nithya L