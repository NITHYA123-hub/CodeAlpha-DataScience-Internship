# Task 4 - Customer Segmentation using K-Means Clustering

## 📌 Objective

The objective of this project is to segment customers into different groups based on their purchasing behavior using the K-Means Clustering algorithm. This helps businesses understand customer patterns and create targeted marketing strategies.

---

## 📂 Project Structure

```
Task4_CustomerSegmentation/
│
├── data/
│   └── Mall_Customers.csv
│
├── notebooks/
│   └── CustomerSegmentation.ipynb
│
├── src/
│   └── customer_segmentation.py
│
├── models/
│   └── kmeans_model.pkl
│
├── requirements.txt
│
└── README.md
```

---

## 📊 Dataset

**Dataset:** Mall Customers Dataset

The dataset contains the following columns:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

---

## 🛠 Technologies Used

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
3. Check for missing values
4. Select important features
5. Scale the features using StandardScaler
6. Train the K-Means Clustering model
7. Assign cluster labels to customers
8. Visualize customer segments
9. Save the trained model
10. Save the clustered dataset

---

## 📈 Machine Learning Algorithm

**Algorithm Used:** K-Means Clustering

Number of Clusters: **5**

Selected Features:

- Annual Income (k$)
- Spending Score (1-100)

---

## 📊 Output

- Customer clusters
- Cluster visualization
- Trained model (`kmeans_model.pkl`)
- Clustered dataset (`customer_segments.csv`)

---

## ▶️ How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Python script:

```bash
python src/customer_segmentation.py
```

Or open the notebook:

```
notebooks/CustomerSegmentation.ipynb
```

Run all cells.

---

## 📚 Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

---

## 👩‍💻 Author

Nithya L
