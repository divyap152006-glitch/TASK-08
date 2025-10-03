# TASK-08
 🧩 K-Means Clustering on Mall Customers Dataset

## 📌 Objective
The goal of this project is to perform customer segmentation using **K-Means Clustering** on the Mall Customers dataset.  
We use **scikit-learn, pandas, numpy, matplotlib, seaborn** for data preprocessing, clustering, and visualization.

---

## 🛠️ Tools & Libraries
- Python 3.x  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 📂 Dataset
We use the **Mall Customers dataset** which contains information about customers such as:
- CustomerID  
- Gender  
- Age  
- Annual Income (k$)  
- Spending Score (1–100)  

👉 Download dataset: [Mall Customers CSV](https://www.kaggle.com/datasets/shwetabh123/mall-customers)  

Update the `file_path` in the code with your dataset location:
```python
file_path = r"C:\Users\Divyap\Downloads\Mall_Customers.csv"
```

---

## 🚀 Steps in the Code

### 1. **Load Dataset**
```python
df = pd.read_csv(file_path)
print(df.head())
```

### 2. **Preprocessing**
- Select only numerical columns (Age, Annual Income, Spending Score).  
- Apply **StandardScaler** for normalization.  

### 3. **Elbow Method**
- Plot inertia vs. `k` (clusters) to find the optimal number of clusters.  

### 4. **Fit KMeans**
- Train KMeans with the optimal `k`.  
- Assign each customer to a cluster.  

### 5. **Visualization (PCA 2D)**
- Reduce features into 2D using **PCA**.  
- Visualize customer groups in 2D space.  

### 6. **Silhouette Score**
- Evaluate clustering quality using **Silhouette Score**.

---

## 📊 Example Outputs

### 🔹 Elbow Method Plot
![Elbow Method](images/elbow_plot.png)

### 🔹 Cluster Visualization (PCA 2D)
![Cluster Visualization](images/cluster_plot.png)

### 🔹 Silhouette Score
```
Silhouette Score: 0.55  (example)
```

---

## ✅ Conclusion
- K-Means successfully groups customers based on age, income, and spending behavior.  
- Businesses can use these segments for **targeted marketing strategies**.  

---

## 📜 Author
Developed as part of a **Machine Learning internship task**.  
