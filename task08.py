import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
# Update path if dataset is local, otherwise download the Mall Customer dataset
file_path =r"C:\Users\Divya P\Downloads\Mall_Customers.csv" # 👈 update with your path
df = pd.read_csv(file_path)

print("Dataset Head:")
print(df.head())

# -------------------------------
# Step 2: Preprocess Data
# -------------------------------
# Select numeric columns for clustering (e.g., Age, Annual Income, Spending Score)
X = df.select_dtypes(include=[np.number])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Step 3: Elbow Method
# -------------------------------
inertia = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

# -------------------------------
# Step 4: Fit KMeans with Optimal K
# -------------------------------
optimal_k = 5   # 👈 choose based on elbow graph
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

# -------------------------------
# Step 5: Visualization (PCA for 2D)
# -------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette="tab10", s=60)
plt.title("Clusters Visualization (PCA 2D)")
plt.show()

# -------------------------------
# Step 6: Silhouette Score
# -------------------------------
score = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {score:.3f}")