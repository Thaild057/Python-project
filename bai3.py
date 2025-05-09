# Import additional libraries for clustering and visualization
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

base_dir = r"C:\Users\luong\Desktop\baitaplon trr"
# Load the saved result.csv
result_path = os.path.join(base_dir, "result.csv")
df = pd.read_csv(result_path, encoding="utf-8-sig")

# Select numerical features for clustering (exclude non-numeric and identifier columns)
numeric_features = [
    "Age", "Matches Played", "Starts", "Minutes", "Gls", "Ast", "crdY", "crdR",
    "xG", "xAG", "PrgC", "PrgP", "PrgR", "Gls per 90", "Ast per 90", "xG per 90", "xAG per 90",
    "Cmp", "Cmp%", "TotDist", "ShortCmp%", "MedCmp%", "LongCmp%", "KP", "Pass into 1_3", "PPA", "CrsPA",
    "SCA", "SCA90", "GCA", "GCA90", "Tkl", "TklW", "Deff Att", "Lost", "Blocks", "Sh", "Pass", "Int",
    "Touches", "Def Pen", "Def 3rd", "Mid 3rd", "Att 3rd", "Att Pen", "Take-Ons Att", "Succ%", "Tkld%",
    "Carries", "ProDist", "Carries 1_3", "CPA", "Mis", "Dis", "Rec", "Rec PrgR",
    "Fls", "Fld", "Off", "Crs", "Recov", "Aerl Won", "Aerl Lost", "Aerl Won%"
]

# Filter the dataframe to include only numeric features and drop rows with missing values
clustering_data = df[numeric_features].dropna()

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
max_clusters = 10
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, max_clusters + 1), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
elbow_plot_path = os.path.join(base_dir, "elbow_plot.png")
plt.savefig(elbow_plot_path)
plt.close()
print(f"📊 Elbow plot saved to {elbow_plot_path}")

# Choose the number of clusters (e.g., based on the elbow plot, let's assume 4 clusters)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(scaled_data)

# Add cluster labels to the original dataframe (only for rows without missing values)
clustering_data['Cluster'] = cluster_labels
df = df.loc[clustering_data.index, :].copy()  # Align with non-missing data
df['Cluster'] = cluster_labels

# Apply PCA to reduce dimensions to 2D
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
explained_variance = pca.explained_variance_ratio_.sum()
print(f"📊 PCA Explained Variance Ratio (2 components): {explained_variance:.2%}")

# Create a 2D scatter plot of the clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.title('2D PCA Cluster Visualization of Players')
plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
cluster_plot_path = os.path.join(base_dir, "cluster_plot.png")
plt.savefig(cluster_plot_path)
plt.close()
print(f"📊 Cluster plot saved to {cluster_plot_path}")

# Analyze cluster characteristics
print("\n📊 Cluster Sizes:")
print(df['Cluster'].value_counts())
print("\n📊 Mean Statistics by Cluster:")
cluster_summary = df.groupby('Cluster')[numeric_features].mean().round(2)
print(cluster_summary.to_string())

# Save the dataframe with cluster labels
clustered_result_path = os.path.join(base_dir, "clustered_result.csv")
df.to_csv(clustered_result_path, index=False, encoding="utf-8-sig", na_rep="N/A")
print(f"✅ Successfully saved clustered data to {clustered_result_path}")