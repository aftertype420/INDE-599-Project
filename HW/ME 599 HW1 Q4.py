import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Load dataset
df = pd.read_csv("ai4i2020.csv")

# Use only continuous variables
X = df[
    [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]
].copy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# K-MEANS FOR MULTIPLE k VALUES
# -----------------------------
k_values = range(2, 11)
inertias = []
sil_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)

    print(f"k={k}, inertia={kmeans.inertia_:.2f}, silhouette={sil:.4f}")

# Plot elbow curve
plt.figure(figsize=(7,5))
plt.plot(list(k_values), inertias, marker='o')
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-means")
plt.grid(True)
plt.show()

# Plot silhouette scores
plt.figure(figsize=(7,5))
plt.plot(list(k_values), sil_scores, marker='o')
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs k")
plt.grid(True)
plt.show()

# -----------------------------
# CHECK LABEL AGREEMENT
# -----------------------------
for k in [3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    ari_type = adjusted_rand_score(df["Type"], cluster_labels)
    nmi_type = normalized_mutual_info_score(df["Type"], cluster_labels)

    ari_fail = adjusted_rand_score(df["Machine failure"], cluster_labels)
    nmi_fail = normalized_mutual_info_score(df["Machine failure"], cluster_labels)

    print(f"\nResults for k={k}")
    print(f"Type label -> ARI: {ari_type:.6f}, NMI: {nmi_type:.6f}")
    print(f"Machine failure label -> ARI: {ari_fail:.6f}, NMI: {nmi_fail:.6f}")

# -----------------------------
# PCA
# -----------------------------
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained)

print("\nExplained variance ratio:")
for i, val in enumerate(explained, start=1):
    print(f"PC{i}: {val:.4f}")

print("\nCumulative explained variance:")
for i, val in enumerate(cum_explained, start=1):
    print(f"PC1 to PC{i}: {val:.4f}")

# Plot cumulative explained variance
plt.figure(figsize=(7,5))
plt.plot(range(1, len(cum_explained)+1), cum_explained, marker='o')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.grid(True)
plt.show()