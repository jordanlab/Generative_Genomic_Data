import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- 1) Load your three CSVs ---
# Assumes: rows = samples, columns = features (e.g., SNPs or numeric variables)
df1 = pd.read_csv("/home/sverma87/Generative_Genomic_Data/data/BAQ/BAQ_unphased_tensor_subset.csv")
df2 = pd.read_csv("/home/sverma87/Generative_Genomic_Data/BAQ_generated_data.csv")
df3 = pd.read_csv("/home/sverma87/Generative_Genomic_Data/data/BAQ/extractedChrAll_unphased_tensor.csv")

# --- 2) Ensure they have identical feature columns (order matters) ---
# Align columns to their intersection
common = df1.columns.intersection(df2.columns).intersection(df3.columns)
df1 = df1[common].copy()
df2 = df2[common].copy()
df3 = df3[common].copy()

# --- 3) Stack samples and build a label vector for coloring ---
X = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
labels = np.array([0] * len(df1) + [1] * len(df2) + [2] * len(df3))  # 0=file1, 1=file2, 2=file3

# --- 4) Scale features (recommended for PCA) ---
X_scaled = StandardScaler().fit_transform(X)

# Still calculate PCA on the scaled data to get the principal components and explained variance ratio 
# W

# --- 5) Run PCA ---
pca = PCA(n_components=2, random_state=0)
PC = pca.fit_transform(X_scaled)
evr = pca.explained_variance_ratio_

# --- 6) Plot with different colors/markers for each file ---
plt.figure(figsize=(7.5, 6))
plt.scatter(PC[labels == 0, 0], PC[labels == 0, 1],
            c="#1f77b4", marker="o", alpha=0.75, edgecolor="white", linewidth=0.6, label="BAQ Subset")
plt.scatter(PC[labels == 1, 0], PC[labels == 1, 1],
            c="#ff7f0e", marker="^", alpha=0.75, edgecolor="white", linewidth=0.6, label="Generated Data")
plt.scatter(PC[labels == 2, 0], PC[labels == 2, 1],
            c="#2ca02c", marker="s", alpha=0.75, edgecolor="white", linewidth=0.6, label="Extracted Chr All")

plt.xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
plt.title("PCA of Genomic Data: Real vs Generated")
plt.legend(frameon=False)
plt.grid(alpha=0.15)
plt.tight_layout()
plt.show()

plt.savefig("pca_three_datasets.png", dpi=300, bbox_inches="tight")
pd.DataFrame(PC, columns=["PC1","PC2"]).assign(source=np.where(labels==0,"BAQ_Subset",np.where(labels==1,"Generated","Extracted_ChrAll"))).to_csv("pca_scores.csv", index=False)