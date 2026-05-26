import numpy as np

# -----------------------------
# Load normalized dataset
# -----------------------------

X = np.load("gene_expression_scaled.npy")

print("Original shape:", X.shape)

# -----------------------------
# Compute variance of each gene
# -----------------------------

gene_variance = np.var(X, axis=0)

# -----------------------------
# Select top 15,000 genes
# -----------------------------

top_gene_indices = np.argsort(gene_variance)[-15000:]

# -----------------------------
# Reduce dataset
# -----------------------------

X_reduced = X[:, top_gene_indices]

print("Reduced shape:", X_reduced.shape)

# -----------------------------
# Save reduced dataset
# -----------------------------

np.save(
    "gene_expression_reduced.npy",
    X_reduced
)

print("✅ Gene reduction complete.")
