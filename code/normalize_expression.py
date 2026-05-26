import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load merged gene matrix
# -----------------------------

df = pd.read_csv(
    "final_gene_expression.tsv",
    sep="\t",
    index_col=0
)

print("Original shape:", df.shape)

# -----------------------------
# Transpose matrix
# Samples x Genes
# -----------------------------

X = df.T

print("After transpose:", X.shape)

# -----------------------------
# Log2 normalization
# -----------------------------

X_log = np.log2(X + 1)

print("Log normalization complete.")

# -----------------------------
# Standard scaling
# -----------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_log)

# Reduce memory usage
X_scaled = X_scaled.astype("float32")

print("Scaling complete.")
print("Scaled shape:", X_scaled.shape)

# -----------------------------
# Save normalized dataset
# -----------------------------

np.save(
    "gene_expression_scaled.npy",
    X_scaled
)

print("✅ Normalization complete.")
