import pandas as pd

# Load reduced gene dataset
df = pd.read_csv(
    "gene_expression_dataset.tsv",
    sep="\t"
)

# Rename columns
df.columns = [
    f"latent_{i}"
    for i in range(df.shape[1])
]

# Save fixed dataset
df.to_csv(
    "gene_expression_fixed.tsv",
    sep="\t",
    index=False
)

print("✅ Fixed gene dataset created.")
