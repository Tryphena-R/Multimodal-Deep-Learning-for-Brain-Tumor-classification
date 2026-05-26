import pandas as pd
import numpy as np

# -----------------------------------
# Load Gene Dataset
# -----------------------------------

gene_df = pd.read_csv(
    "gene_expression_fixed.tsv",
    sep="\t"
)

print("Gene shape:", gene_df.shape)

# -----------------------------------
# Load Synthetic MRI Dataset
# -----------------------------------

mri_df = pd.read_csv(
    "synthetic_mri_only.tsv",
    sep="\t"
)

print("MRI shape:", mri_df.shape)

# -----------------------------------
# Load Real Labels
# -----------------------------------

labels_df = pd.read_csv(
    "real_labels.tsv",
    sep="\t"
)

labels_df = labels_df[["label"]]

print("Labels shape:", labels_df.shape)

# -----------------------------------
# Match sample counts
# -----------------------------------

min_samples = min(
    len(gene_df),
    len(mri_df),
    len(labels_df)
)

gene_df = gene_df.iloc[:min_samples]
mri_df = mri_df.iloc[:min_samples]
labels_df = labels_df.iloc[:min_samples]

# -----------------------------------
# Early Fusion
# -----------------------------------

fusion_df = pd.concat(
    [gene_df, mri_df],
    axis=1
)

# -----------------------------------
# Add labels
# -----------------------------------

fusion_df["label"] = labels_df["label"]

print("Final dataset shape:", fusion_df.shape)

# -----------------------------------
# Save dataset
# -----------------------------------

fusion_df.to_csv(
    "Gene-Expression_dataset.tsv",
    sep="\t",
    index=False
)

print("✅ Final frontend dataset created.")
