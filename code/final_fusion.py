import pandas as pd

# -----------------------------------
# Load gene features
# -----------------------------------

gene_df = pd.read_csv(
    "gene_features_case_id.tsv",
    sep="\t",
    index_col=0
)

print("Gene shape:", gene_df.shape)

# -----------------------------------
# Load MRI features
# -----------------------------------

mri_df = pd.read_csv(
    "mri_patient_features.tsv",
    sep="\t",
    index_col=0
)

print("MRI shape:", mri_df.shape)

# -----------------------------------
# Load real labels
# -----------------------------------

labels_df = pd.read_csv(
    "real_labels.tsv",
    sep="\t"
)

print("Labels shape:", labels_df.shape)

# -----------------------------------
# Keep only Case ID + label
# -----------------------------------

labels_df = labels_df[[
    "Case ID",
    "label"
]]

labels_df = labels_df.rename(
    columns={"Case ID": "Case_ID"}
)

# -----------------------------------
# Reset indexes
# -----------------------------------

gene_df = gene_df.reset_index(drop=True)
mri_df = mri_df.reset_index(drop=True)

# -----------------------------------
# Match minimum sample count
# -----------------------------------

min_samples = min(
    len(gene_df),
    len(mri_df),
    len(labels_df)
)

gene_df = gene_df.iloc[:min_samples]
mri_df = mri_df.iloc[:min_samples]
labels_df = labels_df.iloc[:min_samples]

print("Matched samples:", min_samples)

# -----------------------------------
# Fuse features
# -----------------------------------

fusion_df = pd.concat(
    [gene_df, mri_df],
    axis=1
)

# -----------------------------------
# Add REAL labels
# -----------------------------------

fusion_df["label"] = labels_df["label"].values

print("Fusion shape:", fusion_df.shape)

# -----------------------------------
# Save final multimodal dataset
# -----------------------------------

fusion_df.to_csv(
    "demo_multimodal_dataset.tsv",
    sep="\t",
    index=False
)

print("✅ Real multimodal fusion dataset created.")
