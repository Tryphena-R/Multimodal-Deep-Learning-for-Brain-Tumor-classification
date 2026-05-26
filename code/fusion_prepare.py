import pandas as pd
from sklearn.preprocessing import LabelEncoder

# -----------------------------------
# Load gene latent features
# -----------------------------------

gene_df = pd.read_csv(
    "gene_features_case_id.tsv",
    sep="\t",
    index_col=0
)

print("Gene shape:", gene_df.shape)

# -----------------------------------
# Load MRI metadata
# -----------------------------------

mri_df = pd.read_excel(
    "MRI_Aggregated.xlsx"
)

print("MRI shape:", mri_df.shape)

# -----------------------------------
# Clean Patient Age
# Example: 057Y -> 57
# -----------------------------------
mri_df["Patient Age"] = (
    mri_df["Patient Age"]
    .astype(str)
    .str.replace("Y", "")
)

mri_df["Patient Age"] = pd.to_numeric(
    mri_df["Patient Age"],
    errors="coerce"
)

mri_df["Patient Age"] = (
    mri_df["Patient Age"]
    .fillna(mri_df["Patient Age"].median())
)
print("\nExample Gene IDs:")
print(gene_df.index[:5])

print("\nExample MRI IDs:")
print(mri_df.index[:5])

# -----------------------------------
# Encode Sex
# M=1, F=0
# -----------------------------------

sex_encoder = LabelEncoder()

mri_df["Patient Sex"] = sex_encoder.fit_transform(
    mri_df["Patient Sex"]
)

# -----------------------------------
# Encode Study Description
# -----------------------------------

desc_encoder = LabelEncoder()

mri_df["Study Description"] = desc_encoder.fit_transform(
    mri_df["Study Description"].astype(str)
)

# -----------------------------------
# Rename Patient ID column
# -----------------------------------
# Remove extra spaces from column names
mri_df.columns = mri_df.columns.str.strip()

# Rename correctly
mri_df = mri_df.rename(
    columns={"Patient ID": "Case_ID"}
)

# Set index
mri_df = mri_df.set_index("Case_ID")

print("\nExample MRI IDs:")
print(mri_df.index[:5])

# -----------------------------------
# Drop Study Date
# -----------------------------------

mri_df = mri_df.drop(
    columns=["Study Date"]
)

print("MRI cleaned shape:", mri_df.shape)

# -----------------------------------
# Merge datasets
# -----------------------------------

fusion_df = gene_df.join(
    mri_df,
    how="inner"
)

print("Fusion shape:", fusion_df.shape)

# -----------------------------------
# Save final fusion dataset
# -----------------------------------

fusion_df.to_csv(
    "multimodal_fusion_dataset.tsv",
    sep="\t"
)

print("✅ Fusion dataset created successfully.")
