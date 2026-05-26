import pandas as pd

# -----------------------------------
# Load latent features
# -----------------------------------

latent_df = pd.read_csv(
    "gene_latent_features.tsv",
    sep="\t",
    index_col=0
)

print("Latent shape:", latent_df.shape)

# -----------------------------------
# Load sample sheet
# -----------------------------------

sample_sheet = pd.read_csv(
    "gdc_sample_sheet.tsv",
    sep="\t"
)

print("Sample sheet shape:", sample_sheet.shape)

# -----------------------------------
# Create filename column
# -----------------------------------

sample_sheet["filename_clean"] = (
    sample_sheet["File Name"]
    .str.replace(".tsv", "", regex=False)
)

# -----------------------------------
# Create mapping
# filename -> Case ID
# -----------------------------------

filename_to_case = dict(
    zip(
        sample_sheet["filename_clean"],
        sample_sheet["Case ID"]
    )
)

# -----------------------------------
# Map latent index
# -----------------------------------

latent_df["Case_ID"] = latent_df.index.map(
    filename_to_case
)

# -----------------------------------
# Remove failed mappings
# -----------------------------------

latent_df = latent_df.dropna(subset=["Case_ID"])

# -----------------------------------
# Set Case ID as index
# -----------------------------------

latent_df = latent_df.set_index("Case_ID")

print("Mapped shape:", latent_df.shape)

# -----------------------------------
# Save final fusion-ready dataset
# -----------------------------------

latent_df.to_csv(
    "gene_features_case_id.tsv",
    sep="\t"
)

print("✅ Mapping successful.")
