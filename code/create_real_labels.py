import pandas as pd

# -----------------------------------
# Load sample sheet
# -----------------------------------

sample_sheet = pd.read_csv(
    "gdc_sample_sheet.tsv",
    sep="\t"
)

# -----------------------------------
# Create clean filename column
# -----------------------------------

sample_sheet["filename_clean"] = (
    sample_sheet["File Name"]
    .str.replace(".tsv", "", regex=False)
)

# -----------------------------------
# Create labels
# LGG = 0
# GBM = 1
# -----------------------------------

sample_sheet["label"] = (
    sample_sheet["Project ID"]
    .map({
        "TCGA-LGG": 0,
        "TCGA-GBM": 1
    })
)

# -----------------------------------
# Keep required columns
# -----------------------------------

label_df = sample_sheet[[
    "filename_clean",
    "Case ID",
    "Project ID",
    "label"
]]

# -----------------------------------
# Save labels
# -----------------------------------

label_df.to_csv(
    "real_labels.tsv",
    sep="\t",
    index=False
)

print("✅ Real labels created.")
print(label_df.head())
