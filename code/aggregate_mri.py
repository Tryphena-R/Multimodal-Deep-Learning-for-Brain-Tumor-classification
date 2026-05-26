import pandas as pd

# -----------------------------------
# Load MRI dataset
# -----------------------------------

mri_df = pd.read_excel(
    "MRI_Aggregated.xlsx"
)

print("Original MRI shape:", mri_df.shape)

# -----------------------------------
# Clean column names
# -----------------------------------

mri_df.columns = mri_df.columns.str.strip()

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

# Fill missing values
mri_df["Patient Age"] = (
    mri_df["Patient Age"]
    .fillna(mri_df["Patient Age"].median())
)

# -----------------------------------
# Encode Sex
# M = 1
# F = 0
# -----------------------------------

mri_df["Patient Sex"] = (
    mri_df["Patient Sex"]
    .map({"M": 1, "F": 0})
)

# Fill missing sex values if any
mri_df["Patient Sex"] = (
    mri_df["Patient Sex"]
    .fillna(0)
)

# -----------------------------------
# Create MRI sequence counts
# -----------------------------------

series_counts = pd.crosstab(
    mri_df["Patient ID"],
    mri_df["Series_Type"]
)

print("Series count shape:", series_counts.shape)

# -----------------------------------
# Aggregate numerical features
# -----------------------------------

agg_df = mri_df.groupby("Patient ID").agg({

    "Patient Age": "first",
    "Patient Sex": "first",
    "Image Count": "sum",
    "File Size": "sum"

})

print("Aggregated numerical shape:", agg_df.shape)

# -----------------------------------
# Merge aggregated features
# -----------------------------------

final_mri_df = agg_df.join(
    series_counts,
    how="left"
)

print("Final MRI feature shape:", final_mri_df.shape)

# -----------------------------------
# Save aggregated MRI dataset
# -----------------------------------

final_mri_df.to_csv(
    "mri_patient_features.tsv",
    sep="\t"
)

print("✅ MRI aggregation complete.")
