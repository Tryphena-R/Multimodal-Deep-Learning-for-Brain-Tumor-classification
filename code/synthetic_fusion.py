import pandas as pd
import numpy as np

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
# Load REAL TCGA labels
# -----------------------------------

labels_df = pd.read_csv(
    "real_labels.tsv",
    sep="\t"
)

# Keep only required columns
labels_df = labels_df[[
    "Case ID",
    "label"
]]

print("Original labels shape:", labels_df.shape)

# -----------------------------------
# Balance LGG and GBM samples
# -----------------------------------

lgg_df = labels_df[
    labels_df["label"] == 0
]

gbm_df = labels_df[
    labels_df["label"] == 1
]

print("LGG samples:", len(lgg_df))
print("GBM samples:", len(gbm_df))

# Equal sample count
n = min(
    len(lgg_df),
    len(gbm_df)
)

# Random sampling
lgg_df = lgg_df.sample(
    n,
    random_state=42
)

gbm_df = gbm_df.sample(
    n,
    random_state=42
)

# Combine
labels_df = pd.concat([
    lgg_df,
    gbm_df
])

# Shuffle
labels_df = labels_df.sample(
    frac=1,
    random_state=42
)

labels_df = labels_df.reset_index(drop=True)

print("Balanced labels shape:", labels_df.shape)

# -----------------------------------
# Match gene feature count
# -----------------------------------

gene_df = gene_df.iloc[:len(labels_df)]

print("Matched gene shape:", gene_df.shape)

# -----------------------------------
# Generate SYNTHETIC MRI features
# -----------------------------------

np.random.seed(42)

synthetic_mri = pd.DataFrame({

    "num_T1":
        np.random.randint(
            1,
            6,
            len(labels_df)
        ),

    "num_T2":
        np.random.randint(
            1,
            6,
            len(labels_df)
        ),

    "num_FLAIR":
        np.random.randint(
            1,
            5,
            len(labels_df)
        ),

    "total_images":
        np.random.randint(
            100,
            500,
            len(labels_df)
        ),

    "patient_age":
        np.random.randint(
            20,
            80,
            len(labels_df)
        ),

    "patient_sex":
        np.random.randint(
            0,
            2,
            len(labels_df)
        )

})

print("Synthetic MRI shape:", synthetic_mri.shape)

# -----------------------------------
# Reset indexes
# -----------------------------------

gene_df = gene_df.reset_index(drop=True)

# -----------------------------------
# EARLY FUSION
# -----------------------------------

fusion_df = pd.concat(
    [gene_df, synthetic_mri],
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
    "final_multimodal_dataset.tsv",
    sep="\t",
    index=False
)

print("✅ Final multimodal dataset created.")
