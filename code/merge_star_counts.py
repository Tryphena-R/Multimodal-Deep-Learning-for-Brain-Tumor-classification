import pandas as pd
import os

folder = "expression_tsv"

all_samples = []

for file in os.listdir(folder):

    if file.endswith(".tsv"):

        filepath = os.path.join(folder, file)

        df = pd.read_csv(
            filepath,
            sep="\t",
            comment="#"
        )

        # Remove technical rows
        df = df[~df["gene_id"].str.startswith("N_")]

        # Remove gene version numbers
        df["gene_id"] = df["gene_id"].str.split(".").str[0]

        # Keep required columns
        df = df[["gene_id", "unstranded"]]

        # Use file UUID as sample ID
        sample_id = file.replace(".tsv", "")

        # Rename expression column
        df = df.rename(
            columns={"unstranded": sample_id}
        )

        # Set gene_id as index
        df = df.set_index("gene_id")

        # Store dataframe
        all_samples.append(df)

# Concatenate ALL at once
merged_df = pd.concat(all_samples, axis=1)

print("Merged shape:", merged_df.shape)

# Save final matrix
merged_df.to_csv(
    "final_gene_expression.tsv",
    sep="\t"
)

print("✅ Gene merge complete.")
