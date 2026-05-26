import numpy as np
import pandas as pd

X = np.load("gene_expression_reduced.npy")

df = pd.DataFrame(X)

df.to_csv(
    "gene_expression_dataset.tsv",
    sep="\t",
    index=False
)

print("✅ Gene TSV created.")
