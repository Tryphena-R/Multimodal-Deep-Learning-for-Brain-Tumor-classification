import pandas as pd
import numpy as np

n = 534

np.random.seed(42)

mri_df = pd.DataFrame({

    "num_T1":
        np.random.randint(1, 6, n),

    "num_T2":
        np.random.randint(1, 6, n),

    "num_FLAIR":
        np.random.randint(1, 5, n),

    "total_images":
        np.random.randint(100, 500, n),

    "patient_age":
        np.random.randint(20, 80, n),

    "patient_sex":
        np.random.randint(0, 2, n)

})

mri_df.to_csv(
    "mri_datasets_only.tsv",
    sep="\t",
    index=False
)

print("✅ Synthetic MRI dataset created.")
