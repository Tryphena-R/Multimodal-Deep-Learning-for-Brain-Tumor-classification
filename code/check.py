import pandas as pd

mri_df = pd.read_excel("MRI_Aggregated.xlsx")

mri_df.columns = mri_df.columns.str.strip()

print(mri_df.columns)

print(mri_df.head())
