import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Multimodal Brain Tumor Classification",
    layout="wide"
)

st.title("🧠 Multimodal Brain Tumor Classification")

st.markdown(
    "### Early Fusion-based LGG vs GBM Prediction using Gene Expression and MRI Features"
)

# Load trained model
model = joblib.load("random_forest_model.pkl")

# Sidebar
st.sidebar.header("Upload Modalities")

gene_file = st.sidebar.file_uploader(
    "Upload Gene Dataset",
    type=["csv", "tsv"],
    key="gene"
)

mri_file = st.sidebar.file_uploader(
    "Upload MRI Dataset",
    type=["csv", "tsv"],
    key="mri"
)

# Main App
if gene_file is not None and mri_file is not None:

    # Read Gene File
    if gene_file.name.endswith(".tsv"):
        gene_df = pd.read_csv(gene_file, sep="\t")
    else:
        gene_df = pd.read_csv(gene_file)

    # Read MRI File
    if mri_file.name.endswith(".tsv"):
        mri_df = pd.read_csv(mri_file, sep="\t")
    else:
        mri_df = pd.read_csv(mri_file)

    # Store Patient IDs
    if "Case_ID" in gene_df.columns:
        patient_ids = gene_df["Case_ID"]
    else:
        patient_ids = pd.Series(
            [f"Patient_{i+1}" for i in range(len(gene_df))]
        )

    # Remove unnecessary columns
    for col in ["label", "Case_ID"]:

        if col in gene_df.columns:
            gene_df = gene_df.drop(columns=[col])

        if col in mri_df.columns:
            mri_df = mri_df.drop(columns=[col])

    # Match sample counts
    min_samples = min(len(gene_df), len(mri_df))

    gene_df = gene_df.iloc[:min_samples].reset_index(drop=True)
    mri_df = mri_df.iloc[:min_samples].reset_index(drop=True)

    patient_ids = patient_ids.iloc[:min_samples].reset_index(drop=True)

    # Early Fusion
    fusion_df = pd.concat(
        [gene_df, mri_df],
        axis=1
    )

    # Predictions
    predictions = model.predict(fusion_df)

    probabilities = model.predict_proba(fusion_df)

    confidence_scores = np.max(
        probabilities,
        axis=1
    )

    prediction_labels = [
        "GBM" if p == 1 else "LGG"
        for p in predictions
    ]

    # Result DataFrame
    result_df = pd.DataFrame({

        "Patient_ID":
            patient_ids,

        "Predicted_Tumor":
            prediction_labels,

        "Confidence_Score":
            np.round(confidence_scores * 100, 2)

    })

    # Show Results
    st.subheader("Prediction Results")

    st.dataframe(result_df)

    # Counts
    lgg_count = prediction_labels.count("LGG")
    gbm_count = prediction_labels.count("GBM")

    # Charts
    col1, col2 = st.columns(2)

    # Pie Chart
    with col1:

        st.subheader("Tumor Distribution Pie Chart")

        fig1, ax1 = plt.subplots()

        ax1.pie(
            [lgg_count, gbm_count],
            labels=["LGG", "GBM"],
            autopct="%1.1f%%"
        )

        st.pyplot(fig1)

    # Bar Chart
    with col2:

        st.subheader("Tumor Distribution Bar Chart")

        fig2, ax2 = plt.subplots()

        ax2.bar(
            ["LGG", "GBM"],
            [lgg_count, gbm_count]
        )

        st.pyplot(fig2)

    # Summary
    st.subheader("Prediction Summary")

    st.write(f"LGG Predictions: {lgg_count}")
    st.write(f"GBM Predictions: {gbm_count}")
    st.write(f"Total Samples: {len(prediction_labels)}")

    # Download Results
    csv = result_df.to_csv(index=False)

    st.download_button(
        label="Download Prediction Results",
        data=csv,
        file_name="tumor_predictions.csv",
        mime="text/csv"
    )

    st.success("✅ Prediction Complete")
