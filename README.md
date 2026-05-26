🧠 Multimodal Brain Tumor Classification using Deep Learning

An AI-powered healthcare analytics project that combines RNA-Seq Gene Expression 🧬 and MRI Imaging Features 🩻 for intelligent brain tumor classification using Multimodal Deep Learning.

The system predicts whether a tumor belongs to:

🔹 LGG (Low Grade Glioma)
🔹 GBM (Glioblastoma Multiforme)

along with prediction confidence scores 📊.

🚀 Project Overview
This project implements a complete Multimodal Deep Learning Pipeline involving:

✅ Gene Expression Preprocessing
✅ Normalization & Scaling
✅ Dimensionality Reduction
✅ Deep Autoencoder-based Latent Feature Extraction
✅ MRI Feature Processing
✅ Early Fusion
✅ Deep Neural Network (DNN/MLP) Classification
✅ Streamlit-based Frontend Visualization

The project demonstrates how genomic and imaging modalities can be combined to improve intelligent healthcare diagnostics.

🧬 Technologies Used
Python
TensorFlow / Keras
Scikit-learn
Pandas
NumPy
Matplotlib
Streamlit

📂 Project Structure
TCGA_GBM_LGG_GeneExpression/
│
├── code/
│   ├── aggregate_mri.py
│   ├── app.py
│   ├── create_real_labels.py
│   ├── final_fusion.py
│   ├── fusion_prepare.py
│   ├── map_case_ids.py
│   ├── merge_star_counts.py
│   ├── normalize_expression.py
│   ├── reduce_genes.py
│   ├── train_autoencoder.py
│   └── train_mlp.py
│
├── datasets/
│
├── models/
│
├── README.md
└── requirements.txt

⚙️ Workflow
RNA-Seq Data
↓
Preprocessing & Normalization
↓
Gene Reduction
↓
Deep Autoencoder
↓
128-D Latent Features

MRI Data
↓
MRI Feature Processing

Early Fusion
↓
Combined Multimodal Features

DNN / MLP Classifier
↓
LGG vs GBM Prediction

🧠 Deep Learning Components
🔹 Deep Autoencoder
Used for learning compressed latent genomic representations from high-dimensional gene expression data.

🔹 Early Fusion
Combines genomic latent features and MRI features into a single multimodal feature vector.

🔹 DNN / MLP Classifier
Classifies tumors using fused multimodal features with ReLU and Sigmoid activation functions.

📊 Features
Brain tumor prediction
Confidence score generation
Interactive Streamlit frontend
Multimodal healthcare analytics
Deep learning-based genomic feature extraction

▶️ Run the Project
Install Dependencies
pip install -r requirements.txt
Run Streamlit App
streamlit run app.py

📌 Dataset Sources
TCGA (The Cancer Genome Atlas)
GDC Portal
TCIA MRI Metadata
