import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# -----------------------------------
# Load final multimodal dataset
# -----------------------------------

df = pd.read_csv(
    "final_multimodal_dataset.tsv",
    sep="\t"
)

print("Dataset shape:", df.shape)

# -----------------------------------
# Features and labels
# -----------------------------------

X = df.drop(columns=["label"])

y = df["label"]

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

# -----------------------------------
# Train-test split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -----------------------------------
# Build Random Forest
# -----------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# -----------------------------------
# Train
# -----------------------------------

model.fit(
    X_train,
    y_train
)

# -----------------------------------
# Predict
# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------
# Evaluation
# -----------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\n✅ Random Forest training complete.")

import joblib

joblib.dump(model, "random_forest_model.pkl")

print("✅ Model saved successfully.")
