import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------------
# Load fused dataset
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
# Feature scaling
# -----------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# -----------------------------------
# Build MLP Model
# -----------------------------------

model = Sequential()

model.add(
    Dense(
        256,
        activation='relu',
        input_shape=(X_train.shape[1],)
    )
)

model.add(Dropout(0.3))

model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(Dropout(0.3))

model.add(
    Dense(
        64,
        activation='relu'
    )
)

model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)

# -----------------------------------
# Compile model
# -----------------------------------

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------------
# Early stopping
# -----------------------------------

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# -----------------------------------
# Train model
# -----------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stop]
)

# -----------------------------------
# Predict
# -----------------------------------

y_pred_prob = model.predict(X_test)

y_pred = (y_pred_prob > 0.5).astype(int)

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

print("\n✅ Final MLP training complete.")
