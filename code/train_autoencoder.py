import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------------
# Load reduced dataset
# -----------------------------------

X = np.load("gene_expression_reduced.npy")

print("Input shape:", X.shape)

# -----------------------------------
# Autoencoder parameters
# -----------------------------------

input_dim = X.shape[1]
latent_dim = 128

# -----------------------------------
# Build Autoencoder
# -----------------------------------

input_layer = Input(shape=(input_dim,))

# Encoder
encoded = Dense(4096, activation="relu")(input_layer)
encoded = Dense(1024, activation="relu")(encoded)

latent = Dense(
    latent_dim,
    activation="relu",
    name="latent_layer"
)(encoded)

# Decoder
decoded = Dense(1024, activation="relu")(latent)
decoded = Dense(4096, activation="relu")(decoded)

output_layer = Dense(
    input_dim,
    activation="linear"
)(decoded)

# Build model
autoencoder = Model(
    inputs=input_layer,
    outputs=output_layer
)

# -----------------------------------
# Compile model
# -----------------------------------

autoencoder.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="mse"
)

autoencoder.summary()

# -----------------------------------
# Early stopping
# -----------------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# -----------------------------------
# Train model
# -----------------------------------

history = autoencoder.fit(
    X,
    X,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    shuffle=True,
    callbacks=[early_stop]
)

# -----------------------------------
# Save model weights
# -----------------------------------

autoencoder.save_weights(
    "autoencoder.weights.h5"
)

print("Model weights saved.")

# -----------------------------------
# Extract latent features
# -----------------------------------

encoder = Model(
    inputs=autoencoder.input,
    outputs=autoencoder.get_layer("latent_layer").output
)

X_latent = encoder.predict(X)

print("Latent shape:", X_latent.shape)

# -----------------------------------
# Load sample IDs
# -----------------------------------

df_original = pd.read_csv(
    "final_gene_expression.tsv",
    sep="\t"
)

sample_ids = df_original.columns[1:]

# -----------------------------------
# Save latent features
# -----------------------------------

latent_df = pd.DataFrame(
    X_latent,
    index=sample_ids,
    columns=[
        f"latent_{i}" for i in range(latent_dim)
    ]
)

latent_df.to_csv(
    "gene_latent_features.tsv",
    sep="\t"
)

np.save(
    "gene_latent_features.npy",
    X_latent
)

print("✅ Autoencoder training complete.")
print("Latent features saved.")
