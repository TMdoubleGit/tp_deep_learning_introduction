import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import keras
from keras import layers

# -------------------------------------------------------------------
# 1. Chargement du dataset
# -------------------------------------------------------------------

csv_path = "iris.csv"

df = pd.read_csv(csv_path, sep=r"\s+")

print("Colonnes du CSV :", df.columns.tolist())
print(df.head())

feature_cols = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
label_col = "Species"

X = df[feature_cols].values
y = df[label_col].values

print("Shape X :", X.shape)
print("Espèces :", np.unique(y))

# -------------------------------------------------------------------
# 2. Standardisation + ACP brute
# -------------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_raw = PCA(n_components=2)
X_pca_raw = pca_raw.fit_transform(X_scaled)

species = np.unique(y)
color_map = {
    species[0]: "red",
    species[1]: "green",
    species[2]: "blue",
}

# -------------------------------------------------------------------
# 3. Construction de tokens
# -------------------------------------------------------------------

n_samples, n_features = X_scaled.shape
assert n_features == 4

feature_onehot = np.eye(n_features)

tokens = []
for i in range(n_samples):
    vals = X_scaled[i]
    sample_tokens = []
    for j in range(n_features):
        v = vals[j]
        oh = feature_onehot[j]
        token_vec = np.concatenate([[v], oh])
        sample_tokens.append(token_vec)
    sample_tokens = np.stack(sample_tokens, axis=0)
    tokens.append(sample_tokens)

tokens = np.stack(tokens, axis=0)
print("tokens shape :", tokens.shape)

# -------------------------------------------------------------------
# 4. Encodage des labels + split train/val
# -------------------------------------------------------------------

species_sorted = sorted(np.unique(y))
sp_to_idx = {sp: i for i, sp in enumerate(species_sorted)}
idx_to_sp = {i: sp for sp, i in sp_to_idx.items()}

y_int = np.array([sp_to_idx[sp] for sp in y], dtype="int32")

X_train, X_val, y_train, y_val = train_test_split(
    tokens, y_int, test_size=0.2, random_state=42, stratify=y_int
)

print("X_train shape :", X_train.shape)
print("X_val shape   :", X_val.shape)

# -------------------------------------------------------------------
# 5. Modèle self-attention + Dense pour classification
# -------------------------------------------------------------------

n_features = 4
token_dim = 5
latent_dim = 16

inputs = keras.Input(shape=(n_features, token_dim))

att = layers.MultiHeadAttention(
    num_heads=1,
    key_dim=token_dim,
    name="self_attention",
)(inputs, inputs)

att_pooled = layers.GlobalAveragePooling1D(name="att_pool")(att)

latent = layers.Dense(latent_dim, activation="relu", name="latent_dense")(att_pooled)

outputs = layers.Dense(len(species_sorted), activation="softmax", name="classifier")(latent)

model = keras.Model(inputs=inputs, outputs=outputs, name="iris_attention_classifier")
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=16,
    epochs=100,
    callbacks=callbacks,
    verbose=1,
)

# -------------------------------------------------------------------
# 6. Extraction des représentations
# -------------------------------------------------------------------

encoder_att = keras.Model(
    inputs=model.input,
    outputs=model.get_layer("att_pool").output,
    name="encoder_attention_only",
)

encoder_latent = keras.Model(
    inputs=model.input,
    outputs=model.get_layer("latent_dense").output,
    name="encoder_attention_plus_dense",
)

H_att = encoder_att.predict(tokens, batch_size=32)
H_lat = encoder_latent.predict(tokens, batch_size=32)

H_att_scaled = StandardScaler().fit_transform(H_att)
H_lat_scaled = StandardScaler().fit_transform(H_lat)

pca_att = PCA(n_components=2)
H_att_pca = pca_att.fit_transform(H_att_scaled)

pca_lat = PCA(n_components=2)
H_lat_pca = pca_lat.fit_transform(H_lat_scaled)

# -------------------------------------------------------------------
# 7. Figure avec trois panels ACP
# -------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1 : ACP brute
ax = axes[0]
for sp in species_sorted:
    mask = (y == sp)
    ax.scatter(
        X_pca_raw[mask, 0],
        X_pca_raw[mask, 1],
        label=sp,
        alpha=0.7,
        s=30,
        edgecolor="k",
    )
ax.set_title("ACP brute (features d'origine)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
ax.grid(True)

# Panel 2 : ACP après self-attention (sans Dense)
ax = axes[1]
for sp in species_sorted:
    mask = (y == sp)
    ax.scatter(
        H_att_pca[mask, 0],
        H_att_pca[mask, 1],
        label=sp,
        alpha=0.7,
        s=30,
        edgecolor="k",
    )
ax.set_title("ACP après self-attention seule")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
ax.grid(True)

# Panel 3 : ACP après self-attention + Dense
ax = axes[2]
for sp in species_sorted:
    mask = (y == sp)
    ax.scatter(
        H_lat_pca[mask, 0],
        H_lat_pca[mask, 1],
        label=sp,
        alpha=0.7,
        s=30,
        edgecolor="k",
    )
ax.set_title("ACP après self-attention + Dense")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig("iris_pca_three_panels.png", dpi=150)
plt.close()
