import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import layers, ops
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------
# 1. Load dataset
# -------------------------------------------------------------------

csv_path = "dataset_2.csv"

df = pd.read_csv(csv_path)

df = df.dropna(subset=["seq", "sst3"]).reset_index(drop=True)
df = df[df["seq"].str.len() == df["sst3"].str.len()].reset_index(drop=True)

sequences = df["seq"].astype(str).tolist()
labels_ss = df["sst3"].astype(str).tolist()

print("Nb d'exemples :", len(sequences))
print(df.head())

# -------------------------------------------------------------------
# 2. Encodage AA et SS
# -------------------------------------------------------------------

aa_chars = sorted({c for seq in sequences for c in seq})
print("AA rencontrés :", aa_chars)

aa_to_idx = {c: i + 1 for i, c in enumerate(aa_chars)}
idx_to_aa = {i + 1: c for i, c in enumerate(aa_chars)}
vocab_size = len(aa_to_idx) + 1
print("vocab_size (AA) :", vocab_size)


# # -------------------------------------------------------------------
# # Embedding physico-chimique 5D (Hydrophobicité, Volume, Charge, Polarité, Flexibilité)
# # -------------------------------------------------------------------

# aa_features = {
#     "A": [ 1.8,  88.6,  0, 0, 0.357],
#     "R": [-4.5, 173.4, +1, 1, 0.529],
#     "N": [-3.5, 114.1,  0, 1, 0.463],
#     "D": [-3.5, 111.1, -1, 1, 0.511],
#     "C": [ 2.5, 108.5,  0, 0, 0.346],
#     "Q": [-3.5, 143.8,  0, 1, 0.493],
#     "E": [-3.5, 138.4, -1, 1, 0.497],
#     "G": [-0.4,  60.1,  0, 0, 0.544],
#     "H": [-3.2, 153.2, +1, 1, 0.323],
#     "I": [ 4.5, 166.7,  0, 0, 0.462],
#     "L": [ 3.8, 166.7,  0, 0, 0.365],
#     "K": [-3.9, 168.6, +1, 1, 0.466],
#     "M": [ 1.9, 162.9,  0, 0, 0.295],
#     "F": [ 2.8, 189.9,  0, 0, 0.314],
#     "P": [-1.6, 112.7,  0, 0, 0.509],
#     "S": [-0.8,  89.0,  0, 1, 0.507],
#     "T": [-0.7, 116.1,  0, 1, 0.444],
#     "W": [-0.9, 227.8,  0, 1, 0.305],
#     "Y": [-1.3, 193.6,  0, 1, 0.420],
#     "V": [ 4.2, 140.0,  0, 0, 0.386],

#     # valeurs approximatives pour codes “bizarres”
#     "X": [0.0, 140.0, 0, 0, 0.400],
#     "B": [-3.5, 112.6, -0.5, 1, 0.487],  # N/D mix
#     "Z": [-3.5, 141.1, -0.5, 1, 0.495],  # E/Q mix
#     "J": [ 4.1, 166.7, 0, 0, 0.410],      # I/L mix
#     "U": [ 2.5, 108.0, 0, 0, 0.340],      # Selenocysteine
#     "O": [-3.9, 255.0, +1, 1, 0.450],     # Pyrrolysine
# }

# On construit un tableau brut pour les AA du vocab local
# raw_feats = []
# order = []
# for aa, idx in aa_to_idx.items():
#     raw_feats.append(aa_features.get(aa, [0, 0, 0, 0, 0]))
#     order.append((aa, idx))

# raw_feats = np.array(raw_feats, dtype="float32")  # (vocab_size-1, 5)

# Normalisation (centrage-réduction par dimension)
# mean = raw_feats.mean(axis=0, keepdims=True)
# std = raw_feats.std(axis=0, keepdims=True) + 1e-8
# norm_feats = (raw_feats - mean) / std

# # Matrice finale (0 = padding -> vecteur nul)
# phychem_matrix = np.zeros((vocab_size, 5), dtype="float32")
# for (aa, idx), vec in zip(order, norm_feats):
#     phychem_matrix[idx] = vec

# phy_dim = 5
# print("phychem_matrix shape :", phychem_matrix.shape)


ss_to_idx = {"H": 0, "E": 1, "C": 2}
idx_to_ss = {v: k for k, v in ss_to_idx.items()}
num_classes = 3

encoded_seqs = [[aa_to_idx.get(c, 0) for c in seq] for seq in sequences]

encoded_ss = [[ss_to_idx.get(c, 2) for c in ss] for ss in labels_ss]

# -------------------------------------------------------------------
# 3. Padding
# -------------------------------------------------------------------

maxlen = 128

X = keras.utils.pad_sequences(
    encoded_seqs,
    maxlen=maxlen,
    padding="post",
    truncating="post",
    value=0,
)

y = keras.utils.pad_sequences(
    encoded_ss,
    maxlen=maxlen,
    padding="post",
    truncating="post",
    value=-1,
)

print("X shape :", X.shape)
print("y shape :", y.shape)

# -------------------------------------------------------------------
# 4. Split train / val
# -------------------------------------------------------------------

x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("x_train shape :", x_train.shape)
print("x_val shape   :", x_val.shape)

# -------------------------------------------------------------------
# 5. Transformer block & Token+Position embedding
# -------------------------------------------------------------------

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True,
        )
        self.pos_emb = layers.Embedding(
            input_dim=maxlen,
            output_dim=embed_dim,
        )

    def call(self, x):
        seq_len = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=seq_len, step=1)
        pos = self.pos_emb(positions)
        tok = self.token_emb(x)
        return tok + pos

    def compute_mask(self, inputs, mask=None):
        return self.token_emb.compute_mask(inputs)

# class TokenAndPositionEmbedding(layers.Layer):
#     def __init__(self, maxlen, vocab_size, embed_dim, phychem_matrix):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.phy_dim = phychem_matrix.shape[1]
#         self.model_dim = embed_dim + self.phy_dim

#         # Embedding appris pour AA
#         self.token_emb = layers.Embedding(
#             input_dim=vocab_size,
#             output_dim=embed_dim,
#             mask_zero=True,
#         )
#         # Embedding physico-chimique fixe
#         self.phy_emb = layers.Embedding(
#             input_dim=vocab_size,
#             output_dim=self.phy_dim,
#             weights=[phychem_matrix],
#             trainable=False,
#             mask_zero=True,
#         )
#         # Embedding positionnel sur la dimension totale
#         self.pos_emb = layers.Embedding(
#             input_dim=maxlen,
#             output_dim=self.model_dim,
#         )

#     def call(self, x):
#         seq_len = ops.shape(x)[-1]              # L
#         positions = ops.arange(start=0, stop=seq_len, step=1)
#         pos = self.pos_emb(positions)           # (L, model_dim)

#         tok = self.token_emb(x)                 # (B, L, embed_dim)
#         phy = self.phy_emb(x)                   # (B, L, phy_dim)
#         x = ops.concatenate([tok, phy], axis=-1)  # (B, L, model_dim)

#         return x + pos                          # broadcast sur batch

#     def compute_mask(self, inputs, mask=None):
#         # on propage le mask du token_emb (0 = PAD)
#         return self.token_emb.compute_mask(inputs)


# -------------------------------------------------------------------
# 6. Modèle Transformer : prédiction par résidu (H/E/C)
# -------------------------------------------------------------------

embed_dim = 256
phy_dim = 5
model_dim = embed_dim + phy_dim
num_heads = 8
ff_dim = 1024

inputs = keras.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)

x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
# x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="transformer_ss3")
model.summary()

# -------------------------------------------------------------------
# 7. Loss masquée (on ignore le pad dans la loss)
# -------------------------------------------------------------------

def loss(y_true, y_pred):
    """
    y_true: (B, L), entiers dans [0, num_classes-1] ou -1 pour pad
    y_pred: (B, L, num_classes)
    """
    mask = ops.cast(y_true >= 0, "float32")
    y_true_clipped = ops.maximum(y_true, 0)

    scce = keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction="none",
    )
    loss_per_token = scce(y_true_clipped, y_pred)
    loss_per_token = loss_per_token * mask

    return ops.sum(loss_per_token) / (ops.sum(mask) + 1e-9)


def accuracy(y_true, y_pred):
    mask = ops.cast(y_true >= 0, "float32")
    y_true_clipped = ops.maximum(y_true, 0)

    y_pred_labels = ops.argmax(y_pred, axis=-1)
    matches = ops.cast(ops.equal(y_true_clipped, y_pred_labels), "float32")
    matches = matches * mask
    return ops.sum(matches) / (ops.sum(mask) + 1e-9)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss=loss,
    metrics=[accuracy],
)

# -------------------------------------------------------------------
# 8. Entraînement
# -------------------------------------------------------------------

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
]

history = model.fit(
    x_train,
    y_train,
    batch_size=16,
    epochs=30,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
)

# -------------------------------------------------------------------
# 10. Plot train/val loss & accuracy
# -------------------------------------------------------------------

def plot_history_panels(history, out_path="training_panels.png"):

    train_loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    train_acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])

    epochs = range(1, len(train_loss) + 1)

    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    ax = axs[0]
    ax.plot(epochs, train_loss, label="Train loss")
    ax.plot(epochs, val_loss, label="Val loss", linestyle="--")
    ax.set_title("Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

    ax = axs[1]
    ax.plot(epochs, train_acc, label="Train accuracy")
    ax.plot(epochs, val_acc, label="Val accuracy", linestyle="--")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Panel sauvegardé sous : {out_path}")

plot_history_panels(history, out_path="training_panels.png")
