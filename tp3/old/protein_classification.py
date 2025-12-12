"""
Text / sequence classification with CNN and Transformer on amino-acid sequences.

Dataset: CSV with columns:
    ID, SEQUENCE, CL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import ops
from keras import layers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit

# -------------------------------------------------------------------
# 1. Load and prepare dataset from CSV
# -------------------------------------------------------------------

csv_path = "db_merged_unique_fold.csv"

df = pd.read_csv(csv_path)

df = df.dropna(subset=["SEQUENCE", "CL"])
df = df[df["CL"] != 1000004].reset_index(drop=True)

# Séquences en string et labels en int / str
sequences = df["SEQUENCE"].astype(str).tolist()
labels_raw = df["CL"].tolist()

print(f"Nombre total d'exemples : {len(sequences)}")

# -------------------------------------------------------------------
# 2. Encodage des acides aminés en entiers
# -------------------------------------------------------------------

# Vocab de caractères (acides aminés) présents dans le dataset
chars = sorted({c for seq in sequences for c in seq})
print("Caractères (AA) trouvés dans le dataset :", chars)

# On réserve l'indice 0 pour le padding
char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
idx_to_char = {i + 1: c for i, c in enumerate(chars)}
vocab_size = len(char_to_idx) + 1

# -------------------------------------------------------------------
# 2bis. Embedding Oobatake (non-bonded interaction energies)
# -------------------------------------------------------------------

oobatake_nonbonded = {
    "A": 0.17,
    "R": 0.81,
    "N": -0.42,
    "D": -1.23,
    "C": 0.24,
    "E": -1.27,
    "Q": -0.58,
    "G": 0.01,
    "H": 0.15,
    "I": 0.25,
    "L": 0.53,
    "K": 0.99,
    "M": 0.09,
    "F": 0.37,
    "P": 0.45,
    "S": -0.13,
    "T": -0.14,
    "W": 0.30,
    "Y": 0.27,
    "V": 0.07,

    # ambiguous / extended codes
    "B": (-1.23 + -0.42) / 2,   # D/N
    "Z": (-1.27 + -0.58) / 2,   # E/Q
    "J": (0.25 + 0.53) / 2,     # I/L
    "X": 0.0,                   # unknown
    "U": 0.24,                  # selenocysteine ≈ C
    "O": 0.99,                  # pyrrolysine ≈ K
}

# Dimension de l'embedding Oobatake (ici scalaire)
ooba_dim = 1

# Matrice d'embedding : shape = (vocab_size, ooba_dim)
oobatake_matrix = np.zeros((vocab_size, ooba_dim), dtype="float32")
for aa, idx in char_to_idx.items():
    value = oobatake_nonbonded.get(aa, 0.0)
    oobatake_matrix[idx, 0] = value  # idx 0 (padding) reste à 0

# -------------------------------------------------------------------
# 2ter. Embedding 5D (Kyte-Doolittle, Volume, Charge, Polarity, Flexibility)
# -------------------------------------------------------------------

aa_features = {
    "A": [ 1.8,  88.6,  0, 0, 0.357],
    "R": [-4.5, 173.4, +1, 1, 0.529],
    "N": [-3.5, 114.1,  0, 1, 0.463],
    "D": [-3.5, 111.1, -1, 1, 0.511],
    "C": [ 2.5, 108.5,  0, 0, 0.346],
    "Q": [-3.5, 143.8,  0, 1, 0.493],
    "E": [-3.5, 138.4, -1, 1, 0.497],
    "G": [-0.4,  60.1,  0, 0, 0.544],
    "H": [-3.2, 153.2, +1, 1, 0.323],
    "I": [ 4.5, 166.7,  0, 0, 0.462],
    "L": [ 3.8, 166.7,  0, 0, 0.365],
    "K": [-3.9, 168.6, +1, 1, 0.466],
    "M": [ 1.9, 162.9,  0, 0, 0.295],
    "F": [ 2.8, 189.9,  0, 0, 0.314],
    "P": [-1.6, 112.7,  0, 0, 0.509],
    "S": [-0.8,  89.0,  0, 1, 0.507],
    "T": [-0.7, 116.1,  0, 1, 0.444],
    "W": [-0.9, 227.8,  0, 1, 0.305],
    "Y": [-1.3, 193.6,  0, 1, 0.420],
    "V": [ 4.2, 140.0,  0, 0, 0.386],

    "X": [0.0, 140.0, 0, 0, 0.400],
    "B": [-3.5, 112.6, -0.5, 1, 0.487],  # N/D mix
    "Z": [-3.5, 141.1, -0.5, 1, 0.495],  # E/Q mix
    "J": [ 4.1, 166.7, 0, 0, 0.410],  # I/L mix
    "U": [ 2.5, 108.0, 0, 0, 0.340],  # Selenocysteine
    "O": [-3.9, 255.0, +1, 1, 0.450],  # Pyrrolysine
}

phychem_matrix = np.zeros((vocab_size, 5), dtype="float32")
for aa, idx in char_to_idx.items():
    phychem_matrix[idx] = aa_features.get(aa, [0,0,0,0,0])

print("Taille du vocabulaire (AA) :", vocab_size)

# Encode chaque séquence en liste d'indices
encoded_sequences = [[char_to_idx.get(c, 0) for c in seq] for seq in sequences]

maxlen = 256
print("Longueur maximale retenue pour les séquences :", maxlen)

# Padding / truncation
X = keras.utils.pad_sequences(
    encoded_sequences,
    maxlen=maxlen,
    padding="post",
    truncating="post",
)

# -------------------------------------------------------------------
# 3. Encodage des labels CL
# -------------------------------------------------------------------

# On mappe les valeurs CL réelles (ex: 1000002) vers des indices 0..num_classes-1
unique_classes = sorted(df["CL"].unique())
cl_to_idx = {cl: i for i, cl in enumerate(unique_classes)}
idx_to_cl = {i: cl for cl, i in cl_to_idx.items()}

y = df["CL"].map(cl_to_idx).values
num_classes = len(unique_classes)

print("Nombre de classes CL :", num_classes)
print("Mapping CL -> index :", cl_to_idx)


# np.random.seed(42)
# indices = np.arange(len(X))
# np.random.shuffle(indices)

# train_split = int(0.7 * len(X))
# train_idx, val_idx = indices[:train_split], indices[train_split:]

# x_train, x_val = X[train_idx], X[val_idx]
# y_train, y_val = y[train_idx], y[val_idx]

# print("Taille x_train :", x_train.shape)
# print("Taille x_val   :", x_val.shape)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
(train_idx, val_idx), = sss.split(X, y)

x_train, x_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

print("Taille x_train :", x_train.shape)
print("Taille x_val   :", x_val.shape)

# # ----------------------------------------------------------
# # Distribution des classes dans train vs val
# # ----------------------------------------------------------
# train_classes, train_counts = np.unique(y_train, return_counts=True)
# val_classes, val_counts = np.unique(y_val, return_counts=True)

# train_dist = pd.DataFrame({"class_idx": train_classes, "count": train_counts})
# val_dist = pd.DataFrame({"class_idx": val_classes, "count": val_counts})

# # Fusion pour comparaison
# merged = pd.merge(
#     train_dist, val_dist, on="class_idx", how="outer", suffixes=("_train", "_val")
# ).fillna(0)

# merged["class_label"] = merged["class_idx"].map(idx_to_cl)
# merged = merged.sort_values("class_idx")

# # Plot
# plt.figure(figsize=(10, 5))
# width = 0.4
# x = np.arange(len(merged))

# plt.bar(x - width/2, merged["count_train"], width=width, label="Train", alpha=0.7)
# plt.bar(x + width/2, merged["count_val"], width=width, label="Validation", alpha=0.7)

# plt.xticks(x, merged["class_label"], rotation=45)
# plt.xlabel("Classe (CL)")
# plt.ylabel("Nombre d'exemples")
# plt.title("Distribution des classes : Train vs Validation")
# plt.legend()
# plt.tight_layout()
# plt.show()

# Hyperparamètres communs
embed_dim = 96
num_heads = 3
ff_dim = 192
epochs = 30
batch_size = 64
ooba_dim = 1
# phy_dim = 5
model_dim = embed_dim + ooba_dim

# -------------------------------------------------------------------
# 5. Modèle CNN 1D baseline
# -------------------------------------------------------------------

kernel_size = 5
pool_size = 2
num_filters = 128

# inputs_cnn = keras.Input(shape=(maxlen,))

# x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen)(
#     inputs_cnn
# )
# x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, activation="relu")(x)

inputs_cnn = keras.Input(shape=(maxlen,))

# Embedding appris
tok_emb_cnn = layers.Embedding(
    input_dim=vocab_size,
    output_dim=embed_dim,
    input_length=maxlen,
)

# Embedding Oobatake fixe (non-trainable)
ooba_emb_cnn = layers.Embedding(
    input_dim=vocab_size,
    output_dim=ooba_dim,
    input_length=maxlen,
    weights=[oobatake_matrix],
    trainable=False,
)

# phy_emb_cnn = layers.Embedding(
#     input_dim=vocab_size,
#     output_dim=5,
#     input_length=maxlen,
#     weights=[phychem_matrix],
#     trainable=False,
# )

x_tok = tok_emb_cnn(inputs_cnn)
x_ooba = ooba_emb_cnn(inputs_cnn)
# x_phy  = phy_emb_cnn(inputs_cnn)
x = layers.Concatenate(axis=-1)([x_tok, x_ooba])

x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, activation="relu")(x)


x = layers.MaxPooling1D(pool_size=pool_size)(x)
x = layers.GlobalMaxPooling1D()(x)

x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# Sortie = num_classes, softmax
outputs_cnn = layers.Dense(num_classes, activation="softmax")(x)

model_cnn = keras.Model(inputs_cnn, outputs_cnn, name="cnn_1d_protein")

model_cnn.summary()

model_cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_cnn = model_cnn.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
)

# -------------------------------------------------------------------
# 6. Transformer block et embedding positionnel
# -------------------------------------------------------------------

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2):
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
    def __init__(self, maxlen, vocab_size, embed_dim, oobatake_matrix):
        super().__init__()
        self.embed_dim = embed_dim
        self.ooba_dim = oobatake_matrix.shape[1]
        # self.phy_dim = phychem_matrix.shape[1]

        self.token_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True,
        )
        self.ooba_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=self.ooba_dim,
            weights=[oobatake_matrix],
            trainable=False,
            mask_zero=True,
        )

        # self.phy_emb = layers.Embedding(
        #     input_dim=vocab_size,
        #     output_dim=phy_dim,
        #     weights=[phychem_matrix],
        #     trainable=False,
        #     mask_zero=True,
        # )

        self.pos_emb = layers.Embedding(
            input_dim=maxlen,
            output_dim=embed_dim + self.ooba_dim,
        )

    def call(self, x):
        seq_len = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=seq_len, step=1)
        pos = self.pos_emb(positions)

        tok = self.token_emb(x)
        ooba = self.ooba_emb(x)
        # phy  = self.phy_emb(x)
        x = ops.concatenate([tok, ooba], axis=-1)

        return x + pos

    def compute_mask(self, inputs, mask=None):
        return self.token_emb.compute_mask(inputs)


# -------------------------------------------------------------------
# 7. Modèle Transformer pour classification CL
# -------------------------------------------------------------------

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, oobatake_matrix)
x = embedding_layer(inputs)

transformer_block1 = TransformerBlock(model_dim, num_heads, ff_dim, rate=0.2)
transformer_block2 = TransformerBlock(model_dim, num_heads, ff_dim, rate=0.2)

x = transformer_block1(x)
x = transformer_block2(x)

# x = layers.GlobalAveragePooling1D()(x)

mask = embedding_layer.token_emb.compute_mask(inputs) 
mask = ops.cast(mask, x.dtype)
mask = ops.expand_dims(mask, axis=-1)
x_sum = ops.sum(x * mask, axis=1)
len_nonpad = ops.maximum(ops.sum(mask, axis=1), 1e-9)
x = x_sum / len_nonpad

x = layers.Dropout(0.1)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model_transfo = keras.Model(inputs=inputs, outputs=outputs, name="transformer_protein")

model_transfo.summary()

# model_transfo.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"],
# )

opt = Adam(learning_rate=3e-4, clipnorm=1.0)
loss = keras.losses.SparseCategoricalCrossentropy()
model_transfo.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

history_transfo = model_transfo.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=[early_stop],
)

# -------------------------------------------------------------------
# 8. Exemple : prédire la classe CL pour une séquence donnée
# -------------------------------------------------------------------

def predict_cl_for_sequence(seq):
    """Prend une séquence d'AA (string) et renvoie la CL prédite (valeur d'origine)."""
    encoded = [char_to_idx.get(c, 0) for c in seq]
    encoded = keras.utils.pad_sequences([encoded], maxlen=maxlen, padding="post", truncating="post")
    probs = model_transfo.predict(encoded)
    pred_idx = int(np.argmax(probs, axis=-1)[0])
    return idx_to_cl[pred_idx], probs[0][pred_idx]

# Exemple sur la première séquence du dataset
example_seq = sequences[0]
true_cl = labels_raw[0]
pred_cl, pred_prob = predict_cl_for_sequence(example_seq)
print("\nSéquence exemple :", example_seq[:60], "...")
print("CL réelle :", true_cl)
print("CL prédite :", pred_cl)

# -------------------------------------------------------------------
# 9. Panels de courbes CNN vs Transformer
# -------------------------------------------------------------------

def plot_history_panels(history_cnn, history_transfo, out_path="training_panels.png"):
    # Récupération des historiques
    c_acc = history_cnn.history.get("accuracy", [])
    c_val_acc = history_cnn.history.get("val_accuracy", [])
    c_loss = history_cnn.history.get("loss", [])
    c_val_loss = history_cnn.history.get("val_loss", [])
    c_epochs = range(1, len(c_acc) + 1)

    t_acc = history_transfo.history.get("accuracy", [])
    t_val_acc = history_transfo.history.get("val_accuracy", [])
    t_loss = history_transfo.history.get("loss", [])
    t_val_loss = history_transfo.history.get("val_loss", [])
    t_epochs = range(1, len(t_acc) + 1)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # (1) Loss CNN
    ax = axs[0, 0]
    ax.plot(c_epochs, c_loss, label="Train loss")
    ax.plot(c_epochs, c_val_loss, linestyle="--", label="Val loss")
    ax.set_title("CNN – Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

    # (2) Loss Transformer
    ax = axs[0, 1]
    ax.plot(t_epochs, t_loss, label="Train loss")
    ax.plot(t_epochs, t_val_loss, linestyle="--", label="Val loss")
    ax.set_title("Transformer – Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

    # (3) Accuracy CNN
    ax = axs[1, 0]
    ax.plot(c_epochs, c_acc, label="Train acc")
    ax.plot(c_epochs, c_val_acc, linestyle="--", label="Val acc")
    ax.set_title("CNN – Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.grid(True)
    ax.legend()

    # (4) Accuracy Transformer
    ax = axs[1, 1]
    ax.plot(t_epochs, t_acc, label="Train acc")
    ax.plot(t_epochs, t_val_acc, linestyle="--", label="Val acc")
    ax.set_title("Transformer – Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Panel sauvegardé sous : {out_path}")

# Appel
plot_history_panels(history_cnn, history_transfo, out_path="training_panels.png")

# TOO KEEP

## Fourni aux étudiants : matrice d'Oobatake
# oobatake_nonbonded = {
#     "A": 0.17,
#     "R": 0.81,
#     "N": -0.42,
#     "D": -1.23,
#     "C": 0.24,
#     "E": -1.27,
#     "Q": -0.58,
#     "G": 0.01,
#     "H": 0.15,
#     "I": 0.25,
#     "L": 0.53,
#     "K": 0.99,
#     "M": 0.09,
#     "F": 0.37,
#     "P": 0.45,
#     "S": -0.13,
#     "T": -0.14,
#     "W": 0.30,
#     "Y": 0.27,
#     "V": 0.07,
# }

# oobatake_dim = 1

# oobatake_matrix = np.zeros((vocab_size, oobatake_dim), dtype="float32")
# for aa, idx in char_to_idx.items():
#     value = oobatake_nonbonded.get(aa, 0.0)
#     oobatake_matrix[idx, 0] = value
# 
## Partie CNN
# 
# inputs_cnn = keras.Input(shape=(maxlen,))

# tok_emb_cnn = layers.Embedding(
#     input_dim=vocab_size,
#     output_dim=embed_dim,
#     input_length=maxlen,
# )

# oobatake_emb_cnn = layers.Embedding(
#     input_dim=vocab_size,
#     output_dim=oobatake_dim,
#     input_length=maxlen,
#     weights=[oobatake_matrix],
#     trainable=False,
# )

# x_token = tok_emb_cnn(inputs_cnn)
# x_oobatake = oobatake_emb_cnn(inputs_cnn)
# x = layers.Concatenate(axis=-1)([x_token, x_oobatake])

# x = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, activation="relu")(x)

# x = layers.MaxPooling1D(pool_size=pool_size)(x)
# x = layers.GlobalMaxPooling1D()(x)

# x = layers.Dropout(0.5)(x)
# x = layers.Dense(64, activation="relu")(x)
# x = layers.Dropout(0.5)(x)

# outputs_cnn = layers.Dense(num_classes, activation="softmax")(x)

# model_cnn = keras.Model(inputs_cnn, outputs_cnn, name="cnn_1d_protein")

# model_cnn.summary()

# model_cnn.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"],
# )

# history_cnn = model_cnn.fit(
#     x_train,
#     y_train,
#     batch_size=batch_size,
#     epochs=epochs,
#     validation_data=(x_val, y_val),
# )
# 
## partie self attention
#
#
# inputs_att = keras.Input(shape=(maxlen,), name="inputs_att")

# tok_emb_att = layers.Embedding(
#     input_dim=vocab_size,
#     output_dim=embed_dim,
#     input_length=maxlen,
#     mask_zero=True,
# )

# oobatake_emb_att = layers.Embedding(
#     input_dim=vocab_size,
#     output_dim=oobatake_dim,
#     input_length=maxlen,
#     weights=[oobatake_matrix],
#     trainable=False,
#     mask_zero=True,
# )

# x_token_att = tok_emb_att(inputs_att)
# x_ooba_att = oobatake_emb_att(inputs_att)

# x_att = layers.Concatenate(axis=-1)([x_token_att, x_ooba_att])

# att_layer = layers.Attention(name="self_attention")
# mask = tok_emb_att.compute_mask(inputs_att)
# x_att = att_layer([x_att, x_att], mask=[mask, mask])

# x_att = layers.Conv1D(filters=num_filters, kernel_size=kernel_size, activation="relu")(x_att)
# x_att = layers.MaxPooling1D(pool_size=pool_size)(x_att)
# x_att = layers.GlobalMaxPooling1D()(x_att)

# x_att = layers.Dropout(0.2)(x_att)
# x_att = layers.Dense(64, activation="relu")(x_att)
# x_att = layers.Dropout(0.2)(x_att)

# outputs_att = layers.Dense(num_classes, activation="softmax")(x_att)

# model_att_cnn = keras.Model(inputs_att, outputs_att, name="self_attention_cnn_protein")

# model_att_cnn.summary()

# opt = Adam(learning_rate=3e-4, clipnorm=1.0)
# loss = keras.losses.SparseCategoricalCrossentropy()

# model_att_cnn.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

# early_stop = EarlyStopping(
#     monitor="val_loss",
#     patience=10,
#     restore_best_weights=True,
# )

# history_att_cnn = model_att_cnn.fit(
#     x_train,
#     y_train,
#     batch_size=batch_size,
#     epochs=epochs,
#     validation_data=(x_val, y_val),
#     callbacks=[early_stop],
# )x = layers.MaxPooling1D(pool_size=pool_size)(x)
x = layers.GlobalMaxPooling1D()(x)