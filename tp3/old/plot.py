#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# --- Paramètres ---
csv_path = "db_merged.csv"  # <-- adapte le chemin si besoin
save_path = "class_distribution.png"  # fichier de sortie optionnel

# --- Lecture du CSV ---
df = pd.read_csv(csv_path)
if "CL" not in df.columns:
    raise ValueError("La colonne 'CL' est introuvable dans le CSV.")

# --- Nettoyage minimal ---
df = df.dropna(subset=["CL"])

# --- Comptage des classes ---
class_counts = df["CL"].value_counts().sort_index()

print("Nombre total d'échantillons :", len(df))
print("\nDistribution des classes :")
print(class_counts)

# --- Plot ---
plt.figure(figsize=(10, 6))
bars = plt.bar(class_counts.index.astype(str), class_counts.values)

plt.title("Répartition des séquences par classe CL")
plt.xlabel("Classe CL")
plt.ylabel("Nombre de séquences")
plt.xticks(rotation=45, ha="right")

# Ajout des valeurs sur les barres
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

plt.tight_layout()
plt.savefig(save_path, dpi=150)
plt.show()

print(f"\nGraphique sauvegardé sous : {save_path}")
