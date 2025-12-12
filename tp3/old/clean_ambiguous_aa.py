#!/usr/bin/env python3
import sys
import pandas as pd

# AA non standards à remplacer par '*'
NON_STANDARD_AA = set(["B", "Z", "J", "X", "U", "O"])

def clean_sequence(seq: str) -> str:
    """Remplace B, Z, J, X, U, O par '*' dans une séquence."""
    return "".join("*" if c in NON_STANDARD_AA else c for c in seq)

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} INPUT_CSV OUTPUT_CSV")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    # Charge le dataset
    df = pd.read_csv(input_csv)

    if "SEQUENCE" not in df.columns:
        print("Erreur : la colonne 'SEQUENCE' n'existe pas dans le CSV.")
        print("Colonnes disponibles :", list(df.columns))
        sys.exit(1)

    # Nettoie la colonne 'seq'
    df["SEQUENCE"] = df["SEQUENCE"].astype(str).apply(clean_sequence)

    # Sauvegarde
    df.to_csv(output_csv, index=False)
    print(f"Fichier nettoyé sauvegardé dans : {output_csv}")

if __name__ == "__main__":
    main()
