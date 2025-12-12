#!/usr/bin/env python3
import sys
import csv

def read_fa_file(fa_path):
    """
    Lit le fichier de type FA (fasta-like) et renvoie un dict:
        { FA-DOMID (str) : sequence (str) }
    """
    sequences = {}
    current_id = None
    current_seq_chunks = []

    with open(fa_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq_chunks)
                header = line[1:]  # retire '>'
                current_id = header.split()[0]  # FA-DOMID
                current_seq_chunks = []
            else:
                current_seq_chunks.append(line)

        if current_id is not None:
            sequences[current_id] = "".join(current_seq_chunks)

    return sequences


def read_second_file(cls_path):
    """
    Lit le second extract et retourne un dict:
        { FA-DOMID (str) : {"CL": str, "CF": str} }
    """
    id_to_labels = {}

    with open(cls_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            cl_value, cf_value = None, None

            # Champ style "TP=1,CL=1000003,CF=2000144,SF=3000034,FA=4000057"
            for field in parts:
                if field.startswith("TP="):
                    items = field.split(",")
                    for item in items:
                        if item.startswith("CL="):
                            cl_value = item.split("=", 1)[1]
                        elif item.startswith("CF="):
                            cf_value = item.split("=", 1)[1]
                    break

            if cl_value is None or cf_value is None:
                continue

            # Tous les tokens purement numériques de la ligne = FA-DOMID
            for field in parts:
                if field.isdigit():
                    id_to_labels[field] = {"CL": cl_value, "CF": cf_value}

    return id_to_labels


def merge_files(fa_path, cls_path, out_path):
    """
    Fusionne les deux fichiers :
    - relie FA-DOMID entre FA et second extract
    - garde le plus long représentant par CF
    """
    seqs = read_fa_file(fa_path)
    id_to_labels = read_second_file(cls_path)

    # Dictionnaire temporaire : { CF : (domid, seq, cl) } pour le plus long
    best_by_cf = {}

    for domid, seq in seqs.items():
        labels = id_to_labels.get(domid)
        if not labels:
            continue

        cl = labels["CL"]
        cf = labels["CF"]

        # Si on n’a pas encore ce CF ou si la séquence est plus longue
        if cf not in best_by_cf or len(seq) > len(best_by_cf[cf][1]):
            best_by_cf[cf] = (domid, seq, cl)

    # Écriture finale
    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["ID", "SEQUENCE", "CL", "CF"])  # en-tête

        for cf, (domid, seq, cl) in best_by_cf.items():
            writer.writerow([domid, seq, cl, cf])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} FA_FILE SECOND_FILE OUTPUT_CSV")
        sys.exit(1)

    fa_file = sys.argv[1]
    second_file = sys.argv[2]
    output_file = sys.argv[3]

    merge_files(fa_file, second_file, output_file)
