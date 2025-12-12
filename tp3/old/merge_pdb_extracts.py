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
                # On enregistre l'ancienne séquence
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq_chunks)
                # Nouveau header
                header = line[1:]  # on enlève '>'
                current_id = header.split()[0]  # FA-DOMID
                current_seq_chunks = []
            else:
                current_seq_chunks.append(line)

        # Dernière séquence
        if current_id is not None:
            sequences[current_id] = "".join(current_seq_chunks)

    return sequences


def read_second_file(cls_path):
    """
    Lit le second extract et retourne un dict:
        { FA-DOMID (str) : CL (str) }
    """
    id_to_cl = {}

    with open(cls_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            # Trouver le champ contenant CL
            cl_value = None
            for field in parts:
                if field.startswith("TP="):
                    for item in field.split(","):
                        if item.startswith("CL="):
                            cl_value = item.split("=", 1)[1]
                            break
                if cl_value is not None:
                    break

            if cl_value is None:
                continue

            # Les tokens purement numériques sont les FA-DOMID
            for field in parts:
                if field.isdigit():
                    id_to_cl[field] = cl_value

    return id_to_cl


def merge_files(fa_path, cls_path, out_path):
    seqs = read_fa_file(fa_path)
    id_to_cl = read_second_file(cls_path)

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["ID", "SEQUENCE", "CL"])  # en-tête

        for domid, seq in seqs.items():
            cl = id_to_cl.get(domid)
            if cl is None:
                continue
            writer.writerow([domid, seq, cl])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} FA_FILE SECOND_FILE OUTPUT_CSV")
        sys.exit(1)

    fa_file = sys.argv[1]
    second_file = sys.argv[2]
    output_file = sys.argv[3]

    merge_files(fa_file, second_file, output_file)
