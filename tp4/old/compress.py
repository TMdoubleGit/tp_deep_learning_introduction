import tarfile
import os

def compress_to_tar_gz(input_path, output_path):
    """
    Compresse un fichier ou dossier en .tar.gz
    """
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(input_path, arcname=os.path.basename(input_path))
    print(f"✔ {input_path} compressé dans {output_path}")

def main():
    # Liste des datasets à compresser
    datasets = [
        "dataset.csv",
        "dataset_2.csv",
        "2018-06-06-pdb-intersect-pisces.csv",
        "2018-06-06-ss.cleaned.csv"
    ]

    output_dir = "archives"
    os.makedirs(output_dir, exist_ok=True)

    for ds in datasets:
        if os.path.exists(ds):
            out_file = os.path.join(output_dir, f"{ds}.tar.gz")
            compress_to_tar_gz(ds, out_file)
        else:
            print(f"⚠ Le dataset '{ds}' n'existe pas, ignoré.")

if __name__ == "__main__":
    main()
