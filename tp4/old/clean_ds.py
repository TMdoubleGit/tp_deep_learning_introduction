import pandas as pd


input_csv = "2018-06-06-ss.cleaned.csv"
output_csv = "dataset_2.csv"

df = pd.read_csv(input_csv)

df = df[df["has_nonstd_aa"] == False].reset_index(drop=True)
cols_to_keep = ["pdb_id", "chain_code", "seq", "sst3", "len"]
df_out = df[cols_to_keep].copy()

df_out = df_out.dropna(subset=["seq", "sst3"])
df_out = df_out[df_out["len"] > 30].reset_index(drop=True)
df_out = df_out[df_out["len"] < 128].reset_index(drop=True)

df_out.to_csv(output_csv, index=False)

print(f"Dataset réduit sauvegardé dans : {output_csv}")
print(df_out.head())
