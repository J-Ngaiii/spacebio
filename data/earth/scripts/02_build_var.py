"""
Build the gene table (.var).
The RNA matrices are keyed by mouse gene symbol, not Ensembl ID.
"""
import gzip
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "raw"
OUT = Path(__file__).resolve().parent.parent / "output"
REF = Path(__file__).resolve().parent.parent.parent / "reference"

TISSUES = ["Kidney", "Liver", "Muscle", "Skin"]

spines = {}
for tissue in TISSUES:
    with gzip.open(RAW / f"GSE225576_{tissue}.txt.gz", "rt") as f:
        text = f.read().replace("\r\n", "\n").replace("\r", "\n")
    symbols = [line.split("\t", 1)[0] for line in text.split("\n") if line]
    symbols = symbols[1:]
    spines[tissue] = symbols
    print(f"  {tissue}: {len(symbols)} gene rows")

ref = spines["Kidney"]
for tissue in TISSUES[1:]:
    if set(spines[tissue]) != set(ref):
        raise ValueError(f"{tissue} gene set differs from Kidney -- cannot merge matrices")

if len(set(ref)) != len(ref):
    dupes = pd.Series(ref).value_counts()
    dupes = dupes[dupes > 1]
    raise ValueError(f"{len(dupes)} duplicated gene symbols in RNA spine: "
                     f"{dupes.head(10).to_dict()} -- resolve before continuing")

var = pd.DataFrame({"gene_symbol": ref})

orthologs = pd.read_csv(REF / "mouse_human_orthologs.tsv", sep="\t")
orthologs = orthologs.rename(columns={
    "Gene stable ID": "ensembl_id",
    "Gene name": "mgi_symbol",
    "Human gene name": "human_ortholog",
    "Human homology type": "human_homology_type",
})

per_symbol = orthologs.groupby("mgi_symbol").agg(
    mouse_ensembl_id=("ensembl_id", lambda s: ";".join(sorted(s.dropna().unique())) or pd.NA),
    human_ortholog=("human_ortholog", lambda s: ";".join(sorted(s.dropna().unique())) or pd.NA),
    human_homology_type=("human_homology_type", "first"),
    n_crosswalk_rows=("ensembl_id", "size"),
)
var = var.merge(per_symbol, left_on="gene_symbol", right_index=True, how="left")
var["ambiguous_symbol_map"] = var["n_crosswalk_rows"] > 1

unmapped = var["human_ortholog"].isna().sum()
ambiguous = var["ambiguous_symbol_map"].sum()
print(f"NOTE: {unmapped} RNA symbols have no crosswalk entry -- kept, not dropped")
print(f"NOTE: {ambiguous} symbols map to multiple Ensembl IDs -- flagged ambiguous_symbol_map")

var.to_csv(OUT / "Earth_var.csv", index=False)
print(f"[Earth] var table: {var.shape[0]} genes x {var.shape[1]} columns")