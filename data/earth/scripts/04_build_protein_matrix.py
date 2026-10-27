"""
Build per-tissue protein matrices. Isobarquant signal_sum per TMT channel.

Output: one CSV per tissue (index gene_symbol, 16 sample columns named
{tissue}_{age}M{rep}, the same ids as obs and the RNA matrix).
"""
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "raw"
OUT = Path(__file__).resolve().parent.parent / "output"

PROTEIN_FILES = {
    "Kidney_merged_results_20220530_2359_proteins.txt": ("kidney", "metadata_P2067.csv"),
    "Muscle_merged_results_20220531_1130_proteins.txt": ("muscle", "metadata_P2067.csv"),
    "Skin_merged_results_20220531_1312_proteins.txt": ("skin", "metadata_P2067.csv"),
    "liver_merged_results_20221213_2127_proteins.txt": ("liver", "metadata_P2375.csv"),
}

for result_file, (tissue, meta_name) in PROTEIN_FILES.items():
    meta = pd.read_csv(RAW / meta_name)
    chan_map = meta[meta["file"] == result_file].set_index("col.name")
    sample_of_channel = {
        chan: f"{tissue}_{int(r['condition'].removeprefix('t').removesuffix('mo'))}M"
              f"{int(r['rep'].removeprefix('rep'))}"
        for chan, r in chan_map.iterrows()
    }

    prot = pd.read_csv(RAW / result_file, sep="\t")
    signal_cols = [c for c in prot.columns if c.startswith("signal_sum_")]
    if len(signal_cols) != 16:
        raise ValueError(f"{result_file}: expected 16 signal_sum columns, found {len(signal_cols)}")

    n_raw = len(prot)
    pid = prot["protein_id"].astype(str)
    gname = prot["gene_name"].astype(str)
    decoy = pid.str.contains("REV", na=False) | gname.str.contains("#", na=False)
    prot = prot[~decoy]
    n_decoy = int(decoy.sum())
    if n_decoy:
        print(f"    dropped {n_decoy}/{n_raw} protein-group rows as reverse-decoy "
              f"(###REV###) entries")

    # Drop protein groups attributed to several genes (pipe-joined names)
    ambiguous = gname.loc[prot.index].str.contains(r"\|", regex=True)
    n_ambig = int(ambiguous.sum())
    if n_ambig:
        prot = prot[~ambiguous]
        print(f"    dropped {n_ambig} protein-group rows with pipe-joined "
              f"multi-gene names ")

    per_gene = prot.groupby("gene_name").agg(
        **{sample_of_channel[c]: (c, "sum") for c in signal_cols},
        n_protein_groups=("protein_id", "size"),
        uniprot_accession=("protein_id", lambda s: ";".join(sorted(s.dropna().unique()))),
    )

    if per_gene.index.astype(str).str.contains(r"[#|]", regex=True).any():
        raise ValueError(f"{tissue}: pipe-joined identifiers survived the filter")

    multi = (per_gene["n_protein_groups"] > 1).sum()
    per_gene.to_csv(OUT / f"Earth_{tissue}_protein_matrix.csv")
    print(f"  {tissue}: {per_gene.shape[0]} genes x 16 samples "
          f"({multi} genes from multiple protein groups, signal_sum summed)")