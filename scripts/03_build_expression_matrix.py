"""
Build the expression matrix (.X) for an OSD study.
Align to obs/var row order & save raw counts.
Normalization is a per-task decision made downstream.

Usage: python 03_build_expression_matrix.py [STUDY_ID]
"""
import sys
import pandas as pd
from pathlib import Path

STUDY_ID = sys.argv[1] if len(sys.argv) > 1 else "OSD-105"

RAW = Path(__file__).resolve().parent.parent / "raw"
OUT = Path(__file__).resolve().parent.parent / "output"

obs = pd.read_csv(OUT / f"{STUDY_ID}_obs.csv")
var = pd.read_csv(OUT / f"{STUDY_ID}_var.csv")

count_files = list(RAW.glob("*_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv"))
if len(count_files) != 1:
    raise FileNotFoundError(
        f"Expected exactly 1 RSEM unnormalized counts CSV in {RAW}, "
        f"found {len(count_files)}: {[p.name for p in count_files]}"
    )
counts = pd.read_csv(count_files[0], index_col=0)

# Align to the exact row/column order of var/obs.
counts = counts.loc[var["ensembl_id"], obs["sample_id"]]

counts.to_csv(OUT / f"{STUDY_ID}_expression_counts.csv")
print(f"[{STUDY_ID}] expression matrix: {counts.shape[0]} genes x {counts.shape[1]} samples (raw counts)")