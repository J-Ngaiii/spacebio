"""
Merge RNA raw counts (all studies share the same 57,186-gene Ensembl
spine) + sample metadata. OSD-48 joins as RNA-only (has_protein=False).

Usage: python 06_merge_studies.py   (run from the directory containing the
per-study folders: osd105/ osd101/ osd102/ osd137/ osd48/)
"""
import pandas as pd
import anndata as ad
from pathlib import Path

STUDIES = ["osd105", "osd101", "osd102", "osd137", "osd48"]
STUDY_IDS = {
    "osd105": "OSD-105", "osd101": "OSD-101", "osd102": "OSD-102",
    "osd137": "OSD-137", "osd48": "OSD-48",
}
NO_PROTEIN_STUDIES = {"OSD-48"}

SHARED_VAR_COLS = ["mgi_symbol", "human_ortholog", "human_homology_type"]

ROOT = Path(__file__).resolve().parent

obs_parts, count_parts = [], []
var_reference = None

for folder in STUDIES:
    study_id = STUDY_IDS[folder]
    out = ROOT / folder / "output"

    obs = pd.read_csv(out / f"{study_id}_obs.csv").set_index("sample_id")
    var = pd.read_csv(out / f"{study_id}_var.csv").set_index("ensembl_id")
    counts = pd.read_csv(out / f"{study_id}_expression_counts.csv", index_col=0)

    if var_reference is None:
        var_reference = var
        var_shared = var[SHARED_VAR_COLS]
        spine = var.index
    else:
        if set(var.index) != set(spine):
            raise ValueError(
                f"{study_id}: gene set differs from the reference spine "
                f"({len(var)} vs {len(spine)} genes) — investigate before merging"
            )
        shared_ref = var_reference.loc[var.index, SHARED_VAR_COLS]
        if not shared_ref.equals(var[SHARED_VAR_COLS]):
            raise ValueError(f"{study_id}: shared var columns differ from the reference spine")
        counts = counts.reindex(spine)

    counts = counts.T  # samples as rows
    obs["study_id"] = study_id
    obs["has_protein"] = study_id not in NO_PROTEIN_STUDIES
    obs_parts.append(obs)
    count_parts.append(counts)
    print(f"  {study_id}: {counts.shape[0]} samples "
          f"({'RNA-only' if study_id in NO_PROTEIN_STUDIES else 'paired'})")

obs_merged = pd.concat(obs_parts)
X = pd.concat(count_parts).values

adata = ad.AnnData(X=X, obs=obs_merged, var=var_shared)
adata.X = adata.X.astype("float32")

print(f"\nmerged object: {adata.shape[0]} samples x {adata.shape[1]} genes")
print(f"  studies: {adata.obs['study_id'].value_counts().to_dict()}")
print(f"  tissues: {adata.obs['tissue'].value_counts().to_dict()}")
print(f"  missions: {adata.obs['mission'].value_counts().to_dict()}")
print(f"  arms: {adata.obs['arm'].value_counts().to_dict()}")
print(f"  paired (has_protein): {adata.obs['has_protein'].sum()} of {adata.shape[0]}")

# Task B needs two arms without NaNs.
y = (adata.obs["arm"] == "Flight").astype(int)
if y.isna().any() or y.nunique() != 2:
    raise ValueError("Task B label column broken after merge — check arm mapping")

OUT = ROOT / "merged_output"
OUT.mkdir(exist_ok=True)
obs_merged.to_csv(OUT / "merged_obs.csv")
var_shared.to_csv(OUT / "merged_var.csv")
adata.write_h5ad(OUT / "spacebio_merged_rna.h5ad")
print(f"\nwrote {OUT}/: merged_obs.csv, merged_var.csv, spacebio_merged_rna.h5ad")
print("NOTE: protein matrices stay in their per-study output folders (Task A, within-study).")
