"""
Assemble the AnnData object and verify RNA<->protein pairing per tissue.
"""
import pandas as pd
import anndata as ad
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

OUT = Path(__file__).resolve().parent.parent / "output"
TISSUES = ["kidney", "liver", "muscle", "skin"]

obs = pd.read_csv(OUT / "Earth_obs.csv").set_index("sample_id")
var = pd.read_csv(OUT / "Earth_var.csv").set_index("gene_symbol")
counts = pd.read_csv(OUT / "Earth_expression_counts.csv", index_col=0)

adata = ad.AnnData(X=counts.T.values.astype("float32"), obs=obs, var=var)
print(f"[Earth] AnnData assembled: {adata.shape[0]} samples x {adata.shape[1]} genes")

for tissue in TISSUES:
    prot = pd.read_csv(OUT / f"Earth_{tissue}_protein_matrix.csv", index_col=0)
    rna_ids = [s for s in adata.obs_names if s.startswith(f"{tissue}_")]
    if list(prot[rna_ids].columns) != rna_ids or prot[rna_ids].isna().any().any():
        raise ValueError(
            f"{tissue}: protein sample columns do not 1:1 match RNA ids -- pairing broken"
        )
    print(f"  {tissue}: protein {prot.shape} pairs 1:1 with RNA ({len(rna_ids)} samples)")

for tissue in TISSUES:
    mask = adata.obs["tissue"] == tissue
    sub = adata[mask]
    y = (sub.obs["age_months"] == 30).astype(int)
    scores = cross_val_score(
        LogisticRegression(max_iter=1000), sub.X, y, cv=4, scoring="roc_auc"
    )
    print(f"  {tissue}: 6M-vs-30M probe AUROC = {sorted(scores)} "
          f"(sanity check only, 4v4 labels)")

print("Protein matrices: Earth_{tissue}_protein_matrix.csv (Task A validation, per tissue).")
print("Adequacy gate reminder: RNA values are normalized counts -- Geneformer tokenization")
print("needs raw counts; record the per-model exclusion decision before embedding.")