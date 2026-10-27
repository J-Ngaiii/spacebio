"""
Assemble the AnnData object and verify RNA-protein pairing per tissue.
RNA values are normalized counts.
"""
import pandas as pd
import anndata as ad
from pathlib import Path

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
            f"{tissue}: protein sample columns do not match RNA ids 1 to 1"
        )
    print(f"  {tissue}: protein {prot.shape} pairs 1:1 with RNA ({len(rna_ids)} samples)")