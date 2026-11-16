
import anndata as ad
import numpy as np
import pandas as pd

# Load CSVs (rows = samples, columns = features)
rna = pd.read_csv("osdr_paired_expression_intrain.csv", index_col=0)
protein = pd.read_csv("osdr_paired_protein_intrain.csv", index_col=0)

# Keep samples present in both datasets
common_samples = rna.index.intersection(protein.index)

rna = rna.loc[common_samples]
protein = protein.loc[common_samples]

# Create AnnData
adata = ad.AnnData(
    X=rna.to_numpy(dtype=np.float32),
    obs=pd.DataFrame(index=common_samples.copy()),
    var=pd.DataFrame(index=rna.columns.copy()),
)

# Store protein targets
adata.obsm["protein"] = protein.to_numpy(dtype=np.float32)

# Store protein names
adata.uns["protein_features"] = protein.columns.tolist()

# Save
adata.write_h5ad("gene_protein_intersection_intrain.h5ad")

print(adata)
print("RNA:", adata.X.shape)
print("Protein:", adata.obsm["protein"].shape)
