import anndata as ad
import numpy as np
import pandas as pd


def rna_log_tpm(adata: ad.AnnData) -> np.ndarray:
    """log1p(TPM) model input, shared by every protein model.

    Uses layers["log_tpm"] when prep already computed it with species-correct gene lengths (mouse data from
    prep_bulkformer); otherwise converts raw counts in X with BulkFormer's bundled human gene lengths.
    """
    if "log_tpm" in adata.layers:
        return np.asarray(adata.layers["log_tpm"], dtype=np.float64)
    from bulkformer.preprocessing import normalize_data

    counts = pd.DataFrame(np.asarray(adata.X, dtype=np.float64), index=adata.obs_names, columns=adata.var_names)
    return normalize_data(counts).to_numpy(dtype=np.float64)
