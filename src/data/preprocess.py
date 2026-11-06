import numpy as np
import pandas as pd
import anndata as ad
from typing import Any

import json
import warnings


# --------------------------------------------------
# Dataset standardization
# --------------------------------------------------

def standardize_osdr(data: Any, **kwargs) -> Any:
    """Return a standardized multimodal representation of loaded OSDR data."""
    raise NotImplementedError("Implement OSDR standardization before running preprocessing.")


def standardize_archs4(data: Any, **kwargs) -> Any:
    """Return a standardized representation of loaded ARCHS4 data."""
    raise NotImplementedError("Implement ARCHS4 standardization before running preprocessing.")


# --------------------------------------------------
# Model-specific preprocessing
# --------------------------------------------------
def prep_logreg(data: dict[str, pd.DataFrame], *, include_protein: bool = True, column_annotations=None) -> ad.AnnData:
    """Assemble loaded interim CSVs; the classifier uses RNA counts only."""

    def find_table(suffixes: tuple[str, ...], required: bool = True):
        matches = [name for name in data if name.endswith(suffixes)]
        if len(matches) > 1:
            raise ValueError(f"Multiple matching tables: {matches}")
        if not matches:
            if required:
                raise ValueError(f"Missing table with suffix in {suffixes}")
            return None
        return data[matches[0]].copy()

    obs = find_table(("_obs.csv",)).set_index("sample_id")
    var = find_table(("_var.csv",)).set_index("ensembl_id")
    counts = find_table(("_expression_counts.csv", "_expressioncount.csv"))

    # CSV reader retains the first column containing the gene identifiers.
    counts = counts.set_index(counts.columns[0])

    if not obs.index.is_unique or not var.index.is_unique:
        raise ValueError("Sample and gene identifiers must be unique.")
    if not counts.index.is_unique or not counts.columns.is_unique:
        raise ValueError("Expression matrix identifiers must be unique.")

    if set(counts.columns) != set(obs.index):
        raise ValueError("Expression sample IDs do not match obs.sample_id.")
    if set(counts.index) != set(var.index):
        raise ValueError("Expression gene IDs do not match var.ensembl_id.")

    # Align both axes explicitly before discarding DataFrame labels.
    counts = counts.reindex(index=var.index, columns=obs.index)
    X = counts.T.to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError("RNA expression contains missing or infinite values.")
    if (X < 0).any():
        raise ValueError("Raw RNA counts contain negative values.")

    adata = ad.AnnData(X=X, obs=obs, var=var)
    if include_protein:
        protein = find_table(("_protein_matrix.csv", "_protein_matric.csv"), required=False)
        if protein is not None:
            protein = protein.set_index(list(protein.columns[:2]))
            if not protein.columns.is_unique:
                raise ValueError("Protein sample identifiers must be unique.")

            missing = set(obs.index) - set(protein.columns)
            if missing:
                raise ValueError(f"Protein matrix missing samples: {sorted(missing)}")

            protein = protein.reindex(columns=obs.index)
            adata.obsm["protein"] = protein.T.to_numpy(dtype=float)

            # Keep feature identities, including plex, with the protein matrix.
            adata.uns["protein_features"] = protein.index.to_frame(index=False).to_dict("list")

    if column_annotations is not None:
        annotations = pd.DataFrame(column_annotations, columns=["table", "column_or_axis", "source", "description"])
        adata.uns["column_provenance"] = annotations.to_dict("list")

    return adata

def prep_simple_mlp(data: dict[str, pd.DataFrame], **kwargs) -> ad.AnnData:
    """Same raw-count AnnData as logreg; train_mlp applies log1p and per-fold scaling."""
    return prep_logreg(data, **kwargs)

def _bin_expression(X: np.ndarray, n_bins: int) -> np.ndarray:
    """scGPT-style per-sample quantile binning (zeros stay 0, non-zeros -> 1..n_bins-1)."""
    out = np.zeros(X.shape, dtype=np.int64)
    q = np.linspace(0, 1, n_bins - 1)
    for i, row in enumerate(X):
        nz = row > 0
        if not nz.any():
            continue
        edges = np.quantile(row[nz], q)
        out[i, nz] = np.clip(np.digitize(row[nz], edges, right=True), 1, n_bins - 1)
    return out


def prep_scgpt_lora(
    data: dict[str, pd.DataFrame],
    *,
    vocab_path: str,
    gene_symbol_col: str = "human_symbol",
    n_genes: int = 1200,
    n_bins: int = 51,
    target_sum: float = 1e4,
    include_protein: bool = True,
    column_annotations=None,
) -> ad.AnnData:
    """AnnData ready for scGPT fine-tuning (one dataset at a time).

    adata.X                      log-normalized expression
    adata.layers["binned"]       int bins, the actual scGPT input
    adata.var["scgpt_gene_id"]   vocab ids
    adata.obsm["protein_z"]      within-study z-scored proteins (NaN where missing)
    adata.obsm["protein_mask"]   True where a protein was observed
    """
    adata = prep_logreg(data, include_protein=include_protein, column_annotations=column_annotations)

    if gene_symbol_col not in adata.var.columns:
        raise ValueError(
            f"var has no '{gene_symbol_col}' column; set gene_symbol_col to the column "
            f"holding human ortholog symbols. Available: {list(adata.var.columns)}"
        )
    with open(vocab_path) as f:
        vocab = json.load(f)  # scGPT vocab.json: token -> id

    # Library size from ALL genes, before any subsetting.
    totals = adata.X.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("At least one sample has zero total counts.")

    symbols = adata.var[gene_symbol_col].astype(object).to_numpy()
    in_vocab = np.array([isinstance(s, str) and s in vocab for s in symbols])
    first_occurrence = ~pd.Series(symbols).duplicated().to_numpy()
    keep = in_vocab & first_occurrence
    if keep.sum() == 0:
        raise ValueError("No genes matched scGPT's vocab; check gene_symbol_col and vocab_path.")

    adata = adata[:, keep].copy()
    symbols = symbols[keep]
    X = adata.X / totals * target_sum
    X = np.log1p(X)

    n_genes = min(n_genes, X.shape[1])
    top = np.sort(np.argsort(X.var(axis=0))[::-1][:n_genes])  # keep stable gene order

    out = adata[:, top].copy()
    out.X = X[:, top].astype(np.float32)
    out.layers["binned"] = _bin_expression(X[:, top], n_bins)
    out.var["scgpt_symbol"] = symbols[top]
    out.var["scgpt_gene_id"] = [vocab[s] for s in symbols[top]]

    if "protein" in out.obsm:
        P = np.asarray(out.obsm["protein"], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns
            mu = np.nanmean(P, axis=0)
            sd = np.nanstd(P, axis=0)
        sd = np.where(~np.isfinite(sd) | (sd < 1e-8), 1.0, sd)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        out.obsm["protein_z"] = (P - mu) / sd
        out.obsm["protein_mask"] = ~np.isnan(P)
        out.uns["protein_norm"] = {"mean": mu, "std": sd}

    out.uns["scgpt_prep"] = {
        "n_genes": int(n_genes),
        "n_bins": int(n_bins),
        "target_sum": float(target_sum),
        "genes_in_vocab": int(in_vocab.sum()),
        "genes_before_filter": int(len(in_vocab)),
        "gene_symbol_col": gene_symbol_col,
    }
    return out


def prep_scgpt_deepattn(data: Any, **kwargs) -> Any:
    """Return standardized data prepared for scGPT deep attention."""
    raise NotImplementedError("Implement scGPT deep attention preprocessing.")


def prep_pca(data: Any, **kwargs) -> Any:
    """Return standardized data prepared for PCA."""
    raise NotImplementedError("Implement PCA preprocessing.")


def prep_single_hyena(data: Any, **kwargs) -> Any:
    """Return standardized data prepared for single Hyena."""
    raise NotImplementedError("Implement single Hyena preprocessing.")