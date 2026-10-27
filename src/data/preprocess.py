import numpy as np
import pandas as pd
import anndata as ad
from typing import Any


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

def prep_scgpt_lora(data: Any, **kwargs) -> Any:
    """Return standardized data prepared for scGPT LoRA."""
    raise NotImplementedError("Implement scGPT LoRA preprocessing.")


def prep_scgpt_deepattn(data: Any, **kwargs) -> Any:
    """Return standardized data prepared for scGPT deep attention."""
    raise NotImplementedError("Implement scGPT deep attention preprocessing.")


def prep_pca(data: Any, **kwargs) -> Any:
    """Return standardized data prepared for PCA."""
    raise NotImplementedError("Implement PCA preprocessing.")


def prep_single_hyena(data: Any, **kwargs) -> Any:
    """Return standardized data prepared for single Hyena."""
    raise NotImplementedError("Implement single Hyena preprocessing.")


def prep_bulkformer(data: dict[str, pd.DataFrame], **kwargs) -> ad.AnnData:
    """Return RNA + protein AnnData with genes aligned to the BulkFormer gene vocabulary."""
    # TODO: start from prep_logreg(data, include_protein=True), map mouse Ensembl IDs to one-to-one
    # human orthologs, then reorder genes to the checkpoint's gene list and fill genes it lacks.
    raise NotImplementedError("Implement BulkFormer preprocessing.")