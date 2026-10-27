"""Shared protein-prediction protocol: every model uses the same targets, folds and metrics so their scores compare."""
import anndata as ad
import numpy as np
import torch
from sklearn.model_selection import KFold


def load_protein_targets(adata: ad.AnnData, protein_key: str = "protein", log2: bool = True) -> np.ndarray:
    """Return the sample x protein matrix as float32, log2-transformed if requested; NaN marks unmeasured values."""
    if protein_key not in adata.obsm:
        raise ValueError(f"adata.obsm[{protein_key!r}] is missing; prepare the data with include_protein=True.")
    Y = np.asarray(adata.obsm[protein_key], dtype=np.float64)
    if log2:
        if (Y[~np.isnan(Y)] <= 0).any():
            raise ValueError("log2_protein needs positive protein values; set log2_protein=False if they are already on a log scale.")
        Y = np.log2(Y)
    return Y.astype(np.float32)

def zscore_stats(Y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-protein mean and std from training samples only, so validation values never leak into training."""
    mean = np.nan_to_num(np.nanmean(Y_train, axis=0))
    std = np.nanstd(Y_train, axis=0)
    std[~(std > 0)] = 1.0
    return mean, std

def kfold_splits(n_samples: int, cv_folds: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_index, val_index) pairs shared by every model."""
    # TODO: replace random KFold with the team's shared held-out split (e.g. by study, tissue or plex).
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    return list(cv.split(np.zeros((n_samples, 1))))

def masked_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE over measured proteins only; NaN targets were not measured and are skipped."""
    measured = ~torch.isnan(targets)
    if not measured.any():
        return predictions.sum() * 0.0
    return ((predictions[measured] - targets[measured]) ** 2).mean()

def per_protein_pearson(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Pearson r per protein across samples, using measured values only; NaN where it is undefined."""
    scores = np.full(targets.shape[1], np.nan)
    for j in range(targets.shape[1]):
        measured = ~np.isnan(targets[:, j])
        if measured.sum() < 3 or targets[measured, j].std() == 0 or predictions[measured, j].std() == 0:
            continue
        scores[j] = np.corrcoef(predictions[measured, j], targets[measured, j])[0, 1]
    return scores

def pooled_pearson(predictions: np.ndarray, targets: np.ndarray) -> float:
    """One Pearson r over every measured (sample, protein) value; still defined when each protein has few values in a fold."""
    measured = ~np.isnan(targets)
    if measured.sum() < 3 or targets[measured].std() == 0 or predictions[measured].std() == 0:
        return float("nan")
    return float(np.corrcoef(predictions[measured], targets[measured])[0, 1])
