"""PCA + ridge baseline for predicting protein abundance from bulk RNA, with optional W&B tracking."""
from contextlib import nullcontext
from dataclasses import asdict
from uuid import uuid4
import anndata as ad

import numpy as np
import torch

from src.features.build_features import rna_log_tpm
from src.evaluation.evaluate import kfold_splits, load_protein_targets, masked_mse, per_protein_pearson, pooled_pearson, zscore_stats
from src.models.pca import PCARidge
from src.training.train_conf.pca_config import pca_train_config

def _config_warnings(config: pca_train_config):
    if config.n_components < 1 or config.cv_folds < 2:
        raise ValueError("At least one component and two CV folds are required.")
    if not config.alphas or min(config.alphas) <= 0:
        raise ValueError("alphas must be a nonempty list of positive penalties.")
    if config.wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline or disabled.")

def train_pca(adata: ad.AnnData, config: pca_train_config | None = None) -> dict:
    if config is None:
        config = pca_train_config()

    _config_warnings(config)

    X = np.asarray(adata.X, dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("RNA values must be finite; impute or drop genes with missing values before training.")
    if config.normalize_rna and (X < 0).any():
        raise ValueError("Raw RNA counts cannot be negative; set normalize_rna=False if X is already normalized (e.g. z-scored).")
    if len(X) < config.cv_folds:
        raise ValueError("At least cv_folds samples are required.")

    Y = load_protein_targets(adata, config.protein_key, log2=config.log2_protein)
    if config.normalize_rna:
        # Same RNA input as BulkFormer. TPM is per sample, so it cannot leak across folds.
        X = rna_log_tpm(adata)
    group = config.run_group or f"pca-{uuid4().hex[:12]}"
    wandb = None
    if config.wandb_mode != "disabled":
        import wandb

    def fit(train_index, val_index, name):
        mean, std = zscore_stats(Y[train_index])
        train_Y = (Y[train_index] - mean) / std
        # Scaler, PCA and ridge all see training samples only.
        model = PCARidge(config.n_components, alphas=config.alphas, random_state=config.random_state).fit(X[train_index], train_Y)

        context = wandb.init(project=config.wandb_project, entity=config.wandb_entity, group=group, name=name, config={**asdict(config), "n_genes": X.shape[1], "n_proteins": Y.shape[1], "n_train": len(train_index)}, mode=config.wandb_mode) if wandb is not None else nullcontext(None)
        predictions, score = None, None

        with context as run:
            metrics = {"n_components": model.n_components_, "explained_variance": float(model.pca.explained_variance_ratio_.sum()), "train/loss": masked_mse(torch.from_numpy(model.predict(X[train_index])), torch.from_numpy(train_Y)).item()}
            if val_index is not None:
                val_Y = (Y[val_index] - mean) / std
                outputs = model.predict(X[val_index])
                score = pooled_pearson(outputs, val_Y)
                # Undo the z-score so predictions are on the (log2) protein scale.
                predictions = outputs * std + mean
                metrics.update({"val/loss": masked_mse(torch.from_numpy(outputs), torch.from_numpy(val_Y)).item(), "val/pooled_pearson": score})
            if run is not None:
                run.log(metrics)
        return model, {"mean": mean, "std": std}, predictions, score

    scores, folds = [], []
    oof = np.full(Y.shape, np.nan, dtype=np.float32)
    for fold, (train_index, val_index) in enumerate(kfold_splits(len(X), config.cv_folds, config.random_state), start=1):
        _, _, predictions, score = fit(train_index, val_index, f"fold-{fold}")
        oof[val_index] = predictions
        scores.append(score)
        folds.append({"train_sample_ids": adata.obs_names[train_index].tolist(), "validation_sample_ids": adata.obs_names[val_index].tolist()})
    model, protein_scaling, _, _ = fit(np.arange(len(X)), None, "final-refit")
    oof_pearson = per_protein_pearson(oof, Y)
    return {"model": model, "protein_scaling": protein_scaling, "architecture": {"n_components": model.n_components_, "explained_variance": float(model.pca.explained_variance_ratio_.sum())}, "rna_gene_lengths": ("prep_species_lengths" if "log_tpm" in adata.layers else "bulkformer_human_lengths") if config.normalize_rna else None, "preprocessing": f"{'log_tpm' if config.normalize_rna else 'prenormalized_rna'}_then_train_fold_standard_scaler; {'log2_protein_then_' if config.log2_protein else 'protein_'}train_fold_zscore", "feature_names": adata.var_names.tolist(), "protein_features": adata.uns.get("protein_features"), "sample_ids": adata.obs_names.tolist(), "oof_predictions": oof.tolist(), "folds": folds, "metrics": {"cv_scores": scores, "cv_mean": float(np.nanmean(scores)), "oof_per_protein_pearson": oof_pearson.tolist(), "oof_mean_pearson": float(np.nanmean(oof_pearson))}, "config": asdict(config), "wandb_group": group}
