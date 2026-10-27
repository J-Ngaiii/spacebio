"""Fine-tune BulkFormer to predict protein abundance from bulk RNA, with optional W&B tracking."""
from contextlib import nullcontext
from dataclasses import asdict
from uuid import uuid4
import anndata as ad

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from src.models.bulkformer import BulkFormerRegressor, load_pretrained_bulkformer
from src.training.train_conf.bulkformer_config import bulkformer_train_config

def _config_warnings(config: bulkformer_train_config):
    if config.epochs < 1 or config.batch_size < 1 or config.embed_dim < 1 or config.cv_folds < 2:
        raise ValueError("Positive epochs, batch size, embedding dimension and at least two CV folds are required.")
    if config.lr <= 0 or config.weight_decay < 0 or not 0 <= config.dropout < 1:
        raise ValueError("Invalid learning rate, weight decay or dropout.")
    if config.wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline or disabled.")

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

def train_bulkformer(adata: ad.AnnData, config: bulkformer_train_config | None = None) -> dict:
    if config is None:
        config = bulkformer_train_config()

    _config_warnings(config)

    if config.protein_key not in adata.obsm:
        raise ValueError(f"adata.obsm[{config.protein_key!r}] is missing; prepare the data with include_protein=True.")

    X = np.asarray(adata.X, dtype=np.float32)
    if not np.isfinite(X).all() or (X < 0).any():
        raise ValueError("RNA counts must be finite and nonnegative.")
    if len(X) < config.cv_folds:
        raise ValueError("At least cv_folds samples are required.")

    # TODO: match the input scale BulkFormer was pretrained on (check its repo) instead of log1p counts.
    X = np.log1p(X)
    Y = np.asarray(adata.obsm[config.protein_key], dtype=np.float32)
    device = torch.device(config.device)
    group = config.run_group or f"bulkformer-{uuid4().hex[:12]}"
    wandb = None
    if config.wandb_mode != "disabled":
        import wandb

    def fit(train_index, val_index, name, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Z-score each protein with training-fold statistics only, so validation values never leak into training.
        mean = np.nan_to_num(np.nanmean(Y[train_index], axis=0))
        std = np.nanstd(Y[train_index], axis=0)
        std[~(std > 0)] = 1.0
        train_X = torch.from_numpy(X[train_index])
        train_Y = torch.from_numpy(((Y[train_index] - mean) / std).astype(np.float32))

        loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
        backbone = load_pretrained_bulkformer(config.checkpoint_path, device=config.device)
        model = BulkFormerRegressor(backbone, config.embed_dim, Y.shape[1], dropout=config.dropout, freeze_backbone=config.freeze_backbone).to(device)

        # Only parameters that still require gradients are trained; with a frozen backbone that is just the head.
        optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=config.lr, weight_decay=config.weight_decay)

        val_X = torch.from_numpy(X[val_index]).to(device) if val_index is not None else None
        val_Y = torch.from_numpy(((Y[val_index] - mean) / std).astype(np.float32)).to(device) if val_index is not None else None
        context = wandb.init(project=config.wandb_project, entity=config.wandb_entity, group=group, name=name, config={**asdict(config), "n_genes": X.shape[1], "n_proteins": Y.shape[1], "n_train": len(train_index)}, mode=config.wandb_mode) if wandb is not None else nullcontext(None)
        predictions = None

        with context as run:
            for epoch in range(config.epochs):
                model.train()
                total = 0.0
                for features, targets in loader:
                    optimizer.zero_grad()
                    loss = masked_mse(model(features.to(device)), targets.to(device))
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(targets)
                metrics = {"epoch": epoch + 1, "train/loss": total / len(train_index)}
                if val_X is not None:
                    model.eval()
                    with torch.no_grad():
                        outputs = model(val_X)
                        # Undo the z-score so predictions are on the original protein scale.
                        predictions = outputs.cpu().numpy() * std + mean
                        metrics.update({"val/loss": masked_mse(outputs, val_Y).item(), "val/mean_pearson": float(np.nanmean(per_protein_pearson(predictions, Y[val_index])))})
                if run is not None:
                    run.log(metrics)
            if run is not None and val_index is not None:
                run.summary["final_val_mean_pearson"] = metrics["val/mean_pearson"]
        return model.cpu(), {"mean": mean, "std": std}, predictions

    scores, folds = [], []
    oof = np.full(Y.shape, np.nan, dtype=np.float32)
    # TODO: replace random KFold with the team's shared held-out split (e.g. by study, tissue or plex).
    cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    for fold, (train_index, val_index) in enumerate(cv.split(X), start=1):
        _, _, predictions = fit(train_index, val_index, f"fold-{fold}", config.random_state + fold)
        oof[val_index] = predictions
        scores.append(float(np.nanmean(per_protein_pearson(predictions, Y[val_index]))))
        folds.append({"train_sample_ids": adata.obs_names[train_index].tolist(), "validation_sample_ids": adata.obs_names[val_index].tolist()})
    # Fixed epoch count: CV measures the training procedure; refit uses all samples.
    model, protein_scaling, _ = fit(np.arange(len(X)), None, "final-refit", config.random_state)
    oof_pearson = per_protein_pearson(oof, Y)
    # A frozen backbone is unchanged from the checkpoint, so save only the head.
    state_dict = model.head.state_dict() if config.freeze_backbone else model.state_dict()
    return {"state_dict": state_dict, "saved_weights": "head" if config.freeze_backbone else "full", "protein_scaling": protein_scaling, "architecture": {"checkpoint_path": config.checkpoint_path, "embed_dim": config.embed_dim, "output_dim": Y.shape[1], "dropout": config.dropout}, "preprocessing": "log1p_rna_then_train_fold_protein_zscore", "feature_names": adata.var_names.tolist(), "protein_features": adata.uns.get("protein_features"), "sample_ids": adata.obs_names.tolist(), "oof_predictions": oof.tolist(), "folds": folds, "metrics": {"cv_scores": scores, "cv_mean": float(np.mean(scores)), "oof_per_protein_pearson": oof_pearson.tolist(), "oof_mean_pearson": float(np.nanmean(oof_pearson))}, "config": asdict(config), "wandb_group": group}
