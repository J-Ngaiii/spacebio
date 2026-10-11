"""RNA-only cross-validation and final refit, with optional W&B tracking."""
from contextlib import nullcontext
from dataclasses import asdict
from uuid import uuid4
import anndata as ad

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.simple_mlp import MLP
from src.training.train_conf.mlp_config import mlp_train_config

def _config_warnings(config: mlp_train_config):
    if config.epochs < 1 or config.batch_size < 1 or config.hidden_dim < 1 or config.cv_folds < 2:
        raise ValueError("Positive epochs, batch size, hidden dimension and at least two CV folds are required.")
    if config.lr <= 0 or config.weight_decay < 0 or not 0 <= config.dropout < 1:
        raise ValueError("Invalid learning rate, weight decay or dropout.")
    if config.wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline or disabled.")

def train_mlp(adata: ad.AnnData, config: mlp_train_config | None = None) -> dict:
    if config is None:
        config = mlp_train_config()

    _config_warnings(config)
    
    labels = adata.obs[config.label_column]
    if labels.isna().any() or not labels.isin([config.positive_label, config.negative_label]).all():
        raise ValueError("Labels must be nonmissing and match the configured positive/negative labels.")
    
    y = labels.eq(config.positive_label).to_numpy(dtype=np.float32)
    if np.bincount(y.astype(int), minlength=2).min() < config.cv_folds:
        raise ValueError("Each class must have at least cv_folds samples.")
    
    X = np.asarray(adata.X, dtype=np.float32)
    if not np.isfinite(X).all() or (X < 0).any():
        raise ValueError("RNA counts must be finite and nonnegative.")
    
    X = np.log1p(X)
    device = torch.device(config.device)
    group = config.run_group or f"mlp-{uuid4().hex[:12]}"
    wandb = None
    if config.wandb_mode != "disabled":
        import wandb

    def fit(train_index, val_index, name, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        scaler = StandardScaler().fit(X[train_index])
        train_X = torch.from_numpy(scaler.transform(X[train_index]).astype(np.float32))
        train_y = torch.from_numpy(y[train_index]).reshape(-1, 1)
        
        loader = DataLoader(TensorDataset(train_X, train_y), batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
        model = MLP(X.shape[1], config.hidden_dim, 1, dropout=config.dropout).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        val_X = torch.from_numpy(scaler.transform(X[val_index]).astype(np.float32)).to(device) if val_index is not None else None
        val_y = torch.from_numpy(y[val_index]).reshape(-1, 1).to(device) if val_index is not None else None
        context = wandb.init(project=config.wandb_project, entity=config.wandb_entity, group=group, name=name, config={**asdict(config), "n_genes": X.shape[1], "n_train": len(train_index)}, mode=config.wandb_mode) if wandb is not None else nullcontext(None)
        probabilities = None
        
        with context as run:
            for epoch in range(config.epochs):
                model.train()
                total = 0.0
                for features, targets in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(features.to(device)), targets.to(device))
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(targets)
                metrics = {"epoch": epoch + 1, "train/loss": total / len(train_index)}
                if val_X is not None:
                    model.eval()
                    with torch.no_grad():
                        logits = model(val_X)
                        probabilities = logits.sigmoid().cpu().numpy().ravel()
                        metrics.update({"val/loss": criterion(logits, val_y).item(), "val/auroc": roc_auc_score(y[val_index], probabilities)})
                if run is not None:
                    run.log(metrics)
            if run is not None and val_index is not None:
                run.summary["final_val_auroc"] = metrics["val/auroc"]
        return model.cpu(), scaler, probabilities

    scores, folds = [], []
    oof = np.empty(len(y), dtype=np.float32)
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    for fold, (train_index, val_index) in enumerate(cv.split(X, y), start=1):
        _, _, probabilities = fit(train_index, val_index, f"fold-{fold}", config.random_state + fold)
        oof[val_index] = probabilities
        scores.append(float(roc_auc_score(y[val_index], probabilities)))
        folds.append({"train_sample_ids": adata.obs_names[train_index].tolist(), "validation_sample_ids": adata.obs_names[val_index].tolist()})
    # Fixed epoch count: CV measures the training procedure; refit uses all samples.
    model, scaler, _ = fit(np.arange(len(y)), None, "final-refit", config.random_state)
    return {"state_dict": model.state_dict(), "scaler": scaler, "architecture": {"input_dim": X.shape[1], "hidden_dim": config.hidden_dim, "output_dim": 1, "dropout": config.dropout}, "preprocessing": "log1p_then_standard_scaler", "feature_names": adata.var_names.tolist(), "sample_ids": adata.obs_names.tolist(), "labels": y.tolist(), "oof_probabilities": oof.tolist(), "folds": folds, "metrics": {"cv_scores": scores, "cv_mean": float(np.mean(scores)), "oof_auroc": float(roc_auc_score(y, oof))}, "config": asdict(config), "wandb_group": group}
