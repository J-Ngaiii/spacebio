# src/training/train/logreg_train.py

import numpy as np
import anndata as ad

from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.models.logreg import build_logreg
from src.training.train_conf.logreg_config import logreg_train_config


def train_logreg(adata: ad.AnnData, config: logreg_train_config | None = None) -> dict:
    config = logreg_train_config() if config is None else config

    labels = adata.obs[config.label_column]
    allowed = {config.positive_label, config.negative_label}

    if labels.isna().any():
        raise ValueError("Label column contains missing values.")

    unexpected = set(labels.unique()) - allowed
    if unexpected:
        raise ValueError(f"Unexpected arm labels: {unexpected}")

    y = labels.eq(config.positive_label).astype(int).to_numpy()
    class_counts = np.bincount(y, minlength=2)

    if config.cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if class_counts.min() < config.cv_folds:
        raise ValueError(f"Each class needs at least {config.cv_folds} samples; received {class_counts.tolist()}.")

    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    model = build_logreg(config)
    scores = cross_val_score(model, adata.X, y, cv=cv, scoring=config.scoring, error_score="raise")

    # Cross-validation fits clones. Fit this estimator separately for saving.
    model.fit(adata.X, y)

    return {
        "model": model,
        "metrics": {
            "scoring": config.scoring,
            "cv_scores": scores.tolist(),
            "cv_mean": float(scores.mean()),
            "n_samples": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "class_counts": class_counts.tolist(),
        },
        "feature_names": adata.var_names.tolist(),
        "sample_ids": adata.obs_names.tolist(),
        "positive_label": config.positive_label,
        "config": vars(config).copy(),
    }