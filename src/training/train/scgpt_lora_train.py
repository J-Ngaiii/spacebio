"""scGPT fine-tuning (LoRA or frozen backbone): CV, final refit, optional W&B tracking.

Model code lives in src/models/scgpt_lora.py. Mirrors train_mlp: train_scgpt_lora(adata, config) -> result dict. Expects the AnnData
produced by prep_scgpt_lora. Written against scGPT's TransformerModel API from memory and
not run here: check the "loaded N/M pretrained tensors" line first, and the NOTE comments.
"""
from contextlib import nullcontext
from dataclasses import asdict
from uuid import uuid4

import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.models.scgpt_lora import build_scgpt_lora
from src.training.train_conf.scgpt_lora_config import scgpt_lora_train_config


def _config_warnings(config: scgpt_lora_train_config):
    if not config.checkpoint_dir:
        raise ValueError("checkpoint_dir is required.")
    if config.task not in {"classification", "protein"}:
        raise ValueError("task must be classification or protein.")
    if config.epochs < 1 or config.batch_size < 1 or config.head_hidden_dim < 1 or config.cv_folds < 2:
        raise ValueError("Positive epochs, batch size, hidden dimension and at least two CV folds are required.")
    if config.lr <= 0 or config.lora_lr <= 0 or config.weight_decay < 0 or not 0 <= config.head_dropout < 1:
        raise ValueError("Invalid learning rate, weight decay or dropout.")
    if config.wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline or disabled.")


# --------------------------------------------------
# Metrics
# --------------------------------------------------
def _masked_mse(pred, target, mask):
    return (((pred - target) ** 2) * mask).sum() / mask.sum().clamp(min=1)


def _mean_pearson(pred, y, mask):
    rs = []
    for j in range(y.shape[1]):
        m = mask[:, j]
        if m.sum() >= 3 and pred[m, j].std() > 0 and y[m, j].std() > 0:
            rs.append(np.corrcoef(pred[m, j], y[m, j])[0, 1])
    return float(np.mean(rs)) if rs else float("nan")


# --------------------------------------------------
# Training
# --------------------------------------------------
def train_scgpt_lora(adata: ad.AnnData, config: scgpt_lora_train_config | None = None) -> dict:
    if config is None:
        config = scgpt_lora_train_config()
    _config_warnings(config)

    if "binned" not in adata.layers or "scgpt_gene_id" not in adata.var:
        raise ValueError("adata must come from prep_scgpt_lora (needs layers['binned'] and var['scgpt_gene_id']).")

    ids = torch.tensor(adata.var["scgpt_gene_id"].to_numpy(), dtype=torch.long).unsqueeze(0)
    vals = torch.tensor(adata.layers["binned"], dtype=torch.float32)
    n = len(vals)
    classification = config.task == "classification"

    if classification:
        labels = adata.obs[config.label_column]
        if labels.isna().any() or not labels.isin([config.positive_label, config.negative_label]).all():
            raise ValueError("Labels must be nonmissing and match the configured positive/negative labels.")
        y = labels.eq(config.positive_label).to_numpy(dtype=np.float32).reshape(-1, 1)
        if np.bincount(y.ravel().astype(int), minlength=2).min() < config.cv_folds:
            raise ValueError("Each class must have at least cv_folds samples.")
        mask = np.ones_like(y, dtype=bool)
        cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        splits = list(cv.split(vals, y.ravel().astype(int)))
        score_key = "val/auroc"
    else:
        if "protein_z" not in adata.obsm:
            raise ValueError("task='protein' needs obsm['protein_z'] from prep_scgpt_lora.")
        y = np.nan_to_num(np.asarray(adata.obsm["protein_z"], dtype=np.float32))
        mask = np.asarray(adata.obsm["protein_mask"], dtype=bool)
        groups = adata.obs[config.group_column].to_numpy()
        n_splits = min(config.cv_folds, len(np.unique(groups)))
        splits = list(GroupKFold(n_splits=n_splits).split(vals, groups=groups))
        score_key = "val/pearson"

    yt = torch.from_numpy(y)
    mt = torch.from_numpy(mask.astype(np.float32))
    device = torch.device(config.device)
    group = config.run_group or f"scgpt-{config.task}-{uuid4().hex[:12]}"
    wandb = None
    if config.wandb_mode != "disabled":
        import wandb

    def forward(model, idx):
        return model(ids.expand(len(idx), -1).to(device), vals[idx].to(device))

    def loss_fn(out, idx):
        target, m = yt[idx].to(device), mt[idx].to(device)
        return F.binary_cross_entropy_with_logits(out, target) if classification else _masked_mse(out, target, m)

    def evaluate(model, idx):
        model.eval()
        with torch.no_grad():
            parts = [forward(model, idx[i:i + config.batch_size]) for i in range(0, len(idx), config.batch_size)]
            out = torch.cat(parts)
            loss = loss_fn(out, idx).item()
        if classification:
            preds = out.sigmoid().cpu().numpy().ravel()
            return preds, {"val/loss": loss, "val/auroc": float(roc_auc_score(y[idx].ravel(), preds))}
        preds = out.cpu().numpy()
        return preds, {"val/loss": loss, "val/pearson": _mean_pearson(preds, y[idx], mask[idx])}

    def fit(train_index, val_index, name, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        model = build_scgpt_lora(config, y.shape[1]).to(device)
        model.eval()
        with torch.no_grad():
            chunks = np.array_split(train_index, max(1, len(train_index) // config.batch_size))
            emb = torch.cat([model.embed(ids.expand(len(b), -1).to(device), vals[b].to(device)) for b in chunks])
        model.set_embedding_stats(emb.mean(0), emb.std(0).clamp(min=1e-3))
        groups_ = [{"params": model.head.parameters(), "lr": config.lr}]
        adapters = [p for p in model.backbone.parameters() if p.requires_grad]
        if adapters:
            groups_.append({"params": adapters, "lr": config.lora_lr})
        optimizer = torch.optim.AdamW(groups_, weight_decay=config.weight_decay)

        context = wandb.init(project=config.wandb_project, entity=config.wandb_entity, group=group, name=name, config={**asdict(config), "n_genes": vals.shape[1], "n_train": len(train_index)}, mode=config.wandb_mode) if wandb is not None else nullcontext(None)
        preds, metrics = None, {}
        with context as run:
            for epoch in range(config.epochs):
                model.train()
                total = 0.0
                order = rng.permutation(train_index)
                for i in range(0, len(order), config.batch_size):
                    batch = order[i:i + config.batch_size]
                    loss = loss_fn(forward(model, batch), batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(batch)
                metrics = {"epoch": epoch + 1, "train/loss": total / len(train_index)}
                if val_index is not None:
                    preds, val_metrics = evaluate(model, val_index)
                    metrics.update(val_metrics)
                if run is not None:
                    run.log(metrics)
            if run is not None and val_index is not None:
                run.summary[f"final_{score_key}"] = metrics[score_key]
        return model, preds, metrics

    scores, folds = [], []
    oof = np.full(n if classification else y.shape, np.nan, dtype=np.float32)
    for fold, (train_index, val_index) in enumerate(splits, start=1):
        _, preds, metrics = fit(train_index, val_index, f"fold-{fold}", config.random_state + fold)
        oof[val_index] = preds
        scores.append(float(metrics[score_key]))
        folds.append({"train_sample_ids": adata.obs_names[train_index].tolist(), "validation_sample_ids": adata.obs_names[val_index].tolist()})

    # Fixed epoch count: CV measures the training procedure; refit uses all samples.
    model, _, _ = fit(np.arange(n), None, "final-refit", config.random_state)

    if classification:
        summary = {"oof_auroc": float(roc_auc_score(y.ravel(), oof))}
    else:
        summary = {
            "oof_mse": float(((oof - y) ** 2)[mask].mean()),
            "baseline_mse": float((y ** 2)[mask].mean()),  # predicting the study mean (0 in z-space)
        }
    # Save only what was trained (head + adapters); the backbone reloads from the checkpoint.
    trainable = {name for name, p in model.named_parameters() if p.requires_grad} | {"emb_mean", "emb_std"}
    state = {k: v.cpu() for k, v in model.state_dict().items() if k in trainable}
    return {
        "state_dict": state,
        "architecture": {"n_genes": int(vals.shape[1]), "hidden_dim": config.head_hidden_dim, "output_dim": int(y.shape[1]), "dropout": config.head_dropout, "use_lora": config.use_lora, "lora_r": config.lora_r},
        "preprocessing": "scgpt_binned_expression",
        "feature_names": adata.var["scgpt_symbol"].tolist(),
        "sample_ids": adata.obs_names.tolist(),
        "targets": y.tolist(),
        "oof_predictions": oof.tolist(),
        "folds": folds,
        "metrics": {"cv_scores": scores, "cv_mean": float(np.nanmean(scores)), **summary},
        "config": asdict(config),
        "wandb_group": group,
    }