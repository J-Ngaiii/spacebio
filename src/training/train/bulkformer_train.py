"""Fine-tune BulkFormer to predict protein abundance from bulk RNA, with optional W&B tracking."""
import copy
from contextlib import nullcontext
from dataclasses import asdict
from uuid import uuid4
import anndata as ad

import numpy as np
import pandas as pd
import torch
from bulkformer import extract_features
from bulkformer.preprocessing import align_to_vocab
from torch.utils.data import DataLoader, TensorDataset

from src.features.build_features import rna_log_tpm
from src.evaluation.evaluate import kfold_splits, load_protein_targets, masked_mse, per_protein_pearson, pooled_pearson, zscore_stats
from src.models.bulkformer import BulkFormerRegressor, load_pretrained_bulkformer, protein_head
from src.training.train_conf.bulkformer_config import bulkformer_train_config

def _config_warnings(config: bulkformer_train_config):
    if config.epochs < 1 or config.batch_size < 1 or config.cv_folds < 2:
        raise ValueError("Positive epochs and batch size and at least two CV folds are required.")
    if config.lr <= 0 or config.weight_decay < 0 or not 0 <= config.dropout < 1:
        raise ValueError("Invalid learning rate, weight decay or dropout.")
    if config.wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline or disabled.")

def predict(model: torch.nn.Module, inputs: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """Run the model in batches so large validation sets fit in GPU memory."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            outputs.append(model(inputs[start:start + batch_size].to(device)).cpu())
    return torch.cat(outputs)

def train_bulkformer(adata: ad.AnnData, config: bulkformer_train_config | None = None) -> dict:
    if config is None:
        config = bulkformer_train_config()

    _config_warnings(config)

    X = np.asarray(adata.X, dtype=np.float32)
    if not np.isfinite(X).all() or (X < 0).any():
        raise ValueError("RNA counts must be finite and nonnegative.")
    if len(X) < config.cv_folds:
        raise ValueError("At least cv_folds samples are required.")

    Y = load_protein_targets(adata, config.protein_key, log2=config.log2_protein)

    backbone = load_pretrained_bulkformer(config.checkpoint)
    # BulkFormer expects log TPM in its own gene order; genes it knows but the data lacks are filled with its mask value.
    log_tpm = pd.DataFrame(rna_log_tpm(adata), index=adata.obs_names, columns=adata.var_names)
    aligned, mask_prob = align_to_vocab(log_tpm, backbone.config.gene_ids, mask_token_value=backbone.config.mask_token_value)
    expr = aligned.to_numpy(dtype=np.float32)
    embed_dim = backbone.config.dim + 3

    device = torch.device(config.device)
    if config.freeze_backbone:
        # A frozen backbone gives the same embedding every time, so compute it once and train only the head on it.
        inputs = extract_features(backbone, expr, "sample_level", aggregate_type="mean", device=device, batch_size=config.batch_size, mask_prob=mask_prob, show_progress=False)
        backbone.cpu()
        def make_model():
            return protein_head(embed_dim, Y.shape[1], dropout=config.dropout)
    else:
        inputs = torch.from_numpy(expr)
        # Each fold starts from a fresh copy of the pretrained weights.
        def make_model():
            return BulkFormerRegressor(copy.deepcopy(backbone), Y.shape[1], dropout=config.dropout, mask_prob=mask_prob)

    group = config.run_group or f"bulkformer-{uuid4().hex[:12]}"
    wandb = None
    if config.wandb_mode != "disabled":
        import wandb

    def fit(train_index, val_index, name, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

        mean, std = zscore_stats(Y[train_index])
        train_Y = torch.from_numpy(((Y[train_index] - mean) / std).astype(np.float32))

        loader = DataLoader(TensorDataset(inputs[train_index], train_Y), batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
        model = make_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        val_Y = ((Y[val_index] - mean) / std).astype(np.float32) if val_index is not None else None
        context = wandb.init(project=config.wandb_project, entity=config.wandb_entity, group=group, name=name, config={**asdict(config), "n_genes": X.shape[1], "vocab_coverage": 1 - mask_prob, "n_proteins": Y.shape[1], "n_train": len(train_index)}, mode=config.wandb_mode) if wandb is not None else nullcontext(None)
        predictions, score = None, None

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
                if val_index is not None:
                    outputs = predict(model, inputs[val_index], config.batch_size, device)
                    score = pooled_pearson(outputs.numpy(), val_Y)
                    # Undo the z-score so predictions are on the (log2) protein scale.
                    predictions = outputs.numpy() * std + mean
                    metrics.update({"val/loss": masked_mse(outputs, torch.from_numpy(val_Y)).item(), "val/pooled_pearson": score})
                if run is not None:
                    run.log(metrics)
            if run is not None and val_index is not None:
                run.summary["final_val_pooled_pearson"] = score
        return model.cpu(), {"mean": mean, "std": std}, predictions, score

    scores, folds = [], []
    oof = np.full(Y.shape, np.nan, dtype=np.float32)
    for fold, (train_index, val_index) in enumerate(kfold_splits(len(X), config.cv_folds, config.random_state), start=1):
        _, _, predictions, score = fit(train_index, val_index, f"fold-{fold}", config.random_state + fold)
        oof[val_index] = predictions
        scores.append(score)
        folds.append({"train_sample_ids": adata.obs_names[train_index].tolist(), "validation_sample_ids": adata.obs_names[val_index].tolist()})
    # Fixed epoch count: CV measures the training procedure; refit uses all samples.
    model, protein_scaling, _, _ = fit(np.arange(len(X)), None, "final-refit", config.random_state)
    oof_pearson = per_protein_pearson(oof, Y)
    # With a frozen backbone the model is just the head; the backbone is unchanged from the checkpoint.
    return {"state_dict": model.state_dict(), "saved_weights": "head" if config.freeze_backbone else "full", "protein_scaling": protein_scaling, "architecture": {"checkpoint": config.checkpoint, "embed_dim": embed_dim, "output_dim": Y.shape[1], "dropout": config.dropout}, "mask_prob": mask_prob, "rna_gene_lengths": "prep_species_lengths" if "log_tpm" in adata.layers else "bulkformer_human_lengths", "preprocessing": "log_tpm_aligned_to_vocab; log2_protein_then_train_fold_zscore" if config.log2_protein else "log_tpm_aligned_to_vocab; train_fold_protein_zscore", "feature_names": adata.var_names.tolist(), "protein_features": adata.uns.get("protein_features"), "sample_ids": adata.obs_names.tolist(), "oof_predictions": oof.tolist(), "folds": folds, "metrics": {"cv_scores": scores, "cv_mean": float(np.nanmean(scores)), "vocab_coverage": 1 - mask_prob, "oof_per_protein_pearson": oof_pearson.tolist(), "oof_mean_pearson": float(np.nanmean(oof_pearson))}, "config": asdict(config), "wandb_group": group}
