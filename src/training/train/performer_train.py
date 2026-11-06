import copy
import random
from dataclasses import asdict
from pathlib import Path

import time
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.performer import Performer
from src.training.train_conf.performer_config import PerformerConfig


def encode_metadata(obs, columns, fit_idx):
    """Encode categorical metadata as integer ids.

    Vocabularies are built from the rows in `fit_idx` only (the train split).
    Index 0 is reserved for missing or unseen categories.
    """
    metadata = {}
    vocabularies = {}

    for column in columns:
        if column in obs.columns:
            values = obs[column].astype("string").fillna("<MISSING>")
        else:
            values = pd.Series(
                ["<MISSING>"] * len(obs),
                index=obs.index,
                dtype="string",
            )

        fit_values = values.iloc[fit_idx]

        categories = sorted(
            value for value in fit_values.unique()
            if value != "<MISSING>"
        )

        vocabulary = {
            value: i + 1
            for i, value in enumerate(categories)
        }

        metadata[column] = (
            values.map(vocabulary)
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )

        vocabularies[column] = vocabulary

    return metadata, vocabularies


# NEW: Masked regression counts only observed protein measurements.
def masked_squared_error_sum(prediction, target):
    mask = torch.isfinite(target)
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    error_sum = (prediction - safe_target).square().masked_select(mask).sum()
    return error_sum, int(mask.sum().item())


def masked_mse(prediction, target):
    error_sum, count = masked_squared_error_sum(prediction, target)
    if count == 0:
        raise ValueError("Batch has no observed protein targets")
    return error_sum / count


def train_performer(
    data: dict,
    config: PerformerConfig | None = None,
    output_path: str = "results/performer.pt",
    wandb_run=None,
    wandb_log_model: bool = False,
):
    cfg = config or PerformerConfig()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if cfg.task not in {"regression", "classification"}:
        raise ValueError(f"Unsupported task: {cfg.task}")

    X = np.asarray(data["X"], dtype=np.float32)
    Y = np.asarray(data["Y"])

    if X.ndim not in (2, 3):
        raise ValueError("X must be 2D or 3D")
    if len(X) != len(Y):
        raise ValueError("X and Y sample counts differ")
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite X")

    classes = None

    metadata_columns = [
        "source",
        "lineage",
        "tissue",
        "cancer_type",
    ]

    if cfg.task == "regression":
        Y = np.asarray(Y, dtype=np.float32)

        if Y.ndim == 1:
            Y = Y[:, None]

        # NEW: NaNs indicate unobserved proteins; infinities are invalid.
        if Y.ndim != 2 or np.isinf(Y).any():
            raise ValueError("Regression Y must be 2D without infinities")

        output_dim = Y.shape[1]
        criterion = masked_mse  # NEW: observed targets only

    else:
        if Y.ndim != 1:
            raise ValueError("Classification requires 1D labels")

        classes, Y = np.unique(Y, return_inverse=True)
        Y = Y.astype(np.int64)
        output_dim = len(classes)

        if output_dim < 2:
            raise ValueError("At least two classes required")

        criterion = nn.CrossEntropyLoss()

    length = X.shape[1]
    channels = 1 if X.ndim == 2 else X.shape[2]

    if cfg.input_dim is not None and cfg.input_dim != length:
        raise ValueError("Configured input_dim mismatch")
    if cfg.output_dim is not None and cfg.output_dim != output_dim:
        raise ValueError("Configured output_dim mismatch")
    if cfg.input_channels != channels:
        raise ValueError("Configured input_channels mismatch")

    if (
        not 0 < cfg.val_fraction < 1
        or not 0 < cfg.test_fraction < 1
        or cfg.val_fraction + cfg.test_fraction >= 1
    ):
        raise ValueError("Invalid split fractions")

    indices = np.arange(len(X))

    train_val, test_idx = train_test_split(
        indices,
        test_size=cfg.test_fraction,
        random_state=cfg.seed,
    )

    train_idx, val_idx = train_test_split(
        train_val,
        test_size=cfg.val_fraction / (1 - cfg.test_fraction),
        random_state=cfg.seed,
    )

    # Metadata vocabularies are fit on the train split only.
    metadata, metadata_vocabularies = encode_metadata(
        data["obs"],
        metadata_columns,
        train_idx,
    )

    metadata_cardinalities = {
        name: len(vocabulary) + 1
        for name, vocabulary in metadata_vocabularies.items()
    }

    x_scaler = None

    if cfg.standardize_x:
        x_scaler = StandardScaler()
        flat = X.reshape(len(X), -1)
        x_scaler.fit(flat[train_idx])
        X = x_scaler.transform(flat).reshape(X.shape).astype(np.float32)

    # NEW: Per-protein target scaling using observed TRAIN values only.
    # Proteins with no training observations are excluded from loss/scoring
    # by replacing their targets with NaN throughout the split.
    y_scaler = None
    y_mean = None
    y_std = None
    if cfg.task == "regression":
        train_counts = np.isfinite(Y[train_idx]).sum(axis=0)
        train_supported = train_counts > 0
        if not train_supported.any():
            raise ValueError("No observed protein targets in training split")
        if (~train_supported).any():
            print(f"Warning: {(~train_supported).sum()} proteins have no training labels; excluded from scoring")
            Y[:, ~train_supported] = np.nan
        if cfg.standardize_y:
            y_mean = np.zeros(output_dim, dtype=np.float32)
            y_std = np.ones(output_dim, dtype=np.float32)
            for j in np.flatnonzero(train_supported):
                values = Y[train_idx, j]
                values = values[np.isfinite(values)]
                y_mean[j] = values.mean()
                y_std[j] = max(float(values.std()), 1e-6)
            Y = ((Y - y_mean) / y_std).astype(np.float32)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else cfg.device
    )

    model = Performer(
        input_dim=length,
        output_dim=output_dim,
        input_channels=channels,
        d_model=cfg.d_model,
        depth=cfg.depth,
        heads=cfg.heads,
        dropout=cfg.dropout,
        metadata_cardinalities=metadata_cardinalities,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    if wandb_run is not None:
        wandb_run.watch(model, log="gradients", log_freq=100)
        wandb_run.summary["num_parameters"] = sum(
            p.numel() for p in model.parameters()
        )

    def make_loader(idx, shuffle=False):
        y_dtype = (
            torch.long
            if cfg.task == "classification"
            else torch.float32
        )

        tensors = [
            torch.tensor(X[idx], dtype=torch.float32),
            torch.tensor(Y[idx], dtype=y_dtype),
        ]

        for name in metadata_columns:
            tensors.append(
                torch.tensor(
                    metadata[name][idx],
                    dtype=torch.long,
                )
            )

        dataset = TensorDataset(*tensors)

        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
        )

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx)
    test_loader = make_loader(test_idx)

    def unpack(batch):
        xb, yb, *metadata_batch = batch

        mb = {
            name: tensor.to(device)
            for name, tensor in zip(metadata_columns, metadata_batch)
        }

        return xb.to(device), yb.to(device), mb

    def evaluate(loader, return_predictions=False):
        model.eval()
        total_error = 0.0
        observed_count = 0
        predictions = []

        with torch.no_grad():
            for batch in loader:
                xb, yb, mb = unpack(batch)
                prediction = model(xb, mb)
                if cfg.task == "regression":
                    # NEW: Aggregate over observed entries, not batch sizes.
                    error_sum, n_obs = masked_squared_error_sum(prediction, yb)
                    total_error += error_sum.item()
                    observed_count += n_obs
                else:
                    loss = criterion(prediction, yb)
                    total_error += loss.item() * len(xb)
                    observed_count += len(xb)
                if return_predictions:
                    predictions.append(prediction.cpu().numpy())

        if observed_count == 0:
            raise ValueError("Evaluation split has no observed targets")
        mean_loss = total_error / observed_count
        if return_predictions:
            return mean_loss, np.concatenate(predictions, axis=0)
        return mean_loss

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    history = []

    training_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()

        model.train()
        total_squared_error = 0.0
        total_observed = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1:03d}/{cfg.epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )

        for batch in progress:
            xb, yb, mb = unpack(batch)

            optimizer.zero_grad(set_to_none=True)

            prediction = model(xb, mb)

            # NEW: Only observed protein targets contribute to MSE.
            mask = torch.isfinite(yb)
            n_observed = int(mask.sum().item())

            # Skip batches with no observed targets.
            if n_observed == 0:
                continue

            safe_target = torch.where(
                mask, yb, torch.zeros_like(yb)
            )

            squared_error = (
                (prediction - safe_target).square()
                .masked_select(mask)
                .sum()
            )

            loss = squared_error / n_observed

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # NEW: Accumulate loss by observed protein entries,
            # rather than averaging batch losses equally.
            total_squared_error += squared_error.detach().item()
            total_observed += n_observed

            running_loss = total_squared_error / total_observed

            progress.set_postfix(
                train_loss=f"{running_loss:.5f}"
            )

        if total_observed == 0:
            raise RuntimeError(
                "No observed protein targets in training epoch"
            )

        train_loss = total_squared_error / total_observed

        # NEW: Run validation after every epoch.
        # Assumes evaluate() has been updated for masked MSE.
        val_loss = evaluate(val_loader)

        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_minutes = (
            time.perf_counter() - training_start
        ) / 60

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_seconds": epoch_seconds,
        })

        # NEW: Persistent epoch summary for Slurm logs.
        tqdm.write(
            f"Epoch {epoch + 1:03d}/{cfg.epochs} | "
            f"Train MSE: {train_loss:.5f} | "
            f"Val MSE: {val_loss:.5f} | "
            f"Time: {epoch_seconds:.1f}s | "
            f"Elapsed: {elapsed_minutes:.1f}min"
        )

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/learning_rate":
                    optimizer.param_groups[0]["lr"],
                "train/epoch_seconds": epoch_seconds,
            })

        # EXISTING: Early stopping based on validation loss.
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1

            if wait >= cfg.patience:
                tqdm.write("Early stopping")
                break

    if best_state is None:
        raise RuntimeError("No successful training epoch")

    model.load_state_dict(best_state)
    test_loss, predictions = evaluate(test_loader, return_predictions=True)

    if cfg.task == "regression":
        actual = Y[test_idx]

        # NEW: Inverse transform without destroying NaN observation masks.
        if y_mean is not None:
            predictions = predictions * y_std + y_mean
            actual = actual * y_std + y_mean

        observed = np.isfinite(actual)
        if not observed.any():
            raise ValueError("Test split has no observed protein targets")
        errors = predictions[observed] - actual[observed]
        metrics = {
            "test_mse": float(np.mean(errors ** 2)),
            "test_mae": float(np.mean(np.abs(errors))),
            "test_observed_entries": int(observed.sum()),
            "metadata_columns": metadata_columns,
            "metadata_vocabularies": metadata_vocabularies,
            "metadata_cardinalities": metadata_cardinalities,
        }

    else:
        metrics = {
            "test_accuracy": float(
                np.mean(
                    predictions.argmax(axis=1) == Y[test_idx]
                )
            )
        }

    metrics["test_loss"] = test_loss
    metrics["best_epoch"] = best_epoch

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": best_state,
        "config": asdict(cfg),
        "input_dim": length,
        "output_dim": output_dim,
        "input_channels": channels,
        "classes": classes.tolist() if classes is not None else None,
        "gene_names": data.get("gene_names"),
        "protein_names": data.get("protein_names"),
        "sample_ids": data.get("sample_ids"),
        "x_scaler": x_scaler,
        # NEW: Store train-fitted per-protein scaling parameters.
        "y_scaler": y_scaler,  # Retained for backwards-compatible key only (None).
        "y_mean": y_mean,
        "y_std": y_std,
        "train_supported_proteins": train_supported if cfg.task == "regression" else None,
        "metadata_columns": metadata_columns,
        "metadata_vocabularies": metadata_vocabularies,
        "metadata_cardinalities": metadata_cardinalities,
        "split_indices": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        },
        "history": history,
        "metrics": metrics,
    }, path)

    if wandb_run is not None:
        wandb_run.log({
            "test/loss": test_loss,
            "best/val_loss": best_loss,
            "best/epoch": best_epoch,
            **{
                key.replace("test_", "test/", 1): value
                for key, value in metrics.items()
                if key.startswith("test_")
            },
        })

        wandb_run.summary["best_val_loss"] = best_loss
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["checkpoint_path"] = str(path)

        if wandb_log_model:
            artifact = wandb.Artifact(
                name=f"{wandb_run.id}-checkpoint",
                type="model",
                metadata={
                    "architecture": "performer",
                    "input_dim": length,
                    "output_dim": output_dim,
                },
            )

            artifact.add_file(str(path))
            wandb_run.log_artifact(artifact)

    return metrics