import copy
import random
from dataclasses import asdict
from pathlib import Path

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

        if Y.ndim != 2 or not np.isfinite(Y).all():
            raise ValueError("Regression Y must be finite and 2D")

        output_dim = Y.shape[1]
        criterion = nn.MSELoss()

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

    y_scaler = None

    if cfg.task == "regression" and cfg.standardize_y:
        y_scaler = StandardScaler()
        y_scaler.fit(Y[train_idx])
        Y = y_scaler.transform(Y).astype(np.float32)

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
        total = 0.0
        count = 0
        predictions = []

        with torch.no_grad():
            for batch in loader:
                xb, yb, mb = unpack(batch)

                prediction = model(xb, mb)
                loss = criterion(prediction, yb)

                total += loss.item() * len(xb)
                count += len(xb)

                if return_predictions:
                    predictions.append(prediction.cpu().numpy())

        mean_loss = total / count

        if return_predictions:
            return mean_loss, np.concatenate(predictions, axis=0)

        return mean_loss

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        count = 0

        for batch in train_loader:
            xb, yb, mb = unpack(batch)

            optimizer.zero_grad(set_to_none=True)

            prediction = model(xb, mb)
            loss = criterion(prediction, yb)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += loss.item() * len(xb)
            count += len(xb)

        train_loss = total / count
        val_loss = evaluate(val_loader)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train {train_loss:.5f} | "
            f"Val {val_loss:.5f}"
        )

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            })

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1

            if wait >= cfg.patience:
                print("Early stopping")
                break

    if best_state is None:
        raise RuntimeError("No successful training epoch")

    model.load_state_dict(best_state)
    test_loss, predictions = evaluate(test_loader, return_predictions=True)

    if cfg.task == "regression":
        actual = Y[test_idx]

        if y_scaler is not None:
            predictions = y_scaler.inverse_transform(predictions)
            actual = y_scaler.inverse_transform(actual)

        metrics = {
            "test_mse": float(np.mean((predictions - actual) ** 2)),
            "test_mae": float(np.mean(np.abs(predictions - actual))),
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
        "y_scaler": y_scaler,
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