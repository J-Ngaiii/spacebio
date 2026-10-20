
import argparse
from dataclasses import asdict
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
import wandb

from src.training.train_conf.performer_config import PerformerConfig
from src.training.train_conf.hyena_config import HyenaConfig
from src.training.train.performer_train import train_performer
from src.training.train.hyena_train import train_hyena


def load_data(
    path: Path,
    target_key: str = "protein",
) -> dict:
    """Load RNA features and protein targets from AnnData."""

    if not path.exists():
        raise FileNotFoundError(path)

    adata = ad.read_h5ad(path)

    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    if target_key not in adata.obsm:
        raise KeyError(f"Missing obsm[{target_key!r}]")

    Y = adata.obsm[target_key]
    if sparse.issparse(Y):
        Y = Y.toarray()
    Y = np.asarray(Y, dtype=np.float32)

    gene_names = adata.var_names.astype(str).tolist()

    if "protein_features" not in adata.uns:
        raise KeyError("Missing uns['protein_features']")

    protein_features = adata.uns["protein_features"]

    if isinstance(protein_features, dict):
        if len(protein_features) != 1:
            raise ValueError("Ambiguous protein_features dictionary")
        protein_names = list(next(iter(protein_features.values())))
    else:
        protein_names = list(protein_features)

    protein_names = [str(name) for name in protein_names]

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D")

    if X.shape[0] != Y.shape[0]:
        raise ValueError("RNA/protein sample mismatch")

    if len(gene_names) != X.shape[1]:
        raise ValueError("RNA feature count mismatch")

    if len(protein_names) != Y.shape[1]:
        raise ValueError("Protein feature count mismatch")

    if not np.isfinite(X).all():
        raise ValueError("Nonfinite RNA values")

    if not np.isfinite(Y).all():
        raise ValueError("Nonfinite protein values")

    print(f"Samples: {X.shape[0]}")
    print(f"RNA genes: {X.shape[1]}")
    print(f"Protein targets: {Y.shape[1]}")

    return {
        "X": X,
        "Y": Y,
        "sample_ids": adata.obs_names.astype(str).tolist(),
        "gene_names": gene_names,
        "protein_names": protein_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Performer and Hyena models on AnnData")

    parser.add_argument("--data", type=Path, default=Path("data/processed/sequence/harmonized/harmonized_training_gz.h5ad"))
    parser.add_argument("--model", choices=("performer", "hyena", "both"), default="both")
    parser.add_argument("--target-key", default="protein")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/sequence_models"))

    parser.add_argument("--wandb-project", default="spacebio-sequence-models")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="harmonized-training")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-log-model", action="store_true")

    args = parser.parse_args()

    data = load_data(args.data, args.target_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_dim = data["X"].shape[1]
    output_dim = data["Y"].shape[1]

    models = (
        ["performer", "hyena"]
        if args.model == "both"
        else [args.model]
    )

    for model_name in models:
        print(f"\n{'=' * 60}")
        print(f"Training {model_name.upper()}")
        print(f"{'=' * 60}")

        common = {
            "task": "regression",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
        }

        if model_name == "performer":
            config = PerformerConfig(**common)
            trainer = train_performer
        else:
            config = HyenaConfig(**common)
            trainer = train_hyena

        output_path = args.output_dir / f"{model_name}.pt"

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=f"{model_name}-{args.seed}",
            job_type="training",
            config={
                **asdict(config),
                "architecture": model_name,
                "dataset": args.data.name,
                "n_samples": data["X"].shape[0],
                "n_features": input_dim,
                "n_targets": output_dim,
            },
            mode=args.wandb_mode,
            reinit="finish_previous",
        )

        try:
            metrics = trainer(
                data=data,
                config=config,
                output_path=str(output_path),
                wandb_run=run,
                wandb_log_model=args.wandb_log_model,
            )

            print(f"\n{model_name.upper()} metrics:")

            for metric, value in metrics.items():
                print(f"  {metric}: {value}")

            print(f"Checkpoint: {output_path}")

        finally:
            run.finish()


if __name__ == "__main__":
    main()
