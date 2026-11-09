import argparse
from dataclasses import asdict
from pathlib import Path
import torch

import anndata as ad
import numpy as np
from scipy import sparse
import wandb

from src.training.train_conf.performer_config import PerformerConfig
from src.training.train_conf.hyena_config import HyenaConfig
from src.training.train.performer_train import train_performer
from src.training.train.hyena_train import train_hyena


def load_data(path: Path, target_key: str = "protein") -> dict:
    """Load RNA features and partially observed protein targets from AnnData."""
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

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("RNA/protein sample mismatch")

    # NEW: Resolve names before checking feature counts (previously undefined).
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

    if len(gene_names) != X.shape[1]:
        raise ValueError("RNA feature count mismatch")
    if len(protein_names) != Y.shape[1]:
        raise ValueError("Protein feature count mismatch")

    # EXISTING BEHAVIOR: Drop genes with any nonfinite expression; revisit if
    # missing RNA values are common, as this can remove many useful genes.
    keep_genes = np.isfinite(X).all(axis=0)
    print(f"Genes kept: {keep_genes.sum()}/{len(keep_genes)}")
    X = X[:, keep_genes]
    gene_names = np.asarray(gene_names)[keep_genes].tolist()
    if X.shape[1] == 0:
        raise ValueError("No finite RNA genes remain")

    # NEW: NaN protein entries are missing labels, not invalid samples.
    # Infinities remain invalid and must not enter the regression objective.
    if np.isinf(Y).any():
        raise ValueError("Protein targets contain infinity")
    observed = np.isfinite(Y)
    print(f"Observed protein entries: {observed.sum():,}/{Y.size:,}")
    print(f"Protein observation rate: {observed.mean():.2%}")
    print(f"Proteins with no observations: {(observed.sum(axis=0) == 0).sum()}")
    print(f"Samples with no observations: {(observed.sum(axis=1) == 0).sum()}")
    if not observed.any():
        raise ValueError("No observed protein targets")

    print(f"Samples: {X.shape[0]}")
    print(f"RNA genes: {X.shape[1]}")
    print(f"Protein targets: {Y.shape[1]}")
    return {
        "X": X,
        "Y": Y,
        "obs": adata.obs.copy(),
        "sample_ids": adata.obs_names.astype(str).tolist(),
        "gene_names": gene_names,
        "protein_names": protein_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Performer and Hyena models on AnnData")
    parser.add_argument("--data", type=Path, default=Path("data/processed/sequence/harmonized/harmonized_training_intersection.h5ad"))
    parser.add_argument("--model", choices=("performer", "hyena", "both"), default="performer")
    parser.add_argument("--return-predictions", action="store_true")
    parser.add_argument("--depth", type=int, default=1)
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

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Refusing to train on CPU."
        )

    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    data = load_data(args.data, args.target_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_dim = data["X"].shape[1]
    output_dim = data["Y"].shape[1]
    models = ["performer", "hyena"] if args.model == "both" else [args.model]

    for model_name in models:
        print(f"\n{'=' * 60}\nTraining {model_name.upper()}\n{'=' * 60}")
        common = {
            "task": "regression",
            "return_predictions": args.return_predictions,
            "depth": args.depth,
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
        elif model_name == "hyena":
            # RESTORED: Select Hyena trainer when requested.
            config = HyenaConfig(**common)
            trainer = train_hyena

        # debug: 
        print(config)
        # RESTORED: Define model checkpoint destination.
        output_path = args.output_dir

        # RESTORED: Initialize W&B run before training.
        print(f"Initializing W&B for {model_name}...", flush=True)

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
            # RESTORED: Actually execute the training function.
            print(f"Starting {model_name} training...", flush=True)

            metrics = trainer(
                data=data,
                config=config,
                output_path=str(output_path),
                wandb_run=run,
                wandb_log_model=args.wandb_log_model,
            )

            # RESTORED: Report final training metrics.
            print(f"\n{model_name.upper()} metrics:")

            for metric, value in metrics.items():
                print(f"  {metric}: {value}")

            print(f"Checkpoint: {output_path}")

        finally:
            # RESTORED: Close W&B run even if training fails.
            run.finish()

if __name__ == "__main__":
    main()