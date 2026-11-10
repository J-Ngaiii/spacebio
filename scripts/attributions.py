
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from scipy import sparse

from src.models.performer import Performer


def load_checkpoint(path, device):
    """Restore the trained Performer and its preprocessing metadata."""
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = checkpoint["config"]

    model = Performer(
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        input_channels=checkpoint["input_channels"],
        d_model=cfg["d_model"],
        depth=cfg["depth"],
        heads=cfg["heads"],
        dropout=cfg["dropout"],
        metadata_cardinalities=checkpoint["metadata_cardinalities"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def prepare_data(adata, checkpoint, indices):
    """
    Reconstruct the gene ordering, RNA scaling, and metadata
    encoding used during training.
    """
    gene_names = list(checkpoint["gene_names"])
    protein_names = list(checkpoint["protein_names"])

    # Match RNA genes to checkpoint ordering.
    gene_index = adata.var_names.get_indexer(gene_names)

    if np.any(gene_index < 0):
        missing = np.asarray(gene_names)[gene_index < 0]
        raise ValueError(
            f"Missing checkpoint genes: {missing[:10]}"
        )

    X = adata.X[indices, :]
    X = X[:, gene_index]

    if sparse.issparse(X):
        X = X.toarray()

    X = np.asarray(X, dtype=np.float32)

    if not np.isfinite(X).all():
        raise ValueError("Nonfinite RNA inputs")

    # Apply the original training-fitted scaler.
    x_scaler = checkpoint["x_scaler"]

    if x_scaler is not None:
        X = x_scaler.transform(X).astype(np.float32)

    obs = adata.obs.iloc[indices]
    metadata = {}

    for name in checkpoint["metadata_columns"]:
        vocabulary = checkpoint["metadata_vocabularies"][name]

        if name in obs.columns:
            values = (
                obs[name]
                .astype("string")
                .fillna("<MISSING>")
            )
        else:
            values = pd.Series(
                ["<MISSING>"] * len(obs),
                index=obs.index,
                dtype="string",
            )

        metadata[name] = torch.tensor(
            values.map(vocabulary)
            .fillna(0)
            .to_numpy(dtype=np.int64),
            dtype=torch.long,
        )

    return X, metadata, gene_names, protein_names


def compute_attributions(
    model,
    X,
    metadata,
    protein_idx,
    device,
    batch_size,
    n_steps,
    internal_batch_size,
):
    """
    Compute Integrated Gradients for one predicted protein.

    The baseline is zero in standardized RNA coordinates,
    corresponding to the training-set mean expression.

    Returns:
        attributions: (num_samples, num_genes)
        deltas:       (num_samples,)
    """

    metadata_keys = tuple(metadata.keys())

    # Captum expands tensor arguments across interpolation steps.
    # Pass metadata as individual tensors rather than a dictionary.
    def forward_protein(x, *metadata_tensors):
        metadata_batch = dict(
            zip(metadata_keys, metadata_tensors)
        )

        return model(x, metadata_batch)[:, protein_idx]

    ig = IntegratedGradients(forward_protein)

    all_attributions = []
    all_deltas = []

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))

        xb = torch.from_numpy(
            X[start:end]
        ).to(device)

        metadata_args = tuple(
            metadata[name][start:end].to(device)
            for name in metadata_keys
        )

        attributions, delta = ig.attribute(
            inputs=xb,
            baselines=torch.zeros_like(xb),
            additional_forward_args=metadata_args,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=True,
        )

        all_attributions.append(
            attributions.detach().cpu().numpy()
        )

        all_deltas.append(
            delta.detach().cpu().numpy()
        )

        print(
            f"Processed {end}/{len(X)} samples",
            flush=True,
        )

    return (
        np.concatenate(all_attributions, axis=0),
        np.concatenate(all_deltas, axis=0),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute Integrated Gradients for Performer protein predictions"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--protein",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--internal-batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    if (
        args.num_samples < 1
        or args.batch_size < 1
        or args.n_steps < 1
    ):
        raise ValueError(
            "num-samples, batch-size, and n-steps must be positive"
        )

    if args.internal_batch_size < args.batch_size:
        raise ValueError(
            "internal-batch-size must be >= batch-size"
        )

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    if not args.data.exists():
        raise FileNotFoundError(args.data)

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    # --------------------------------------------------
    # 1. Restore trained model
    # --------------------------------------------------

    model, checkpoint = load_checkpoint(
        args.checkpoint,
        device,
    )

    # --------------------------------------------------
    # 2. Load original AnnData
    # --------------------------------------------------

    adata = ad.read_h5ad(args.data)

    protein_names = list(checkpoint["protein_names"])

    if args.protein not in protein_names:
        raise ValueError(
            f"Protein {args.protein!r} not found. "
            f"Example proteins: {protein_names[:10]}"
        )

    protein_idx = protein_names.index(args.protein)

    # Ensure the output was supervised during training.
    supported = checkpoint.get("train_supported_proteins")

    if supported is not None and not supported[protein_idx]:
        raise ValueError(
            f"Protein {args.protein!r} had no training labels"
        )

    # --------------------------------------------------
    # 3. Reconstruct original held-out test split
    # --------------------------------------------------

    test_indices = np.asarray(
        checkpoint["split_indices"]["test"]
    )

    if adata.n_obs != len(checkpoint["sample_ids"]):
        raise ValueError(
            "Dataset sample count mismatch"
        )

    if (
        adata.obs_names.astype(str).tolist()
        != list(checkpoint["sample_ids"])
    ):
        raise ValueError(
            "Dataset sample ordering mismatch"
        )

    # Validate protein ordering against AnnData.
    protein_features = adata.uns["protein_features"]

    if isinstance(protein_features, dict):
        if len(protein_features) != 1:
            raise ValueError(
                "Ambiguous protein_features dictionary"
            )
        data_protein_names = list(
            next(iter(protein_features.values()))
        )
    else:
        data_protein_names = list(protein_features)

    data_protein_names = [
        str(name) for name in data_protein_names
    ]

    if data_protein_names != protein_names:
        raise ValueError(
            "Protein ordering differs from checkpoint"
        )

    # --------------------------------------------------
    # 4. Select samples with observed protein labels
    # --------------------------------------------------

    Y = adata.obsm["protein"]

    if sparse.issparse(Y):
        Y = Y.toarray()

    Y = np.asarray(Y)

    if Y.shape[1] != len(protein_names):
        raise ValueError(
            "Protein target count mismatch"
        )

    observed_indices = test_indices[
        np.isfinite(Y[test_indices, protein_idx])
    ]

    if len(observed_indices) == 0:
        raise ValueError(
            f"No observed test labels for {args.protein}"
        )

    rng = np.random.default_rng(args.seed)

    n = min(
        args.num_samples,
        len(observed_indices),
    )

    selected_indices = np.sort(
        rng.choice(
            observed_indices,
            size=n,
            replace=False,
        )
    )

    # --------------------------------------------------
    # 5. Reconstruct standardized model inputs
    # --------------------------------------------------

    X, metadata, gene_names, _ = prepare_data(
        adata,
        checkpoint,
        selected_indices,
    )

    print(f"Protein: {args.protein}")
    print(f"Protein index: {protein_idx}")
    print(f"Samples: {len(X)}")
    print(f"Genes: {X.shape[1]}")
    print(f"Device: {device}")
    print(f"Integration steps: {args.n_steps}")

    # --------------------------------------------------
    # 6. Compute Integrated Gradients
    # --------------------------------------------------

    attributions, deltas = compute_attributions(
        model=model,
        X=X,
        metadata=metadata,
        protein_idx=protein_idx,
        device=device,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        internal_batch_size=args.internal_batch_size,
    )

    # --------------------------------------------------
    # 7. Aggregate gene-level importance
    # --------------------------------------------------

    mean_abs_ig = np.abs(attributions).mean(axis=0)
    mean_signed_ig = attributions.mean(axis=0)

    ranking = pd.DataFrame({
        "gene": gene_names,
        "mean_abs_ig": mean_abs_ig,
        "mean_signed_ig": mean_signed_ig,
    }).sort_values(
        "mean_abs_ig",
        ascending=False,
    )

    # --------------------------------------------------
    # 8. Save outputs
    # --------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_protein = "".join(
        c if c.isalnum() or c in "-_." else "_"
        for c in args.protein
    )

    npz_path = (
        args.output_dir / f"{safe_protein}_ig.npz"
    )

    csv_path = (
        args.output_dir / f"{safe_protein}_gene_ranking.csv"
    )

    np.savez_compressed(
        npz_path,
        attributions=attributions,
        convergence_delta=deltas,
        gene_names=np.asarray(gene_names),
        sample_ids=np.asarray(
            checkpoint["sample_ids"]
        )[selected_indices],
        sample_indices=selected_indices,
        protein_name=args.protein,
        protein_idx=protein_idx,
        baseline="zero_standardized_expression",
    )

    ranking.to_csv(
        csv_path,
        index=False,
    )

    print("\nTop 20 genes:")
    print(
        ranking.head(20).to_string(index=False)
    )

    print(
        "\nMean absolute convergence delta:",
        np.abs(deltas).mean(),
    )

    print(f"\nSaved: {npz_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
