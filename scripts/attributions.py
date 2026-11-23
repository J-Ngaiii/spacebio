
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from scipy import sparse

from src.models.performer import Performer
from src.models.hyena import Hyena


def load_checkpoint(path, device, architecture="auto"):
    """Restore a Performer or Hyena using its saved configuration."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    if architecture == "auto":
        architecture = cfg.get("architecture") or cfg.get("model")
        if architecture is None:
            folder = path.parent.name.lower()
            architecture = next(
                (name for name in ("performer", "hyena") if folder.startswith(name)),
                None,
            )
        if architecture is None:
            raise ValueError(
                "Cannot infer architecture. Supply --model performer or --model hyena."
            )

    shared = dict(
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        input_channels=checkpoint["input_channels"],
        d_model=cfg["d_model"],
        depth=cfg["depth"],
        dropout=cfg["dropout"],
        metadata_cardinalities=checkpoint.get("metadata_cardinalities") or {},
    )
    if architecture == "performer":
        model = Performer(**shared, heads=cfg["heads"])
    elif architecture == "hyena":
        model = Hyena(
            **shared,
            order=cfg["order"],
            kernel_size=cfg["kernel_size"],
        )
    else:
        raise ValueError(f"Unsupported model: {architecture}")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint, architecture


def prepare_data(adata, checkpoint, indices):
    """
    Reconstruct the gene ordering, RNA scaling, and metadata
    encoding used during training.
    """
    gene_names = list(checkpoint["gene_names"])
    protein_names = list(checkpoint["protein_names"])

    # Keep the checkpoint feature space; unmeasured genes are zero in
    # standardized coordinates (the training-set mean).
    if not adata.var_names.is_unique:
        raise ValueError("Duplicate RNA gene names in dataset")
    gene_index = adata.var_names.get_indexer(gene_names)
    present = gene_index >= 0
    if not present.any():
        raise ValueError("No checkpoint RNA genes found in dataset")
    print(f"RNA genes present: {present.sum()}/{len(gene_names)}; "
          f"missing (baseline-imputed): {(~present).sum()}", flush=True)

    X = np.zeros((len(indices), len(gene_names)), dtype=np.float32)
    X_present = adata.X[indices, :][:, gene_index[present]]
    if sparse.issparse(X_present):
        X_present = X_present.toarray()
    X_present = np.asarray(X_present, dtype=np.float32)
    if not np.isfinite(X_present).all():
        raise ValueError("Nonfinite observed RNA inputs")

    x_scaler = checkpoint.get("x_scaler")
    if x_scaler is not None:
        X_present = (X_present - np.asarray(x_scaler.mean_)[present]) / np.asarray(x_scaler.scale_)[present]
    X[:, present] = X_present.astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite standardized RNA inputs")

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

    return X, metadata, gene_names, protein_names, present


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
        description="Compute Integrated Gradients for Performer or Hyena protein predictions"
    )

    parser.add_argument("--model", choices=("auto", "performer", "hyena"), default="auto")
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
        "--external-data", action="store_true",
        help="Evaluate independent dataset rather than checkpoint test split",
    )
    parser.add_argument(
        "--all-samples", action="store_true",
        help="Use all held-out samples with observed labels for this protein",
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
        or args.internal_batch_size < 1
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

    model, checkpoint, architecture = load_checkpoint(
        args.checkpoint, device, args.model
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

    if args.external_data:
        candidate_indices = np.arange(adata.n_obs)
        print(f"External dataset: {adata.n_obs} samples", flush=True)
    else:
        if adata.n_obs != len(checkpoint["sample_ids"]):
            raise ValueError("Dataset sample count mismatch; use --external-data for OOD data")
        if adata.obs_names.astype(str).tolist() != list(checkpoint["sample_ids"]):
            raise ValueError("Dataset sample ordering mismatch; use --external-data for OOD data")
        candidate_indices = np.asarray(checkpoint["split_indices"]["test"])

    if args.external_data:
        # IG does not require measured protein targets on external samples.
        eligible_indices = candidate_indices
        print("External IG: no protein-label requirement", flush=True)
    else:
        protein_features = adata.uns["protein_features"]
        if isinstance(protein_features, dict):
            if len(protein_features) != 1:
                raise ValueError("Ambiguous protein_features dictionary")
            protein_features = next(iter(protein_features.values()))
        data_protein_names = [str(name) for name in protein_features]
        if data_protein_names != protein_names:
            raise ValueError("Protein ordering differs from checkpoint")
        Y = adata.obsm["protein"]
        if Y.shape[1] != len(protein_names):
            raise ValueError("Protein target count mismatch")
        if sparse.issparse(Y):
            target = np.asarray(Y[candidate_indices, protein_idx].toarray()).ravel()
        else:
            target = np.asarray(Y[candidate_indices, protein_idx]).ravel()
        eligible_indices = candidate_indices[np.isfinite(target)]

    if len(eligible_indices) == 0:
        raise ValueError("No eligible samples")
    if args.all_samples:
        selected_indices = np.sort(eligible_indices)
    else:
        rng = np.random.default_rng(args.seed)
        n = min(args.num_samples, len(eligible_indices))
        selected_indices = np.sort(rng.choice(eligible_indices, size=n, replace=False))
    print(f"Selected {len(selected_indices)} of {len(eligible_indices)} eligible samples")

    X, metadata, gene_names, _, present = prepare_data(
        adata, checkpoint, selected_indices
    )

    print(f"Model: {architecture}")
    print(f"Depth: {checkpoint['config']['depth']}")
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
    })[present].sort_values(
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
        gene_present_mask=present,
        sample_ids=np.asarray(
            adata.obs_names[selected_indices].tolist(),
            dtype=str,
        ),
        sample_indices=selected_indices,
        protein_name=args.protein,
        architecture=architecture,
        checkpoint_path=str(args.checkpoint),
        protein_idx=protein_idx,
        baseline="zero_standardized_expression",
        selection_mode=("all_eligible_external" if args.all_samples else "random_eligible_external")
            if args.external_data else ("all_eligible_test" if args.all_samples else "random_eligible_test"),
        dataset_path=str(args.data),
        external_data=args.external_data,
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
