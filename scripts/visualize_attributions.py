"""Visualize top genes, then top samples, from a complete IG matrix."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Integrated Gradients .npz file")
    parser.add_argument("--output-dir", type=Path, default=Path("data/visuals"))
    parser.add_argument("--space-or-earth", type=str, default="earth")
    parser.add_argument("--top-genes", type=int, default=5)
    parser.add_argument("--top-samples", type=int, default=20)
    parser.add_argument("--sample-ranking", choices=("mean_abs", "max_abs"), default="mean_abs")
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    args = parser.parse_args()

    if args.top_genes < 1 or args.top_samples < 1:
        parser.error("--top-genes and --top-samples must be positive")

    with np.load(args.input, allow_pickle=False) as data:
        A = np.asarray(data["attributions"], dtype=np.float64)
        genes = np.asarray(data["gene_names"]).astype(str)
        samples = np.asarray(data["sample_ids"]).astype(str)
        protein = (str(data["protein_name"].item()) if "protein_name" in data
                   else args.input.stem.removesuffix("_ig"))

    if A.ndim != 2 or A.shape != (len(samples), len(genes)):
        raise ValueError("Attribution dimensions do not match sample/gene identifiers")
    if not A.size or not np.isfinite(A).all():
        raise ValueError("Attribution matrix is empty or contains nonfinite values")

    # Rank genes across ALL available samples before selecting any samples.
    gene_scores = np.mean(np.abs(A), axis=0)
    n_genes = min(args.top_genes, A.shape[1])
    gene_indices = np.argsort(-gene_scores, kind="stable")[:n_genes]
    A_genes = A[:, gene_indices]

    # Rank samples using only the selected genes.
    if args.sample_ranking == "mean_abs":
        sample_scores = np.mean(np.abs(A_genes), axis=1)
    else:
        sample_scores = np.max(np.abs(A_genes), axis=1)
    n_samples = min(args.top_samples, A.shape[0])
    sample_indices = np.argsort(-sample_scores, kind="stable")[:n_samples]

    selected = A_genes[sample_indices, :]
    selected_genes = genes[gene_indices]
    selected_samples = samples[sample_indices]
    limit = float(np.max(np.abs(selected)))
    if limit == 0:
        limit = 1.0

    fig_width = max(11, n_genes * 0.26)
    fig_height = max(5, min(28, n_samples * 0.18 + 2.5))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(selected, cmap="RdBu_r", aspect="auto",
                   interpolation="nearest", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(n_genes), labels=selected_genes, rotation=90, fontsize=8)
    if n_samples <= 40:
        ax.set_yticks(np.arange(n_samples), labels=selected_samples, fontsize=7)
    else:
        ax.set_yticks([])
    ax.set_xlabel("Genes (ranked by mean absolute IG over all samples)")
    ax.set_ylabel(f"Samples (ranked by {args.sample_ranking} IG over selected genes)")
    ax.set_title(f"{protein} protein prediction — Integrated Gradients "
                 f"({n_samples} of {A.shape[0]} samples; {n_genes} genes)")
    fig.colorbar(im, ax=ax, label="Signed IG attribution", fraction=0.025, pad=0.02)
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.input.parent.name
    safe_protein = "".join(c if c.isalnum() or c in "-_." else "_" for c in protein)
    prefix = (f"{run_name}_{safe_protein}_top{n_genes}genes_"
              f"top{n_samples}samples_{args.sample_ranking}_{args.space_or_earth}")
    output = args.output_dir / f"{prefix}_ig_heatmap.{args.format}"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({"gene": selected_genes,
                  "mean_abs_ig_all_samples": gene_scores[gene_indices],
                  "original_gene_index": gene_indices}).to_csv(
                      args.output_dir / f"{prefix}_gene_ranking.csv", index=False)
    pd.DataFrame({"sample_id": selected_samples,
                  "sample_score": sample_scores[sample_indices],
                  "original_sample_index": sample_indices}).to_csv(
                      args.output_dir / f"{prefix}_sample_ranking.csv", index=False)
    np.savez_compressed(args.output_dir / f"{prefix}_selected_ig.npz",
                        attributions=selected, gene_names=selected_genes,
                        sample_ids=selected_samples, protein_name=protein,
                        original_gene_indices=gene_indices,
                        original_sample_indices=sample_indices)
    print(f"Full IG matrix: {A.shape}; selected heatmap: {selected.shape}")
    print(f"Saved heatmap: {output}")


if __name__ == "__main__":
    main()
