import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Integrated Gradients .npz file')
    parser.add_argument('--output-dir', type=Path, default=Path('data/visuals'))
    parser.add_argument('--top-genes', type=int, default=50)
    parser.add_argument('--format', choices=['png', 'pdf'], default='png')
    args = parser.parse_args()

    if args.top_genes < 1:
        parser.error('--top-genes must be positive')
    with np.load(args.input, allow_pickle=False) as data:
        A = np.asarray(data['attributions'], dtype=np.float64)
        genes = np.asarray(data['gene_names']).astype(str)
        samples = np.asarray(data['sample_ids']).astype(str)
        protein = str(data['protein_name'].item()) if 'protein_name' in data else args.input.stem.replace('_ig', '')

    if A.ndim != 2 or A.shape != (len(samples), len(genes)):
        raise ValueError('Attribution dimensions do not match sample/gene identifiers')
    if not np.isfinite(A).all():
        raise ValueError('Attribution matrix contains NaN or infinity')
    if A.shape[0] == 0 or A.shape[1] == 0:
        raise ValueError('Attribution matrix is empty')

    n = min(args.top_genes, A.shape[1])
    top = np.argsort(-np.mean(np.abs(A), axis=0), kind='stable')[:n]
    selected = A[:, top]
    selected_genes = genes[top]
    limit = np.max(np.abs(selected))
    if limit == 0:
        limit = 1.0

    fig_width = max(11, n * 0.26)
    fig_height = max(5, min(28, A.shape[0] * 0.18 + 2.5))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(selected, cmap='RdBu_r', aspect='auto', interpolation='nearest', vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(n), labels=selected_genes, rotation=90, fontsize=8)
    if len(samples) <= 40:
        ax.set_yticks(np.arange(len(samples)), labels=samples, fontsize=7)
    else:
        ax.set_yticks([])
    ax.set_xlabel('Genes (ranked by mean absolute IG)')
    ax.set_ylabel('Held-out test samples')
    ax.set_title(f'{protein} protein prediction — Integrated Gradients ({len(samples)} samples)')
    fig.colorbar(im, ax=ax, label='Signed IG attribution', fraction=0.025, pad=0.02)
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.input.parent.name
    output = args.output_dir / f'{run_name}_{protein}_top{n}_ig_heatmap.{args.format}'
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved heatmap: {output}')


if __name__ == '__main__':
    main()