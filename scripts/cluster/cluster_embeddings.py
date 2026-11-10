"""Cluster raw and expression-residualized gene embeddings across Performer layers."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embeddings', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--k', type=int, nargs='+', default=[10, 20, 50, 100])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pca-components', type=int, default=30)
    args = p.parse_args()
    data = np.load(args.embeddings, allow_pickle=False)
    genes = data['gene_names'].astype(str)
    covariate = data['expression_covariate'].astype(np.float64)
    if not np.isfinite(covariate).all():
        raise ValueError('Nonfinite expression covariate')
    layers = ['static_gene_id', 'input'] + sorted(
        (name for name in data.files if name.startswith('layer_')),
        key=lambda name: int(name.split('_')[1]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    design = np.column_stack([np.ones(len(genes)), covariate])
    for layer in layers:
        embeddings = data[layer].astype(np.float64)
        if not np.isfinite(embeddings).all():
            raise ValueError(f'Nonfinite embeddings: {layer}')
        # OLS on every coordinate simultaneously; fit intercept and log expression.
        coefficients = np.linalg.lstsq(design, embeddings, rcond=None)[0]
        residuals = embeddings - design @ coefficients
        for condition, values in [('original', embeddings), ('expression_adjusted', residuals)]:
            # Scale embedding dimensions, then PCA to denoise before K-means.
            standardized = StandardScaler().fit_transform(values)
            n_components = min(args.pca_components, standardized.shape[0] - 1, standardized.shape[1])
            if n_components < 2:
                raise ValueError('Insufficient genes/embedding dimensions')
            pca = PCA(n_components=n_components, random_state=args.seed)
            reduced = pca.fit_transform(standardized)
            pd.DataFrame({'gene': genes, 'PC1': reduced[:, 0], 'PC2': reduced[:, 1]}).to_csv(
                args.output_dir / f'{layer}__{condition}__pca.csv', index=False)
            for k in args.k:
                if k < 2 or k >= len(genes):
                    raise ValueError(f'K must be between 2 and {len(genes)-1}: {k}')
                labels = KMeans(n_clusters=k, random_state=args.seed, n_init=10).fit_predict(reduced)
                pd.DataFrame({'gene': genes, 'cluster': labels}).to_csv(
                    args.output_dir / f'{layer}__{condition}__k{k}__clusters.csv', index=False)
                # Approximate silhouette on a fixed subsample for speed.
                sil = silhouette_score(reduced, labels, sample_size=min(2000, len(genes)), random_state=args.seed)
                metrics.append({'layer': layer, 'condition': condition, 'k': k,
                                'silhouette': sil, 'pca_variance_explained': float(pca.explained_variance_ratio_.sum())})
                print(f'{layer} {condition} K={k}: silhouette={sil:.3f}', flush=True)
    pd.DataFrame(metrics).to_csv(args.output_dir / 'clustering_metrics.csv', index=False)
    (args.output_dir / 'clustering_manifest.json').write_text(json.dumps({
        'embedding_file': str(args.embeddings), 'layers': layers, 'k': args.k,
        'expression_adjustment': 'Coordinate-wise OLS residuals against expression_covariate with intercept',
        'clustering': 'StandardScaler -> PCA -> KMeans; same procedure for original and adjusted embeddings',
        'background_genes': len(genes)}, indent=2))


if __name__ == '__main__':
    main()
