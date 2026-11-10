"""Offline GO BP / KEGG overrepresentation analysis from local GMT gene-set files.

GMT format: pathway<TAB>description<TAB>gene1<TAB>gene2...
Download gene sets for the correct organism and gene ID type beforehand.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


def read_gmt(path):
    sets = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            if len(fields) >= 3:
                sets[fields[0]] = set(fields[2:])
    if not sets:
        raise ValueError(f'No gene sets in {path}')
    return sets


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    q = np.empty(len(p))
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1].clip(0, 1)
    return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clusters-dir', type=Path, required=True)
    p.add_argument('--embeddings', type=Path, required=True)
    p.add_argument('--go-gmt', type=Path, required=True, help='GO Biological Process GMT matching gene identifiers')
    p.add_argument('--kegg-gmt', type=Path, required=True, help='KEGG GMT matching gene identifiers')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--min-set-size', type=int, default=5)
    p.add_argument('--max-set-size', type=int, default=500)
    args = p.parse_args()
    if args.min_set_size < 1 or args.max_set_size < args.min_set_size:
        p.error('Invalid gene-set size range')
    genes = set(np.load(args.embeddings, allow_pickle=False)['gene_names'].astype(str))
    if len(genes) == 0:
        raise ValueError('Empty gene universe')
    libraries = {'GO_BP': read_gmt(args.go_gmt), 'KEGG': read_gmt(args.kegg_gmt)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for file in sorted(args.clusters_dir.glob('*__clusters.csv')):
        assignments = pd.read_csv(file, dtype={'gene': str})
        if assignments['gene'].duplicated().any() or set(assignments['gene']) != genes:
            raise ValueError(f'Cluster gene universe differs from embedding universe: {file}')
        for cluster, group in assignments.groupby('cluster'):
            selected = set(group['gene'])
            rows = []
            for database, pathways in libraries.items():
                for term, members in pathways.items():
                    eligible = members & genes
                    if not args.min_set_size <= len(eligible) <= args.max_set_size:
                        continue
                    overlap = selected & eligible
                    # Test all eligible pathways, including those with zero overlap,
                    # so BH correction includes the full tested family.
                    pval = hypergeom.sf(len(overlap)-1, len(genes), len(eligible), len(selected))
                    rows.append({'clustering': file.stem, 'cluster': cluster, 'database': database,
                                 'pathway': term, 'cluster_size': len(selected),
                                 'pathway_size_in_background': len(eligible),
                                 'overlap_count': len(overlap), 'p_value': pval,
                                 'overlap_genes': ';'.join(sorted(overlap))})
            if rows:
                df = pd.DataFrame(rows)
                # BH per cluster, across both GO BP and KEGG hypotheses.
                df['q_value'] = bh_adjust(df['p_value'].to_numpy())
                all_results.append(df)
        print(f'Enriched {file.name}', flush=True)
    if not all_results:
        raise ValueError('No cluster assignments or eligible gene sets found')
    result = pd.concat(all_results, ignore_index=True)
    result.sort_values(['clustering', 'cluster', 'q_value', 'p_value'], inplace=True)
    result.to_csv(args.output_dir / 'all_enrichment.csv', index=False)
    result[(result.q_value < 0.05) & (result.overlap_count > 0)].to_csv(
        args.output_dir / 'significant_enrichment.csv', index=False)
    (args.output_dir / 'enrichment_manifest.json').write_text(json.dumps({
        'background': 'All model input genes (from embedding NPZ)',
        'go_gmt': str(args.go_gmt), 'kegg_gmt': str(args.kegg_gmt),
        'test': 'One-sided hypergeometric overrepresentation',
        'multiple_testing': 'Benjamini-Hochberg within each clustering and cluster across both libraries',
        'min_set_size': args.min_set_size, 'max_set_size': args.max_set_size,
        'gene_id_requirement': 'GMT identifiers must match checkpoint gene names exactly'}, indent=2))
    print(f'Saved {args.output_dir / "all_enrichment.csv"}')


if __name__ == '__main__':
    main()
