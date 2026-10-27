"""Map mouse OSDR RNA matrix to human gene symbols.

Input:
merged mouse OSDR h5ad
Output:
out/osdr_paired_human_mapped.h5ad
out/ortholog_coverage_report.txt (per-study mapping coverage)
out/mouse2human_mapping_used.csv

Mapping:
1. Only keep ortholog_one2only rows
2. compare with the available table
3. duplicate human symbols (multiple mouse genes -> same symbol) collapsed by mean
"""
import argparse
import os
import pathlib
import sys
import urllib.request

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

REFERENCE_URL = ('https://raw.githubusercontent.com/alwalt/bridge-rna/'
              'sample_embeddings/data/ensembl/orthologs_one2one.txt')


def fetch_reference_table(dest):
    if not os.path.exists(dest):
        print(f'downloading reference ortholog table -> {dest}')
        urllib.request.urlretrieve(REFERENCE_URL, dest)
    return pd.read_csv(dest, sep='\t')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5ad', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--all-samples', action='store_true',
                    help='keep all 433 samples (default: paired RNA+protein only)')
    ap.add_argument('--reference-table', default='../raw/orthologs_one2one.txt')
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out) if args.out else \
        pathlib.Path(__file__).resolve().parent / 'out'
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = pathlib.Path(__file__).resolve().parent / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    reference_path = pathlib.Path(args.reference_table)
    if not reference_path.exists():
        reference_path = raw_dir / 'orthologs_one2one.txt'

    A = ad.read_h5ad(args.h5ad)
    print(f'loaded {A.shape[0]} samples x {A.shape[1]} genes')

    for col in ('mgi_symbol', 'human_ortholog', 'human_homology_type'):
        if col not in A.var.columns:
            sys.exit(f'var column {col!r} missing. Rebuild the merged matrix '
                     'with ortholog annotation first')

    var = A.var.copy()
    o2o = var.human_homology_type == 'ortholog_one2one'
    kept = var[o2o & var.human_ortholog.notna()].copy()
    print(f'one2one mapping rows: {len(kept)} / {A.n_vars} genes '
          f'({100 * len(kept) / A.n_vars:.0f}%)')

    # compare to the reference table
    reference = fetch_reference_table(reference_path)
    reference_pairs = set(zip(reference['Gene name'], reference['Human gene name']))
    ours = set(zip(kept.mgi_symbol, kept.human_ortholog))
    report = ['# mouse -> human ortholog mapping report',
              f'input: {args.h5ad}',
              f'genes total: {A.n_vars}',
              f'one2one mapped: {len(kept)} ({100 * len(kept) / A.n_vars:.1f}%)',
              f'one2many dropped: {(var.human_homology_type == "ortholog_one2many").sum()}',
              f'many2many dropped: {(var.human_homology_type == "ortholog_many2many").sum()}',
              f'unmapped (no ortholog): {var.human_homology_type.isna().sum()}',
              f'agreement with reference table: {len(ours & reference_pairs)} of '
              f'{len(reference_pairs)} reference pairs reproduced; '
              f'{len(ours - reference_pairs)} pairs newer release']

    # collapse duplicate human symbols
    dup_mask = kept.human_ortholog.duplicated(keep=False)
    dup_symbols = sorted(kept.human_ortholog[dup_mask].unique())
    report.append(f'duplicate human symbols collapsed by mean: {len(dup_symbols)} '
                  f'({", ".join(dup_symbols[:10])}{"..." if len(dup_symbols) > 10 else ""})')

    X = A[:, kept.index].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    M = pd.DataFrame(X, index=A.obs_names,
                     columns=kept.human_ortholog.astype(str).values)
    n_before = M.shape[1]
    M = M.T.groupby(level=0).mean().T
    report.append(f'gene columns: {n_before} -> {M.shape[1]} after collapse')

    # sample subset
    if args.all_samples:
        sub = A.obs_names
        label = 'all samples'
    else:
        sub = A.obs_names[A.obs.has_protein == True]
        label = 'paired RNA+protein samples only'
    M = M.loc[sub]
    report.append(f'samples kept: {M.shape[0]} ({label})')

    # per-study coverage
    report.append('')
    report.append('per-study mapping coverage (fraction of expressed genes '
                  'with a one2one human ortholog):')
    expr = (M.values > 0) if not sp.issparse(X) else None
    for osd, idx in A.obs.groupby('osd_id').groups.items():
        idx = [s for s in idx if s in M.index]
        if not idx:
            continue
        rows = M.loc[idx].values
        cov = (rows > 0).mean()
        report.append(f'  {osd}: {len(idx):3d} samples, {100 * cov:.0f}% of '
                      'expressed gene entries mapped')

    # Write outputs
    import shutil
    import tempfile
    out_h5ad = out_dir / ('osdr_paired_human_mapped.h5ad' if not args.all_samples
                          else 'osdr_all_human_mapped.h5ad')
    obs = A.obs.loc[sub].copy()
    for c in obs.columns:
        dt = obs[c].dtype
        numeric = pd.api.types.is_numeric_dtype(dt) or dt == bool
        if not numeric:
            obs[c] = obs[c].astype(object).fillna('').astype(str).astype(object)
    obs.index = pd.Index([str(s) for s in obs.index],
                         dtype=object, name=obs.index.name)
    B = ad.AnnData(X=M.values.astype(np.float32), obs=obs,
                   var=pd.DataFrame(index=pd.Index(
                       [str(c) for c in M.columns], dtype=object)))
    B.uns['mapping'] = ('Ensembl Compara one2one orthologs; duplicates '
                        'collapsed by mean')
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / out_h5ad.name
        B.write_h5ad(tmp)
        shutil.copyfile(str(tmp), out_h5ad)

    M.to_csv(out_dir / ('osdr_paired_human_mapped.csv.gz' if not args.all_samples
                        else 'osdr_all_human_mapped.csv.gz'), compression='gzip')
    map_df = kept.reset_index()
    map_df = map_df.rename(columns={map_df.columns[0]: 'ensembl_id'})
    map_df[['ensembl_id', 'mgi_symbol', 'human_ortholog']].to_csv(
        out_dir / 'mouse2human_mapping_used.csv', index=False)
    (out_dir / 'ortholog_coverage_report.txt').write_text('\n'.join(report) + '\n')

    print('\n'.join(report))
    print(f'\nwrote: {out_h5ad}')
    print('done.')


if __name__ == '__main__':
    main()