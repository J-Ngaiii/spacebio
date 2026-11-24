"""Build harmonized_training_bulkformer.h5ad: RNA as log2(TPM+1) per source, not z-scored."""
import argparse
import tempfile
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ad.settings.allow_write_nullable_strings = True

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

SOURCES = ['ccle', 'cptac_brca', 'cptac_lscc', 'cptac_ccrcc', 'cptac_gbm',
           'cptac_hnscc', 'cptac_luad', 'cptac_ov', 'cptac_pdac', 'cptac_ucec',
           'egtex', 'tcga']
LOGGED_P999 = 30.0  # linear TPM / FPKM tops out far above this; log2 values do not

ap = argparse.ArgumentParser()
ap.add_argument('--lengths', default=None,
                help='CSV mapping genes to lengths; needed for tcga')
ap.add_argument('--indir', default=None,
                help='per-source input dir (default: <script dir>/out)')
ap.add_argument('--outdir', default=OUT)
ap.add_argument('--egtex-pseudocount', type=float, default=0.0,
                help='eGTEx sheet holds log2(TPM + this); 0 = log2(TPM), 1 = log2(TPM+1)')
args = ap.parse_args()
if args.indir:
    OUT = Path(args.indir)
BF = Path(args.outdir) / 'bulkformer'
BF.mkdir(parents=True, exist_ok=True)


def _read_either(d: Path, stem: str) -> pd.DataFrame:
    for suffix in ('.csv', '.csv.gz'):
        p = d / f'{stem}{suffix}'
        if p.exists():
            return pd.read_csv(p, index_col=0)
    raise FileNotFoundError(f'{d / stem}.csv[.gz] not found')


def looks_logged(mat: pd.DataFrame) -> bool:
    return float(np.nanpercentile(mat.to_numpy(dtype=float), 99.9)) < LOGGED_P999


def clip_at_zero(lin: pd.DataFrame, source: str) -> tuple[pd.DataFrame, int]:
    '''Linear values below 0 are impossible as TPM; set them to 0 and count them.'''
    n_neg = int((lin < 0).sum().sum())
    if n_neg:
        print(f'  [{source}] {n_neg} values below 0 set to 0')
    return lin.clip(lower=0), n_neg


def rescale_to_million(lin: pd.DataFrame) -> pd.DataFrame:
    '''Each sample (column) sums to 1,000,000 over the genes it has.'''
    col_sum = lin.sum(axis=0, skipna=True)
    return lin.div(col_sum.where(col_sum > 0), axis=1) * 1e6


def to_log2tpm1(mat: pd.DataFrame, source: str) -> tuple[pd.DataFrame, str, int]:
    '''Returns (log2(TPM+1), note, number of values set to 0).'''
    if source == 'ccle':
        return mat, 'already log2(TPM+1)', 0

    if source == 'egtex':
        lin = np.power(2.0, mat) - args.egtex_pseudocount
        lin, n_neg = clip_at_zero(lin, source)
        note = (f'eGTEx un-logged as 2^x - {args.egtex_pseudocount:g}, '
                f'rescaled to 1e6 per sample')
        return np.log2(rescale_to_million(lin) + 1), note, n_neg

    if source.startswith('cptac_'):
        if looks_logged(mat):
            print(f'  [{source}] values look logged (99.9th pct below '
                  f'{LOGGED_P999:g}); un-logging as 2^x - 1 first')
            lin, note = np.power(2.0, mat) - 1.0, 'un-logged (2^x - 1), then '
        else:
            lin, note = mat, ''
        lin, n_neg = clip_at_zero(lin, source)
        return (np.log2(rescale_to_million(lin) + 1),
                note + 'FPKM to TPM per-sample rescale', n_neg)

    if source == 'tcga':
        if args.lengths is None:
            raise SystemExit('tcga needs --lengths')
        lens = pd.read_csv(args.lengths)
        lens.columns = [c.lower() for c in lens.columns]
        sym_col = next((c for c in lens.columns if 'symbol' in c or c == 'gene'), None)
        len_col = next(c for c in lens.columns if 'length' in c)
        id_col = next((c for c in lens.columns if 'ensg' in c or 'ensembl' in c),
                      None)
        idx = mat.index.to_series().astype(str)
        candidates = {}
        if sym_col is not None:
            key = dict(zip(lens[sym_col].astype(str), lens[len_col]))
            candidates['symbol'] = idx.map(key)
        if id_col is not None:
            key = dict(zip(lens[id_col].astype(str).str.split('.').str[0],
                           lens[len_col]))
            candidates['ensg'] = idx.str.split('.').str[0].map(key)
        if not candidates:
            raise SystemExit('--lengths needs a symbol/gene or ensembl column')
        best = max(candidates, key=lambda k: candidates[k].notna().mean())
        L = candidates[best]
        L.index = mat.index
        print(f'  gene lengths matched by {best}: '
              f'{L.notna().sum()}/{len(L)} genes')
        uq, n_neg = clip_at_zero(np.power(2.0, mat) - 1.0, source)
        fpkm = uq.div(L, axis=0) * 1e3
        return (np.log2(rescale_to_million(fpkm) + 1),
                'log2(UQ+1) to TPM via gene lengths, below 0 set to 0', n_neg)

    raise SystemExit(f'unknown source {source}')


def zscore_per_gene(df: pd.DataFrame, source: str) -> pd.DataFrame:
    '''per-gene z across samples; <3 obs or zero variance assigned NaN'''
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0)
    n_obs = df.notna().sum(axis=1)
    dead = (n_obs < 3) | (std == 0) | std.isna()
    z = df.sub(mean, axis=0).div(std, axis=0)
    z[dead] = np.nan
    if dead.any():
        print(f'  [{source}] protein: {int(dead.sum())} dead genes -> NaN')
    return z


def sample_key(source: str, name: str) -> str:
    name = str(name)
    if source == 'ccle':
        return '_'.join(name.split('_')[:-1])
    return name


report, rna_parts, prot_parts = [], [], []
src_rna = {}
for src in SOURCES:
    d = OUT / src
    if not d.exists():
        print(f'!! {d} missing -- skipping {src}')
        report.append({'source': src, 'status': 'DIR MISSING'})
        continue
    print(f'[{src}]')
    rna = _read_either(d, f'{src}_rna')
    prot = _read_either(d, f'{src}_protein')

    prot.columns = [sample_key(src, c) for c in prot.columns]
    if prot.columns.duplicated().any():
        n = int(prot.columns.duplicated().sum())
        print(f'  {n} duplicated protein sample keys')
        prot = prot.T.groupby(level=0).mean().T

    shared = rna.columns.intersection(prot.columns)
    print(f'  paired samples: {len(shared)} '
          f'(RNA {rna.shape[1]}, protein {prot.shape[1]})')
    if len(shared) == 0:
        print(f' no paired samples')
        report.append({'source': src, 'status': 'no paired samples'})
        continue
    rna, prot = rna[shared], prot[shared]

    if rna.index.duplicated().any():
        rna = rna.groupby(level=0).mean()
    if prot.index.duplicated().any():
        prot = prot.groupby(level=0).mean()

    raw_p999 = float(np.nanpercentile(rna.to_numpy(dtype=float), 99.9))
    rna_log, note, n_neg = to_log2tpm1(rna, src)
    rna_log.to_csv(BF / f'rna_log2tpm1_{src}.csv')
    prot_z = zscore_per_gene(prot, src)

    src_rna[src] = rna_log
    rna_parts.append(rna_log.add_suffix(f'__{src}'))
    prot_parts.append(prot_z.add_suffix(f'__{src}'))
    report.append({'source': src, 'status': 'ok', 'note': note,
                   'n_samples': len(shared), 'n_rna_genes': rna_log.shape[0],
                   'n_protein_genes': prot_z.shape[0],
                   'raw_rna_p999': round(raw_p999, 2),
                   'n_set_to_zero': n_neg,
                   'n_rna_blank': int(rna_log.isna().sum().sum()),
                   'p99_rna_log2tpm1': round(float(
                       np.nanpercentile(rna_log.values, 99)), 2)})

rna_all = pd.concat(rna_parts, axis=1)
prot_all = pd.concat(prot_parts, axis=1)
source_of = {c: s for s, m in src_rna.items() for c in m.columns}
rna_all.columns = rna_all.columns.str.split('__').str[0]
prot_all.columns = prot_all.columns.str.split('__').str[0]
rna_all = rna_all.groupby(level=0).mean()
prot_all = prot_all.groupby(level=0).mean()

measured = {}
for src, mat in src_rna.items():
    has_val = pd.Series(False, index=rna_all.index)
    common_genes_src = rna_all.index.intersection(mat.index)
    sub = mat.loc[common_genes_src]
    has_val.loc[common_genes_src] = np.any(~np.isnan(sub.values), axis=1)
    measured[src] = has_val
    print(f'  [{src}] {int(has_val.sum())} genes measured')

common_genes = [g for g in rna_all.index
                if all(measured[s].loc[g] for s in measured)]
print(f'\nintersection: '
      f'{len(common_genes)} genes')
if not common_genes:
    raise SystemExit('empty intersection')

# Report column: share of each sample's 1,000,000 TPM held by the final genes (median),
# before the rescale below. Low means the source spends much of its TPM on other genes.
for row in report:
    if row['source'] in src_rna:
        tpm_final = np.power(2.0, src_rna[row['source']].loc[common_genes]) - 1.0
        row['tpm_share_in_final_genes'] = round(
            float(np.median(np.nansum(tpm_final.to_numpy(), axis=0))) / 1e6, 3)

# Rescale each sample to 1,000,000 TPM over the shared genes only (protein-coding; BulkFormer's
# vocabulary is protein-coding too). Without this, genes a source measures outside the shared
# set (e.g. small non-coding RNAs in cptac_lscc) shrink its values.
tpm_inter = np.power(2.0, rna_all.loc[common_genes]) - 1.0
rna_inter = np.log2(rescale_to_million(tpm_inter) + 1)
prot_all.index = pd.Index([str(g) for g in prot_all.index], dtype=object)

obs = pd.DataFrame({
    'sample_id': rna_inter.columns.astype(str),
    'source': [source_of.get(c, 'na') for c in rna_inter.columns],
}).set_index('sample_id')

adata = ad.AnnData(
    X=rna_inter.T.to_numpy(dtype=np.float32),
    obs=obs,
    var=pd.DataFrame(index=[str(g) for g in rna_inter.index]),
)
adata.obsm['protein'] = (prot_all.T.reindex(adata.obs_names)
                         .to_numpy(dtype=np.float32))
assert not np.isnan(adata.obsm['protein']).all(axis=1).any(), \
    'some RNA samples have no protein row'
adata.uns['protein_features'] = [str(g) for g in prot_all.index]
adata.uns['alignment'] = (
    'X: RNA as log2(TPM+1) per source, not z-scored; values below 0 set to 0; '
    'genes intersected across sources, then each sample rescaled to 1e6 TPM over the shared genes. '
    + '; '.join(f"{r['source']}: {r['note']}" for r in report if r.get('status') == 'ok'))

dest = Path(args.outdir) / 'harmonized_training_bulkformer.h5ad'
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td) / 'harmonized_training_bulkformer.h5ad'
    adata.write_h5ad(tmp)
    shutil.copyfile(tmp, dest)
(BF / 'bulkformer_intersection_genes.txt').write_text(
    '\n'.join(map(str, common_genes)) + '\n')
pd.DataFrame(report).to_csv(BF / 'bulkformer_harmonize_report.csv',
                            index=False)
print(f'\nwritten {dest}: {adata.shape[0]} samples x {adata.shape[1]} genes '
      f'(log2 TPM+1) | {len(prot_all.index)} protein targets')
