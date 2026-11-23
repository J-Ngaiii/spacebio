"""Test robustness of a trained Performer/Hyena to random RNA gene masking.

Run from the SpaceBio repository root: python -m scripts.masking_pearson ...
Masking is applied AFTER checkpoint X standardization; 0 = training mean.
"""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from tqdm.auto import tqdm

from src.models.performer import Performer
from src.models.hyena import Hyena


def correlations(y_true, y_pred):
    n_proteins = y_true.shape[1]
    r = np.full(n_proteins, np.nan, dtype=np.float64)
    counts = np.zeros(n_proteins, dtype=np.int64)
    for j in range(n_proteins):
        valid = np.isfinite(y_true[:, j]) & np.isfinite(y_pred[:, j])
        counts[j] = valid.sum()
        if counts[j] < 3:
            continue
        a = y_true[valid, j].astype(np.float64)
        b = y_pred[valid, j].astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom > 0:
            r[j] = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return r, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--target-key', default='protein')
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--external-data', action='store_true',
                   help='Evaluate all samples in an independent dataset, ignoring saved split indices')
    p.add_argument('--percentages', nargs='+', type=int, default=[0, 10, 20, 30, 50, 70, 90])
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    if args.batch_size < 1 or any(not 0 <= v <= 100 for v in args.percentages):
        p.error('batch-size must be positive; percentages must be in [0, 100]')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device(args.device)

    # Checkpoints contain trusted project-local sklearn scalers; do not load untrusted files.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = checkpoint['config']
    if cfg.get('task', 'regression') != 'regression':
        raise ValueError('Only regression checkpoints are supported')
    names = checkpoint.get('gene_names')
    protein_names = checkpoint.get('protein_names')
    if names is None or protein_names is None:
        raise ValueError('Checkpoint must contain gene_names and protein_names')
    names = [str(x) for x in names]
    protein_names = [str(x) for x in protein_names]
    if len(set(names)) != len(names) or len(set(protein_names)) != len(protein_names):
        raise ValueError('Duplicate gene or protein names in checkpoint')

    adata = ad.read_h5ad(args.data)
    if args.target_key not in adata.obsm:
        raise KeyError(f'Missing obsm[{args.target_key!r}]')
    if not adata.var_names.is_unique or not adata.obs_names.is_unique:
        raise ValueError('AnnData gene and sample identifiers must be unique')
    source_proteins = adata.uns.get('protein_features')
    if isinstance(source_proteins, dict):
        if len(source_proteins) != 1:
            raise ValueError('Ambiguous protein_features dictionary')
        source_proteins = next(iter(source_proteins.values()))
    if source_proteins is None:
        raise KeyError("Missing uns['protein_features']")
    source_proteins = [str(x) for x in source_proteins]
    if len(set(source_proteins)) != len(source_proteins):
        raise ValueError('Duplicate dataset protein names')
    gene_index = adata.var_names.get_indexer(names)
    protein_index = pd.Index(source_proteins).get_indexer(protein_names)
    if (gene_index < 0).any() or (protein_index < 0).any():
        raise ValueError('Dataset is missing checkpoint genes or proteins')

    if args.external_data:
        # External OSDR dataset: saved split indices and sample IDs do not apply.
        idx = np.arange(adata.n_obs, dtype=np.int64)
        split_label = 'external'
    else:
        saved_ids = checkpoint.get('sample_ids')
        if saved_ids is None:
            raise ValueError('Checkpoint sample_ids missing; cannot safely use saved split indices')
        saved_ids = [str(x) for x in saved_ids]
        if len(saved_ids) != adata.n_obs or saved_ids != adata.obs_names.astype(str).tolist():
            raise ValueError('Sample IDs/order differ from checkpoint; use --external-data for OSDR')
        idx = np.asarray(checkpoint['split_indices'][args.split], dtype=np.int64)
        split_label = args.split

    if len(idx) == 0 or idx.min() < 0 or idx.max() >= adata.n_obs:
        raise ValueError('Invalid or empty evaluation indices')
    print(f'Dataset: {args.data} | evaluation: {split_label} | samples: {len(idx)}', flush=True)

    # Subset BEFORE densifying; retains exact checkpoint feature order.
    X = adata.X[idx, :][:, gene_index]
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite RNA inputs in selected split')
    Y = adata.obsm[args.target_key]
    if sparse.issparse(Y):
        Y = Y[idx, :][:, protein_index].toarray()
    else:
        Y = np.asarray(Y)[idx, :][:, protein_index]
    Y = np.asarray(Y, dtype=np.float32)
    if np.isinf(Y).any():
        raise ValueError('Infinite protein targets')
    supported = checkpoint.get('train_supported_proteins')
    if supported is not None:
        supported = np.asarray(supported, dtype=bool)
        if len(supported) != Y.shape[1]:
            raise ValueError('Invalid train_supported_proteins length')
        Y[:, ~supported] = np.nan

    scaler = checkpoint.get('x_scaler')
    if scaler is not None:
        X = scaler.transform(X).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite standardized RNA inputs')
    if X.shape[1] != checkpoint['input_dim']:
        raise ValueError('Input dimension mismatch')

    vocabularies = checkpoint.get('metadata_vocabularies', {})
    columns = checkpoint.get('metadata_columns', list(vocabularies))
    metadata = {}
    for column in columns:
        if column in adata.obs.columns:
            values = adata.obs[column].iloc[idx].astype('string').fillna('<MISSING>')
            metadata[column] = values.map(vocabularies.get(column, {})).fillna(0).to_numpy(dtype=np.int64)
        else:
            metadata[column] = np.zeros(len(idx), dtype=np.int64)
    cardinalities = checkpoint.get('metadata_cardinalities', {})
    if 'order' in cfg and 'kernel_size' in cfg:
        architecture = 'hyena'
        model = Hyena(input_dim=checkpoint['input_dim'], output_dim=checkpoint['output_dim'],
                      input_channels=checkpoint['input_channels'], d_model=cfg['d_model'],
                      depth=cfg['depth'], order=cfg['order'], kernel_size=cfg['kernel_size'],
                      dropout=cfg['dropout'], metadata_cardinalities=cardinalities)
    elif 'heads' in cfg:
        architecture = 'performer'
        model = Performer(input_dim=checkpoint['input_dim'], output_dim=checkpoint['output_dim'],
                          input_channels=checkpoint['input_channels'], d_model=cfg['d_model'],
                          depth=cfg['depth'], heads=cfg['heads'], dropout=cfg['dropout'],
                          metadata_cardinalities=cardinalities)
    else:
        raise ValueError('Cannot identify architecture from checkpoint config')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device).eval()

    rng = np.random.default_rng(args.seed)
    # Same per-sample random gene order for all percentages (nested masking).
    # int32 ranks use ~4 bytes per sample-gene entry, instead of float64 noise.
    ranks = np.argsort(rng.random(X.shape, dtype=np.float32), axis=1).argsort(axis=1).astype(np.int32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    protein_rows = []
    for pct in tqdm(sorted(set(args.percentages)), desc='Masking levels'):
        k = round(X.shape[1] * pct / 100)
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(X), args.batch_size):
                end = min(start + args.batch_size, len(X))
                xb = X[start:end].copy()
                if k:
                    xb[ranks[start:end] < k] = 0.0
                if checkpoint['input_channels'] != 1:
                    raise ValueError('Only single-channel RNA inputs supported')
                # Performer training accepts (N,D) and (N,D,1); use (N,D) for both.
                mb = {name: torch.as_tensor(metadata[name][start:end], dtype=torch.long, device=device)
                      for name in columns}
                pred = model(torch.as_tensor(xb, dtype=torch.float32, device=device), mb)
                predictions.append(pred.cpu().numpy())
        pred = np.concatenate(predictions, axis=0)
        if pred.shape != Y.shape:
            raise ValueError(f'Prediction shape {pred.shape} != target shape {Y.shape}')
        r, counts = correlations(Y, pred)
        valid = np.isfinite(r)
        summary_rows.append({'mask_percent': pct, 'masked_genes_per_sample': k,
                             'median_pearson': float(np.median(r[valid])) if valid.any() else np.nan,
                             'mean_pearson': float(np.mean(r[valid])) if valid.any() else np.nan,
                             'n_valid_proteins': int(valid.sum()), 'n_samples': len(idx)})
        for j, name in enumerate(protein_names):
            protein_rows.append({'mask_percent': pct, 'protein': name,
                                 'pearson': r[j], 'n_observed_samples': int(counts[j])})
        print(f'{architecture} {split_label} | masked={pct:3d}% | '
              f'median={summary_rows[-1]["median_pearson"]:.4f} | '
              f'mean={summary_rows[-1]["mean_pearson"]:.4f} | '
              f'valid_proteins={valid.sum()}', flush=True)

    prefix = f'{architecture}_{split_label}_masking'
    pd.DataFrame(summary_rows).to_csv(args.output_dir / f'{prefix}_summary.csv', index=False)
    pd.DataFrame(protein_rows).to_csv(args.output_dir / f'{prefix}_per_protein.csv', index=False)
    print(f'Saved results to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
