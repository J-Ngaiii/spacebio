"""Extract mean gene-token representations from a trained SpaceBio Performer.

Run from the project root: python -m scripts.extract_embeddings ...
"""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from src.models.performer import Performer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-samples', type=int, default=0, help='0 means all held-out test samples')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.batch_size < 1 or args.num_samples < 0:
        p.error('batch-size must be positive and num-samples nonnegative')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    model = Performer(input_dim=ck['input_dim'], output_dim=ck['output_dim'],
                      input_channels=ck['input_channels'], d_model=cfg['d_model'],
                      depth=cfg['depth'], heads=cfg['heads'], dropout=cfg['dropout'],
                      metadata_cardinalities=ck['metadata_cardinalities'])
    model.load_state_dict(ck['model_state_dict'])
    model.to(device).eval()
    adata = ad.read_h5ad(args.data, backed='r')
    if adata.n_obs != len(ck['sample_ids']) or list(adata.obs_names.astype(str)) != list(ck['sample_ids']):
        raise ValueError('AnnData samples do not match checkpoint sample order')
    genes = np.asarray(ck['gene_names'], dtype=str)
    cols = adata.var_names.get_indexer(genes)
    if (cols < 0).any():
        raise ValueError(f'Missing checkpoint genes: {genes[cols < 0][:10]}')
    indices = np.asarray(ck['split_indices']['test'], dtype=np.int64)
    if args.num_samples and args.num_samples < len(indices):
        indices = np.sort(np.random.default_rng(args.seed).choice(indices, args.num_samples, replace=False))
    if len(indices) == 0:
        raise ValueError('Empty test split')

    # Register one hook on the embedding output and one on each backbone layer.
    # performer-pytorch's encoder uses a Transformer with a ModuleList named layers.
    layers = model.encoder.net.layers
    if len(layers) != cfg['depth']:
        raise ValueError('Encoder layer count does not match checkpoint depth')
    names = ['input'] + [f'layer_{i+1}' for i in range(len(layers))]
    sums = {name: np.zeros((len(genes), cfg['d_model']), dtype=np.float64) for name in names}
    counts = {name: 0 for name in names}

    def hook_for(name):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            if h.ndim != 3 or h.shape[1:] != (len(genes), cfg['d_model']):
                raise ValueError(f'Unexpected {name} activation shape: {tuple(h.shape)}')
            sums[name][:] += h.detach().float().sum(dim=0).cpu().numpy()
            counts[name] += h.shape[0]
        return hook

    handles = [model.embedding.register_forward_hook(hook_for('input'))]
    handles += [layer.register_forward_hook(hook_for(f'layer_{i+1}')) for i, layer in enumerate(layers)]
    raw_sum = np.zeros(len(genes), dtype=np.float64)
    raw_abs_sum = np.zeros(len(genes), dtype=np.float64)
    raw_nonnegative = True
    try:
        with torch.no_grad():
            for start in range(0, len(indices), args.batch_size):
                ix = indices[start:start + args.batch_size]
                # h5py backed fancy indexing requires sorted indices.
                raw = adata.X[ix, :]
                if sparse.issparse(raw):
                    raw = raw.toarray()
                raw = np.asarray(raw, dtype=np.float32)[:, cols]
                if not np.isfinite(raw).all():
                    raise ValueError('Nonfinite RNA values in selected genes')
                raw_nonnegative &= bool(np.all(raw >= 0))
                raw_sum += raw.sum(axis=0, dtype=np.float64)
                raw_abs_sum += np.abs(raw).sum(axis=0, dtype=np.float64)
                scaler = ck.get('x_scaler')
                X = scaler.transform(raw).astype(np.float32) if scaler is not None else raw
                obs = adata.obs.iloc[ix]
                metadata = {}
                for name in ck['metadata_columns']:
                    vocab = ck['metadata_vocabularies'][name]
                    if name in obs:
                        values = obs[name].astype('string').fillna('<MISSING>')
                    else:
                        values = pd.Series(['<MISSING>'] * len(ix), index=obs.index, dtype='string')
                    metadata[name] = torch.tensor(values.map(vocab).fillna(0).to_numpy(dtype=np.int64), device=device)
                _ = model(torch.from_numpy(X).to(device), metadata)
                print(f'Processed {min(start + args.batch_size, len(indices))}/{len(indices)}', flush=True)
    finally:
        for handle in handles:
            handle.remove()
        if getattr(adata, 'file', None) is not None:
            adata.file.close()

    if any(count != len(indices) for count in counts.values()):
        raise RuntimeError(f'Incomplete layer capture: {counts}')
    # The original input token is gene ID + expression projection + metadata.
    static_gene_ids = model.embedding.gene_embedding.weight.detach().cpu().numpy().astype(np.float32)
    mean_raw = raw_sum / len(indices)
    if raw_nonnegative:
        expr_covariate = np.log1p(mean_raw)
        expr_label = 'log1p(mean_input_expression)'
    else:
        # A valid nonnegative proxy for already standardized/signed data; not TPM.
        expr_covariate = np.log1p(raw_abs_sum / len(indices))
        expr_label = 'log1p(mean_absolute_input_expression); signed-input proxy, NOT TPM'
        print('WARNING: negative expression values detected; using log1p(mean absolute expression) proxy, not TPM.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {name: (sums[name] / counts[name]).astype(np.float32) for name in names}
    np.savez_compressed(args.output_dir / 'gene_embeddings.npz',
                        gene_names=genes, sample_indices=indices,
                        sample_ids=np.asarray(ck['sample_ids'], dtype=str)[indices],
                        static_gene_id=static_gene_ids,
                        mean_input_expression=mean_raw.astype(np.float32),
                        expression_covariate=expr_covariate.astype(np.float32),
                        **arrays)
    manifest = {'checkpoint': str(args.checkpoint), 'data': str(args.data),
                'num_samples': len(indices), 'num_genes': len(genes),
                'd_model': cfg['d_model'], 'layers': names,
                'expression_covariate': expr_label,
                'notes': 'Mean contextual gene-token activations over held-out samples; static_gene_id saved separately.'}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Saved {args.output_dir / "gene_embeddings.npz"}')


if __name__ == '__main__':
    main()
