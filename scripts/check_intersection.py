import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def to_dense(M):
    if sparse.issparse(M):
        M = M.toarray()
    return np.asarray(M, dtype=np.float32)


def summarize(name, A, obs, source_col, names):
    print(f"\n{'=' * 70}\n{name}: shape {A.shape}\n{'=' * 70}")

    nan = np.isnan(A)
    inf = np.isinf(A)
    bad = nan | inf

    print(f"NaN entries: {nan.sum():,} ({nan.mean():.2%}) | inf entries: {inf.sum():,}")
    if (~bad).any():
        print(f"finite min/max: {A[~bad].min():.4f} / {A[~bad].max():.4f}")

    # Per sample
    row_bad = bad.mean(axis=1)
    print("\nPer-sample fraction bad:")
    print(pd.Series(row_bad).describe().to_string())
    print(f"samples with any bad: {bad.any(axis=1).sum()}/{A.shape[0]}")
    print(f"samples 100% bad:     {bad.all(axis=1).sum()}")
    print(f"samples >50% bad:     {(row_bad > 0.5).sum()}")

    # Per feature
    col_bad = bad.mean(axis=0)
    print("\nPer-feature fraction bad:")
    print(f"features with any bad:   {bad.any(axis=0).sum()}/{A.shape[1]}")
    print(f"features 100% bad:       {bad.all(axis=0).sum()}")
    print(f"features >50% bad:       {(col_bad > 0.5).sum()}")
    print(f"features >20% bad:       {(col_bad > 0.2).sum()}")
    print(f"features fully finite:   {(~bad.any(axis=0)).sum()}")

    worst = np.argsort(-col_bad)[:10]
    print("worst 10 features:", [(names[i], round(float(col_bad[i]), 3)) for i in worst])

    # Per source
    if source_col in obs.columns:
        src = obs[source_col].astype(str).values
        rows = []
        for s in np.unique(src):
            m = src == s
            sb = bad[m]
            col_frac = sb.mean(axis=0)
            rows.append({
                "source": s,
                "n_samples": int(m.sum()),
                "mean_row_bad": float(sb.mean()),
                "features_all_bad": int((col_frac == 1).sum()),
                "features_some_bad": int(((col_frac > 0) & (col_frac < 1)).sum()),
                "features_clean": int((col_frac == 0).sum()),
            })
        tbl = pd.DataFrame(rows).set_index("source")
        print("\nPer-source breakdown:")
        print(tbl.to_string())

        # Structured missingness: within a source, is the mask identical across samples?
        print("\nWithin-source mask consistency (1.0 = every sample misses the same features):")
        for s in np.unique(src):
            sb = bad[src == s]
            if len(sb) > 1:
                consistent = float((sb == sb[0]).all(axis=0).mean())
                print(f"  {s:<14} {consistent:.4f}")

        # Feature coverage across sources
        cov = pd.DataFrame({
            s: ~bad[src == s].any(axis=0) for s in np.unique(src)
        })
        n_src = cov.sum(axis=1)
        print("\nFeatures finite in k sources (k: count):")
        print(n_src.value_counts().sort_index().to_string())
        print(f"features finite in ALL {cov.shape[1]} sources: {(n_src == cov.shape[1]).sum()}")

    return bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path,
                   default=Path("data/processed/sequence/harmonized/harmonized_training_intersection.h5ad"))
    p.add_argument("--target-key", default="protein")
    p.add_argument("--source-col", default="source")
    args = p.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(args.data)

    a = ad.read_h5ad(args.data)
    print(f"File: {args.data}")
    print(f"obs columns: {list(a.obs.columns)}")
    print(f"obsm keys:   {list(a.obsm.keys())}")
    print(f"uns keys:    {list(a.uns.keys())}")

    gene_names = np.asarray(a.var_names)
    print(f"var_names unique: {len(set(gene_names))}/{len(gene_names)}")

    X = to_dense(a.X)
    bad_x = summarize("X (RNA)", X, a.obs, args.source_col, gene_names)

    if args.target_key in a.obsm:
        Y = to_dense(a.obsm[args.target_key])
        pf = a.uns.get("protein_features")
        if isinstance(pf, dict) and len(pf) == 1:
            pf = next(iter(pf.values()))
        pnames = np.asarray(pf).astype(str) if pf is not None else np.arange(Y.shape[1]).astype(str)
        bad_y = summarize(f"Y ({args.target_key})", Y, a.obs, args.source_col, pnames)

        print(f"\n{'=' * 70}\nJoint: samples usable if both X and Y are fully finite\n{'=' * 70}")
        ok = ~bad_x.any(axis=1) & ~bad_y.any(axis=1)
        print(f"fully finite X and Y: {ok.sum()}/{X.shape[0]}")
        if args.source_col in a.obs.columns:
            print(a.obs.loc[ok, args.source_col].value_counts().to_string())
    else:
        print(f"\nobsm[{args.target_key!r}] not found")


if __name__ == "__main__":
    main()