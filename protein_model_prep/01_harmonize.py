"""Harmonize paired RNA + protein sources into one training object.

  1. Join on gene symbol.
  2. RNA: log-transform if not already, then per-source per-gene z-score across samples.
  3. Protein: per-source per-gene z-score (even if already reference-normalized)
  4. Missing values stay NaN.
  5. Every sample keeps its source label (obs['source'])

Input layout:
  out/<source>/<source>_rna.csv       genes x samples
  out/<source>/<source>_protein.csv   genes x samples
  out/<source>/<source>_samples.csv   sample metadata

Output:
  out/harmonized_training.h5ad   X = RNA (z-scored), obsm['protein'] =
                                 protein (z-scored), obs['source'] = label
  out/harmonize_report.txt       per-source stats + QC notes

python3 01_harmonize.py            # all sources found under out/
python3 01_harmonize.py ccle       # one source only
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# Sources with paired RNA and protein.
SOURCES = ["ccle", "egtex", "cptac", "tcga"]

def log_transform_rna(df: pd.DataFrame, source: str) -> pd.DataFrame:
    vals = df.to_numpy(dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        raise ValueError(f"{source}: RNA matrix is empty")
    p99 = np.percentile(vals, 99.9)
    if p99 > 100:
        print(f"  [{source}] RNA looks linear-scale (99.9pct={p99:.0f}). Do log2(x+1)")
        return np.log2(df + 1)
    if p99 > 25:
        print(f"  [{source}] RNA already log-scale (99.9pct={p99:.1f}). Passthrough")
        return df
    print(f"  [{source}] RNA looks log/z-scale already (99.9pct={p99:.1f}). Passthrough")
    return df


def zscore_per_gene(df: pd.DataFrame, source: str, kind: str) -> pd.DataFrame:
    """Per-gene (row) z-score across samples (columns).

    Genes with <3 measured samples or zero variance are assigned NaN.
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0)
    n_obs = df.notna().sum(axis=1)
    dead = (n_obs < 3) | (std == 0) | std.isna()
    z = df.sub(mean, axis=0).div(std, axis=0)
    z[dead] = np.nan
    n_dead = int(dead.sum())
    if n_dead:
        print(f"  [{source}] {kind}: {n_dead} genes with <3 obs or zero variance. Kept as NaN")
    return z


def sample_key(source: str, name: str) -> str:
    name = str(name)
    if source == "ccle":
        return "_".join(name.split("_")[:-1])
    return name


def _read_either(d: Path, stem: str) -> pd.DataFrame:
    """Read <stem>.csv or <stem>.csv.gz."""
    for suffix in (".csv", ".csv.gz"):
        p = d / f"{stem}{suffix}"
        if p.exists():
            return pd.read_csv(p, index_col=0)
    raise FileNotFoundError(f"{d / stem}.csv[.gz] not found")


def load_source(source: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (rna genes x samples, protein genes x samples, sample meta)."""
    d = OUT / source
    rna = _read_either(d, f"{source}_rna")
    prot = _read_either(d, f"{source}_protein")
    prot.columns = [sample_key(source, c) for c in prot.columns]
    if prot.columns.duplicated().any():
        n_dup = int(prot.columns.duplicated().sum())
        print(f"  [{source}] {n_dup} samples share a cell-line key. Do mean-collapse")
        prot = prot.T.groupby(level=0).mean().T
    meta_p = d / f"{source}_samples.csv"
    meta = pd.read_csv(meta_p) if meta_p.exists() else pd.DataFrame(index=rna.columns)
    return rna, prot, meta


def harmonize(sources: list[str]):
    rna_parts, prot_parts, obs_parts = [], [], []
    report = []

    for source in sources:
        print(f"[{source}]")
        rna, prot, meta = load_source(source)

        shared_samples = rna.columns.intersection(prot.columns)
        if len(shared_samples) < len(rna.columns) or len(shared_samples) < len(prot.columns):
            print(f"  sample mismatch: RNA {len(rna.columns)} vs protein "
                  f"{len(prot.columns)} -> keeping {len(shared_samples)} paired")
        rna, prot = rna[shared_samples], prot[shared_samples]

        if rna.index.duplicated().any():
            print(f"  [{source}] {int(rna.index.duplicated().sum())} duplicate RNA "
                  f"gene names. Do mean-collapse")
            rna = rna.groupby(level=0).mean()
        if prot.index.duplicated().any():
            print(f"  [{source}] {int(prot.index.duplicated().sum())} duplicate protein "
                  f"gene names. Do mean-collapse")
            prot = prot.groupby(level=0).mean()

        rna = log_transform_rna(rna, source)
        rna_z = zscore_per_gene(rna, source, "RNA")
        prot_z = zscore_per_gene(prot, source, "protein")

        rna_parts.append(rna_z.add_suffix(f"__{source}"))
        prot_parts.append(prot_z.add_suffix(f"__{source}"))
        if "sample_id" in meta.columns:
            meta = meta.set_index("sample_id")
            meta.index = [sample_key(source, s) for s in meta.index]
            meta = meta[~meta.index.duplicated(keep="first")]
            meta = meta.loc[shared_samples]
        else:
            meta = meta.loc[shared_samples] if len(meta) == len(shared_samples) else meta
        meta["source"] = source
        obs_parts.append(meta)

        report.append({"source": source, "n_samples": len(shared_samples),
                       "n_rna_genes": int(rna_z.notna().any(axis=1).sum()),
                       "n_protein_genes": int(prot_z.notna().any(axis=1).sum())})

    # stack samples across sources
    rna_all = pd.concat(rna_parts, axis=1)   # genes x samples
    prot_all = pd.concat(prot_parts, axis=1)
    obs_all = pd.concat(obs_parts, axis=0)

    # For the same gene from multiple sources, take mean of the per-source z-scores
    rna_all.columns = rna_all.columns.str.split("__").str[0]
    prot_all.columns = prot_all.columns.str.split("__").str[0]
    rna_all = rna_all.groupby(level=0).mean()   # genes x samples, deduped
    prot_all = prot_all.groupby(level=0).mean()  # protein genes x samples, deduped

    obs_all.index = pd.Index([str(i) for i in obs_all.index], dtype=object, name="sample_id")
    for c in obs_all.columns:
        if not pd.api.types.is_numeric_dtype(obs_all[c]):
            obs_all[c] = (obs_all[c].astype(object).fillna("")
                          .astype(str).astype(object))

    import anndata as ad

    rna_all.index = pd.Index([str(g) for g in rna_all.index], dtype=object)
    prot_all.index = pd.Index([str(g) for g in prot_all.index], dtype=object)

    adata = ad.AnnData(
        X=rna_all.T.to_numpy(dtype=np.float32),   # samples x genes
        obs=obs_all,
        var=pd.DataFrame(index=rna_all.index),
    )
    adata.obsm["protein"] = prot_all.T.loc[adata.obs_names].to_numpy(dtype=np.float32)
    adata.uns["protein_features"] = [str(g) for g in prot_all.index]

    import tempfile, shutil
    dest = OUT / "harmonized_training.h5ad"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "harmonized_training.h5ad"
        adata.write_h5ad(tmp)
        shutil.copyfile(tmp, dest)
    pd.DataFrame(report).to_csv(OUT / "harmonize_report.csv", index=False)
    print(f"\nwritten {dest}: {adata.shape[0]} samples x {adata.shape[1]} RNA genes "
          f"| {len(prot_all.index)} protein targets | sources: "
          f"{obs_all['source'].value_counts().to_dict()}")


def main() -> None:
    if len(sys.argv) > 1:
        sources = [sys.argv[1]]
    else:
        sources = sorted(d.name for d in OUT.iterdir()
                         if d.is_dir() and (list(d.glob("*_protein.csv"))
                                            or list(d.glob("*_protein.csv.gz"))))
    available = [s for s in sources
                 if (OUT / s / f"{s}_protein.csv").exists()
                 or (OUT / s / f"{s}_protein.csv.gz").exists()]
    harmonize(available)


if __name__ == "__main__":
    main()
