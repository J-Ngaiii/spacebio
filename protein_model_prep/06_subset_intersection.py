"""Subset the harmonized dataset to genes measured in every source.

The harmonizer outputs the gene union.

  1. loads out/harmonized_training.h5ad
  2. for each source (obs['source']), finds genes with >=1 non-NaN value
     among that source's samples
  3. writes out/harmonized_training_intersection.h5ad and out/intersection_genes.txt
  4. prints per-source gene counts and the intersection size

"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "out" / "harmonized_training.h5ad"
DEST = HERE / "out" / "harmonized_training_intersection.h5ad"
GENES_TXT = HERE / "out" / "intersection_genes.txt"


def main() -> None:
    import anndata as ad

    print(f"loading {SRC.name} ...")
    adata = ad.read_h5ad(SRC)
    sources = adata.obs["source"].astype(str)
    X = adata.X  # samples x genes, NaN preserved

    measured = {}
    for s in sorted(sources.unique()):
        rows = (sources == s).to_numpy()
        # a gene is "measured" in a source if any of its samples has a value
        measured[s] = np.any(~np.isnan(X[rows]), axis=0)
        print(f"  [{s}] {int(measured[s].sum()):>6} genes measured "
              f"({len(adata.var):>6} in union)")

    # pairwise overlap check
    names = sorted(measured)
    print("\npairwise overlap (rows/cols = sources):")
    header = " " * 14 + " ".join(f"{n[:12]:>12}" for n in names)
    print(header)
    for a in names:
        cells = [f"{int((measured[a] & measured[b]).sum()):>12}" for b in names]
        print(f"{a[:14]:<14}" + " ".join(cells))
    print("\nexample labels measured only in that source:")
    for a in names:
        others = np.logical_and.reduce([measured[b] for b in names if b != a])
        excl = adata.var_names[measured[a] & ~others][:5]
        print(f"  [{a}] {list(map(str, excl))}")

    inter = np.logical_and.reduce([m for m in measured.values()])
    n_inter = int(inter.sum())
    print(f"\nintersection (measured in all {len(measured)} sources): "
          f"{n_inter} genes")

    sub = adata[:, inter].copy()
    sub.uns["intersection_of_sources"] = sorted(measured)
    sub.obs.index = pd.Index([str(i) for i in sub.obs.index], dtype=object, name="sample_id")
    for c in sub.obs.columns:
        if not pd.api.types.is_numeric_dtype(sub.obs[c]):
            sub.obs[c] = (sub.obs[c].astype(object).fillna("")
                          .astype(str).astype(object))
    sub.var.index = pd.Index([str(g) for g in sub.var.index], dtype=object)
    GENES_TXT.write_text("\n".join(map(str, sub.var_names)) + "\n")

    import tempfile, shutil
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "intersection.h5ad"
        sub.write_h5ad(tmp)
        shutil.copyfile(tmp, DEST)
    print(f"written {DEST}: {sub.shape[0]} samples x {sub.shape[1]} genes "
          f"| {len(sub.uns['protein_features'])} protein targets")


if __name__ == "__main__":
    sys.exit(main())
