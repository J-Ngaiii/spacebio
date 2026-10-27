"""Ingest CCLE proteomics into the standard source layout.
   sample names: e.g. MDAMB468_BREAST_TenPx01 (cell line, lineage, plex)
Output:
  ccle_protein.csv   genes x samples (gene symbol index, NaN = missing)
  ccle_samples.csv   sample metadata: cell_line, lineage, plex_id

"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://gygi.hms.harvard.edu/data/ccle/protein_quant_current_normalized.csv.gz"
HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "ccle"
RAW = HERE / "raw"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)


def ensure_raw(path: str | None) -> Path:
    if path:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(src)
        return src
    dest = RAW / "protein_quant_current_normalized.csv.gz"
    if not dest.exists():
        import requests

        print(f"downloading {URL}")
        r = requests.get(URL, timeout=600)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"  ok: {len(r.content) / 1e6:.1f} MB")
    return dest


def main() -> None:
    src = ensure_raw(sys.argv[1] if len(sys.argv) > 1 else None)
    df = pd.read_csv(src)

    meta_cols = ["Protein_Id", "Gene_Symbol", "Description", "Group_ID",
                 "Uniprot", "Uniprot_Acc"]
    missing_meta = [c for c in meta_cols if c not in df.columns]
    if missing_meta:
        raise ValueError(f"unexpected CCLE file layout, missing: {missing_meta}")

    sample_cols = [c for c in df.columns if c not in meta_cols]
    pep_cols = [c for c in sample_cols if c.endswith("_Peptides")]
    value_cols = [c for c in sample_cols if not c.endswith("_Peptides")]
    print(f"rows: {len(df)} | sample columns: {len(value_cols)} "
          f"(dropped {len(pep_cols)} QC '_Peptides' columns)")

    def parse_sample(col: str) -> dict:
        parts = col.split("_")
        return {"sample_id": col, "cell_line": "_".join(parts[:-2]),
                "lineage": parts[-2], "plex_id": parts[-1]}

    samples = pd.DataFrame([parse_sample(c) for c in value_cols])

    prot = df[["Gene_Symbol"] + value_cols].copy()
    prot["Gene_Symbol"] = prot["Gene_Symbol"].astype(str).str.strip()
    bad = (prot["Gene_Symbol"].isin({"nan", "None", ""})
           | prot["Gene_Symbol"].str.contains(r"[|#]", regex=True))
    if bad.any():
        print(f"  dropping {int(bad.sum())} rows with missing gene symbols")
        prot = prot[~bad]

    # collapse protein groups mapped to the same symbol by mean
    n_dup = int(prot["Gene_Symbol"].duplicated().sum())
    prot = prot.groupby("Gene_Symbol").mean(numeric_only=True)
    if n_dup:
        print(f"  collapsed {n_dup} duplicate-symbol rows by mean")

    prot.to_csv(OUT / "ccle_protein.csv")
    samples.to_csv(OUT / "ccle_samples.csv", index=False)

    vals = prot.to_numpy(dtype=float).ravel()
    vals = vals[~np.isnan(vals)]

if __name__ == "__main__":
    main()
