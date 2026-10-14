"""
Build the merged RNA expression matrix.
Values came in as normalized counts.
"""
import gzip
import io
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "raw"
OUT = Path(__file__).resolve().parent.parent / "output"

TISSUES = ["Kidney", "Liver", "Muscle", "Skin"]


def read_cr_tsv(path):
    """Read a TSV that uses \r line endings."""
    with gzip.open(path, "rt") as f:
        text = f.read()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return pd.read_csv(io.StringIO(text), sep="\t", index_col=0)


parts = []
for tissue in TISSUES:
    df = read_cr_tsv(RAW / f"GSE225576_{tissue}.txt.gz")
    df.columns = [f"{tissue.lower()}_{c}" for c in df.columns]
    parts.append(df)
    print(f"  {tissue}: {df.shape[0]} genes x {df.shape[1]} samples")

counts = pd.concat(parts, axis=1)
if counts.isna().any().any():
    raise ValueError("NaNs in the merged RNA matrix -- gene spines drifted between tissues")

counts.to_csv(OUT / "Earth_expression_counts.csv")
print(f"[Earth] expression matrix: {counts.shape[0]} genes x {counts.shape[1]} samples "
      f"(normalized counts, values are fractional)")