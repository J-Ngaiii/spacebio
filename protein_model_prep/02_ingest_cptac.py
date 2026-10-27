"""Ingest CPTAC proteomics and matched RNA.

Protein: log2 ratios vs the study's pooled reference. 
RNA: gene-level FPKM. One output folder per cancer type:

  out/cptac_<type>/cptac_<type>_rna.csv       genes x samples (FPKM)
  out/cptac_<type>/cptac_<type>_protein.csv   genes x samples (log2 ratios)
  out/cptac_<type>/cptac_<type>_samples.csv   sample metadata (sample_id, cancer_type)

Tumor and normal-adjacent (NAT) samples are both kept.

python3 02_ingest_cptac.py            # all available cancer types
python3 02_ingest_cptac.py Ccrcc      # one type
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

CANCER_TYPES = ["Brca", "Ccrcc", "Coad", "Gbm", "Hnscc", "Luad", "Lscc",
                "Ov", "Pdac", "Ucec"]
RNA_SOURCE = {"Lscc": "broad"}


def flatten_gene_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a MultiIndex (Name, ...) to plain gene symbols."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values("Name")
    # mean over duplicate gene symbols
    return df.T.groupby(level=0).mean().T


def ingest(cancer: str) -> None:
    import cptac

    cls = getattr(cptac, cancer, None)
    if cls is None:
        print(f"[{cancer}] not available in this cptac version")
        return
    cn = cls()

    prot = cn.get_proteomics("umich")
    prot = flatten_gene_columns(prot)
    rna_src = RNA_SOURCE.get(cancer, "washu")
    rna = cn.get_transcriptomics(rna_src)
    rna = flatten_gene_columns(rna)

    shared = prot.index.intersection(rna.index)
    prot, rna = prot.loc[shared], rna.loc[shared]

    # genes as rows, samples as columns
    prot, rna = prot.T, rna.T

    dest = OUT / f"cptac_{cancer.lower()}"
    dest.mkdir(parents=True, exist_ok=True)
    rna.to_csv(dest / f"cptac_{cancer.lower()}_rna.csv")
    prot.to_csv(dest / f"cptac_{cancer.lower()}_protein.csv")
    pd.DataFrame({"sample_id": shared, "cancer_type": cancer.lower()}).to_csv(
        dest / f"cptac_{cancer.lower()}_samples.csv", index=False)

    miss_p = prot.isna().sum().sum() / prot.size
    print(f"[{cancer}] {len(shared)} samples | RNA {rna.shape[0]} genes | "
          f"protein {prot.shape[0]} genes | protein missing {miss_p:.0%}")


def main() -> None:
    types = sys.argv[1:] or CANCER_TYPES
    for cancer in types:
        try:
            ingest(cancer)
        except Exception as e:
            print(f"[{cancer}] FAILED: {str(e)[:200]}")


if __name__ == "__main__":
    main()
