"""Build a gene length table (union of all exons per gene, in bp) from an Ensembl GTF, for TPM normalization.

Example (the table committed at data/reference/mouse_gene_lengths.csv):
    curl -LO https://ftp.ensembl.org/pub/release-116/gtf/mus_musculus/Mus_musculus.GRCm39.116.gtf.gz
    python scripts/build_gene_lengths.py --gtf Mus_musculus.GRCm39.116.gtf.gz --out data/reference/mouse_gene_lengths.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def union_exon_lengths(gtf_path: Path) -> pd.Series:
    """Gene ID -> total length of the gene's exons after merging overlaps across all its transcripts."""
    chunks = []
    for chunk in pd.read_csv(gtf_path, sep="\t", comment="#", header=None, usecols=[2, 3, 4, 8], names=["feature", "start", "end", "attributes"], chunksize=500_000):
        exons = chunk[chunk["feature"] == "exon"]
        chunks.append(pd.DataFrame({"gene_id": exons["attributes"].str.extract(r'gene_id "([^"]+)"', expand=False), "start": exons["start"], "end": exons["end"]}))
    exons = pd.concat(chunks).sort_values(["gene_id", "start"], ignore_index=True)

    # An exon starts a new merged block when it begins after every earlier exon of the same gene has ended.
    reach = exons.groupby("gene_id")["end"].cummax()
    previous_reach = reach.groupby(exons["gene_id"]).shift()
    block = (previous_reach.isna() | (exons["start"] > previous_reach)).cumsum()
    blocks = exons.groupby(block).agg(gene_id=("gene_id", "first"), start=("start", "min"), end=("end", "max"))
    return (blocks["end"] - blocks["start"] + 1).groupby(blocks["gene_id"]).sum().rename("length")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gtf", type=Path, required=True, help="Ensembl GTF, optionally gzipped.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lengths = union_exon_lengths(args.gtf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lengths.rename_axis("ensembl_id").reset_index().to_csv(args.out, index=False, lineterminator="\n")
    print(f"Wrote {len(lengths)} gene lengths to {args.out}")


if __name__ == "__main__":
    main()
