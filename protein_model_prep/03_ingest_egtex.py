"""Parse the eGTEx dataset from GTEx Portal into the standard source layout.

Input structure:
  Table_S1, sheet 'G protein cleaned relative abundance':
      header row = tissue labels; first data row = GTEx sample IDs;
      values = log2 abundance ratio.
  Table_S2, sheet 'A RNA cleaned log TPM'.
  Both tables keyed by Ensembl gene id (columns 'gene.id.full', 'gene.id')
  and mapped to gene symbols via mygene.info.

Output: out/egtex/egtex_{protein,rna,samples}.csv (genes x samples).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = HERE / "out" / "egtex"

PROT_XLSX = RAW / "Table_S1_gene_info_at_protein_level.xlsx"
RNA_XLSX = RAW / "Table_S2_gene_info_at_RNA_level.xlsx"
PROT_SHEET = "G protein cleaned relative abundance"
RNA_SHEET = "A RNA cleaned log TPM"


def parse_sheet(xlsx: Path, sheet: str) -> tuple[pd.DataFrame, pd.Series]:
    """Returns genes x samples matrix indexed by ENSG, tissue per sample."""
    df = pd.read_excel(xlsx, sheet_name=sheet)
    sample_ids = df.iloc[0, 2:].astype(str).tolist()
    tissues = pd.Series(df.columns[2:].astype(str)).str.replace(r"\.\d+$", "",
                                                                regex=True)
    tissues.index = sample_ids

    mat = df.iloc[1:, 2:].apply(pd.to_numeric, errors="coerce")
    mat.columns = sample_ids
    mat.index = df.iloc[1:, 1].astype(str)
    mat.index.name = "ensg"

    # drop all-NaN rows and any duplicated column ids
    mat = mat.dropna(how="all")
    mat = mat.loc[:, ~pd.Index(sample_ids).duplicated()]
    print(f"  parsed {mat.shape[0]} genes x {mat.shape[1]} samples")
    return mat, tissues


def ensg_to_symbol(ensg_ids: list[str]) -> pd.Series:
    """ENSG to gene symbol via mygene.info."""
    ids = sorted(set(ensg_ids))
    out = {}
    for i in range(0, len(ids), 900):
        batch = ids[i:i + 900]
        for attempt in range(4):
            try:
                r = requests.post("https://mygene.info/v3/query",
                                  data={"q": "\n".join(batch),
                                        "scopes": "ensembl.gene",
                                        "fields": "symbol",
                                        "species": "human"},
                                  timeout=120)
                r.raise_for_status()
                for hit in r.json():
                    q = hit.get("query")
                    if q and hit.get("symbol"):
                        out[q] = hit["symbol"]
                    elif q and hit.get("hits"):
                        sym = hit["hits"][0].get("symbol")
                        if sym:
                            out[q] = sym
                break
            except Exception as e:
                if attempt == 3:
                    print(f"  mygene batch failed: {str(e)[:100]}")
                time.sleep(5 * (attempt + 1))
        print(f"  mapped {min(i + 900, len(ids))}/{len(ids)} ENSG ids")
    return pd.Series(out)


def find_sheet(xl: pd.ExcelFile, key: str) -> str:
    matches = [s for s in xl.sheet_names if key.lower() in s.lower()]
    if len(matches) != 1:
        raise ValueError(f"expected exactly 1 sheet matching '{key}', "
                         f"found {matches}")
    return matches[0]


def main() -> None:
    for f in (PROT_XLSX, RNA_XLSX):
        if not f.exists():
            raise SystemExit(f"missing {f} (eGTEx proteomics tables from the GTEx Portal, expected in raw/)")

    xl_p = pd.ExcelFile(PROT_XLSX)
    prot, tissues_p = parse_sheet(PROT_XLSX, find_sheet(xl_p, "relative"))
    xl_r = pd.ExcelFile(RNA_XLSX)
    rna, tissues_r = parse_sheet(RNA_XLSX, find_sheet(xl_r, "cleaned log TPM"))

    def to_aliquot(ids):
        return ["-".join(str(s).split("-")[:3]) for s in ids]

    p_al, r_al = to_aliquot(prot.columns), to_aliquot(rna.columns)
    for name, ids in (("protein", p_al), ("rna", r_al)):
        dups = sorted({x for x in ids if ids.count(x) > 1})
        if dups:
            print(f"  Duplicate aliquot ids in {name}, dropped: {dups}")
    prot.columns, rna.columns = p_al, r_al
    tissues_p.index, tissues_r.index = p_al, r_al

    shared = prot.columns.intersection(rna.columns)
    print(f"paired samples: {len(shared)} of "
          f"protein {prot.shape[1]} / RNA {rna.shape[1]}")
    if len(shared) == 0:
        raise SystemExit("no shared sample ids between protein and RNA tables")

    # tissue labels kept consistent across assays for the same aliquot
    mism = [a for a in shared if tissues_p[a] != tissues_r[a]]
    if mism:
        print(f"  Tissue label mismatch for {len(mism)} aliquots "
              f"(dropped): {mism[:5]}")
        shared = [a for a in shared if a not in mism]
    prot, rna = prot[shared], rna[shared]

    sym = ensg_to_symbol(list(prot.index) + list(rna.index))
    print(f"  {len(sym)}/{len(set(prot.index) | set(rna.index))} ids mapped")

    def rename_genes(df: pd.DataFrame) -> pd.DataFrame:
        g = df.index.to_series().map(sym)
        keep = g.notna()
        df = df[keep.to_numpy()]
        g = g[keep].astype(str)
        df.index = g
        df.index.name = "gene_symbol"
        return df.groupby(level=0).mean(numeric_only=True)

    prot_s, rna_s = rename_genes(prot), rename_genes(rna)
    print(f"  protein {prot_s.shape}, RNA {rna_s.shape}")

    OUT.mkdir(parents=True, exist_ok=True)
    prot_s.to_csv(OUT / "egtex_protein.csv")
    rna_s.to_csv(OUT / "egtex_rna.csv")
    meta = pd.DataFrame({"sample_id": shared,
                         "tissue": tissues_p.loc[shared].to_numpy()})
    meta["source"] = "egtex"
    meta.to_csv(OUT / "egtex_samples.csv", index=False)


if __name__ == "__main__":
    main()
