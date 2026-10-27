"""Parse the RNA info.

Download manually from
https://depmap.org/portal/data_page/?tab=allData:
  1. OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv
       -> save as raw/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv
  2. Model.csv  (metadata)
       -> save as raw/Model.csv

Format:
row: sequencing profile;
columns:
ProfileID, is_default_entry, ModelID,'SYMBOL (ENTREZ_ID)'
Values are log2(TPM+1).

Maps ModelID to CCLE cell-line name via Model.csv, transposes to genes x samples, and matches cell lines to the
protein matrix.

Writes out/ccle/ccle_rna.csv (genes x matched samples).
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_EXPR = HERE / "raw" / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
RAW_MODEL = HERE / "raw" / "Model.csv"
PROT = HERE / "out" / "ccle" / "ccle_protein.csv"
OUT = HERE / "out" / "ccle"


def gene_symbol(col: str) -> str:
    """Extract the gene symbol from a DepMap column name.

    There are multiple formats: 'TSPAN6 (7105)' (SYMBOL + ENTREZ),
    'TSPAN6 (ENSG000...)', and 'ENSG00000000003 (TSPAN6)'.
    My current rule is that the symbol is whatever is neither an ENSG id nor numeric.
    """
    col = str(col)
    if "(" not in col:
        return col.split(" ")[0].strip()
    before = col.split("(")[0].strip()
    inside = col.split("(")[-1].rstrip(")").strip()
    for cand in (before, inside):
        if cand and not cand.startswith("ENSG") and not cand.isdigit():
            return cand
    return before or inside


def main() -> None:
    for p in (RAW_EXPR, RAW_MODEL, PROT):
        if not p.exists():
            raise SystemExit(f"missing {p.name}")

    print("reading Model.csv...")
    model = pd.read_csv(RAW_MODEL, low_memory=False)
    id_col = "ModelID" if "ModelID" in model.columns else model.columns[0]
    name_col = ("CCLEName" if "CCLEName" in model.columns
                else "StrippedCellLineName")
    id2name = dict(zip(model[id_col].astype(str), model[name_col].astype(str)))
    print(f"  {len(id2name)} models; e.g. "
          f"{list(id2name.items())[:2]}")

    expr = pd.read_csv(RAW_EXPR, index_col=None)
    META_NAMES = {"Unnamed: 0", "ProfileID", "SequencingID",
                  "ModelConditionID", "ModelID", "IsDefaultEntryForMC",
                  "IsDefaultEntryForModel", "is_default_entry"}
    meta_cols = [c for c in expr.columns if str(c) in META_NAMES]
    gene_cols = [c for c in expr.columns if str(c) not in META_NAMES]
    print(f"  raw shape: {expr.shape}; meta columns: {meta_cols}")

    for flag_col in ("IsDefaultEntryForModel", "IsDefaultEntryForMC",
                     "is_default_entry"):
        if flag_col in expr.columns:
            flag = expr[flag_col].astype(str).str.lower()
            keep = flag.isin(("true", "yes", "1"))
            print(f"  default profiles ({flag_col}): "
                  f"{keep.sum()}/{len(expr)}")
            expr = expr[keep]
            break
    id_series = expr["ModelID"].astype(str)
    expr = expr[gene_cols]

    expr.index = [id2name.get(m, m) for m in id_series]
    if expr.index.duplicated().any():
        print(f"  collapsing {expr.index.duplicated().sum()} duplicate "
              "profiles by mean")
        expr = expr.groupby(level=0).mean()

    expr.columns = [gene_symbol(c) for c in expr.columns]
    rna = expr.T.groupby(level=0).mean().T  # collapse dup symbols
    rna.index.name = "sample_id"

    # match cell lines to the protein matrix
    prot = pd.read_csv(PROT, index_col=0, nrows=1)
    prot_keys = ["_".join(s.split("_")[:-1]) for s in prot.columns]
    keep = [i for i, name in enumerate(rna.index) if str(name) in set(prot_keys)]
    print(f"  matched {len(keep)}/{len(rna.index)} RNA cell lines to the "
          f"protein matrix ({len(prot.columns)} protein samples)")

    rna_m = rna.iloc[keep].T
    rna_m.columns = [str(c) for c in rna_m.columns]
    rna_m.index.name = "gene_symbol"

    OUT.mkdir(parents=True, exist_ok=True)
    rna_m.to_csv(OUT / "ccle_rna.csv")
    print(f"written: {rna_m.shape[0]} genes x {rna_m.shape[1]} samples "
          f"-> {OUT / 'ccle_rna.csv'}")


if __name__ == "__main__":
    main()
