"""
Build the gene table (.var) for an OSD study.
Row spine = every gene actually present in the count matrix 
joined to the ortholog crosswalk and the protein accession table.

Join direction: UniProt -> gene (protein rows are scarce).

Usage: python 02_build_var.py [STUDY_ID]
"""
import sys
import pandas as pd
from pathlib import Path

STUDY_ID = sys.argv[1] if len(sys.argv) > 1 else "OSD-105"

PROTEIN_NA_STUDIES = {
    "OSD-48": (
        "processed proteomics is pool-level only (no per-animal values); "
        "protein-coverage columns are filled with NA"
    ),
}

RNA_ONLY_STUDIES = {
    "OSD-100", "OSD-194", "OSD-238", "OSD-239", "OSD-243", "OSD-244",
    "OSD-254", "OSD-289", "OSD-397", "OSD-421", "OSD-515", "OSD-666",
}
for _s in RNA_ONLY_STUDIES:
    PROTEIN_NA_STUDIES[_s] = "RNA-only study (no proteomics in design); protein-coverage columns are NA"

RAW = Path(__file__).resolve().parent.parent / "raw"
OUT = Path(__file__).resolve().parent.parent / "output"
REF = Path(__file__).resolve().parent.parent.parent / "reference"

count_files = list(RAW.glob("*_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv"))
if len(count_files) != 1:
    raise FileNotFoundError(
        f"Expected exactly 1 RSEM unnormalized counts CSV in {RAW}, "
        f"found {len(count_files)}: {[p.name for p in count_files]}"
    )
counts = pd.read_csv(count_files[0], index_col=0)
var = pd.DataFrame({"ensembl_id": counts.index})

orthologs = pd.read_csv(REF / "mouse_human_orthologs.tsv", sep="\t")
orthologs = orthologs.rename(columns={
    "Gene stable ID": "ensembl_id",
    "Gene name": "mgi_symbol",
    "Human gene name": "human_ortholog",
    "Human homology type": "human_homology_type",
})
var = var.merge(orthologs, on="ensembl_id", how="left")

# Real genomes have duplicate ensembl_id rows in BioMart exports when a gene
# has multiple homology-type entries.
dupe_ensembl = var["ensembl_id"].duplicated(keep=False)
if dupe_ensembl.sum():
    print(f"NOTE: {dupe_ensembl.sum()} BioMart rows had duplicate ensembl_id "
          f"— collapsing to one row per gene before continuing")
    var = (
        var.groupby("ensembl_id", as_index=False)
        .agg({
            "mgi_symbol": "first",
            "human_ortholog": lambda s: ";".join(sorted(s.dropna().unique())) or pd.NA,
            "human_homology_type": "first",
        })
    )

missing_ortholog = var["human_ortholog"].isna().sum()
if missing_ortholog:
    print(f"NOTE: {missing_ortholog} genes had no human ortholog — kept, not dropped")

if STUDY_ID in PROTEIN_NA_STUDIES:
    print(f"NOTE: [{STUDY_ID}] {PROTEIN_NA_STUDIES[STUDY_ID]}")
    var["uniprot_accession"] = pd.NA
    var["n_protein_groups"] = 0
    var["ambiguous_protein_map"] = False
    mapped_genes = 0
    n_protein_groups = 0
    unmapped_protein_groups = 0
else:
    protein_files = sorted(RAW.glob("*_TargetProtein.txt"))
    if len(protein_files) != 2:
        raise FileNotFoundError(
            f"Expected exactly 2 TargetProtein tables (one per TMT plex) in {RAW}, "
            f"found {len(protein_files)}: {[p.name for p in protein_files]}"
        )
    protein_table = pd.concat(
        [pd.read_csv(p, sep="\t") for p in protein_files], ignore_index=True
    )
    protein_table["gene_symbol"] = protein_table["Description"].str.extract(r"GN=(\S+)")

    n_protein_groups = len(protein_table)

    protein_per_gene = (
        protein_table.groupby("gene_symbol", as_index=False)["Accession"]
        .agg(lambda s: ";".join(sorted(s)))
        .rename(columns={"Accession": "uniprot_accession"})
    )
    protein_per_gene["n_protein_groups"] = protein_table.groupby("gene_symbol").size().values

    var = var.merge(
        protein_per_gene, left_on="mgi_symbol", right_on="gene_symbol", how="left"
    ).drop(columns=["gene_symbol"])
    var["ambiguous_protein_map"] = var["n_protein_groups"].fillna(0) > 1

    mapped_genes = var["uniprot_accession"].notna().sum()
    unmapped_protein_groups = n_protein_groups - var["n_protein_groups"].fillna(0).sum()
print(f"Protein mapping yield: {mapped_genes} genes reached by protein data, "
      f"out of {n_protein_groups} total protein groups measured "
      f"({unmapped_protein_groups:.0f} protein groups did not reach any gene row)")

var.to_csv(OUT / f"{STUDY_ID}_var.csv", index=False)
print(f"[{STUDY_ID}] var table: {var.shape[0]} genes x {var.shape[1]} columns "
      f"(row count should equal the count matrix's {counts.shape[0]} genes)")
assert var.shape[0] == counts.shape[0], "var row count drifted from counts — dedup bug"
