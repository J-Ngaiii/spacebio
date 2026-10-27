"""
Assemble the per-study AnnData object (RNA in X, protein in obsm when available),
check arm labels and protein alignment, and write the column annotations.

"""
import sys
import pandas as pd
import anndata as ad
from pathlib import Path

STUDY_ID = sys.argv[1] if len(sys.argv) > 1 else "OSD-105"

NO_PROTEIN_STUDIES = {"OSD-48"}

OUT = Path(__file__).resolve().parent.parent / "output"

obs = pd.read_csv(OUT / f"{STUDY_ID}_obs.csv").set_index("sample_id")
var = pd.read_csv(OUT / f"{STUDY_ID}_var.csv").set_index("ensembl_id")
counts = pd.read_csv(OUT / f"{STUDY_ID}_expression_counts.csv", index_col=0)

adata = ad.AnnData(
    X=counts.T.values,       # AnnData convention: samples are rows
    obs=obs,
    var=var,
)

if STUDY_ID in NO_PROTEIN_STUDIES:
    print(f"NOTE: [{STUDY_ID}] no per-sample protein matrix (pool-level proteomics only); "
          f"assembled RNA-only")
else:
    # Script 04 saves a (gene_symbol, tmt_plex) MultiIndex.
    protein = pd.read_csv(OUT / f"{STUDY_ID}_protein_matrix.csv", index_col=[0, 1])
    # Reindex protein columns to obs order so obsm rows align with obs.
    missing = set(adata.obs_names) - set(protein.columns)
    if missing:
        raise ValueError(f"protein matrix missing samples: {sorted(missing)}")
    protein = protein.reindex(columns=adata.obs_names)
    adata.obsm["protein"] = protein.T.values  # paired modality, rows aligned with obs
    print(f"  paired protein modality: {adata.obsm['protein'].shape}")

print(f"[{STUDY_ID}] AnnData assembled: {adata.shape[0]} samples x {adata.shape[1]} genes")

if adata.obs["arm"].isna().any():
    raise ValueError("arm mapping failed for some samples")
if adata.obs["arm"].nunique() < 2:
    raise ValueError("Only one arm present; check tissue/arm filtering upstream")

# Column-level provenance: source and description of each column.
COLUMN_ANNOTATIONS = [
    # (table, column, source, description)
    ("obs", "sample_id", "runsheet.csv: Sample Name",
     "Primary key. GLDS sample name encoding organism_strain_tissue_arm_replicate_animal. Joins obs, expression columns, and protein columns."),
    ("obs", "organism", "runsheet.csv: organism", "Organism scientific name (Mus musculus)."),
    ("obs", "has_ercc", "runsheet.csv: has_ERCC", "Whether ERCC spike-ins were added to the library (QC metadata)."),
    ("obs", "organism_code", "parsed from sample_id by script 01 regex", "First token of the sample name (Mmus). Redundant with organism; kept for name round-tripping."),
    ("obs", "strain", "parsed from sample_id by script 01 regex", "Mouse strain token (C57-6J = C57BL/6J)."),
    ("obs", "tissue_code", "parsed from sample_id by script 01 regex", "Raw tissue abbreviation from the sample name (TA/GST/KDN)."),
    ("obs", "arm_code", "parsed from sample_id by script 01 regex", "Raw mission-arm abbreviation from the sample name (FLT/GC)."),
    ("obs", "replicate", "parsed from sample_id by script 01 regex", "Replicate number within arm (1-6). Note: NOT the animal number -- Rep1 maps to M23 in FLT but M33 in GC."),
    ("obs", "animal_id", "script 01: animal_prefix + parsed digits", "Individual animal identifier (M23-M38). The join key between RNA and protein assays of the same animal."),
    ("obs", "tissue", "script 01: TISSUE_MAP lookup on tissue_code", "Standardized tissue name. Use this, not tissue_code, for cross-study pooling."),
    ("obs", "arm", "script 01: ARM_MAP lookup, cross-checked against runsheet Factor Value[Spaceflight]",
     "Standardized mission arm: Flight or Ground.Control. This is the label for Task B classification."),
    ("obs", "osd_id", "script 01: STUDY_CONFIGS constant", "OSDR study accession (OSD-105/101/102)."),
    ("obs", "mission", "script 01: STUDY_CONFIGS constant (from ISA)", "Spaceflight mission (RR-1 = SpaceX CRS-4, 37-day round trip to ISS)."),
    ("obs", "spacecraft", "script 01: STUDY_CONFIGS constant (from ISA)", "Vehicle that returned the samples (SpaceX-4)."),
    ("obs", "preservation_method", "empty for OSD-105/101/102 (not recorded in ISA)", "Preservation token; populated only for studies like OSD-48 where it varies and acts as a confound."),
    ("obs", "tmt_plex", "script 04: ISA assay file run-identity column",
     "Which TMT plex the animal was labeled in (plex_1/plex_2), derived from the ISA run column. Batch covariate for protein values."),
    ("obs", "tmt_channel", "script 04: ISA assay file channel column",
     "The animal's TMT tag within its plex (126, 127N, ...). Channel 126 is reused across plexes, so channel alone does not identify an animal."),
    ("var", "ensembl_id", "count matrix row index (GLDS RSEM unnormalized counts)",
     "Mouse Ensembl gene ID. Primary key of var and of the expression matrix rows."),
    ("var", "mgi_symbol", "BioMart ortholog export: 'Gene name'",
     "Mouse gene symbol. Join key to the protein matrix's gene_symbol level."),
    ("var", "human_ortholog", "BioMart ortholog export: 'Human gene name'",
     "Human gene symbol(s), semicolon-joined when multiple. For cross-species comparison and readability."),
    ("var", "human_homology_type", "BioMart ortholog export",
     "Type of orthology (ortholog_one2one, ortholog_one2many, ...). Genes with no human ortholog (37,074 rows, mostly mt-/Rpl/Rps) are kept with NA."),
    ("var", "uniprot_accession", "TargetProtein.txt: Accession, collapsed per gene by script 02",
     "UniProt accession(s) of protein groups mapping to this gene, semicolon-joined. NA = gene was not detected in the proteome."),
    ("var", "n_protein_groups", "script 02 groupby count", "Number of protein groups (across both plexes) that mapped to this gene. >1 flags isoform ambiguity."),
    ("var", "ambiguous_protein_map", "script 02: n_protein_groups > 1", "Boolean flag: protein groups for this gene could not be resolved to a single accession."),
    ("expression_counts", "(columns = sample_id)", "GLDS RSEM Unnormalized Counts CSV",
     "Raw (unnormalized, unfiltered) gene-level counts. Rows = var.ensembl_id, columns = obs.sample_id. Normalization is deliberately left downstream."),
    ("protein_matrix", "(index = gene_symbol, tmt_plex)", "TargetProtein.txt ratio columns, via script 04",
     "TMT reporter-ion ratio to the 131 reference channel (a pool of all 12 animals -- so values are relative to the cohort mean, not to a control animal). One row per gene per plex: a gene quantified in both plexes has two rows, because plex is a batch effect."),
    ("protein_matrix", "(columns = sample_id)", "script 04: ISA channel map",
     "Same sample_ids as obs/expression. Values for the 6 samples NOT in that row's plex are structurally NA (that channel was not labeled in that plex)."),
]
ann = pd.DataFrame(COLUMN_ANNOTATIONS, columns=["table", "column_or_axis", "source", "description"])
ann.to_csv(OUT / f"{STUDY_ID}_column_annotations.csv", index=False)
adata.uns["column_provenance"] = ann.to_dict("list")
print(f"column annotations: {len(ann)} entries -> {STUDY_ID}_column_annotations.csv")