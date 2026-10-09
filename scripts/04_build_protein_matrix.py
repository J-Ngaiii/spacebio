"""
Build the protein matrix for an OSD study. Linked by sample_id/animal_id.

Usage: python 04_build_protein_matrix.py [STUDY_ID]

CHANNEL->SAMPLE SOURCE:
The TMT-labels xlsx is a peptide-yield worksheet and doesn't
contain channel assignments. Refer to ISA archive's
proteomics assay file (a_OSD-<id>_protein-expression-profiling_*.txt):
  - the channel column ("Label" or "Labeled Extract Name", depending on
    study) gives each sample's TMT channel
  - the run-identity column ("Parameter Value[Run]" or "Comment[Run Number]")
    gives the plex.
Rows whose Sample Name contains Pool/Empty are dropped (reference channels,
pool channels, empty wells).

The 131 reference is a pool of the study's animals. Ratio-to-131 values are
relative to the cohort mean.

Protein tables are matched to plexes via the ISA raw-file stem when recorded
(OSD-137), else the run digit in the filename (OSD-105/101/102)
"""
import sys
import re
import zipfile
import pandas as pd
from pathlib import Path

STUDY_ID = sys.argv[1] if len(sys.argv) > 1 else "OSD-105"

RAW = Path(__file__).resolve().parent.parent / "raw"
OUT = Path(__file__).resolve().parent.parent / "output"

CHANNEL_RE = re.compile(r"^(126|127[NC]|128[NC]|129[NC]|130[NC]|131)$")

# --- Step 1: locate the ISA zip and its proteomics assay file ---
isa_zips = list(RAW.glob("*.zip"))
if len(isa_zips) != 1:
    raise FileNotFoundError(f"Expected exactly 1 zip (the ISA archive) in {RAW}, found {isa_zips}")
with zipfile.ZipFile(isa_zips[0]) as z:
    assay_names = [n for n in z.namelist() if "protein-expression-profiling" in n]
    if len(assay_names) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 protein-expression assay file inside {isa_zips[0].name}, "
            f"found {assay_names}"
        )
    isa = pd.read_csv(z.open(assay_names[0]), sep="\t")

# --- Step 2: channel & plex map from the ISA assay file ---
def find_channel_column(isa):
    for col in ("Label", "Labeled Extract Name"):
        if col in isa.columns:
            vals = isa[col].dropna().astype(str).str.strip()
            if vals.str.match(CHANNEL_RE).all():
                return col
    raise ValueError("No column with clean TMT channel values found in ISA assay")

def find_run_column(isa):
    run_cols = [c for c in isa.columns if "run" in c.lower()]
    if len(run_cols) != 1:
        raise ValueError(f"Expected exactly 1 run-identity column, found {run_cols}")
    return run_cols[0]

channel_col = find_channel_column(isa)
run_col = find_run_column(isa)
channel_map = (
    isa[["Sample Name", channel_col, run_col]]
    .dropna(subset=[channel_col])
    .rename(columns={channel_col: "tmt_channel", run_col: "isa_run"})
)
channel_map["tmt_channel"] = channel_map["tmt_channel"].astype(str).str.strip()
# Keep only real animal samples; pools/reference/empty are not observations.
channel_map = channel_map[~channel_map["Sample Name"].str.contains("Pool|Empty", case=False)]
channel_map["animal_id"] = channel_map["Sample Name"].str.extract(r"_([A-Z]+\d+)$")
if channel_map["animal_id"].isna().any():
    raise ValueError("ISA sample names did not yield animal IDs — check naming")
if ~channel_map["tmt_channel"].str.match(CHANNEL_RE).all():
    raise ValueError("Animal sample mapped to a non-channel value — check ISA")

channel_map["run_num"] = channel_map["isa_run"].astype(str).str.extract(r"(\d+)")
channel_map["tmt_plex"] = "plex_" + channel_map["run_num"]
if channel_map["tmt_plex"].isna().any():
    bad = channel_map[channel_map["tmt_plex"].isna()]["isa_run"].unique().tolist()
    raise ValueError(f"Unrecognized ISA run values (expected a run number): {bad}")
for plex, grp in channel_map.groupby("tmt_plex"):
    dupes = grp["tmt_channel"].duplicated()
    if dupes.any():
        raise ValueError(f"{plex}: channel assigned twice: {grp[dupes]['tmt_channel'].tolist()}")
n_plexes = channel_map["tmt_plex"].nunique()

# --- Step 3: map each TargetProtein table to its plex ---
# Every plex uses the same channel set. Two mechanisms, in order:
#   1. If the ISA records per-plex raw-file stems (OSD-137:
#      '..._G1_B3_Frac_*.raw' for run 1), match the processed filename
#      against the stem.
#   2. Else, read the run digit from the protein filename
#      ('..._TA_1_TargetProtein.txt' -> run 1). Verified for OSD-105/101/102.
protein_files = sorted(RAW.glob("*_TargetProtein.txt"))
protein_files = [p for p in protein_files if not p.name.endswith("_TargetProteinGroup.txt")]
if len(protein_files) != n_plexes:
    raise FileNotFoundError(
        f"Expected exactly {n_plexes} TargetProtein tables (one per plex) in {RAW}, "
        f"found {len(protein_files)}: {[p.name for p in protein_files]}"
    )

stem_col = next((c for c in isa.columns if "Raw Spectral Data File Name" in c), None)
run_to_stem = {}
if stem_col:
    stems = isa[["Sample Name", stem_col]].dropna()
    stems = stems[~stems["Sample Name"].str.contains("Pool|Empty", case=False)]
    stems["run_num"] = stems["Sample Name"].map(
        dict(zip(channel_map["Sample Name"], channel_map["run_num"]))
    )
    stems["stem"] = stems[stem_col].str.replace(r"_Frac.*\.raw$", "", regex=True)
    run_to_stem = dict(zip(stems["run_num"], stems["stem"]))

file_to_plex = {}
for path in protein_files:
    plex_num = None
    for run_num, stem in run_to_stem.items():
        if stem in path.name:
            plex_num = run_num
            break
    if plex_num is None:
        m = re.search(r"_(\d+)_TargetProtein\.txt$", path.name)
        if m:
            plex_num = m.group(1)
    if plex_num is None or f"plex_{plex_num}" not in set(channel_map["tmt_plex"]):
        raise ValueError(
            f"{path.name}: could not determine which plex this file quantifies — "
            f"no ISA raw-file stem match and no run digit in the filename"
        )
    file_to_plex[path] = f"plex_{plex_num}"
if sorted(file_to_plex.values()) != sorted(channel_map["tmt_plex"].unique()):
    raise ValueError(
        f"Plex-to-file assignment incomplete: {file_to_plex} vs plexes {sorted(channel_map['tmt_plex'].unique())}"
    )

# --- Step 4: build the protein matrix from the ratio columns ---
# Each plex file reports Ratios: (channel) / (131). Take the animal
# channels for that plex and rename them to sample_ids. Pools/reference are
# not carried as sample columns.
frames = []
for path, plex in file_to_plex.items():
    tab = pd.read_csv(path, sep="\t")
    tab["gene_symbol"] = tab["Description"].str.extract(r"GN=(\S+)")
    plex_map = channel_map[channel_map["tmt_plex"] == plex]
    chan_to_sample = dict(zip(plex_map["tmt_channel"], plex_map["Sample Name"]))
    cols = {}
    for chan, sample_id in chan_to_sample.items():
        chan_col = re.sub(r"([NC])$", r"_\1", chan)
        ratio_col = f"Ratios: ({chan_col}) / (131)"
        if ratio_col not in tab.columns:
            raise ValueError(f"{plex} ({path.name}): expected column {ratio_col!r} missing")
        cols[sample_id] = tab[ratio_col]
    sub = pd.DataFrame(cols)
    sub["gene_symbol"] = tab["gene_symbol"]
    sub["tmt_plex"] = plex
    frames.append(sub)

protein_long = pd.concat(frames, ignore_index=True)

# --- Step 5: collapse to one row per (gene, plex); plex is a batch covariate
# downstream, so a gene quantified in both plexes keeps two rows. ---
protein_matrix = protein_long.set_index(["gene_symbol", "tmt_plex"])

# --- Step 6: update obs with resolved plex/channel ---
obs = pd.read_csv(OUT / f"{STUDY_ID}_obs.csv")
obs = obs.drop(columns=["tmt_plex", "tmt_channel"], errors="ignore")
obs = obs.merge(
    channel_map[["Sample Name", "tmt_channel", "tmt_plex"]]
    .rename(columns={"Sample Name": "sample_id"}),
    on="sample_id", how="left",
)
obs.to_csv(OUT / f"{STUDY_ID}_obs.csv", index=False)

sample_columns = set(protein_long.columns) - {"gene_symbol", "tmt_plex"}
expected = set(obs["sample_id"])
if expected != sample_columns:
    raise ValueError(
        f"Sample ID mismatch between obs and protein matrix.\n"
        f"  In obs but not protein: {expected - sample_columns}\n"
        f"  In protein but not obs: {sample_columns - expected}"
    )

protein_matrix.to_csv(OUT / f"{STUDY_ID}_protein_matrix.csv")
print(f"[{STUDY_ID}] protein matrix: {protein_matrix.shape[0]} gene-plex rows "
      f"x {len(sample_columns)} sample columns")
print(protein_long.groupby("tmt_plex").size().rename("rows per plex").to_string())
print("obs updated with resolved tmt_plex/tmt_channel.")
print("Saved separately from expression matrix — linked by sample_id, not merged.")
