"""
Build the sample table (.obs) for an OSD study. One row per animal x tissue x assay.
Usage: python 01_build_obs.py [STUDY_ID]

"""
import sys
import pandas as pd
import re
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "raw"
OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)

STUDY_CONFIGS = {
    "OSD-105": {
        "pattern": re.compile(
            r"^(?P<organism_code>\w+)_(?P<strain>[\w-]+)_(?P<tissue_code>\w+)_"
            r"(?P<arm_code>\w+)_Rep(?P<replicate>\d+)_M(?P<animal_num>\d+)$"
        ),
        "animal_prefix": "M",  # animal_id = prefix + animal_num
        "mission": "RR-1",
        "spacecraft": "SpaceX-4",
        "runsheet": "GLDS-105_rna_seq_bulkRNASeq_v2_runsheet.csv",
    },
    "OSD-101": {  # gastrocnemius, same RR-1 cohort (M23-M38) as OSD-105
        "pattern": re.compile(
            r"^(?P<organism_code>\w+)_(?P<strain>[\w-]+)_(?P<tissue_code>\w+)_"
            r"(?P<arm_code>\w+)_Rep(?P<replicate>\d+)_M(?P<animal_num>\d+)$"
        ),
        "animal_prefix": "M",
        "mission": "RR-1",
        "spacecraft": "SpaceX-4",
        "runsheet": "GLDS-101_rna_seq_bulkRNASeq_v1_runsheet.csv",
    },
    "OSD-102": {  # kidney, same RR-1 cohort (M23-M38) as OSD-105
        "pattern": re.compile(
            r"^(?P<organism_code>\w+)_(?P<strain>[\w-]+)_(?P<tissue_code>\w+)_"
            r"(?P<arm_code>\w+)_Rep(?P<replicate>\d+)_M(?P<animal_num>\d+)$"
        ),
        "animal_prefix": "M",
        "mission": "RR-1",
        "spacecraft": "SpaceX-4",
        "runsheet": "GLDS-102_rna_seq_bulkRNASeq_v1_runsheet.csv",
    },
    "OSD-137": {  # liver, RR-3, 3 arms; animal codes embed the arm letter
        # (B*=Basal, G*=GC, F*=FLT) and are NOT contiguous (no B5, no G4).
        "pattern": re.compile(
            r"^(?P<organism_code>\w+)_(?P<strain>[\w-]+)_(?P<tissue_code>\w+)_"
            r"(?P<arm_code>\w+)_Rep(?P<replicate>\d+)_(?P<animal_code>[A-Z]+\d+)$"
        ),
        "animal_prefix": "",  # animal_id = full captured code (B1, G5, F6, ...)
        "mission": "RR-3",
        "spacecraft": "SpaceX-8",
        "runsheet": "GLDS-137_rna_seq_bulkRNASeq_v2_runsheet.csv",
    },
    "OSD-48": {
        "pattern": re.compile(
            r"^(?P<organism_code>\w+)_(?P<strain>[\w-]+)_(?P<tissue_code>\w+)_"
            r"(?P<arm_code>\w+)_(?P<preservation>[IC])_Rep(?P<replicate>\d+)_"
            r"M(?P<animal_num>\d+)$"
        ),
        "animal_prefix": "M",
        "mission": "RR-1",
        "spacecraft": "SpaceX-4",
        "runsheet": "GLDS-48_rna_seq_bulkRNASeq_v2_runsheet.csv",
    },
}
STUDY_ID = sys.argv[1] if len(sys.argv) > 1 else "OSD-105"
if STUDY_ID not in STUDY_CONFIGS:
    raise KeyError(f"No STUDY_CONFIGS entry for {STUDY_ID} — add one, don't guess")
config = STUDY_CONFIGS[STUDY_ID]

# --- Step 1: runsheet---
runsheet = pd.read_csv(RAW / config["runsheet"])
obs = runsheet[["Sample Name", "organism", "has_ERCC", "Factor Value[Spaceflight]"]].copy()
obs = obs.rename(columns={
    "Sample Name": "sample_id",
    "has_ERCC": "has_ercc",
    "Factor Value[Spaceflight]": "factor_value_raw",
})

# --- Step 2: parse the structured sample name using this study's config ---
def parse_sample_name(name: str) -> dict:
    m = config["pattern"].match(name)
    if not m:
        raise ValueError(f"Sample name did not match expected pattern: {name!r}")
    return m.groupdict()

parsed = obs["sample_id"].apply(parse_sample_name).apply(pd.Series)
obs = pd.concat([obs, parsed], axis=1)

# Animal ID as the join key between RNA and protein assays.
if "animal_code" in obs.columns:
    obs["animal_id"] = obs["animal_code"]
    obs = obs.drop(columns=["animal_code"])
else:
    obs["animal_id"] = config["animal_prefix"] + obs["animal_num"]
    obs = obs.drop(columns=["animal_num"])

# Preservation token (OSD-48)
PRESERVATION_MAP = {"I": "upon_euthanasia", "C": "carcass"}
if "preservation" in obs.columns:
    obs["preservation_method"] = obs["preservation"].map(PRESERVATION_MAP)
    obs = obs.drop(columns=["preservation"])
    if obs["preservation_method"].isna().any():
        raise ValueError("Unmapped preservation token — extend PRESERVATION_MAP")
    if "Factor Value[Dissection Condition]" in runsheet.columns:
        dc = runsheet["Factor Value[Dissection Condition]"].map(
            {"Upon euthanasia": "upon_euthanasia", "Carcass": "carcass"}
        )
        if (obs["preservation_method"] != dc).any():
            raise ValueError(
                "Parsed preservation token disagrees with runsheet "
                "Factor Value[Dissection Condition] — look before proceeding"
            )

# --- Step 3: standardize tissue and arm names ---
TISSUE_MAP = {
    "TA": "tibialis_anterior",
    "SOL": "soleus",
    "GST": "gastrocnemius",
    "KDN": "kidney",
    "LVR": "liver",
}
ARM_MAP = {"FLT": "Flight", "GC": "Ground.Control", "VC": "Vivarium", "BC": "Basal", "BSL": "Basal"}

obs["tissue"] = obs["tissue_code"].map(TISSUE_MAP)
obs["arm"] = obs["arm_code"].map(ARM_MAP)

unmapped_tissue = obs[obs["tissue"].isna()]["tissue_code"].unique()
unmapped_arm = obs[obs["arm"].isna()]["arm_code"].unique()
if len(unmapped_tissue) or len(unmapped_arm):
    raise ValueError(
        f"Unmapped codes found — add to the lookup table, don't guess.\n"
        f"  tissue_code: {list(unmapped_tissue)}\n"
        f"  arm_code: {list(unmapped_arm)}"
    )

FACTOR_VALUE_MAP = {
    "Space Flight": "Flight",
    "Ground Control": "Ground.Control",
    "Basal Control": "Basal",
}
obs["arm_from_factor_value"] = obs["factor_value_raw"].map(FACTOR_VALUE_MAP)
mismatches = obs[obs["arm"] != obs["arm_from_factor_value"]]
if len(mismatches):
    raise ValueError(
        f"Regex-parsed arm disagrees with runsheet Factor Value for "
        f"{len(mismatches)} samples:\n{mismatches[['sample_id', 'arm', 'arm_from_factor_value']]}"
    )
obs = obs.drop(columns=["factor_value_raw", "arm_from_factor_value"])

# --- Step 4: study-level constants, pulled from this study's config (ISA.zip)---
obs["osd_id"] = STUDY_ID
obs["mission"] = config["mission"]
obs["spacecraft"] = config["spacecraft"]

# --- Step 5: flag fields not parsed above ---
if "preservation_method" not in obs.columns:
    obs["preservation_method"] = pd.Series([pd.NA] * len(obs), dtype="object")
obs["tmt_plex"] = pd.Series([pd.NA] * len(obs), dtype="object")
obs["tmt_channel"] = pd.Series([pd.NA] * len(obs), dtype="object")

obs.to_csv(OUT / f"{STUDY_ID}_obs.csv", index=False)
print(f"[{STUDY_ID}] obs table: {obs.shape[0]} samples x {obs.shape[1]} columns")
print(obs[["sample_id", "tissue", "arm", "animal_id", "mission"]])
