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
    "OSD-102": { 
        "pattern": re.compile(
            r"^(?P<organism_code>\w+)_(?P<strain>[\w-]+)_(?P<tissue_code>\w+)_"
            r"(?P<arm_code>\w+)_Rep(?P<replicate>\d+)_M(?P<animal_num>\d+)$"
        ),
        "animal_prefix": "M",
        "mission": "RR-1",
        "spacecraft": "SpaceX-4",
        "runsheet": "GLDS-102_rna_seq_bulkRNASeq_v1_runsheet.csv",
    },
    "OSD-137": {
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
    
    "OSD-100": {
        "pattern": re.compile(
            r"^(?P<organism_code>\w+)_(?P<strain>[\w-]+)_(?P<tissue_code>\w+)_"
            r"(?P<arm_code>\w+)_Rep(?P<replicate>\d+)_M(?P<animal_num>\d+)$"
        ),
        "animal_prefix": "M",
        "mission": "RR-1",
        "spacecraft": "SpaceX-4",
        "runsheet": "GLDS-100_rna_seq_bulkRNASeq_v2_runsheet.csv",
    },
    "OSD-194": {
        "pattern": re.compile(
            r"^(?P<organism_code>\w+)_(?P<strain>[\w-]+)_(?P<tissue_code>\w+)_"
            r"(?P<arm_code>\w+)_Rep(?P<replicate>\d+)_(?P<animal_code>[A-Z]+\d+)$"
        ),
        "animal_prefix": "",
        "mission": "RR-3",
        "spacecraft": "SpaceX-8",
        "runsheet": "GLDS-194_rna_seq_bulkRNASeq_v2_runsheet.csv",
        "confounds": ["strain BAL-TAL (BALB/c-derived) differs from the rest of the corpus"],
    },
    "OSD-238": {
        "pattern": re.compile(
            r"^(?P<organism_code>\w+)_(?P<strain>[\w-]+)_(?P<tissue_code>\w+)_"
            r"(?P<arm_code>\w+)_(?P<gravity>[\w-]+)_(?P<diet>\w+)_"
            r"Rep(?P<replicate>\d+)_(?P<animal_code>[A-Z]+\d+)$"
        ),
        "animal_prefix": "",
        "mission": "MHU-2",
        "spacecraft": "SpaceX-12",
        "runsheet": "GLDS-238_rna_seq_bulkRNASeq_v2_runsheet.csv",
        "confounds": ["diet arm (JC vs JCwFOS) is confounded with housing; VIV arm excluded from Task B"],
    },
    "OSD-239": {
        "pattern": re.compile(
            r"^(?P<organism_code>\w+)_(?P<strain>[\w-]+)_(?P<tissue_code>\w+)_"
            r"(?P<arm_code>\w+)_(?P<gravity>[\w-]+)_(?P<diet>\w+)_"
            r"Rep(?P<replicate>\d+)_(?P<animal_code>[A-Z]+\d+)$"
        ),
        "animal_prefix": "",
        "mission": "MHU-2",
        "spacecraft": "SpaceX-12",
        "runsheet": "GLDS-239_rna_seq_bulkRNASeq_v2_runsheet.csv",
        "confounds": ["diet arm (JC vs JCwFOS); VIV arm excluded from Task B"],
    },
    "OSD-243": {
        "pattern": re.compile(
            r"^(?P<organism_code>\w+)_(?P<strain>[\w-]+)_(?P<tissue_code>\w+)_"
            r"(?P<arm_code>\w+)_(?P<dissection>[A-Z-]+)_Rep(?P<replicate>\d+)_"
            r"(?P<animal_code>[A-Z]+\d+)$"
        ),
        "animal_prefix": "",
        "mission": "RR-6",
        "spacecraft": "SpaceX-13",
        "runsheet": "GLDS-243_rna_seq_bulkRNASeq_v2_runsheet.csv",
        "confounds": ["dissection condition (LAR vs ISS-T) tracks arm; keep as covariate"],
    },
    "OSD-244": {
        "pattern": re.compile(
            r"^(?P<organism_code>\w+)_(?P<strain>[\w-]+)_(?P<tissue_code>\w+)_"
            r"(?P<arm_code>\w+)_(?P<dissection>[A-Z-]+)_Rep(?P<replicate>\d+)_"
            r"(?P<animal_code>[A-Z]+\d+)$"
        ),
        "animal_prefix": "",
        "mission": "RR-6",
        "spacecraft": "SpaceX-13",
        "runsheet": "GLDS-244_rna_seq_bulkRNASeq_v2_runsheet.csv",
        "confounds": ["dissection condition (LAR vs ISS-T) tracks arm; keep as covariate"],
    },
    "OSD-254": {
        "pattern": re.compile(
            r"^(?P<organism_code>\w+)_(?P<strain>[\w-]+)_(?P<tissue_code>\w+)_"
            r"(?P<arm_code>\w+)_(?P<duration>[\w-]+)_Rep(?P<replicate>\d+)_"
            r"(?P<animal_code>[A-Z]+\d+)$"
        ),
        "animal_prefix": "",
        "mission": "RR-7",
        "spacecraft": "SpaceX-15",
        "runsheet": "GLDS-254_rna_seq_bulkRNASeq_v2_runsheet.csv",
        "confounds": ["two strains by design (C57BL/6J, C3H/HeJ); VIV arm excluded from Task B"],
    },
    "OSD-289": {
        "pattern": re.compile(
            r"^(?P<organism_code>\w+)_(?P<strain>[\w-]+)_(?P<tissue_code>\w+)_"
            r"(?P<mission_token>MHU\d+)_(?P<arm_code>\w+)_(?P<gravity>[\w-]+)_"
            r"Rep(?P<replicate>\d+)$"
        ),
        "animal_prefix": "",
        "mission": None,
        "spacecraft": "SpaceX-9 (MHU-1), SpaceX-12 (MHU-2)",
        "runsheet": "GLDS-289_rna_seq_bulkRNASeq_v2_runsheet.csv",
        "animal_from_source": re.compile(r"^MHU\d+_(?P<animal_code>.+)$"),
        "arm_source": "factor_value",
        "confounds": ["MHU-1: arm == strain (FLT=C57-6J, GC=C57-6CR) -- do not pool MHU-1 arms for strain-sensitive analyses"],
    },
    "OSD-397": {
        "pattern": re.compile(r"^(?P<animal_code>M\d+)$"),
        "animal_prefix": "",
        "mission": "RR-1-era (SpaceX-4 per ISA; RR number not stated)",
        "spacecraft": "SpaceX-4",
        "runsheet": "GLDS-397_rna_seq_bulkRNASeq_v2_runsheet.csv",
        "tissue_override": "retina",
        "confounds": ["arm not parseable from name -- taken from runsheet Factor Value (no independent cross-check)"],
    },
    "OSD-421": {
        "pattern": re.compile(
            r"^RR(?P<mission_num>\d+)_(?P<tissue_code>[\w-]+)_"
            r"(?P<arm_code>\w+)_(?P<animal_code>[A-Z]+\d+)$"
        ),
        "animal_prefix": "",
        "mission": None,  # per-sample, from mission_num
        "spacecraft": "SpaceX-12",
        "runsheet": "GLDS-421_rna_seq_bulkRNASeq_v2_runsheet.csv",
    },
    "OSD-515": {
        "pattern": re.compile(
            r"^RR(?P<mission_num>\d+)_(?P<tissue_code>[\w-]+)_"
            r"(?P<arm_code>\w+)_(?P<animal_code>[A-Z]+\d+)(?P<techrep>_techrep\d+)?$"
        ),
        "animal_prefix": "",
        "mission": None,  # per-sample, from mission_num
        "spacecraft": "SpaceX-21",
        "runsheet": "GLDS-515_rna_seq_bulkRNASeq_v2_runsheet.csv",
        "drop_techrep": True,
    },
    "OSD-666": {
        "pattern": re.compile(
            r"^RR(?P<mission_num>\d+)_(?P<tissue_code>[\w-]+)_"
            r"(?P<arm_code>\w+)_(?P<animal_code>[A-Z]+\d+)(?P<techrep>_techrep\d+)?$"
        ),
        "animal_prefix": "",
        "mission": None,  # per-sample, from mission_num
        "spacecraft": "SpaceX-21",
        "runsheet": "GLDS-605_rna_seq_bulkRNASeq_v2_runsheet.csv",
        "drop_techrep": True,
    },
}
STUDY_ID = sys.argv[1] if len(sys.argv) > 1 else "OSD-105"
if STUDY_ID not in STUDY_CONFIGS:
    raise KeyError(f"No STUDY_CONFIGS entry for {STUDY_ID} — add one, don't guess")
config = STUDY_CONFIGS[STUDY_ID]

runsheet = pd.read_csv(RAW / config["runsheet"])
obs = runsheet[["Sample Name", "organism", "has_ERCC", "Factor Value[Spaceflight]"]].copy()
obs = obs.rename(columns={
    "Sample Name": "sample_id",
    "has_ERCC": "has_ercc",
    "Factor Value[Spaceflight]": "factor_value_raw",
})

def parse_sample_name(name: str) -> dict:
    m = config["pattern"].match(name)
    if not m:
        raise ValueError(f"Sample name did not match expected pattern: {name!r}")
    return m.groupdict()

parsed = obs["sample_id"].apply(parse_sample_name).apply(pd.Series)
obs = pd.concat([obs, parsed], axis=1)

if "techrep" in obs.columns:
    n_drop = obs["techrep"].notna().sum()
    if config.get("drop_techrep"):
        print(f"NOTE: dropping {n_drop} technical-replicate rows (re-sequenced "
              f"libraries of the same animals): "
              f"{obs.loc[obs['techrep'].notna(), 'sample_id'].tolist()[:6]}...")
        obs = obs[obs["techrep"].isna()].copy()
    obs = obs.drop(columns=["techrep"])

if "animal_from_source" in config:
    src = runsheet.loc[obs.index, "Source Name"] if "Source Name" in runsheet.columns else None
    if src is None:
        raise ValueError("config needs animal_from_source but runsheet has no Source Name")
    ext = src.astype(str).str.extract(config["animal_from_source"])
    obs["animal_id"] = ext["animal_code"].values
    if obs["animal_id"].isna().any():
        raise ValueError("Source Name did not yield animal IDs -- check config regex")
elif "animal_code" in obs.columns:
    obs["animal_id"] = obs["animal_code"]
    obs = obs.drop(columns=["animal_code"])
else:
    obs["animal_id"] = config["animal_prefix"] + obs["animal_num"]
    obs = obs.drop(columns=["animal_num"])

PRESERVATION_MAP = {"I": "upon_euthanasia", "C": "carcass"}
DISSECTION_MAP = {"LAR": "upon_euthanasia", "ISS-T": "carcass_iss_transfer"}
pres_col = None
if "preservation" in obs.columns:
    pres_col = "preservation"
    pmap = PRESERVATION_MAP
elif "dissection" in obs.columns:
    pres_col = "dissection"
    pmap = DISSECTION_MAP
if pres_col:
    obs["preservation_method"] = obs[pres_col].map(pmap)
    obs = obs.drop(columns=[pres_col])
    if obs["preservation_method"].isna().any():
        raise ValueError(f"Unmapped {pres_col} token — extend the map, don't guess")
    factor_col = next((c for c in runsheet.columns if "Dissection Condition" in c), None)
    if factor_col:
        dc = runsheet.loc[obs.index, factor_col].map(
            {"Upon euthanasia": "upon_euthanasia", "Carcass": "carcass"}
        )
        lar_mask = obs["preservation_method"] == "upon_euthanasia"
        if (obs.loc[lar_mask, "preservation_method"].values != dc[lar_mask].values).any():
            raise ValueError(
                "Parsed dissection token disagrees with runsheet "
                "Factor Value[Dissection Condition] — look before proceeding"
            )

TISSUE_MAP = {
    "TA": "tibialis_anterior",
    "SOL": "soleus",
    "GST": "gastrocnemius",
    "KDN": "kidney",
    "LVR": "liver",
    "EYE": "eye",
    "LRTN": "retina_left",
    "RRTN": "retina_right",
    "DSKN": "skin_dorsal",
    "FSKN": "skin_femoral",
    "SKN": "skin",
    "TMS": "thymus",
    "R-Quad": "quadriceps_right",
}
ARM_MAP = {"FLT": "Flight", "GC": "Ground.Control", "VC": "Vivarium",
           "VIV": "Vivarium", "BC": "Basal", "BSL": "Basal"}

if "tissue_code" in obs.columns:
    obs["tissue"] = obs["tissue_code"].map(TISSUE_MAP)
else:
    obs["tissue"] = config["tissue_override"]  # OSD-397: names carry no tissue
unmapped_tissue = obs[obs["tissue"].isna()]["sample_id"].tolist()
if unmapped_tissue:
    codes = obs.loc[obs["tissue"].isna()].get(
        "tissue_code", pd.Series(config.get("tissue_override"), index=obs.index)
    ).unique()
    raise ValueError(f"Unmapped tissue codes {list(codes)} in {unmapped_tissue[:5]} — add to TISSUE_MAP")

FACTOR_VALUE_MAP = {
    "Space Flight": "Flight",
    "Ground Control": "Ground.Control",
    "Basal Control": "Basal",
    "Vivarium Control": "Vivarium",
}
obs["arm_from_factor_value"] = obs["factor_value_raw"].map(FACTOR_VALUE_MAP)
if config.get("arm_source") == "factor_value":
    disagree = obs[obs["arm_code"].map(ARM_MAP) != obs["arm_from_factor_value"]]
    if len(disagree):
        print(f"NOTE [{STUDY_ID}]: arm taken from runsheet Factor Value; "
              f"{len(disagree)} sample names disagree and are kept in 'arm_code':")
        print(disagree[["sample_id", "arm_code", "arm_from_factor_value"]].to_string())
    obs["arm"] = obs["arm_from_factor_value"]
elif "arm_code" in obs.columns:
    obs["arm"] = obs["arm_code"].map(ARM_MAP)
    unmapped_arm = obs[obs["arm"].isna()]["arm_code"].unique()
    if len(unmapped_arm):
        raise ValueError(f"Unmapped arm codes {list(unmapped_arm)} — add to ARM_MAP, don't guess")
    mismatches = obs[obs["arm"] != obs["arm_from_factor_value"]]
    if len(mismatches):
        raise ValueError(
            f"Regex-parsed arm disagrees with runsheet Factor Value for "
            f"{len(mismatches)} samples:\n"
            f"{mismatches[['sample_id', 'arm', 'arm_from_factor_value']]}"
        )
else:
    obs["arm"] = obs["arm_from_factor_value"]
    if obs["arm"].isna().any():
        raise ValueError("Factor Value[Spaceflight] has values outside FACTOR_VALUE_MAP")
if obs["arm_from_factor_value"].isna().any():
    raise ValueError("Factor Value[Spaceflight] has values outside FACTOR_VALUE_MAP")
obs = obs.drop(columns=["factor_value_raw", "arm_from_factor_value"])

obs["osd_id"] = STUDY_ID
if "mission_token" in obs.columns:
    obs["mission"] = obs["mission_token"].str.replace(r"MHU(\d+)", r"MHU-\1", regex=True)
    obs = obs.drop(columns=["mission_token"])
elif "mission_num" in obs.columns:
    obs["mission"] = "RR-" + obs["mission_num"]
    obs = obs.drop(columns=["mission_num"])
else:
    obs["mission"] = config["mission"]
obs["spacecraft"] = config["spacecraft"]

# Extra design tokens parsed from names (kept as covariates)
for tok, col in [("gravity", "gravity_condition"), ("diet", "diet"), ("duration", "flight_duration")]:
    if tok in obs.columns:
        obs[col] = obs[tok]
        obs = obs.drop(columns=[tok])

if "preservation_method" not in obs.columns:
    obs["preservation_method"] = pd.Series([pd.NA] * len(obs), dtype="object")
obs["tmt_plex"] = pd.Series([pd.NA] * len(obs), dtype="object")
obs["tmt_channel"] = pd.Series([pd.NA] * len(obs), dtype="object")

obs.to_csv(OUT / f"{STUDY_ID}_obs.csv", index=False)
print(f"[{STUDY_ID}] obs table: {obs.shape[0]} samples x {obs.shape[1]} columns")
print(obs[["sample_id", "tissue", "arm", "animal_id", "mission"]].head(3).to_string())
for note in config.get("confounds", []):
    print(f"CONFOUND NOTE [{STUDY_ID}]: {note}")
