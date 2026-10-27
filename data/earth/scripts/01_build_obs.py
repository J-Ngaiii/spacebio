"""
Build the sample table (.obs) for the Earth aging cohort (GSE225576 RNA, PXD040722 TMT proteomics).
"""
import gzip

import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "raw"
OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)

# PRIDE metadata file -> protein result files it describes (16 TMT channels each)
METADATA_FILES = {
    "metadata_P2067.csv": ["Kidney_merged_results_20220530_2359_proteins.txt",
                           "Muscle_merged_results_20220531_1130_proteins.txt",
                           "Skin_merged_results_20220531_1312_proteins.txt"],
    "metadata_P2375.csv": ["liver_merged_results_20221213_2127_proteins.txt"],
}
TISSUE_FROM_FILE = {
    "Kidney_merged_results_20220530_2359_proteins.txt": "kidney",
    "Muscle_merged_results_20220531_1130_proteins.txt": "muscle",
    "Skin_merged_results_20220531_1312_proteins.txt": "skin",
    "liver_merged_results_20221213_2127_proteins.txt": "liver",
}

rows = []
for meta_name, result_files in METADATA_FILES.items():
    meta = pd.read_csv(RAW / meta_name)
    for result_file in result_files:
        sub = meta[meta["file"] == result_file]
        if len(sub) != 16:
            raise ValueError(
                f"{meta_name}: expected 16 channel rows for {result_file}, found {len(sub)}"
            )
        tissue = TISSUE_FROM_FILE[result_file]
        for _, r in sub.iterrows():
            age_months = int(r["condition"].removeprefix("t").removesuffix("mo"))
            replicate = int(r["rep"].removeprefix("rep"))
            rows.append({
                "sample_id": f"{tissue}_{age_months}M{replicate}",
                "tissue": tissue,
                "age_months": age_months,
                "replicate": replicate,
                "tmt_channel": r["col.name"].removeprefix("signal_sum_"),
                "source_rna": "GSE225576",
                "source_protein": "PXD040722",
            })

obs = pd.DataFrame(rows)
obs["is_anchor"] = obs["age_months"] == 6
obs["has_protein"] = True

for tissue in ["Kidney", "Liver", "Muscle", "Skin"]:
    with gzip.open(RAW / f"GSE225576_{tissue}.txt.gz", "rt") as f:
        header = f.readline().rstrip("\r").split("\t")
    rna_ids = {f"{tissue.lower()}_{c}" for c in header[1:]}
    expected = {f"{tissue.lower()}_{a}M{r}" for a in [6, 15, 24, 30] for r in [1, 2, 3, 4]}
    if rna_ids != expected:
        raise ValueError(f"{tissue}: RNA columns {sorted(rna_ids)} != expected {sorted(expected)}")

if len(obs) != 64 or obs["sample_id"].duplicated().any():
    raise ValueError(f"obs build broken: {len(obs)} rows, "
                     f"{obs['sample_id'].duplicated().sum()} duplicate ids")

obs = obs.sort_values(["tissue", "age_months", "replicate"]).set_index("sample_id")
obs.to_csv(OUT / "Earth_obs.csv")
print(f"[Earth] obs table: {len(obs)} samples x {obs.shape[1]} columns")
print(f"  tissues: {obs['tissue'].value_counts().to_dict()}")
print(f"  ages: {obs['age_months'].value_counts().sort_index().to_dict()}")
print(f"  6-month samples: {obs['is_anchor'].sum()}")