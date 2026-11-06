import pickle
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import save_file

from src.config.config import DATASETS
from src.data.artifact import DataArtifact
from src.data.modelprep import ModelPrep
from src.utils import normalize_name
from src.training.train.scgpt_lora_train import train_scgpt_lora
from src.training.train_conf.scgpt_lora_config import scgpt_lora_train_config

CKPT = Path("checkpoints/scGPT_human")
OUT = Path("data/predictions/osd-105/scgpt-lora")
OUT.mkdir(parents=True, exist_ok=True)

missing = [f for f in ("best_model.pt", "vocab.json", "args.json") if not (CKPT / f).exists()]
if missing:
    raise SystemExit(f"Missing checkpoint files in {CKPT}: {missing}")

name = normalize_name("OSD-105")
settings = DATASETS[name]
metadata = deepcopy(settings.get("metadata", {}))
metadata.update({"source": settings["source"], "study_id": settings["study_id"]})
interim = DataArtifact(name=name, stage="interim", path=Path("data/interim/osd105"), metadata=metadata)

processed = ModelPrep.from_interim(
    interim,
    model_type="scgpt-lora",
    vocab_path=str(CKPT / "vocab.json"),
    gene_symbol_col="human_ortholog",
)
p = Path(processed.path)
if p.is_dir():
    p = p / "model_data.pkl"
adata = pickle.load(open(p, "rb"))
print(adata.uns["scgpt_prep"])
print(adata.obs["arm"].value_counts())

cfg = scgpt_lora_train_config(
    checkpoint_dir=str(CKPT),
    task="classification",
    epochs=20, cv_folds=3, batch_size=4, use_lora=False,
    device="cpu", wandb_mode="disabled",
)

t0 = time.time()
result = train_scgpt_lora(adata, cfg)
print(f"training took {(time.time() - t0) / 60:.1f} min")
print(result["metrics"])

# --- save everything ---
save_file({k: v.contiguous().clone() for k, v in result["state_dict"].items()}, str(OUT / "scgpt_lora_weights.safetensors"))

preds = pd.DataFrame({
    "sample_id": result["sample_ids"],
    "arm": adata.obs["arm"].to_numpy(),
    "target": [t[0] for t in result["targets"]],
    "oof_prediction": result["oof_predictions"],
})
preds.to_csv(OUT / "oof_predictions.csv", index=False)

with open(OUT / "scgpt_lora_results.pkl", "wb") as f:
    pickle.dump(result, f)

print("saved to", OUT)
print(preds)
