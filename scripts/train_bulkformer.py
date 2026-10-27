"""Fine-tune BulkFormer to predict protein from RNA and optionally track it with W&B."""
import argparse
from pathlib import Path
import sys
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train.bulkformer_train import train_bulkformer
from src.training.train_conf.bulkformer_config import bulkformer_train_config
from src.utils import normalize_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_pipeline(dataset: str = "OSD-105", *, data_dir: Path | None = None, config: bulkformer_train_config | None = None):
    input_path = PROJECT_ROOT / "data" / "processed" / "bulkformer" / normalize_name(dataset) / "model_data.pkl"
    with input_path.open("rb") as stream:
        adata = pickle.load(stream)
    result = train_bulkformer(adata, config)
    metrics = result["metrics"]
    print(f"[{dataset}] {adata.n_obs} samples x {adata.n_vars} genes; {metrics['vocab_coverage']:.1%} of BulkFormer's genes present")
    print(f"CV pooled Pearson: {metrics['cv_scores']}; OOF mean per-protein Pearson: {metrics['oof_mean_pearson']:.3f}")

    output_path = PROJECT_ROOT / "data" / "predictions" / normalize_name(dataset) / "bulkformer" / "bulkformer_results.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(result, stream)
    print(f"Saved results: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="OSD-105")
    parser.add_argument("--data-dir", type=Path)
    defaults = bulkformer_train_config()
    for name, kind in (("checkpoint", str), ("dropout", float), ("batch_size", int), ("epochs", int), ("lr", float), ("weight_decay", float), ("cv_folds", int), ("random_state", int), ("device", str), ("protein_key", str), ("wandb_project", str), ("wandb_entity", str), ("run_group", str)):
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=getattr(defaults, name))
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=defaults.freeze_backbone)
    parser.add_argument("--log2-protein", action=argparse.BooleanOptionalAction, default=defaults.log2_protein)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=defaults.wandb_mode)
    args = vars(parser.parse_args())
    dataset, data_dir = args.pop("dataset"), args.pop("data_dir")
    run_pipeline(dataset, data_dir=data_dir, config=bulkformer_train_config(**args))


if __name__ == "__main__":
    main()
