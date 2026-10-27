"""Train the PCA + ridge baseline to predict protein from RNA and optionally track it with W&B."""
import argparse
from pathlib import Path
import sys
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train.pca_train import train_pca
from src.training.train_conf.pca_config import pca_train_config
from src.utils import normalize_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_pipeline(dataset: str = "OSD-105", *, data_dir: Path | None = None, config: pca_train_config | None = None):
    input_path = PROJECT_ROOT / "data" / "processed" / "pca" / normalize_name(dataset) / "model_data.pkl"
    with input_path.open("rb") as stream:
        adata = pickle.load(stream)
    result = train_pca(adata, config)
    metrics = result["metrics"]
    print(f"[{dataset}] {adata.n_obs} samples x {adata.n_vars} genes; {result['architecture']['n_components']} components explain {result['architecture']['explained_variance']:.1%} of variance")
    print(f"CV pooled Pearson: {metrics['cv_scores']}; OOF mean per-protein Pearson: {metrics['oof_mean_pearson']:.3f}")

    output_path = PROJECT_ROOT / "data" / "predictions" / normalize_name(dataset) / "pca" / "pca_results.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(result, stream)
    print(f"Saved results: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="OSD-105")
    parser.add_argument("--data-dir", type=Path)
    defaults = pca_train_config()
    for name, kind in (("n_components", int), ("cv_folds", int), ("random_state", int), ("protein_key", str), ("wandb_project", str), ("wandb_entity", str), ("run_group", str)):
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=getattr(defaults, name))
    parser.add_argument("--alphas", type=float, nargs="+", default=list(defaults.alphas))
    parser.add_argument("--log2-protein", action=argparse.BooleanOptionalAction, default=defaults.log2_protein)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=defaults.wandb_mode)
    args = vars(parser.parse_args())
    dataset, data_dir = args.pop("dataset"), args.pop("data_dir")
    args["alphas"] = tuple(args["alphas"])
    run_pipeline(dataset, data_dir=data_dir, config=pca_train_config(**args))


if __name__ == "__main__":
    main()
