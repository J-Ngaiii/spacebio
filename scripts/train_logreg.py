"""Train the RNA count probe from existing interim CSVs; no download or stage rebuild."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import pandas as pd
import pickle
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.config import DATASETS
from src.data.artifact import DataArtifact
from src.data.modelprep import ModelPrep
from src.data.read_writer import DATA_ROOT, PROJECT_ROOT, read_writer
from src.training.train.logreg_train import train_logreg
from src.training.train_conf.logreg_config import logreg_train_config
from src.utils import normalize_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_pipeline(dataset: str = "OSD-105", *, data_dir: Path | None = None, config: logreg_train_config | None = None) -> DataArtifact:
    input_path = PROJECT_ROOT / "data" / "processed" / "logreg" / normalize_name(dataset) / "model_data.pkl"
    with input_path.open("rb") as stream:
        adata = pickle.load(stream)
    result = train_logreg(adata, config)
    metrics = result["metrics"]
    print(f"[{dataset}] {metrics['n_samples']} samples x {metrics['n_genes']} genes; CV AUROC: {metrics['cv_scores']}")
    print("Raw RNA count schema sanity check; protein is retained but is not used for fitting.")
    # save results in predictions

    return 


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="OSD-105")
    parser.add_argument("--data-dir", type=Path, help="Override the existing interim study directory.")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(args.dataset, data_dir=args.data_dir, config=logreg_train_config(max_iter=args.max_iter, cv_folds=args.cv_folds, random_state=args.seed))


if __name__ == "__main__":
    main()
