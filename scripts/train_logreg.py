"""Train the RNA count probe from existing interim CSVs; no download or stage rebuild."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.config import DATASETS
from src.data.artifact import DataArtifact
from src.data.modelprep import ModelPrep
from src.data.read_writer import DATA_ROOT, read_writer
from src.training.train.logreg_train import train_logreg
from src.training.train_conf.logreg_config import logreg_train_config
from src.utils import normalize_name


def run_pipeline(dataset: str = "OSD-105", *, data_dir: Path | None = None, config: logreg_train_config | None = None) -> DataArtifact:
    name = normalize_name(dataset)
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    study_id = DATASETS[name]["study_id"]
    if data_dir is None:
        candidates = [DATA_ROOT / "interim" / folder for folder in (study_id, name, name.replace("-", ""))]
        data_dir = next((path for path in candidates if path.is_dir()), candidates[0])
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)

    # Retain actual filenames, including OSD_105 prefixes and expressioncount spelling.
    suffixes = (("_obs.csv",), ("_var.csv",), ("_expression_counts.csv", "_expressioncount.csv"))
    filenames = []
    prefixes = (study_id, study_id.replace("-", "_"))
    for endings in suffixes:
        matches = [path.name for path in data_dir.glob("*.csv") if path.name.startswith(prefixes) and path.name.endswith(endings)]
        if len(matches) != 1:
            raise ValueError(f"Expected one {endings} table in {data_dir}, found {matches}.")
        filenames.extend(matches)
    proteins = [path.name for path in data_dir.glob("*.csv") if path.name.startswith(prefixes) and path.name.endswith(("_protein_matrix.csv", "_protein_matric.csv"))]
    if len(proteins) > 1:
        raise ValueError(f"Multiple protein matrices: {proteins}")
    filenames.extend(proteins)

    
    adata = pd.read_csv(f"data/processed/logreg/{dataset}")
    result = train_logreg(adata, config)
    metrics = result["metrics"]
    print(f"[{study_id}] {metrics['n_samples']} samples x {metrics['n_genes']} genes; CV AUROC: {metrics['cv_scores']}")
    print("Raw RNA count schema sanity check; protein is retained but is not used for fitting.")
    # save results in predictions

    return output


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
