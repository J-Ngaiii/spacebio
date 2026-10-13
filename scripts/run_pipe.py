import argparse
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.config import DATASETS, MODELS
from src.data.artifact import DataArtifact
from src.data.dataprep import DataPrep
from src.data.modelprep import ModelPrep
from src.data.read_writer import DATA_ROOT
from src.utils import normalize_name


def run_pipeline(dataset: str, stage: str | None, path: str | Path, model_type: str) -> DataArtifact:
    dataset = normalize_name(dataset)
    model_type = normalize_name(model_type)
    stage = normalize_name(stage) if stage is not None else None
    path = Path(path)

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    if model_type not in MODELS:
        raise ValueError(f"Unknown model type: {model_type!r}")
    if stage not in {None, "raw", "interim"}:
        raise ValueError(f"Unknown stage: {stage!r}")
    if not path.is_dir():
        raise FileNotFoundError(path)

    settings = DATASETS[dataset]
    metadata = deepcopy(settings.get("metadata", {}))
    metadata.update({"source": settings["source"], "study_id": settings["study_id"]})

    if stage is None or stage == "raw":
        raw_artifact = DataArtifact(name=dataset, stage="raw", path=path, metadata=metadata)
        interim_artifact = DataPrep.from_raw(raw_artifact, standardizer="default")
    if stage == "interim":
        interim_artifact = DataArtifact(name=dataset, stage="interim", path=path, metadata=metadata)
    
    processed_artifact = ModelPrep.from_interim(interim_artifact, model_type=model_type)
    print(f"[{dataset}] Prepared {model_type}: {processed_artifact.path}")
    return processed_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["OSD-105"])
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=["logreg", "simple-mlp"])
    parser.add_argument("--stage", choices=("raw", "interim"), default=None, help="Starting stage; defaults to raw.")
    parser.add_argument("--path", type=Path, help="Input dataset directory; requires exactly one dataset.")
    args = parser.parse_args()

    if args.path is not None and len(args.datasets) != 1:
        parser.error("--path requires exactly one dataset.")

    for dataset in args.datasets:
        name = normalize_name(dataset)
        if name not in DATASETS:
            parser.error(f"Unknown dataset: {dataset!r}")

        input_path = args.path
        if input_path is None:
            input_stage = args.stage or "raw"
            study_id = DATASETS[name]["study_id"]
            candidates = [DATA_ROOT / input_stage / folder for folder in (study_id, name, name.replace("-", ""))]
            input_path = next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])

        for model_type in args.models:
            run_pipeline(dataset, args.stage, input_path, model_type)


if __name__ == "__main__":
    main()