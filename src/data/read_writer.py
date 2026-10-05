from pathlib import Path
from typing import Any
import json
import pickle

from src.config.config import DATASETS, MODELS
from src.data.artifact import DataArtifact
from src.utils import normalize_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


def _dataset_dir(stage: str, dataset_name: str) -> Path:
    return DATA_ROOT / stage / dataset_name


def read_writer(operation: str, dataset_name: str, *, action: str = "write", data: Any = None, filename: str | None = None, model_name: str | None = None, artifact: DataArtifact | None = None) -> Any:
    """Read stage inputs or save stage outputs and return a DataArtifact."""
    operation = normalize_name(operation)
    action = normalize_name(action)
    dataset_name = normalize_name(dataset_name)

    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")
    if action not in {"read", "write"}:
        raise ValueError(f"Unknown action: {action!r}")
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("A plain filename is required.")

    metadata = {}

    match operation:
        case "curl":
            input_stage = None
            output_stage = "raw"
            output_dir = _dataset_dir(output_stage, dataset_name)

        case "preprocess":
            input_stage = "raw"
            output_stage = "interim"
            output_dir = _dataset_dir(output_stage, dataset_name)

        case "model":
            input_stage = "interim"
            output_stage = "processed"

            if model_name is None:
                raise ValueError("model_name required for model preprocessing.")

            model_name = normalize_name(model_name)
            if model_name not in MODELS:
                raise ValueError(f"Unknown model type: {model_name!r}")

            output_dir = _dataset_dir(output_stage, dataset_name) / model_name
            metadata["model_type"] = model_name

        case _:
            raise ValueError(f"Unknown operation: {operation!r}")

    if artifact is not None:
        if normalize_name(artifact.name) != dataset_name:
            raise ValueError("Artifact name does not match dataset_name.")
        if artifact.stage != input_stage:
            raise ValueError(f"Expected stage={input_stage!r}, received {artifact.stage!r}.")
        metadata = {**artifact.metadata, **metadata} # when we match to a model, we need to udpate the artifact metadata

    if action == "read":
        if input_stage is None:
            raise ValueError("'curl' supports writing only.")

        if artifact is not None:
            input_dir = Path(artifact.path) 
        else:
            input_dir = _dataset_dir(input_stage, dataset_name)
        return _read(input_dir / filename)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write(data, output_dir / filename)

    return DataArtifact(name=dataset_name, stage=output_stage, path=output_dir, metadata=metadata)


def _read(path: Path) -> Any:
    """Select a loader using the file extension."""
    match path.suffix.lower():
        case ".json":
            with path.open("r", encoding="utf-8") as stream:
                return json.load(stream)
        case ".txt":
            return path.read_text(encoding="utf-8")
        case ".pkl" | ".pickle":
            with path.open("rb") as stream:
                return pickle.load(stream)
        case ".bin":
            return path.read_bytes()
        case _:
            raise ValueError(f"No reader registered for extension: {path.suffix!r}")


def _write(data: Any, path: Path) -> None:
    """Select a serializer using the file extension."""
    match path.suffix.lower():
        case ".json":
            with path.open("w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2)
        case ".txt":
            if not isinstance(data, str):
                raise TypeError("Writing .txt requires a string.")
            path.write_text(data, encoding="utf-8")
        case ".pkl" | ".pickle":
            with path.open("wb") as stream:
                pickle.dump(data, stream)
        case ".bin":
            if not isinstance(data, bytes):
                raise TypeError("Writing .bin requires bytes.")
            path.write_bytes(data)
        case _:
            raise ValueError(f"No writer registered for extension: {path.suffix!r}")