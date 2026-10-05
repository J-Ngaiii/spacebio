from pathlib import Path

from src.config.config import DATASETS, STANDARDIZERS
from src.data import preprocess
from src.data.artifact import DataArtifact
from src.data.read_writer import read_writer
from src.utils import normalize_name


class DataPrep:
    """Resolve a registered transformation and return its output artifact."""

    def __init__(self, *args, **kwargs):
        raise TypeError("Use DataPrep.from_pretrained(...).")

    @classmethod
    def from_pretrained(cls, artifact: DataArtifact, standardizer: str = "default", **kwargs) -> DataArtifact:
        if not isinstance(artifact, DataArtifact):
            raise TypeError("DataPrep requires a DataArtifact.")
        if artifact.stage != "raw":
            raise ValueError(f"DataPrep requires stage='raw', received {artifact.stage!r}.")
        
        name: str = normalize_name(artifact.name)
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name!r}")
        elif not Path(artifact.path).is_dir():
            raise FileNotFoundError(artifact.path)

        # identify the standardizer we want based on dataset name 
        standardizer: str = normalize_name(standardizer)
        if standardizer == "default":
            standardizer = normalize_name(DATASETS[name]["standardizer"])
        elif standardizer not in STANDARDIZERS:
            raise ValueError(f"Unknown standardizer: {standardizer!r}")

        # get the actual standardizer function
        transform: callable = getattr(preprocess, STANDARDIZERS[standardizer])
        output = read_writer("preprocess", name)
        transform(input_dir=Path(artifact.path), output_dir=output.path, **kwargs)
        output.metadata = {**artifact.metadata, "standardizer": standardizer}
        
        return output
