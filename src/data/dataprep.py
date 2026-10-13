from collections.abc import Iterable
from pathlib import Path

from src.config.config import BATCHPROCESSING_SETTINGS, DATASETS, STANDARDIZERS
from src.data import preprocess
from src.data.artifact import DataArtifact
from src.data.read_writer import read_writer
from src.utils import normalize_name


class DataPrep:
    """Standardize one raw artifact or a configured batch of raw artifacts."""

    def __init__(self, *args, **kwargs):
        raise TypeError("Use DataPrep.from_raw(...) or DataPrep.from_batch(...).")

    @classmethod
    def from_raw(cls, artifact: DataArtifact, standardizer: str = "default", **kwargs) -> DataArtifact:
        if not isinstance(artifact, DataArtifact):
            raise TypeError("DataPrep requires a DataArtifact.")
        if artifact.stage != "raw":
            raise ValueError(f"DataPrep requires stage='raw', received {artifact.stage!r}.")

        name = normalize_name(artifact.name)
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name!r}")
        if not Path(artifact.path).is_dir():
            raise FileNotFoundError(artifact.path)

        standardizer = normalize_name(standardizer)
        if standardizer == "default":
            standardizer = normalize_name(DATASETS[name]["standardizer"])
        if standardizer not in STANDARDIZERS:
            raise ValueError(f"Unknown standardizer: {standardizer!r}")

        transform = getattr(preprocess, STANDARDIZERS[standardizer])
        settings = artifact.metadata.get(artifact.stage, {})
        reader_options, writer_options =  dict(settings.get("reader_dict", {})), dict(settings.get("writer_dict", {}))
        input_filename = reader_options.pop("filenames", "metadata.json")
        output_filename = writer_options.pop("filenames", "standardized.pkl")

        raw_data = read_writer("preprocess", name, action="read", artifact=artifact, filename=input_filename, read_writer_kwargs=reader_options)
        transformed_data = transform(raw_data, **kwargs)
        output = read_writer("preprocess", name, action="write", artifact=artifact, data=transformed_data, filename=output_filename, read_writer_kwargs=writer_options)
        output.metadata["standardizer"] = standardizer

        return output

    @classmethod
    def from_batch(cls, artifacts: Iterable[DataArtifact], standardizer: str = "default", **kwargs) -> tuple[DataArtifact, ...]:
        return tuple(cls.from_raw(artifact, standardizer=standardizer, **kwargs) for artifact in artifacts)