from collections.abc import Iterable
import Path

from src.config.config import BATCHPROCESSING_SETTINGS, DATASETS, MODELS
from src.data import preprocess
from src.data.artifact import DataArtifact
from src.data.read_writer import read_writer
from src.utils import normalize_name


class ModelPrep:
    """Prepare one interim artifact or a configured batch for a model."""

    def __init__(self, *args, **kwargs):
        raise TypeError("Use ModelPrep.from_pretrained(...) or ModelPrep.from_batch(...).")

    @classmethod
    def from_interim(cls, artifact: DataArtifact, model_type: str, *, input_filename: str | tuple[str, ...] | None = None, output_filename: str | None = None, **kwargs) -> DataArtifact:
        if not isinstance(artifact, DataArtifact):
            raise TypeError("ModelPrep requires a DataArtifact.")
        if artifact.stage != "interim":
            raise ValueError(f"ModelPrep requires stage='interim', received {artifact.stage!r}.")

        name = normalize_name(artifact.name)
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name!r}")

        model_type = normalize_name(model_type)
        if model_type not in MODELS:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        transform = getattr(preprocess, MODELS[model_type]["cleaner"])
        settings = artifact.metadata.get(artifact.stage, {})
        reader_options, writer_options = dict(settings.get("reader_dict", {})), dict(settings.get("writer_dict", {}))
        configured_input = reader_options.pop("filenames", "standardized.pkl")
        configured_output = writer_options.pop("filenames", "model_data.pkl")
        input_filename = configured_input if input_filename is None else input_filename
        base_output = configured_output if output_filename is None else output_filename
        output_filename = str(Path(model_type) / name / base_output) # validate that this works
        
        data = read_writer("model", name, action="read", artifact=artifact, filename=input_filename, model_name=model_type, read_writer_kwargs=reader_options)
        transformed_data = transform(data, **kwargs)
        output = read_writer("model", name, action="write", artifact=artifact, data=transformed_data, filename=output_filename, model_name=model_type, read_writer_kwargs=writer_options)

        return output

    @classmethod
    def from_batch(cls, artifacts: Iterable[DataArtifact], model_type: str, *, input_filename: str | tuple[str, ...] | None = None, output_filename: str | None = None, **kwargs) -> tuple[DataArtifact, ...]:
        return tuple(cls.from_interim(artifact, model_type=model_type, input_filename=input_filename, output_filename=output_filename, **kwargs) for artifact in artifacts)