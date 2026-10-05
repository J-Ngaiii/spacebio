from src.config.config import DATASETS, MODELS
from src.data import preprocess
from src.data.artifact import DataArtifact
from src.data.read_writer import read_writer
from src.utils import normalize_name


class ModelPrep:
    """Resolve a model transformation, load its input, and save its output."""

    def __init__(self, *args, **kwargs):
        raise TypeError("Use ModelPrep.from_pretrained(...).")

    @classmethod
    def from_pretrained(cls, artifact: DataArtifact, model_type: str, *, input_filename: str = "standardized.pkl", output_filename: str = "model_data.pkl", **kwargs) -> DataArtifact:
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

        transform = getattr(preprocess, MODELS[model_type])

        data = read_writer("model", name, action="read", artifact=artifact, filename=input_filename, model_name=model_type)
        transformed_data = transform(data, **kwargs)
        output = read_writer("model", name, action="write", artifact=artifact, data=transformed_data, filename=output_filename, model_name=model_type)

        return output