import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.artifact import DataArtifact
from src.data.dataprep import DataPrep
from src.data.modelprep import ModelPrep
from src.data.read_writer import read_writer


def run_pipeline(dataset: str, stage:str , path:str, model_type: str):
    match stage:
        case None:
            # Run pipeline from scratch

            # Artifact exists to track metadata as we read, write and process data
            # Prerpocesing organizers just read directly from the Artifact then apply transformations
            # Less messey than having transformation functions pass metadata and dataframes in and out at the same time
            rawDatasetArtifact = DataArtifact(name=dataset, stage="raw", path=path) # runtime class to track metadata as we proces
            # DataPrep is an organizer class that then reads the metadata in an artifact, manages pulling from preprocessing functions in preprocess.py, 
            # applies the functions then writes the transformed outputs to interim, passing an updated DataArtifact out
            interimDatasetArtifact = DataPrep.from_raw(artifact=rawDatasetArtifact, standardizer="default") 
            # ModelPrep is also an organizer class tghat reads metadata in an artifact, applies transformations and output an udpated Artifact
            processedDatasetArtifact = ModelPrep.from_interim(artifact=interimDatasetArtifact, model_type=model_type) 
            # END. Train.py takes over to read from processed
        case "interim":
            # Run only the interim data stage

            interimDatasetArtifact = DataArtifact(name=dataset, stage="interim", path=path)
            processedDatasetArtifact = ModelPrep.from_interim(artifact=interimDatasetArtifact, model_type=model_type) 
            # END
        case _:
            raise ValueError(f"Unknown stage: {stage}")


    # Keep filenames unique when multiple studies share the model directory.
    processed = ModelPrep.from_pretrained(source, model_type, input_filename=tuple(source.metadata["input_files"]), output_filename=f"{name}_assembled.pkl")
    adata = read_writer("train", name, action="read", artifact=processed, filename=f"{name}_assembled.pkl", model_name=model_type)

    if "arm" not in adata.obs:
        raise ValueError("Sample metadata must contain the arm label.")
    if "arm" in adata.var_names or "sample_id" in adata.var_names:
        raise ValueError("Gene IDs conflict with reserved CSV columns: arm or sample_id.")

    table = adata.to_df()
    table.index.name = "sample_id"
    table["arm"] = adata.obs["arm"]

    output = read_writer("model", name, artifact=source, data=table, filename=f"{name}.csv", model_name=model_type, read_writer_kwargs={"index": True})
    print(f"Saved {output.path / f'{name}.csv'}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["OSD-105"])
    parser.add_argument("--models", nargs="+", choices=("logreg", "simple-mlp"), default=["logreg", "simple-mlp"])
    args = parser.parse_args()

    for dataset in args.datasets:
        for model_type in args.models:
            run_pipeline(dataset, model_type)


if __name__ == "__main__":
    main()