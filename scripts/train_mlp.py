"""Train an MLP on existing interim RNA counts and optionally track it with W&B."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.modelprep import ModelPrep
from src.data.read_writer import read_writer
from src.training.train.mlp_train import train_mlp
from src.training.train_conf.mlp_config import mlp_train_config

def run_pipeline(dataset: str = "OSD-105", *, data_dir: Path | None = None, config: mlp_train_config | None = None):
    source = #TODO
    processed = ModelPrep.from_pretrained(source, "simple-mlp", input_filename=tuple(source.metadata["input_files"]), output_filename="assembled.pkl")
    adata = read_writer("train", source.name, action="read", artifact=processed, filename="assembled.pkl", model_name="simple-mlp")
    result = train_mlp(adata, config)
    output = read_writer("train", source.name, artifact=processed, data=result, filename="mlp_results.pkl", model_name="simple-mlp")
    print(f"CV AUROC: {result['metrics']['cv_scores']}; saved {output.path / 'mlp_results.pkl'}")
    print("RNA-only schema probe; these small-sample scores are not a benchmark.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="OSD-105")
    parser.add_argument("--data-dir", type=Path)
    defaults = mlp_train_config()
    for name, kind in (("hidden_dim", int), ("dropout", float), ("batch_size", int), ("epochs", int), ("lr", float), ("weight_decay", float), ("cv_folds", int), ("random_state", int), ("device", str), ("wandb_project", str), ("wandb_entity", str), ("run_group", str)):
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=getattr(defaults, name))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=defaults.wandb_mode)
    args = vars(parser.parse_args())
    dataset, data_dir = args.pop("dataset"), args.pop("data_dir")
    run_pipeline(dataset, data_dir=data_dir, config=mlp_train_config(**args))


if __name__ == "__main__":
    main()
