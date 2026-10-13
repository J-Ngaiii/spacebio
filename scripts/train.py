"""Central launcher for logistic regression and MLP training."""
import argparse
from dataclasses import fields
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.artifact import DataArtifact
from src.data.modelprep import ModelPrep
from src.data.read_writer import read_writer
from src.config.config import TRAINERS


# REFACTOR


# REFACTOR
def load_config(model_name: str, config_path: Path | None, overrides: list[str]):
    settings = TRAINERS[model_name]
    module = importlib.import_module(settings["config_module"]) # just rename the whole module and import everything
    config_class = getattr(module, settings["config_class"])

    options = {}
    if config_path is not None:
        with config_path.open("r", encoding="utf-8") as stream:
            options = json.load(stream)
        if not isinstance(options, dict):
            raise ValueError("The config file must contain a JSON object.")

    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator or not key:
            raise ValueError(f"Expected --set key=value, received {override!r}.")
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            options[key] = value

    allowed = {field.name for field in fields(config_class) if field.init}
    unknown = options.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {model_name} config fields: {sorted(unknown)}")

    return config_class(**options)


def run_pipeline(model_name: str, dataset: str, *, data_dir: Path | None = None, config=None) -> DataArtifact:
    settings = TRAINERS[model_name]
    if config is None:
        config = load_config(model_name, None, [])

    if settings["torchrun"]:
        device = torch.device(config.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)

    train_module = importlib.import_module(settings["train_module"]) # just rename the whole module and import everything
    train_function = getattr(train_module, settings["train_function"])

    source = #TODO
    processed: DataArtifact = ModelPrep.from_pretrained(source, model_name, input_filename=tuple(source.metadata["input_files"]), output_filename="assembled.pkl") # should not happen here should be in pipe
    adata = read_writer("train", source.name, action="read", artifact=None, filename="assembled.pkl", model_name=model_name) # not how read writer is used, 

    result = train_function(adata, config)
    output = read_writer("train", source.name, artifact=processed, data=result, filename=settings["output_filename"], model_name=model_name) # don't use readwriter, just read from processed directly using a different processor.

    print(f"[{source.metadata['study_id']}] Model: {model_name}")
    print(f"CV AUROC: {result['metrics']['cv_scores']}")
    print(f"Saved results: {output.path / settings['output_filename']}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(TRAINERS))
    parser.add_argument("--dataset", default="OSD-105")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--config", type=Path, help="JSON overrides for the model's dataclass defaults.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a config field; repeat as needed.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Prevent independent workers from overwriting the same output.
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("The current trainers support one process only; multi-GPU training requires DDP.")

    try:
        config = load_config(args.model, args.config, args.set)
    except (ValueError, TypeError) as error:
        parser.error(str(error))

    launched = args.worker or "LOCAL_RANK" in os.environ
    if TRAINERS[args.model]["torchrun"] and not launched:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=1", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        completed = subprocess.run(command, check=False)
        raise SystemExit(completed.returncode)

    run_pipeline(args.model, args.dataset, data_dir=args.data_dir, config=config)


if __name__ == "__main__":
    main()