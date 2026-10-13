# SpaceBio

Classify mouse tissue samples as spaceflight (`Flight`) or ground control
(`Ground.Control`) from bulk RNA-seq, using NASA Open Science Data Repository
(OSDR) studies from the Rodent Research-1 mission. Paired proteomics is loaded
alongside RNA but is not yet used for training.

# Data

Each study is one tissue from the same RR-1 animal cohort:

| Study | Tissue | Samples | Arms | Per-sample protein |
| --- | --- | --- | --- | --- |
| OSD-105 | tibialis anterior | 12 | 6 Flight / 6 Ground.Control | yes |
| OSD-101 | gastrocnemius | 12 | 6 / 6 | yes |
| OSD-102 | kidney | 12 | 6 / 6 | yes |
| OSD-137 | liver | 18 | 6 / 6 / 6 Basal | yes |
| OSD-48 | liver | 14 | 7 / 7 | no (pool-level only) |

`data/interim/<study>/` holds one study as CSVs that together form an AnnData object:

| File | AnnData slot | Contents |
| --- | --- | --- |
| `*_obs.csv` | `.obs` | One row per sample: `sample_id`, `arm`, `animal_id`, `tissue`, TMT channel/plex |
| `*_var.csv` | `.var` | One row per gene: `ensembl_id`, mouse symbol, human ortholog, UniProt mapping |
| `*_expression_counts.csv` | `.X` | Genes x samples, raw RSEM counts (normalization is per model) |
| `*_protein_matrix.csv` | `.obsm["protein"]` | Protein x sample abundances |

`data/interim/merged/` holds all studies merged (RNA only). These files were
built from OSDR downloads by `scripts/01`–`06`, which expect their own
per-study `raw/` and `output/` folder layout; see the docstrings in each script.
`data/reference/` holds the mouse–human ortholog table.

# Setup

The repo targets Python 3.9–3.11:

```bash
conda create -n spacebio python=3.11.8 -y
conda activate spacebio
pip install -r requirements.txt
pip install -e .
```

# Usage

Run from the repository root.

```bash
# interim CSVs -> data/processed/<model>/<dataset>/model_data.pkl
python scripts/run_pipe.py --stage interim                   # all models, OSD-105
python scripts/run_pipe.py --stage interim --models simple-mlp

# train with stratified k-fold cross-validation; prints CV AUROC
python scripts/train_logreg.py
python scripts/train_mlp.py --wandb-mode disabled            # --help lists hyperparameters
```

`train_mlp.py` logs to Weights & Biases by default; use `--wandb-mode offline`
or `disabled` to run without a W&B login. With 12 samples per study, CV scores
are a pipeline check, not a benchmark.

## Running on Savio

`slurm/train_mlp.sbatch` prepares the data and trains the MLP on one GPU under
the `ic_cdss170fall` account. Each person runs it with their own Savio login.

One-time setup on a login node:

```bash
cd /global/scratch/users/$USER
git clone https://github.com/J-Ngaiii/spacebio.git && cd spacebio
module load anaconda3
conda create -n spacebio python=3.11 -y && conda activate spacebio
pip install -r requirements.txt && pip install -e .
```

Submit from the repository root:

```bash
sbatch slurm/train_mlp.sbatch                # defaults
sbatch slurm/train_mlp.sbatch --epochs 50    # extra arguments go to train_mlp.py
squeue -u $USER                              # job status
```

Output is written to `slurm-<jobid>.out` and results to
`data/predictions/osd-105/simple-mlp/mlp_results.pkl`.

- **Different account, partition, or GPU:** flags given to `sbatch` override the
  script, e.g. `sbatch --partition=<p> --gres=gpu:<type>:1 --cpus-per-task=<n> slurm/train_mlp.sbatch`.
  `sacctmgr -p show associations user=$USER` lists the accounts, partitions,
  and QoS available to you.
- **Different conda environment:** `SPACEBIO_ENV=<name> sbatch slurm/train_mlp.sbatch`.
- **W&B:** jobs log offline. To upload, run `wandb login` once on a login node,
  then `wandb sync wandb/offline-run-*`.

# Pipeline

Each stage returns a `DataArtifact`: a lightweight handle with `name`, `stage`,
`path`, and `metadata` that points to saved files rather than holding data.

| Stage | Entry point | Input | Saved output |
| --- | --- | --- | --- |
| Ingestion | `nasa.pull_dataset(...)` | Study accession from `DATASETS` | `data/raw/<dataset>/metadata.json` |
| Standardization | `DataPrep.from_raw(...)` | A raw artifact | `data/interim/<dataset>/standardized.pkl` |
| Model preparation | `ModelPrep.from_interim(...)` | An interim artifact | `data/processed/<model>/<dataset>/model_data.pkl` |
| Training | `scripts/train_<model>.py` | Processed `model_data.pkl` | MLP: `data/predictions/<dataset>/simple-mlp/mlp_results.pkl`; logreg: printed metrics only |

Current data enters at the interim stage. The NASA puller downloads study
metadata only, and the standardizers in `preprocess.py` are placeholders.

Models registered in `MODELS`: `logreg` and `simple-mlp` are implemented;
`pca`, `scgpt-lora`, `scgpt-deepattn`, and `single-hyena` are placeholders that
raise `NotImplementedError`. Only OSD-105's config entry lists its interim CSV
filenames, so `run_pipe.py` currently works for OSD-105 only.

## Code layout

- `src/config/config.py` registers datasets, standardizers, model prep functions,
  and trainers. Registry keys are lowercase; input is normalized with
  `normalize_name` from `src/utils.py`.
- `src/data/read_writer.py` owns file loading, directory creation, and
  serialization (`.csv` read; `.json`, `.txt`, `.pkl`, `.bin` read and write).
- `src/data/preprocess.py` contains transformations. They take loaded data and
  return transformed data without reading or writing files.
- `src/data/dataprep.py` and `src/data/modelprep.py` validate the input stage,
  resolve a registered transformation, and orchestrate read, transform, write.
- `src/models/` defines the models; `src/training/train/` holds the CV training
  loops and `src/training/train_conf/` their dataclass configs.

Only load trusted pickle files.

# Tests

```bash
python -m unittest tests.test_mlp -v
```

`tests/test.py` and `tests/test_logreg.py` predate the current API and need updating.
