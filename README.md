# SpaceBio

Classify mouse tissue samples as spaceflight (`Flight`) or ground control
(`Ground.Control`) from bulk RNA-seq, using NASA Open Science Data Repository
(OSDR) studies from the Rodent Research-1 mission. Paired proteomics is loaded
alongside RNA but is not yet used for training.

# Setup

Initialize a conda enviornment. This repository uses Python<3.12. 
```bash
conda create -n spacebio python=3.11.8 -y
conda activate spacebio
pip install -r requirements.txt
pip install -e .
```

# Data Pipeline Component

Jonathan's description of the data pipeline architecture below. 

## Conceptual Overview 

The data pipeline is designed to move datasets across three stages represented by three folders. Raw data which is just the base data itself and is housed within `data/raw`. Interim data which is data that has been subject to general cleaning and transformations but has not yet been formatted for a specific model and is housed within `data/interim`. Processed data which is data that is ready for specific models to operate on and is housed within `data/processed`. 

The actual code to move the data through those three stages is done by code that has two abstraction layers. One layer for actual data transformations and another layer on top for coordinating the application of different transformation functions and moving data throughout the repository as it gets cleaned.  

On the first layer are the actual data transformation functions that calculate means, normalize data, etc. These functions are housed in `src/data/preprocess.py`. Different parts of the pipeline import from this file to select and use the correct data transformation function at different stages of the pipeline. All functions in `preprocess.py` are dedicated to data transformation ONLY. Some important properties of the transformation functions in `preprocess.py` include: 
- They are pure functions that do not modify any global variables or states, they simply execute on the inputted data then output the data
- They operate entirely in runtime and do not execute any read or write calls to local storage whatsoever. 

The second layer of abstraction are the data preparer classes and DataArtifacts which are primarily dedicated to applying the correct transformation functions from `preprocess.py` and handling reads/writes to disk. First the `DataArtifact` class within `src/data/artifact.py` is a class that serves as a runtime object to track relevant metadata about a particular dataset. DataArtifacts track information about specific dataset CSVs like the processing stage the CSV is in throughout the pipeline and its path on disk. Generally as a rule of thumb we want to **instantiate only one `DataArtifact` instance per csv file**. - Note: Previous implementations of `DataArtifact` were not per CSV but per OSD dataset (eg. contained paths for all sub CSVs under OSD-105 like OSD-105_expression_counts.csv, OSD-105_obs.csv, etc). This was deprecated but some legacy code may still be operating on this scheme. Generally this scheme should be discouraged because it creates too much complexity. 

Next there are two data preparer classes: `DataPrep` within `src/data/dataprep.py` and `ModelPrep` within `src/data/modelprep.py`. These two classes are in charge of managing the application of data transformation functions, read/write operations and generally moving the data through the pipeline stages. They interact with the `DataArtifact` instances, configurations in `src/config/config.py` and the data transformation functions in `preprocess.py`. Generally both of these functions will:
- Read off data about a particular csv from a `DataArtifact` object
- Compare dataset information with `config.py` to find the correct data transformation functions for the given dataset at the current stage 
- Read the data into runtime and import the relevant data transformation functions specified in `config.py` to execute the transformations
- Write the transformed data to paths pulled from `config.py`
- Return an updated `DataArtifact` object. 

The specific function call is `DataPrep.from_raw(...)` and `ModelPrep.from_interim(...)`. These are class methods and do not require an instance of the data preparer classes to be first instantiated at runtime. 

`ModelPrep` is a little special because it also reads in information about a model and is the class responsible for creating specific folders and writing to the `data/processed/{model_name}/{actual_data}` path. 

Both `ModelPrep` and `DataPrep` have a batch mode that technically allows for taking in multiple CSVs at once but this may be buggy. 

For syntax on how to call the data preparer classes, check `scripts/run_pipeline.py` and refer to the 'Entry Point and Executing the Pipeline' Section below. 

## Helper Functions and Files

The `read_writer` function in read_write.py is a dedicated read/write function for the data pipeline ONLY. It is only used by the data pipelining functions to execute reads and writes throughout data processing. 

The `utils.py` file has utility functions like normalizing the naming. 

## Pipeline Naming Conventions

Earlier versions of the pipeline generally kept all names in lower case (eg osd-105 instead of OSD-105 or Osd-105). Some more recent code might break from this format but generally this should and can easily be adhered to by applying the `normalize_name` function in `utils.py`.

## Entry Point into Executing the Pipeline

The pipeline entry point file is `run_pipe.py`. It takes in four arguments via argparse from terminal.
- datasets: specify the dataset name or study id (eg OSD-105)
- models: specify the list of models to clean the data for (eg. ['logreg', 'mlp'])
- stage: specify the stage of data processing (eg. start from 'interim' data). Will start from scratch and begin with 'raw` data if no argument is given.
- path: the path of the dataset that needs to be flowed through the pipeline. Meant to take in only one folder per dataset currently (eg. OSD-105 folder with OSD-105_obs.csv, OSD-105_expression_counts.csv files within is OKAY). Might be a bit buggy on path intake. 

Example terminal call: 
```bash
python scripts/run_pipe.py --datasets OSD-105 --models logreg --stage interim --path data/interim/osd105
```
For a specific model like logistic regression, this call would check the interim folder for OSD-105 data, transform it into an `annData` object then save it as a .pkl file for the training script to load in the repository's Model Training Component (which is explained below).

## [IMPORTANT] Steps to Add New Datasets
- Write the relevant data transformation functions and put them in `preprocess.py`. Pay attention to how the specific functions are named in here because both data preparer classes.
- Update the configurations in `src/config/config.py`. 
- Input the write path when you run `scripts/run_pipeline.py`

That should largely be it. Instantiation of DataArtifacts happens in `run_pipeline.py`. Then the `DataPrep` and `ModelPrep` are executed by `run_pipeline.py`. As those functions execute they read off the specific `DataArtifact` and `config.py` file to find and execute the transformations in `preprocess.py`. 

## [IMPORTANT] Known Errors
- Batch mode in `DataPrep` and `ModelPrep` should be verified
- `run_pipeline.py` has only been tested on the interim stage for logistic regression and OSD-105. More tests and runs should be devised. 
- The `tests/test.py` file contains the main tests for the data pipeline component and is largely outdated. They should be updated. 
- The github workflow was previously set to run `tests/test.py` on push to the virtual repository but this part of the code has since been commented out since `test/test.py` fell out of date. 

## [IMPORTANT] To-dos: 
- `ModelPrep` and `DataPrep` should be able to take in and combine multiple OSD or other datasets. We will have to add another section in the `config.py` that specifies different sets of data we want to combine and how for a specific model. 
- Either `ModelPrep` or a new `TrainingPrep` class needs to handle executing a train test split. For deep learning pre-training or fine-tuning where we might want to specifically hold out data by certain biological covariates (eg. species, organs, etc) we need additional we need to have a class that handles that. 

# Model Training Component

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

Models registered in `MODELS`: `logreg` and `simple-mlp` are implemented
flight-vs-ground classifiers; `bulkformer` and `pca` predict protein from RNA;
`scgpt-lora`, `scgpt-deepattn`, and `single-hyena` are placeholders that
raise `NotImplementedError`. `pca` is the PCA + ridge baseline for `bulkformer`:
it uses the same genes, folds and metrics (`src/evaluation/evaluate.py`). Only OSD-105's config entry lists its interim CSV
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
