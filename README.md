# ML Dev Template
This repository is a basic template for a machine learning model developement repo. 

# Setup
For Python version, this repo requires at least Python 3.9 and strictly less than Python 3.12. You can do this by creating a conda enviornment:
```bash
conda create -n spacebio python=3.11.8 -y
conda activate spacebio
pip install -r requirements.txt
pip install -e .
```

## Data processing

SpaceBio separates data retrieval, dataset standardization, and model-specific
preparation. Each stage returns a `DataArtifact`, a lightweight handle containing
`name`, `stage`, `path`, and `metadata`. The artifact points to saved files rather than holding the biological data itself.

### Pipeline stages

| Stage | Entry point | Input | Saved output |
| --- | --- | --- | --- |
| Ingestion | `nasa.pull_dataset(...)` | NASA study accession resolved from `DATASETS` | `data/raw/<dataset>/metadata.json` |
| Standardization | `DataPrep.from_pretrained(...)` | A raw artifact | `data/interim/<dataset>/standardized.pkl` |
| Model preparation | `ModelPrep.from_pretrained(...)` | An interim artifact | `data/processed/<dataset>/<model>/model_data.pkl` |

The NASA puller currently downloads study metadata. Downloading biological data
files such as expression matrices requires additional ingestion logic. ARCHS4 is
registered in configuration, but its source adapter still needs implementation.
Train/validation/test splitting is a planned subsequent stage.

### Responsibilities

- `src/config/config.py` registers dataset names, source accessions, standardizer
  names, and model preprocessing names. Registry keys are lowercase; user input
  is normalized with `normalize_name` from `src/utils.py`.
- `src/data/artifact.py` defines the shared `DataArtifact` handle.
- `src/data/sources/nasa.py` retrieves NASA metadata and passes it to the writer.
- `src/data/read_writer.py` owns pipeline file loading, directory creation, and
  serialization. `action="read"` loads the source file; `action="write"` saves a
  result and returns its artifact. Reads use the supplied artifact's path.
- `src/data/preprocess.py` contains transformations. Each function accepts loaded
  data and optional keyword arguments, then returns transformed data. These
  functions do not read or write files.
- `DataPrep` and `ModelPrep` validate the input stage, resolve a registered
  transformation, and orchestrate reading, transformation, and writing.

Supported serialization formats are JSON (`.json`), text (`.txt`), pickle
(`.pkl` or `.pickle`), and bytes (`.bin`). CSV and HDF5 require additional handlers.
Only load trusted pickle files.

### Example

Run this code from the repository root after implementing the selected
transformations in `src/data/preprocess.py`:

```python
from src.data.sources.nasa import pull_dataset
from src.data.dataprep import DataPrep
from src.data.modelprep import ModelPrep

raw = pull_dataset("osdr-114")
interim = DataPrep.from_pretrained(raw, standardizer="default")
processed = ModelPrep.from_pretrained(interim, model_type="pca")

print(processed.path)
print(processed.metadata)
```

`standardizer="default"` selects the dataset's configured standardizer. Currently
registered model keywords are `scgpt-lora`, `scgpt-deepattn`, `pca`, and
`single-hyena`. Transformation placeholders raise `NotImplementedError` until
implemented; registering a name does not implement the transformation.

Additional transformation options are passed through `**kwargs`. `ModelPrep`
also accepts `input_filename` and `output_filename` to override its default
`standardized.pkl` input and `model_data.pkl` output. Process multiple datasets
by passing one artifact at a time through the factories.

Output metadata preserves the source metadata and records the selected
`standardizer` or `model_type`. This supplies basic processing provenance without
mutating the input artifact. Metadata is currently held on the artifact in memory;
it is not automatically saved as a manifest alongside the data.

### Tests

Run the test suite from the repository root:

```bash
python tests/test.py
```