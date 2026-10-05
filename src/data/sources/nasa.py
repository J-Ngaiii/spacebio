from urllib.parse import quote

import requests

from src.config.config import DATASETS, OSDR_BASE_URL, OSDR_ENDPOINTS
from src.data.artifact import DataArtifact
from src.data.read_writer import read_writer
from src.utils import normalize_name


def _endpoint(name: str, **kwargs) -> str:
    """Construct an OSDR API URL."""
    encoded = {
        key: quote(str(value), safe="")
        for key, value in kwargs.items()
    }
    path = OSDR_ENDPOINTS[name].format(**encoded)
    return f"{OSDR_BASE_URL}{path}"


def _get(name: str, params=None, **kwargs):
    """GET an OSDR endpoint and return JSON."""

    url = _endpoint(name, **kwargs)
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_datasets():
    return _get("datasets")


def get_dataset(accession: str):
    return _get("dataset", accession=accession)


def get_assays(accession: str):
    return _get("dataset_assays", accession=accession)


def get_files(accession: str):
    return _get("dataset_files", accession=accession)


def get_samples(accession: str, assay: str):
    return _get("assay_samples", accession=accession, assay=assay)


def get_sample(accession: str, assay: str, sample: str):
    return _get("sample", accession=accession, assay=assay, sample=sample)


def pull_dataset(dataset_name: str) -> DataArtifact:
    """Fetch and save OSDR study metadata; biological files are separate downloads."""
    dataset_name = normalize_name(dataset_name)
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")
    
    config = DATASETS[dataset_name]
    if config["source"] != "nasa":
        raise ValueError(f"Dataset {dataset_name!r} is not a NASA dataset.")
    accession = config.get("accession", dataset_name.upper().replace("OSDR-", "OSD-", 1))
    data = get_dataset(accession)
    
    # use readwriter to instatiate the artifact then update its metadata
    artifact = read_writer("curl", dataset_name, data=data, filename="metadata.json")
    artifact.metadata = {"source": "nasa", "accession": accession, "files": ["metadata.json"]}
    return artifact
