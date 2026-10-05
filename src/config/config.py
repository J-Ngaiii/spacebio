DATASETS = {
    "osdr-114": {
        "source": "nasa",
        "standardizer": "osdr",
    },
    "archs4-a": {
        "source": "archs4",
        "standardizer": "archs4",
    },
}
STANDARDIZERS = {
    "osdr": "standardize_osdr",
    "archs4": "standardize_archs4",
}
MODELS = {
    "scgpt-lora": "prep_scgpt_lora",
    "scgpt-deepattn": "prep_scgpt_deepattn",
    "pca": "prep_pca",
    "single-hyena": "prep_single_hyena",
}

# Phase 0: Curling Raw Data from APIs

# OSDR
OSDR_BASE_URL = "https://visualization.osdr.nasa.gov/biodata/api/v2"

OSDR_ENDPOINTS = {
    # REST traversal
    "datasets": "/datasets/",
    "dataset": "/dataset/{accession}/",
    "dataset_assays": "/dataset/{accession}/assays/",
    "dataset_files": "/dataset/{accession}/files/",

    "assay": "/dataset/{accession}/assay/{assay}/",
    "assay_samples": "/dataset/{accession}/assay/{assay}/samples/",
    "assay_files": "/dataset/{accession}/assay/{assay}/files/",

    "sample": (
        "/dataset/{accession}/assay/{assay}/"
        "sample/{sample}/"
    ),
    "sample_files": (
        "/dataset/{accession}/assay/{assay}/"
        "sample/{sample}/files/"
    ),

    "file": "/dataset/{accession}/file/{filename}/",

    # Query interface
    "query_metadata": "/query/metadata/",
    "query_samples": "/query/samples/",
    "query_assays": "/query/assays/",
    "query_data": "/query/data/",
}