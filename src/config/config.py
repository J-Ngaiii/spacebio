# TRAINING
TRAINERS = {
    "logreg": {
        "config_module": "src.training.train_conf.logreg_config",
        "config_class": "logreg_train_config",
        "train_module": "src.training.train.logreg_train",
        "train_function": "train_logreg",
        "output_filename": "probe.pkl",
        "torchrun": False,
    },
    "simple-mlp": {
        "config_module": "src.training.train_conf.mlp_config",
        "config_class": "mlp_train_config",
        "train_module": "src.training.train.mlp_train",
        "train_function": "train_mlp",
        "output_filename": "mlp_results.pkl",
        "torchrun": True,
    },
}

# PREPROCESSING
# eveytime we add a dataset we instantiate a standardizer and source
BATCHPROCESSING_SETTINGS = {
    "raw": ("osd-105", "osd-101", "osd-102", "osd-137", "osd-48"),
    "interim": ("osd-105", "osd-101", "osd-102", "osd-137", "osd-48"),
}

def osdr_dataset_config(study_id: str, tissue: str) -> dict:
    return {
        "source": "nasa",
        "study_id": study_id,
        "standardizer": "osdr",
        "metadata": {
            "type": tissue,
            "raw": {
                "reader_dict": {
                    "filenames": (
                        f"{study_id}_expression_counts.csv",
                        f"{study_id}_obs.csv",
                        f"{study_id}_protein_matric.csv",
                        f"{study_id}_var.csv",
                    ),
                },
                "writer_dict": {
                    "filenames": "standardized.pkl",
                },
            },
            "interim": {
                "reader_dict": {
                    "filenames": "standardized.pkl",
                },
                "writer_dict": {
                    "filenames": "model_data.pkl",
                },
            },
        },
    }


DATASETS = {
    "osd-105": {
        "source": "nasa",
        "study_id": "OSD-105",
        "standardizer": "osdr",
        "metadata": {
            "type": "tibialis anterior",
            "raw": {
                "reader_dict": {
                    "filenames": (
                        "osd-105_expression_counts.csv",
                        "osd-105_obs.csv",
                        "osd-105_protein_matrix.csv",
                        "osd-105_var.csv",
                    ),
                },
                "writer_dict": {
                    "filenames": "standardized.pkl",
                },
            },
            "interim": {
                "reader_dict": {
                    "filenames": (
                        "OSD-105_obs.csv",
                        "OSD-105_var.csv",
                        "OSD-105_expression_counts.csv",
                        "OSD-105_protein_matrix.csv",
                    ),
                },
                "writer_dict": {
                    "filenames": "model_data.pkl",
                },
            },
        },
    },
    "osd-101": osdr_dataset_config("OSD-101", "gastrocnemius"),
    "osd-102": osdr_dataset_config("OSD-102", "kidney"),
    "osd-137": osdr_dataset_config("OSD-137", "liver"),
    "osd-48": osdr_dataset_config("OSD-48", "liver"),
}

STANDARDIZERS = {
    "osdr": "standardize_osdr",
    "archs4": "standardize_archs4",
}

MODELS = {
    "scgpt-lora": {
        "cleaner": "prep_scgpt_lora"
    },
    "scgpt-deepattn": {
        "cleaner": "prep_scgpt_deepattn"
    },
    "pca": {
        "cleaner": "prep_pca"
    },
    "single-hyena": {
        "cleaner": "prep_single_hyena"
    },
    "logreg": {
        "cleaner": "prep_logreg",
    },
    "simple-mlp": {
        "cleaner": "prep_simple_mlp"
    }
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