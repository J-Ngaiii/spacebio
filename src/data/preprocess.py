from typing import Any


# --------------------------------------------------
# Dataset standardization
# --------------------------------------------------

def standardize_osdr(data: Any, **kwargs) -> Any:
    """Return a standardized multimodal representation of loaded OSDR data."""
    raise NotImplementedError("Implement OSDR standardization before running preprocessing.")


def standardize_archs4(data: Any, **kwargs) -> Any:
    """Return a standardized representation of loaded ARCHS4 data."""
    raise NotImplementedError("Implement ARCHS4 standardization before running preprocessing.")


# --------------------------------------------------
# Model-specific preprocessing
# --------------------------------------------------

def prep_scgpt_lora(data: Any, **kwargs) -> Any:
    """Return standardized data prepared for scGPT LoRA."""
    raise NotImplementedError("Implement scGPT LoRA preprocessing.")


def prep_scgpt_deepattn(data: Any, **kwargs) -> Any:
    """Return standardized data prepared for scGPT deep attention."""
    raise NotImplementedError("Implement scGPT deep attention preprocessing.")


def prep_pca(data: Any, **kwargs) -> Any:
    """Return standardized data prepared for PCA."""
    raise NotImplementedError("Implement PCA preprocessing.")


def prep_single_hyena(data: Any, **kwargs) -> Any:
    """Return standardized data prepared for single Hyena."""
    raise NotImplementedError("Implement single Hyena preprocessing.")