
from dataclasses import dataclass, field


@dataclass
class PerformerConfig:
    task: str = "regression"
    return_predictions: bool = False

    input_dim: int | None = None
    output_dim: int | None = None
    input_channels: int = 1

    d_model: int = 128
    depth: int = 4
    heads: int = 4
    dropout: float = 0.1

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 15

    val_fraction: float = 0.15
    test_fraction: float = 0.15

    standardize_x: bool = True

    # NEW: For regression, standardize each protein target
    # using only its observed values in the training split.
    # NaN entries remain NaN and are excluded from the loss.
    standardize_y: bool = True

    # If non-empty, these sources are the test set
    # (whole-cohort holdout).
    # NOTE: Requires implementation in performer_train.py.
    holdout_sources: list[str] = field(default_factory=list)

    # Minimum observed values for a (source, protein)
    # pair to be used or scored.
    # NOTE: Requires implementation in performer_train.py.
    min_source_obs: int = 10

    seed: int = 42
    device: str = "auto"
