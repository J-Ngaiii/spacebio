
from dataclasses import dataclass


@dataclass
class HyenaConfig:
    task: str = "regression"

    input_dim: int | None = None
    output_dim: int | None = None
    input_channels: int = 1

    d_model: int = 128
    depth: int = 4
    order: int = 2
    kernel_size: int = 129
    dropout: float = 0.1

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 15

    val_fraction: float = 0.15
    test_fraction: float = 0.15

    standardize_x: bool = True
    standardize_y: bool = True

    seed: int = 42
    device: str = "auto"
