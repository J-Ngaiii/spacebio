from dataclasses import dataclass

@dataclass
class mlp_train_config:
    input_dim: int=10
    hidden_dim: int=32
    output_dim: int=1
    batch_size: int=32
    epochs: int=20
    lr: float=1e-3