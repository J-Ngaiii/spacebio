from dataclasses import dataclass, field


@dataclass
class mlp_train_config:
    hidden_dim: int = 32
    dropout: float = 0.2
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-2
    cv_folds: int = 3
    random_state: int = 42

    device: str = "cpu"
    precision: str = "fp32"  # fp32, bf16, fp16; fp16 requires CUDA

    optimizer: str = "adamw"  # adamw, adam, sgd
    optimizer_kwargs: dict = field(default_factory=dict)
    # Examples: {"momentum": 0.9} for SGD; {"betas": (0.9, 0.95)} for AdamW

    criterion: str = "bce_with_logits"  # bce_with_logits, mse
    criterion_kwargs: dict = field(default_factory=dict)
    # MSE applies sigmoid first; BCEWithLogitsLoss receives raw logits.

    label_column: str = "arm"
    positive_label: str = "Flight"
    negative_label: str = "Ground.Control"

    wandb_project: str = "spacebio"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    run_group: str | None = None