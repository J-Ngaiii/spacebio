from dataclasses import dataclass


@dataclass
class bulkformer_train_config:
    checkpoint_path: str = "PATH/to/bulkformer_checkpoint.pt"
    embed_dim: int = 128  # TODO: set to the checkpoint's embedding size
    freeze_backbone: bool = True  # True trains only the regression head; False fine-tunes the whole model
    dropout: float = 0.1
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-2
    cv_folds: int = 3
    random_state: int = 42

    device: str = "cpu"

    protein_key: str = "protein"  # adata.obsm key holding the sample x protein matrix

    wandb_project: str = "spacebio"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    run_group: str | None = None
