from dataclasses import dataclass


@dataclass
class bulkformer_train_config:
    checkpoint: str = "zhfanrui/BulkFormer-37M"  # Hugging Face model ID or a local folder
    freeze_backbone: bool = True  # True trains only the protein head on precomputed embeddings; False fine-tunes the whole model
    dropout: float = 0.1
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-3  # for the head alone; the model card uses 1e-5 for full fine-tuning
    weight_decay: float = 1e-2
    cv_folds: int = 3
    random_state: int = 42

    device: str = "cpu"

    protein_key: str = "protein"  # adata.obsm key holding the sample x protein matrix
    log2_protein: bool = True  # OSDR TMT values are ratios; log2 makes them symmetric before z-scoring

    wandb_project: str = "spacebio"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    run_group: str | None = None
