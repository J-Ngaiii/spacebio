from dataclasses import dataclass


@dataclass
class pca_train_config:
    n_components: int = 5  # capped at (training samples - 1) inside each fold
    alphas: tuple = (0.1, 1.0, 10.0, 100.0, 1000.0)  # ridge penalties; the best is picked within each training fold
    cv_folds: int = 3
    random_state: int = 42

    protein_key: str = "protein"  # adata.obsm key holding the sample x protein matrix
    log2_protein: bool = True  # OSDR TMT values are ratios; log2 makes them symmetric before z-scoring

    wandb_project: str = "spacebio"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    run_group: str | None = None
