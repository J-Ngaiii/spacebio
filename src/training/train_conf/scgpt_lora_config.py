from dataclasses import dataclass


@dataclass
class scgpt_lora_train_config:
    checkpoint_dir: str = ""            # folder with best_model.pt, vocab.json, args.json (required)
    task: str = "classification"        # classification (flight vs ground, AUROC) | protein (RNA -> protein)

    use_lora: bool = True               # False -> frozen backbone, train the head only
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    head_hidden_dim: int = 32
    head_dropout: float = 0.2
    pad_value: int = -2                 # take this from the checkpoint's args.json

    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-3                    # head
    lora_lr: float = 1e-4               # LoRA adapters
    weight_decay: float = 1e-2
    cv_folds: int = 3
    random_state: int = 42
    device: str = "cuda"

    # classification task
    label_column: str = "arm"
    positive_label: str = "Flight"
    negative_label: str = "Ground.Control"

    # protein task: folds are grouped so an animal never spans train and validation
    group_column: str = "animal_id"

    wandb_project: str = "spacebio"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    run_group: str | None = None