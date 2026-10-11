from dataclasses import dataclass


@dataclass
class logreg_train_config:
    """Parameters for the RNA count logistic regression probe."""
    max_iter: int = 1000
    C: float = 1.0
    cv_folds: int = 3
    random_state: int = 42
    scoring: str = "roc_auc"
    label_column: str = "arm"
    positive_label: str = "Flight"
    negative_label: str = "Ground.Control"
