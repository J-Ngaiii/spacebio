from sklearn.linear_model import LogisticRegression

from src.training.train_conf.logreg_config import logreg_train_config


def build_logreg(config: logreg_train_config | None = None) -> LogisticRegression:
    config = logreg_train_config() if config is None else config
    return LogisticRegression(max_iter=config.max_iter, C=config.C, random_state=config.random_state)
