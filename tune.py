import json
import logging
import math
import os

import optuna

import config
from src.dataset import get_dataloaders
from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel
from src.trainer import Trainer
from src.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _objective_lstm(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    seq_len = trial.suggest_categorical("seq_len", [15, 30, 60])
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)

    set_seed(config.SEED)
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size, seq_len=seq_len)
    model = LSTMModel(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    trainer = Trainer(model_name="lstm_tune", lr=lr, weight_decay=weight_decay)
    best_val_loss = trainer.train(model, train_loader, val_loader)
    return math.sqrt(best_val_loss)


def _objective_transformer(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    seq_len = trial.suggest_categorical("seq_len", [15, 30, 60])
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    set_seed(config.SEED)
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size, seq_len=seq_len)
    model = TransformerModel(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
    trainer = Trainer(model_name="transformer_tune", lr=lr, weight_decay=weight_decay)
    best_val_loss = trainer.train(model, train_loader, val_loader)
    return math.sqrt(best_val_loss)


def _retrain_best(model_name: str, best_params: dict) -> None:
    set_seed(config.SEED)
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=best_params["batch_size"],
        seq_len=best_params["seq_len"],
    )
    if model_name == "lstm":
        model = LSTMModel(
            hidden_size=best_params["hidden_size"],
            num_layers=best_params["num_layers"],
            dropout=best_params["dropout"],
        )
    else:
        model = TransformerModel(
            d_model=best_params["d_model"],
            nhead=best_params["nhead"],
            num_layers=best_params["num_layers"],
            dropout=best_params["dropout"],
        )
    trainer = Trainer(model_name=model_name, lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    trainer.train(model, train_loader, val_loader)


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objectives = [
        ("lstm", _objective_lstm),
        ("transformer", _objective_transformer),
    ]
    results: dict = {}

    for model_name, objective in objectives:
        logger.info("=== Tuning: %s (%d trials) ===", model_name, config.N_TRIALS)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=config.N_TRIALS, show_progress_bar=True)

        best = study.best_trial
        logger.info(
            "%s best val_rmse=%.6f  params=%s", model_name, best.value, best.params
        )
        results[model_name] = {"val_rmse": best.value, "params": dict(best.params)}

        logger.info("=== Retraining best %s ===", model_name)
        _retrain_best(model_name, best.params)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, "best_params.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Best params saved: %s", out_path)


if __name__ == "__main__":
    main()
