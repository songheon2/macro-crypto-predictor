import os
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from src.dataset import MacroDataset
from src.models.lstm import LSTMModel
from src.trainer import Trainer
from src.utils import load_checkpoint


def _make_loader(n: int = 80, shuffle: bool = False) -> DataLoader:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((n, config.INPUT_SIZE)).astype(np.float64)
    target = rng.standard_normal(n).astype(np.float64)
    return DataLoader(MacroDataset(features, target), batch_size=16, shuffle=shuffle)


def test_trainer_runs_without_error(tmp_path: "pathlib.Path") -> None:
    original_ckpt_dir = config.CHECKPOINT_DIR
    config.CHECKPOINT_DIR = str(tmp_path / "checkpoints")
    config.EPOCHS = 3
    config.EARLY_STOPPING_PATIENCE = 10

    model = LSTMModel()
    trainer = Trainer("lstm_test")
    train_loader = _make_loader(80, shuffle=True)
    val_loader = _make_loader(40)

    trainer.train(model, train_loader, val_loader)

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "lstm_test_best.pt")
    assert os.path.exists(ckpt_path), "Best checkpoint must be saved"

    config.CHECKPOINT_DIR = original_ckpt_dir
    config.EPOCHS = 100
    config.EARLY_STOPPING_PATIENCE = 10


def test_early_stopping(tmp_path: "pathlib.Path") -> None:
    """With patience=2 and constant val loss, training should stop early."""
    original_ckpt_dir = config.CHECKPOINT_DIR
    original_epochs = config.EPOCHS
    original_patience = config.EARLY_STOPPING_PATIENCE

    config.CHECKPOINT_DIR = str(tmp_path / "checkpoints")
    config.EPOCHS = 50
    config.EARLY_STOPPING_PATIENCE = 2

    model = LSTMModel()
    trainer = Trainer("lstm_early_stop")
    train_loader = _make_loader(80, shuffle=True)
    val_loader = _make_loader(40)

    trainer.train(model, train_loader, val_loader)

    config.CHECKPOINT_DIR = original_ckpt_dir
    config.EPOCHS = original_epochs
    config.EARLY_STOPPING_PATIENCE = original_patience


def test_checkpoint_loadable_after_training(tmp_path: "pathlib.Path") -> None:
    original_ckpt_dir = config.CHECKPOINT_DIR
    config.CHECKPOINT_DIR = str(tmp_path / "checkpoints")
    config.EPOCHS = 2
    config.EARLY_STOPPING_PATIENCE = 10

    model = LSTMModel()
    trainer = Trainer("lstm_ckpt")
    trainer.train(model, _make_loader(80, True), _make_loader(40))

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "lstm_ckpt_best.pt")
    model2 = LSTMModel()
    meta = load_checkpoint(model2, ckpt_path)
    assert "epoch" in meta
    assert "val_loss" in meta

    config.CHECKPOINT_DIR = original_ckpt_dir
    config.EPOCHS = 100
    config.EARLY_STOPPING_PATIENCE = 10
