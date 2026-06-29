import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: nn.Module,
    path: str,
    epoch: int,
    val_loss: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    logger.info("Checkpoint saved: %s (epoch=%d, val_loss=%.6f)", path, epoch, val_loss)


def load_scaler_params(path: str, col: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler params not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    # 실제 파일 형식: {"columns": [...], "mean": [...], "scale": [...]}
    idx = params["columns"].index(col)
    return {"mean": float(params["mean"][idx]), "std": float(params["scale"][idx])}


def invert_target(z: np.ndarray, mean: float, std: float) -> np.ndarray:
    log_return = z * std + mean          # inverse StandardScaler
    pct_return = np.exp(log_return) - 1  # inverse log diff → % 수익률
    return pct_return


def load_checkpoint(model: nn.Module, path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        "Checkpoint loaded: %s (epoch=%d, val_loss=%.6f)",
        path,
        checkpoint["epoch"],
        checkpoint["val_loss"],
    )
    return checkpoint
