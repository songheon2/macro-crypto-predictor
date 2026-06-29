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
