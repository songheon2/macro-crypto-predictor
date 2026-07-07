import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from src.utils import save_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model_name: str, lr: float = config.LEARNING_RATE) -> None:
        self._model_name = model_name
        self._lr = lr
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Trainer device: %s", self._device)

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> float:
        model.to(self._device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR, f"{self._model_name}_best.pt"
        )

        for epoch in range(1, config.EPOCHS + 1):
            train_loss = self._run_epoch(model, train_loader, criterion, optimizer, train=True)
            val_loss = self._run_epoch(model, val_loader, criterion, optimizer=None, train=False)

            logger.info(
                "Epoch %3d/%d — train_loss=%.6f  val_loss=%.6f",
                epoch,
                config.EPOCHS,
                train_loss,
                val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(model, checkpoint_path, epoch=epoch, val_loss=val_loss)
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch,
                        config.EARLY_STOPPING_PATIENCE,
                    )
                    break

        return best_val_loss

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        train: bool,
    ) -> float:
        model.train(train)
        total_loss = 0.0
        n_batches = 0

        with torch.set_grad_enabled(train):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                preds = model(x_batch)
                loss = criterion(preds, y_batch)

                if train and optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0
