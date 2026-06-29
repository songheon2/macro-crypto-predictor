import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from src.utils import load_checkpoint

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(
        self,
        model: nn.Module,
        model_name: str,
        checkpoint_path: str,
        test_loader: DataLoader,
    ) -> dict:
        load_checkpoint(model, checkpoint_path)
        model.to(self._device)
        model.eval()

        all_preds: list[float] = []
        all_labels: list[float] = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self._device)
                preds = model(x_batch).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(y_batch.numpy().tolist())

        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)
        rmse = float(np.sqrt(np.mean((preds_arr - labels_arr) ** 2)))

        result = {"model": model_name, "rmse": rmse, "n_samples": len(labels_arr)}
        logger.info("Evaluation — %s: RMSE=%.6f (n=%d)", model_name, rmse, len(labels_arr))

        self._save_result(result)
        return result

    def _save_result(self, result: dict) -> None:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        results_path = os.path.join(config.RESULTS_DIR, "evaluation_results.csv")

        row = pd.DataFrame([result])
        if os.path.exists(results_path):
            existing = pd.read_csv(results_path)
            combined = pd.concat(
                [existing[existing["model"] != result["model"]], row],
                ignore_index=True,
            )
        else:
            combined = row

        combined.to_csv(results_path, index=False)
        logger.info("Results saved to %s", results_path)
