import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from src.utils import invert_target, load_checkpoint, load_scaler_params

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

        result = {
            "model": model_name,
            "rmse": rmse,
            "n_samples": len(labels_arr),
            "preds": preds_arr,
            "labels": labels_arr,
        }
        logger.info("Evaluation — %s: RMSE=%.6f (n=%d)", model_name, rmse, len(labels_arr))

        self._save_prediction_plot(model_name, preds_arr, labels_arr)
        return result

    def save_comparison_chart(self, results: List[dict]) -> None:
        if not results:
            logger.warning("No results to plot for comparison chart.")
            return

        os.makedirs(config.RESULTS_DIR, exist_ok=True)

        model_names = [r["model"] for r in results]
        rmse_values = [r["rmse"] for r in results]

        fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 2), 5))
        bars = ax.bar(model_names, rmse_values)
        ax.set_title("Model Comparison — RMSE (Test Set)")
        ax.set_xlabel("Model")
        ax.set_ylabel("RMSE")
        for bar, val in zip(bars, rmse_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        save_path = os.path.join(config.RESULTS_DIR, "model_comparison_rmse.png")
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        logger.info("Comparison chart saved: %s", save_path)

    def _save_prediction_plot(
        self,
        model_name: str,
        preds_arr: np.ndarray,
        labels_arr: np.ndarray,
    ) -> None:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)

        plot_preds = preds_arr
        plot_labels = labels_arr
        y_label = "btc_open (z-score)"

        try:
            params = load_scaler_params(config.SCALER_PARAMS_PATH, config.TARGET_COL)
            plot_preds = invert_target(preds_arr, params["mean"], params["std"])
            plot_labels = invert_target(labels_arr, params["mean"], params["std"])
            y_label = "btc_open (% return)"
        except FileNotFoundError:
            logger.warning(
                "scaler_params.json not found at %s — plotting in z-score units.",
                config.SCALER_PARAMS_PATH,
            )

        x = np.arange(len(plot_labels))
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(x, plot_labels, label="actual", linewidth=1.0)
        ax.plot(x, plot_preds, label="predicted", linewidth=1.0, alpha=0.8)
        ax.set_title(f"Prediction vs Actual — {model_name} (Test Set)")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel(y_label)
        ax.legend()

        save_path = os.path.join(
            config.RESULTS_DIR, f"prediction_vs_actual_{model_name}.png"
        )
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        logger.info("Prediction plot saved: %s", save_path)
