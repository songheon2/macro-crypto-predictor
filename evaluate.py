import logging
import os

import config
from src.dataset import get_dataloaders
from src.evaluator import Evaluator
from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel
from src.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    set_seed(config.SEED)

    _, _, test_loader = get_dataloaders()

    evaluator = Evaluator()
    models: dict = {
        "lstm": LSTMModel(),
        "transformer": TransformerModel(),
    }

    results = []
    for model_name, model in models.items():
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pt")
        if not os.path.exists(ckpt_path):
            logger.warning("Checkpoint not found for %s: %s", model_name, ckpt_path)
            continue
        result = evaluator.evaluate(model, model_name, ckpt_path, test_loader)
        results.append(result)

    if results:
        evaluator.save_comparison_chart(results)
        logger.info("=== Evaluation Summary ===")
        for r in sorted(results, key=lambda x: x["rmse"]):
            logger.info("  %-15s RMSE=%.6f", r["model"], r["rmse"])
    else:
        logger.warning("No evaluation results — run train.py first.")


if __name__ == "__main__":
    main()
