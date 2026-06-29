import logging

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


def main() -> None:
    set_seed(config.SEED)
    logger.info("Seed set to %d", config.SEED)

    train_loader, val_loader, _ = get_dataloaders()

    models: dict = {
        "lstm": LSTMModel(),
        "transformer": TransformerModel(),
    }

    for model_name, model in models.items():
        logger.info("=== Training: %s ===", model_name)
        trainer = Trainer(model_name)
        trainer.train(model, train_loader, val_loader)
        logger.info("=== Done: %s ===", model_name)


if __name__ == "__main__":
    main()
