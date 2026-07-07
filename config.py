DATA_DIR: str = "data"
OUTPUT_DIR: str = "outputs"
CHECKPOINT_DIR: str = "outputs/checkpoints"
RESULTS_DIR: str = "outputs/results"

SEED: int = 42
BATCH_SIZE: int = 64
EPOCHS: int = 100
LEARNING_RATE: float = 1e-3
EARLY_STOPPING_PATIENCE: int = 10
SEQ_LEN: int = 30
INPUT_SIZE: int = 18

LSTM_HIDDEN_SIZE: int = 128
LSTM_NUM_LAYERS: int = 2
LSTM_DROPOUT: float = 0.2

TRANSFORMER_D_MODEL: int = 64
TRANSFORMER_NHEAD: int = 4
TRANSFORMER_NUM_LAYERS: int = 2
TRANSFORMER_DROPOUT: float = 0.1

TARGET_COL: str = "btc_open"
SCALER_PARAMS_PATH: str = "data/scaler_params.json"

N_TRIALS: int = 50
WEIGHT_DECAY: float = 1e-4
LR_SCHEDULER_PATIENCE: int = 5
LR_SCHEDULER_FACTOR: float = 0.5
GRAD_CLIP_NORM: float = 1.0
