import torch
import torch.nn as nn

import config


class LSTMModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.INPUT_SIZE,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            dropout=config.LSTM_DROPOUT if config.LSTM_NUM_LAYERS > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(config.LSTM_HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, SEQ_LEN, INPUT_SIZE)
        out, _ = self.lstm(x)          # (batch, SEQ_LEN, hidden)
        last = out[:, -1, :]           # (batch, hidden)
        return self.fc(last).squeeze(-1)  # (batch,)
