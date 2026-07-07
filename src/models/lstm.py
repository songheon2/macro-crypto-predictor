import torch
import torch.nn as nn

import config


class LSTMModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = config.LSTM_HIDDEN_SIZE,
        num_layers: int = config.LSTM_NUM_LAYERS,
        dropout: float = config.LSTM_DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.INPUT_SIZE,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, SEQ_LEN, INPUT_SIZE)
        out, _ = self.lstm(x)          # (batch, SEQ_LEN, hidden)
        last = out[:, -1, :]           # (batch, hidden)
        return self.fc(last).squeeze(-1)  # (batch,)
