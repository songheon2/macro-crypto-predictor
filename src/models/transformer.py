import math

import torch
import torch.nn as nn

import config


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model: int = config.TRANSFORMER_D_MODEL,
        nhead: int = config.TRANSFORMER_NHEAD,
        num_layers: int = config.TRANSFORMER_NUM_LAYERS,
        dropout: float = config.TRANSFORMER_DROPOUT,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(config.INPUT_SIZE, d_model)
        self.pos_enc = _PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, SEQ_LEN, INPUT_SIZE)
        x = self.input_proj(x)         # (batch, SEQ_LEN, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)            # (batch, SEQ_LEN, d_model)
        last = x[:, -1, :]             # (batch, d_model)
        return self.fc(last).squeeze(-1)  # (batch,)
