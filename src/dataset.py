import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config

logger = logging.getLogger(__name__)


class MacroDataset(Dataset):
    def __init__(
        self, features: np.ndarray, target: np.ndarray, seq_len: int = config.SEQ_LEN
    ) -> None:
        # features: (T, INPUT_SIZE), target: (T,)  — already aligned after dropna
        self._features = features
        self._target = target
        self._seq_len = seq_len
        n = len(target)
        if n < self._seq_len:
            raise ValueError(
                f"Dataset length {n} is shorter than SEQ_LEN={self._seq_len}"
            )
        self._length = n - self._seq_len + 1

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._features[idx : idx + self._seq_len]          # (SEQ_LEN, INPUT_SIZE)
        y = self._target[idx + self._seq_len - 1]              # scalar — btc_open_{t+1}
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def _load_split(path: str) -> Tuple[np.ndarray, np.ndarray]:
    # 1. CSV 로드
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # 2. 초반 연속 NaN 블록 제거 — 유효 시작일을 데이터에서 동적으로 탐지
    first_valid = df.dropna(how="any").index[0]
    df = df.loc[first_valid:]

    # 3. y 생성: shift(-1) 적용 — X 배열과 독립적으로 별도 생성
    y_series = df[config.TARGET_COL].shift(-1)

    # 4. 마지막 행 NaN 제거 (shift로 생긴 NaN)
    df = df.iloc[:-1]
    y_series = y_series.iloc[:-1]

    features = df.values.astype(np.float64)    # (T, INPUT_SIZE)
    target = y_series.values.astype(np.float64)  # (T,)

    logger.info("Loaded %s: %d samples (features=%s)", path, len(target), features.shape)
    return features, target


def get_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    seq_len: int = config.SEQ_LEN,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_path = os.path.join(config.DATA_DIR, "macro_features_train_scaled.csv")
    val_path = os.path.join(config.DATA_DIR, "macro_features_val_scaled.csv")
    test_path = os.path.join(config.DATA_DIR, "macro_features_test_scaled.csv")

    train_features, train_target = _load_split(train_path)
    val_features, val_target = _load_split(val_path)
    test_features, test_target = _load_split(test_path)

    train_loader = DataLoader(
        MacroDataset(train_features, train_target, seq_len=seq_len),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        MacroDataset(val_features, val_target, seq_len=seq_len),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        MacroDataset(test_features, test_target, seq_len=seq_len),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
