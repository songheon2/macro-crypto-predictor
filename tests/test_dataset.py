import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

import config
from src.dataset import MacroDataset, _load_split


_COLS = [
    "open", "volume", "gov_1y", "usd_krw", "gold_usd",
    "nasdaq_close", "nasdaq_volume", "btc_open", "btc_volume",
    "btc_rsi", "eth_open", "eth_volume", "eth_rsi",
    "volatility_20d", "term_spread_10y1y", "term_spread_3y1y",
    "credit_spread_aa", "credit_spread_bbb",
]


def _make_csv(tmpdir: str, name: str, n_rows: int = 50) -> str:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((n_rows, len(_COLS))), columns=_COLS)
    path = os.path.join(tmpdir, name)
    df.to_csv(path)
    return path


def _make_csv_with_leading_nans(
    tmpdir: str, name: str, n_leading: int = 5, n_rows: int = 50
) -> str:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((n_rows, len(_COLS))), columns=_COLS)
    df.iloc[:n_leading, 0] = float("nan")  # 첫 n_leading행의 첫 컬럼에 NaN 삽입
    path = os.path.join(tmpdir, name)
    df.to_csv(path)
    return path


def test_dataset_length() -> None:
    rng = np.random.default_rng(1)
    n = 50
    features = rng.standard_normal((n, config.INPUT_SIZE))
    target = rng.standard_normal(n)
    ds = MacroDataset(features, target)
    assert len(ds) == n - config.SEQ_LEN + 1


def test_dataset_item_shapes() -> None:
    rng = np.random.default_rng(2)
    n = 50
    features = rng.standard_normal((n, config.INPUT_SIZE))
    target = rng.standard_normal(n)
    ds = MacroDataset(features, target)
    x, y = ds[0]
    assert x.shape == (config.SEQ_LEN, config.INPUT_SIZE)
    assert y.shape == ()


def test_dataset_sliding_window_alignment() -> None:
    """y[i] must correspond to target[i + SEQ_LEN - 1]."""
    rng = np.random.default_rng(3)
    n = 50
    features = rng.standard_normal((n, config.INPUT_SIZE))
    target = np.arange(n, dtype=np.float64)
    ds = MacroDataset(features, target)
    for i in range(len(ds)):
        _, y = ds[i]
        expected = target[i + config.SEQ_LEN - 1]
        assert abs(y.item() - expected) < 1e-9, f"Mismatch at index {i}"


def test_dataset_too_short() -> None:
    features = np.zeros((config.SEQ_LEN - 1, config.INPUT_SIZE))
    target = np.zeros(config.SEQ_LEN - 1)
    with pytest.raises(ValueError):
        MacroDataset(features, target)


def test_load_split_no_leakage(tmp_path: "pathlib.Path") -> None:
    """Verify y is shift(-1) of btc_open, not btc_open itself."""
    path = _make_csv(str(tmp_path), "train.csv", n_rows=50)

    df_orig = pd.read_csv(path, index_col=0, parse_dates=True)
    expected_y = df_orig[config.TARGET_COL].shift(-1).dropna().values

    features, target = _load_split(path)

    assert len(features) == len(target)
    assert len(target) == len(expected_y)
    assert np.allclose(target, expected_y), "y must equal btc_open.shift(-1)"

    # features의 btc_open 컬럼(인덱스 7)과 y가 다른지 확인
    btc_open_col_idx = 7
    assert not np.allclose(
        features[:, btc_open_col_idx], target
    ), "Feature btc_open[t] must differ from target btc_open[t+1]"


def test_load_split_removes_last_row(tmp_path: "pathlib.Path") -> None:
    path = _make_csv(str(tmp_path), "train.csv", n_rows=50)
    features, target = _load_split(path)
    assert len(target) == 49, "shift(-1) + iloc[:-1] should remove exactly 1 row"
    assert not np.any(np.isnan(target)), "No NaN should remain in target"


def test_load_split_leading_nan_removed(tmp_path: "pathlib.Path") -> None:
    """Leading NaN rows must be stripped before sliding window construction."""
    n_leading = 5
    n_rows = 50
    path = _make_csv_with_leading_nans(str(tmp_path), "train.csv", n_leading, n_rows)
    features, target = _load_split(path)
    # n_rows - n_leading - 1(shift) = 44
    expected_len = n_rows - n_leading - 1
    assert len(target) == expected_len, (
        f"Expected {expected_len} rows after leading NaN removal, got {len(target)}"
    )
    assert not np.any(np.isnan(features)), "No NaN should remain in features"
    assert not np.any(np.isnan(target)), "No NaN should remain in target"
