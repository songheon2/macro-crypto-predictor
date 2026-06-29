import json
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

from src.utils import (
    invert_target,
    load_checkpoint,
    load_scaler_params,
    save_checkpoint,
    set_seed,
)


class _SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_set_seed_reproducibility() -> None:
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    assert np.allclose(a, b), "set_seed must produce reproducible numpy results"


def test_set_seed_torch() -> None:
    set_seed(42)
    a = torch.rand(5)
    set_seed(42)
    b = torch.rand(5)
    assert torch.allclose(a, b), "set_seed must produce reproducible torch results"


def test_save_and_load_checkpoint() -> None:
    model = _SimpleModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt", "model.pt")
        save_checkpoint(model, path, epoch=3, val_loss=0.123)
        assert os.path.exists(path)

        model2 = _SimpleModel()
        meta = load_checkpoint(model2, path)
        assert meta["epoch"] == 3
        assert abs(meta["val_loss"] - 0.123) < 1e-9

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Loaded weights must match saved weights"


def test_load_checkpoint_missing_file() -> None:
    model = _SimpleModel()
    try:
        load_checkpoint(model, "/nonexistent/path/model.pt")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_invert_target_values() -> None:
    mean, std = 0.001, 0.05
    z = np.array([0.0, 1.0, -1.0])
    result = invert_target(z, mean, std)
    expected_log = z * std + mean
    expected_pct = np.exp(expected_log) - 1
    assert np.allclose(result, expected_pct), "invert_target must match manual calculation"


def test_invert_target_zero_z_score() -> None:
    mean, std = 0.002, 0.03
    z = np.array([0.0])
    result = invert_target(z, mean, std)
    expected = np.exp(mean) - 1
    assert np.allclose(result, expected), "z=0 should give exp(mean)-1"


def test_invert_target_shape_preserved() -> None:
    z = np.zeros((5, 3))
    result = invert_target(z, mean=0.0, std=1.0)
    assert result.shape == z.shape, "Output shape must match input shape"


def test_load_scaler_params(tmp_path: "pathlib.Path") -> None:
    params = {
        "columns": ["open", "btc_open", "volume"],
        "mean": [0.0, 0.001, 0.5],
        "scale": [1.0, 0.05, 2.0],
    }
    path = str(tmp_path / "scaler_params.json")
    with open(path, "w") as f:
        json.dump(params, f)
    loaded = load_scaler_params(path, col="btc_open")
    assert abs(loaded["mean"] - 0.001) < 1e-12
    assert abs(loaded["std"] - 0.05) < 1e-12


def test_load_scaler_params_missing_file() -> None:
    try:
        load_scaler_params("/nonexistent/scaler_params.json", col="btc_open")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
