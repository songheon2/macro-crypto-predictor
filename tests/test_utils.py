import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

from src.utils import load_checkpoint, save_checkpoint, set_seed


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
