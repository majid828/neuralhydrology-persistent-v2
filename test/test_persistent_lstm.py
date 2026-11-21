"""Tests of the persistent-state LSTM method (manuscript Sec. 2) that need no hydrological data.

Run from the repository root:  pytest -q test/test_persistent_lstm.py
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from neuralhydrology.modelzoo import get_model
from neuralhydrology.training import get_loss_obj
from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.persistentutils import (contiguous_batches, select_window_ends, unflatten_predictions,
                                                   validate_persistent_config)

CFG_FILE = Path(__file__).parent / "test_configs" / "persistent_lstm_for_long_memory_basins.test.yml"
B, L = 8, 2  # batch_size and seq_length of the daily persistent configs


def _cfg() -> Config:
    return Config(CFG_FILE)


def _inputs(cfg: Config, T: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    x_d = {k: torch.randn(1, T, 1, generator=g) for k in cfg.dynamic_inputs}
    x_s = torch.randn(1, len(cfg.static_attributes), generator=g)
    return {"x_d": x_d, "x_s": x_s}


def test_flattened_persistent_equals_one_continuous_run():
    """Carrying detached (h, c) across flattened [B, L] batches reproduces one uninterrupted LSTM run (Eq. 7-13)."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = get_model(cfg).eval()
    n_batches = 3
    full = _inputs(cfg, B * L * n_batches, seed=1)
    with torch.no_grad():
        reference = model(full)["y_hat"]
        hidden, outs = None, []
        for k in range(n_batches):
            sl = slice(k * B * L, (k + 1) * B * L)
            x_d = {f: BaseTrainer._flatten_time_batch(v[:, sl].reshape(B, L, 1)) for f, v in full["x_d"].items()}
            pred = model({"x_d": x_d, "x_s": full["x_s"]}, hidden_state=hidden)
            h, c = pred["hidden_state"]
            hidden = (h.detach(), c.detach())
            outs.append(pred["y_hat"])
    torch.testing.assert_close(torch.cat(outs, dim=1), reference)


def test_cell_state_is_carried():
    """Both h and c are carried: dropping the cached c changes the predictions."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = get_model(cfg).eval()
    a, b = _inputs(cfg, 16, seed=2), _inputs(cfg, 16, seed=3)
    with torch.no_grad():
        h, c = model(a)["hidden_state"]
        with_c = model(b, hidden_state=(h, c))["y_hat"]
        h_only = model(b, hidden_state=(h, torch.zeros_like(c)))["y_hat"]
    assert not torch.allclose(with_c, h_only)


def test_detach_truncates_gradient_but_keeps_values():
    """Truncated BPTT: values persist, the computation graph does not cross the batch boundary (Sec. 2.4)."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = get_model(cfg).train()
    seg1, seg2 = _inputs(cfg, 16, seed=4), _inputs(cfg, 16, seed=5)
    for v in seg1["x_d"].values():
        v.requires_grad_(True)

    h, c = model(seg1)["hidden_state"]
    model(seg2, hidden_state=(h.detach(), c.detach()))["y_hat"].sum().backward()
    assert all(v.grad is None for v in seg1["x_d"].values())

    model.zero_grad()
    model(seg2, hidden_state=model(seg1)["hidden_state"])["y_hat"].sum().backward()
    assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in seg1["x_d"].values())


def test_loss_scores_every_step_of_flattened_batch():
    """Eq. (17)/(20): the trainer's loss covers all B*L steps of the flattened stream."""
    cfg = _cfg()
    loss_obj = get_loss_obj(cfg)
    torch.manual_seed(0)
    y = torch.randn(B, L, 1)
    y[0, 0, 0] = float("nan")  # missing targets are excluded
    preds_flat = {"y_hat": torch.randn(1, B * L, 1), "hidden_state": (torch.zeros(1, 1, 4), torch.zeros(1, 1, 4))}

    loss, _ = loss_obj(unflatten_predictions(preds_flat, B, L), {"y": y})
    y_hat = preds_flat["y_hat"].reshape(B, L, 1)
    mask = ~torch.isnan(y)
    expected = 0.5 * torch.mean((y_hat[mask] - y[mask]) ** 2)  # NeuralHydrology's masked MSE carries a factor 1/2
    torch.testing.assert_close(loss, expected)

    y_changed = y.clone()
    y_changed[0, 1] += 10.0  # a target in the FIRST segment of the batch
    changed, _ = loss_obj(unflatten_predictions(preds_flat, B, L), {"y": y_changed})
    assert not torch.isclose(loss, changed)


@pytest.mark.parametrize("update, should_raise", [
    ({}, False),
    ({"non_overlapping_sequences": False}, True),
    ({"predict_last_n": 1}, True),
    ({"seq_stride": 3}, True),
    ({"eval_warmup_steps": 61}, True),
    ({"persistent_state_init_file": "states.p"}, True),
    ({"model": "cudalstm"}, True),
    ({"persistent_state": False}, True),  # persistentlstm would otherwise silently train like a standard LSTM
])
def test_validate_persistent_config(update, should_raise):
    cfg = _cfg()
    if update:
        cfg.update_config(update)
    if should_raise:
        with pytest.raises(ValueError):
            validate_persistent_config(cfg)
    else:
        validate_persistent_config(cfg)


def test_select_window_ends_keeps_a_fixed_grid():
    """Segments end at k*L-1 (Eq. 4-6); an invalid window does not shift the following ones."""
    flag = np.ones(20)
    flag[9] = 0  # window ending at index 9 is invalid (e.g. NaN forcing)
    ends = select_window_ends(flag, seq_len=2, stride=2)
    np.testing.assert_array_equal(ends, [1, 3, 5, 7, 11, 13, 15, 17, 19])


def test_contiguous_batches_split_at_gaps():
    """Batches never contain a time gap; a gap starts a new batch."""
    day = np.timedelta64(1, "D")
    start = np.datetime64("2000-01-01")
    ends = [1, 3, 5, 7, 11, 13, 15, 17, 19]
    items = [(i, start + (e - 1) * day, start + e * day) for i, e in enumerate(ends)]
    batches = contiguous_batches(items, batch_size=3, step=day)
    assert batches == [[0, 1, 2], [3], [4, 5, 6], [7, 8]]
