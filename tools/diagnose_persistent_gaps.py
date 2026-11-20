"""Summarize how a persistent-state training run will traverse YOUR data (needs the data and PyTorch).

Usage, from the repository root:
    python tools/diagnose_persistent_gaps.py test/test_configs/persistent_lstm_hourly.test.yml

It builds the training dataset exactly as BaseTrainer does and reports per basin: segments, time-contiguous runs
(each run starts from a zero state), batches, segments without any observed target (state propagated, no loss), and
the number of finite training targets that enter the loss (all of them). The scaler is written to a temporary folder,
not to your run directory.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.persistentutils import contiguous_batches, infer_time_step, validate_persistent_config

cfg = Config(Path(sys.argv[1]))
validate_persistent_config(cfg)
cfg.train_dir = Path(tempfile.mkdtemp())
ds = get_dataset(cfg=cfg, is_train=True, period="train", scaler={})
L = int(ds.seq_len[0])
B = int(cfg.batch_size)
freq = ds.frequencies[0]

items, step = {}, None
for i in range(len(ds)):
    basin, idx = ds.lookup_table[i]
    end = int(idx[0])
    dates = np.asarray(ds._dates[basin][freq][end - L + 1:end + 1])
    if step is None:
        step = infer_time_step(dates[None, :], freq)
    items.setdefault(basin, []).append((i, end, dates[0], dates[-1]))

print(f"config={sys.argv[1]}  seq_length={L}  batch_size={B}  frequency={freq}  basins={len(items)}")
print(f"{'basin':<12}{'segments':>9}{'runs':>6}{'batches':>9}{'no-target segs':>16}{'finite targets':>16}")
totals = np.zeros(5, dtype=int)
for basin, its in items.items():
    batches = contiguous_batches([(i, d0, d1) for i, _, d0, d1 in its], batch_size=B, step=step)
    sorted_its = sorted(its, key=lambda x: x[2])
    runs = 1 + sum(int(b[2] != a[3] + step) for a, b in zip(sorted_its, sorted_its[1:]))
    y = np.asarray(ds._y[basin][freq]).reshape(len(ds._y[basin][freq]), -1)[:, 0]
    no_target = sum(int(not np.isfinite(y[e - L + 1:e + 1]).any()) for _, e, _, _ in its)
    finite = sum(int(np.isfinite(y[e - L + 1:e + 1]).sum()) for _, e, _, _ in its)
    row = np.array([len(its), runs, len(batches), no_target, finite])
    totals += row
    print(f"{basin:<12}{row[0]:>9}{row[1]:>6}{row[2]:>9}{row[3]:>16}{row[4]:>16}")
print(f"{'TOTAL':<12}{totals[0]:>9}{totals[1]:>6}{totals[2]:>9}{totals[3]:>16}{totals[4]:>16}")
print(f"State resets at forcing gaps per epoch: {totals[1] - len(items)}. All {totals[4]} finite training targets "
      f"enter the loss once per epoch.")
