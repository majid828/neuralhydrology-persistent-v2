"""Helpers shared by persistent-state LSTM training (BaseTrainer) and evaluation (BaseTester).

The persistent-state method (manuscript Sec. 2) requires:
  * single-frequency data split into non-overlapping, chronological segments of length L (Eq. 4-6),
  * B consecutive segments of one basin flattened into one continuous stream of length T = B*L (Eq. 7-8),
  * the loss computed over every valid time step of that stream (Eq. 17, 20),
  * basin-specific (h, c) caches that are only carried between segments that are contiguous in time.

The functions below implement the parts of this logic that both the trainer and the tester need.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pandas.tseries.frequencies import to_offset

LOGGER = logging.getLogger(__name__)


def uses_persistent_state(cfg) -> bool:
    """True if the run uses persistent-state training/evaluation."""
    return bool(getattr(cfg, "persistent_state", False)) and str(cfg.model).lower() == "persistentlstm"


def _single_value(value: Union[int, Dict[str, int]], name: str) -> int:
    if isinstance(value, dict):
        if len(value) != 1:
            raise ValueError(f"Persistent-state training supports a single frequency, but '{name}' has "
                             f"{len(value)} entries: {value}")
        return int(next(iter(value.values())))
    return int(value)


def single_seq_length(cfg) -> int:
    """Segment length L (seq_length) of a single-frequency run."""
    return _single_value(cfg.seq_length, "seq_length")


def single_predict_last_n(cfg) -> int:
    """predict_last_n of a single-frequency run."""
    return _single_value(cfg.predict_last_n, "predict_last_n")


def validate_persistent_config(cfg) -> None:
    """Raise a ValueError if the configuration cannot reproduce the persistent-state method.

    Without these checks, wrong settings fail silently: overlapping windows would re-process time steps under a
    carried state, predict_last_n < seq_length would shrink the loss to part of each segment, and a model other
    than 'persistentlstm' would ignore persistent_state altogether.
    """
    if not bool(getattr(cfg, "persistent_state", False)):
        if bool(getattr(cfg, "ablate_state_carry", False)):
            raise ValueError("'ablate_state_carry' only applies to runs with 'persistent_state: true'.")
        return

    errors = []
    if str(cfg.model).lower() != "persistentlstm":
        errors.append(f"'persistent_state: true' requires 'model: persistentlstm' (got '{cfg.model}').")
    if cfg.use_frequencies and len(cfg.use_frequencies) > 1:
        errors.append(f"persistent-state training supports one frequency only (got {cfg.use_frequencies}).")
    if not cfg.non_overlapping_sequences:
        errors.append("'non_overlapping_sequences' must be true: overlapping windows would re-process time steps "
                      "that the carried state has already seen.")
    try:
        seq_len = single_seq_length(cfg)
        predict_last_n = single_predict_last_n(cfg)
    except ValueError as err:
        errors.append(str(err))
    else:
        if predict_last_n != seq_len:
            errors.append(f"'predict_last_n' ({predict_last_n}) must equal 'seq_length' ({seq_len}) so that the "
                          f"loss covers every time step of each segment (manuscript Eq. 17).")
        stride = int(cfg.seq_stride)
        if stride not in (1, seq_len):
            errors.append(f"'seq_stride' must be 1 or equal to 'seq_length' ({seq_len}) for persistent-state "
                          f"training (got {stride}).")
        warmup = int(cfg.eval_warmup_steps)
        if cfg.eval_warmup_from_preceding_period and warmup % seq_len != 0:
            errors.append(f"'eval_warmup_steps' ({warmup}) must be a multiple of 'seq_length' ({seq_len}) when "
                          f"'eval_warmup_from_preceding_period' is true, so that scoring starts exactly at the "
                          f"period start date.")
    if cfg.persistent_state_init_file is not None and cfg.eval_warmup_from_preceding_period:
        errors.append("'persistent_state_init_file' and 'eval_warmup_from_preceding_period' are mutually exclusive: "
                      "a loaded state already summarizes the history before the period start.")
    if errors:
        raise ValueError("Invalid persistent-state configuration:\n  - " + "\n  - ".join(errors))


def frequency_time_step(freq: str) -> Optional[np.timedelta64]:
    """Length of one time step of a fixed frequency (e.g. '1D', '1h', '15min'), or None if not fixed."""
    try:
        return np.timedelta64(int(to_offset(freq).nanos), "ns")
    except (ValueError, TypeError, AttributeError):
        return None


def infer_time_step(dates: np.ndarray, freq: Optional[str] = None) -> np.timedelta64:
    """Time step of a dataset, from its frequency string or, if needed, from a [.., L] array of window dates."""
    step = frequency_time_step(freq) if freq is not None else None
    if step is not None:
        return step
    dates = np.asarray(dates)
    if dates.ndim >= 1 and dates.shape[-1] >= 2:
        return (dates.reshape(-1, dates.shape[-1])[0, 1] - dates.reshape(-1, dates.shape[-1])[0, 0]).astype(
            "timedelta64[ns]")
    raise ValueError(f"Cannot determine the time step for frequency '{freq}' with seq_length 1.")


def select_window_ends(flag: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    """Window end indices for non-overlapping, chronological segments on a FIXED time grid (manuscript Eq. 4-6).

    Segment k covers steps (k-1)*L .. k*L-1 (0-based), i.e. it ends at index k*L-1. Invalid windows (flag == 0) are
    left out, but the grid itself never shifts, so every kept window still starts exactly L steps after the previous
    grid position. (Thinning the list of valid windows by position, as before, shifted the grid after every gap.)
    """
    stride = int(seq_len) if int(stride) <= 1 else int(stride)
    grid = np.arange(int(seq_len) - 1, len(flag), stride)
    return grid[np.asarray(flag)[grid] == 1]


def contiguous_batches(items: List[Tuple[int, np.datetime64, np.datetime64]], batch_size: int,
                       step: np.timedelta64) -> List[List[int]]:
    """Group one basin's windows into batches of up to `batch_size` windows that are contiguous in time.

    `items` holds (dataset index, first date, last date) per window. Windows are sorted chronologically and split
    into runs wherever a window does not start exactly one time step after the previous window ended (a data gap).
    Each run is cut into batches of `batch_size` consecutive windows; only the last batch of a run can be shorter.
    Flattening a batch therefore always gives a truly continuous stream (manuscript Sec. 2.3).
    """
    items = sorted(items, key=lambda x: x[1])
    batches: List[List[int]] = []
    current: List[int] = []
    previous_end = None
    for idx, start, end in items:
        breaks_run = previous_end is not None and start != previous_end + step
        if current and (breaks_run or len(current) == batch_size):
            batches.append(current)
            current = []
        current.append(idx)
        previous_end = end
    if current:
        batches.append(current)
    return batches


def unflatten_time_batch(value: Any, batch_size: int, seq_len: int) -> Any:
    """Inverse of the [B, L, ...] -> [1, B*L, ...] flattening, for tensors that have that flattened shape."""
    if torch.is_tensor(value) and value.dim() >= 2 and value.shape[0] == 1 and value.shape[1] == batch_size * seq_len:
        return value.reshape(batch_size, seq_len, *value.shape[2:])
    return value


def unflatten_predictions(predictions: Dict[str, Any], batch_size: int, seq_len: int) -> Dict[str, Any]:
    """Model outputs reshaped back to [B, L, ...] so that the loss scores every step of every segment."""
    return {key: unflatten_time_batch(value, batch_size, seq_len)
            for key, value in predictions.items() if key != "hidden_state"}


def has_finite_targets(data: Dict[str, Any]) -> bool:
    """True if any target value in the batch is finite (otherwise the batch carries no supervision)."""
    for key, value in data.items():
        if key.startswith("y") and torch.is_tensor(value) and bool(torch.isfinite(value).any()):
            return True
    return False


def state_is_finite(hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> bool:
    """True if a cached (h, c) contains only finite values."""
    if hidden is None:
        return True
    return all(bool(torch.isfinite(t).all()) for t in hidden)
