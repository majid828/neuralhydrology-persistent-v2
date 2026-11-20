"""
Checks of the persistent-state code that run WITHOUT PyTorch (e.g. on a laptop without a GPU environment).

The repository's own source is extracted verbatim with `ast` (sampler, batch builder, training epoch loop, loss,
window selection, evaluation loop, 15-min loader) and executed with a small NumPy stand-in for the tensor operations
those code paths use. nn.LSTM is replaced by a deterministic "counter" model whose (h, c) record the last absolute
time index processed and the number of steps since the last reset, which makes every state hand-off checkable.

What this establishes: ordering, state hand-offs and resets, flattening, static reattachment, which time steps enter
the loss, gap handling, warm-up and scoring window, state save/load, configuration guards.
What it does not establish: autograd/detach semantics and real LSTM numerics -> see test/test_persistent_lstm.py.
"""
import ast
import logging
import random
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else ".")  # repository root
EVENTS: List[str] = []


# ----------------------------------------------------------------------------- faketorch
class Tensor(np.ndarray):
    def __new__(cls, a):
        return np.asarray(a).view(cls)

    def view(self, *shape):
        if len(shape) == 1 and (isinstance(shape[0], type) or isinstance(shape[0], np.dtype)):
            return np.ndarray.view(self, shape[0])
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return np.ndarray.reshape(self, shape)

    def contiguous(self):
        return self

    def dim(self):
        return self.ndim

    def to(self, *a, **k):
        return self

    def detach(self):
        return np.ndarray.copy(self)

    def clone(self):
        return np.ndarray.copy(self)

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def expand(self, *shape):
        return Tensor(np.broadcast_to(np.asarray(self), shape))

    def backward(self):
        EVENTS.append("backward")


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Module:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return self.forward(*a, **k)

    def to(self, *a, **k):
        return self


class _Sampler:
    def __init__(self, *a, **k):
        pass

    def __class_getitem__(cls, item):
        return cls


class _DataLoader:
    """Mimics torch DataLoader(batch_sampler=..., collate_fn=...): fetch each index, then collate."""

    def __init__(self, ds, batch_size=None, batch_sampler=None, num_workers=0, collate_fn=None, shuffle=False):
        self.ds, self.batch_sampler, self.collate_fn = ds, batch_sampler, collate_fn

    def __iter__(self):
        for batch in self.batch_sampler:
            yield self.collate_fn([self.ds[i] for i in batch])

    def __len__(self):
        return len(self.batch_sampler)


class _NS:
    pass


faketorch = _NS()
faketorch.Tensor = Tensor
faketorch.is_tensor = lambda x: isinstance(x, Tensor)
faketorch.tensor = lambda x, dtype=None: Tensor(np.asarray(x, dtype=dtype))
faketorch.long = np.int64
faketorch.stack = lambda seq, dim=0: Tensor(np.stack([np.asarray(s) for s in seq], axis=dim))
faketorch.cat = lambda seq, dim=0: Tensor(np.concatenate([np.asarray(s) for s in seq], axis=dim))
faketorch.all = lambda x: bool(np.all(np.asarray(x)))
faketorch.isnan = lambda x: Tensor(np.isnan(np.asarray(x)))
faketorch.isfinite = lambda x: Tensor(np.isfinite(np.asarray(x)))
faketorch.from_numpy = lambda a: Tensor(np.asarray(a))
faketorch.mean = lambda x: Tensor(np.asarray(np.mean(np.asarray(x))))
faketorch.sum = lambda x: Tensor(np.asarray(np.sum(np.asarray(x))))
faketorch.no_grad = _NoGrad
faketorch.nn = _NS()
faketorch.optim = _NS()
faketorch.optim.Optimizer = type("Optimizer", (), {})
faketorch.nn.Module = _Module
faketorch.utils = _NS()
faketorch.utils.data = _NS()
faketorch.utils.data.Sampler = _Sampler
faketorch.utils.data.DataLoader = _DataLoader


class _Tqdm:
    def __init__(self, it, **k):
        self.it = it

    def __iter__(self):
        return iter(self.it)

    def set_description(self, *a):
        pass

    def set_postfix_str(self, *a):
        pass


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __class_getitem__(cls, item):
        return cls


# ----------------------------------------------------------------------------- extraction
def extract(rel_path: str, names: List[str]) -> Dict[str, str]:
    src = (REPO / rel_path).read_text()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            seg = ast.get_source_segment(src, node)
            # keep decorators (e.g. @njit) out; we exec the bare function
            out[node.name] = seg
    missing = set(names) - set(out)
    assert not missing, f"not found in {rel_path}: {missing}"
    return out


def extract_lines(rel_path: str, first_line_text: str, last_line_text: str) -> str:
    lines = (REPO / rel_path).read_text().splitlines()
    i0 = next(i for i, l in enumerate(lines) if l.strip() == first_line_text)
    i1 = next(i for i in range(i0, len(lines)) if lines[i].strip() == last_line_text and i > i0 + 15)
    return textwrap.dedent("\n".join(lines[i0:i1 + 1])), i0 + 1, i1 + 1


NS: Dict[str, Any] = dict(
    np=np, torch=faketorch, random=random, sys=sys, logging=logging, LOGGER=logging.getLogger("audit"),
    defaultdict=defaultdict, tqdm=_Tqdm, prange=range, Any=Any, Dict=Dict, Iterator=Iterator, List=List,
    Optional=Optional, Tuple=Tuple, Union=Union, Sampler=_Sampler, DataLoader=_DataLoader, Config=_Dummy,
    BaseDataset=_Dummy, BaseTester=_Dummy, BaseModel=_Dummy, BaseRegularization=_Dummy, Logger=_Dummy,
    Path=Path,
)
NS["loss"] = _NS()
NS["loss"].BaseLoss = _Dummy

from pandas.tseries.frequencies import to_offset
NS["to_offset"] = to_offset
for rel, names in [
    ("neuralhydrology/utils/persistentutils.py", ["uses_persistent_state", "_single_value", "single_seq_length",
                                                  "single_predict_last_n", "validate_persistent_config",
                                                  "frequency_time_step", "infer_time_step", "select_window_ends",
                                                  "contiguous_batches", "unflatten_time_batch",
                                                  "unflatten_predictions", "has_finite_targets", "state_is_finite"]),
    ("neuralhydrology/training/loss.py", ["BaseLoss", "MaskedMSELoss", "_get_predict_last_n"]),
    ("neuralhydrology/datasetzoo/basedataset.py", ["_validate_samples"]),
    ("neuralhydrology/training/basetrainer.py", ["BasinChronoInterleaveBatchSampler", "BaseTrainer"]),
    ("neuralhydrology/evaluation/tester.py", ["BaseTester", "RegressionTester"]),
]:
    for name, code in extract(rel, names).items():
        exec(compile(code, f"{rel}::{name}", "exec"), NS)
        if name == "BaseTester":
            NS["BaseTester"] = NS["BaseTester"]  # real class now replaces dummy

# BaseDataset.__getitem__ and collate_fn (methods): pull from the class body
_bd_src = (REPO / "neuralhydrology/datasetzoo/basedataset.py").read_text()
_bd_tree = ast.parse(_bd_src)
_bd_cls = next(n for n in _bd_tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseDataset")
_bd_methods = {n.name: " " * n.col_offset + ast.get_source_segment(_bd_src, n) for n in _bd_cls.body if isinstance(n, ast.FunctionDef)}
_cls_code = "class DatasetUnderTest:\n" + textwrap.indent(
    "@staticmethod\n" + textwrap.dedent(_bd_methods["collate_fn"]).replace("@staticmethod\n", "") + "\n\n"
    + textwrap.dedent(_bd_methods["__getitem__"]) + "\n\n"
    + "def __len__(self):\n    return len(self.lookup_table)\n", "    ")
exec(compile(_cls_code, "basedataset.py::BaseDataset(__getitem__, collate_fn)", "exec"), NS)

# ----------------------------------------------------------------------------- synthetic data
def build_dataset(series: Dict[str, Dict[str, np.ndarray]], seq_len: int, stride: int, is_train: bool,
                  non_overlapping=True, predict_last_n=None, persistent=True):
    """Builds a DatasetUnderTest. Window selection mirrors basedataset._create_lookup_table of the v2 code:
    _validate_samples (repo code) with y=None for persistent training, then select_window_ends (repo code)."""
    predict_last_n = seq_len if predict_last_n is None else predict_last_n
    ds = NS["DatasetUnderTest"]()
    ds.frequencies = ["1D"]
    ds.seq_len = [seq_len]
    ds.seq_stride = stride
    ds.non_overlapping_sequences = non_overlapping
    ds.is_train = is_train
    ds._persistent_training = is_train and persistent
    ds._basin_to_idx = {b: i for i, b in enumerate(series)}
    cfg = _NS()
    cfg.forecast_seq_length = None
    cfg.hindcast_inputs_flattened = []
    cfg.forecast_inputs_flattened = []
    cfg.nan_step_probability = None
    cfg.nan_sequence_probability = None
    cfg.timestep_counter = False
    ds.cfg = cfg
    ds._x_d, ds._y, ds._dates, ds._attributes, ds._x_s = {}, {}, {}, {}, {}
    ds._per_basin_target_stds, ds.id_to_int = {}, {}
    lookup = []
    for basin, d in series.items():
        n = len(d["t"])
        x_d = {"t": d["t"].reshape(-1, 1), "b": d["b"].reshape(-1, 1)}
        y = d["y"].reshape(-1, 1)
        frequency_maps = {"1D": np.arange(n)}
        x_d_validate = [np.concatenate([x_d["t"], x_d["b"]], axis=-1)] if is_train else None
        flag = NS["_validate_samples"](x_d=x_d_validate, x_s=None,
                                       y=[y] if (is_train and not ds._persistent_training) else None,
                                       seq_length=[seq_len], predict_last_n=[predict_last_n],
                                       frequency_maps=[frequency_maps["1D"]])
        if non_overlapping:
            valid = NS["select_window_ends"](flag, seq_len=seq_len, stride=stride)
            if ds._persistent_training and not np.any(np.isfinite(y)):
                valid = valid[:0]
        else:
            valid = np.argwhere(flag == 1).flatten()
            if stride > 1:
                valid = valid[::stride]
        for f in valid:
            lookup.append((basin, [frequency_maps[freq][int(f)] for freq in ds.frequencies]))
        ds._x_d[basin] = {"1D": {k: Tensor(v.astype(np.float32)) for k, v in x_d.items()}}
        ds._y[basin] = {"1D": Tensor(y.astype(np.float32))}
        ds._dates[basin] = {"1D": d["dates"]}
        ds._attributes[basin] = Tensor(np.array([float(ds._basin_to_idx[basin])], dtype=np.float32))
    ds.lookup_table = {i: e for i, e in enumerate(lookup)}
    return ds


def make_series(lengths: Dict[str, int], target_gap: Dict[str, Tuple[int, int]] = None,
                input_nan: Dict[str, int] = None, start="1990-01-01"):
    target_gap = target_gap or {}
    input_nan = input_nan or {}
    out = {}
    for code, (basin, n) in enumerate(lengths.items()):
        t = np.arange(n, dtype=np.float64)
        y = t.copy()
        if basin in target_gap:
            a, b = target_gap[basin]
            y[a:b] = np.nan
        tt = t.copy()
        if basin in input_nan:
            tt[input_nan[basin]] = np.nan
        out[basin] = dict(t=tt, b=np.full(n, float(code)), y=y,
                          dates=np.datetime64(start, 'D') + np.arange(n).astype('timedelta64[D]'))
    return out


# ----------------------------------------------------------------------------- counter model
class CounterLSTM:
    """Stand-in for nn.LSTM inside PersistentLSTMModel. Same call signature as
    PersistentLSTMModel.forward(data, hidden_state=None) and same output keys.
    h = [last absolute time index processed, basin code]; c = [steps since reset].
    y_hat[t] = steps since reset (so a continuous stream yields consecutive integers)."""

    def __init__(self):
        self.calls = []

    def train(self):
        pass

    def eval(self):
        pass

    def pre_model_hook(self, data, is_train):
        return data

    def __call__(self, data, hidden_state=None):
        t = np.asarray(data["x_d"]["t"])[0, :, 0]
        bcode = float(np.asarray(data["x_d"]["b"])[0, 0, 0])
        xs = np.asarray(data["x_s"])
        rec = dict(basin=bcode, t_first=t[0], t_last=t[-1], T=len(t), xs=xs.copy(), xs_shape=xs.shape,
                   internal_jumps=int(np.sum(np.diff(t) != 1)), jump_sizes=[int(x) - 1 for x in np.diff(t) if np.isfinite(x) and x != 1], h0=None)
        if hidden_state is None:
            steps0 = 0.0
        else:
            h, c = hidden_state
            rec["h0"] = (float(h[0, 0, 0]), float(h[0, 0, 1]), float(c[0, 0, 0]))
            steps0 = float(c[0, 0, 0])
        self.calls.append(rec)
        EVENTS.append("forward")
        y_hat = Tensor((steps0 + np.arange(1, len(t) + 1, dtype=np.float32)).reshape(1, -1, 1))
        if np.any(np.isnan(t)):
            steps0 = np.nan
            y_hat = Tensor(np.full((1, len(t), 1), np.nan, dtype=np.float32))
        h_new = Tensor(np.array([[[t[-1], bcode]]], dtype=np.float32))
        c_new = Tensor(np.array([[[steps0 + len(t)]]], dtype=np.float32))
        return {"y_hat": y_hat, "hidden_state": (h_new, c_new)}


class RecordingMSE:
    """Real MaskedMSELoss from the repo, wrapped to record which target values enter the loss."""

    def __init__(self, predict_last_n):
        cfg = _NS()
        cfg.predict_last_n = predict_last_n
        cfg.no_loss_frequencies = []
        cfg.target_loss_weights = None
        cfg.target_variables = ["y"]
        self.inner = NS["MaskedMSELoss"](cfg)
        self.seen = []
        inner_get = self.inner._get_loss

        def rec(prediction, ground_truth, **kw):
            g = np.asarray(ground_truth["y"]).reshape(-1)
            self.seen.append(g[~np.isnan(g)].copy())
            return inner_get(prediction, ground_truth, **kw)

        self.inner._get_loss = rec

    def __call__(self, preds, data):
        return self.inner(preds, data)

    def to(self, *a):
        return self


_orig_cache = NS["BaseTrainer"]._cache_persistent_state


def _recording_cache(cache, cache_end, basin_id, new_hidden, dates):
    EVENTS.append("cache")
    return _orig_cache(cache, cache_end, basin_id, new_hidden, dates)


NS["BaseTrainer"]._cache_persistent_state = staticmethod(_recording_cache)


class FakeOptim:
    def zero_grad(self):
        EVENTS.append("zero_grad")

    def step(self):
        EVENTS.append("step")


class FakeLogger:
    def train(self):
        pass

    def log_step(self, **k):
        pass


def make_trainer(ds, batch_size, seq_len, seed=0, ablate=False):
    BT = NS["BaseTrainer"]
    tr = object.__new__(BT)
    cfg = _NS()
    cfg.model = "persistentlstm"
    cfg.batch_size = batch_size
    cfg.clip_gradient_norm = None
    tr.cfg = cfg
    tr.persistent_state = True
    tr.device = "cpu"
    tr._max_updates_per_epoch = None
    tr.noise_sampler_y = None
    cfg.ablate_state_carry = ablate
    tr._disable_pbar = True
    tr._allow_subsequent_nan_losses = 0
    tr.model = CounterLSTM()
    tr.loss_obj = RecordingMSE(seq_len)
    tr.optimizer = FakeOptim()
    tr.experiment_logger = FakeLogger()
    batches = tr._build_basin_chrono_batches(ds)
    tr._basin_batch_sampler = NS["BasinChronoInterleaveBatchSampler"](basin_to_batches=batches, seed=seed)
    tr.loader = _DataLoader(ds, batch_sampler=tr._basin_batch_sampler, collate_fn=ds.collate_fn)
    return tr


def run_epochs(tr, epochs):
    per_epoch = []
    for ep in epochs:
        tr._basin_batch_sampler.set_epoch(ep)
        tr.model.calls.clear()
        tr.loss_obj.seen.clear()
        tr._train_epoch(epoch=ep)
        per_epoch.append((list(tr.model.calls), [s.copy() for s in tr.loss_obj.seen]))
    return per_epoch
