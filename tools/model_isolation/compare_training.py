"""Check that the persistent-state changes of this fork leave every other model's training and evaluation unchanged.

The script trains and evaluates the same configurations with several code trees in the same Python environment and
compares the outputs bit for bit:

* ``upstream``      original NeuralHydrology (commit 01510be, the base of this fork)
* ``upstream303``   the same commit plus the upstream fix #303 of ``dynamic_learning_rate`` (commit 9528d1d)
* ``v1``            this fork before version 2 (commit a3025c8)
* ``v2``            this fork, version 2 as first merged (commit 0a40022)
* ``v2fixed``       the checked-out tree (version 2 with the model-isolation fixes)

Every run happens in a fresh process whose ``neuralhydrology`` package is imported from one tree only (checked).
Inputs are identical for all trees: the configurations and test data of the upstream tree.

Compared per run: all saved weights and optimizer states (``torch.equal`` on every tensor), test and validation
results (every array, NaN-aware), metric files, the logged epoch losses, the learning rate at the start of every
epoch, the type of the training data sampler, the keys of a training sample and a hash of every training sample.

numba's random generator (used by the random holdout of dynamic inputs) is seeded identically for every run, because
NeuralHydrology's seeding does not reach it.

It also runs two checks that only apply to the fork:

* ``persistent``: instruments a Persistent-State LSTM run and checks the state hand-over batch by batch (state of
  basin b in batch t+1 == detached final state of basin b in batch t, reset at every epoch and at every basin in
  evaluation, update before caching, every training window once per epoch).
* ``switching``: runs Persistent-State LSTM, CudaLSTM and MTS-LSTM one after another in one process and compares the
  CudaLSTM/MTS-LSTM outputs with runs in fresh processes.

Usage (see .github/workflows/model-isolation.yml)::

    python compare_training.py drive --group core --data-root <upstream tree> --out <dir> \
        --tree upstream=<path> --tree v1=<path> --tree v2=<path> --tree v2fixed=<path> ...

The exit code is 1 if any expectation fails (``v2fixed`` must match its reference everywhere; the persistent checks
must pass). Differences of ``v1`` and ``v2`` are reported, not enforced.
"""
import argparse
import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from pathlib import Path

DAILY = {
    "dataset": "camels_us",
    "forcings": "daymet",
    "dynamic_inputs": ["prcp(mm/day)", "tmax(C)"],
    "target_variables": ["QObs(mm/d)"],
    "data_dir": "test/test_data/camels_us",
}

UNCERTAINTY_COMMON = {"n_samples": 10, "negative_sample_handling": "clip", "negative_sample_max_retries": 1}
UNCERTAINTY_HEADS = {
    "umal": {"head": "umal", "loss": "UMALLoss", "n_taus": 32, "umal_extend_batch": True, "tau_down": 0.1,
             "tau_up": 0.9},
    "cmal": {"head": "cmal", "loss": "CMALLoss", "n_distributions": 3},
    "gmm": {"head": "gmm", "loss": "GMMLoss", "n_distributions": 3},
}

# Settings of the Persistent-State LSTM run used by the 'persistent' and 'switching' checks (test data, 4 basins).
PERSISTENT = dict(DAILY, model="persistentlstm", persistent_state=True, non_overlapping_sequences=True, seq_length=30,
                  predict_last_n=30, seq_stride=30, batch_size=4, epochs=2, eval_warmup_steps=60,
                  eval_warmup_from_preceding_period=True, log_n_figures=0)

ALL_TREES = ["upstream", "v1", "v2", "v2fixed"]

# The manuscript configs of this repository (test/test_configs), run as they are except for what points to data and
# hardware: the 4-basin test data, its periods, CPU, 2 epochs, validation and weights after every epoch.
REPO_TEST_DATA = {
    "train_basin_file": "test/test_data/4_basins_test_set.txt",
    "validation_basin_file": "test/test_data/4_basins_test_set.txt",
    "test_basin_file": "test/test_data/4_basins_test_set.txt",
    "train_start_date": "01/01/2000", "train_end_date": "31/12/2001",
    "validation_start_date": "01/01/2001", "validation_end_date": "31/12/2002",
    "test_start_date": "01/01/2001", "test_end_date": "31/12/2002",
    "data_dir": "test/test_data/camels_us", "device": "cpu", "num_workers": 0, "epochs": 2, "validate_every": 1,
    "save_weights_every": 1, "log_tensorboard": False,
}
REPO_BASELINE_CONFIGS = ["cuda_lstm_for_long_memory_basins", "cuda_lstm_nonstationary_basins", "mts_lstm_hourly"]
REPO_PERSISTENT_CONFIGS = ["persistent_lstm_for_long_memory_basins", "persistent_lstm_nonstationary_basins",
                           "persistent_lstm_hourly"]  # persistent_lstm_15_mint: no 15-min test data
# the nonstationary configs use a 'Tmean(C)' forcing column that the CAMELS-US NLDAS test files do not have
NEEDS_TMEAN = {"cuda_lstm_nonstationary_basins", "persistent_lstm_nonstationary_basins"}


def test_data_with_tmean(data_root: Path, out: Path) -> Path:
    """Copy of the test data whose NLDAS files gain a 'Tmean(C)' column = (Tmax + Tmin) / 2 (test input only)."""
    import shutil
    target = out / "test_data_tmean" / "camels_us"
    if target.is_dir():
        return target
    shutil.copytree(data_root / "test" / "test_data" / "camels_us", target)
    for f in sorted((target / "basin_mean_forcing" / "nldas").glob("*_forcing_leap.txt")):
        lines = f.read_text().splitlines()
        header = lines[3].split()
        i_max, i_min = header.index("Tmax(C)"), header.index("Tmin(C)")
        new = lines[:3] + [lines[3] + "\tTmean(C)"]
        for line in lines[4:]:
            tokens = line.split()
            new.append(line + f"\t{(float(tokens[i_max]) + float(tokens[i_min])) / 2:.2f}" if tokens else line)
        f.write_text("\n".join(new) + "\n")
    return target


def _repo_update(name: str, tmean_dir):
    update = dict(REPO_TEST_DATA)
    if name in NEEDS_TMEAN:
        update["data_dir"] = str(tmean_dir)  # absolute, so it is not joined to the data root
    return update


def _spec(config, update=None, kind="train_eval", eval_epoch=None, finetune=None, config_file=None):
    return {"config": config, "update": update or {}, "kind": kind, "eval_epoch": eval_epoch, "finetune": finetune,
            "config_file": config_file}


def scenarios(group, repo=None, tmean_dir=None):
    """Scenario name -> (run spec, trees to run, reference tree, trees that must match the reference)."""
    out = {}
    if group == "repo_baselines":
        for name in REPO_BASELINE_CONFIGS:
            out[f"repo_{name}"] = (_spec(name, _repo_update(name, tmean_dir), eval_epoch=2,
                                         config_file=str(repo / "test" / "test_configs" / f"{name}.test.yml")),
                                   ["upstream", "v1", "v2fixed"], "upstream", ["v2fixed"])
    if group in ("core", "all"):
        cuda = dict(DAILY, model="cudalstm", epochs=2, validate_n_random_basins=2, save_validation_results=True)
        out["cudalstm"] = (_spec("daily_regression", cuda, eval_epoch=2),
                           ["upstream", "upstream303", "v1", "v2", "v2fixed"], "upstream",
                           ["upstream303", "v2fixed"])
        base = dict(DAILY, model="cudalstm")
        out["finetune_head"] = (_spec("daily_regression", base, kind="finetune",
                                      finetune={"finetune_modules": ["head"], "epochs": 1,
                                                "learning_rate": 0.001}),
                                ALL_TREES, "upstream", ["v2fixed"])
        out["target_noise"] = (_spec("daily_regression", dict(base, target_noise_std=0.2), eval_epoch=1),
                               ALL_TREES, "upstream", ["v2fixed"])
        dyn = dict(base, dynamic_learning_rate=True, patience_dynamic_learning_rate=0, factor_dynamic_learning_rate=0.5,
                   validate_every=1, epochs=4, validate_n_random_basins=2, save_validation_results=True)
        out["dynamic_lr"] = (_spec("daily_regression", dyn, eval_epoch=4),
                             ["upstream303", "upstream", "v1", "v2", "v2fixed"], "upstream303", ["v2fixed"])
        # options that exist only in the fork (not in upstream): reference is v1, the fork before version 2
        fork_trees = ["v1", "v2", "v2fixed"]
        out["fork_non_overlapping"] = (_spec("daily_regression", dict(base, non_overlapping_sequences=True),
                                             eval_epoch=1), fork_trees, "v1", ["v2fixed"])
        out["fork_non_overlapping_stride7"] = (_spec("daily_regression",
                                                     dict(base, non_overlapping_sequences=True, seq_stride=7),
                                                     eval_epoch=1), fork_trees, "v1", ["v2fixed"])
        out["fork_seq_stride3"] = (_spec("daily_regression", dict(base, seq_stride=3), eval_epoch=1), fork_trees,
                                   "v1", ["v2fixed"])
    if group in ("models", "all"):
        for model in ["ealstm", "customlstm", "gru"]:
            out[model] = (_spec("daily_regression", dict(DAILY, model=model), eval_epoch=1), ALL_TREES, "upstream",
                          ["v2fixed"])
        out["arlstm"] = (_spec("autoregression_daily_regression", dict(DAILY, model="arlstm"), eval_epoch=1),
                         ALL_TREES, "upstream", ["v2fixed"])
        out["transformer"] = (_spec("transformer_daily_regression", eval_epoch=1), ALL_TREES, "upstream", ["v2fixed"])
        out["embedding_cudalstm"] = (_spec("daily_regression_with_embedding", eval_epoch=1), ALL_TREES, "upstream",
                                     ["v2fixed"])
        out["additional_features"] = (_spec("daily_regression_additional_features", eval_epoch=1), ALL_TREES,
                                      "upstream", ["v2fixed"])
        out["nan_targets"] = (_spec("daily_regression_nan_targets", eval_epoch=1), ALL_TREES, "upstream",
                              ["v2fixed"])
        for model in ["mtslstm", "odelstm"]:
            out[model] = (_spec("multi_timescale_regression", {"model": model}, eval_epoch=1), ALL_TREES, "upstream",
                          ["v2fixed"])
        for head, head_cfg in UNCERTAINTY_HEADS.items():
            upd = dict(DAILY, **UNCERTAINTY_COMMON, **head_cfg, mc_dropout=False)
            out[f"uncertainty_{head}"] = (_spec("daily_uncertainty", upd, eval_epoch=1), ALL_TREES, "upstream",
                                          ["v2fixed"])
        upd = dict(DAILY, **UNCERTAINTY_COMMON, **UNCERTAINTY_HEADS["cmal"], mc_dropout=True)
        out["uncertainty_cmal_mcdropout"] = (_spec("daily_uncertainty", upd, eval_epoch=1), ALL_TREES, "upstream",
                                             ["v2fixed"])
        # models without an upstream test configuration
        out["embcudalstm"] = (_spec("daily_regression_with_embedding", {"model": "embcudalstm"}, eval_epoch=1),
                              ALL_TREES, "upstream", ["v2fixed"])
        out["mclstm"] = (_spec("daily_regression", dict(DAILY, model="mclstm", mass_inputs=["prcp(mm/day)"],
                                                         dynamic_inputs=["tmax(C)"]), eval_epoch=1),
                         ALL_TREES, "upstream", ["v2fixed"])
        forecast = dict(DAILY, hindcast_inputs=["prcp(mm/day)", "tmax(C)"], forecast_inputs=["prcp(mm/day)", "tmax(C)"],
                        seq_length=30, forecast_seq_length=7, predict_last_n=7)  # Config checks both together
        fc = {"type": "fc", "hiddens": [16], "activation": "tanh", "dropout": 0.0}
        forecast_models = {
            "handoff_forecast_lstm": dict(forecast, forecast_overlap=3, hindcast_hidden_size=16, forecast_hidden_size=16,
                                          state_handoff_network=fc),
            "multihead_forecast_lstm": dict(forecast, forecast_network=fc),
            "sequential_forecast_lstm": dict(forecast),
            # stacked model: hindcast length (seq_length - forecast_seq_length + forecast_overlap) = forecast_seq_length
            "stacked_forecast_lstm": dict(forecast, forecast_seq_length=20, forecast_overlap=10, hindcast_hidden_size=16,
                                          forecast_hidden_size=16, bidirectional_stacked_forecast_lstm=False),
        }
        for model, upd in forecast_models.items():
            out[model] = (_spec("daily_regression", dict(upd, model=model), eval_epoch=1), ALL_TREES, "upstream",
                          ["v2fixed"])
    return out


# ======================================================================================================================
# Part 1: runs inside one tree (fresh process)
# ======================================================================================================================


def _absolute_paths(update: dict, data_root: Path) -> dict:
    update = copy.deepcopy(update)
    for key in update:
        if key == "data_dir" or key.endswith("_basin_file"):
            update[key] = str(data_root / update[key])
    return update


def _hash_value(h, value):
    import numpy as np
    import torch
    if isinstance(value, dict):
        for k in sorted(value):
            h.update(str(k).encode())
            _hash_value(h, value[k])
    elif torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
        h.update(str(arr.dtype).encode() + str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    elif isinstance(value, np.ndarray):
        h.update(str(value.dtype).encode() + str(value.shape).encode())
        if value.dtype.kind in "OUS":
            h.update(str(value.tolist()).encode())
        else:
            h.update(np.ascontiguousarray(value).tobytes())
    else:
        h.update(repr(value).encode())


def _dataset_fingerprint(ds) -> dict:
    full, without_basin_idx = hashlib.sha256(), hashlib.sha256()
    keys = None
    for i in range(len(ds)):
        sample = ds[i]
        if keys is None:
            keys = sorted(sample.keys())
        _hash_value(full, sample)
        _hash_value(without_basin_idx, {k: v for k, v in sample.items() if k != "basin_idx"})
    lookup = hashlib.sha256(repr([ds.lookup_table[i] for i in range(len(ds))]).encode()).hexdigest()
    return {"n_samples": len(ds), "sample_keys": keys, "lookup_sha256": lookup, "samples_sha256": full.hexdigest(),
            "samples_without_basin_idx_sha256": without_basin_idx.hexdigest()}


class Probe:
    """Wraps NeuralHydrology functions to record what a run executes. Upstream functions are wrapped without changing
    their behavior; fork-only functions are counted if they exist."""

    FORK_ONLY = [("neuralhydrology.training.basetrainer", "BaseTrainer", "_build_basin_chrono_batches"),
                 ("neuralhydrology.training.basetrainer", "BaseTrainer", "_cache_persistent_state"),
                 ("neuralhydrology.training.basetrainer", "BasinChronoInterleaveBatchSampler", "__iter__"),
                 ("neuralhydrology.evaluation.tester", "BaseTester", "_get_chrono_loader"),
                 ("neuralhydrology.evaluation.tester", "BaseTester", "_evaluate_persistent_match_training_with_flatten")]

    def __init__(self, instrument_persistent: bool = False):
        import importlib
        import torch
        self.torch = torch
        self.trainers = []
        self.lr_trace = []
        self.counts = {}
        self.events = []
        self.instrument_persistent = instrument_persistent
        basetrainer = importlib.import_module("neuralhydrology.training.basetrainer")
        tr = basetrainer.BaseTrainer
        probe = self

        orig_init = tr.initialize_training

        def initialize_training(trainer, *a, **k):
            result = orig_init(trainer, *a, **k)
            probe.trainers.append(trainer)
            if probe.instrument_persistent:
                orig_step = trainer.optimizer.step

                def step(*sa, **sk):
                    probe.events.append({"ev": "step"})
                    return orig_step(*sa, **sk)

                trainer.optimizer.step = step
            return result

        tr.initialize_training = initialize_training

        orig_epoch = tr._train_epoch

        def _train_epoch(trainer, epoch):
            probe.lr_trace.append([epoch, [g["lr"] for g in trainer.optimizer.param_groups]])
            probe.events.append({"ev": "epoch", "epoch": epoch})
            return orig_epoch(trainer, epoch=epoch)

        tr._train_epoch = _train_epoch

        for module_name, cls_name, fn_name in self.FORK_ONLY:
            module = importlib.import_module(module_name)
            cls = getattr(module, cls_name, None)
            if cls is None or fn_name not in cls.__dict__:
                continue
            self._count(cls, fn_name, f"{cls_name}.{fn_name}")

        if instrument_persistent:
            self._instrument_persistent()

    def _count(self, cls, fn_name, label):
        raw = cls.__dict__[fn_name]
        is_static = isinstance(raw, staticmethod)
        func = raw.__func__ if is_static else raw
        probe = self
        self.counts[label] = 0

        def wrapper(*a, **k):
            probe.counts[label] += 1
            if probe.instrument_persistent:
                if label == "BaseTrainer._cache_persistent_state":
                    # (cache, cache_end, basin_id, new_hidden, dates)
                    import numpy as np
                    basin_id, new_hidden, dates = a[2], a[3], np.asarray(a[4])
                    probe.events.append({"ev": "cache", "basin": int(basin_id), "first": dates[0, 0],
                                         "last": dates[-1, -1], "n_windows": int(dates.shape[0]),
                                         "window_starts": [d for d in dates[:, 0]],
                                         "h": None if new_hidden is None else new_hidden[0].detach().clone(),
                                         "c": None if new_hidden is None else new_hidden[1].detach().clone()})
                elif label == "BaseTester._evaluate_persistent_match_training_with_flatten":
                    tester = a[0]
                    probe.events.append({"ev": "eval_basin", "basin": getattr(tester, "_persistent_basin_eval", None),
                                         "period": tester.period})
                    try:
                        return func(*a, **k)
                    finally:
                        probe.events.append({"ev": "eval_basin_end"})
            return func(*a, **k)

        setattr(cls, fn_name, staticmethod(wrapper) if is_static else wrapper)

    def _instrument_persistent(self):
        from neuralhydrology.modelzoo import persistentlstm
        cls = persistentlstm.PersistentLSTMModel
        orig_forward = cls.forward
        probe = self

        def forward(model, data, hidden_state=None, **kwargs):
            out = orig_forward(model, data, hidden_state=hidden_state, **kwargs)
            any_xd = next(iter(data["x_d"].values()))
            h_out, c_out = out["hidden_state"]
            probe.events.append({
                "ev": "fwd", "training": bool(model.training), "B": int(any_xd.shape[0]), "T": int(any_xd.shape[1]),
                "h_in": None if hidden_state is None else hidden_state[0].detach().clone(),
                "c_in": None if hidden_state is None else hidden_state[1].detach().clone(),
                "h_in_detached": None if hidden_state is None else (hidden_state[0].grad_fn is None
                                                                    and not hidden_state[0].requires_grad
                                                                    and hidden_state[1].grad_fn is None),
                "h_out_has_graph": h_out.grad_fn is not None,
                "h_out": h_out.detach().clone(), "c_out": c_out.detach().clone()})
            return out

        cls.forward = forward


def _equal_state(a, b, torch):
    if a is None or b is None:
        return a is None and b is None
    return torch.equal(a, b)


def analyse_persistent(probe: Probe, step) -> dict:
    """Checks of the state hand-over from the recorded events (manuscript Eq. 11-14, Algorithm 1)."""
    import numpy as np
    torch = probe.torch
    trainer = probe.trainers[-1]
    ds = trainer.loader.dataset
    windows_per_basin = {}
    for i in range(len(ds)):
        basin, _ = ds.lookup_table[i]
        b = ds._basin_to_idx[basin]
        windows_per_basin[b] = windows_per_basin.get(b, 0) + 1

    res = {"train_batches": 0, "train_updates": 0, "first_batch_of_basin_with_zero_state": 0,
           "first_batch_of_basin_with_carried_state": 0, "carried_state_equals_own_previous_final_state": 0,
           "carried_state_mismatch": 0, "carried_state_not_detached": 0, "carried_state_from_other_basin": 0,
           "gap_resets": 0, "missed_carry": 0, "non_chronological": 0, "order_violations": 0,
           "output_state_without_graph": 0, "windows_per_epoch_ok": True, "duplicate_windows": 0,
           "basin_switches_per_epoch": [], "basin_orders_differ_between_epochs": None,
           "eval_streams": 0, "eval_stream_start_with_zero_state": 0, "eval_stream_start_with_carried_state": 0,
           "eval_continuations": 0, "eval_continuation_equal": 0, "eval_continuation_reset_or_mismatch": 0,
           "eval_steps_per_stream": {}}

    epoch, last, seen, order, orders = None, {}, {}, [], []
    pending_fwd, pending_step = None, False
    prev_basin, prev_basin_state = None, None
    eval_stack, eval_prev_out, eval_period = None, None, None

    def close_epoch():
        if epoch is None:
            return
        res["basin_switches_per_epoch"].append(sum(1 for i in range(1, len(order)) if order[i] != order[i - 1]))
        orders.append(list(order))
        for b, n_expected in windows_per_basin.items():
            starts = seen.get(b, [])
            if len(starts) != n_expected:
                res["windows_per_epoch_ok"] = False
            res["duplicate_windows"] += len(starts) - len(set(starts))

    for ev in probe.events:
        kind = ev["ev"]
        if kind == "epoch":
            close_epoch()
            epoch, last, seen, order = ev["epoch"], {}, {}, []
            prev_basin, prev_basin_state = None, None
        elif kind == "fwd" and ev["training"]:
            if pending_fwd is not None:
                res["order_violations"] += 1  # a forward pass without a cache update
            pending_fwd, pending_step = ev, False
        elif kind == "step":
            if pending_fwd is None:
                res["order_violations"] += 1  # an update outside a batch
            pending_step = True
        elif kind == "cache":
            if pending_fwd is None:
                res["order_violations"] += 1
                continue
            fwd, b = pending_fwd, ev["basin"]
            res["train_batches"] += 1
            res["train_updates"] += int(pending_step)
            if not fwd["h_out_has_graph"]:
                res["output_state_without_graph"] += 1
            order.append(b)
            seen.setdefault(b, []).extend(ev["window_starts"])
            if b not in last:
                if fwd["h_in"] is None:
                    res["first_batch_of_basin_with_zero_state"] += 1
                else:
                    res["first_batch_of_basin_with_carried_state"] += 1
            else:
                prev_last, prev_h, prev_c = last[b]
                if ev["first"] <= prev_last:
                    res["non_chronological"] += 1
                contiguous = ev["first"] == prev_last + step
                if fwd["h_in"] is None:
                    res["gap_resets" if not contiguous else "missed_carry"] += 1
                else:
                    if not fwd["h_in_detached"]:
                        res["carried_state_not_detached"] += 1
                    if contiguous and _equal_state(fwd["h_in"], prev_h, torch) and _equal_state(fwd["c_in"], prev_c,
                                                                                               torch):
                        res["carried_state_equals_own_previous_final_state"] += 1
                    else:
                        res["carried_state_mismatch"] += 1
            if fwd["h_in"] is not None and prev_basin is not None and prev_basin != b and \
                    _equal_state(fwd["h_in"], prev_basin_state, torch):
                res["carried_state_from_other_basin"] += 1
            last[b] = (ev["last"], ev["h"], ev["c"])
            prev_basin, prev_basin_state = b, ev["h"]
            pending_fwd = None
        elif kind == "eval_basin":
            eval_stack, eval_prev_out, eval_period = ev["basin"], None, ev["period"]
            res["eval_streams"] += 1
        elif kind == "eval_basin_end":
            eval_stack = None
        elif kind == "fwd" and not ev["training"]:
            if eval_stack is None:
                continue
            key = f"{eval_period}/{eval_stack}"
            res["eval_steps_per_stream"][key] = res["eval_steps_per_stream"].get(key, 0) + ev["B"] * ev["T"]
            if eval_prev_out is None:
                if ev["h_in"] is None:
                    res["eval_stream_start_with_zero_state"] += 1
                else:
                    res["eval_stream_start_with_carried_state"] += 1
            else:
                res["eval_continuations"] += 1
                if _equal_state(ev["h_in"], eval_prev_out[0], torch) and _equal_state(ev["c_in"], eval_prev_out[1],
                                                                                      torch):
                    res["eval_continuation_equal"] += 1
                else:
                    res["eval_continuation_reset_or_mismatch"] += 1
            eval_prev_out = (ev["h_out"], ev["c_out"])
    close_epoch()
    if len(orders) >= 2:
        res["basin_orders_differ_between_epochs"] = orders[0] != orders[1]
    res["windows_per_basin"] = {str(k): v for k, v in windows_per_basin.items()}
    res["n_basins"] = len(windows_per_basin)
    # a few eval streams are enough in the report
    res["eval_steps_per_stream"] = dict(list(res["eval_steps_per_stream"].items())[:8])
    passed = (res["train_batches"] > 0 and res["order_violations"] == 0
              and res["first_batch_of_basin_with_carried_state"] == 0
              and res["first_batch_of_basin_with_zero_state"] == res["n_basins"] * len(orders)
              and res["carried_state_mismatch"] == 0 and res["carried_state_not_detached"] == 0
              and res["carried_state_from_other_basin"] == 0 and res["missed_carry"] == 0
              and res["non_chronological"] == 0 and res["output_state_without_graph"] == 0
              and res["windows_per_epoch_ok"] and res["duplicate_windows"] == 0
              and res["carried_state_equals_own_previous_final_state"] > 0
              and res["eval_streams"] > 0 and res["eval_stream_start_with_carried_state"] == 0
              and res["eval_continuation_reset_or_mismatch"] == 0 and res["eval_continuation_equal"] > 0)
    res["passed"] = bool(passed)
    return res


def _train_and_eval(spec: dict, data_root: Path, run_parent: Path, meta: dict, probe: Probe):
    from neuralhydrology.evaluation.evaluate import start_evaluation
    from neuralhydrology.training.train import start_training
    from neuralhydrology.utils.config import Config

    config_file = spec.get("config_file") or data_root / "test" / "test_configs" / f"{spec['config']}.test.yml"
    cfg = Config(Path(config_file))
    cfg.update_config(_absolute_paths(spec["update"], data_root))
    run_parent.mkdir(parents=True, exist_ok=True)
    cfg.run_dir = run_parent

    try:  # fork only: the persistent-config validation must not touch a non-persistent config
        from neuralhydrology.utils.persistentutils import validate_persistent_config
        before = repr(sorted(cfg.as_dict().items()))
        validate_persistent_config(cfg)
        meta["validate_persistent_config_left_cfg_unchanged"] = before == repr(sorted(cfg.as_dict().items()))
    except ImportError:
        pass

    start_training(cfg)
    run_dir = cfg.run_dir
    meta["run_dir"] = str(run_dir)
    trainer = probe.trainers[-1]
    loader = trainer.loader
    meta["loader"] = {"sampler": type(loader.sampler).__name__,
                      "batch_sampler": type(loader.batch_sampler).__name__ if loader.batch_sampler is not None else None,
                      "batch_size": loader.batch_size, "drop_last": loader.drop_last}

    if spec["kind"] == "finetune":
        from neuralhydrology import nh_run
        ft = dict(spec["finetune"], base_run_dir=str(run_dir), run_dir=str(run_parent / "finetune"),
                  experiment_name="finetune")
        (run_parent / "finetune").mkdir(parents=True, exist_ok=True)
        ft_file = run_parent / "finetune.yml"
        from ruamel.yaml import YAML
        with ft_file.open("w") as fp:
            YAML().dump(ft, fp)
        n_before = len(probe.trainers)
        nh_run.finetune(config_file=ft_file)
        ft_trainer = probe.trainers[n_before]
        meta["finetune_run_dir"] = str(ft_trainer.cfg.run_dir)
        meta["finetune_trainable"] = sorted(n for n, p in ft_trainer.model.named_parameters() if p.requires_grad)
        meta["finetune_frozen"] = sorted(n for n, p in ft_trainer.model.named_parameters() if not p.requires_grad)
        start_evaluation(cfg=ft_trainer.cfg, run_dir=ft_trainer.cfg.run_dir, epoch=1, period="test")
    else:
        start_evaluation(cfg=cfg, run_dir=run_dir, epoch=spec["eval_epoch"], period="test")

    meta["dataset"] = _dataset_fingerprint(trainer.loader.dataset)
    try:
        meta["test_result_dates"] = _result_dates(Path(meta.get("finetune_run_dir", run_dir)))
    except Exception as err:  # informational only
        meta["test_result_dates"] = {"error": repr(err)}
    return trainer


def _result_dates(run_dir: Path) -> dict:
    """Per frequency: number of scored dates of the first basin, first and last date (test period)."""
    import numpy as np
    files = sorted(run_dir.glob("test/*/test_results.p"))
    if not files:
        return {}
    with files[-1].open("rb") as fp:
        results = pickle.load(fp)
    basin = sorted(results)[0]
    out = {"basin": basin}
    for freq, res in results[basin].items():
        xr = res["xr"]
        sim = [v for v in xr.data_vars if v.endswith("_sim")][0]
        values = xr[sim]
        dim = "date" if "date" in values.dims else "datetime"
        if "time_step" in values.dims:
            values = values.isel(time_step=-1)
        if "samples" in values.dims:
            values = values.mean("samples")
        dates = values[dim].values[np.isfinite(values.values)]
        out[freq] = {"n_scored": int(dates.size), "first": str(dates.min()) if dates.size else None,
                     "last": str(dates.max()) if dates.size else None}
    return out


def _seed_numba_rng(seed: int):
    """Seed numba's own random generator, the same way in every tree.

    NeuralHydrology seeds Python, NumPy and torch, but numba-compiled functions draw from numba's separate generator,
    which np.random.seed does not reach. The random holdout of dynamic inputs (samplingutils.bernoulli_subseries_sampler,
    used by the ARLSTM test configuration) draws from it, so without this even two upstream runs differ.
    """
    import numpy as np
    from numba import njit

    @njit
    def _seed(value):
        np.random.seed(value)

    _seed(seed)


def run_in_tree(args):
    """Entry point of a fresh process: one or several runs with the neuralhydrology package of one tree."""
    tree = Path(args.tree).resolve()
    sys.path.insert(0, str(tree))
    import neuralhydrology
    import torch
    nh_file = Path(neuralhydrology.__file__).resolve()
    if tree not in nh_file.parents:
        raise RuntimeError(f"neuralhydrology was imported from {nh_file}, not from tree {tree}")
    torch.set_num_threads(1)

    specs = json.loads(Path(args.spec).read_text())
    out = Path(args.out)
    probe = Probe(instrument_persistent=args.instrument_persistent)

    # NeuralHydrology logs through logging.basicConfig, which configures the root logger only once per process: in a
    # process with several runs, later runs write no output.log of their own. The handler below records the messages of
    # every run in the same way, without touching the root logger.
    import logging
    messages = []

    class _Collect(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if any(k in msg for k in LOG_KEYS):
                messages.append(msg.strip())

    logging.getLogger("neuralhydrology").addHandler(_Collect(level=logging.INFO))
    for i, spec in enumerate(specs):
        sub = out / (f"step{i}_{spec['name']}" if len(specs) > 1 else "")
        sub.mkdir(parents=True, exist_ok=True)
        meta = {"name": spec["name"], "tree": str(tree), "nh_file": str(nh_file), "torch": torch.__version__,
                "started": time.time()}
        probe.lr_trace, probe.events = [], []
        messages.clear()
        _seed_numba_rng(111)
        try:
            _train_and_eval(spec, Path(args.data_root), sub / "runs", meta, probe)
            meta["lr_trace"] = probe.lr_trace
            meta["log_messages"] = list(messages)
            log_file = Path(meta["run_dir"]) / "output.log"
            meta["output_log_lines"] = len(log_file.read_text().splitlines()) if log_file.is_file() else None
            meta["probe_counts"] = dict(probe.counts)
            if args.instrument_persistent:
                import numpy as np
                step = getattr(probe.trainers[-1], "_persistent_time_step", None)
                meta["persistent_time_step"] = str(step)
                meta["persistent_checks"] = analyse_persistent(probe, step if step is not None else
                                                               np.timedelta64(1, "D"))
            meta["error"] = None
        except Exception:  # record the failure and keep the remaining runs going
            meta["error"] = traceback.format_exc()
        meta["seconds"] = round(time.time() - meta["started"], 1)
        (sub / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        probe.counts = {k: 0 for k in probe.counts}


# ======================================================================================================================
# Part 2: comparison of two finished runs
# ======================================================================================================================


def _compare_objects(a, b, path=""):
    import numpy as np
    import xarray as xr
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys differ ({sorted(map(str, set(a) ^ set(b)))[:6]})"]
        diffs = []
        for k in sorted(a, key=str):
            diffs += _compare_objects(a[k], b[k], f"{path}/{k}")
        return diffs
    if isinstance(a, xr.Dataset) and isinstance(b, xr.Dataset):
        diffs = []
        if set(a.data_vars) != set(b.data_vars) or set(a.coords) != set(b.coords):
            return [f"{path}: variables differ"]
        for name in list(a.data_vars) + list(a.coords):
            diffs += _compare_objects(a[name].values, b[name].values, f"{path}/{name}")
        return diffs
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return [f"{path}: shape {a.shape} vs {b.shape}"]
        nan_ok = a.dtype.kind in "fcmM" and b.dtype.kind in "fcmM"
        if not np.array_equal(a, b, equal_nan=nan_ok):
            if a.dtype.kind == "f":
                diff = np.nanmax(np.abs(a.astype(float) - b.astype(float)))
                return [f"{path}: values differ (max abs diff {diff:.3g})"]
            return [f"{path}: values differ"]
        return []
    if isinstance(a, float) and isinstance(b, float):
        return [] if (a == b or (a != a and b != b)) else [f"{path}: {a} vs {b}"]
    return [] if a == b else [f"{path}: {str(a)[:60]} vs {str(b)[:60]}"]


def _files(run_dir: Path, pattern: str):
    return {str(p.relative_to(run_dir)): p for p in sorted(run_dir.glob(pattern))}


LOG_KEYS = ("average loss", "average validation loss", "Setting learning rate")


def _log_lines(run_dir: Path):
    log = run_dir / "output.log"
    if not log.is_file():
        return []
    return [line.split(": ", 1)[-1].strip() for line in log.read_text().splitlines() if any(k in line for k in LOG_KEYS)]


def compare_runs(ref_dir: Path, cand_dir: Path) -> dict:
    import torch
    ref_meta = json.loads((ref_dir / "meta.json").read_text())
    cand_meta = json.loads((cand_dir / "meta.json").read_text())
    if ref_meta.get("error") or cand_meta.get("error"):
        return {"status": "error", "details": [f"reference error: {bool(ref_meta.get('error'))}",
                                               f"candidate error: {bool(cand_meta.get('error'))}"]}
    details = []
    for key in ["run_dir", "finetune_run_dir"]:
        if key not in ref_meta:
            continue
        r, c = Path(ref_meta[key]), Path(cand_meta[key])
        tag = "finetune " if key.startswith("finetune") else ""
        for pattern in ["model_epoch*.pt", "optimizer_state_epoch*.pt"]:
            rf, cf = _files(r, pattern), _files(c, pattern)
            if set(rf) != set(cf):
                details.append(f"{tag}{pattern}: files differ ({sorted(rf)} vs {sorted(cf)})")
                continue
            for name in rf:
                sa = torch.load(rf[name], map_location="cpu", weights_only=False)
                sb = torch.load(cf[name], map_location="cpu", weights_only=False)
                if pattern.startswith("model"):
                    bad = [k for k in sa if k not in sb or not torch.equal(sa[k], sb[k])]
                    if bad or set(sa) != set(sb):
                        details.append(f"{tag}{name}: {len(bad)} of {len(sa)} tensors differ (e.g. {bad[:3]})")
                else:
                    diffs = _compare_tensors_nested(sa, sb, torch)
                    if diffs:
                        details.append(f"{tag}{name}: optimizer state differs ({diffs[0]})")
        for pattern in ["test/*/test_results.p", "validation/*/validation_results.p"]:
            rf, cf = _files(r, pattern), _files(c, pattern)
            if set(rf) != set(cf):
                details.append(f"{tag}{pattern}: files differ")
                continue
            for name in rf:
                with rf[name].open("rb") as fa, cf[name].open("rb") as fb:
                    diffs = _compare_objects(pickle.load(fa), pickle.load(fb))
                if diffs:
                    details.append(f"{tag}{name}: {len(diffs)} differences, e.g. {diffs[0]}")
        for pattern in ["test/*/test_metrics.csv", "validation/*/validation_metrics.csv"]:
            rf, cf = _files(r, pattern), _files(c, pattern)
            for name in set(rf) | set(cf):
                if name not in rf or name not in cf or rf[name].read_text() != cf[name].read_text():
                    details.append(f"{tag}{name}: differs")
        if "log_messages" in ref_meta and "log_messages" in cand_meta:
            same_log = ref_meta["log_messages"] == cand_meta["log_messages"]
        else:
            same_log = _log_lines(r) == _log_lines(c)
        if not tag and not same_log:
            details.append("logged losses / learning rates differ")
    for key in ["lr_trace", "loader", "finetune_trainable"]:
        if ref_meta.get(key) != cand_meta.get(key):
            details.append(f"{key}: {ref_meta.get(key)} vs {cand_meta.get(key)}")
    rd, cd = ref_meta["dataset"], cand_meta["dataset"]
    for key in ["n_samples", "lookup_sha256", "sample_keys", "samples_sha256"]:
        if rd[key] != cd[key]:
            details.append(f"training samples: {key} differs" + (f" ({rd[key]} vs {cd[key]})" if key in
                                                                ("n_samples", "sample_keys") else ""))
    only_basin_idx = (rd["samples_sha256"] != cd["samples_sha256"]
                      and rd["samples_without_basin_idx_sha256"] == cd["samples_without_basin_idx_sha256"]
                      and set(rd["sample_keys"]) ^ set(cd["sample_keys"]) == {"basin_idx"})
    if only_basin_idx:
        details = [d for d in details if not d.startswith("training samples: ")]
        details.append("training samples: identical apart from the extra 'basin_idx' entry")
        status = "identical_except_basin_idx" if len(details) == 1 else "different"
    else:
        status = "identical" if not details else "different"
    return {"status": status, "details": details}


def _compare_tensors_nested(a, b, torch, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path}: keys differ"]
        out = []
        for k in a:
            out += _compare_tensors_nested(a[k], b[k], torch, f"{path}/{k}")
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length differs"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _compare_tensors_nested(x, y, torch, f"{path}[{i}]")
        return out
    if torch.is_tensor(a) and torch.is_tensor(b):
        return [] if torch.equal(a, b) else [f"{path}: tensor differs"]
    return [] if a == b else [f"{path}: {a} vs {b}"]


def frozen_check(meta_file: Path) -> dict:
    """After finetuning only 'head': every non-head tensor must equal the base model, the head must have changed."""
    import torch
    meta = json.loads(meta_file.read_text())
    if meta.get("error"):
        return {"status": "error"}
    base = sorted(Path(meta["run_dir"]).glob("model_epoch*.pt"))[-1]
    ft = sorted(Path(meta["finetune_run_dir"]).glob("model_epoch*.pt"))[-1]
    a = torch.load(base, map_location="cpu", weights_only=False)
    b = torch.load(ft, map_location="cpu", weights_only=False)
    changed = sorted(k for k in a if not torch.equal(a[k], b[k]))
    frozen_changed = [k for k in changed if not k.startswith("head.")]
    head_changed = [k for k in changed if k.startswith("head.")]
    return {"status": "frozen" if not frozen_changed and head_changed else "NOT frozen",
            "non_head_tensors_changed": frozen_changed, "head_tensors_changed": head_changed,
            "trainable": meta.get("finetune_trainable")}


# ======================================================================================================================
# Part 3: driver
# ======================================================================================================================


def _launch(tree: Path, specs: list, out: Path, data_root: Path, instrument: bool = False):
    out.mkdir(parents=True, exist_ok=True)
    spec_file = out / "spec.json"
    spec_file.write_text(json.dumps(specs, indent=1))
    cmd = [sys.executable, str(Path(__file__).resolve()), "run", "--tree", str(tree), "--spec", str(spec_file),
           "--out", str(out), "--data-root", str(data_root)]
    if instrument:
        cmd.append("--instrument-persistent")
    env = dict(os.environ, PYTHONPATH=str(tree), MPLBACKEND="Agg", PYTHONHASHSEED="0")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(data_root), env=env, capture_output=True, text=True)
    (out / "stdout.txt").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    print(f"    ran {[s['name'] for s in specs]} with {tree.name} in {time.time() - t0:.0f}s (exit {proc.returncode})",
          flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-3000:], flush=True)


def _error_of(meta_file: Path):
    if not meta_file.is_file():
        return "no meta.json (process crashed)"
    err = json.loads(meta_file.read_text()).get("error")
    return err.strip().splitlines()[-1] if err else None


def drive(args):
    trees = dict(t.split("=", 1) for t in args.tree)
    trees = {k: Path(v).resolve() for k, v in trees.items()}
    data_root = Path(args.data_root).resolve()
    out = Path(args.out).resolve()
    report = {"group": args.group, "scenarios": {}, "failures": []}

    tmean_dir = test_data_with_tmean(data_root, out) if args.group.startswith("repo_") else None
    for name, (spec, run_trees, reference, must_match) in scenarios(args.group, repo=trees["v2fixed"],
                                                                    tmean_dir=tmean_dir).items():
        print(f"== scenario {name}", flush=True)
        spec = dict(spec, name=name)
        # every reference run is repeated in a second fresh process: 'identical' only means something if the
        # reference itself is reproducible
        repeat = f"{reference}_repeat"
        trees.setdefault(repeat, trees[reference])
        if repeat not in run_trees:
            run_trees = [reference, repeat] + [t for t in run_trees if t != reference]
        must_match = list(must_match) + [repeat]
        for label in run_trees:
            _launch(trees[label], [spec], out / name / label, data_root)
        entry = {"reference": reference, "results": {}}
        ref_dir = out / name / reference
        ref_err = _error_of(ref_dir / "meta.json")
        for label in run_trees:
            if label == reference:
                entry["results"][label] = {"status": "reference" if not ref_err else "error", "error": ref_err}
                continue
            err = _error_of(out / name / label / "meta.json")
            if err or ref_err:
                result = {"status": "error", "error": err, "reference_error": ref_err}
                print(f"   {label:>16} ERROR: {err or ''} | reference: {ref_err or ''}", flush=True)
            else:
                result = compare_runs(ref_dir, out / name / label)
            entry["results"][label] = result
            print(f"   {label:>16} vs {reference}: {result['status']}", flush=True)
            for d in result.get("details", [])[:6]:
                print(f"{'':22}- {d}", flush=True)
            accepted = {"identical"}
            if reference.startswith("v1") and label == "v2fixed":
                accepted.add("identical_except_basin_idx")  # v2fixed drops v1's basin_idx for non-persistent models
            if label in must_match and result["status"] not in accepted:
                report["failures"].append(f"{name}: {label} is not identical to {reference}")
        if spec["kind"] == "finetune":
            entry["frozen"] = {label: frozen_check(out / name / label / "meta.json") for label in run_trees}
            for label, fz in entry["frozen"].items():
                print(f"   {label:>16} finetune freeze: {fz['status']} "
                      f"(non-head tensors changed: {len(fz.get('non_head_tensors_changed', []))})", flush=True)
            for label in ["upstream", "v2fixed"]:
                if entry["frozen"][label]["status"] != "frozen":
                    report["failures"].append(f"{name}: {label} does not freeze the non-finetuned modules")
        probes = {}
        for label in run_trees:
            mf = out / name / label / "meta.json"
            if mf.is_file():
                m = json.loads(mf.read_text())
                probes[label] = {"probe_counts": m.get("probe_counts"), "loader": m.get("loader"),
                                 "sample_keys": (m.get("dataset") or {}).get("sample_keys"),
                                 "validate_persistent_config_left_cfg_unchanged":
                                     m.get("validate_persistent_config_left_cfg_unchanged"),
                                 "lr_trace": m.get("lr_trace"), "seconds": m.get("seconds")}
                if label == "v2fixed":
                    executed = {k: v for k, v in (m.get("probe_counts") or {}).items() if v}
                    if executed:
                        report["failures"].append(f"{name}: v2fixed executed persistent-only code {executed}")
                    if "basin_idx" in ((m.get("dataset") or {}).get("sample_keys") or []):
                        report["failures"].append(f"{name}: v2fixed samples contain basin_idx")
                    if m.get("validate_persistent_config_left_cfg_unchanged") is False:
                        report["failures"].append(f"{name}: validate_persistent_config changed the config")
        entry["probes"] = probes
        report["scenarios"][name] = entry

    if args.group in ("persistent", "all"):
        report["persistent"] = persistent_and_switching(trees, data_root, out, report["failures"])
    if args.group == "repo_persistent":
        report["repo_persistent"] = repo_persistent(trees, data_root, out, report["failures"], tmean_dir)

    (out / f"report_{args.group}.json").write_text(json.dumps(report, indent=1, default=str))
    print("\n== SUMMARY", flush=True)
    print(json.dumps({k: {t: r["status"] for t, r in v["results"].items()} for k, v in report["scenarios"].items()},
                     indent=1), flush=True)
    if report["failures"]:
        print("FAILED EXPECTATIONS:\n  " + "\n  ".join(report["failures"]), flush=True)
        sys.exit(1)
    print("All expectations met.", flush=True)


def repo_persistent(trees, data_root, out, failures, tmean_dir):
    """The repository's Persistent-State LSTM configs, instrumented (Test 1 with the manuscript settings)."""
    res = {}
    for name in REPO_PERSISTENT_CONFIGS:
        print(f"== repository config {name} (instrumented)", flush=True)
        spec = dict(_spec(name, _repo_update(name, tmean_dir), eval_epoch=2,
                          config_file=str(trees["v2fixed"] / "test" / "test_configs" / f"{name}.test.yml")),
                    name=name)
        _launch(trees["v2fixed"], [spec], out / "repo_persistent" / name, data_root, instrument=True)
        mf = out / "repo_persistent" / name / "meta.json"
        err = _error_of(mf)
        meta = {} if err else json.loads(mf.read_text())
        checks = meta.get("persistent_checks")
        executed = {k: v for k, v in (meta.get("probe_counts") or {}).items() if v}
        res[name] = {"error": err, "checks": checks, "persistent_code_calls": executed, "loader": meta.get("loader"),
                     "time_step": meta.get("persistent_time_step")}
        status = "ERROR " + err if err else ("PASSED" if checks["passed"] else "FAILED")
        print(f"   {status}; time step {meta.get('persistent_time_step')}; loader {meta.get('loader')}; "
              f"persistent code calls {executed}", flush=True)
        if checks:
            print("   " + json.dumps(checks), flush=True)
        if err or not checks["passed"] or not executed.get("BaseTrainer._cache_persistent_state"):
            failures.append(f"repository config {name}: persistent checks failed")
    return res


def persistent_and_switching(trees, data_root, out, failures):
    res = {}
    cuda = _spec("daily_regression", dict(DAILY, model="cudalstm", epochs=2, validate_n_random_basins=2,
                                          save_validation_results=True), eval_epoch=2)
    mts = _spec("multi_timescale_regression", {"model": "mtslstm"}, eval_epoch=1)
    pers = _spec("daily_regression", PERSISTENT, eval_epoch=2)
    cuda["name"], mts["name"], pers["name"] = "cudalstm", "mtslstm", "persistentlstm"

    # Test 1: persistent state hand-over, instrumented (v2 as merged and v2 with the isolation fixes)
    print("== persistent state continuity (instrumented)", flush=True)
    for label in ["v2", "v2fixed"]:
        _launch(trees[label], [pers], out / "persistent" / label, data_root, instrument=True)
        mf = out / "persistent" / label / "meta.json"
        err = _error_of(mf)
        checks = None if err else json.loads(mf.read_text())["persistent_checks"]
        res[f"continuity_{label}"] = {"error": err, "checks": checks}
        print(f"   {label}: {'ERROR ' + err if err else ('PASSED' if checks['passed'] else 'FAILED')}", flush=True)
        if checks:
            print("   " + json.dumps(checks), flush=True)
        if label == "v2fixed" and (err or not checks["passed"]):
            failures.append("persistent continuity checks failed for v2fixed")
    cmp = compare_runs(out / "persistent" / "v2", out / "persistent" / "v2fixed")
    res["persistent_v2_vs_v2fixed"] = cmp
    print(f"   persistent run, v2fixed vs v2: {cmp['status']} {cmp.get('details', [])[:4]}", flush=True)
    if cmp["status"] != "identical":
        failures.append("the isolation fixes changed the Persistent-State LSTM run")

    # Test 4: model switching in one process versus fresh processes
    print("== model switching", flush=True)
    for spec in [cuda, mts, pers]:
        _launch(trees["v2fixed"], [spec], out / "switching" / f"fresh_{spec['name']}", data_root)
    for spec in [cuda, mts]:
        _launch(trees["upstream"], [spec], out / "switching" / f"upstream_{spec['name']}", data_root)
    _launch(trees["v2fixed"], [pers, cuda, mts], out / "switching" / "sequence_P_C_M", data_root)
    _launch(trees["v2fixed"], [cuda, pers], out / "switching" / "sequence_C_P", data_root)
    pairs = [("sequence_P_C_M/step1_cudalstm", "fresh_cudalstm"), ("sequence_P_C_M/step1_cudalstm", "upstream_cudalstm"),
             ("sequence_P_C_M/step2_mtslstm", "fresh_mtslstm"), ("sequence_P_C_M/step2_mtslstm", "upstream_mtslstm"),
             ("sequence_P_C_M/step0_persistentlstm", "fresh_persistentlstm"),
             ("sequence_C_P/step0_cudalstm", "fresh_cudalstm"),
             ("sequence_C_P/step1_persistentlstm", "fresh_persistentlstm"),
             ("fresh_cudalstm", "upstream_cudalstm"), ("fresh_mtslstm", "upstream_mtslstm")]
    for cand, ref in pairs:
        cd, rd = out / "switching" / cand, out / "switching" / ref
        errs = [e for e in (_error_of(cd / "meta.json"), _error_of(rd / "meta.json")) if e]
        r = {"status": "error", "details": errs} if errs else compare_runs(rd, cd)
        res[f"{cand} vs {ref}"] = r
        print(f"   {cand} vs {ref}: {r['status']} {r.get('details', [])[:4]}", flush=True)
        if r["status"] != "identical":
            failures.append(f"switching: {cand} differs from {ref}")
    for step in ["sequence_P_C_M/step1_cudalstm", "sequence_P_C_M/step2_mtslstm", "sequence_C_P/step0_cudalstm"]:
        mf = out / "switching" / step / "meta.json"
        if mf.is_file():
            m = json.loads(mf.read_text())
            info = {"probe_counts": m.get("probe_counts"), "loader": m.get("loader"),
                    "sample_keys": (m.get("dataset") or {}).get("sample_keys")}
            res[f"{step} probes"] = info
            print(f"   {step}: {info}", flush=True)
            if any((m.get("probe_counts") or {}).values()) or "basin_idx" in (info["sample_keys"] or []):
                failures.append(f"switching: {step} executed persistent-only code or got basin_idx")
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("drive")
    d.add_argument("--group", choices=["core", "models", "persistent", "repo_baselines", "repo_persistent", "all"],
                   required=True)
    d.add_argument("--data-root", required=True, help="upstream tree (its test configs and test data are used)")
    d.add_argument("--out", required=True)
    d.add_argument("--tree", action="append", required=True, help="label=path")
    r = sub.add_parser("run")
    r.add_argument("--tree", required=True)
    r.add_argument("--spec", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--data-root", required=True)
    r.add_argument("--instrument-persistent", action="store_true")
    args = parser.parse_args()
    if args.cmd == "drive":
        drive(args)
    else:
        run_in_tree(args)


if __name__ == "__main__":
    main()
