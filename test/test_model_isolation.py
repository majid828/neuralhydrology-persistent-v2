"""Regression tests: the persistent-state changes must not change how other models are trained or evaluated.

Each test pins down one place where version 1 of this fork had changed shared code for every model (found while
comparing this fork with upstream NeuralHydrology, commit 01510be). The bit-for-bit comparison with upstream runs in
.github/workflows/model-isolation.yml (tools/model_isolation/compare_training.py).

Run from the repository root:  pytest -q test/test_model_isolation.py
"""
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
from ruamel.yaml import YAML
from torch.utils.data import BatchSampler, RandomSampler

from neuralhydrology import nh_run
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.training.train import start_training
from neuralhydrology.utils.config import Config
from test import Fixture

DAILY = {
    "model": "cudalstm",
    "dataset": "camels_us",
    "forcings": "daymet",
    "dynamic_inputs": ["prcp(mm/day)", "tmax(C)"],
    "target_variables": ["QObs(mm/d)"],
}
PERSISTENT = {"model": "persistentlstm", "persistent_state": True, "non_overlapping_sequences": True,
              "seq_length": 30, "predict_last_n": 30, "seq_stride": 30, "batch_size": 4}


def _daily_config(get_config: Callable[[str], Config], **update) -> Config:
    config = get_config("daily_regression")
    config.update_config(dict(DAILY, data_dir=config.data_dir / "camels_us", **update))
    return config


def _train_dataset(config: Config, tmp: Path):
    config.train_dir = tmp / "train_data"
    config.train_dir.mkdir(parents=True, exist_ok=True)
    return get_dataset(config, is_train=True, period="train")


def _windows_per_basin(ds) -> dict:
    windows = defaultdict(list)
    for i in range(len(ds)):
        basin, indices = ds.lookup_table[i]
        windows[basin].append(int(indices[0]))
    return dict(windows)


def test_non_persistent_samples_have_no_basin_idx(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    """Only persistent-state runs add 'basin_idx'; every other model gets the upstream sample."""
    ds = _train_dataset(_daily_config(get_config), tmp_path / "cuda")
    assert "basin_idx" not in ds[0]

    ds = _train_dataset(_daily_config(get_config, **PERSISTENT), tmp_path / "persistent")
    assert "basin_idx" in ds[0]


def test_non_persistent_training_uses_upstream_loader(get_config: Fixture[Callable[[str], Config]]):
    """CudaLSTM is trained from a shuffled DataLoader of the given batch size, as in upstream."""
    trainer = BaseTrainer(cfg=_daily_config(get_config))
    trainer.initialize_training()
    assert isinstance(trainer.loader.sampler, RandomSampler)
    assert type(trainer.loader.batch_sampler) is BatchSampler
    assert trainer.loader.batch_size == trainer.cfg.batch_size
    assert not trainer.loader.drop_last


def test_non_overlapping_option_for_other_models_is_unchanged(get_config: Fixture[Callable[[str], Config]],
                                                              tmp_path: Path):
    """For models other than persistentlstm, 'non_overlapping_sequences' keeps every seq_length-th valid window
    (behaviour of the fork before version 2); the fixed window grid applies to persistent-state runs only."""
    all_windows = _windows_per_basin(_train_dataset(_daily_config(get_config), tmp_path / "all"))
    thinned = _windows_per_basin(
        _train_dataset(_daily_config(get_config, non_overlapping_sequences=True), tmp_path / "thinned"))
    seq_len = 30
    assert set(thinned) == set(all_windows)
    for basin, windows in all_windows.items():
        assert thinned[basin] == windows[::seq_len]


def test_dynamic_learning_rate_reads_its_own_key(get_config: Fixture[Callable[[str], Config]]):
    """Same as upstream after fix #303: the flag is 'dynamic_learning_rate', not 'early_stopping'."""
    config = _daily_config(get_config, early_stopping=True, dynamic_learning_rate=False)
    assert config.dynamic_learning_rate is False
    config = _daily_config(get_config, early_stopping=False, dynamic_learning_rate=True)
    assert config.dynamic_learning_rate is True


def test_finetuning_freezes_the_other_modules(get_config: Fixture[Callable[[str], Config]], tmp_path: Path):
    """Finetuning trains only 'finetune_modules', as in upstream (version 1 of the fork trained all weights)."""
    config = _daily_config(get_config)
    start_training(config)
    base_dir = config.run_dir

    finetune_file = tmp_path / "finetune.yml"
    with finetune_file.open("w") as fp:
        YAML().dump({"base_run_dir": str(base_dir), "finetune_modules": ["head"], "epochs": 1,
                     "learning_rate": {0: 0.001}, "run_dir": str(tmp_path / "finetune"),
                     "experiment_name": "finetune"}, fp)
    (tmp_path / "finetune").mkdir()
    nh_run.finetune(config_file=finetune_file)

    finetuned_dir = next((tmp_path / "finetune").glob("finetune_*"))
    base = torch.load(sorted(base_dir.glob("model_epoch*.pt"))[-1], map_location="cpu")
    tuned = torch.load(sorted(finetuned_dir.glob("model_epoch*.pt"))[-1], map_location="cpu")
    changed = {k for k in base if not torch.equal(base[k], tuned[k])}
    assert changed, "finetuning did not change any weight"
    assert all(k.startswith("head.") for k in changed), f"frozen modules changed: {sorted(changed)}"
