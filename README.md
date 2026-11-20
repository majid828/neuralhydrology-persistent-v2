
# Persistent LSTM Extension for NeuralHydrology (v2)

This code is derived from the open-source NeuralHydrology framework and extends its training workflow by adding persistent hidden- and cell-state handling across sequential hydrological batches. The LSTM carries hydrological memory across non-overlapping, chronological segments of each basin instead of resetting its states for every training sequence.

**This is version 2 of the repository.** Version 1 (`majid828/neuralhydrology-persistent`, release `v1.0.0`, Zenodo DOI below) is kept unchanged. Version 2 corrects the places where the v1 code did not do what the manuscript describes; see [Changes in version 2](#changes-in-version-2). Results produced with v1 and v2 are not interchangeable.

---

## Main modifications relative to NeuralHydrology

1. Persistent LSTM model: `neuralhydrology/modelzoo/persistentlstm.py`
2. Training workflow (`neuralhydrology/training/basetrainer.py`):
   - per-basin cache of the detached hidden and cell states, carried between chronological segments of the same basin
   - mini-batches of consecutive, time-contiguous segments of one basin, flattened `[B, L, d] -> [1, B*L, d]` into one continuous stream; static attributes reattached
   - basins randomly interleaved every epoch, chronological order within each basin
   - loss over every valid time step of the flattened stream; truncated backpropagation through time (states detached after each optimizer update)
   - states reset at the start of every epoch, at a basin's first segment, and where the forcing record has a gap
3. Evaluation workflow (`neuralhydrology/evaluation/tester.py`): one continuous stream per basin, optional causal warm-up (from the forcing before the period start), optional saving and reloading of final states
4. Dataset handling (`neuralhydrology/datasetzoo/basedataset.py`): non-overlapping segments on a fixed time grid, basin index per sample
5. Shared helpers and configuration checks: `neuralhydrology/utils/persistentutils.py`
6. 15-minute CAMELS-US dataset: `neuralhydrology/datasetzoo/fifteenminutecamelsus.py`

---

## Changes in version 2

| # | Manuscript statement | v1 behaviour | v2 behaviour | Files |
|---|---|---|---|---|
| 1 | Loss over all valid steps of the flattened segment, T = BL (Sec. 2.6, Eq. 17, 20) | Only the last `L` steps of each flattened batch were scored (12.5% of training targets for the daily runs, 6.25% for the hourly and 15-min runs) | Predictions are mapped back to `[B, L, ...]`; every valid target of every segment is scored | `training/basetrainer.py`, `utils/persistentutils.py` |
| 2 | Non-overlapping chronological segments `S_k = {(k-1)L+1, ..., kL}`; continuous stream; resets only at basin/epoch boundaries (Sec. 2.2-2.4) | Valid windows were thinned by position, so the grid shifted after gaps and the state was carried across missing data | Segments on a fixed grid; windows with missing targets stay in the stream (loss masks them); batches never span a gap; the state is reset only where the *forcing* has a gap (logged) | `datasetzoo/basedataset.py`, `training/basetrainer.py` |
| 3 | Only the final incomplete segment is discarded (Sec. 2.2) | `drop_last=True` discarded each basin's last `n mod B` full segments every epoch | Every full segment is used once per epoch | `training/basetrainer.py` |
| 4 | States detached and cached after the optimizer update (Eq. 13, Algorithm 1 step 16) | Cached before the update (same values) | Cached after the update | `training/basetrainer.py` |
| 5 | Evaluation over the test period (Sec. 2.9) | Warm-up consumed the first 30-90 days of the test period, which were then not scored | New option `eval_warmup_from_preceding_period: true`: the warm-up runs on the forcing before the period start, so the whole period is scored (same dates as the baselines) | `datasetzoo/basedataset.py`, configs |
| 6 | States are saved and reloaded for operational initialization (Sec. 1) | Not implemented | `save_persistent_states: true` writes each basin's final `(h, c)` and time stamp; `persistent_state_init_file` initializes a following period from them | `evaluation/tester.py`, `utils/config.py` |
| 7 | Released configs reproduce the manuscript runs (Sec. 5) | Hourly YAML did not parse; file names with commas; no seed in the daily configs | Fixed indentation, renamed files, seeds added | `test/test_configs/` |
| 8 | 15-min forcing and discharge describe the same time steps (Sec. 3.1) | USGS local clock time used as if UTC | Converted to UTC with the `tz_cd` column (`qobs_15min_to_utc`, default true) | `datasetzoo/fifteenminutecamelsus.py` |

Further v2 changes:
- Configuration guards (`validate_persistent_config`) stop runs whose settings would silently break the method (overlapping windows, `predict_last_n != seq_length`, `seq_stride` not in {1, `seq_length`}, multi-frequency, a model other than `persistentlstm`).
- At evaluation, a non-finite state (e.g. after a NaN forcing value) is reset instead of turning the rest of the basin's predictions into NaN.
- The evaluation loss covers every scored time step.
- Ablation switch `ablate_state_carry: true`: every flattened batch starts from a zero state, with everything else unchanged, to isolate the effect of carrying the state.
- The non-persistent (baseline) training and evaluation paths are identical to upstream NeuralHydrology again (target noise and `UncertaintyTester` sampling restored).

Not changed in the code (manuscript wording):
- NeuralHydrology's masked MSE contains a factor 1/2 that Eq. 17 and 20 do not show. Changing it would change the effective learning rate of every model, so the equations should mention it instead.
- Each optimizer update backpropagates through `B*L` time steps (16 days for the daily runs, 16 days hourly, 4 days at 15 min), not only through the `seq_length` window. `batch_size` therefore sets the gradient horizon.

### New configuration keys

| Key | Default | Meaning |
|---|---|---|
| `eval_warmup_from_preceding_period` | `false` | Take the `eval_warmup_steps` warm-up from the forcing before the period start (targets there are always NaN). Requires `eval_warmup_steps` to be a multiple of `seq_length`. |
| `save_persistent_states` | `false` | Write `<run_dir>/<period>/<model_epochXXX>/<period>_persistent_states.p` when results are saved. |
| `persistent_state_init_file` | none | Initialize each basin from a saved state that ends one time step before the evaluation stream starts (stand-alone evaluation only; not combinable with `eval_warmup_from_preceding_period`). |
| `ablate_state_carry` | `false` | Start every flattened batch from a zero state (training and evaluation). |
| `qobs_15min_to_utc` | `true` | Convert USGS 15-min discharge time stamps to UTC using `tz_cd`. |

### Tests

- `pytest -q test/test_persistent_lstm.py` (needs PyTorch, no data): state continuity with the real LSTM, cell-state carry-over, gradient truncation, loss coverage, configuration guards, segment grid and batching.
- `python3 -I tools/persistent_checks/run_checks.py .` (needs only NumPy and pandas): runs the repository's own sampler, training loop, loss and evaluation code with a counting stand-in for the LSTM and checks every state hand-off, reset, scored time step, gap, warm-up and saved state.
- `python tools/diagnose_persistent_gaps.py <config>` (needs your data): per-basin segments, contiguous runs, batches and targets for a real configuration.

---

## Relationship to NeuralHydrology

This repository is a derivative of the original NeuralHydrology package. The original NeuralHydrology code was used as the base hydrological deep-learning framework, and this repository extends it by adding a Persistent LSTM model and modifying the training and dataset workflow to support persistent hidden-state handling across sequential hydrological batches.

The original NeuralHydrology software should be cited as:

```bibtex
@article{kratzert2022joss,
  title = {NeuralHydrology -- A Python library for Deep Learning research in hydrology},
  author = {Frederik Kratzert and Martin Gauch and Grey Nearing and Daniel Klotz},
  journal = {Journal of Open Source Software},
  publisher = {The Open Journal},
  year = {2022},
  volume = {7},
  number = {71},
  pages = {4050},
  doi = {10.21105/joss.04050},
  url = {https://doi.org/10.21105/joss.04050}
}
```

---

## Citation for this repository

Version 1.0.0 (the original repository, `majid828/neuralhydrology-persistent`) is archived on Zenodo with DOI:

https://doi.org/10.5281/zenodo.20479712

Please cite this repository as:

```bibtex
@software{shah_persistent_lstm_2026,
  author = {Shah, Majid Hussain},
  title = {Persistent LSTM Extension for NeuralHydrology},
  year = {2026},
  publisher = {Zenodo},
  version = {v1.0.0},
  doi = {10.5281/zenodo.20479712},
  url = {https://doi.org/10.5281/zenodo.20479712}
}
```

---

## Manuscript configuration files

The configuration files used to generate the manuscript results are provided in:

`test/test_configs/`

The following configuration files were used for the experiments reported in the manuscript:

| Configuration file | Purpose |
|---|---|
| `persistent_lstm_hourly.test.yml` | Persistent LSTM experiment using hourly hydrological data |
| `persistent_lstm_15_mint.test.yml` | Persistent LSTM experiment using 15-minute hydrological data |
| `persistent_lstm_for_long_memory_basins.test.yml` | Persistent LSTM experiment for long-memory basin analysis |
| `persistent_lstm_nonstationary_basins.test.yml` | Persistent LSTM experiment for nonstationary basin analysis |
| `cuda_lstm_for_long_memory_basins.test.yml` | Standard CUDA LSTM comparison for long-memory basins |
| `cuda_lstm_nonstationary_basins.test.yml` | Standard CUDA LSTM comparison for nonstationary basins |
| `mts_lstm_hourly.test.yml` | Multi-timescale LSTM experiment using hourly hydrological data |

These files define the model settings used in the manuscript, including the model type, basin selection, input features, training period, validation period, test period, sequence length, batch size, and other training options.

Before running them on another machine, adjust `data_dir`, `run_dir`, `device` and the basin files. All configs currently point to the same basin list (`list_250_OWP6.txt`); store one basin list per experiment (high-resolution Washington basins, nonstationary basins, snow-influenced Pacific Northwest basins) and point each config to its own list.

---
