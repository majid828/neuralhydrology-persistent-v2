import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

import neuralhydrology.training.loss as loss
from neuralhydrology.datautils.utils import load_basin_file, load_scaler
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.evaluation import get_tester
from neuralhydrology.evaluation.tester import BaseTester
from neuralhydrology.modelzoo import get_model
from neuralhydrology.training import get_loss_obj, get_optimizer, get_regularization_obj
from neuralhydrology.training.earlystopper import EarlyStopper
from neuralhydrology.training.logger import Logger
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.logging_utils import setup_logging
from neuralhydrology.utils.persistentutils import (contiguous_batches, has_finite_targets, infer_time_step,
                                                   unflatten_predictions, validate_persistent_config)

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Basin-chronological, interleaved batch sampler for persistent-state training
# =============================================================================
class BasinChronoInterleaveBatchSampler(Sampler[List[int]]):
    """Yields mini-batches for persistent-state training (manuscript Sec. 2.5, Algorithm 1 step 3).

    - Every batch holds consecutive, time-contiguous segments of ONE basin, so that flattening the batch gives one
      continuous stream (Sec. 2.3). The batches are built by `persistentutils.contiguous_batches`.
    - Within a basin, batches are always yielded in chronological order (A1 before A2 before A3, Eq. 15).
    - Across basins, the order is re-randomized every epoch by randomly interleaving basins (A1, C1, B1, A2, ..., Eq. 16).
    - Every batch is used exactly once per epoch. Only the last batch of a contiguous run of segments can hold fewer
      than batch_size segments; no full segment is dropped (Sec. 2.2).
    """

    def __init__(self, basin_to_batches: Dict[int, List[List[int]]], seed: int = 0):
        self.basin_to_batches = {b: [list(batch) for batch in batches]
                                 for b, batches in basin_to_batches.items() if len(batches) > 0}
        self.seed = int(seed)
        self._epoch = 0
        self._length = sum(len(batches) for batches in self.basin_to_batches.values())

    def set_epoch(self, epoch: int):
        """Call once per epoch so that the interleaving changes (deterministically, given the seed)."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._epoch)
        # pointer to the next chronological batch of each basin
        ptr: Dict[int, int] = {b: 0 for b in self.basin_to_batches}
        active = sorted(self.basin_to_batches)
        while active:
            basin_id = rng.choice(active)
            yield self.basin_to_batches[basin_id][ptr[basin_id]]
            ptr[basin_id] += 1
            if ptr[basin_id] >= len(self.basin_to_batches[basin_id]):
                active.remove(basin_id)


class BaseTrainer(object):
    """Default class to train a model."""

    def __init__(self, cfg: Config):
        super(BaseTrainer, self).__init__()
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.loss_obj = None
        self.experiment_logger = None
        self.loader = None
        self.validator = None
        self.noise_sampler_y = None
        self._target_mean = None
        self._target_std = None
        self._scaler = {}
        self._allow_subsequent_nan_losses = cfg.allow_subsequent_nan_losses
        self._disable_pbar = cfg.verbose == 0
        self._max_updates_per_epoch = cfg.max_updates_per_epoch
        self._early_stopping = cfg.early_stopping
        self._patience_early_stopping = cfg.patience_early_stopping
        self._minimum_epochs_before_early_stopping = cfg.minimum_epochs_before_early_stopping
        self._dynamic_learning_rate = cfg.dynamic_learning_rate
        self._patience_dynamic_learning_rate = cfg.patience_dynamic_learning_rate
        self._factor_dynamic_learning_rate = cfg.factor_dynamic_learning_rate

        # Persistent-state training switch. Invalid persistent settings raise here, before any run folder is created.
        validate_persistent_config(self.cfg)
        self.persistent_state = getattr(self.cfg, "persistent_state", False)
        self._persistent_time_step: Optional[np.timedelta64] = None

        # Basin list
        self.basins = load_basin_file(cfg.train_basin_file)
        self.cfg.number_of_basins = len(self.basins)

        # epoch start
        self._epoch = self._get_start_epoch_number()

        self._create_folder_structure()
        setup_logging(str(self.cfg.run_dir / "output.log"))
        LOGGER.info(f"### Folder structure created at {self.cfg.run_dir}")

        if self.cfg.is_continue_training:
            LOGGER.info(f"### Continue training of run stored in {self.cfg.base_run_dir}")
        if self.cfg.is_finetuning:
            LOGGER.info(f"### Start finetuning with pretrained model stored in {self.cfg.base_run_dir}")

        LOGGER.info(f"### Run configurations for {self.cfg.experiment_name}")
        for key, val in self.cfg.as_dict().items():
            LOGGER.info(f"{key}: {val}")

        self._set_random_seeds()
        self._set_device()

        # Will be created in initialize_training
        self._basin_batch_sampler: Optional[BasinChronoInterleaveBatchSampler] = None

    # ------------------------------------------------------------------
    # Helpers for persistent LSTM reshaping and slicing
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_time_batch(t: torch.Tensor) -> torch.Tensor:
        """
        Flatten [B, L, ...] -> [1, B*L, ...].

        Why:
          Your persistentlstm forward expects a single "continuous" time axis for state carryover.
          By flattening across the DataLoader batch dimension, we treat the batch as one long segment.
        """
        if not torch.is_tensor(t) or t.dim() < 2:
            return t
        t = t.contiguous()
        b, l = t.shape[0], t.shape[1]
        return t.view(1, b * l, *t.shape[2:])

    def _slice_time_segment(self, data_dict: Dict[str, object], segment_slice: slice) -> Dict[str, object]:
        """Slice a contiguous segment along time axis from already reshaped tensors."""
        seg: Dict[str, object] = {}
        for key, val in data_dict.items():
            if key.startswith("x_d"):
                seg[key] = {feat: v[:, segment_slice, ...] for feat, v in val.items()}
            elif key.startswith("x_s"):
                seg[key] = val  # fixed to batch=1 later
            elif key.startswith("y") or key.startswith("basin_idx"):
                if torch.is_tensor(val):
                    if val.dim() == 1:
                        seg[key] = val[segment_slice]
                    else:
                        seg[key] = val[:, segment_slice, ...]
                else:
                    seg[key] = val
            elif torch.is_tensor(val) and val.dim() >= 2:
                seg[key] = val[:, segment_slice, ...]
            else:
                seg[key] = val
        return seg

    def _get_dataset(self, basin: Optional[str] = None) -> BaseDataset:
        return get_dataset(cfg=self.cfg, period="train", is_train=True, scaler=self._scaler, basin=basin)

    def _get_model(self) -> torch.nn.Module:
        return get_model(cfg=self.cfg)

    def _get_optimizer(self) -> torch.optim.Optimizer:
        return get_optimizer(model=self.model, cfg=self.cfg)

    def _get_loss_obj(self) -> loss.BaseLoss:
        return get_loss_obj(cfg=self.cfg)

    def _set_regularization(self):
        self.loss_obj.set_regularization_terms(get_regularization_obj(cfg=self.cfg))

    def _get_tester(self) -> BaseTester:
        return get_tester(cfg=self.cfg, run_dir=self.cfg.run_dir, period="validation", init_model=False)

    # ------------------------------------------------------------------
    # Build basin -> chronological, time-contiguous batches from the dataset
    # ------------------------------------------------------------------
    def _build_basin_chrono_batches(self, ds: BaseDataset) -> Dict[int, List[List[int]]]:
        """Map basin_id -> list of batches (lists of dataset indices) in chronological order.

        Each batch holds up to batch_size consecutive segments of one basin that are contiguous in time, so the
        flattened batch is one continuous stream. A data gap (segments removed because of missing forcing) ends a
        batch; the trainer then starts the next batch of that basin from a fresh state instead of carrying the state
        across the gap.
        """
        LOGGER.info("### Building chronological, time-contiguous batches per basin (one pass over dataset) ...")
        basin_to_items: Dict[int, List[Tuple[int, Any, Any]]] = {}
        step = None
        for idx in range(len(ds)):
            sample = ds[idx]

            if "basin_idx" not in sample:
                raise RuntimeError("Dataset sample does not contain 'basin_idx'. Required for persistent batching.")
            bidx = sample["basin_idx"]
            if torch.is_tensor(bidx):
                basin_id = int(bidx.view(-1)[0].item())
            else:
                basin_id = int(np.array(bidx).reshape(-1)[0])

            if "date" not in sample:
                raise RuntimeError("Dataset sample does not contain 'date'. Required to order segments in time.")
            dates = np.asarray(sample["date"]).reshape(-1)
            if step is None:
                step = infer_time_step(dates[None, :], ds.frequencies[0] if ds.frequencies else None)

            basin_to_items.setdefault(basin_id, []).append((idx, dates[0], dates[-1]))

        self._persistent_time_step = step
        basin_to_batches = {
            basin_id: contiguous_batches(items, batch_size=self.cfg.batch_size, step=step)
            for basin_id, items in basin_to_items.items()
        }
        n_segments = sum(len(items) for items in basin_to_items.values())
        n_batches = sum(len(batches) for batches in basin_to_batches.values())
        LOGGER.info(f"### {n_segments} segments of {len(basin_to_items)} basins grouped into {n_batches} "
                    f"time-contiguous batches.")
        return basin_to_batches

    def initialize_training(self):
        """Initialize the training class."""
        if self.cfg.is_finetuning:
            self._scaler = load_scaler(self.cfg.base_run_dir)

        ds = self._get_dataset()
        if len(ds) == 0:
            raise ValueError("Dataset contains no samples.")

        # ------------------------------------------------------------------
        # CORRECT persistent batching
        # ------------------------------------------------------------------
        if self.persistent_state and self.cfg.model.lower() == "persistentlstm":
            # Persistent across batches of the same basin within an epoch, random interleaving of basins.
            basin_to_batches = self._build_basin_chrono_batches(ds)

            self._basin_batch_sampler = BasinChronoInterleaveBatchSampler(
                basin_to_batches=basin_to_batches,
                seed=self.cfg.seed if self.cfg.seed is not None else 0,
            )

            # DataLoader uses our sampler; don't enable shuffle here.
            self.loader = DataLoader(
                ds,
                batch_sampler=self._basin_batch_sampler,
                num_workers=self.cfg.num_workers,
                collate_fn=ds.collate_fn,
            )
            LOGGER.info("### Using BasinChronoInterleaveBatchSampler (interleaved basins, chrono within basin).")
        else:
            # Original non-persistent training
            self.loader = DataLoader(
                ds,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                collate_fn=ds.collate_fn,
            )

        self.model = self._get_model().to(self.device)

        if self.cfg.checkpoint_path is not None:
            LOGGER.info(f"Starting training from Checkpoint {self.cfg.checkpoint_path}")
            self.model.load_state_dict(torch.load(str(self.cfg.checkpoint_path), map_location=self.device))
        elif self.cfg.checkpoint_path is None and self.cfg.is_finetuning:
            checkpoint_path = [x for x in sorted(list(self.cfg.base_run_dir.glob("model_epoch*.pt")))][-1]
            LOGGER.info(f"Starting training from checkpoint {checkpoint_path}")
            self.model.load_state_dict(torch.load(str(checkpoint_path), map_location=self.device))

        # Freeze model parts from pre-trained model (as in upstream NeuralHydrology; the v1 fork had dropped this).
        if self.cfg.is_finetuning:
            self._freeze_model_parts()

        self.optimizer = self._get_optimizer()
        self.loss_obj = self._get_loss_obj().to(self.device)
        self._set_regularization()

        if self.cfg.is_continue_training:
            self._restore_training_state()

        self.experiment_logger = Logger(cfg=self.cfg)
        if self.cfg.log_tensorboard:
            self.experiment_logger.start_tb()

        if self.cfg.is_continue_training:
            self.experiment_logger.epoch = self._epoch
            self.experiment_logger.update = len(self.loader) * self._epoch

        if self.cfg.validate_every is not None:
            if self.cfg.validate_n_random_basins < 1:
                warn_msg = [
                    f"Validation set to validate every {self.cfg.validate_every} epoch(s), but ",
                    "'validate_n_random_basins' not set or set to zero. Will validate on the entire validation set.",
                ]
                LOGGER.warning("".join(warn_msg))
                self.cfg.validate_n_random_basins = self.cfg.number_of_basins
            self.validator = self._get_tester()

        if self.cfg.target_noise_std is not None:
            self.noise_sampler_y = torch.distributions.Normal(loc=0, scale=self.cfg.target_noise_std)
            self._target_mean = torch.from_numpy(
                ds.scaler["xarray_feature_center"][self.cfg.target_variables].to_array().values
            ).to(self.device)
            self._target_std = torch.from_numpy(
                ds.scaler["xarray_feature_scale"][self.cfg.target_variables].to_array().values
            ).to(self.device)

    def _freeze_model_parts(self):
        # freeze all model weights
        for param in self.model.parameters():
            param.requires_grad = False

        unresolved_modules = []

        # unfreeze parameters specified in config as tuneable parameters
        if isinstance(self.cfg.finetune_modules, list):
            for module_part in self.cfg.finetune_modules:
                if module_part in self.model.module_parts:
                    module = getattr(self.model, module_part)
                    for param in module.parameters():
                        param.requires_grad = True
                else:
                    unresolved_modules.append(module_part)
        else:
            # if it was no list, it has to be a dictionary
            for module_group, module_parts in self.cfg.finetune_modules.items():
                if module_group in self.model.module_parts:
                    if isinstance(module_parts, str):
                        module_parts = [module_parts]
                    for module_part in module_parts:
                        module = getattr(self.model, module_group)[module_part]
                        for param in module.parameters():
                            param.requires_grad = True
                else:
                    unresolved_modules.append(module_group)
        if unresolved_modules:
            LOGGER.warning(f"Could not resolve the following module parts for finetuning: {unresolved_modules}")

    def train_and_validate(self):
        """Train and validate the model."""
        if self._early_stopping:
            if self.cfg.is_continue_training:
                LOGGER.warning("Early stopping state is reset.")
            early_stopper = EarlyStopper(patience=self._patience_early_stopping, min_delta=0.0001)

        if self._dynamic_learning_rate:
            if self.cfg.is_continue_training:
                LOGGER.warning("Scheduler state is reset.")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self._factor_dynamic_learning_rate,
                patience=self._patience_dynamic_learning_rate,
            )

        for epoch in range(self._epoch + 1, self._epoch + self.cfg.epochs + 1):
            # IMPORTANT: changes interleaving pattern each epoch (still chrono within basin)
            if self._basin_batch_sampler is not None:
                self._basin_batch_sampler.set_epoch(epoch)

            if not self._dynamic_learning_rate:
                if epoch in self.cfg.learning_rate.keys():
                    LOGGER.info(f"Setting learning rate to {self.cfg.learning_rate[epoch]}")
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.cfg.learning_rate[epoch]

            self._train_epoch(epoch=epoch)

            avg_losses = self.experiment_logger.summarise()
            loss_str = ", ".join(f"{k}: {v:.5f}" for k, v in avg_losses.items())
            LOGGER.info(f"Epoch {epoch} average loss: {loss_str}")

            if epoch % self.cfg.save_weights_every == 0:
                self._save_weights_and_optimizer(epoch)

            if (self.validator is not None) and (epoch % self.cfg.validate_every == 0):
                self.validator.evaluate(
                    epoch=epoch,
                    save_results=self.cfg.save_validation_results,
                    save_all_output=self.cfg.save_all_output,
                    metrics=self.cfg.metrics,
                    model=self.model,
                    experiment_logger=self.experiment_logger.valid(),
                )

                valid_metrics = self.experiment_logger.summarise()
                print_msg = f"Epoch {epoch} average validation loss: {valid_metrics['avg_total_loss']:.5f}"
                if self.cfg.metrics:
                    print_msg += " -- Median validation metrics: "
                    print_msg += ", ".join(f"{k}: {v:.5f}" for k, v in valid_metrics.items() if k != "avg_total_loss")
                LOGGER.info(print_msg)

                if (
                    self._early_stopping
                    and epoch > self._minimum_epochs_before_early_stopping
                    and early_stopper.check_early_stopping(valid_metrics["avg_total_loss"])
                ):
                    LOGGER.info(
                        f"Early stopping triggered at epoch {epoch} with validation loss "
                        f"{valid_metrics['avg_total_loss']:.5f}. Training stopped."
                    )
                    break

                if self._dynamic_learning_rate:
                    scheduler.step(valid_metrics["avg_total_loss"])

        if self.cfg.log_tensorboard:
            self.experiment_logger.stop_tb()

    def _get_start_epoch_number(self):
        if self.cfg.is_continue_training:
            if self.cfg.continue_from_epoch is not None:
                epoch = self.cfg.continue_from_epoch
            else:
                weight_path = [x for x in sorted(list(self.cfg.run_dir.glob("model_epoch*.pt")))][-1]
                epoch = weight_path.name[-6:-3]
        else:
            epoch = 0
        return int(epoch)

    def _restore_training_state(self):
        if self.cfg.continue_from_epoch is not None:
            epoch = f"{self.cfg.continue_from_epoch:03d}"
            weight_path = self.cfg.base_run_dir / f"model_epoch{epoch}.pt"
        else:
            weight_path = [x for x in sorted(list(self.cfg.base_run_dir.glob("model_epoch*.pt")))][-1]
            epoch = weight_path.name[-6:-3]

        optimizer_path = self.cfg.base_run_dir / f"optimizer_state_epoch{epoch}.pt"

        LOGGER.info(f"Continue training from epoch {int(epoch)}")
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(str(optimizer_path), map_location=self.device))

    def _save_weights_and_optimizer(self, epoch: int):
        weight_path = self.cfg.run_dir / f"model_epoch{epoch:03d}.pt"
        torch.save(self.model.state_dict(), str(weight_path))

        optimizer_path = self.cfg.run_dir / f"optimizer_state_epoch{epoch:03d}.pt"
        torch.save(self.optimizer.state_dict(), str(optimizer_path))

    @staticmethod
    def _cache_persistent_state(cache: Dict[int, Optional[Tuple[torch.Tensor, torch.Tensor]]],
                                cache_end: Dict[int, Any], basin_id: int,
                                new_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]], dates: np.ndarray):
        """C[b] <- detach(h_T, c_T) (manuscript Eq. 13), plus the time stamp of the last processed step."""
        if new_hidden is None:
            cache[basin_id] = None
        else:
            h, c = new_hidden
            cache[basin_id] = (h.detach(), c.detach())
        cache_end[basin_id] = np.asarray(dates)[-1, -1]

    def _add_target_noise(self, data: Dict[str, torch.Tensor]):
        if self.noise_sampler_y is not None:
            for key in filter(lambda k: 'y' in k, data.keys()):
                noise = self.noise_sampler_y.sample(data[key].shape)
                # make sure we add near-zero noise to originally near-zero targets
                data[key] += (data[key] + self._target_mean / self._target_std) * noise.to(self.device)

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.experiment_logger.train()

        # ============================================================
        # Persistent-state path (manuscript Algorithm 1)
        # ============================================================
        if self.persistent_state and self.cfg.model.lower() == "persistentlstm":
            # Algorithm 1 step 2 / Eq. (14): empty state cache at the start of every epoch (no carry across epochs)
            epoch_state_cache: Dict[int, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {}
            epoch_state_end: Dict[int, Any] = {}
            ablate_state_carry = bool(getattr(self.cfg, "ablate_state_carry", False))
            step = self._persistent_time_step
            n_updates, n_no_target, n_gap_resets = 0, 0, 0

            n_iter = min(self._max_updates_per_epoch, len(self.loader)) if self._max_updates_per_epoch is not None else None
            pbar = tqdm(self.loader, file=sys.stdout, disable=self._disable_pbar, total=n_iter)
            pbar.set_description(f"# Epoch {epoch} (interleaved basins, chrono within basin)")

            nan_count = 0

            for i, data in enumerate(pbar):
                if self._max_updates_per_epoch is not None and i >= self._max_updates_per_epoch:
                    break

                # Move tensors to device
                for key in data.keys():
                    if key.startswith("x_d"):
                        data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                    elif not key.startswith("date"):
                        data[key] = data[key].to(self.device)

                data = self.model.pre_model_hook(data, is_train=True)

                # Determine basin for this batch
                if "basin_idx" not in data or (not torch.is_tensor(data["basin_idx"])):
                    raise RuntimeError("Persistent training requires basin_idx in batch.")

                bidx = data["basin_idx"]

                # HARD SAFETY CHECK: ensure the batch contains only ONE basin
                flat = bidx.view(-1)
                if not torch.all(flat == flat[0]):
                    raise RuntimeError("Batch contains multiple basins -> hidden-state leakage risk. Fix sampler/dataset.")

                # Extract basin id
                basin_id = int(flat[0].item())

                # Infer B,L from any dynamic input
                any_xd = None
                if "x_d" in data and isinstance(data["x_d"], dict) and len(data["x_d"]) > 0:
                    any_xd = next(iter(data["x_d"].values()))
                if any_xd is None or any_xd.dim() < 2:
                    raise RuntimeError("Persistent training: could not infer [B,L] from x_d.")
                B_inferred, L_inferred = int(any_xd.shape[0]), int(any_xd.shape[1])

                # Time stamps of the B consecutive segments, shape [B, L]
                dates = np.asarray(data["date"])
                if dates.shape[:2] != (B_inferred, L_inferred):
                    raise RuntimeError(f"Persistent training: date array {dates.shape} does not match x_d "
                                       f"[{B_inferred}, {L_inferred}].")
                # The sampler only groups time-contiguous segments, so the flattened stream is continuous (Sec. 2.3).
                if B_inferred > 1 and not np.all(dates[1:, 0] == dates[:-1, -1] + step):
                    raise RuntimeError("Persistent training: a batch holds segments that are not contiguous in time.")

                # Eq. (11): cached state of this basin, or init() (= zero state) if there is none. The cached state is
                # only used if it ended exactly one time step before this batch starts; after a data gap (segments
                # removed because of missing forcing) the basin restarts from init() instead of jumping the gap.
                hidden = epoch_state_cache.get(basin_id, None)
                if hidden is not None and dates[0, 0] != epoch_state_end[basin_id] + step:
                    hidden = None
                    n_gap_resets += 1
                if ablate_state_carry:
                    hidden = None

                # Flatten [B, L, ...] -> [1, B*L, ...] so the model sees one continuous segment (Eq. 7-8)
                reshaped: Dict[str, object] = {}
                for key, val in data.items():
                    if key.startswith("x_d"):
                        reshaped[key] = {feat: self._flatten_time_batch(v) for feat, v in val.items()}
                    elif key.startswith("x_s"):
                        reshaped[key] = val
                    elif key.startswith("y") and torch.is_tensor(val):
                        reshaped[key] = self._flatten_time_batch(val)
                    elif key == "basin_idx" and torch.is_tensor(val):
                        # Ensure basin_idx becomes [B,L] before flatten so it aligns with time
                        if val.dim() == 1:
                            basin_2d = val.view(B_inferred, 1).expand(B_inferred, L_inferred)
                        else:
                            basin_2d = val
                        reshaped[key] = self._flatten_time_batch(basin_2d)
                    else:
                        if torch.is_tensor(val) and val.dim() >= 2:
                            reshaped[key] = self._flatten_time_batch(val)
                        else:
                            reshaped[key] = val

                seg_len = B_inferred * L_inferred
                seg_data = self._slice_time_segment(reshaped, slice(0, seg_len))

                # Eq. (9): reattach the basin's static attributes (batch=1; static features are per basin, not per time)
                if "x_s" in data:
                    if isinstance(data["x_s"], dict):
                        seg_data["x_s"] = {feat: v[0:1, ...] for feat, v in data["x_s"].items()}
                    else:
                        seg_data["x_s"] = data["x_s"][0:1, ...]

                # Eq. (12): forward pass over the flattened stream, starting from (h0, c0)
                preds = self.model(seg_data, hidden_state=hidden)
                new_hidden = preds.get("hidden_state", None)

                # A batch without any observed target (e.g. a gap in the discharge record) carries no supervision:
                # propagate the state through it, but do not update the parameters.
                if not has_finite_targets(data):
                    self._cache_persistent_state(epoch_state_cache, epoch_state_end, basin_id, new_hidden, dates)
                    n_no_target += 1
                    continue

                self._add_target_noise(data)

                # Eq. (17)/(20): loss over EVERY valid time step of the flattened segment T = B*L. The predictions are
                # mapped back to [B, L, ...] and scored against the original [B, L, ...] targets; with
                # predict_last_n == seq_length (enforced by validate_persistent_config) BaseLoss keeps all L steps of
                # each of the B segments. (Scoring the flattened [1, B*L] tensors would keep only the last L steps.)
                preds_for_loss = unflatten_predictions(preds, B_inferred, L_inferred)
                loss_val, all_losses = self.loss_obj(preds_for_loss, data)

                if torch.isnan(loss_val):
                    nan_count += 1
                    if nan_count > self._allow_subsequent_nan_losses:
                        raise RuntimeError(f"Loss was NaN for {nan_count} times in a row. Stopped training.")
                    LOGGER.warning(f"Loss is NaN; ignoring step. (#{nan_count}/{self._allow_subsequent_nan_losses})")
                    self._cache_persistent_state(epoch_state_cache, epoch_state_end, basin_id, new_hidden, dates)
                    continue
                nan_count = 0

                # Algorithm 1 step 15: update the parameters by backpropagation through the current segment only
                self.optimizer.zero_grad()
                loss_val.backward()

                if self.cfg.clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradient_norm)

                self.optimizer.step()
                n_updates += 1

                # Eq. (13), Algorithm 1 step 16: after the optimizer update, store the detached final state.
                # Detaching stops gradients at the batch boundary (truncated BPTT); the values are carried forward.
                self._cache_persistent_state(epoch_state_cache, epoch_state_end, basin_id, new_hidden, dates)

                pbar.set_postfix_str(f"Loss: {loss_val.item():.4f}")
                self.experiment_logger.log_step(**{k: v.item() for k, v in all_losses.items()})

            LOGGER.info(f"Epoch {epoch}: {n_updates} optimizer updates; {n_no_target} batches without observed "
                        f"targets (state propagated, no update); {n_gap_resets} state resets at data gaps.")
            return

        # ============================================================
        # Original non-persistent path (as in upstream NeuralHydrology)
        # ============================================================
        n_iter = min(self._max_updates_per_epoch, len(self.loader)) if self._max_updates_per_epoch is not None else None
        pbar = tqdm(self.loader, file=sys.stdout, disable=self._disable_pbar, total=n_iter)
        pbar.set_description(f"# Epoch {epoch}")

        nan_count = 0
        for i, data in enumerate(pbar):
            if self._max_updates_per_epoch is not None and i >= self._max_updates_per_epoch:
                break

            for key in data.keys():
                if key.startswith("x_d"):
                    data[key] = {k: v.to(self.device) for k, v in data[key].items()}
                elif not key.startswith("date"):
                    data[key] = data[key].to(self.device)

            # apply possible pre-processing to the batch before the forward pass
            data = self.model.pre_model_hook(data, is_train=True)

            # get predictions
            predictions = self.model(data)

            self._add_target_noise(data)

            loss_val, all_losses = self.loss_obj(predictions, data)

            # early stop training if loss is NaN
            if torch.isnan(loss_val):
                nan_count += 1
                if nan_count > self._allow_subsequent_nan_losses:
                    raise RuntimeError(f"Loss was NaN for {nan_count} times in a row. Stopped training.")
                LOGGER.warning(f"Loss is Nan; ignoring step. (#{nan_count}/{self._allow_subsequent_nan_losses})")
            else:
                nan_count = 0
                self.optimizer.zero_grad()
                loss_val.backward()

                if self.cfg.clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradient_norm)

                self.optimizer.step()

            pbar.set_postfix_str(f"Loss: {loss_val.item():.4f}")
            self.experiment_logger.log_step(**{k: v.item() for k, v in all_losses.items()})

    def _set_random_seeds(self):
        if self.cfg.seed is None:
            self.cfg.seed = int(np.random.uniform(low=0, high=1e6))
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    def _set_device(self):
        if self.cfg.device is not None:
            if self.cfg.device.startswith("cuda"):
                gpu_id = int(self.cfg.device.split(":")[-1])
                if gpu_id > torch.cuda.device_count():
                    raise RuntimeError(f"This machine does not have GPU #{gpu_id} ")
                self.device = torch.device(self.cfg.device)
            elif self.cfg.device == "mps":
                if torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    raise RuntimeError("MPS device is not available.")
            else:
                self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        LOGGER.info(f"### Device {self.device} will be used for training")

    def _create_folder_structure(self):
        if self.cfg.is_continue_training:
            folder_name = f"continue_training_from_epoch{self._epoch:03d}"
            self.cfg.base_run_dir = self.cfg.run_dir
            self.cfg.run_dir = self.cfg.run_dir / folder_name
        else:
            now = datetime.now()
            day = f"{now.day}".zfill(2)
            month = f"{now.month}".zfill(2)
            hour = f"{now.hour}".zfill(2)
            minute = f"{now.minute}".zfill(2)
            second = f"{now.second}".zfill(2)
            run_name = f"{self.cfg.experiment_name}_{day}{month}_{hour}{minute}{second}"

            if self.cfg.run_dir is None:
                self.cfg.run_dir = Path().cwd() / "runs" / run_name
            else:
                self.cfg.run_dir = self.cfg.run_dir / run_name

        if not self.cfg.run_dir.is_dir():
            self.cfg.train_dir = self.cfg.run_dir / "train_data"
            self.cfg.train_dir.mkdir(parents=True)
        else:
            raise RuntimeError(f"There is already a folder at {self.cfg.run_dir}")

        if self.cfg.log_n_figures is not None:
            self.cfg.img_log_dir = self.cfg.run_dir / "img_log"
            self.cfg.img_log_dir.mkdir(parents=True)
