"""Training loop: masked MSE, early stopping, one self-contained checkpoint.

The checkpoint carries the config, the fitted normalizer and the flight ids of each split
alongside the weights. That is what makes inference reproducible without re-deriving
anything: ``forecast.py`` loads a checkpoint and knows the resample step, the channel
order, the horizon mode, and which flights the model must not be evaluated on.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from channels import CHANNELS
from batching import resolve_batch_size
from config import SAMPLING_AIRPORT_FLIGHT_BALANCED, TSConfig
from dataset import (
    ARRIVAL_DATA_PROVENANCE_SCHEMA,
    FlightSeries,
    Normalizer,
    TrajectoryWindows,
    iter_batches,
    provenance_manifest_digests,
    split_by_flight,
    window_anchors,
)
from metrics import error_by_horizon, trajectory_metrics
from models import build_model, parameter_count, resolve_device

CHECKPOINT_NAME = "checkpoint.pt"
CHECKPOINT_METADATA_NAME = "checkpoint_metadata.json"
CHECKPOINT_METADATA_SCHEMA = "ts-checkpoint-metadata-v2-multi-airport"
HISTORY_NAME = "history.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_sha256(series: Sequence[FlightSeries]) -> str:
    """Stable audit digest shared conceptually with cross-validation's split record."""
    payload = "\n".join(sorted(item.dataset_id for item in series)).encode()
    return hashlib.sha256(payload).hexdigest()


def masked_mse(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Weighted MSE over supervised channel values only.

    Legacy ``[B,H]`` masks still broadcast across channels. New datasets pass ``[B,H,C]``:
    measured rows weight all channels equally, while fitted rows weight position only.
    """
    if mask.ndim == predicted.ndim - 1:
        weights = mask.unsqueeze(-1).expand_as(predicted)
    elif mask.shape == predicted.shape:
        weights = mask
    else:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} cannot weight prediction {tuple(predicted.shape)}"
        )
    error = (predicted - target) ** 2 * weights
    denominator = weights.sum()
    return error.sum() / denominator.clamp(min=1.0)


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    seconds: float
    val_by_airport: dict[str, float]


@dataclass
class FitResult:
    """In-memory result shared by final training and cross-validation folds."""

    model: nn.Module
    config: TSConfig
    normalizer: Normalizer
    device: torch.device
    history: list[EpochResult]
    best_val_loss: float
    train_windows: int
    val_windows: int


def _predict_split(
    model: nn.Module,
    dataset: TrajectoryWindows,
    normalizer: Normalizer,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a split through the model; return PHYSICAL-unit ``(predicted, truth, mask)``."""
    model.eval()
    predicted_chunks, truth_chunks, mask_chunks = [], [], []
    with torch.no_grad():
        for x, y, mask in iter_batches(dataset, batch_size, shuffle=False, seed=0):
            out = model(x.to(device)).cpu().numpy()
            # Decode in float64 (the normalizer stats' dtype), store float32: a full-mode
            # split is tens of thousands of [H, C] windows held live at once, and float64
            # doubled the peak for precision the metre-scale metrics cannot use — this
            # machine is 16 GB and frequently swap-bound.
            predicted_chunks.append(normalizer.decode(out.astype(np.float64)).astype(np.float32))
            truth_chunks.append(normalizer.decode(y.numpy().astype(np.float64)).astype(np.float32))
            raw_mask = mask.numpy()
            # Headline ADE/FDE remain measured-data metrics. Position-only fitted rows are
            # training labels, not observations, and therefore stay out of this mask.
            if raw_mask.ndim == 3:
                raw_mask = np.all(raw_mask > 0.0, axis=-1).astype(np.float32)
            mask_chunks.append(raw_mask)
    return (
        np.concatenate(predicted_chunks), np.concatenate(truth_chunks), np.concatenate(mask_chunks)
    )


def evaluate_split(
    model: nn.Module,
    dataset: TrajectoryWindows,
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Physical-unit metrics for a split (ADE/FDE + the decomposed errors + the horizon curve)."""
    predicted, truth, mask = _predict_split(model, dataset, normalizer, device, config.batch_size)
    block = trajectory_metrics(predicted, truth, mask)
    block["by_horizon"] = error_by_horizon(predicted, truth, mask, config.dt_s)
    return block


def usable_series(
    series: Sequence[FlightSeries], config: TSConfig, *, verbose: bool = True
) -> list[FlightSeries]:
    """Drop flights that cannot yield one model window, once and with an audit count."""
    usable = [item for item in series if len(window_anchors(item, config)) > 0]
    if len(usable) < len(series) and verbose:
        need = config.seq_len + config.pred_len
        print(f"  excluded   {len(series) - len(usable)} flight(s) too short to yield one "
              f"training window (need {need} samples = {need * config.dt_s:.0f}s)")
    return usable


def _validation_datasets(
    series: Sequence[FlightSeries], config: TSConfig, normalizer: Normalizer
) -> dict[str, TrajectoryWindows]:
    by_airport: dict[str, list[FlightSeries]] = {}
    for item in series:
        by_airport.setdefault(item.airport or "<unknown>", []).append(item)
    return {
        airport: TrajectoryWindows(
            group, config, normalizer, anchor_policy=config.eval_anchor_policy
        )
        for airport, group in sorted(by_airport.items())
    }


def _dataset_loss(
    model: nn.Module,
    dataset: TrajectoryWindows,
    device: torch.device,
    batch_size: int,
) -> float:
    weighted_error = 0.0
    weight_total = 0.0
    with torch.no_grad():
        for x, y, mask in iter_batches(dataset, batch_size, shuffle=False, seed=0):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            predicted = model(x)
            weights = mask if mask.shape == predicted.shape else mask.unsqueeze(-1).expand_as(predicted)
            weighted_error += float((((predicted - y) ** 2) * weights).sum())
            weight_total += float(weights.sum())
    return weighted_error / max(weight_total, 1.0)


def fit_model(
    train_series: Sequence[FlightSeries],
    val_series: Sequence[FlightSeries],
    config: TSConfig,
    *,
    auto_batch_size: bool = False,
    verbose: bool = True,
) -> FitResult:
    """Fit one model against explicit train/validation flights, without touching test."""
    if not train_series or not val_series:
        raise ValueError("fit_model requires non-empty train and validation flights")

    device = resolve_device(config.device)
    batch_size = resolve_batch_size(config, device, auto=auto_batch_size, verbose=verbose)
    config = replace(config, batch_size=batch_size)

    # Reset after the isolated auto-batch probe so probing cannot change final initialisation.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    balanced = config.sampling_strategy == SAMPLING_AIRPORT_FLIGHT_BALANCED
    normalizer = Normalizer.fit(
        train_series, balance_airports_and_flights=balanced
    )
    train_set = TrajectoryWindows(train_series, config, normalizer)
    val_sets = _validation_datasets(val_series, config, normalizer)
    val_window_count = sum(len(dataset) for dataset in val_sets.values())
    if not len(train_set) or not val_window_count:
        raise ValueError(
            f"empty window set (train={len(train_set)}, val={val_window_count}) — "
            f"seq_len={config.seq_len} + pred_len={config.pred_len} may exceed the track lengths"
        )

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    requested_samples = config.train_samples_per_epoch or len(train_set)
    samples_per_epoch = (
        requested_samples if balanced else min(requested_samples, len(train_set))
    )
    if verbose:
        print(f"  model      {config.model} ({parameter_count(model):,} params) on {device}")
        print(f"  horizon    {config.horizon_mode}: L={config.seq_len} "
              f"({config.lookback_s:.0f}s) -> H={config.pred_len} ({config.horizon_s:.0f}s)")
        print(f"  flights    train {len(train_series)} / val {len(val_series)}")
        print(f"  windows    train {len(train_set)} / val {val_window_count} "
              f"(eval anchors: {config.eval_anchor_policy})")
        sampling = "airport -> flight -> anchor" if balanced else "all windows"
        print(f"  sampling   {sampling}; {samples_per_epoch} sample(s)/epoch")

    history: list[EpochResult] = []
    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()

        model.train()
        train_total, train_steps = 0.0, 0
        for x, y, mask in iter_batches(
            train_set,
            config.batch_size,
            shuffle=not balanced,
            seed=config.seed + epoch,
            balanced=balanced,
            num_samples=samples_per_epoch,
        ):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = masked_mse(model(x), y, mask)
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach())
            train_steps += 1

        model.eval()
        val_by_airport = {
            airport: _dataset_loss(model, dataset, device, config.batch_size)
            for airport, dataset in val_sets.items()
        }
        train_loss = train_total / max(train_steps, 1)
        # Equal airport weight: a large/long airport cannot control early stopping alone.
        val_loss = float(np.mean(list(val_by_airport.values())))
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            raise RuntimeError(
                f"training diverged at epoch {epoch} (train {train_loss}, val {val_loss}) "
                f"— no checkpoint written; lower --learning-rate"
            )
        scheduler.step(val_loss)
        history.append(EpochResult(
            epoch, train_loss, val_loss, time.perf_counter() - started, val_by_airport
        ))

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
            marker = " *"
        else:
            epochs_without_improvement += 1
            marker = ""

        if verbose:
            print(f"  epoch {epoch:3d}/{config.epochs}  train {train_loss:.6f}  "
                  f"val-macro {val_loss:.6f}  {history[-1].seconds:5.1f}s{marker}")

        if epochs_without_improvement >= config.patience:
            if verbose:
                print(f"  early stop: {config.patience} epochs without improvement")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return FitResult(
        model=model,
        config=config,
        normalizer=normalizer,
        device=device,
        history=history,
        best_val_loss=best_val,
        train_windows=len(train_set),
        val_windows=val_window_count,
    )


def train(
    series: Sequence[FlightSeries],
    config: TSConfig,
    *,
    output_dir: str | Path,
    data_provenance: dict[str, Any],
    auto_batch_size: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train one model on ``series``; write ``checkpoint.pt`` + ``history.json``.

    Returns the run summary (also embedded in the history file).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if data_provenance.get("schema_version") != ARRIVAL_DATA_PROVENANCE_SCHEMA:
        raise ValueError("data_provenance is not a TS arrival-data fingerprint")
    manifest_digests = provenance_manifest_digests(data_provenance)

    series = usable_series(series, config, verbose=verbose)
    train_series, val_series, test_series = split_by_flight(series, config)
    fit = fit_model(
        train_series,
        val_series,
        config,
        auto_batch_size=auto_batch_size,
        verbose=verbose,
    )
    model, config, normalizer, device = (
        fit.model, fit.config, fit.normalizer, fit.device
    )
    val_set = TrajectoryWindows(
        val_series, config, normalizer, anchor_policy=config.eval_anchor_policy
    )
    test_window_count = sum(
        min(len(window_anchors(item, config)), 1)
        if config.eval_anchor_policy == "first"
        else len(window_anchors(item, config))
        for item in test_series
    )
    split_metrics = {"val": evaluate_split(model, val_set, normalizer, config, device)}

    checkpoint_payload = {
        "config": config.to_dict(),
        "model_state": model.state_dict(),
        "normalizer": normalizer.to_dict(),
        "split": {
            "train": [s.dataset_id for s in train_series],
            "val": [s.dataset_id for s in val_series],
            "test": [s.dataset_id for s in test_series],
        },
        "best_val_loss": fit.best_val_loss,
        "data_provenance": data_provenance,
    }
    checkpoint_path = out / CHECKPOINT_NAME
    checkpoint_tmp = out / f"{CHECKPOINT_NAME}.tmp"
    torch.save(checkpoint_payload, checkpoint_tmp)
    checkpoint_tmp.replace(checkpoint_path)
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    checkpoint_metadata = {
        "schema_version": CHECKPOINT_METADATA_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "arrival_manifests": manifest_digests,
        "split_sha256": {
            "train": _split_sha256(train_series),
            "val": _split_sha256(val_series),
            "test": _split_sha256(test_series),
        },
    }
    metadata_path = out / CHECKPOINT_METADATA_NAME
    metadata_tmp = out / f"{CHECKPOINT_METADATA_NAME}.tmp"
    metadata_tmp.write_text(json.dumps(checkpoint_metadata, indent=2), encoding="utf-8")
    metadata_tmp.replace(metadata_path)

    summary = {
        "config": config.to_dict(),
        "parameters": parameter_count(model),
        "device": str(device),
        "epochs_run": len(fit.history),
        "best_val_loss": fit.best_val_loss,
        "flights": {"train": len(train_series), "val": len(val_series), "test": len(test_series)},
        "windows": {"train": fit.train_windows, "val": len(val_set), "test": test_window_count},
        "metrics": split_metrics,
        "data_provenance": {
            "schema_version": data_provenance["schema_version"],
            "arrival_manifests": manifest_digests,
            "source_record_count": sum(
                len(entry["source_records"]) for entry in data_provenance["manifests"]
            ),
        },
        "history": [vars(h) for h in fit.history],
    }
    (out / HISTORY_NAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        for split, block in split_metrics.items():
            print(f"  {split:5s}  ADE {block['ade_m']:7.1f} m   FDE {block['fde_m']:7.1f} m   "
                  f"cross-track p95 {block['cross_track_m']['p95_abs']:7.1f} m   "
                  f"alt p95 {block['altitude_m']['p95_abs']:6.1f} m")
        print(f"✓ wrote {out / CHECKPOINT_NAME} and {out / HISTORY_NAME}")

    return summary


def load_checkpoint(path: str | Path) -> tuple[nn.Module, TSConfig, Normalizer, dict[str, Any]]:
    """Rebuild a trained model from a checkpoint written by :func:`train`."""
    # weights_only=True: the payload is tensors + primitives only (config/normalizer are
    # plain dicts and lists), so nothing here needs — or should get — pickle execution.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    config = TSConfig.from_dict(payload["config"])
    if config.channels != CHANNELS:
        raise ValueError(
            f"checkpoint channel contract {config.channels} != this build's "
            f"channels.CHANNELS {CHANNELS} — the model and normalizer index the old "
            f"contract, the data build the new one, and a same-length mismatch would load "
            f"cleanly but silently mis-map (or mis-scale: ve/vn/vu -> edot/ndot/udot was a "
            f"semantics change) every channel. Re-train, or run the matching code version."
        )
    normalizer = Normalizer.from_dict(payload["normalizer"])
    model = build_model(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config, normalizer, payload
