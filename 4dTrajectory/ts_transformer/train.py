"""Training loop: normalized-progress state loss plus final-time loss.

The checkpoint carries the config, the fitted normalizer and the flight ids of each split
alongside the weights. That is what makes inference reproducible without re-deriving
anything: ``forecast.py`` loads a checkpoint and knows the resample step, channel order,
normalized segment count, and which flights the model must not be evaluated on.
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
from config import TSConfig
from dataset import (
    ARRIVAL_DATA_PROVENANCE_SCHEMA,
    FixedAnchorTrajectoryWindows,
    FlightSeries,
    Normalizer,
    RandomAnchorTrajectoryWindows,
    TrajectoryWindows,
    iter_batches,
    provenance_manifest_digests,
    split_by_flight,
    window_anchors,
)
from metrics import error_by_progress, trajectory_metrics
from models import build_model, parameter_count, resolve_device
from prediction_outputs import StatePrediction

CHECKPOINT_NAME = "checkpoint.pt"
CHECKPOINT_METADATA_NAME = "checkpoint_metadata.json"
CHECKPOINT_METADATA_SCHEMA = "ts-checkpoint-metadata-v7-anchor-policy"
TARGET_CONTRACT = "normalized-time-runway-crossing-v1"
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

    All three tensors use ``[B,N,C]``. Measured rows weight all channels equally, while
    fitted rows weight position only.
    """
    error = (predicted - target) ** 2 * mask
    denominator = mask.sum()
    return error.sum() / denominator.clamp(min=1.0)


def prediction_loss(
    prediction: StatePrediction,
    target_states: torch.Tensor,
    state_weights: torch.Tensor,
    target_final_time_s: torch.Tensor,
    flight_weights: torch.Tensor,
    config: TSConfig,
) -> torch.Tensor:
    """Airport-macro joint loss, with every flight represented once per epoch."""
    state_error = ((prediction.states - target_states) ** 2 * state_weights).sum(
        dim=(1, 2)
    )
    state_denominator = state_weights.sum(dim=(1, 2)).clamp(min=1.0)
    state_loss = state_error / state_denominator
    time_loss = (
        (prediction.final_time_s - target_final_time_s) / config.final_time_scale_s
    ).square()
    per_flight = state_loss + config.final_time_loss_weight * time_loss
    # Weights are normalized to mean one across the complete epoch. Keeping the minibatch
    # denominator independent of its airport composition gives an unbiased stochastic
    # estimate of that fixed airport-macro objective.
    return (per_flight * flight_weights).mean()


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return physical state arrays, state masks, predicted time and true time."""
    model.eval()
    predicted_chunks, truth_chunks, mask_chunks = [], [], []
    predicted_time_chunks, truth_time_chunks = [], []
    with torch.no_grad():
        for x, y, mask, final_time_s, _flight_weights in iter_batches(
            dataset, batch_size, shuffle=False, seed=0
        ):
            output = model(x.to(device))
            out = output.states.cpu().numpy()
            # Decode in float64 (the normalizer stats' dtype), store float32: a pooled
            # split is tens of thousands of [N, C] windows held live at once, and float64
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
            predicted_time_chunks.append(output.final_time_s.cpu().numpy())
            truth_time_chunks.append(final_time_s.numpy())
    return (
        np.concatenate(predicted_chunks),
        np.concatenate(truth_chunks),
        np.concatenate(mask_chunks),
        np.concatenate(predicted_time_chunks),
        np.concatenate(truth_time_chunks),
    )


def evaluate_split(
    model: nn.Module,
    dataset: TrajectoryWindows,
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Physical-unit state and final-time metrics for a split."""
    predicted, truth, mask, predicted_time, truth_time = _predict_split(
        model, dataset, normalizer, device, config.batch_size
    )
    block = trajectory_metrics(predicted, truth, mask)
    block["by_progress"] = error_by_progress(predicted, truth, mask)
    time_error = predicted_time - truth_time
    block["final_time_s"] = {
        "mae": float(np.abs(time_error).mean()),
        "rmse": float(np.sqrt(np.mean(time_error**2))),
        "mean_signed": float(time_error.mean()),
    }
    return block


def usable_series(
    series: Sequence[FlightSeries],
    config: TSConfig,
    *,
    minimum_anchor_index: int | None = None,
    verbose: bool = True,
) -> list[FlightSeries]:
    """Drop flights that cannot yield one model window, once and with an audit count."""
    usable = [
        item for item in series
        if len(window_anchors(
            item, config, minimum_anchor_index=minimum_anchor_index
        )) > 0
    ]
    if len(usable) < len(series) and verbose:
        need = config.seq_len + 1
        print(f"  excluded   {len(series) - len(usable)} flight(s) too short to yield one "
              f"training window (need {need} samples = {need * config.dt_s:.0f}s)")
    return usable


def _validation_datasets(
    series: Sequence[FlightSeries],
    config: TSConfig,
    normalizer: Normalizer,
    *,
    minimum_anchor_index: int | None = None,
) -> dict[str, TrajectoryWindows]:
    by_airport: dict[str, list[FlightSeries]] = {}
    for item in series:
        by_airport.setdefault(item.airport or "<unknown>", []).append(item)
    return {
        airport: FixedAnchorTrajectoryWindows(
            group,
            config,
            normalizer,
            minimum_anchor_index=minimum_anchor_index,
        )
        for airport, group in sorted(by_airport.items())
    }


def _dataset_loss(
    model: nn.Module,
    dataset: TrajectoryWindows,
    device: torch.device,
    batch_size: int,
) -> float:
    loss_total = 0.0
    flight_weight_total = 0.0
    with torch.no_grad():
        for x, y, mask, final_time_s, flight_weights in iter_batches(
            dataset, batch_size, shuffle=False, seed=0
        ):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            final_time_s = final_time_s.to(device)
            flight_weights = flight_weights.to(device)
            prediction = model(x)
            state_error = (((prediction.states - y) ** 2) * mask).sum(dim=(1, 2))
            state_loss = state_error / mask.sum(dim=(1, 2)).clamp(min=1.0)
            time_loss = (
                (prediction.final_time_s - final_time_s)
                / dataset.config.final_time_scale_s
            ).square()
            per_flight = state_loss + dataset.config.final_time_loss_weight * time_loss
            loss_total += float((per_flight * flight_weights).sum())
            flight_weight_total += float(flight_weights.sum())
    return loss_total / max(flight_weight_total, 1.0)


def fit_model(
    train_series: Sequence[FlightSeries],
    val_series: Sequence[FlightSeries],
    config: TSConfig,
    *,
    auto_batch_size: bool = False,
    minimum_anchor_index: int | None = None,
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
    normalizer = Normalizer.fit(train_series, balance_airports_and_flights=True)
    training_dataset_class = {
        False: FixedAnchorTrajectoryWindows,
        True: RandomAnchorTrajectoryWindows,
    }[config.random_train_anchor]
    train_set = training_dataset_class(
        train_series,
        config,
        normalizer,
        minimum_anchor_index=minimum_anchor_index,
    )
    val_sets = _validation_datasets(
        val_series,
        config,
        normalizer,
        minimum_anchor_index=minimum_anchor_index,
    )
    val_window_count = sum(len(dataset) for dataset in val_sets.values())
    if not len(train_set) or not val_window_count:
        raise ValueError(
            f"empty window set (train={len(train_set)}, val={val_window_count}) — "
            f"seq_len={config.seq_len}, minimum_anchor_index={minimum_anchor_index!r} "
            "leaves no future remainder in these tracks"
        )

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    flights_per_epoch = sum(
        count > 0 for _start, count in train_set.series_ranges.values()
    )
    if verbose:
        print(f"  model      {config.model} ({parameter_count(model):,} params) on {device}")
        print(f"  prediction L={config.seq_len} ({config.lookback_s:.0f}s history) -> "
              f"N={config.n_segments} normalized progress segments + final_time_s")
        print(f"  flights    train {len(train_series)} / val {len(val_series)}")
        print(f"  windows    train {len(train_set)} / val {val_window_count} "
              "(validation anchor: fixed L-1)")
        print(f"  anchors    {train_set.anchor_description}")
        if minimum_anchor_index is not None:
            print(f"  anchor     common minimum index {minimum_anchor_index} "
                  f"({minimum_anchor_index * config.dt_s:.0f}s after track entry)")
        print(
            f"  sampling   one shuffled sample/flight; {flights_per_epoch} flight(s)/epoch; "
            "airport-macro loss weights"
        )

    history: list[EpochResult] = []
    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()

        model.train()
        train_total, train_weight_total = 0.0, 0.0
        for x, y, mask, final_time_s, flight_weights in iter_batches(
            train_set,
            config.batch_size,
            shuffle=True,
            seed=config.seed + epoch,
        ):
            batch_count = len(flight_weights)
            batch_weight = float(flight_weights.sum())
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            final_time_s = final_time_s.to(device)
            flight_weights = flight_weights.to(device)
            optimizer.zero_grad()
            loss = prediction_loss(
                model(x), y, mask, final_time_s, flight_weights, config
            )
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach()) * batch_count
            train_weight_total += batch_weight

        model.eval()
        val_by_airport = {
            airport: _dataset_loss(model, dataset, device, config.batch_size)
            for airport, dataset in val_sets.items()
        }
        train_loss = train_total / max(train_weight_total, 1.0)
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
    val_set = FixedAnchorTrajectoryWindows(val_series, config, normalizer)
    test_window_count = sum(
        min(len(window_anchors(item, config)), 1)
        for item in test_series
    )
    split_metrics = {"val": evaluate_split(model, val_set, normalizer, config, device)}

    checkpoint_payload = {
        "target_contract": TARGET_CONTRACT,
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
        "random_train_anchor": config.random_train_anchor,
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
                  f"alt p95 {block['altitude_m']['p95_abs']:6.1f} m   "
                  f"time MAE {block['final_time_s']['mae']:5.1f} s")
        print(f"✓ wrote {out / CHECKPOINT_NAME} and {out / HISTORY_NAME}")

    return summary


def load_checkpoint(path: str | Path) -> tuple[nn.Module, TSConfig, Normalizer, dict[str, Any]]:
    """Rebuild a trained model from a checkpoint written by :func:`train`."""
    # weights_only=True: the payload is tensors + primitives only (config/normalizer are
    # plain dicts and lists), so nothing here needs — or should get — pickle execution.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("target_contract") != TARGET_CONTRACT:
        raise ValueError(
            "checkpoint predates the runway-crossing target contract; retrain it"
        )
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
