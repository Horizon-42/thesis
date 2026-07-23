"""Training loop: masked MSE, early stopping, one self-contained checkpoint.

The checkpoint carries the config, the fitted normalizer and the flight ids of each split
alongside the weights. That is what makes inference reproducible without re-deriving
anything: ``forecast.py`` loads a checkpoint and knows the resample step, the channel
order, the horizon mode, and which flights the model must not be evaluated on.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from channels import CHANNELS
from config import TSConfig
from dataset import (
    FlightSeries, Normalizer, TrajectoryWindows, iter_batches, split_by_flight, window_anchors,
)
from metrics import error_by_horizon, trajectory_metrics
from models import build_model, parameter_count, resolve_device

CHECKPOINT_NAME = "checkpoint.pt"
HISTORY_NAME = "history.json"


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


def train(
    series: Sequence[FlightSeries],
    config: TSConfig,
    *,
    output_dir: str | Path,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train one model on ``series``; write ``checkpoint.pt`` + ``history.json``.

    Returns the run summary (also embedded in the history file).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.device)

    # In window mode a flight can be long enough to BUILD (seq_len + 1 samples) yet too
    # short to yield a single training window (seq_len + pred_len). Excluding those here —
    # counted, never silent — keeps them out of the split roster, where they would occupy
    # slots while contributing zero windows.
    usable = [s for s in series if len(window_anchors(s, config)) > 0]
    if len(usable) < len(series) and verbose:
        need = config.seq_len + config.pred_len
        print(f"  excluded   {len(series) - len(usable)} flight(s) too short to yield one "
              f"training window (need {need} samples = {need * config.dt_s:.0f}s)")
    series = usable

    train_series, val_series, test_series = split_by_flight(series, config)
    normalizer = Normalizer.fit(train_series)

    train_set = TrajectoryWindows(train_series, config, normalizer)
    val_set = TrajectoryWindows(val_series, config, normalizer)
    test_set = TrajectoryWindows(test_series, config, normalizer)
    if not len(train_set) or not len(val_set):
        raise ValueError(
            f"empty window set (train={len(train_set)}, val={len(val_set)}) — "
            f"seq_len={config.seq_len} + pred_len={config.pred_len} may exceed the track lengths"
        )

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    if verbose:
        print(f"  model      {config.model} ({parameter_count(model):,} params) on {device}")
        print(f"  horizon    {config.horizon_mode}: L={config.seq_len} "
              f"({config.lookback_s:.0f}s) -> H={config.pred_len} ({config.horizon_s:.0f}s)")
        print(f"  flights    train {len(train_series)} / val {len(val_series)} / test {len(test_series)}")
        print(f"  windows    train {len(train_set)} / val {len(val_set)} / test {len(test_set)}")

    history: list[EpochResult] = []
    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()

        model.train()
        train_total, train_steps = 0.0, 0
        for x, y, mask in iter_batches(train_set, config.batch_size, shuffle=True,
                                       seed=config.seed + epoch):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = masked_mse(model(x), y, mask)
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach())
            train_steps += 1

        model.eval()
        val_total, val_steps = 0.0, 0
        with torch.no_grad():
            for x, y, mask in iter_batches(val_set, config.batch_size, shuffle=False, seed=0):
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                val_total += float(masked_mse(model(x), y, mask))
                val_steps += 1

        train_loss = train_total / max(train_steps, 1)
        val_loss = val_total / max(val_steps, 1)
        # NaN compares False against everything, so a diverged run would sail through the
        # best-val bookkeeping below: best_state never captured, the LAST (NaN) weights
        # checkpointed, and "✓ wrote checkpoint.pt" printed as if the run succeeded.
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            raise RuntimeError(
                f"training diverged at epoch {epoch} (train {train_loss}, val {val_loss}) "
                f"— no checkpoint written; lower --learning-rate"
            )
        scheduler.step(val_loss)
        history.append(EpochResult(epoch, train_loss, val_loss, time.perf_counter() - started))

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            # .clone() — state_dict() returns live tensor references, so without it the
            # "best" snapshot keeps mutating as training continues past the best epoch.
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            marker = " *"
        else:
            epochs_without_improvement += 1
            marker = ""

        if verbose:
            print(f"  epoch {epoch:3d}/{config.epochs}  train {train_loss:.6f}  "
                  f"val {val_loss:.6f}  {history[-1].seconds:5.1f}s{marker}")

        if epochs_without_improvement >= config.patience:
            if verbose:
                print(f"  early stop: {config.patience} epochs without improvement")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    split_metrics = {"val": evaluate_split(model, val_set, normalizer, config, device)}
    if len(test_set):
        split_metrics["test"] = evaluate_split(model, test_set, normalizer, config, device)

    torch.save({
        "config": config.to_dict(),
        "model_state": model.state_dict(),
        "normalizer": normalizer.to_dict(),
        "split": {
            "train": [s.flight_id for s in train_series],
            "val": [s.flight_id for s in val_series],
            "test": [s.flight_id for s in test_series],
        },
        "best_val_loss": best_val,
    }, out / CHECKPOINT_NAME)

    summary = {
        "config": config.to_dict(),
        "parameters": parameter_count(model),
        "device": str(device),
        "epochs_run": len(history),
        "best_val_loss": best_val,
        "flights": {"train": len(train_series), "val": len(val_series), "test": len(test_series)},
        "windows": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
        "metrics": split_metrics,
        "history": [vars(h) for h in history],
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
