"""Training loop: configured-time-grid state loss plus final-time/physics losses.

The checkpoint carries the config, the fitted normalizer and the flight ids of each split
alongside the weights. That is what makes inference reproducible without re-deriving
anything: ``forecast.py`` loads a checkpoint and knows the resample step, channel order,
output length/time mode, and which flights the model must not be evaluated on.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from channels import CHANNELS, IDX, POSITION_IDX
from batching import resolve_batch_size
from config import (
    HORIZON_FULL,
    HORIZON_NORMALIZED,
    HORIZON_WINDOW,
    PREDICTION_CONTROL,
    PREDICTION_STATE,
    TSConfig,
)
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
from evaluation_protocol import (
    TEST_RELEASE_NAME,
    TEST_RELEASE_PROTOCOL_FIELD,
    TEST_RELEASE_SCHEMA,
)
from metrics import error_by_progress, raw_kinematic_metrics, trajectory_metrics
from models import build_model, parameter_count, resolve_device
from prediction_outputs import ControlPrediction, StatePrediction
from time_grids import batch_time_grid, numpy_inference_time_grid
from aerodynamic_model.torch_dynamics import (
    geodetic_states_to_channels,
    rollout_piecewise_constant as torch_control_rollout,
)

CHECKPOINT_NAME = "checkpoint.pt"
CHECKPOINT_METADATA_NAME = "checkpoint_metadata.json"
CHECKPOINT_METADATA_SCHEMA = "ts-checkpoint-metadata-v14-aircraft-filter-audit"
STATE_TARGET_CONTRACTS = {
    HORIZON_NORMALIZED: "normalized-time-runway-crossing-displacement-kinematic-v3",
    HORIZON_FULL: "full-horizon-fixed-time-displacement-kinematic-v2",
    HORIZON_WINDOW: "recursive-window-fixed-time-displacement-kinematic-v2",
}
HISTORY_NAME = "history.json"
FIT_EVALUATION_NAME = "fit_evaluation.json"
FIT_EVALUATION_SCHEMA = "ts-fit-evaluation-v1-best-checkpoint-fixed-anchor"
STATE_LOSS_COMPONENT_NAMES = ("state", "final_time", "kinematic", "terminal")
CONTROL_LOSS_COMPONENT_NAMES = (
    "state", "final_time", "kinematic", "terminal", "control_effort", "control_smoothness"
)


def target_contract(config: TSConfig) -> str:
    if config.prediction_output == PREDICTION_STATE:
        return STATE_TARGET_CONTRACTS[config.horizon_mode]
    return "bounded-control-nonuniform-duration-casadi-rollout-clock-aligned-v2"


def loss_component_names(config: TSConfig) -> tuple[str, ...]:
    return (
        CONTROL_LOSS_COMPONENT_NAMES
        if config.prediction_output == PREDICTION_CONTROL
        else STATE_LOSS_COMPONENT_NAMES
    )


def move_dynamics(
    dynamics: dict[str, torch.Tensor] | None, device: torch.device
) -> dict[str, torch.Tensor] | None:
    if dynamics is None:
        return None
    return {name: value.to(device) for name, value in dynamics.items()}


def unpack_batch(batch: tuple) -> tuple:
    if len(batch) == 5:
        return (*batch, None)
    if len(batch) == 6:
        return batch
    raise ValueError(f"unexpected trajectory batch with {len(batch)} fields")


def model_forward(
    model: nn.Module,
    history: torch.Tensor,
    dynamics: dict[str, torch.Tensor] | None,
):
    return model(history) if dynamics is None else model(history, dynamics)


def control_rollout_channels(
    prediction: ControlPrediction,
    dynamics: dict[str, torch.Tensor],
    config: TSConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return physical chart channels and geodetic segment endpoints."""
    # ECEF coordinates are O(6e6 m); float32 subtraction would quantize local offsets at
    # roughly half-metre resolution and accumulate through a long rollout.  The dynamics
    # contract therefore runs in float64 even when the network itself trains in float32;
    # ``to(float64)`` remains differentiable and gradients return to the FP32 parameters.
    rollout_dtype = torch.float64
    geodetic = torch_control_rollout(
        dynamics["initial_state"].to(rollout_dtype),
        prediction.controls.to(rollout_dtype),
        prediction.segment_durations.to(rollout_dtype),
        dynamics["aero_params"].to(rollout_dtype),
        integrator_dt_s=config.control_rollout_integrator_dt_s,
    )
    channels = geodetic_states_to_channels(
        geodetic,
        dynamics["frame_params"].to(rollout_dtype),
        runway_aligned=config.coordinate_frame == "runway-aligned",
    )
    return channels, geodetic


def align_control_targets_to_prediction_clock(
    normalized_anchor_state: torch.Tensor,
    target_states: torch.Tensor,
    state_weights: torch.Tensor,
    predicted_segment_durations_s: torch.Tensor,
    target_final_time_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate normalized truth onto learned cumulative control timestamps.

    Control targets arrive on the normalized true clock ``i * T_true / N``. A learned
    non-uniform partition instead produces endpoints at ``cumsum(Delta_t_hat)``. Comparing
    rows by index would therefore compare different physical times. Prepending the observed
    anchor supplies the ``t=0`` node; queries after the true endpoint clamp to its terminal
    state while the separate final-time loss continues to penalize their clock error.
    """
    if target_states.shape != state_weights.shape or target_states.ndim != 3:
        raise ValueError("control targets and weights must be aligned [B,N,C] tensors")
    batch, segments, channels = target_states.shape
    if normalized_anchor_state.shape != (batch, channels):
        raise ValueError("normalized control anchors must be [B,C]")
    if predicted_segment_durations_s.shape != (batch, segments):
        raise ValueError("predicted control durations must be [B,N]")
    if target_final_time_s.shape != (batch,):
        raise ValueError("target final time must be [B]")
    if torch.any(target_final_time_s <= 0.0):
        raise ValueError("control target final time must be positive")

    dtype, device = target_states.dtype, target_states.device
    anchor = normalized_anchor_state.to(dtype=dtype, device=device).unsqueeze(1)
    source_states = torch.cat((anchor, target_states), dim=1)
    # The anchor is always an observed input row, so all channels carry the measured-row
    # weight used by dataset._build_supervision. Later fitted-tail masks come from targets.
    anchor_weights = torch.full(
        (batch, 1, channels),
        1.0 / channels,
        dtype=state_weights.dtype,
        device=state_weights.device,
    )
    source_weights = torch.cat((anchor_weights, state_weights), dim=1)

    query_progress = (
        predicted_segment_durations_s.to(dtype=dtype, device=device).cumsum(dim=1)
        / target_final_time_s.to(dtype=dtype, device=device).unsqueeze(1)
    ).clamp(min=0.0, max=1.0)
    source_coordinate = query_progress * segments
    left_index = torch.floor(source_coordinate).to(torch.long).clamp(max=segments)
    right_index = (left_index + 1).clamp(max=segments)
    fraction = source_coordinate - left_index.to(dtype)

    def interpolate(source: torch.Tensor) -> torch.Tensor:
        gather_shape = left_index.unsqueeze(-1).expand(-1, -1, channels)
        left = torch.gather(source, 1, gather_shape)
        right = torch.gather(
            source,
            1,
            right_index.unsqueeze(-1).expand(-1, -1, channels),
        )
        return left + fraction.unsqueeze(-1) * (right - left)

    return interpolate(source_states), interpolate(source_weights)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_sha256(series: Sequence[FlightSeries]) -> str:
    """Stable audit digest shared conceptually with cross-validation's split record."""
    return _keys_sha256(item.dataset_id for item in series)


def _keys_sha256(keys: Iterable[str]) -> str:
    payload = "\n".join(sorted(keys)).encode()
    return hashlib.sha256(payload).hexdigest()


def masked_mse(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Weighted MSE over supervised channel values only.

    All three tensors use ``[B,N,C]``. Measured rows weight all channels equally, while
    fitted rows weight position only.
    """
    error = (predicted - target) ** 2 * mask
    denominator = mask.sum()
    return error.sum() / denominator.clamp(min=1.0)


@dataclass(frozen=True)
class LossComponents:
    """Weighted scalar contributions whose sum is the optimization objective."""

    state: torch.Tensor
    final_time: torch.Tensor
    kinematic: torch.Tensor
    terminal: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def total(self) -> torch.Tensor:
        return (
            self.state + self.final_time + self.kinematic + self.terminal
            + sum(self.extras.values(), self.state.new_zeros(()))
        )

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "state": self.state,
            "final_time": self.final_time,
            "kinematic": self.kinematic,
            "terminal": self.terminal,
            **self.extras,
        }


def position_velocity_consistency_loss(
    normalized_anchor_state: torch.Tensor,
    normalized_states: torch.Tensor,
    target_final_time_s: torch.Tensor,
    normalizer: Normalizer,
    *,
    config: TSConfig | None = None,
    state_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-flight displacement implied by position versus integrated velocity.

    State predictions are standardized channel-wise, so the positions and velocities are
    decoded before differencing. The displacement residual is divided by each fitted
    position scale. Unlike dividing finite-difference velocity by velocity scale, this does
    not make the position gradient grow as ``1 / dt`` when N increases. Ground-truth
    duration defines ``dt`` during training, so the time head cannot shrink this loss.
    """
    _batch_size, n_segments, _channels = normalized_states.shape
    normalized_states = torch.cat(
        (normalized_anchor_state.unsqueeze(1), normalized_states), dim=1
    )
    position_indices = list(POSITION_IDX)
    velocity_indices = [IDX["edot"], IDX["ndot"], IDX["udot"]]
    dtype, device = normalized_states.dtype, normalized_states.device
    position_mean = torch.as_tensor(
        normalizer.mean[position_indices], dtype=dtype, device=device
    )
    position_scale = torch.as_tensor(
        normalizer.std[position_indices], dtype=dtype, device=device
    )
    velocity_mean = torch.as_tensor(
        normalizer.mean[velocity_indices], dtype=dtype, device=device
    )
    velocity_scale = torch.as_tensor(
        normalizer.std[velocity_indices], dtype=dtype, device=device
    )

    positions = (
        normalized_states[..., position_indices] * position_scale + position_mean
    )
    velocities = (
        normalized_states[..., velocity_indices] * velocity_scale + velocity_mean
    )
    if config is None:
        durations = (target_final_time_s / n_segments).to(dtype=dtype).view(-1, 1)
        durations = durations.expand(-1, n_segments)
        active = torch.ones_like(durations, dtype=torch.bool)
    else:
        durations, active = batch_time_grid(target_final_time_s.to(dtype=dtype), config)
    if state_weights is not None:
        active = active & (state_weights.sum(dim=-1) > 0.0)
    interval_velocity = 0.5 * (velocities[:, 1:] + velocities[:, :-1])
    displacement_residual = (
        positions[:, 1:] - positions[:, :-1]
        - interval_velocity * durations.unsqueeze(-1)
    )
    normalized_residual = displacement_residual / position_scale
    squared = normalized_residual.square() * active.unsqueeze(-1)
    denominator = (active.sum(dim=1) * len(position_indices)).clamp(min=1)
    return squared.sum(dim=(1, 2)) / denominator


def control_prediction_loss_components(
    prediction: ControlPrediction,
    normalized_anchor_state: torch.Tensor,
    target_states: torch.Tensor,
    state_weights: torch.Tensor,
    target_final_time_s: torch.Tensor,
    flight_weights: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor],
) -> LossComponents:
    """State-space supervision through the differentiable control rollout."""
    physical_channels, _geodetic = control_rollout_channels(prediction, dynamics, config)
    dtype, device = physical_channels.dtype, physical_channels.device
    mean = torch.as_tensor(normalizer.mean, dtype=dtype, device=device)
    scale = torch.as_tensor(normalizer.std, dtype=dtype, device=device)
    normalized_states = (physical_channels - mean) / scale
    aligned_targets, aligned_weights = align_control_targets_to_prediction_clock(
        normalized_anchor_state,
        target_states,
        state_weights,
        prediction.segment_durations,
        target_final_time_s,
    )
    state_error = (
        (normalized_states - aligned_targets) ** 2 * aligned_weights
    ).sum(dim=(1, 2))
    state_loss = state_error / aligned_weights.sum(dim=(1, 2)).clamp(min=1.0)
    time_loss = (
        (prediction.final_time_s - target_final_time_s) / config.final_time_scale_s
    ).square()
    terminal_delta = normalized_states[:, -1, list(POSITION_IDX)] - target_states[
        :, -1, list(POSITION_IDX)
    ]
    terminal_loss = terminal_delta.square().mean(dim=1)

    lower = dynamics["control_lower"]
    upper = dynamics["control_upper"]
    effort_scale = torch.stack(
        (upper[:, 0], torch.full_like(upper[:, 1], math.pi / 2.0), upper[:, 2]), dim=-1
    )
    scaled_controls = prediction.controls / effort_scale.unsqueeze(1)
    effort = scaled_controls.square().mean(dim=(1, 2))
    if prediction.controls.shape[1] > 1:
        change_scale = (upper - lower).unsqueeze(1)
        changes = torch.diff(prediction.controls, dim=1) / change_scale
        smoothness = changes.square().mean(dim=(1, 2))
    else:
        smoothness = effort.new_zeros(effort.shape)

    def weighted_mean(values: torch.Tensor) -> torch.Tensor:
        return (values * flight_weights).mean()

    zero = weighted_mean(state_loss.new_zeros(state_loss.shape))
    return LossComponents(
        state=weighted_mean(state_loss),
        final_time=config.final_time_loss_weight * weighted_mean(time_loss),
        # State/velocity consistency is structural because both come from one dynamics rollout.
        kinematic=zero,
        terminal=config.terminal_loss_weight * weighted_mean(terminal_loss),
        extras={
            "control_effort": config.control_effort_loss_weight * weighted_mean(effort),
            "control_smoothness": (
                config.control_smoothness_loss_weight * weighted_mean(smoothness)
            ),
        },
    )


def prediction_loss_components(
    prediction: StatePrediction | ControlPrediction,
    normalized_anchor_state: torch.Tensor,
    target_states: torch.Tensor,
    state_weights: torch.Tensor,
    target_final_time_s: torch.Tensor,
    flight_weights: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor] | None = None,
) -> LossComponents:
    """Return the selected output strategy's airport-macro loss contributions."""
    if isinstance(prediction, ControlPrediction):
        if dynamics is None:
            raise ValueError("control prediction loss requires per-flight dynamics")
        return control_prediction_loss_components(
            prediction,
            normalized_anchor_state,
            target_states,
            state_weights,
            target_final_time_s,
            flight_weights,
            config,
            normalizer,
            dynamics,
        )
    state_error = ((prediction.states - target_states) ** 2 * state_weights).sum(
        dim=(1, 2)
    )
    state_denominator = state_weights.sum(dim=(1, 2)).clamp(min=1.0)
    state_loss = state_error / state_denominator
    time_loss = (
        (prediction.final_time_s - target_final_time_s) / config.final_time_scale_s
    ).square()
    if config.kinematic_consistency_loss_weight:
        kinematic_loss = position_velocity_consistency_loss(
            normalized_anchor_state,
            prediction.states,
            target_final_time_s,
            normalizer,
            config=config,
            state_weights=state_weights,
        )
    else:
        kinematic_loss = state_loss.new_zeros(state_loss.shape)
    if config.terminal_loss_weight:
        position_valid = state_weights[..., list(POSITION_IDX)].sum(dim=-1) > 0.0
        terminal_index = {
            HORIZON_NORMALIZED: lambda: torch.full(
                (len(target_states),),
                target_states.shape[1] - 1,
                dtype=torch.long,
                device=target_states.device,
            ),
            HORIZON_FULL: lambda: position_valid.long().sum(dim=1).clamp(min=1) - 1,
            HORIZON_WINDOW: lambda: position_valid.long().sum(dim=1).clamp(min=1) - 1,
        }[config.horizon_mode]()
        rows = torch.arange(len(target_states), device=target_states.device)
        terminal_delta = (
            prediction.states[rows, terminal_index][:, list(POSITION_IDX)]
            - target_states[rows, terminal_index][:, list(POSITION_IDX)]
        )
        terminal_loss = terminal_delta.square().mean(dim=1)
    else:
        terminal_loss = state_loss.new_zeros(state_loss.shape)

    def weighted_mean(values: torch.Tensor) -> torch.Tensor:
        return (values * flight_weights).mean()

    return LossComponents(
        state=weighted_mean(state_loss),
        final_time=config.final_time_loss_weight * weighted_mean(time_loss),
        kinematic=(
            config.kinematic_consistency_loss_weight * weighted_mean(kinematic_loss)
        ),
        terminal=config.terminal_loss_weight * weighted_mean(terminal_loss),
    )


def prediction_loss(
    prediction: StatePrediction | ControlPrediction,
    normalized_anchor_state: torch.Tensor,
    target_states: torch.Tensor,
    state_weights: torch.Tensor,
    target_final_time_s: torch.Tensor,
    flight_weights: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Airport-macro state/time/physics loss, one sample per flight and epoch."""
    # Weights are normalized to mean one across the complete epoch. Keeping the minibatch
    # denominator independent of its airport composition gives an unbiased stochastic
    # estimate of that fixed airport-macro objective.
    return prediction_loss_components(
        prediction,
        normalized_anchor_state,
        target_states,
        state_weights,
        target_final_time_s,
        flight_weights,
        config,
        normalizer,
        dynamics,
    ).total


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    optimizer_updates: int
    seconds: float
    val_by_airport: dict[str, float]
    train_components: dict[str, float]
    val_components: dict[str, float]


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return physical anchor/output arrays, masks, predicted time and true time."""
    model.eval()
    predicted_chunks, truth_chunks, mask_chunks = [], [], []
    predicted_time_chunks, truth_time_chunks, anchor_chunks, duration_chunks = [], [], [], []
    with torch.no_grad():
        for raw_batch in iter_batches(dataset, batch_size, shuffle=False, seed=0):
            x, y, mask, final_time_s, _flight_weights, dynamics = unpack_batch(raw_batch)
            x_device = x.to(device)
            dynamics_device = move_dynamics(dynamics, device)
            output = model_forward(model, x_device, dynamics_device)
            metric_targets = y
            metric_weights = mask
            if isinstance(output, ControlPrediction):
                physical, _geodetic = control_rollout_channels(
                    output, dynamics_device, dataset.config
                )
                out = physical.cpu().numpy().astype(np.float32)
                predicted_physical = out
                metric_targets, metric_weights = align_control_targets_to_prediction_clock(
                    x_device[:, -1],
                    y.to(device),
                    mask.to(device),
                    output.segment_durations,
                    final_time_s.to(device),
                )
                metric_targets = metric_targets.cpu()
                metric_weights = metric_weights.cpu()
                duration_chunks.append(output.segment_durations.cpu().numpy())
            else:
                out = output.states.cpu().numpy()
                predicted_physical = normalizer.decode(out.astype(np.float64)).astype(
                    np.float32
                )
                duration_chunks.append(
                    numpy_inference_time_grid(
                        output.final_time_s.cpu().numpy(), dataset.config
                    )[0]
                )
            # Decode in float64 (the normalizer stats' dtype), store float32: a pooled
            # split is tens of thousands of [N, C] windows held live at once, and float64
            # doubled the peak for precision the metre-scale metrics cannot use — this
            # machine is 16 GB and frequently swap-bound.
            predicted_chunks.append(predicted_physical)
            truth_chunks.append(
                normalizer.decode(metric_targets.numpy().astype(np.float64)).astype(
                    np.float32
                )
            )
            anchor_chunks.append(
                normalizer.decode(x[:, -1].numpy().astype(np.float64)).astype(np.float32)
            )
            raw_mask = metric_weights.numpy()
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
        np.concatenate(anchor_chunks),
        np.concatenate(duration_chunks),
    )


def evaluate_split(
    model: nn.Module,
    dataset: TrajectoryWindows,
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Physical-unit state and final-time metrics for a split."""
    predicted, truth, mask, predicted_time, truth_time, anchors, segment_durations_s = _predict_split(
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
    # Raw model nodes on their own predicted clock: no measured-track interpolation,
    # spline, filtering or CZML resampling. Durations are explicit [B,N] so this call site
    # remains valid when the output layer moves from uniform to nonuniform segments.
    active_segments = segment_durations_s > 0.0
    block["raw_kinematics"] = raw_kinematic_metrics(
        anchors, predicted, segment_durations_s, valid_segments=active_segments
    )
    return block


def _generalization_metric(train_value: float, val_value: float) -> dict[str, float | None]:
    return {
        "train": float(train_value),
        "val": float(val_value),
        "absolute_gap": float(val_value - train_value),
        "ratio": float(val_value / train_value) if train_value != 0.0 else None,
    }


def _training_objective_diagnostics(
    history: Sequence[EpochResult | dict[str, Any]], config: TSConfig
) -> dict[str, Any] | None:
    if not history:
        return None
    rows = [vars(row) if isinstance(row, EpochResult) else row for row in history]
    best = min(rows, key=lambda row: row["val_loss"])
    result = {
        "best_epoch": int(best["epoch"]),
        "epochs_run": len(rows),
        "reached_epoch_budget": len(rows) == config.epochs,
        "train_loss_at_best_epoch": float(best["train_loss"]),
        "val_loss_at_best_epoch": float(best["val_loss"]),
        "absolute_gap": float(best["val_loss"] - best["train_loss"]),
        "ratio": (
            float(best["val_loss"] / best["train_loss"])
            if best["train_loss"] != 0.0 else None
        ),
    }
    last = rows[-1]
    if "learning_rate" in last:
        result["final_learning_rate"] = float(last["learning_rate"])
    if "optimizer_updates" in last:
        result["total_optimizer_updates"] = int(last["optimizer_updates"])
    return result


def evaluate_fit_splits(
    model: nn.Module,
    train_series: Sequence[FlightSeries],
    val_series: Sequence[FlightSeries],
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    *,
    history: Sequence[EpochResult | dict[str, Any]] = (),
) -> dict[str, Any]:
    """Replay the best model deterministically on fixed-anchor train and validation.

    This is the shared post-fit/standalone evaluation seam. It deliberately ignores the
    training anchor policy: both splits use one fixed ``L-1`` anchor, sequential batches,
    ``model.eval()`` and no gradients so train/validation metrics are directly comparable.
    """
    if not train_series or not val_series:
        raise ValueError("fit evaluation requires non-empty train and validation splits")
    model.to(device).eval()
    split_series = {"train": train_series, "val": val_series}
    splits: dict[str, dict[str, Any]] = {}
    for split, group in split_series.items():
        dataset = FixedAnchorTrajectoryWindows(group, config, normalizer)
        if len(dataset) != len(group):
            raise ValueError(
                f"fixed-anchor {split} replay covers {len(dataset)}/{len(group)} flights"
            )
        splits[split] = {
            "flights": len(group),
            "windows": len(dataset),
            "split_sha256": _split_sha256(group),
            "metrics": evaluate_split(model, dataset, normalizer, config, device),
        }

    train_metrics = splits["train"]["metrics"]
    val_metrics = splits["val"]["metrics"]
    diagnostics: dict[str, Any] = {
        "native_generalization": {
            "ade_m": _generalization_metric(train_metrics["ade_m"], val_metrics["ade_m"]),
            "fde_m": _generalization_metric(train_metrics["fde_m"], val_metrics["fde_m"]),
            "final_time_mae_s": _generalization_metric(
                train_metrics["final_time_s"]["mae"],
                val_metrics["final_time_s"]["mae"],
            ),
        }
    }
    objective = _training_objective_diagnostics(history, config)
    if objective is not None:
        diagnostics["training_objective"] = objective

    return {
        "schema_version": FIT_EVALUATION_SCHEMA,
        "evaluation_contract": {
            "model_mode": "eval",
            "dropout": "disabled",
            "anchor": "fixed L-1",
            "batch_order": "sequential (shuffle disabled)",
            "splits": ["train", "val"],
            "metric_grid": "native target grid; measured-data mask",
        },
        "config": config.to_dict(),
        "splits": splits,
        "diagnostics": diagnostics,
    }


def write_fit_evaluation(
    evaluation: dict[str, Any],
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Bind a deterministic fit replay to its exact checkpoint and write it atomically."""
    checkpoint = Path(checkpoint_path).resolve()
    document = dict(evaluation)
    document["checkpoint"] = {
        "path": str(checkpoint),
        "sha256": _file_sha256(checkpoint),
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / FIT_EVALUATION_NAME
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary.replace(output)
    return document


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


def _dataset_loss_components(
    model: nn.Module,
    dataset: TrajectoryWindows,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    names = loss_component_names(dataset.config)
    component_totals = {name: 0.0 for name in names}
    flight_weight_total = 0.0
    with torch.no_grad():
        for raw_batch in iter_batches(dataset, batch_size, shuffle=False, seed=0):
            x, y, mask, final_time_s, flight_weights, dynamics = unpack_batch(raw_batch)
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            final_time_s = final_time_s.to(device)
            flight_weights = flight_weights.to(device)
            dynamics = move_dynamics(dynamics, device)
            prediction = model_forward(model, x, dynamics)
            components = prediction_loss_components(
                prediction,
                x[:, -1],
                y,
                mask,
                final_time_s,
                flight_weights,
                dataset.config,
                dataset.normalizer,
                dynamics,
            )
            for name, value in components.tensors().items():
                component_totals[name] += float(value) * len(flight_weights)
            flight_weight_total += float(flight_weights.sum())
    denominator = max(flight_weight_total, 1.0)
    return {name: value / denominator for name, value in component_totals.items()}


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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.lr_plateau_factor,
        patience=config.lr_plateau_patience,
    )

    flights_per_epoch = sum(
        count > 0 for _start, count in train_set.series_ranges.values()
    )
    if verbose:
        print(
            f"  model      {config.model}/{config.prediction_output} "
            f"({parameter_count(model):,} params) on {device}"
        )
        output_grid = {
            HORIZON_NORMALIZED: f"N={config.n_segments} normalized progress segments",
            HORIZON_FULL: (
                f"H={config.full_horizon_steps} physical {config.dt_s:g}s steps, one pass"
            ),
            HORIZON_WINDOW: (
                f"H={config.window_horizon_steps} physical {config.dt_s:g}s steps per pass"
            ),
        }[config.horizon_mode]
        print(
            f"  prediction L={config.seq_len} ({config.lookback_s:.0f}s history) -> "
            f"{output_grid} + final_time_s"
        )
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
    optimizer_updates = 0
    component_names = loss_component_names(config)

    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])

        model.train()
        train_component_totals = {name: 0.0 for name in component_names}
        train_weight_total = 0.0
        for raw_batch in iter_batches(
            train_set, config.batch_size, shuffle=True, seed=config.seed + epoch
        ):
            x, y, mask, final_time_s, flight_weights, dynamics = unpack_batch(raw_batch)
            batch_count = len(flight_weights)
            batch_weight = float(flight_weights.sum())
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            final_time_s = final_time_s.to(device)
            flight_weights = flight_weights.to(device)
            dynamics = move_dynamics(dynamics, device)
            optimizer.zero_grad()
            components = prediction_loss_components(
                model_forward(model, x, dynamics),
                x[:, -1],
                y,
                mask,
                final_time_s,
                flight_weights,
                config,
                normalizer,
                dynamics,
            )
            loss = components.total
            loss.backward()
            optimizer.step()
            optimizer_updates += 1
            for name, value in components.tensors().items():
                train_component_totals[name] += float(value.detach()) * batch_count
            train_weight_total += batch_weight

        model.eval()
        val_components_by_airport = {
            airport: _dataset_loss_components(model, dataset, device, config.batch_size)
            for airport, dataset in val_sets.items()
        }
        train_components = {
            name: value / max(train_weight_total, 1.0)
            for name, value in train_component_totals.items()
        }
        train_loss = sum(train_components.values())
        val_by_airport = {
            airport: sum(components.values())
            for airport, components in val_components_by_airport.items()
        }
        val_components = {
            name: float(np.mean([
                components[name] for components in val_components_by_airport.values()
            ]))
            for name in component_names
        }
        # Equal airport weight: a large/long airport cannot control early stopping alone.
        val_loss = float(np.mean(list(val_by_airport.values())))
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            raise RuntimeError(
                f"training diverged at epoch {epoch} (train {train_loss}, val {val_loss}) "
                f"— no checkpoint written; lower --learning-rate"
            )
        scheduler.step(val_loss)
        history.append(EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=epoch_learning_rate,
            optimizer_updates=optimizer_updates,
            seconds=time.perf_counter() - started,
            val_by_airport=val_by_airport,
            train_components=train_components,
            val_components=val_components,
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
                  f"val-macro {val_loss:.6f}  lr {epoch_learning_rate:.2e}  "
                  f"updates {optimizer_updates:5d}  {history[-1].seconds:5.1f}s{marker}")
            print(
                "             val parts  "
                + "  ".join(
                    f"{name}={val_components[name]:.4f}" for name in component_names
                )
            )

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
    reserved_test_keys: Sequence[str] | None = None,
    data_selection: dict[str, Any] | None = None,
    auto_batch_size: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train one model on ``series``; write ``checkpoint.pt`` + ``history.json``.

    Returns the run summary (also embedded in the history file).
    """
    out = Path(output_dir)
    release_path = out / TEST_RELEASE_NAME
    if release_path.exists():
        raise RuntimeError(
            f"refusing to train in {out}: test release ledger {release_path} protects "
            "the checkpoint already published from this directory"
        )
    out.mkdir(parents=True, exist_ok=True)
    if data_provenance.get("schema_version") != ARRIVAL_DATA_PROVENANCE_SCHEMA:
        raise ValueError("data_provenance is not a TS arrival-data fingerprint")
    manifest_digests = provenance_manifest_digests(data_provenance)

    series = usable_series(series, config, verbose=verbose)
    train_series, val_series, test_series = split_by_flight(series, config)
    if reserved_test_keys is not None and test_series:
        raise ValueError(
            "development training received outer-test series even though sealed test "
            "identities were supplied separately"
        )
    checkpoint_test_keys = (
        list(reserved_test_keys)
        if reserved_test_keys is not None
        else [s.dataset_id for s in test_series]
    )
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
    test_window_count = (
        None
        if reserved_test_keys is not None
        else sum(min(len(window_anchors(item, config)), 1) for item in test_series)
    )
    fit_evaluation = evaluate_fit_splits(
        model,
        train_series,
        val_series,
        normalizer,
        config,
        device,
        history=fit.history,
    )
    split_metrics = {
        split: block["metrics"]
        for split, block in fit_evaluation["splits"].items()
    }

    checkpoint_payload = {
        "target_contract": target_contract(config),
        TEST_RELEASE_PROTOCOL_FIELD: TEST_RELEASE_SCHEMA,
        "config": config.to_dict(),
        "model_state": model.state_dict(),
        "normalizer": normalizer.to_dict(),
        "split": {
            "train": [s.dataset_id for s in train_series],
            "val": [s.dataset_id for s in val_series],
            "test": checkpoint_test_keys,
        },
        "best_val_loss": fit.best_val_loss,
        "data_provenance": data_provenance,
        "data_selection": data_selection,
    }
    checkpoint_path = out / CHECKPOINT_NAME
    checkpoint_tmp = out / f"{CHECKPOINT_NAME}.tmp"
    # Freeze-test may be run by another process while fitting. Recheck immediately before
    # the first checkpoint write so an already-bound checkpoint is never replaced.
    if release_path.exists():
        raise RuntimeError(
            f"refusing to replace {checkpoint_path}: test release ledger {release_path} "
            "was created while training"
        )
    torch.save(checkpoint_payload, checkpoint_tmp)
    checkpoint_tmp.replace(checkpoint_path)
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    fit_evaluation = write_fit_evaluation(
        fit_evaluation,
        checkpoint_path=checkpoint_path,
        output_dir=out,
    )
    checkpoint_metadata = {
        "schema_version": CHECKPOINT_METADATA_SCHEMA,
        TEST_RELEASE_PROTOCOL_FIELD: TEST_RELEASE_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "arrival_manifests": manifest_digests,
        "random_train_anchor": config.random_train_anchor,
        "horizon_mode": config.horizon_mode,
        "prediction_output": config.prediction_output,
        "aircraft_filter": config.aircraft_filter,
        "pred_len": config.pred_len,
        "full_horizon_steps": config.full_horizon_steps,
        "lr_scheduler": {
            "name": "ReduceLROnPlateau",
            "factor": config.lr_plateau_factor,
            "patience": config.lr_plateau_patience,
        },
        "split_sha256": {
            "train": _split_sha256(train_series),
            "val": _split_sha256(val_series),
            "test": _keys_sha256(checkpoint_test_keys),
        },
    }
    if data_selection is not None:
        selection_path = out / "data_selection.json"
        selection_tmp = out / "data_selection.json.tmp"
        selection_tmp.write_text(json.dumps(data_selection, indent=2), encoding="utf-8")
        selection_tmp.replace(selection_path)
        checkpoint_metadata["data_selection_sha256"] = _file_sha256(selection_path)
    if config.prediction_output == PREDICTION_CONTROL:
        checkpoint_metadata["control_recipe"] = {
            "effort_loss_weight": config.control_effort_loss_weight,
            "smoothness_loss_weight": config.control_smoothness_loss_weight,
            "rollout_integrator_dt_s": config.control_rollout_integrator_dt_s,
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
        "optimizer_updates": fit.history[-1].optimizer_updates,
        "final_learning_rate": fit.history[-1].learning_rate,
        "best_val_loss": fit.best_val_loss,
        "flights": {
            "train": len(train_series),
            "val": len(val_series),
            "test": len(checkpoint_test_keys),
        },
        "windows": {
            "train": fit.train_windows,
            "val": fit_evaluation["splits"]["val"]["windows"],
            "test": test_window_count,
        },
        "metrics": split_metrics,
        "fit_diagnostics": fit_evaluation["diagnostics"],
        "data_provenance": {
            "schema_version": data_provenance["schema_version"],
            "arrival_manifests": manifest_digests,
            "source_record_count": sum(
                len(entry["source_records"]) for entry in data_provenance["manifests"]
            ),
        },
        "data_selection": data_selection,
        "history": [vars(h) for h in fit.history],
    }
    (out / HISTORY_NAME).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        for split, block in split_metrics.items():
            print(f"  {split:5s}  ADE {block['ade_m']:7.1f} m   FDE {block['fde_m']:7.1f} m   "
                  f"cross-track p95 {block['cross_track_m']['p95_abs']:7.1f} m   "
                  f"alt p95 {block['altitude_m']['p95_abs']:6.1f} m   "
                  f"time MAE {block['final_time_s']['mae']:5.1f} s")
            raw = block["raw_kinematics"]
            print(
                "         raw kinematics  "
                f"pos/vel RMSE {raw['position_velocity_rmse_mps']:.2f} m/s   "
                f"heading p95 {raw['heading_consistency_p95_deg']:.2f} deg   "
                f"turn {raw['turn_rate_p95_deg_s']:.2f} deg/s   "
                f"accel {raw['acceleration_p95_mps2']:.2f} m/s²   "
                f"jerk {raw['jerk_p95_mps3']:.2f} m/s³"
            )
        print(
            f"✓ wrote {out / CHECKPOINT_NAME}, {out / HISTORY_NAME} and "
            f"{out / FIT_EVALUATION_NAME}"
        )

    return summary


def load_checkpoint(path: str | Path) -> tuple[nn.Module, TSConfig, Normalizer, dict[str, Any]]:
    """Rebuild a trained model from a checkpoint written by :func:`train`."""
    # weights_only=True: the payload is tensors + primitives only (config/normalizer are
    # plain dicts and lists), so nothing here needs — or should get — pickle execution.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    config = TSConfig.from_dict(payload["config"])
    expected_contract = target_contract(config)
    if payload.get("target_contract") != expected_contract:
        raise ValueError(
            f"checkpoint target contract {payload.get('target_contract')!r} does not match "
            f"the configured {config.prediction_output!r} output contract "
            f"{expected_contract!r}; retrain it"
        )
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
