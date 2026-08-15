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
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from channels import CHANNELS, IDX, POSITION_IDX
from batching import resolve_batch_size
from config import (
    CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
    CHECKPOINT_SELECTION_COMMON_GRID_ADE,
    CHECKPOINT_SELECTION_COMMON_GRID_CRITERIA,
    CHECKPOINT_SELECTION_OBJECTIVE,
    CHECKPOINT_SELECTION_TERMINAL_STATE,
    CONTROL_ARC_LOCAL_VELOCITY_TANGENT_SPEED,
    CONTROL_ARC_LOCAL_VELOCITY_VECTOR,
    CONTROL_ARC_TERMINAL_RUNWAY_COMPONENTS,
    CONTROL_ARC_TERMINAL_VECTOR_NORM,
    CONTROL_DYNAMICS_REANCHORED_RK4,
    CONTROL_DURATION_DIRECT,
    CONTROL_DURATION_FACTORIZED,
    CONTROL_VALUE_TRIM_RESIDUAL,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_CLOCK_PREDICTED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    CONTROL_STATE_OBJECTIVE_TERMINAL_STATE,
    CONTROL_TERMINAL_CLOCK_STATE_SUPERVISION,
    HORIZON_FULL,
    HORIZON_NORMALIZED,
    HORIZON_WINDOW,
    PREDICTION_CONTROL,
    PREDICTION_CONTROL_MIXTURE,
    PREDICTION_STATE,
    TSConfig,
    control_recipe,
    uses_control_dynamics,
)
from control_mixture import ControlMixturePrediction
from control_dynamics_backends import (
    DenseControlRolloutChannels,
    control_dynamics_backend,
)
from control_loss_components import (
    ControlStateLossResult,
    control_tracking_loss_terms,
)
from control_prediction_adapters import deployable_control_prediction
from control_regularization import control_regularization_signals
from control_training_curriculum import (
    ControlTrainingStage,
    build_control_training_stage_view,
    build_control_training_stages,
    control_training_stage_for_epoch,
)
from control_training_diagnostics import ControlTrainingDiagnosticsAccumulator
from control_terminal_clock import apply_control_terminal_clock
from dataset import (
    ARRIVAL_DATA_PROVENANCE_SCHEMA,
    FixedAnchorTrajectoryWindows,
    FlightSeries,
    Normalizer,
    RandomAnchorTrajectoryWindows,
    TrajectoryWindows,
    iter_batches,
    provenance_eligibility_digests,
    provenance_manifest_digests,
    split_by_flight,
    window_anchors,
)
from evaluation_protocol import (
    TEST_RELEASE_NAME,
    TEST_RELEASE_PROTOCOL_FIELD,
    TEST_RELEASE_SCHEMA,
)
from fixed_anchor_validation import fixed_anchor_common_grid_metrics
from fixed_dt_supervision import FixedDTControlSupervision
from metrics import error_by_progress, raw_kinematic_metrics, trajectory_metrics
from models import build_model, parameter_count, resolve_device
from prediction_outputs import ControlPrediction, StatePrediction
from physical_criteria import physical_criteria_loss
from time_grids import batch_time_grid, numpy_inference_time_grid
from training_performance import EpochProfiler

CHECKPOINT_NAME = "checkpoint.pt"
CHECKPOINT_METADATA_NAME = "checkpoint_metadata.json"
CHECKPOINT_METADATA_SCHEMA = "ts-checkpoint-metadata-v33-lateral-eligibility"
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
CONTROL_MIXTURE_LOSS_COMPONENT_NAMES = (
    *CONTROL_LOSS_COMPONENT_NAMES,
    "mixture_selector",
    "mixture_diversity",
)

CONTROL_TARGET_CONTRACTS = {
    (
        CONTROL_DURATION_FACTORIZED,
        CONTROL_STATE_CLOCK_PREDICTED,
        CONTROL_STATE_LOSS_GRID_NATIVE,
    ): (
        "bounded-control-nonuniform-duration-casadi-rollout-clock-aligned-v2"
    ),
    (
        CONTROL_DURATION_FACTORIZED,
        CONTROL_STATE_CLOCK_OBSERVED,
        CONTROL_STATE_LOSS_GRID_NATIVE,
    ): (
        "bounded-control-nonuniform-duration-casadi-rollout-observed-clock-aligned-v3"
    ),
    (
        CONTROL_DURATION_DIRECT,
        CONTROL_STATE_CLOCK_PREDICTED,
        CONTROL_STATE_LOSS_GRID_NATIVE,
    ): (
        "bounded-control-direct-duration-casadi-rollout-clock-aligned-v1"
    ),
    (
        CONTROL_DURATION_DIRECT,
        CONTROL_STATE_CLOCK_OBSERVED,
        CONTROL_STATE_LOSS_GRID_NATIVE,
    ): (
        "bounded-control-direct-duration-casadi-rollout-observed-clock-aligned-v1"
    ),
    (
        CONTROL_DURATION_FACTORIZED,
        CONTROL_STATE_CLOCK_OBSERVED,
        CONTROL_STATE_LOSS_GRID_FIXED_DT,
    ): "bounded-control-nonuniform-duration-fixed-dt-state-loss-v1",
    (
        CONTROL_DURATION_DIRECT,
        CONTROL_STATE_CLOCK_OBSERVED,
        CONTROL_STATE_LOSS_GRID_FIXED_DT,
    ): "bounded-control-direct-duration-fixed-dt-state-loss-v1",
}


def target_contract(config: TSConfig) -> str:
    if config.prediction_output == PREDICTION_STATE:
        return STATE_TARGET_CONTRACTS[config.horizon_mode]
    if config.prediction_output == PREDICTION_CONTROL_MIXTURE:
        base = (
            "bounded-control-mixture-best-of-k-selector-v1"
            f"+duration-uniform-floor={config.control_duration_uniform_floor:g}-v1"
        )
        if config.control_dynamics_backend != CONTROL_DYNAMICS_REANCHORED_RK4:
            base += f"+dynamics={config.control_dynamics_backend}-v1"
        return base
    base = CONTROL_TARGET_CONTRACTS[
        (
            config.control_duration_parameterization,
            config.control_state_supervision_clock,
            config.control_state_loss_grid,
        )
    ]
    base += (
        f"+duration-uniform-floor={config.control_duration_uniform_floor:g}-v1"
    )
    if config.control_value_parameterization == CONTROL_VALUE_TRIM_RESIDUAL:
        base += "+trim-residual-control-v1"
    if config.control_dynamics_backend != CONTROL_DYNAMICS_REANCHORED_RK4:
        base += f"+dynamics={config.control_dynamics_backend}-v1"
    if (
        config.control_state_objective == CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE
        and config.control_state_duration_gradient
    ):
        return base
    gradient_contract = (
        "joint-duration-gradient"
        if config.control_state_duration_gradient
        else "detached-duration-gradient"
    )
    contract = f"{base}+{config.control_state_objective}+{gradient_contract}"
    if config.control_horizon_curriculum_s:
        horizons = ",".join(f"{value:g}" for value in config.control_horizon_curriculum_s)
        contract += (
            f"+horizon-curriculum={horizons}s"
            f"x{config.control_horizon_curriculum_stage_epochs}epochs"
        )
    if (
        config.control_terminal_supervision_clock
        != CONTROL_TERMINAL_CLOCK_STATE_SUPERVISION
    ):
        contract += (
            f"+terminal-clock={config.control_terminal_supervision_clock}-v1"
        )
    return contract


def loss_component_names(config: TSConfig) -> tuple[str, ...]:
    names = {
        PREDICTION_STATE: STATE_LOSS_COMPONENT_NAMES,
        PREDICTION_CONTROL: CONTROL_LOSS_COMPONENT_NAMES,
        PREDICTION_CONTROL_MIXTURE: CONTROL_MIXTURE_LOSS_COMPONENT_NAMES,
    }[config.prediction_output]
    if config.prediction_output != PREDICTION_CONTROL:
        return names
    arc_velocity_extensions = {
        CONTROL_ARC_LOCAL_VELOCITY_VECTOR: (
            "arc_horizontal_velocity",
            "arc_vertical_velocity",
        ),
        CONTROL_ARC_LOCAL_VELOCITY_TANGENT_SPEED: (
            "arc_horizontal_tangent",
            "arc_horizontal_speed",
            "arc_vertical_velocity",
        ),
    }
    extensions = {
        CONTROL_STATE_OBJECTIVE_TERMINAL_STATE: ("terminal_velocity",),
        CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY: (
            "terminal_velocity",
            *arc_velocity_extensions[
                config.control_arc_local_velocity_parameterization
            ],
        ),
    }
    return (*names, *extensions.get(config.control_state_objective, ()))


def move_dynamics(
    dynamics: dict[str, torch.Tensor] | None, device: torch.device
) -> dict[str, torch.Tensor] | None:
    if dynamics is None:
        return None
    return {name: value.to(device) for name, value in dynamics.items()}


def move_fixed_dt_supervision(
    supervision: FixedDTControlSupervision | None,
    device: torch.device,
) -> FixedDTControlSupervision | None:
    return None if supervision is None else supervision.to(device)


def unpack_batch(batch: tuple) -> tuple:
    if len(batch) == 5:
        return (*batch, None, None)
    if len(batch) == 6:
        return (*batch, None)
    if len(batch) == 7:
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
    rollout = control_dynamics_backend(config).endpoint_rollout(
        dynamics["initial_state"].to(rollout_dtype),
        prediction.controls.to(rollout_dtype),
        prediction.segment_durations.to(rollout_dtype),
        dynamics["aero_params"].to(rollout_dtype),
        dynamics["frame_params"].to(rollout_dtype),
        config,
    )
    return rollout.channels, rollout.geodetic_states


def control_dense_rollout_channels(
    prediction: ControlPrediction,
    dynamics: dict[str, torch.Tensor],
    query_offsets_s: torch.Tensor,
    query_valid: torch.Tensor,
    config: TSConfig,
) -> DenseControlRolloutChannels:
    """Return actual RK4 states at requested physical times and segment boundaries.

    This is the deployable counterpart to :func:`control_rollout_channels`.  Callers that
    need a trajectory curve must use query states from this function; segment endpoints are
    intentionally too sparse to stand in for the integrated path.
    """
    rollout_dtype = torch.float64
    return control_dynamics_backend(config).dense_rollout(
        dynamics["initial_state"].to(rollout_dtype),
        prediction.controls.to(rollout_dtype),
        prediction.segment_durations.to(rollout_dtype),
        dynamics["aero_params"].to(rollout_dtype),
        dynamics["frame_params"].to(rollout_dtype),
        query_offsets_s.to(dtype=rollout_dtype, device=prediction.controls.device),
        query_valid.to(device=prediction.controls.device),
        config,
        segment_valid=None,
    )


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
    query_offsets_s = predicted_segment_durations_s.cumsum(dim=1)
    return align_control_targets_to_query_clock(
        normalized_anchor_state,
        target_states,
        state_weights,
        query_offsets_s,
        target_final_time_s,
    )


def align_control_targets_to_query_clock(
    normalized_anchor_state: torch.Tensor,
    target_states: torch.Tensor,
    state_weights: torch.Tensor,
    query_offsets_s: torch.Tensor,
    target_final_time_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate normalized truth onto arbitrary predicted physical timestamps."""
    if target_states.shape != state_weights.shape or target_states.ndim != 3:
        raise ValueError("control targets and weights must be aligned [B,N,C] tensors")
    batch, segments, channels = target_states.shape
    if normalized_anchor_state.shape != (batch, channels):
        raise ValueError("normalized control anchors must be [B,C]")
    if query_offsets_s.ndim != 2 or query_offsets_s.shape[0] != batch:
        raise ValueError("control target query offsets must be [B,M]")
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
        query_offsets_s.to(dtype=dtype, device=device)
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


@dataclass(frozen=True)
class ControlLossTerms:
    """Per-flight weighted control objectives before airport-macro reduction."""

    state: torch.Tensor
    final_time: torch.Tensor
    terminal: torch.Tensor
    effort: torch.Tensor
    smoothness: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def total(self) -> torch.Tensor:
        return (
            self.state
            + self.final_time
            + self.terminal
            + self.effort
            + self.smoothness
            + sum(self.extras.values(), self.state.new_zeros(()))
        )


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


def control_state_supervision_prediction(
    prediction: ControlPrediction,
    target_final_time_s: torch.Tensor,
    config: TSConfig,
) -> ControlPrediction:
    """Select the training clock without changing the deployable model output.

    The observed-clock candidate preserves the model's learned non-uniform duration
    fractions but scales them to the known train/validation total for state supervision.
    The original prediction remains available to the independent final-time loss and is
    still the only clock used at inference.
    """
    if config.control_state_supervision_clock != CONTROL_STATE_CLOCK_OBSERVED:
        return prediction
    fractions = prediction.segment_durations / prediction.segment_durations.sum(
        dim=1, keepdim=True
    )
    if not config.control_state_duration_gradient:
        fractions = fractions.detach()
    durations = fractions * target_final_time_s.unsqueeze(1)
    return ControlPrediction(
        controls=prediction.controls,
        segment_durations=durations,
        final_time_s=target_final_time_s,
    )


def _native_endpoint_control_state_loss(
    prediction: ControlPrediction,
    normalized_anchor_state: torch.Tensor,
    target_states: torch.Tensor,
    state_weights: torch.Tensor,
    target_final_time_s: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor],
    dense_supervision: FixedDTControlSupervision | None,
    segment_valid: torch.Tensor | None,
) -> ControlStateLossResult:
    """Historical loss on learned segment endpoints, isolated from dense supervision."""
    del dense_supervision
    if segment_valid is not None:
        raise ValueError("native endpoint state loss does not support horizon curriculum")
    physical_channels, _geodetic = control_rollout_channels(
        prediction, dynamics, config
    )
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
    return ControlStateLossResult(state_loss, normalized_states)


def _fixed_dt_control_state_loss(
    prediction: ControlPrediction,
    normalized_anchor_state: torch.Tensor,
    target_states: torch.Tensor,
    state_weights: torch.Tensor,
    target_final_time_s: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor],
    dense_supervision: FixedDTControlSupervision | None,
    segment_valid: torch.Tensor | None,
) -> ControlStateLossResult:
    """Dense regular-dt strategy; data preparation and rollout live in separate modules."""
    del normalized_anchor_state, target_states, state_weights, target_final_time_s
    if dense_supervision is None:
        raise ValueError("fixed-dt control state loss requires dense supervision targets")
    from fixed_dt_control_loss import fixed_dt_control_state_loss

    result = fixed_dt_control_state_loss(
        prediction,
        dense_supervision,
        config,
        normalizer,
        dynamics,
        segment_valid=segment_valid,
    )
    return ControlStateLossResult(
        result.per_flight_loss,
        result.normalized_segment_end_states,
        result.physical_query_states,
    )


_CONTROL_STATE_LOSS_HANDLERS = {
    CONTROL_STATE_LOSS_GRID_NATIVE: _native_endpoint_control_state_loss,
    CONTROL_STATE_LOSS_GRID_FIXED_DT: _fixed_dt_control_state_loss,
}


def control_prediction_loss_terms(
    prediction: ControlPrediction,
    normalized_anchor_state: torch.Tensor,
    target_states: torch.Tensor,
    state_weights: torch.Tensor,
    target_final_time_s: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor],
    dense_supervision: FixedDTControlSupervision | None = None,
    training_stage: ControlTrainingStage | None = None,
) -> ControlLossTerms:
    """Per-flight state/control terms through the differentiable dynamics rollout."""
    state_prediction = control_state_supervision_prediction(
        prediction, target_final_time_s, config
    )
    terminal_target = target_states[:, -1]
    segment_valid = None
    active_stage = training_stage or ControlTrainingStage("full", None, 1, None)
    if dense_supervision is not None:
        stage_view = build_control_training_stage_view(
            state_prediction,
            dense_supervision,
            terminal_target,
            target_final_time_s,
            active_stage,
        )
        state_prediction = stage_view.prediction
        dense_supervision = stage_view.supervision
        terminal_target = stage_view.terminal_target
        segment_valid = stage_view.segment_valid
    elif training_stage is not None:
        raise ValueError("control horizon curriculum requires dense supervision")
    rollout_loss = _CONTROL_STATE_LOSS_HANDLERS[
        config.control_state_loss_grid
    ](
        state_prediction,
        normalized_anchor_state,
        target_states,
        state_weights,
        target_final_time_s,
        config,
        normalizer,
        dynamics,
        dense_supervision,
        segment_valid,
    )
    rollout_loss = apply_control_terminal_clock(
        rollout_loss,
        prediction,
        dynamics,
        config,
        normalizer,
        active_stage,
    )
    time_loss = (
        (prediction.final_time_s - target_final_time_s) / config.final_time_scale_s
    ).square()
    runway_heading_rad = (
        dynamics["runway_heading_rad"]
        if config.control_state_objective
        == CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY
        else normalized_anchor_state.new_zeros(len(normalized_anchor_state))
    )
    tracking = control_tracking_loss_terms(
        rollout_loss,
        normalized_anchor_state,
        terminal_target,
        config,
        normalizer,
        dense_supervision,
        runway_heading_rad,
    )

    effort_signal, smoothness_signal = control_regularization_signals(
        prediction, dynamics, config
    )
    active_segments = (
        torch.ones_like(prediction.segment_durations, dtype=torch.bool)
        if segment_valid is None
        else segment_valid
    )
    active_float = active_segments.to(dtype=effort_signal.dtype)
    effort = (
        effort_signal.square() * active_float.unsqueeze(-1)
    ).sum(dim=(1, 2)) / (
        active_float.sum(dim=1) * effort_signal.shape[-1]
    ).clamp(min=1.0)
    if prediction.controls.shape[1] > 1:
        changes = torch.diff(smoothness_signal, dim=1)
        active_pairs = active_segments[:, 1:] & active_segments[:, :-1]
        pair_float = active_pairs.to(dtype=changes.dtype)
        smoothness = (
            changes.square() * pair_float.unsqueeze(-1)
        ).sum(dim=(1, 2)) / (
            pair_float.sum(dim=1) * changes.shape[-1]
        ).clamp(min=1.0)
    else:
        smoothness = effort.new_zeros(effort.shape)

    return ControlLossTerms(
        state=tracking.state,
        final_time=config.final_time_loss_weight * time_loss,
        terminal=tracking.terminal_position,
        effort=config.control_effort_loss_weight * effort,
        smoothness=config.control_smoothness_loss_weight * smoothness,
        extras=tracking.extras,
    )


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
    dense_supervision: FixedDTControlSupervision | None = None,
    training_stage: ControlTrainingStage | None = None,
) -> LossComponents:
    terms = control_prediction_loss_terms(
        prediction,
        normalized_anchor_state,
        target_states,
        state_weights,
        target_final_time_s,
        config,
        normalizer,
        dynamics,
        dense_supervision,
        training_stage,
    )

    def weighted_mean(values: torch.Tensor) -> torch.Tensor:
        return (values * flight_weights).mean()

    zero = weighted_mean(terms.state.new_zeros(terms.state.shape))
    return LossComponents(
        state=weighted_mean(terms.state),
        final_time=weighted_mean(terms.final_time),
        # State/velocity consistency is structural because both come from one dynamics rollout.
        kinematic=zero,
        terminal=weighted_mean(terms.terminal),
        extras={
            "control_effort": weighted_mean(terms.effort),
            "control_smoothness": weighted_mean(terms.smoothness),
            **{
                name: weighted_mean(value)
                for name, value in terms.extras.items()
            },
        },
    )


def state_prediction_loss_components(
    prediction: StatePrediction,
    normalized_anchor_state: torch.Tensor,
    target_states: torch.Tensor,
    state_weights: torch.Tensor,
    target_final_time_s: torch.Tensor,
    flight_weights: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor] | None = None,
    dense_supervision: FixedDTControlSupervision | None = None,
    training_stage: ControlTrainingStage | None = None,
) -> LossComponents:
    """Return the direct-state strategy's airport-macro loss contributions."""
    del dynamics, dense_supervision
    if training_stage is not None:
        raise ValueError("horizon curriculum is not supported by state prediction")
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


def _control_loss_adapter(
    prediction,
    normalized_anchor_state,
    target_states,
    state_weights,
    target_final_time_s,
    flight_weights,
    config,
    normalizer,
    dynamics,
    dense_supervision,
    training_stage,
) -> LossComponents:
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
        dense_supervision,
        training_stage,
    )


def _mixture_loss_adapter(
    prediction,
    normalized_anchor_state,
    target_states,
    state_weights,
    target_final_time_s,
    flight_weights,
    config,
    normalizer,
    dynamics,
    dense_supervision,
    training_stage,
) -> LossComponents:
    if dynamics is None:
        raise ValueError("control-mixture loss requires per-flight dynamics")
    if dense_supervision is not None:
        raise ValueError("control-mixture does not support fixed-dt state supervision")
    if training_stage is not None:
        raise ValueError("horizon curriculum is not supported by control-mixture")
    from control_mixture_loss import mixture_prediction_loss_components

    return mixture_prediction_loss_components(
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


PredictionLossHandler = Callable[..., LossComponents]
PREDICTION_LOSS_HANDLERS: dict[type, PredictionLossHandler] = {
    StatePrediction: state_prediction_loss_components,
    ControlPrediction: _control_loss_adapter,
    ControlMixturePrediction: _mixture_loss_adapter,
}


def prediction_loss_components(
    prediction: StatePrediction | ControlPrediction | ControlMixturePrediction,
    normalized_anchor_state: torch.Tensor,
    target_states: torch.Tensor,
    state_weights: torch.Tensor,
    target_final_time_s: torch.Tensor,
    flight_weights: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor] | None = None,
    dense_supervision: FixedDTControlSupervision | None = None,
    training_stage: ControlTrainingStage | None = None,
) -> LossComponents:
    """Dispatch the configured output contract to its isolated objective."""
    try:
        handler = PREDICTION_LOSS_HANDLERS[type(prediction)]
    except KeyError as error:
        raise TypeError(
            f"unsupported prediction type: {type(prediction).__name__}"
        ) from error
    return handler(
        prediction,
        normalized_anchor_state,
        target_states,
        state_weights,
        target_final_time_s,
        flight_weights,
        config,
        normalizer,
        dynamics,
        dense_supervision,
        training_stage,
    )


def prediction_loss(
    prediction: StatePrediction | ControlPrediction | ControlMixturePrediction,
    normalized_anchor_state: torch.Tensor,
    target_states: torch.Tensor,
    state_weights: torch.Tensor,
    target_final_time_s: torch.Tensor,
    flight_weights: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor] | None = None,
    dense_supervision: FixedDTControlSupervision | None = None,
    training_stage: ControlTrainingStage | None = None,
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
        dense_supervision,
        training_stage,
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
    validation_selection_metric: str = CHECKPOINT_SELECTION_OBJECTIVE
    validation_selection_value: float | None = None
    validation_selection_by_airport: dict[str, float] = field(default_factory=dict)
    train_anchor_sampling: dict[str, Any] = field(default_factory=dict)
    training_stage: dict[str, Any] = field(default_factory=dict)
    control_training_diagnostics: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    validation_profile_by_airport: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )


@dataclass
class FitResult:
    """In-memory result shared by final training and cross-validation folds."""

    model: nn.Module
    config: TSConfig
    normalizer: Normalizer
    device: torch.device
    history: list[EpochResult]
    best_val_loss: float
    best_validation_selection: float
    train_windows: int
    val_windows: int
    model_pretraining: dict[str, Any] | None = None


ModelPretrainer = Callable[
    [nn.Module, Sequence[FlightSeries], Normalizer, TSConfig, torch.device],
    dict[str, Any],
]


@dataclass(frozen=True)
class SplitPredictionReplay:
    """One immutable deployable prediction pass reused by every metric view."""

    predicted: np.ndarray
    truth: np.ndarray
    mask: np.ndarray
    predicted_time_s: np.ndarray
    truth_time_s: np.ndarray
    anchors: np.ndarray
    segment_durations_s: np.ndarray


def _prediction_batch_replay(
    output: StatePrediction | ControlPrediction | ControlMixturePrediction,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    final_time_s: torch.Tensor,
    dynamics: dict[str, torch.Tensor] | None,
    dataset: TrajectoryWindows,
) -> SplitPredictionReplay:
    """Materialize deployable physical arrays from an already-computed model output."""
    metric_targets = y
    metric_weights = mask
    if uses_control_dynamics(dataset.config.prediction_output):
        deployable = deployable_control_prediction(output)
        if dynamics is None:
            raise ValueError("control replay requires per-flight dynamics")
        points = dataset.config.validation_common_grid_points
        progress = torch.arange(
            1,
            points + 1,
            dtype=torch.float64,
            device=deployable.segment_durations.device,
        ) / points
        predicted_total_s = deployable.segment_durations.to(torch.float64).sum(dim=1)
        query_offsets_s = predicted_total_s.unsqueeze(1) * progress.unsqueeze(0)
        query_valid = torch.ones_like(query_offsets_s, dtype=torch.bool)
        rollout = control_dense_rollout_channels(
            deployable,
            dynamics,
            query_offsets_s,
            query_valid,
            dataset.config,
        )
        predicted_physical = (
            rollout.query_channels.detach().cpu().numpy().astype(np.float32)
        )
        metric_targets, metric_weights = align_control_targets_to_query_clock(
            x[:, -1],
            y,
            mask,
            query_offsets_s,
            final_time_s,
        )
        segment_durations_s = np.broadcast_to(
            (
                predicted_total_s.detach().cpu().numpy().astype(np.float64)
                / points
            )[:, None],
            (len(x), points),
        ).copy()
        predicted_time_s = deployable.final_time_s.detach().cpu().numpy()
    else:
        if not isinstance(output, StatePrediction):
            raise TypeError("state replay requires StatePrediction")
        out = output.states.detach().cpu().numpy()
        predicted_physical = dataset.normalizer.decode(
            out.astype(np.float64)
        ).astype(np.float32)
        segment_durations_s = numpy_inference_time_grid(
            output.final_time_s.detach().cpu().numpy(), dataset.config
        )[0]
        predicted_time_s = output.final_time_s.detach().cpu().numpy()

    # Decode in float64 (the normalizer stats' dtype), store float32: a pooled split is
    # tens of thousands of [N,C] windows and metre-scale metrics do not need float64 storage.
    truth = dataset.normalizer.decode(
        metric_targets.detach().cpu().numpy().astype(np.float64)
    ).astype(np.float32)
    anchors = dataset.normalizer.decode(
        x[:, -1].detach().cpu().numpy().astype(np.float64)
    ).astype(np.float32)
    raw_mask = metric_weights.detach().cpu().numpy()
    if raw_mask.ndim == 3:
        raw_mask = np.all(raw_mask > 0.0, axis=-1).astype(np.float32)
    return SplitPredictionReplay(
        predicted=predicted_physical,
        truth=truth,
        mask=raw_mask,
        predicted_time_s=predicted_time_s,
        truth_time_s=final_time_s.detach().cpu().numpy(),
        anchors=anchors,
        segment_durations_s=segment_durations_s,
    )


def _merge_prediction_replays(
    chunks: Sequence[tuple[np.ndarray, SplitPredictionReplay]],
    *,
    count: int,
) -> SplitPredictionReplay:
    """Restore dataset order after validation-only duration bucketing."""
    if not chunks:
        raise ValueError("prediction replay requires at least one batch")
    indices = np.concatenate([item[0] for item in chunks])
    order = np.argsort(indices, kind="stable")
    if not np.array_equal(indices[order], np.arange(count, dtype=np.int64)):
        raise ValueError("prediction replay indices must cover the dataset exactly once")

    def merged(name: str) -> np.ndarray:
        return np.concatenate(
            [getattr(item[1], name) for item in chunks], axis=0
        )[order]

    return SplitPredictionReplay(
        predicted=merged("predicted"),
        truth=merged("truth"),
        mask=merged("mask"),
        predicted_time_s=merged("predicted_time_s"),
        truth_time_s=merged("truth_time_s"),
        anchors=merged("anchors"),
        segment_durations_s=merged("segment_durations_s"),
    )


def _predict_split(
    model: nn.Module,
    dataset: TrajectoryWindows,
    normalizer: Normalizer,
    device: torch.device,
    batch_size: int,
) -> SplitPredictionReplay:
    """Return physical anchor/output arrays, masks, predicted time and true time."""
    model.eval()
    chunks: list[tuple[np.ndarray, SplitPredictionReplay]] = []
    cursor = 0
    with torch.no_grad():
        for raw_batch in iter_batches(dataset, batch_size, shuffle=False, seed=0):
            (
                x,
                y,
                mask,
                final_time_s,
                _flight_weights,
                dynamics,
                _dense_supervision,
            ) = unpack_batch(raw_batch)
            x_device = x.to(device)
            y_device = y.to(device)
            mask_device = mask.to(device)
            final_time_device = final_time_s.to(device)
            dynamics_device = move_dynamics(dynamics, device)
            output = model_forward(model, x_device, dynamics_device)
            batch_replay = _prediction_batch_replay(
                output,
                x_device,
                y_device,
                mask_device,
                final_time_device,
                dynamics_device,
                dataset,
            )
            batch_count = len(x)
            chunks.append(
                (np.arange(cursor, cursor + batch_count, dtype=np.int64), batch_replay)
            )
            cursor += batch_count
    return _merge_prediction_replays(
        chunks,
        count=len(dataset),
    )


def evaluate_split(
    model: nn.Module,
    dataset: TrajectoryWindows,
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    *,
    replay: SplitPredictionReplay | None = None,
) -> dict[str, Any]:
    """Physical-unit state and final-time metrics for a split."""
    replay = replay or _predict_split(
        model, dataset, normalizer, device, config.batch_size
    )
    block = trajectory_metrics(replay.predicted, replay.truth, replay.mask)
    block["by_progress"] = error_by_progress(
        replay.predicted, replay.truth, replay.mask
    )
    time_error = replay.predicted_time_s - replay.truth_time_s
    block["final_time_s"] = {
        "mae": float(np.abs(time_error).mean()),
        "rmse": float(np.sqrt(np.mean(time_error**2))),
        "mean_signed": float(time_error.mean()),
    }
    # Raw model nodes on their own predicted clock: no measured-track interpolation,
    # spline, filtering or CZML resampling. Durations are explicit [B,N] so this call site
    # remains valid when the output layer moves from uniform to nonuniform segments.
    active_segments = replay.segment_durations_s > 0.0
    block["raw_kinematics"] = raw_kinematic_metrics(
        replay.anchors,
        replay.predicted,
        replay.segment_durations_s,
        valid_segments=active_segments,
    )
    return block


def evaluate_fixed_anchor_common_grid(
    model: nn.Module,
    dataset: TrajectoryWindows,
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    *,
    replay: SplitPredictionReplay | None = None,
) -> dict[str, Any]:
    """Deployable fixed-anchor metrics on one shared physical-time grid."""
    replay = replay or _predict_split(
        model, dataset, normalizer, device, config.batch_size
    )
    block = fixed_anchor_common_grid_metrics(
        dataset.series,
        config,
        replay.anchors,
        replay.predicted,
        replay.predicted_time_s,
        replay.segment_durations_s,
        points=config.validation_common_grid_points,
        normalizer=normalizer,
    )
    return {
        key: value
        for key, value in block.items()
        if not isinstance(value, np.ndarray)
    }


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
    eligible_rows = [
        row
        for row in rows
        if not row.get("training_stage")
        or row["training_stage"].get("is_full_horizon", True)
    ]
    if not eligible_rows:
        raise ValueError("training history contains no full-horizon epoch")
    best = min(
        eligible_rows,
        key=lambda row: (
            row.get("validation_selection_value")
            if row.get("validation_selection_value") is not None
            else row["val_loss"]
        ),
    )
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
        "checkpoint_selection_metric": best.get(
            "validation_selection_metric", CHECKPOINT_SELECTION_OBJECTIVE
        ),
        "checkpoint_selection_value": float(
            best.get("validation_selection_value")
            if best.get("validation_selection_value") is not None
            else best["val_loss"]
        ),
    }
    last = rows[-1]
    if "learning_rate" in last:
        result["final_learning_rate"] = float(last["learning_rate"])
    if "optimizer_updates" in last:
        result["total_optimizer_updates"] = int(last["optimizer_updates"])
    return result


def _epoch_performance_summary(
    history: Sequence[EpochResult | dict[str, Any]],
    *,
    warmup_epochs: int = 3,
) -> dict[str, Any]:
    """Report median/p90 after a fixed warm-up without affecting model selection."""
    rows = [vars(row) if isinstance(row, EpochResult) else row for row in history]
    excluded = warmup_epochs if len(rows) > warmup_epochs else 0
    measured = rows[excluded:]
    timing_keys = sorted({
        key
        for row in measured
        for key, value in row.get("timing", {}).items()
        if isinstance(value, (int, float))
    })
    metrics = {}
    for key in timing_keys:
        values = np.asarray(
            [row["timing"][key] for row in measured if key in row.get("timing", {})],
            dtype=np.float64,
        )
        if len(values):
            metrics[key] = {
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
            }
    return {
        "warmup_epochs_excluded": excluded,
        "measured_epochs": len(measured),
        "timing": metrics,
    }


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
        splits[split] = evaluate_fixed_anchor_series(
            model, group, normalizer, config, device, split_name=split
        )

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


def evaluate_fixed_anchor_series(
    model: nn.Module,
    series: Sequence[FlightSeries],
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    *,
    split_name: str = "cohort",
) -> dict[str, Any]:
    """Evaluate one explicit flight cohort once on both native and common grids."""
    dataset = FixedAnchorTrajectoryWindows(series, config, normalizer)
    if len(dataset) != len(series):
        raise ValueError(
            f"fixed-anchor {split_name} replay covers {len(dataset)}/{len(series)} flights"
        )
    replay = _predict_split(model, dataset, normalizer, device, config.batch_size)
    return {
        "flights": len(series),
        "windows": len(dataset),
        "split_sha256": _split_sha256(series),
        "metrics": evaluate_split(
            model, dataset, normalizer, config, device, replay=replay
        ),
        "common_grid_metrics": evaluate_fixed_anchor_common_grid(
            model, dataset, normalizer, config, device, replay=replay
        ),
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


def filter_training_cohort(
    series: Sequence[FlightSeries],
    config: TSConfig,
    *,
    verbose: bool = True,
) -> tuple[list[FlightSeries], dict[str, Any]]:
    """Apply a predeclared future-duration floor to train flights only."""
    minimum = float(config.training_cohort_min_future_s)
    anchor = config.seq_len - 1
    retained: list[FlightSeries] = []
    excluded: list[dict[str, Any]] = []
    for item in series:
        remaining = float(item.supervision_times[-1] - item.times[anchor])
        if remaining >= minimum - 1e-9:
            retained.append(item)
        else:
            excluded.append({
                "dataset_id": item.dataset_id,
                "fixed_anchor_remaining_s": remaining,
            })
    audit = {
        "scope": "train only after by-flight split",
        "anchor": "fixed L-1",
        "minimum_future_s": minimum,
        "input_flights": len(series),
        "retained_flights": len(retained),
        "excluded_flights": len(excluded),
        "excluded": excluded,
    }
    if verbose and excluded:
        print(
            f"  cohort     retained {len(retained)}/{len(series)} train flights with "
            f">= {minimum:g}s after fixed L-1 anchor; excluded {len(excluded)}"
        )
    if not retained:
        raise ValueError("training cohort future-duration floor removed every train flight")
    return retained, audit


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


VALIDATION_DURATION_BUCKETS_S = (180.0, 360.0, 600.0)


@dataclass(frozen=True)
class ValidationBatch:
    indices: np.ndarray
    raw_batch: tuple
    bucket: str
    query_points: int


@dataclass(frozen=True)
class ValidationBatchPlan:
    """Cached, validation-only batches grouped by fixed-anchor remaining duration."""

    dataset: TrajectoryWindows
    batches: tuple[ValidationBatch, ...]
    bucket_flights: dict[str, int]

    @property
    def flights(self) -> int:
        return len(self.dataset)

    @property
    def query_points(self) -> int:
        return sum(batch.query_points for batch in self.batches)


def _duration_bucket_label(bucket: int) -> str:
    lower = 0.0 if bucket == 0 else VALIDATION_DURATION_BUCKETS_S[bucket - 1]
    if bucket < len(VALIDATION_DURATION_BUCKETS_S):
        upper = VALIDATION_DURATION_BUCKETS_S[bucket]
        return f"({lower:g},{upper:g}]s"
    return f"({lower:g},inf)s"


def build_validation_batch_plan(
    dataset: TrajectoryWindows,
    batch_size: int,
    *,
    duration_bucketed: bool = False,
) -> ValidationBatchPlan:
    """Build each fixed validation tensor once; optionally group no-grad rows by duration."""
    if not isinstance(dataset, FixedAnchorTrajectoryWindows):
        raise TypeError("validation batching requires a fixed-anchor dataset")
    durations = np.asarray([
        dataset.series[series_index].supervision_times[-1]
        - dataset.series[series_index].times[anchor]
        for series_index, anchor in dataset.index
    ], dtype=np.float64)
    bucket_ids = (
        np.searchsorted(
            np.asarray(VALIDATION_DURATION_BUCKETS_S, dtype=np.float64),
            durations,
            side="left",
        )
        if duration_bucketed
        else np.zeros(len(dataset), dtype=np.int64)
    )
    batches: list[ValidationBatch] = []
    bucket_flights: dict[str, int] = {}
    bucket_count = len(VALIDATION_DURATION_BUCKETS_S) + 1 if duration_bucketed else 1
    for bucket in range(bucket_count):
        indices = np.flatnonzero(bucket_ids == bucket).astype(np.int64, copy=False)
        if not len(indices):
            continue
        label = _duration_bucket_label(bucket) if duration_bucketed else "unbucketed"
        bucket_flights[label] = len(indices)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            raw_batch = dataset.batch(batch_indices)
            dense = unpack_batch(raw_batch)[-1]
            query_points = (
                int(dense.valid.sum())
                if dense is not None
                else len(batch_indices) * dataset.config.pred_len
            )
            batches.append(
                ValidationBatch(
                    indices=batch_indices,
                    raw_batch=raw_batch,
                    bucket=label,
                    query_points=query_points,
                )
            )
    covered = (
        np.concatenate([batch.indices for batch in batches])
        if batches else np.array([], dtype=np.int64)
    )
    if not np.array_equal(np.sort(covered), np.arange(len(dataset), dtype=np.int64)):
        raise ValueError("validation duration buckets must cover every row exactly once")
    return ValidationBatchPlan(
        dataset=dataset,
        batches=tuple(batches),
        bucket_flights=bucket_flights,
    )


@dataclass(frozen=True)
class ValidationAirportEvaluation:
    components: dict[str, float]
    replay: SplitPredictionReplay | None
    profile: dict[str, Any]


def _evaluate_validation_airport(
    model: nn.Module,
    plan: ValidationBatchPlan,
    device: torch.device,
    training_stage: ControlTrainingStage | None,
    *,
    include_deployable_replay: bool,
    profiler: EpochProfiler | None = None,
) -> ValidationAirportEvaluation:
    """Evaluate both validation clocks from one model forward per cached batch."""
    dataset = plan.dataset
    names = loss_component_names(dataset.config)
    component_totals = {name: 0.0 for name in names}
    flight_weight_total = 0.0
    replay_chunks: list[tuple[np.ndarray, SplitPredictionReplay]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in plan.batches:
            section = profiler.section if profiler is not None else None
            data_context = section("val_data_s") if section else nullcontext()
            with data_context:
                (
                    x,
                    y,
                    mask,
                    final_time_s,
                    flight_weights,
                    dynamics,
                    dense_supervision,
                ) = unpack_batch(batch.raw_batch)
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                final_time_s = final_time_s.to(device)
                flight_weights = flight_weights.to(device)
                dynamics = move_dynamics(dynamics, device)
                dense_supervision = move_fixed_dt_supervision(
                    dense_supervision, device
                )
            objective_context = (
                section("val_objective_s") if section else nullcontext()
            )
            with objective_context:
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
                    dense_supervision,
                    training_stage,
                )
            for name, value in components.tensors().items():
                component_totals[name] += float(value) * len(flight_weights)
            flight_weight_total += float(flight_weights.sum())
            if include_deployable_replay:
                selection_context = (
                    section("val_checkpoint_selection_s")
                    if section else nullcontext()
                )
                with selection_context:
                    replay_chunks.append((
                        batch.indices,
                        _prediction_batch_replay(
                            prediction,
                            x,
                            y,
                            mask,
                            final_time_s,
                            dynamics,
                            dataset,
                        ),
                    ))
    denominator = max(flight_weight_total, 1.0)
    replay = (
        _merge_prediction_replays(replay_chunks, count=len(dataset))
        if include_deployable_replay else None
    )
    profile = {
        "flights": plan.flights,
        "batches": len(plan.batches),
        "query_points": plan.query_points,
        "wall_s": time.perf_counter() - started,
        "duration_bucket_flights": dict(plan.bucket_flights),
    }
    return ValidationAirportEvaluation(
        components={
            name: value / denominator for name, value in component_totals.items()
        },
        replay=replay,
        profile=profile,
    )


def _dataset_loss_components(
    model: nn.Module,
    dataset: TrajectoryWindows,
    device: torch.device,
    batch_size: int,
    training_stage: ControlTrainingStage | None = None,
) -> dict[str, float]:
    names = loss_component_names(dataset.config)
    component_totals = {name: 0.0 for name in names}
    flight_weight_total = 0.0
    with torch.no_grad():
        for raw_batch in iter_batches(dataset, batch_size, shuffle=False, seed=0):
            (
                x,
                y,
                mask,
                final_time_s,
                flight_weights,
                dynamics,
                dense_supervision,
            ) = unpack_batch(raw_batch)
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            final_time_s = final_time_s.to(device)
            flight_weights = flight_weights.to(device)
            dynamics = move_dynamics(dynamics, device)
            dense_supervision = move_fixed_dt_supervision(dense_supervision, device)
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
                dense_supervision,
                training_stage,
            )
            for name, value in components.tensors().items():
                component_totals[name] += float(value) * len(flight_weights)
            flight_weight_total += float(flight_weights.sum())
    denominator = max(flight_weight_total, 1.0)
    return {name: value / denominator for name, value in component_totals.items()}


@dataclass(frozen=True)
class ValidationSelection:
    """One deterministic checkpoint-selection result over fixed-anchor validation."""

    metric: str
    value: float
    by_airport: dict[str, float]
    details_by_airport: dict[str, dict[str, Any]] = field(default_factory=dict)


def _objective_validation_selection(
    *,
    model: nn.Module,
    val_sets: dict[str, TrajectoryWindows],
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    val_by_airport: dict[str, float],
    precomputed_details_by_airport: dict[str, dict[str, Any]] | None = None,
) -> ValidationSelection:
    del model, val_sets, normalizer, config, device, precomputed_details_by_airport
    return ValidationSelection(
        metric=CHECKPOINT_SELECTION_OBJECTIVE,
        value=float(np.mean(list(val_by_airport.values()))),
        by_airport=dict(val_by_airport),
    )


def _common_grid_validation_details(
    *,
    model: nn.Module,
    val_sets: dict[str, TrajectoryWindows],
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    val_by_airport: dict[str, float],
    precomputed_details_by_airport: dict[str, dict[str, Any]] | None = None,
    replays_by_airport: dict[str, SplitPredictionReplay] | None = None,
) -> dict[str, dict[str, Any]]:
    del val_by_airport
    if precomputed_details_by_airport is not None:
        if set(precomputed_details_by_airport) != set(val_sets):
            raise ValueError("precomputed validation details do not match airport sets")
        return precomputed_details_by_airport
    if replays_by_airport is not None and set(replays_by_airport) != set(val_sets):
        raise ValueError("validation replays do not match airport sets")
    details: dict[str, dict[str, Any]] = {}
    for airport, dataset in val_sets.items():
        block = evaluate_fixed_anchor_common_grid(
            model,
            dataset,
            normalizer,
            config,
            device,
            replay=(
                replays_by_airport[airport]
                if replays_by_airport is not None else None
            ),
        )
        details[airport] = {
            "ade_m": block["ade_m"],
            "fde_m": block["fde_m"],
            "dense_state_loss": block["dense_state_loss"],
            "terminal_velocity_error_mps": block[
                "terminal_velocity_error_mps"
            ],
            "final_time_mae_s": block["final_time_mae_s"],
            "flights": block["flights"],
            "arc_length_geometry_loss": block["arc_length_geometry_loss"],
            "arc_length_geometry_unweighted_loss": block[
                "arc_length_geometry_unweighted_loss"
            ],
            "arc_length_distance_mean_m": block["arc_length_distance_mean_m"],
            "arc_length_path_length_ratio": block[
                "arc_length_path_length_ratio"
            ],
            "arc_length_path_length_log_error": block[
                "arc_length_path_length_log_error"
            ],
            "arc_length_horizontal_velocity_mae_mps": block[
                "arc_length_horizontal_velocity_mae_mps"
            ],
            "arc_length_horizontal_velocity_p95_mps": block[
                "arc_length_horizontal_velocity_p95_mps"
            ],
            "arc_length_horizontal_tangent_mean": block[
                "arc_length_horizontal_tangent_mean"
            ],
            "arc_length_horizontal_tangent_p95": block[
                "arc_length_horizontal_tangent_p95"
            ],
            "arc_length_horizontal_speed_mae_mps": block[
                "arc_length_horizontal_speed_mae_mps"
            ],
            "arc_length_horizontal_speed_p95_mps": block[
                "arc_length_horizontal_speed_p95_mps"
            ],
            "arc_length_vertical_velocity_mae_mps": block[
                "arc_length_vertical_velocity_mae_mps"
            ],
            "arc_length_vertical_velocity_p95_mps": block[
                "arc_length_vertical_velocity_p95_mps"
            ],
            "arc_length_horizontal_mean_m": block[
                "arc_length_horizontal_mean_m"
            ],
            "arc_length_horizontal_p95_m": block[
                "arc_length_horizontal_p95_m"
            ],
            "arc_length_vertical_mae_m": block["arc_length_vertical_mae_m"],
            "arc_length_vertical_p95_m": block["arc_length_vertical_p95_m"],
            "arc_length_terminal_position_m": block[
                "arc_length_terminal_position_m"
            ],
            "arc_length_terminal_velocity_error_mps": block[
                "arc_length_terminal_velocity_error_mps"
            ],
            "arc_length_terminal_position_runway_components_m": block[
                "arc_length_terminal_position_runway_components_m"
            ],
            "arc_length_terminal_velocity_runway_components_mps": block[
                "arc_length_terminal_velocity_runway_components_mps"
            ],
        }
    return details


def _common_grid_validation_selection(
    *,
    model: nn.Module,
    val_sets: dict[str, TrajectoryWindows],
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    val_by_airport: dict[str, float],
    precomputed_details_by_airport: dict[str, dict[str, Any]] | None = None,
) -> ValidationSelection:
    details = _common_grid_validation_details(
        model=model, val_sets=val_sets, normalizer=normalizer, config=config,
        device=device, val_by_airport=val_by_airport,
        precomputed_details_by_airport=precomputed_details_by_airport,
    )
    by_airport = {
        airport: float(block["ade_m"]) for airport, block in details.items()
    }
    return ValidationSelection(
        metric=CHECKPOINT_SELECTION_COMMON_GRID_ADE,
        value=float(np.mean(list(by_airport.values()))),
        by_airport=by_airport,
        details_by_airport=details,
    )


def _common_grid_criteria_validation_selection(
    *,
    model: nn.Module,
    val_sets: dict[str, TrajectoryWindows],
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    val_by_airport: dict[str, float],
    precomputed_details_by_airport: dict[str, dict[str, Any]] | None = None,
) -> ValidationSelection:
    details = _common_grid_validation_details(
        model=model, val_sets=val_sets, normalizer=normalizer, config=config,
        device=device, val_by_airport=val_by_airport,
        precomputed_details_by_airport=precomputed_details_by_airport,
    )
    by_airport: dict[str, float] = {}
    for airport, block in details.items():
        criterion = physical_criteria_loss(
            torch.tensor(float(block["ade_m"])),
            torch.tensor(float(block["fde_m"])),
        )
        value = float(criterion)
        by_airport[airport] = value
        block["physical_criteria"] = value
    return ValidationSelection(
        metric=CHECKPOINT_SELECTION_COMMON_GRID_CRITERIA,
        value=float(np.mean(list(by_airport.values()))),
        by_airport=by_airport,
        details_by_airport=details,
    )


def _terminal_state_validation_selection(
    *,
    model: nn.Module,
    val_sets: dict[str, TrajectoryWindows],
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    val_by_airport: dict[str, float],
    precomputed_details_by_airport: dict[str, dict[str, Any]] | None = None,
) -> ValidationSelection:
    details = _common_grid_validation_details(
        model=model,
        val_sets=val_sets,
        normalizer=normalizer,
        config=config,
        device=device,
        val_by_airport=val_by_airport,
        precomputed_details_by_airport=precomputed_details_by_airport,
    )
    by_airport: dict[str, float] = {}
    for airport, block in details.items():
        value = (
            config.control_dense_state_loss_weight * block["dense_state_loss"]
            + config.control_terminal_position_loss_weight
            * block["fde_m"]
            / config.control_terminal_position_scale_m
            + config.control_terminal_velocity_loss_weight
            * block["terminal_velocity_error_mps"]
            / config.control_terminal_velocity_scale_mps
        )
        by_airport[airport] = float(value)
        block["terminal_state_criterion"] = float(value)
    return ValidationSelection(
        metric=CHECKPOINT_SELECTION_TERMINAL_STATE,
        value=float(np.mean(list(by_airport.values()))),
        by_airport=by_airport,
        details_by_airport=details,
    )


def _arc_length_geometry_validation_selection(
    *,
    model: nn.Module,
    val_sets: dict[str, TrajectoryWindows],
    normalizer: Normalizer,
    config: TSConfig,
    device: torch.device,
    val_by_airport: dict[str, float],
    precomputed_details_by_airport: dict[str, dict[str, Any]] | None = None,
) -> ValidationSelection:
    details = _common_grid_validation_details(
        model=model,
        val_sets=val_sets,
        normalizer=normalizer,
        config=config,
        device=device,
        val_by_airport=val_by_airport,
        precomputed_details_by_airport=precomputed_details_by_airport,
    )
    by_airport: dict[str, float] = {}
    velocity_value = {
        CONTROL_ARC_LOCAL_VELOCITY_VECTOR: lambda block: (
            config.control_arc_horizontal_velocity_loss_weight
            * block["arc_length_horizontal_velocity_mae_mps"]
            / config.control_arc_horizontal_velocity_scale_mps
            + config.control_arc_vertical_velocity_loss_weight
            * block["arc_length_vertical_velocity_mae_mps"]
            / config.control_arc_vertical_velocity_scale_mps
        ),
        CONTROL_ARC_LOCAL_VELOCITY_TANGENT_SPEED: lambda block: (
            config.control_arc_tangent_loss_weight
            * block["arc_length_horizontal_tangent_mean"]
            + config.control_arc_horizontal_velocity_loss_weight
            * block["arc_length_horizontal_speed_mae_mps"]
            / config.control_arc_horizontal_velocity_scale_mps
            + config.control_arc_vertical_velocity_loss_weight
            * block["arc_length_vertical_velocity_mae_mps"]
            / config.control_arc_vertical_velocity_scale_mps
        ),
    }[config.control_arc_local_velocity_parameterization]
    terminal_position_key, terminal_velocity_key = {
        CONTROL_ARC_TERMINAL_VECTOR_NORM: (
            "arc_length_terminal_position_m",
            "arc_length_terminal_velocity_error_mps",
        ),
        CONTROL_ARC_TERMINAL_RUNWAY_COMPONENTS: (
            "arc_length_terminal_position_runway_components_m",
            "arc_length_terminal_velocity_runway_components_mps",
        ),
    }[config.control_arc_terminal_parameterization]
    for airport, block in details.items():
        value = (
            config.control_geometry_loss_weight
            * block["arc_length_geometry_loss"]
            + velocity_value(block)
            + config.control_terminal_position_loss_weight
            * block[terminal_position_key]
            / config.control_terminal_position_scale_m
            + config.control_terminal_velocity_loss_weight
            * block[terminal_velocity_key]
            / config.control_terminal_velocity_scale_mps
        )
        by_airport[airport] = float(value)
        block["arc_length_geometry_criterion"] = float(value)
    return ValidationSelection(
        metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
        value=float(np.mean(list(by_airport.values()))),
        by_airport=by_airport,
        details_by_airport=details,
    )


_VALIDATION_SELECTIONS: dict[str, Callable[..., ValidationSelection]] = {
    CHECKPOINT_SELECTION_OBJECTIVE: _objective_validation_selection,
    CHECKPOINT_SELECTION_COMMON_GRID_ADE: _common_grid_validation_selection,
    CHECKPOINT_SELECTION_COMMON_GRID_CRITERIA: _common_grid_criteria_validation_selection,
    CHECKPOINT_SELECTION_TERMINAL_STATE: _terminal_state_validation_selection,
    CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY: (
        _arc_length_geometry_validation_selection
    ),
}


def fit_model(
    train_series: Sequence[FlightSeries],
    val_series: Sequence[FlightSeries],
    config: TSConfig,
    *,
    auto_batch_size: bool = False,
    minimum_anchor_index: int | None = None,
    verbose: bool = True,
    model_pretrainer: ModelPretrainer | None = None,
) -> FitResult:
    """Fit one model against explicit train/validation flights, without touching test."""
    if not train_series or not val_series:
        raise ValueError("fit_model requires non-empty train and validation flights")
    cohort_floor = config.training_cohort_min_future_s
    cohort_anchor = config.seq_len - 1
    cohort_ineligible = [
        item for item in train_series
        if item.supervision_times[-1] - item.times[cohort_anchor] < cohort_floor - 1e-9
    ]
    if cohort_ineligible:
        raise ValueError(
            f"fit_model received {len(cohort_ineligible)} train flight(s) below the "
            f"predeclared {cohort_floor:g} s cohort floor; filter the train cohort before fit"
        )

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
    val_batch_plans = {
        airport: build_validation_batch_plan(dataset, config.batch_size)
        for airport, dataset in val_sets.items()
    }
    val_window_count = sum(len(dataset) for dataset in val_sets.values())
    if not len(train_set) or not val_window_count:
        raise ValueError(
            f"empty window set (train={len(train_set)}, val={val_window_count}) — "
            f"seq_len={config.seq_len}, minimum_anchor_index={minimum_anchor_index!r} "
            "leaves no future remainder in these tracks"
        )

    model = build_model(config).to(device)
    model_pretraining = (
        model_pretrainer(model, train_series, normalizer, config, device)
        if model_pretrainer is not None
        else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.lr_plateau_factor,
        patience=config.lr_plateau_patience,
    )
    curriculum_stages = build_control_training_stages(
        config.control_horizon_curriculum_s,
        epochs_per_stage=config.control_horizon_curriculum_stage_epochs,
        total_epochs=config.epochs,
    )

    flights_per_epoch = sum(
        count > 0 for _start, count in train_set.series_ranges.values()
    )
    if config.random_train_anchor and flights_per_epoch != len(train_series):
        raise ValueError(
            f"random-anchor {config.random_train_anchor_min_future_s:g} s future contract "
            f"covers {flights_per_epoch}/"
            f"{len(train_series)} train flights; adjust the train roster explicitly "
            "instead of silently changing experiment membership"
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
        if config.random_train_anchor:
            print(
                f"  anchor min {config.random_train_anchor_min_future_s:g}s future; "
                f"eligible {flights_per_epoch}/{len(train_series)} train flights"
            )
        print(
            f"  selection  {config.checkpoint_selection_metric} on fixed L-1 validation"
        )
        if minimum_anchor_index is not None:
            print(f"  anchor     common minimum index {minimum_anchor_index} "
                  f"({minimum_anchor_index * config.dt_s:.0f}s after track entry)")
        print(
            f"  sampling   one shuffled sample/flight; {flights_per_epoch} flight(s)/epoch; "
            "airport-macro loss weights"
        )
        if config.control_horizon_curriculum_s:
            schedule = " -> ".join(
                (
                    f"{stage.label} (epochs {stage.start_epoch}-{stage.end_epoch})"
                    if stage.end_epoch is not None
                    else f"{stage.label} (epochs {stage.start_epoch}+ )"
                )
                for stage in curriculum_stages
            )
            print(f"  curriculum {schedule}")
            print("             short stages select nothing; checkpoint selection starts at full")

    history: list[EpochResult] = []
    best_val = math.inf
    best_val_loss_at_selection = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    optimizer_updates = 0
    component_names = loss_component_names(config)
    full_stage_started = not bool(config.control_horizon_curriculum_s)
    previous_stage_label: str | None = None

    for epoch in range(1, config.epochs + 1):
        curriculum_stage = control_training_stage_for_epoch(curriculum_stages, epoch)
        training_stage = (
            curriculum_stage if config.control_horizon_curriculum_s else None
        )
        if curriculum_stage.is_full_horizon and not full_stage_started:
            # Prefix stages are initialization only. Their easier objective must not set
            # the LR schedule, early-stop counter, or deployable checkpoint baseline.
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config.lr_plateau_factor,
                patience=config.lr_plateau_patience,
            )
            best_val = math.inf
            best_val_loss_at_selection = math.inf
            best_state = None
            epochs_without_improvement = 0
            full_stage_started = True
        if verbose and curriculum_stage.label != previous_stage_label:
            print(
                f"  curriculum stage {curriculum_stage.label} starts at epoch {epoch}"
            )
        previous_stage_label = curriculum_stage.label
        profiler = EpochProfiler(device)
        epoch_start_optimizer_updates = optimizer_updates
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        train_anchor_sampling = train_set.anchor_statistics(config.seed + epoch)

        model.train()
        train_component_totals = {name: 0.0 for name in component_names}
        train_weight_total = 0.0
        control_diagnostics = (
            ControlTrainingDiagnosticsAccumulator(
                config.control_gradient_clip_norm,
                policy=config.control_gradient_clip_policy,
            )
            if config.control_gradient_clip_norm > 0.0
            else None
        )
        train_batches = iter(iter_batches(
            train_set, config.batch_size, shuffle=True, seed=config.seed + epoch
        ))
        while True:
            data_started = time.perf_counter()
            try:
                raw_batch = next(train_batches)
            except StopIteration:
                break
            (
                x,
                y,
                mask,
                final_time_s,
                flight_weights,
                dynamics,
                dense_supervision,
            ) = unpack_batch(raw_batch)
            batch_count = len(flight_weights)
            batch_weight = float(flight_weights.sum())
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            final_time_s = final_time_s.to(device)
            flight_weights = flight_weights.to(device)
            dynamics = move_dynamics(dynamics, device)
            dense_supervision = move_fixed_dt_supervision(dense_supervision, device)
            profiler.add_cpu_seconds(
                "train_data_s", time.perf_counter() - data_started
            )
            optimizer.zero_grad()
            with profiler.section("train_forward_s"):
                prediction = model_forward(model, x, dynamics)
            with profiler.section("train_rollout_loss_s"):
                if control_diagnostics is not None:
                    if not isinstance(prediction, ControlPrediction) or dynamics is None:
                        raise RuntimeError(
                            "control gradient diagnostics require deterministic control output"
                        )
                    control_diagnostics.record_prediction(prediction, dynamics)
                components = prediction_loss_components(
                    prediction,
                    x[:, -1],
                    y,
                    mask,
                    final_time_s,
                    flight_weights,
                    config,
                    normalizer,
                    dynamics,
                    dense_supervision,
                    training_stage,
                )
            loss = components.total
            with profiler.section("train_backward_step_s"):
                loss.backward()
                if control_diagnostics is not None:
                    control_diagnostics.record_gradients_and_clip(model)
                optimizer.step()
            optimizer_updates += 1
            for name, value in components.tensors().items():
                train_component_totals[name] += float(value.detach()) * batch_count
            train_weight_total += batch_weight

        model.eval()
        include_deployable_replay = curriculum_stage.is_full_horizon
        val_evaluations = {
            airport: _evaluate_validation_airport(
                model,
                plan,
                device,
                training_stage,
                include_deployable_replay=include_deployable_replay,
                profiler=profiler,
            )
            for airport, plan in val_batch_plans.items()
        }
        val_components_by_airport = {
            airport: evaluation.components
            for airport, evaluation in val_evaluations.items()
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
        control_training_diagnostics = (
            control_diagnostics.summary() if control_diagnostics is not None else {}
        )
        if curriculum_stage.is_full_horizon:
            replay_by_airport: dict[str, SplitPredictionReplay] = {}
            for airport, evaluation in val_evaluations.items():
                if evaluation.replay is None:
                    raise RuntimeError(
                        "full-horizon validation did not create deployable replay"
                    )
                replay_by_airport[airport] = evaluation.replay
            selection_metrics_started = time.perf_counter()
            common_grid_details = _common_grid_validation_details(
                model=model,
                val_sets=val_sets,
                normalizer=normalizer,
                config=config,
                device=device,
                val_by_airport=val_by_airport,
                replays_by_airport=replay_by_airport,
            )
            profiler.add_cpu_seconds(
                "val_checkpoint_selection_s",
                time.perf_counter() - selection_metrics_started,
            )
            validation_selection = _VALIDATION_SELECTIONS[
                config.checkpoint_selection_metric
            ](
                model=model,
                val_sets=val_sets,
                normalizer=normalizer,
                config=config,
                device=device,
                val_by_airport=val_by_airport,
                precomputed_details_by_airport=common_grid_details,
            )
        else:
            validation_selection = ValidationSelection(
                metric="curriculum-prefix-objective",
                value=val_loss,
                by_airport=dict(val_by_airport),
            )
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            raise RuntimeError(
                f"training diverged at epoch {epoch} (train {train_loss}, val {val_loss}) "
                f"— no checkpoint written; lower --learning-rate"
            )
        if not math.isfinite(validation_selection.value):
            raise RuntimeError(
                f"validation selection metric {validation_selection.metric} is not finite"
            )
        if curriculum_stage.is_full_horizon:
            scheduler.step(validation_selection.value)
        timing = profiler.finish(
            optimizer_updates=optimizer_updates - epoch_start_optimizer_updates
        )
        history.append(EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=epoch_learning_rate,
            optimizer_updates=optimizer_updates,
            seconds=timing["epoch_total_s"],
            val_by_airport=val_by_airport,
            train_components=train_components,
            val_components=val_components,
            validation_selection_metric=validation_selection.metric,
            validation_selection_value=validation_selection.value,
            validation_selection_by_airport=validation_selection.by_airport,
            train_anchor_sampling=train_anchor_sampling,
            training_stage=(
                {
                    "label": curriculum_stage.label,
                    "horizon_s": curriculum_stage.horizon_s,
                    "is_full_horizon": curriculum_stage.is_full_horizon,
                }
                if config.control_horizon_curriculum_s
                else {}
            ),
            control_training_diagnostics=control_training_diagnostics,
            timing=timing,
            validation_profile_by_airport={
                airport: evaluation.profile
                for airport, evaluation in val_evaluations.items()
            },
        ))

        if (
            curriculum_stage.is_full_horizon
            and validation_selection.value < best_val - 1e-9
        ):
            best_val = validation_selection.value
            best_val_loss_at_selection = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
            marker = " *"
        elif curriculum_stage.is_full_horizon:
            epochs_without_improvement += 1
            marker = ""
        else:
            marker = ""

        if verbose:
            stage_suffix = (
                f"  stage {curriculum_stage.label}"
                if config.control_horizon_curriculum_s
                else ""
            )
            print(f"  epoch {epoch:3d}/{config.epochs}  train {train_loss:.6f}  "
                  f"val-macro {val_loss:.6f}  lr {epoch_learning_rate:.2e}  "
                  f"updates {optimizer_updates:5d}  {history[-1].seconds:5.1f}s"
                  f"{stage_suffix}{marker}")
            if validation_selection.metric != CHECKPOINT_SELECTION_OBJECTIVE:
                selection_label = (
                    "checkpoint"
                    if curriculum_stage.is_full_horizon
                    else "prefix val"
                )
                print(
                    f"             {selection_label:10s}  "
                    f"{validation_selection.metric}="
                    f"{validation_selection.value:.1f}"
                )
            print(
                "             val parts  "
                + "  ".join(
                    f"{name}={val_components[name]:.4f}" for name in component_names
                )
            )
            if control_training_diagnostics:
                gradients = control_training_diagnostics["gradient_norm_pre_clip"]
                clip = control_training_diagnostics["clip"]
                saturation = control_training_diagnostics["control_saturation"]
                print(
                    "             gradients  "
                    f"total mean/max={gradients['mean']['total']:.2f}/"
                    f"{gradients['max']['total']:.2f}  "
                    f"backbone={gradients['max']['backbone']:.2f}  "
                    f"control={gradients['max']['control_head']:.2f}  "
                    f"time={gradients['max']['final_time_head']:.2f}"
                )
                print(
                    "             stability "
                    f"clip={clip['policy']} "
                    f"{clip['triggered_batches']}/{clip['batches']}  "
                    f"saturation={saturation['overall_rate']:.3%}"
                )

        if (
            curriculum_stage.is_full_horizon
            and epochs_without_improvement >= config.patience
        ):
            if verbose:
                print(f"  early stop: {config.patience} epochs without improvement")
            break

    if best_state is None:
        raise RuntimeError("training completed without a full-horizon checkpoint")
    model.load_state_dict(best_state)
    return FitResult(
        model=model,
        config=config,
        normalizer=normalizer,
        device=device,
        history=history,
        best_val_loss=best_val_loss_at_selection,
        best_validation_selection=best_val,
        train_windows=len(train_set),
        val_windows=val_window_count,
        model_pretraining=model_pretraining,
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
    model_pretrainer: ModelPretrainer | None = None,
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
    eligibility_digests = provenance_eligibility_digests(data_provenance)

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
    train_series, training_cohort = filter_training_cohort(
        train_series, config, verbose=verbose
    )
    fit = fit_model(
        train_series,
        val_series,
        config,
        auto_batch_size=auto_batch_size,
        verbose=verbose,
        model_pretrainer=model_pretrainer,
    )
    model, config, normalizer, device = (
        fit.model, fit.config, fit.normalizer, fit.device
    )
    test_window_count = (
        None
        if reserved_test_keys is not None
        else sum(min(len(window_anchors(item, config)), 1) for item in test_series)
    )
    anchor_audit = fit.history[0].train_anchor_sampling
    training_anchor_contract = {
        key: anchor_audit[key]
        for key in (
            "policy",
            "sampling_version",
            "minimum_future_s",
            "eligibility_policy",
            "temporal_candidate_anchors",
            "eligible_candidate_anchors",
            "excluded_candidate_anchors",
        )
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
        "validation_selection": {
            "metric": config.checkpoint_selection_metric,
            "best_value": fit.best_validation_selection,
            "anchor": "fixed L-1",
            "common_grid_points": config.validation_common_grid_points,
        },
        "training_anchor_contract": training_anchor_contract,
        "training_cohort": training_cohort,
        "data_provenance": data_provenance,
        "data_selection": data_selection,
    }
    if fit.model_pretraining is not None:
        checkpoint_payload["model_pretraining"] = fit.model_pretraining
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
    # The model artifact is complete before any derived metric/report replay starts. A
    # reporting failure can therefore be resumed with ``evaluate-fit`` without retraining.
    fit_evaluation = evaluate_fit_splits(
        model,
        train_series,
        val_series,
        normalizer,
        config,
        device,
        history=fit.history,
    )
    fit_evaluation = write_fit_evaluation(
        fit_evaluation,
        checkpoint_path=checkpoint_path,
        output_dir=out,
    )
    split_metrics = {
        split: block["metrics"]
        for split, block in fit_evaluation["splits"].items()
    }
    common_grid_metrics = {
        split: block["common_grid_metrics"]
        for split, block in fit_evaluation["splits"].items()
    }
    checkpoint_metadata = {
        "schema_version": CHECKPOINT_METADATA_SCHEMA,
        TEST_RELEASE_PROTOCOL_FIELD: TEST_RELEASE_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "arrival_manifests": manifest_digests,
        "random_train_anchor": config.random_train_anchor,
        "training_anchor_contract": training_anchor_contract,
        "training_cohort_min_future_s": config.training_cohort_min_future_s,
        "training_cohort_excluded_flights": training_cohort["excluded_flights"],
        "random_train_anchor_min_future_s": config.random_train_anchor_min_future_s,
        "checkpoint_selection_metric": config.checkpoint_selection_metric,
        "validation_common_grid_points": config.validation_common_grid_points,
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
    if eligibility_digests:
        checkpoint_metadata["eligibility_rosters"] = eligibility_digests
    if fit.model_pretraining is not None:
        checkpoint_metadata["model_pretraining"] = fit.model_pretraining
    if data_selection is not None:
        selection_path = out / "data_selection.json"
        selection_tmp = out / "data_selection.json.tmp"
        selection_tmp.write_text(json.dumps(data_selection, indent=2), encoding="utf-8")
        selection_tmp.replace(selection_path)
        checkpoint_metadata["data_selection_sha256"] = _file_sha256(selection_path)
    if uses_control_dynamics(config.prediction_output):
        checkpoint_metadata["control_recipe"] = control_recipe(config)
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
        "validation_selection": {
            "metric": config.checkpoint_selection_metric,
            "best_value": fit.best_validation_selection,
            "anchor": "fixed L-1",
            "common_grid_points": config.validation_common_grid_points,
        },
        "training_anchor_contract": training_anchor_contract,
        "training_cohort": training_cohort,
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
        "common_grid_metrics": common_grid_metrics,
        "fit_diagnostics": fit_evaluation["diagnostics"],
        "performance": _epoch_performance_summary(fit.history),
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
    if eligibility_digests:
        summary["data_provenance"]["eligibility_rosters"] = eligibility_digests
    if fit.model_pretraining is not None:
        summary["model_pretraining"] = fit.model_pretraining
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
