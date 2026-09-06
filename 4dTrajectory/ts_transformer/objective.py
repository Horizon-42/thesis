"""The training objective: what a prediction is scored against, and how.

Every loss the package optimizes is built here — the state, control and closure
objectives, the procedure penalty, the supervision terms and the dispatch that picks one
from the config. It is deliberately NOT part of the training loop: `train.fit_model` is
one consumer, and the batch-size probe (`batching`), the capacity-ceiling runner and the
benchmark are others that want the objective without the loop around it. That is the
same reason `batch_contract` exists, one layer down.

Adding a term: register its name in :func:`loss_component_names` as well as putting it in
the objective's ``extras``, or the first batch raises ``KeyError`` — after the slow
dataset build.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from aerodynamic_model.torch_dynamics import heading_rate_rad_s
from batch_contract import LossComponents
from channels import IDX, POSITION_IDX, VELOCITY_IDX
from closure_output import ClosurePrediction, closure_loss_components
from config import (
    CONTROL_DURATION_FACTORIZED,
    CONTROL_DURATION_UNIFORM,
    CONTROL_DYNAMICS_REANCHORED_RK4,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_CLOCK_PREDICTED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    HORIZON_FULL,
    HORIZON_NORMALIZED,
    HORIZON_WINDOW,
    PREDICTION_CLOSURE,
    PREDICTION_STATE,
    TSConfig,
)
from control.constraints import build_command_hook
from control.dynamics import rollout as control_rollout
from control.dynamics.backends import EndpointControlRollout
from control.envelope import BANK_INDEX, CONTROL_HALF_WIDTH, physical_controls
from control.latent import LATENT_KL_COMPONENT, LatentControlPrediction, with_latent_kl
from control.loss.components import ControlStateLossResult, control_tracking_loss_terms
from dataset import Normalizer
from final_approach_geometry import corridor_violations, runway_axes, truth_final_gate
from fixed_dt_supervision import FixedDTControlSupervision
from prediction_outputs import ControlPrediction, StatePrediction
from time_grids import batch_time_grid


STATE_TARGET_CONTRACTS = {
    HORIZON_NORMALIZED: "normalized-output-true-time-physical-position-duration-v1",
    HORIZON_FULL: "full-horizon-physical-position-duration-v1",
    HORIZON_WINDOW: "recursive-window-physical-position-duration-v1",
}

STATE_LOSS_COMPONENT_NAMES = ("state", "final_time", "kinematic", "terminal", "procedure")
CONTROL_LOSS_COMPONENT_NAMES = ("state", "final_time", "kinematic", "terminal")
# The closure output's regression groups wear the four fixed names (LossComponents
# always emits them): state = geometry, final_time = slowness in seconds, kinematic =
# height, terminal = 0.
CLOSURE_LOSS_COMPONENT_NAMES = ("state", "final_time", "kinematic", "terminal")
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
        CONTROL_DURATION_FACTORIZED,
        CONTROL_STATE_CLOCK_OBSERVED,
        CONTROL_STATE_LOSS_GRID_FIXED_DT,
    ): "bounded-control-nonuniform-duration-fixed-dt-state-loss-v1",
    (
        CONTROL_DURATION_UNIFORM,
        CONTROL_STATE_CLOCK_PREDICTED,
        CONTROL_STATE_LOSS_GRID_NATIVE,
    ): "bounded-control-uniform-duration-casadi-rollout-clock-aligned-v1",
    (
        CONTROL_DURATION_UNIFORM,
        CONTROL_STATE_CLOCK_OBSERVED,
        CONTROL_STATE_LOSS_GRID_NATIVE,
    ): "bounded-control-uniform-duration-casadi-rollout-observed-clock-aligned-v1",
    (
        CONTROL_DURATION_UNIFORM,
        CONTROL_STATE_CLOCK_OBSERVED,
        CONTROL_STATE_LOSS_GRID_FIXED_DT,
    ): "bounded-control-uniform-duration-fixed-dt-state-loss-v1",
}


def target_contract(config: TSConfig) -> str:
    if config.prediction_output == PREDICTION_STATE:
        return STATE_TARGET_CONTRACTS[config.horizon_mode]
    if config.prediction_output == PREDICTION_CLOSURE:
        # The decision vector's shape IS the contract: a different knot count is a
        # different head, and a checkpoint of one must not load into the other.
        return (
            f"closure-v1-slowness{config.closure_slowness_knots}"
            f"-height{config.closure_height_knots}"
        )
    base = CONTROL_TARGET_CONTRACTS[
        (
            config.control_duration_parameterization,
            config.control_state_supervision_clock,
            config.control_state_loss_grid,
        )
    ]
    if config.control_duration_parameterization != CONTROL_DURATION_UNIFORM:
        base += (
            f"+duration-uniform-floor={config.control_duration_uniform_floor:g}-v1"
        )
    if config.control_dynamics_backend != CONTROL_DYNAMICS_REANCHORED_RK4:
        base += f"+dynamics={config.control_dynamics_backend}-v1"
    if config.control_duration_parameterization == CONTROL_DURATION_UNIFORM:
        contract = (
            base
            if config.control_state_objective == CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE
            else f"{base}+{config.control_state_objective}"
        )
    elif (
        config.control_state_objective == CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE
        and config.control_state_duration_gradient
    ):
        return base
    else:
        gradient_contract = (
            "joint-duration-gradient"
            if config.control_state_duration_gradient
            else "detached-duration-gradient"
        )
        contract = f"{base}+{config.control_state_objective}+{gradient_contract}"
    return contract


def loss_component_names(config: TSConfig) -> tuple[str, ...]:
    if config.prediction_output == PREDICTION_STATE:
        return STATE_LOSS_COMPONENT_NAMES
    if config.prediction_output == PREDICTION_CLOSURE:
        return CLOSURE_LOSS_COMPONENT_NAMES
    names = CONTROL_LOSS_COMPONENT_NAMES
    extensions = {
        CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION: (
            *(("velocity",) if config.control_velocity_loss_weight else ()),
            *(("imitation",) if config.control_imitation_loss_weight else ()),
            *(("heading_rate",) if config.control_heading_rate_loss_weight else ()),
            *(("bank_tv",) if config.control_bank_tv_loss_weight else ()),
        ),
    }
    return (
        *names,
        *extensions.get(config.control_state_objective, ()),
        *(("procedure",) if config.procedure_loss_active else ()),
        *((LATENT_KL_COMPONENT,) if config.latent_dim > 0 else ()),
    )


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


def masked_mse(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Weighted MSE over supervised channel values only.

    All three tensors use ``[B,N,C]``. Measured rows weight all channels equally, while
    fitted rows weight position only.
    """
    error = (predicted - target) ** 2 * mask
    denominator = mask.sum()
    return error.sum() / denominator.clamp(min=1.0)


@dataclass(frozen=True)
class ControlLossTerms:
    """Per-flight weighted control objectives before airport-macro reduction."""

    state: torch.Tensor
    final_time: torch.Tensor
    terminal: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
    # Batch-level counts that are not objectives (see LossComponents.diagnostics).
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def total(self) -> torch.Tensor:
        return (
            self.state
            + self.final_time
            + self.terminal
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


def control_imitation_mse(
    prediction: ControlPrediction,
    config: TSConfig,
    dynamics: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    """Per-flight MSE between the predicted schedule and the one the flown track implies.

    Both sides live in the dimensionless control box, and each channel is divided by half
    its box width so a full-scale error costs the same in thrust, bank and load factor.
    Under ``control_imitation_target="inverse-dynamics"`` the segments past the last
    measured velocity carry zero weight (see :func:`dataset.reference_control_supervision`);
    under ``"fitted"`` every segment carries weight one, because that schedule was fitted
    over the whole supervised horizon. This function reads whichever pair the dataset put
    in the batch and cannot tell them apart — which is the point of the axis.
    """
    if not config.control_imitation_loss_weight:
        return None
    target = dynamics["reference_controls"].to(prediction.controls.dtype)
    weight = dynamics["reference_control_weight"].to(prediction.controls.dtype)
    scale = torch.as_tensor(
        CONTROL_HALF_WIDTH, dtype=prediction.controls.dtype, device=prediction.controls.device
    )
    delta = (prediction.controls - target) / scale
    return (delta.square().mean(dim=-1) * weight).sum(dim=1) / weight.sum(dim=1).clamp(
        min=1.0
    )


def control_heading_rate_mse(
    rollout: EndpointControlRollout,
    config: TSConfig,
    dynamics: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    """Per-flight MSE between the ROLLOUT's own turn rate and the flown track's, deg/s.

    The predicted side is read out of the RHS the rollout integrates
    (:func:`aerodynamic_model.torch_dynamics.heading_rate_rad_s`) at each segment END,
    evaluated on the state the rollout reached there and on the controls the aircraft had
    ACTUALLY reached (the commands under the point-mass model, the actuator states after
    the lag). Nothing about the coordinated-turn identity is restated, so this term cannot
    ask for a turn the model would not fly — which is exactly what the imitation teacher's
    inverse-dynamics target does not guarantee.

    Both sides are divided by ``control_heading_rate_loss_scale_dps``, and endpoints past
    the last measured velocity carry zero weight (see
    :func:`dataset.reference_heading_rate_supervision`) — the same cut the velocity term
    makes, not the same numbers (that mask is an interpolated per-channel weight, this one
    a hard 0/1 step).

    **The two sides are paired BY INDEX, and that is only a like-for-like comparison
    because of one chain**: this term is built by the ``true-time-position`` objective, which
    ``TSConfig`` admits only with ``control_duration_parameterization="uniform"``, so the
    rollout's ``cumsum(segment_durations)`` is exactly the target's ``(k+1)·T/N``. Row k of
    each side is therefore the same physical instant. Admitting a non-uniform partition here
    would silently compare different times, exactly as it would for the imitation term.
    """
    if not config.control_heading_rate_loss_weight:
        return None
    states = rollout.geodetic_states
    dtype, device = states.dtype, states.device

    def cast(value: torch.Tensor) -> torch.Tensor:
        return value.to(dtype=dtype, device=device)

    predicted_dps = torch.rad2deg(
        heading_rate_rad_s(
            states,
            physical_controls(
                cast(rollout.actual_controls), cast(dynamics["max_thrust_n"])
            ),
            # Per-flight ``[B,6]`` against per-endpoint ``[B,N,7]`` states.
            cast(dynamics["aero_params"]).unsqueeze(-2),
        )
    )
    target = cast(dynamics["reference_heading_rate_dps"])
    weight = cast(dynamics["reference_heading_rate_weight"])
    delta = (predicted_dps - target) / config.control_heading_rate_loss_scale_dps
    return (delta.square() * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)


def control_bank_total_variation(
    controls: torch.Tensor, config: TSConfig
) -> torch.Tensor | None:
    """Per-flight mean |bank step| between adjacent COMMANDED segments, in half-box units.

    A structural constraint rather than a target: it prices the schedule's roughness
    without naming a value, so it can only remove the wiggle the teacherless arms grew, not
    put a shape in. The commanded schedule is the one the head owns — penalising the lagged
    actual bank would charge the actuator for the command it was given.

    **What it actually prices is REVERSALS, not slope**, and the gradient says so twice.
    ``|x|`` has subgradient ``sign(x)``, so on any run of segments banking monotonically
    the interior terms cancel (segment k gets ``+1`` from its left step and ``-1`` from its
    right) and only the run's two ends are charged: a smooth roll-in costs the same as a
    step of the same total size, while a wiggle that turns around pays at every turn. At
    EXACT flatness it is a stationary point — value 0 and gradient 0 together, which is
    where ``control.heads._initialize_control_head`` starts every run (a zeroed projection
    makes all N commands identical) — so this term alone never leaves the flat schedule;
    the position, velocity and heading-rate terms do, and only then does it begin to bind.
    If arm ③ reads "the TV term changed nothing", that is the live explanation to check
    first, before concluding the dose was too small.
    """
    if not config.control_bank_tv_loss_weight:
        return None
    bank = controls[..., BANK_INDEX]
    return (bank[:, 1:] - bank[:, :-1]).abs().mean(dim=1) / float(
        CONTROL_HALF_WIDTH[BANK_INDEX]
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
) -> ControlStateLossResult:
    """Historical loss on learned segment endpoints, isolated from dense supervision."""
    del dense_supervision
    command_hook = build_command_hook(config, dynamics)
    rollout = control_rollout.rollout_control_endpoints(
        prediction.controls,
        prediction.segment_durations,
        dynamics,
        config,
        command_hook=command_hook,
    )
    physical_channels = rollout.channels
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
    def physical_channel_mse(indices, physical_scale: float) -> torch.Tensor:
        """Weighted per-flight MSE over one channel group, in its own physical unit.

        The supervision weights carry the masking: fitted-tail velocity rows are already
        zero, so a velocity term never trains on the extrapolated placeholders.
        """
        columns = list(indices)
        weights = aligned_weights[..., columns].sum(dim=-1)
        delta = (
            normalized_states[..., columns] - aligned_targets[..., columns]
        ) * scale[columns]
        return (
            (delta.square().sum(dim=-1) * weights).sum(dim=1)
            / weights.sum(dim=1).clamp(min=1.0)
            / (physical_scale**2)
        )

    physical_position_mse = physical_channel_mse(
        POSITION_IDX, config.position_loss_scale_m
    )
    physical_velocity_mse = physical_channel_mse(
        VELOCITY_IDX, config.control_velocity_loss_scale_mps
    )
    return ControlStateLossResult(
        normalized_mse=state_loss,
        normalized_segment_end_states=normalized_states,
        physical_position_mse=physical_position_mse,
        physical_velocity_mse=physical_velocity_mse,
        control_imitation_mse=control_imitation_mse(prediction, config, dynamics),
        control_heading_rate_mse=control_heading_rate_mse(rollout, config, dynamics),
        control_bank_tv=control_bank_total_variation(prediction.controls, config),
        aligned_targets=aligned_targets,
        aligned_weights=aligned_weights,
        hook_diagnostics=command_hook.diagnostics() if command_hook is not None else {},
    )


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
) -> ControlStateLossResult:
    """Dense regular-dt strategy; data preparation and rollout live in separate modules."""
    del normalized_anchor_state, target_states, state_weights, target_final_time_s
    if dense_supervision is None:
        raise ValueError("fixed-dt control state loss requires dense supervision targets")
    from control.loss.fixed_dt import fixed_dt_control_state_loss

    result = fixed_dt_control_state_loss(
        prediction,
        dense_supervision,
        config,
        normalizer,
        dynamics,
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
    *,
    multipliers: "ProcedureMultipliers | None" = None,
) -> ControlLossTerms:
    """Per-flight state/control terms through the differentiable dynamics rollout."""
    state_prediction = control_state_supervision_prediction(
        prediction, target_final_time_s, config
    )
    terminal_target = target_states[:, -1]
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
    )
    # The final-approach penalty on the ROLLED-OUT states: the same hinge as the state
    # path, gated by the truth rows aligned to the segment endpoints, so the constraint
    # reaches the controls through the dynamics — a dynamically admissible path that is
    # pushed toward the corridor, never a clamped one.
    procedure_extra: dict[str, torch.Tensor] = {}
    procedure_diagnostics: dict[str, torch.Tensor] = {}
    if config.procedure_loss_active:
        # TSConfig admits the penalty on the native grid only, which fills these.
        procedure, procedure_diagnostics = procedure_loss(
            rollout_loss.normalized_segment_end_states,
            rollout_loss.aligned_targets,
            rollout_loss.aligned_weights[..., list(POSITION_IDX)].sum(dim=-1),
            config,
            normalizer,
            dynamics,
            multipliers,
        )
        procedure_extra = {"procedure": procedure}
    time_loss = (
        (prediction.final_time_s - target_final_time_s) / config.final_time_scale_s
    ).square()
    tracking = control_tracking_loss_terms(
        rollout_loss,
        normalized_anchor_state,
        terminal_target,
        config,
        normalizer,
        dense_supervision,
    )

    return ControlLossTerms(
        state=tracking.state,
        final_time=config.final_time_loss_weight * time_loss,
        terminal=tracking.terminal_position,
        extras={**tracking.extras, **procedure_extra},
        diagnostics={**rollout_loss.hook_diagnostics, **procedure_diagnostics},
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
    *,
    multipliers: "ProcedureMultipliers | None" = None,
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
        multipliers=multipliers,
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
            name: weighted_mean(value)
            for name, value in terms.extras.items()
        },
        diagnostics=terms.diagnostics,
    )


def _sample_uniform_progress_nodes(
    nodes: torch.Tensor,
    query_progress: torch.Tensor,
) -> torch.Tensor:
    """Differentiably sample ``[B,N+1,D]`` nodes defined at progress ``0..1``."""
    if nodes.ndim != 3 or query_progress.ndim != 2:
        raise ValueError("progress sampling requires [B,N+1,D] nodes and [B,Q] queries")
    if nodes.shape[0] != query_progress.shape[0] or nodes.shape[1] < 2:
        raise ValueError("progress sampling batch/segment shapes do not align")
    segments = nodes.shape[1] - 1
    scaled = query_progress.clamp(min=0.0, max=1.0) * segments
    left = torch.floor(scaled).to(torch.long).clamp(max=segments - 1)
    fraction = (scaled - left.to(scaled.dtype)).unsqueeze(-1)
    gather_index = left.unsqueeze(-1).expand(-1, -1, nodes.shape[-1])
    left_values = torch.gather(nodes, 1, gather_index)
    right_values = torch.gather(nodes, 1, gather_index + 1)
    return left_values + fraction * (right_values - left_values)


@dataclass
class ProcedureMultipliers:
    """The procedure penalty's weights λ, one per constraint family.

    Fixed at the configured weights when ``procedure_loss_dual_step`` is zero; otherwise
    the dual variables of ``min L_pred  s.t.  violation rate ≤ ε``, raised once per epoch
    by the measured excess (dual ascent on the RATE, with the hinge² as the primal
    surrogate), so the weight is found rather than swept.
    """

    lateral: float
    vertical: float

    @classmethod
    def from_config(cls, config: TSConfig) -> "ProcedureMultipliers | None":
        if not config.procedure_loss_active:
            return None
        return cls(
            lateral=config.procedure_loss_lateral_weight,
            vertical=config.procedure_loss_vertical_weight,
        )

    def update(self, lateral_rate: float, vertical_rate: float, config: TSConfig) -> None:
        step = config.procedure_loss_dual_step
        if step <= 0.0:
            return
        self.lateral = max(0.0, self.lateral + step * (lateral_rate - config.procedure_loss_epsilon))
        self.vertical = max(0.0, self.vertical + step * (vertical_rate - config.procedure_loss_epsilon))

    def to_dict(self) -> dict[str, float]:
        return {"lateral": self.lateral, "vertical": self.vertical}


PROCEDURE_DIAGNOSTICS = (
    "procedure_gated_rows", "procedure_lateral_violations", "procedure_vertical_violations",
)


def procedure_loss(
    predicted_states: torch.Tensor,
    target_states: torch.Tensor,
    point_weights: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor] | None,
    multipliers: ProcedureMultipliers | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The final-approach penalty, PER FLIGHT ``[B]``: metres outside the corridor /
    glidepath window, squared at the runway scale, on rows where the OBSERVED track is
    established (the truth gate), weighted by λ — and the counts the dual update needs.
    ``predicted_states`` are normalized rows aligned index-for-index with ``target_states``
    (the state output, or a control rollout's segment endpoints).

    Rows are paired by index: the same physical time under the ``full``/``window`` grids,
    the same progress fraction under ``normalized``. The gate is decided by the truth row
    (and a predicted row already past the threshold, ``d ≤ 0``, is not charged — it is
    truncated at inference), the violation measured on the predicted row, so a model that
    is early or late onto the final is charged where the flight actually was on it.
    """
    if not config.procedure_loss_active:
        return point_weights.new_zeros(point_weights.shape[0]), {}
    if dynamics is None:
        raise ValueError(
            "the procedure loss needs the per-flight final-approach context in the batch"
        )
    dtype, device = predicted_states.dtype, predicted_states.device
    mean = torch.as_tensor(normalizer.mean, dtype=dtype, device=device)
    std = torch.as_tensor(normalizer.std, dtype=dtype, device=device)
    predicted = predicted_states * std + mean
    truth = target_states.to(dtype) * std + mean
    psi = dynamics["runway_heading_rad"].to(dtype)
    tan_gpa = dynamics["glidepath_tan"].to(dtype)
    valid = point_weights > 0.0
    d_truth, xt_truth = runway_axes(truth[..., IDX["e"]], truth[..., IDX["n"]], psi)
    d_pred, xt_pred = runway_axes(predicted[..., IDX["e"]], predicted[..., IDX["n"]], psi)
    gate = truth_final_gate(d_truth, xt_truth, valid) & (d_pred > 0.0)
    lateral_m, vertical_m = corridor_violations(d_pred, xt_pred, predicted[..., IDX["u"]], tan_gpa)
    gate_weight = gate.to(dtype)
    gated_rows = gate_weight.sum(dim=1)
    lateral_sq = ((lateral_m / config.procedure_loss_lateral_scale_m) ** 2 * gate_weight).sum(dim=1)
    vertical_sq = ((vertical_m / config.procedure_loss_vertical_scale_m) ** 2 * gate_weight).sum(dim=1)
    per_flight_lateral = lateral_sq / gated_rows.clamp(min=1.0)
    per_flight_vertical = vertical_sq / gated_rows.clamp(min=1.0)
    weights = multipliers or ProcedureMultipliers.from_config(config)
    term = weights.lateral * per_flight_lateral + weights.vertical * per_flight_vertical
    diagnostics = {
        "procedure_gated_rows": gate.sum().detach(),
        "procedure_lateral_violations": ((lateral_m > 0.0) & gate).sum().detach(),
        "procedure_vertical_violations": ((vertical_m > 0.0) & gate).sum().detach(),
    }
    return term, diagnostics


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
    *,
    multipliers: ProcedureMultipliers | None = None,
) -> LossComponents:
    """Return the direct-state physical-position/time airport-macro objective."""
    del dense_supervision
    if prediction.states.shape != target_states.shape:
        raise ValueError("state prediction and target tensors must align")

    position_indices = list(POSITION_IDX)
    position_std = torch.as_tensor(
        normalizer.std[position_indices],
        dtype=prediction.states.dtype,
        device=prediction.states.device,
    )
    predicted_position = prediction.states[..., position_indices]
    target_position = target_states[..., position_indices]
    point_weights = state_weights[..., position_indices].sum(dim=-1)

    # The output endpoint is the last position carrying supervision, not necessarily the
    # final tensor row: fixed-horizon targets can contain a padded suffix. This is a second
    # task over the same physical target, not a runway-centre prior or a fitted trajectory.
    valid_position = point_weights > 0.0
    if not torch.all(valid_position.any(dim=1)):
        raise ValueError("every state target must contain a supervised position endpoint")
    if torch.any(valid_position[:, 1:] & ~valid_position[:, :-1]):
        raise ValueError("supervised state positions must form a contiguous prefix")
    last_index = valid_position.sum(dim=1) - 1
    gather_index = last_index[:, None, None].expand(-1, 1, len(position_indices))
    predicted_endpoint = torch.gather(predicted_position, 1, gather_index).squeeze(1)
    target_endpoint = torch.gather(target_position, 1, gather_index).squeeze(1)

    if config.horizon_mode == HORIZON_NORMALIZED:
        points = config.validation_common_grid_points
        progress = torch.arange(
            1,
            points + 1,
            dtype=prediction.states.dtype,
            device=prediction.states.device,
        ) / points
        truth_progress = progress.unsqueeze(0).expand(len(target_states), -1)
        prediction_progress = (
            truth_progress
            * target_final_time_s.unsqueeze(1)
            / prediction.final_time_s.unsqueeze(1).clamp(min=1e-6)
        ).clamp(min=0.0, max=1.0)
        anchor_position = normalized_anchor_state[:, None, position_indices]
        predicted_nodes = torch.cat((anchor_position, predicted_position), dim=1)
        target_nodes = torch.cat((anchor_position, target_position), dim=1)
        anchor_weight = point_weights[:, :1]
        weight_nodes = torch.cat((anchor_weight, point_weights), dim=1)
        predicted_position = _sample_uniform_progress_nodes(
            predicted_nodes, prediction_progress
        )
        target_position = _sample_uniform_progress_nodes(
            target_nodes, truth_progress
        )
        point_weights = _sample_uniform_progress_nodes(
            weight_nodes.unsqueeze(-1), truth_progress
        ).squeeze(-1)

    physical_delta = (predicted_position - target_position) * position_std
    squared_distance = physical_delta.square().sum(dim=-1)
    state_loss = (
        (squared_distance * point_weights).sum(dim=1)
        / point_weights.sum(dim=1).clamp(min=1e-12)
        / (config.position_loss_scale_m**2)
    )
    endpoint_delta = (predicted_endpoint - target_endpoint) * position_std
    endpoint_loss = (
        endpoint_delta.square().sum(dim=-1)
        / (config.position_loss_scale_m**2)
    )
    time_loss = (
        (prediction.final_time_s - target_final_time_s) / config.final_time_scale_s
    ).square()
    zero = state_loss.new_zeros(state_loss.shape)
    # On the row grid (not the resampled progress nodes): the gate is a per-row decision.
    procedure, procedure_diagnostics = procedure_loss(
        prediction.states,
        target_states,
        state_weights[..., position_indices].sum(dim=-1),
        config,
        normalizer,
        dynamics,
        multipliers,
    )

    def weighted_mean(values: torch.Tensor) -> torch.Tensor:
        return (values * flight_weights).mean()

    return LossComponents(
        state=weighted_mean(state_loss),
        final_time=config.final_time_loss_weight * weighted_mean(time_loss),
        kinematic=weighted_mean(zero),
        terminal=config.state_endpoint_loss_weight * weighted_mean(endpoint_loss),
        extras={"procedure": weighted_mean(procedure)},
        diagnostics=procedure_diagnostics,
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
    *,
    multipliers: ProcedureMultipliers | None = None,
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
        multipliers=multipliers,
    )


def _latent_control_loss_adapter(
    prediction: LatentControlPrediction,
    normalized_anchor_state,
    target_states,
    state_weights,
    target_final_time_s,
    flight_weights,
    config,
    normalizer,
    dynamics,
    dense_supervision,
    *,
    multipliers: ProcedureMultipliers | None = None,
) -> LossComponents:
    """The control objective on the decoded schedule, plus the latent's KL term."""
    components = _control_loss_adapter(
        prediction, normalized_anchor_state, target_states, state_weights,
        target_final_time_s, flight_weights, config, normalizer, dynamics,
        dense_supervision, multipliers=multipliers,
    )
    return with_latent_kl(components, prediction, config, flight_weights)


PredictionLossHandler = Callable[..., LossComponents]
PREDICTION_LOSS_HANDLERS: dict[type, PredictionLossHandler] = {
    StatePrediction: state_prediction_loss_components,
    ControlPrediction: _control_loss_adapter,
    LatentControlPrediction: _latent_control_loss_adapter,
    ClosurePrediction: closure_loss_components,
}


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
    dense_supervision: FixedDTControlSupervision | None = None,
    *,
    multipliers: ProcedureMultipliers | None = None,
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
        multipliers=multipliers,
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
    dense_supervision: FixedDTControlSupervision | None = None,
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
    ).total
