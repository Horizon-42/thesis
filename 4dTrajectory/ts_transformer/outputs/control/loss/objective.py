"""The control objective: the tracking terms on the rolled-out states (native segment
endpoints or the fixed-dt grid), the supervision terms (velocity, imitation, heading rate,
bank total variation) and the duration terms, assembled into `LossComponents`."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from aerodynamic_model.torch_dynamics import heading_rate_rad_s
from ts_transformer.data.batch_contract import LossComponents
from ts_transformer.data.channels import POSITION_IDX, VELOCITY_IDX
from ts_transformer.config import (
    CONTROL_DURATION_FACTORIZED,
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_CLOCK_PREDICTED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    DURATION_HEAD_QUANTILE,
    DURATION_HEAD_TWO_HEAD,
    TSConfig,
)
from ts_transformer.data.dataset import Normalizer
from ts_transformer.data.fixed_dt_supervision import FixedDTControlSupervision
from ts_transformer.training.objective import ProcedureMultipliers, procedure_loss
from ts_transformer.outputs.constraints import build_command_hook
from ts_transformer.outputs.dynamics import rollout as control_rollout
from ts_transformer.outputs.dynamics.backends import EndpointControlRollout
from ts_transformer.outputs.envelope import BANK_INDEX, control_contract
from ts_transformer.outputs.control.heads import ControlPrediction
from ts_transformer.outputs.control.loss.components import ControlStateLossResult, control_tracking_loss_terms
from ts_transformer.outputs.control.loss.fixed_dt import fixed_dt_control_state_loss
from ts_transformer.outputs.duration_heads import pinball_duration_loss


CONTROL_LOSS_COMPONENT_NAMES = ("state", "final_time", "kinematic", "terminal")
#: B1.b: the quantile head's own component, present under `duration_head='two-head'` alone.
#: Under `quantile` the pinball sum REPLACED the point term and kept the name `final_time`
#: (B1, and every stored history row keys on it); under `two-head` both terms exist, so
#: neither can be read out of the other and the pinball gets a name of its own.
DURATION_QUANTILE_COMPONENT = "duration_quantile"
# The closure output's regression groups wear the four fixed names (LossComponents
# always emits them): state = geometry, final_time = slowness in seconds, kinematic =
# height, terminal = 0.


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

    Both sides live in the run's control contract, and each channel is divided by half its
    box width (``ControlContract.half_width``) so a full-scale error costs the same in the
    longitudinal column, bank and load factor. The specific-force box's half width is the
    thrust-fraction box's times the fleet-median T_max/W, so for the median airframe one
    weight prices a given speed-rate error the same under both contracts.
    Under ``control_imitation_target="inverse-dynamics"`` the segments past the last
    measured velocity carry zero weight (see :func:`dataset.reference_control_supervision`);
    under ``"fitted"`` every segment carries weight one, because that schedule was fitted
    over the whole supervised horizon. This function reads whichever pair the dataset put
    in the batch and cannot tell them apart — which is the point of the axis.

    A flight whose weight vector is all zero (see
    :func:`dataset.reference_control_supervision`) scores exactly 0: the clamped
    denominator makes it a zero contribution and a zero gradient, and it DILUTES the
    per-flight mean this term reports — it is not a perfect imitation (review C-17).
    """
    if not config.control_imitation_loss_weight:
        return None
    target = dynamics["reference_controls"].to(prediction.controls.dtype)
    weight = dynamics["reference_control_weight"].to(prediction.controls.dtype)
    scale = torch.as_tensor(
        control_contract(config.control_thrust_parameterization).half_width,
        dtype=prediction.controls.dtype,
        device=prediction.controls.device,
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

    actual = cast(rollout.actual_controls)
    # The load the rollout FLEW is the contract law's resolution of the actuators at each
    # endpoint: the third actuator itself under every law but the path-angle one, whose loop
    # re-solves it from the state (`_ControlLaw.geodetic_load`) — so under that contract γ*
    # reaches the turn rate through the lift, as the load does under the others.
    load = control_contract(config.control_thrust_parameterization).law.geodetic_load(
        states, actual,
        max_thrust_n=cast(dynamics["max_thrust_n"]),
        initial_geodetic_states=cast(dynamics["initial_state"]),
    )
    predicted_dps = torch.rad2deg(
        heading_rate_rad_s(
            states,
            # The psi row reads bank and the (stall-limited) load factor only, never the
            # thrust, so the thrust column is passed as ZERO under every contract: exact, and
            # the resolved thrust would need each endpoint's drag for a value the row never
            # reads (`tests/test_specific_force.py::test_the_turn_rate_row_never_reads_the_thrust_column`
            # pins the independence).
            torch.stack((torch.zeros_like(load), actual[..., 1], load), dim=-1),
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
        control_contract(config.control_thrust_parameterization).half_width[BANK_INDEX]
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
    # B1: under `quantile` the sum of the five pinball losses REPLACES the point head's
    # squared residual — same component name, same units — so `loss_component_names` is
    # unchanged and every history row and readout keys on `final_time` exactly as before.
    # Its weight moved to `duration_quantile_loss_weight` when B1.b split the two terms;
    # both default to 1.0 and `final_time_loss_weight` is refused off its default under
    # `quantile`, so the number multiplying the pinball there is the one it always was.
    time_loss_weight = (
        config.duration_quantile_loss_weight
        if config.duration_head == DURATION_HEAD_QUANTILE
        else config.final_time_loss_weight
    )
    # Under a fixed horizon (`control_horizon_s`) both sides are Δ: the residual is
    # structurally zero and the config pins its weight at 0 — the component keeps its place
    # among the four fixed names as a stated zero, exactly as `kinematic` does on this path.
    time_loss = (
        pinball_duration_loss(
            prediction.duration_quantiles_s, target_final_time_s, config.final_time_scale_s
        )
        if config.duration_head == DURATION_HEAD_QUANTILE
        else ((prediction.final_time_s - target_final_time_s) / config.final_time_scale_s).square()
    )
    # B1.b: `two-head` has BOTH heads, so `final_time` above stays the POINT head's squared
    # residual at its own weight (the term the ROLLOUT's duration comes from) and the
    # pinball sum rides beside it in its own component — the two gains B1 delivered priced
    # independently, which is the whole construction.
    duration_quantile_extra = (
        {DURATION_QUANTILE_COMPONENT: config.duration_quantile_loss_weight
         * pinball_duration_loss(
             prediction.duration_quantiles_s, target_final_time_s, config.final_time_scale_s
         )}
        if config.duration_head == DURATION_HEAD_TWO_HEAD else {}
    )
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
        final_time=time_loss_weight * time_loss,
        terminal=tracking.terminal_position,
        extras={**tracking.extras, **procedure_extra, **duration_quantile_extra},
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
