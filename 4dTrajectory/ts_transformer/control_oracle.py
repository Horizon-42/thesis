"""Direct-shooting control oracle for one known future trajectory.

The oracle deliberately contains no history encoder or prediction network.  Its trainable
parameters are the physical control schedule itself (and, optionally, its time partition).
This isolates control/dynamics representability from Transformer capacity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import torch
from torch import nn

from channels import POSITION_IDX
from config import TSConfig
from dataset import Normalizer
from fixed_dt_control_loss import (
    FixedDTStateLossResult,
    fixed_dt_control_state_loss,
)
from fixed_dt_supervision import FixedDTControlSupervision
from physical_criteria import (
    PHYSICAL_CRITERIA_DISTANCE_SCALE_M,
    fixed_dt_position_ade_m,
    physical_criteria_loss,
    smooth_maximum,
)
from prediction_outputs import ControlPrediction


ORACLE_DURATION_UNIFORM = "uniform"
ORACLE_DURATION_LEARNED = "learned"
ORACLE_DURATION_MODES = (ORACLE_DURATION_UNIFORM, ORACLE_DURATION_LEARNED)
ORACLE_OBJECTIVE_ALL_STATE = "all-state"
ORACLE_OBJECTIVE_POSITION_ONLY = "position-only"
ORACLE_OBJECTIVE_PHYSICAL_ADE = "physical-ade"
ORACLE_OBJECTIVE_PHYSICAL_CRITERIA = "physical-criteria"
ORACLE_OBJECTIVE_MODES = (
    ORACLE_OBJECTIVE_ALL_STATE,
    ORACLE_OBJECTIVE_POSITION_ONLY,
    ORACLE_OBJECTIVE_PHYSICAL_ADE,
    ORACLE_OBJECTIVE_PHYSICAL_CRITERIA,
)
ORACLE_PHYSICAL_DISTANCE_SCALE_M = 1_000.0
ORACLE_SUCCESS_DISTANCE_SCALE_M = PHYSICAL_CRITERIA_DISTANCE_SCALE_M


def _logit(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(min=1e-6, max=1.0 - 1e-6)
    return torch.log(value) - torch.log1p(-value)


class DirectControlOracle(nn.Module):
    """A bounded piecewise-constant control schedule, not a neural network."""

    def __init__(
        self,
        *,
        n_segments: int,
        control_lower: torch.Tensor,
        control_upper: torch.Tensor,
        total_duration_s: float | torch.Tensor,
        duration_mode: str,
        initial_controls: torch.Tensor | None = None,
        initial_segment_durations_s: torch.Tensor | None = None,
        control_noise_std: float = 0.0,
        duration_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        if n_segments < 1:
            raise ValueError("n_segments must be positive")
        if duration_mode not in ORACLE_DURATION_MODES:
            raise ValueError(
                f"unknown oracle duration mode {duration_mode!r}; "
                f"expected one of {ORACLE_DURATION_MODES}"
            )
        lower = torch.as_tensor(control_lower).reshape(-1)
        upper = torch.as_tensor(control_upper).reshape(-1)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("control bounds must each contain three values")
        if not torch.all(lower < upper):
            raise ValueError("every control lower bound must be below its upper bound")
        duration = torch.as_tensor(
            total_duration_s, dtype=lower.dtype, device=lower.device
        ).reshape(())
        if not torch.isfinite(duration) or duration <= 0.0:
            raise ValueError("total_duration_s must be positive and finite")
        if control_noise_std < 0.0 or duration_noise_std < 0.0:
            raise ValueError("initialization noise must be non-negative")

        self.n_segments = int(n_segments)
        self.duration_mode = duration_mode
        self.register_buffer("control_lower", lower)
        self.register_buffer("control_upper", upper)
        self.register_buffer("total_duration_s", duration)

        if initial_controls is None:
            # Match the deterministic control head's neutral initialization: 20% thrust,
            # zero bank, load factor 1.0.
            unit_controls = lower.new_tensor((0.2, 0.5, 1.0 / 3.0)).repeat(
                self.n_segments, 1
            )
        else:
            physical = torch.as_tensor(
                initial_controls, dtype=lower.dtype, device=lower.device
            )
            if physical.shape != (self.n_segments, 3):
                raise ValueError(
                    "initial_controls must be [N,3], got "
                    f"{tuple(physical.shape)} for N={self.n_segments}"
                )
            if not torch.all(torch.isfinite(physical)):
                raise ValueError("initial_controls must be finite")
            unit_controls = (physical - lower) / (upper - lower)
        # Noise is applied in unconstrained logit space after either initialization.
        control_logits = _logit(unit_controls)
        if control_noise_std:
            control_logits = control_logits + torch.randn_like(control_logits) * control_noise_std
        self.control_logits = nn.Parameter(control_logits)

        if duration_mode == ORACLE_DURATION_LEARNED:
            if initial_segment_durations_s is None:
                duration_logits = lower.new_zeros(self.n_segments)
            else:
                initial_durations = torch.as_tensor(
                    initial_segment_durations_s,
                    dtype=lower.dtype,
                    device=lower.device,
                )
                if initial_durations.shape != (self.n_segments,):
                    raise ValueError(
                        "initial_segment_durations_s must be [N], got "
                        f"{tuple(initial_durations.shape)} for N={self.n_segments}"
                    )
                if (
                    not torch.all(torch.isfinite(initial_durations))
                    or not torch.all(initial_durations > 0.0)
                    or not torch.isclose(
                        initial_durations.sum(), duration, rtol=1e-6, atol=1e-6
                    )
                ):
                    raise ValueError(
                        "initial segment durations must be positive, finite, and sum "
                        "to total_duration_s"
                    )
                duration_logits = torch.log(initial_durations / duration)
            if duration_noise_std:
                duration_logits = (
                    duration_logits
                    + torch.randn_like(duration_logits) * duration_noise_std
                )
            self.duration_logits = nn.Parameter(duration_logits)
        else:
            if initial_segment_durations_s is not None:
                raise ValueError(
                    "initial segment durations require duration_mode='learned'"
                )
            self.register_parameter("duration_logits", None)

    def forward(self) -> ControlPrediction:
        unit_controls = torch.sigmoid(self.control_logits)
        controls = self.control_lower + unit_controls * (
            self.control_upper - self.control_lower
        )
        if self.duration_logits is None:
            durations = self.total_duration_s.expand(self.n_segments) / self.n_segments
        else:
            durations = torch.softmax(self.duration_logits, dim=0) * self.total_duration_s
        return ControlPrediction(
            controls=controls.unsqueeze(0),
            segment_durations=durations.unsqueeze(0),
            final_time_s=self.total_duration_s.reshape(1),
        )


@dataclass(frozen=True)
class OracleObjective:
    """Scalar terms optimized by direct shooting."""

    total: torch.Tensor
    state: torch.Tensor
    terminal: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "state": float(self.state.detach().cpu()),
            "terminal": float(self.terminal.detach().cpu()),
        }


@dataclass(frozen=True)
class OracleEvaluation:
    objective: OracleObjective
    rollout: FixedDTStateLossResult
    prediction: ControlPrediction


def oracle_state_loss(
    rollout: FixedDTStateLossResult,
    supervision: FixedDTControlSupervision,
    normalizer: Normalizer,
    objective_mode: str,
) -> torch.Tensor:
    """Select the diagnostic tracking channels without changing the shared rollout."""
    if objective_mode == ORACLE_OBJECTIVE_ALL_STATE:
        return rollout.per_flight_loss.mean()
    if objective_mode not in (
        ORACLE_OBJECTIVE_POSITION_ONLY,
        ORACLE_OBJECTIVE_PHYSICAL_ADE,
        ORACLE_OBJECTIVE_PHYSICAL_CRITERIA,
    ):
        raise ValueError(
            f"unknown oracle objective mode {objective_mode!r}; "
            f"expected one of {ORACLE_OBJECTIVE_MODES}"
        )
    dtype = rollout.physical_query_states.dtype
    device = rollout.physical_query_states.device
    indices = list(POSITION_IDX)
    mean = torch.as_tensor(normalizer.mean[indices], dtype=dtype, device=device)
    scale = torch.as_tensor(normalizer.std[indices], dtype=dtype, device=device)
    predicted = (
        rollout.physical_query_states[..., indices] - mean
    ) / scale
    targets = supervision.states[..., indices].to(dtype=dtype, device=device)
    weights = supervision.weights[..., indices].to(dtype=dtype, device=device)
    weights = weights * supervision.valid.to(device=device).unsqueeze(-1)
    if objective_mode in (
        ORACLE_OBJECTIVE_PHYSICAL_ADE,
        ORACLE_OBJECTIVE_PHYSICAL_CRITERIA,
    ):
        return fixed_dt_position_ade_m(
            rollout.physical_query_states, supervision, normalizer
        ).mean() / ORACLE_PHYSICAL_DISTANCE_SCALE_M
    squared = (predicted - targets).square() * weights
    denominator = weights.sum(dim=(1, 2)).clamp(min=1.0)
    return (squared.sum(dim=(1, 2)) / denominator).mean()


def evaluate_control_prediction(
    prediction: ControlPrediction,
    *,
    supervision: FixedDTControlSupervision,
    terminal_target: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor],
    objective_mode: str = ORACLE_OBJECTIVE_ALL_STATE,
) -> OracleEvaluation:
    """Apply the exact fixed-dt control loss used by the Transformer experiment."""
    rollout = fixed_dt_control_state_loss(
        prediction, supervision, config, normalizer, dynamics
    )
    target = terminal_target.to(
        dtype=rollout.normalized_segment_end_states.dtype,
        device=rollout.normalized_segment_end_states.device,
    )
    if target.shape != (1, len(normalizer.mean)):
        raise ValueError(
            "terminal_target must be [1,C], got "
            f"{tuple(target.shape)}"
        )
    endpoint_position = rollout.normalized_segment_end_states[
        :, -1, list(POSITION_IDX)
    ]
    terminal_target_position = target[:, list(POSITION_IDX)]
    state = oracle_state_loss(rollout, supervision, normalizer, objective_mode)
    if objective_mode in (
        ORACLE_OBJECTIVE_PHYSICAL_ADE,
        ORACLE_OBJECTIVE_PHYSICAL_CRITERIA,
    ):
        position_scale = torch.as_tensor(
            normalizer.std[list(POSITION_IDX)],
            dtype=endpoint_position.dtype,
            device=endpoint_position.device,
        )
        terminal_distance = torch.linalg.vector_norm(
            (endpoint_position - terminal_target_position) * position_scale,
            dim=-1,
        ).mean()
        terminal_distance_scaled = (
            terminal_distance / ORACLE_PHYSICAL_DISTANCE_SCALE_M
        )
        if objective_mode == ORACLE_OBJECTIVE_PHYSICAL_CRITERIA:
            state = state * (
                ORACLE_PHYSICAL_DISTANCE_SCALE_M
                / ORACLE_SUCCESS_DISTANCE_SCALE_M
            )
            terminal = terminal_distance / ORACLE_SUCCESS_DISTANCE_SCALE_M
            total = physical_criteria_loss(
                state * ORACLE_SUCCESS_DISTANCE_SCALE_M,
                terminal_distance,
            )
            objective = OracleObjective(
                total=total,
                state=state,
                terminal=terminal,
            )
            return OracleEvaluation(objective, rollout, prediction)
        terminal = config.terminal_loss_weight * terminal_distance_scaled
    else:
        terminal = (
            config.terminal_loss_weight
            * (endpoint_position - terminal_target_position).square().mean()
        )
    objective = OracleObjective(total=state + terminal, state=state, terminal=terminal)
    return OracleEvaluation(objective, rollout, prediction)


def evaluate_control_oracle(
    oracle: DirectControlOracle,
    *,
    supervision: FixedDTControlSupervision,
    terminal_target: torch.Tensor,
    config: TSConfig,
    normalizer: Normalizer,
    dynamics: dict[str, torch.Tensor],
    objective_mode: str = ORACLE_OBJECTIVE_ALL_STATE,
) -> OracleEvaluation:
    return evaluate_control_prediction(
        oracle(),
        supervision=supervision,
        terminal_target=terminal_target,
        config=config,
        normalizer=normalizer,
        dynamics=dynamics,
        objective_mode=objective_mode,
    )


@dataclass(frozen=True)
class OracleHistoryRow:
    step: int
    total: float
    state: float
    terminal: float
    gradient_norm: float | None


@dataclass(frozen=True)
class OracleFitResult:
    best_step: int
    best_objective: dict[str, float]
    history: tuple[OracleHistoryRow, ...]
    updates_run: int


def fit_control_oracle(
    oracle: DirectControlOracle,
    objective_fn: Callable[[], OracleObjective],
    *,
    steps: int,
    control_learning_rate: float,
    duration_learning_rate: float,
    optimize_duration: bool = True,
    gradient_clip_norm: float | None = None,
    record_every: int = 10,
    progress: Callable[[OracleHistoryRow], None] | None = None,
) -> OracleFitResult:
    """Optimize one oracle with Adam and restore its best observed parameter state."""
    if steps < 1 or record_every < 1:
        raise ValueError("steps and record_every must be positive")
    if control_learning_rate <= 0.0 or duration_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive when provided")

    optimized_parameters = [oracle.control_logits]
    groups = [{"params": [oracle.control_logits], "lr": control_learning_rate}]
    if oracle.duration_logits is not None and optimize_duration:
        optimized_parameters.append(oracle.duration_logits)
        groups.append(
            {"params": [oracle.duration_logits], "lr": duration_learning_rate}
        )
    optimizer = torch.optim.Adam(groups)
    history: list[OracleHistoryRow] = []
    best_total = math.inf
    best_step = -1
    best_objective: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None

    def observe(step: int, value: OracleObjective, gradient_norm: float | None) -> None:
        nonlocal best_total, best_step, best_objective, best_state
        scalars = value.detached()
        if not math.isfinite(scalars["total"]):
            raise FloatingPointError(
                f"oracle objective became non-finite at step {step}: {scalars}"
            )
        if scalars["total"] < best_total:
            best_total = scalars["total"]
            best_step = step
            best_objective = scalars
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in oracle.state_dict().items()
            }
        if step == 0 or step == steps or step % record_every == 0:
            row = OracleHistoryRow(
                step=step,
                total=scalars["total"],
                state=scalars["state"],
                terminal=scalars["terminal"],
                gradient_norm=gradient_norm,
            )
            history.append(row)

    for step in range(steps):
        oracle.zero_grad(set_to_none=True)
        value = objective_fn()
        observe(step, value, None)
        value.total.backward()
        if gradient_clip_norm is None:
            squared_norm = sum(
                parameter.grad.detach().square().sum()
                for parameter in optimized_parameters
                if parameter.grad is not None
            )
            gradient_norm = float(torch.sqrt(squared_norm).cpu())
        else:
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    optimized_parameters, gradient_clip_norm
                ).detach().cpu()
            )
        # Update the most recent recorded row with its gradient norm without adding another
        # expensive rollout. Non-recorded steps still participate in best-state selection.
        if history and history[-1].step == step:
            history[-1] = OracleHistoryRow(
                step=history[-1].step,
                total=history[-1].total,
                state=history[-1].state,
                terminal=history[-1].terminal,
                gradient_norm=gradient_norm,
            )
            if progress is not None:
                progress(history[-1])
        optimizer.step()

    oracle.zero_grad(set_to_none=True)
    observe(steps, objective_fn(), None)
    if progress is not None:
        progress(history[-1])
    if best_state is None or best_objective is None:
        raise RuntimeError("oracle optimizer did not observe a finite candidate")
    oracle.load_state_dict(best_state)
    return OracleFitResult(
        best_step=best_step,
        best_objective=best_objective,
        history=tuple(history),
        updates_run=steps,
    )
