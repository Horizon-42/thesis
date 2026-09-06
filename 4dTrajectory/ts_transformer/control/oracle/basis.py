"""Batched direct-shooting fit of a piecewise-constant control BASIS to known futures.

``control/oracle/shooting.py`` fits one flight with a curriculum and restarts; this module
fits a whole cohort at once at a chosen width, which is what a width study needs: the
flights carry independent parameters, so one batch is one rollout and the answer per
flight is its own best step, never the batch mean's.

Two axes, both from the collocation optimiser's parameterisation (``DEFAULT_N_SEGMENTS``
= 8 control segments for a whole unconstrained arrival, ``DEFAULT_N_SEG_PER_PHASE`` = 3
per procedure leg, each phase with its own free duration):

* the number of segments N — the decoder's output width;
* how the GIVEN total duration is partitioned over them, ``uniform`` or ``free``.

The total duration is never a parameter here. A width study measures how well N segments
can reproduce an observed PATH; predicting the duration is a separate question with its
own measurements.

Because a width study reads "the error that remains" as a property of the WIDTH, the fit
has to be able to prove it converged: :class:`BasisFitResult` carries the seed's own
objective (step 0, before any update), each flight's best step, and the share of flights
still improving when the budget ran out. Without those three an under-converged optimiser
reads as "N segments are not enough", which is the one wrong answer this study can give.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from channels import states_from_channels
from config import TSConfig
from control.dynamics.inverse import segment_controls
from prediction_outputs import ControlPrediction

DURATION_UNIFORM = "uniform"
DURATION_FREE = "free"
DURATION_MODES = (DURATION_UNIFORM, DURATION_FREE)

# The fit's parameters are logits of the box-scaled controls, so an initial control ON a
# bound has no finite logit; this is the margin the seed is held inside the box by.
UNIT_LOGIT_MARGIN = 1e-6
# A flight whose best step is this far into the budget was still improving when it ran out.
STILL_IMPROVING_FRACTION = 0.95
# The anchor cross-check catches a mis-ALIGNED batch (kilometres, whole flights), not float
# noise, so its tolerances are stated per component in the units each one is measured in:
# degrees, metres, m/s, radians. 1e-5 deg is about a metre.
_ANCHOR_TOLERANCE = np.array([1e-5, 1e-5, 1.0, 0.1, 1e-4, 1e-4, 1.0])


def free_number_count(n_segments: int, duration_mode: str) -> int:
    """How many free numbers one flight's schedule carries at this width and mode.

    The total duration is given, so this is 3N (uniform) or 4N (free) — one less than the
    deployed control head's count at the same width, which also predicts the total.
    """
    if duration_mode not in DURATION_MODES:
        raise ValueError(f"unknown duration mode {duration_mode!r}; expected {DURATION_MODES}")
    return n_segments * 3 + (n_segments if duration_mode == DURATION_FREE else 0)


class BasisSchedule(nn.Module):
    """One bounded piecewise-constant schedule per flight, with the total duration given."""

    def __init__(
        self,
        initial_controls: torch.Tensor,      # [B, N, 3], physical, inside the box
        control_lower: torch.Tensor,         # [B, 3]
        control_upper: torch.Tensor,         # [B, 3]
        final_time_s: torch.Tensor,          # [B]
        duration_mode: str = DURATION_UNIFORM,
    ) -> None:
        super().__init__()
        if duration_mode not in DURATION_MODES:
            raise ValueError(f"unknown duration mode {duration_mode!r}; expected {DURATION_MODES}")
        if initial_controls.ndim != 3 or initial_controls.shape[-1] != 3:
            raise ValueError(f"initial_controls must be [B, N, 3], got {tuple(initial_controls.shape)}")
        batch, n_segments, _ = initial_controls.shape
        if control_lower.shape != (batch, 3) or control_upper.shape != (batch, 3):
            raise ValueError("control bounds must each be [B, 3] and match initial_controls")
        if final_time_s.shape != (batch,):
            raise ValueError(f"final_time_s must be [B], got {tuple(final_time_s.shape)}")
        if not torch.all(control_lower < control_upper):
            raise ValueError("every control lower bound must be below its upper bound")
        if not torch.all(torch.isfinite(final_time_s)) or not torch.all(final_time_s > 0.0):
            raise ValueError("every final_time_s must be positive and finite")
        if not torch.all(torch.isfinite(initial_controls)):
            raise ValueError("initial_controls must be finite")

        unit = (initial_controls - control_lower.unsqueeze(1)) / (
            control_upper - control_lower
        ).unsqueeze(1)
        unit = unit.clamp(min=UNIT_LOGIT_MARGIN, max=1.0 - UNIT_LOGIT_MARGIN)
        self.duration_mode = duration_mode
        self.n_segments = int(n_segments)
        self.control_logits = nn.Parameter(torch.log(unit) - torch.log1p(-unit))
        self.register_buffer("control_lower", control_lower)
        self.register_buffer("control_upper", control_upper)
        self.register_buffer("final_time_s", final_time_s)
        if duration_mode == DURATION_FREE:
            self.duration_logits = nn.Parameter(
                torch.zeros(batch, self.n_segments, dtype=initial_controls.dtype,
                            device=initial_controls.device)
            )
        else:
            self.register_parameter("duration_logits", None)

    @property
    def fitted_parameters(self) -> tuple[str, ...]:
        """The parameter names a per-flight best-state selection has to carry."""
        return ("control_logits",) if self.duration_logits is None else (
            "control_logits", "duration_logits"
        )

    def parameter_groups(self, control_lr: float, duration_lr: float) -> list[dict]:
        groups: list[dict] = [{"params": [self.control_logits], "lr": control_lr}]
        if self.duration_logits is not None:
            groups.append({"params": [self.duration_logits], "lr": duration_lr})
        return groups

    def forward(self) -> ControlPrediction:
        unit = torch.sigmoid(self.control_logits)
        controls = self.control_lower.unsqueeze(1) + unit * (
            self.control_upper - self.control_lower
        ).unsqueeze(1)
        if self.duration_logits is None:
            fractions = controls.new_full((len(controls), self.n_segments), 1.0 / self.n_segments)
        else:
            fractions = torch.softmax(self.duration_logits, dim=1)
        return ControlPrediction(
            controls=controls,
            segment_durations=fractions * self.final_time_s.unsqueeze(1),
            final_time_s=self.final_time_s,
        )


def inverse_dynamics_seed(
    series_batch,
    anchor: int,
    dynamics: dict[str, torch.Tensor],
    *,
    config: TSConfig,
    n_segments: int,
    final_time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Seed ``[B, N, 3]`` from each truth track's own inverse dynamics, and the clipped share.

    The reference states come from the series' MEASURED physical channels (the anchor row
    onwards), never from the batch history and never from the fitted approach tail: a
    history row is ``[C + K]`` with the conditioning columns (the trap
    ``batch_contract.anchor_state`` exists for), and the fitted tail carries the terminal
    observed velocity copied onto fitted positions, whose heading step the inverse's
    ``unwrap``+``gradient`` turns into a bank spike (`dataset.reference_control_supervision`
    excludes it for the same reason). The anchor state the dynamics dict carries is checked
    against the one this reconstructs, so a mis-aligned batch fails loudly instead of
    seeding every flight from someone else's anchor.
    """
    lower = dynamics["control_lower"].detach().cpu().numpy().astype(np.float64)
    upper = dynamics["control_upper"].detach().cpu().numpy().astype(np.float64)
    initial_state = dynamics["initial_state"].detach().cpu().numpy().astype(np.float64)
    if len(series_batch) != len(initial_state):
        raise ValueError("the series batch and the dynamics dict must cover the same flights")
    controls = np.empty((len(series_batch), n_segments, 3), dtype=np.float64)
    clipped = np.empty(len(series_batch), dtype=np.float64)
    for row, series in enumerate(series_batch):
        values = series.values[anchor:]
        times = series.times[anchor:] - series.times[anchor]
        mass_kg = float(initial_state[row, -1])
        states = np.asarray(
            [
                [s.latitude, s.longitude, s.altitude, s.V, s.psi, s.gamma, s.m]
                for _t, s in states_from_channels(times, values, series.frame, mass_kg=mass_kg)
            ],
            dtype=np.float64,
        )
        if np.any(np.abs(states[0] - initial_state[row]) > _ANCHOR_TOLERANCE):
            raise RuntimeError(
                f"flight {series.flight_id}: the reconstructed anchor state does not match the "
                f"batch's initial_state ({states[0]} vs {initial_state[row]})"
            )
        inverted = segment_controls(
            states, times,
            config=config,
            aero_params=dynamics["aero_params"][row].detach().cpu().numpy().astype(np.float64),
            max_thrust_n=float(dynamics["max_thrust_n"][row]),
            control_lower=lower[row], control_upper=upper[row],
            n_segments=n_segments, total_duration_s=float(final_time_s[row]),
        )
        controls[row] = inverted.controls
        clipped[row] = float(np.mean(inverted.clipped_fraction))
    return controls, clipped


def clip_gradients_per_flight(
    parameters: list[torch.nn.Parameter], max_norm: float
) -> torch.Tensor:
    """Scale each FLIGHT's gradient to ``max_norm``; return the pre-clip per-flight norms.

    The flights share no parameter, so a joint norm would let the batch's worst flights
    throttle every other flight's step — and by a factor that moves with the batch size
    and its composition, which Adam does not absorb.
    """
    squared: torch.Tensor | None = None
    grads = [p.grad for p in parameters if p.grad is not None]
    for grad in grads:
        rows = grad.detach().reshape(len(grad), -1).square().sum(dim=1)
        squared = rows if squared is None else squared + rows
    if squared is None:
        raise RuntimeError("no parameter carries a gradient")
    norms = squared.sqrt()
    scale = (max_norm / (norms + 1e-12)).clamp(max=1.0)
    for grad in grads:
        grad.mul_(scale.reshape((-1,) + (1,) * (grad.dim() - 1)))
    return norms


@dataclass(frozen=True)
class BasisFitResult:
    """Per-flight outcome of one batched fit, with the evidence that it converged."""

    best_value: torch.Tensor      # [B] the best objective reached
    best_step: np.ndarray         # [B] the step it was reached at
    seed_value: torch.Tensor      # [B] the objective at step 0, before any update
    clipped_share: float          # share of (flight, step) pairs the gradient clip fired on
    steps: int

    @property
    def still_improving(self) -> np.ndarray:
        """Flights whose best step is at the end of the budget: the fit ran out, not converged."""
        return self.best_step >= STILL_IMPROVING_FRACTION * self.steps


def fit_basis_schedules(
    schedule: BasisSchedule,
    objective: Callable[[ControlPrediction], torch.Tensor],
    *,
    steps: int,
    control_learning_rate: float,
    duration_learning_rate: float,
    gradient_clip_norm: float,
) -> BasisFitResult:
    """Adam on the batch; restore each flight's OWN best-objective parameters.

    ``objective`` returns one scalar per flight. The batch mean drives the gradient (the
    flights share no parameter, and Adam is per-parameter scale-invariant), but the best
    state is selected per flight: a batch-level best would discard a flight that converged
    late because another one diverged. Step 0 is evaluated before any update, so the result
    is never worse than the seed and ``seed_value`` records what the seed alone was worth.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    for name, value in (
        ("control_learning_rate", control_learning_rate),
        ("duration_learning_rate", duration_learning_rate),
        ("gradient_clip_norm", gradient_clip_norm),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")

    groups = schedule.parameter_groups(control_learning_rate, duration_learning_rate)
    parameters = [parameter for group in groups for parameter in group["params"]]
    optimizer = torch.optim.Adam(groups)
    fitted = schedule.fitted_parameters
    batch = len(schedule.final_time_s)
    best_value: torch.Tensor | None = None
    seed_value: torch.Tensor | None = None
    best_state = {name: getattr(schedule, name).detach().clone() for name in fitted}
    best_step = np.zeros(batch, dtype=int)
    clipped = 0

    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        per_flight = objective(schedule())
        if per_flight.shape != (batch,):
            raise ValueError(
                f"the objective must return one value per flight, got {tuple(per_flight.shape)}"
            )
        if not torch.all(torch.isfinite(per_flight)):
            raise FloatingPointError(f"non-finite fit objective at step {step}")
        with torch.no_grad():
            value = per_flight.detach()
            if seed_value is None:
                seed_value = value.clone()
            improved = (
                torch.ones_like(value, dtype=torch.bool) if best_value is None else value < best_value
            )
            if improved.any():
                best_value = value.clone() if best_value is None else torch.where(
                    improved, value, best_value
                )
                for name in fitted:
                    tensor = getattr(schedule, name).detach()
                    view = improved.reshape((-1,) + (1,) * (tensor.dim() - 1))
                    best_state[name] = torch.where(view, tensor, best_state[name])
                best_step[improved.cpu().numpy()] = step
        if step == steps:
            break
        per_flight.mean().backward()
        norms = clip_gradients_per_flight(parameters, gradient_clip_norm)
        clipped += int((norms > gradient_clip_norm).sum())
        optimizer.step()

    with torch.no_grad():
        for name in fitted:
            getattr(schedule, name).copy_(best_state[name])
    return BasisFitResult(
        best_value=best_value,
        best_step=best_step,
        seed_value=seed_value,
        clipped_share=clipped / float(batch * steps),
        steps=steps,
    )
