"""Short-to-long horizon views for single-shooting oracle curriculum.

Cropping only the loss mask would still integrate the full trajectory.  These helpers crop
the control schedule and its final active duration as well, so each early stage genuinely
has a shorter forward and backward RK4 chain.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch

from fixed_dt_supervision import FixedDTControlSupervision
from prediction_outputs import ControlPrediction


@dataclass(frozen=True)
class HorizonCurriculumStage:
    """One single-shooting stage with an explicit duration-optimization contract."""

    horizon_s: float
    steps: int
    is_full_horizon: bool
    optimize_duration: bool


@dataclass(frozen=True)
class HorizonStageView:
    """Prediction and references that belong to one curriculum horizon."""

    prediction: ControlPrediction
    supervision: FixedDTControlSupervision
    terminal_target: torch.Tensor


def build_horizon_curriculum(
    horizon_tokens: Sequence[str],
    stage_steps: Sequence[int],
    *,
    total_duration_s: float,
    supervision_dt_s: float,
    optimize_full_duration: bool = True,
) -> tuple[HorizonCurriculumStage, ...]:
    """Validate CLI stage tokens and resolve the terminal ``full`` stage."""
    if not horizon_tokens:
        raise ValueError("horizon curriculum requires at least one stage")
    if len(horizon_tokens) != len(stage_steps):
        raise ValueError("stage horizons and stage steps must have the same length")
    if not math.isfinite(total_duration_s) or total_duration_s <= 0.0:
        raise ValueError("total trajectory duration must be positive and finite")
    if not math.isfinite(supervision_dt_s) or supervision_dt_s <= 0.0:
        raise ValueError("supervision dt must be positive and finite")
    if any(steps <= 0 for steps in stage_steps):
        raise ValueError("every curriculum stage must have positive steps")

    stages: list[HorizonCurriculumStage] = []
    previous_horizon = 0.0
    for index, (token, steps) in enumerate(zip(horizon_tokens, stage_steps)):
        normalized = token.strip().lower()
        is_full = normalized == "full"
        if is_full:
            if index != len(horizon_tokens) - 1:
                raise ValueError("full horizon must be the final curriculum stage")
            horizon = total_duration_s
        else:
            try:
                horizon = float(normalized)
            except ValueError as exc:
                raise ValueError(
                    f"invalid curriculum horizon {token!r}; expected seconds or 'full'"
                ) from exc
            if not math.isfinite(horizon) or horizon <= 0.0:
                raise ValueError("numeric curriculum horizons must be positive and finite")
            if horizon >= total_duration_s:
                raise ValueError(
                    "numeric curriculum horizons must be shorter than the full trajectory"
                )
            grid_multiple = round(horizon / supervision_dt_s)
            if not math.isclose(
                horizon,
                grid_multiple * supervision_dt_s,
                rel_tol=0.0,
                abs_tol=max(1e-7, supervision_dt_s * 1e-7),
            ):
                raise ValueError(
                    "numeric curriculum horizons must align with the fixed-dt reference grid"
                )
        if horizon <= previous_horizon:
            raise ValueError("curriculum horizons must be strictly increasing")
        stages.append(
            HorizonCurriculumStage(
                horizon_s=horizon,
                steps=int(steps),
                is_full_horizon=is_full,
                optimize_duration=is_full and optimize_full_duration,
            )
        )
        previous_horizon = horizon
    if not stages[-1].is_full_horizon:
        raise ValueError("curriculum must end with a full horizon stage")
    return tuple(stages)


def truncate_supervision(
    supervision: FixedDTControlSupervision,
    horizon_s: float,
) -> FixedDTControlSupervision:
    if horizon_s <= 0.0:
        raise ValueError("curriculum horizon must be positive")
    tolerance = max(1e-9, horizon_s * 1e-9)
    valid = supervision.valid & (
        supervision.query_offsets_s <= horizon_s + tolerance
    )
    if not torch.all(valid.any(dim=1)):
        raise ValueError("curriculum horizon contains no fixed-dt supervision point")
    # Fixed-dt rows are valid prefixes. Slice away later columns so the dense rollout does
    # not retain a full-horizon tail of invalid, zero-duration events.
    counts = valid.sum(dim=1)
    max_points = int(counts.max().detach().cpu())
    return FixedDTControlSupervision(
        query_offsets_s=supervision.query_offsets_s[:, :max_points],
        states=supervision.states[:, :max_points],
        weights=supervision.weights[:, :max_points],
        valid=valid[:, :max_points],
    )


def stage_terminal_target(
    supervision: FixedDTControlSupervision,
    horizon_s: float,
) -> torch.Tensor:
    """Return the normalized reference state exactly at a fixed-dt stage boundary."""
    tolerance = max(1e-7, horizon_s * 1e-7)
    matches = supervision.valid & torch.isclose(
        supervision.query_offsets_s,
        torch.as_tensor(
            horizon_s,
            dtype=supervision.query_offsets_s.dtype,
            device=supervision.query_offsets_s.device,
        ),
        rtol=0.0,
        atol=tolerance,
    )
    if not torch.all(matches.sum(dim=1) == 1):
        raise ValueError(
            "each curriculum horizon must match exactly one fixed-dt reference point"
        )
    indices = matches.to(torch.int64).argmax(dim=1)
    rows = torch.arange(len(indices), device=indices.device)
    return supervision.states[rows, indices]


def truncate_prediction(
    prediction: ControlPrediction,
    horizon_s: float,
    *,
    detach_durations: bool,
) -> ControlPrediction:
    """Crop a single-flight schedule at ``horizon_s`` without rescaling its prefix."""
    if prediction.controls.shape[0] != 1:
        raise ValueError("staged oracle currently requires exactly one flight")
    total = float(prediction.final_time_s[0].detach().cpu())
    if horizon_s <= 0.0 or horizon_s > total + max(1e-7, total * 1e-7):
        raise ValueError("curriculum horizon must be within the prediction duration")
    if horizon_s >= total - max(1e-7, total * 1e-7):
        durations = prediction.segment_durations
        if detach_durations:
            durations = durations.detach()
        return ControlPrediction(
            controls=prediction.controls,
            segment_durations=durations,
            final_time_s=(
                prediction.final_time_s.detach()
                if detach_durations
                else prediction.final_time_s
            ),
        )

    durations = prediction.segment_durations[0]
    boundaries = durations.detach().cumsum(dim=0)
    horizon = torch.as_tensor(horizon_s, dtype=durations.dtype, device=durations.device)
    active_count = int(torch.searchsorted(boundaries, horizon, right=False).item()) + 1
    prefix_before_last = durations[: active_count - 1]
    final_duration = horizon - prefix_before_last.sum()
    if float(final_duration.detach().cpu()) <= 0.0:
        raise RuntimeError("curriculum produced a non-positive final control duration")
    stage_durations = torch.cat((prefix_before_last, final_duration.reshape(1)))
    if detach_durations:
        stage_durations = stage_durations.detach()
    return ControlPrediction(
        controls=prediction.controls[:, :active_count],
        segment_durations=stage_durations.unsqueeze(0),
        final_time_s=horizon.reshape(1),
    )


def build_horizon_stage_view(
    prediction: ControlPrediction,
    supervision: FixedDTControlSupervision,
    terminal_target: torch.Tensor,
    stage: HorizonCurriculumStage,
) -> HorizonStageView:
    """Construct either a cropped early-stage view or the untouched full view."""
    if stage.is_full_horizon:
        return HorizonStageView(prediction, supervision, terminal_target)
    return HorizonStageView(
        prediction=truncate_prediction(
            prediction,
            stage.horizon_s,
            detach_durations=True,
        ),
        supervision=truncate_supervision(supervision, stage.horizon_s),
        terminal_target=stage_terminal_target(supervision, stage.horizon_s),
    )
