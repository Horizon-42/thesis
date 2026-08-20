"""Physical-time curriculum views for batched learned-control training.

The model always emits its full control schedule. Early stages retain only the positive
prefix needed to reach the configured physical horizon, so both the forward RK4 chain and
its adjoint are genuinely shorter. Flights already shorter than a stage keep their full
trajectory and runway terminal target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from fixed_dt_supervision import FixedDTControlSupervision
from prediction_outputs import ControlPrediction


@dataclass(frozen=True)
class ControlTrainingStage:
    label: str
    horizon_s: float | None
    start_epoch: int
    end_epoch: int | None

    @property
    def is_full_horizon(self) -> bool:
        return self.horizon_s is None


@dataclass(frozen=True)
class ControlTrainingStageView:
    prediction: ControlPrediction
    supervision: FixedDTControlSupervision
    terminal_target: torch.Tensor
    segment_valid: torch.Tensor | None


def build_control_training_stages(
    horizons_s: Sequence[float],
    *,
    epochs_per_stage: int,
    total_epochs: int,
) -> tuple[ControlTrainingStage, ...]:
    """Return fixed-length prefix stages followed by one full-horizon stage."""
    if epochs_per_stage <= 0:
        raise ValueError("curriculum epochs per stage must be positive")
    if total_epochs <= len(horizons_s) * epochs_per_stage:
        raise ValueError("epoch budget must leave at least one full-horizon epoch")
    stages: list[ControlTrainingStage] = []
    for index, horizon_s in enumerate(horizons_s):
        start = index * epochs_per_stage + 1
        end = (index + 1) * epochs_per_stage
        stages.append(
            ControlTrainingStage(
                label=f"{horizon_s:g}s",
                horizon_s=float(horizon_s),
                start_epoch=start,
                end_epoch=end,
            )
        )
    full_start = len(horizons_s) * epochs_per_stage + 1
    stages.append(
        ControlTrainingStage(
            label="full",
            horizon_s=None,
            start_epoch=full_start,
            end_epoch=None,
        )
    )
    return tuple(stages)


def control_training_stage_for_epoch(
    stages: Sequence[ControlTrainingStage], epoch: int
) -> ControlTrainingStage:
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    for stage in stages:
        if epoch >= stage.start_epoch and (
            stage.end_epoch is None or epoch <= stage.end_epoch
        ):
            return stage
    raise ValueError(f"epoch {epoch} is outside the curriculum schedule")


def _truncate_supervision(
    supervision: FixedDTControlSupervision,
    effective_horizon_s: torch.Tensor,
) -> FixedDTControlSupervision:
    tolerance = torch.finfo(supervision.query_offsets_s.dtype).eps * (
        effective_horizon_s.abs().clamp(min=1.0)
    ) * 16
    valid = supervision.valid & (
        supervision.query_offsets_s
        <= effective_horizon_s.unsqueeze(1) + tolerance.unsqueeze(1)
    )
    if not torch.all(valid.any(dim=1)):
        raise ValueError("curriculum stage contains no fixed-dt supervision point")
    max_points = int(valid.sum(dim=1).max().detach().cpu())
    return FixedDTControlSupervision(
        query_offsets_s=supervision.query_offsets_s[:, :max_points],
        states=supervision.states[:, :max_points],
        weights=supervision.weights[:, :max_points],
        valid=valid[:, :max_points],
    )


def _stage_terminal_target(
    supervision: FixedDTControlSupervision,
    full_terminal_target: torch.Tensor,
    target_final_time_s: torch.Tensor,
    horizon_s: float,
) -> torch.Tensor:
    """Use the prefix boundary target, except for flights already at full horizon."""
    dtype = supervision.query_offsets_s.dtype
    device = supervision.query_offsets_s.device
    horizon = torch.as_tensor(horizon_s, dtype=dtype, device=device)
    tolerance = max(1e-7, horizon_s * 1e-7)
    shortened = target_final_time_s.to(dtype=dtype, device=device) > horizon + tolerance
    matches = supervision.valid & torch.isclose(
        supervision.query_offsets_s,
        horizon,
        rtol=0.0,
        atol=tolerance,
    )
    if torch.any(shortened & (matches.sum(dim=1) != 1)):
        raise ValueError(
            "every shortened curriculum trajectory must contain its fixed-dt boundary"
        )
    indices = matches.to(torch.int64).argmax(dim=1)
    rows = torch.arange(len(indices), device=device)
    prefix_target = supervision.states[rows, indices].to(
        dtype=full_terminal_target.dtype,
        device=full_terminal_target.device,
    )
    return torch.where(
        shortened.to(device=full_terminal_target.device).unsqueeze(1),
        prefix_target,
        full_terminal_target,
    )


def close_duration_prefix(
    durations: torch.Tensor,
    segment_valid: torch.Tensor,
    total_duration: torch.Tensor,
) -> torch.Tensor:
    """Close the final valid segment on the caller's exact physical-time clock."""

    if durations.ndim != 2 or segment_valid.shape != durations.shape:
        raise ValueError("durations and segment_valid must be matching [batch, segment] tensors")
    if total_duration.shape != durations.shape[:1]:
        raise ValueError("total_duration must contain one value per trajectory")
    if not torch.all(segment_valid.any(dim=1)):
        raise ValueError("every trajectory must contain at least one valid segment")
    if torch.any(segment_valid[:, 1:] & ~segment_valid[:, :-1]):
        raise ValueError("valid control segments must form a prefix")

    durations = torch.where(segment_valid, durations, torch.zeros_like(durations))
    last_valid = segment_valid.sum(dim=1, dtype=torch.long) - 1
    cumulative = durations.cumsum(dim=1)
    previous_index = (last_valid - 1).clamp(min=0)
    previous_total = torch.gather(
        cumulative, 1, previous_index.unsqueeze(1)
    ).squeeze(1)
    previous_total = torch.where(
        last_valid > 0, previous_total, torch.zeros_like(previous_total)
    )
    corrected_last = total_duration - previous_total
    closed = durations.scatter(
        1, last_valid.unsqueeze(1), corrected_last.unsqueeze(1)
    )
    if not torch.allclose(
        closed.sum(dim=1), total_duration, rtol=1e-6, atol=1e-6
    ):
        raise RuntimeError("control durations do not sum to the physical-time horizon")
    return closed


def build_control_training_stage_view(
    prediction: ControlPrediction,
    supervision: FixedDTControlSupervision,
    full_terminal_target: torch.Tensor,
    target_final_time_s: torch.Tensor,
    stage: ControlTrainingStage,
) -> ControlTrainingStageView:
    """Build a batched exact-prefix view without changing model output dimensions."""
    if stage.is_full_horizon:
        dtype = supervision.query_offsets_s.dtype
        device = prediction.segment_durations.device
        durations = prediction.segment_durations.to(dtype=dtype, device=device)
        segment_valid = torch.ones_like(durations, dtype=torch.bool)
        total_duration = prediction.final_time_s.to(dtype=dtype, device=device)
        durations = close_duration_prefix(
            durations, segment_valid, total_duration
        )
        return ControlTrainingStageView(
            prediction=ControlPrediction(
                controls=prediction.controls,
                segment_durations=durations,
                final_time_s=prediction.final_time_s,
            ),
            supervision=supervision,
            terminal_target=full_terminal_target,
            segment_valid=None,
        )
    if stage.horizon_s is None or not math.isfinite(stage.horizon_s):
        raise ValueError("numeric curriculum stage requires a finite horizon")

    # Curriculum boundaries belong to the fixed-dt reference clock (float64). Cropping
    # float32 duration fractions and only then promoting them can leave an exact 60 s
    # query a few microseconds beyond the reconstructed total.
    device = prediction.segment_durations.device
    dtype = supervision.query_offsets_s.dtype
    durations = prediction.segment_durations.to(dtype=dtype, device=device)
    target_total = target_final_time_s.to(dtype=dtype, device=device)
    horizon = torch.full_like(target_total, stage.horizon_s)
    effective_horizon = torch.minimum(target_total, horizon)
    starts = torch.cat(
        (torch.zeros_like(durations[:, :1]), durations.cumsum(dim=1)[:, :-1]),
        dim=1,
    )
    remaining = (effective_horizon.unsqueeze(1) - starts).clamp(min=0.0)
    stage_durations = torch.minimum(durations, remaining)
    segment_valid = stage_durations > 0.0
    if not torch.all(segment_valid.any(dim=1)):
        raise ValueError("curriculum stage retained no control segment")
    if torch.any(segment_valid[:, 1:] & ~segment_valid[:, :-1]):
        raise RuntimeError("curriculum control mask must be a valid prefix")
    stage_durations = torch.where(
        segment_valid, stage_durations, torch.zeros_like(stage_durations)
    )
    # Close the prefix on the float64 reference clock. Merely promoting float32 model
    # durations preserves their accumulated rounding error; for a flight ending exactly on
    # a fixed-dt query, that can leave the query a few microseconds beyond the rollout.
    # Recompute the last valid duration from the preceding float64 cumulative boundary so
    # the dense scheduler sees the intended effective horizon exactly.
    stage_durations = close_duration_prefix(
        stage_durations, segment_valid, effective_horizon
    )

    stage_prediction = ControlPrediction(
        controls=prediction.controls,
        segment_durations=stage_durations,
        final_time_s=effective_horizon,
    )
    terminal_target = _stage_terminal_target(
        supervision,
        full_terminal_target,
        target_final_time_s,
        stage.horizon_s,
    )
    stage_supervision = _truncate_supervision(
        supervision,
        effective_horizon.to(
            dtype=supervision.query_offsets_s.dtype,
            device=supervision.query_offsets_s.device,
        ),
    )
    return ControlTrainingStageView(
        prediction=stage_prediction,
        supervision=stage_supervision,
        terminal_target=terminal_target,
        segment_valid=segment_valid,
    )
