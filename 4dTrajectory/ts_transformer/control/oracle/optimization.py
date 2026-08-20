"""Batched direct-shooting refinement for train-only oracle teacher schedules."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from config import TSConfig
from control.training.curriculum import ControlTrainingStage
from dataset import Normalizer
from fixed_dt_supervision import FixedDTControlSupervision
from prediction_outputs import ControlPrediction


# The one short-to-long refinement schedule every teacher cohort is fitted with. Early
# stages integrate a genuinely shorter RK4 chain (the prefix crop in
# ``control_training_curriculum``), so a cohort converges before it ever sees a 600 s
# adjoint. Both the production generator and the paired-CV ablation call this, so a
# schedule change cannot land in one and not the other.
TEACHER_REFINEMENT_HORIZONS_S = (60.0, 120.0, 240.0)


# train.prediction_loss_components, injected. Typed loosely on purpose: spelling out the
# eleven-argument signature here would duplicate a contract that already has one home.
LossComponentsFn = Callable[..., Any]


def teacher_optimization_stages(
    prefix_steps: int, full_steps: int
) -> tuple[tuple[ControlTrainingStage, int], ...]:
    """Return the canonical ``(stage, steps)`` refinement schedule."""
    if prefix_steps < 1 or full_steps < 1:
        raise ValueError("teacher stage steps must be positive")
    prefix = tuple(
        (ControlTrainingStage(f"{horizon_s:g}s", horizon_s, 1, None), prefix_steps)
        for horizon_s in TEACHER_REFINEMENT_HORIZONS_S
    )
    return (*prefix, (ControlTrainingStage("full", None, 1, None), full_steps))


class BatchedOracleTeacher(nn.Module):
    """One independent bounded control schedule per train flight."""

    def __init__(
        self,
        initial_controls: torch.Tensor,
        control_lower: torch.Tensor,
        control_upper: torch.Tensor,
        final_time_s: torch.Tensor,
    ) -> None:
        super().__init__()
        unit = (initial_controls - control_lower.unsqueeze(1)) / (
            control_upper - control_lower
        ).unsqueeze(1)
        unit = unit.clamp(min=1e-6, max=1.0 - 1e-6)
        self.control_logits = nn.Parameter(torch.log(unit) - torch.log1p(-unit))
        self.register_buffer("control_lower", control_lower)
        self.register_buffer("control_upper", control_upper)
        self.register_buffer("final_time_s", final_time_s)

    def forward(self) -> ControlPrediction:
        unit = torch.sigmoid(self.control_logits)
        controls = self.control_lower.unsqueeze(1) + unit * (
            self.control_upper - self.control_lower
        ).unsqueeze(1)
        segments = controls.shape[1]
        durations = self.final_time_s.unsqueeze(1).expand(-1, segments) / segments
        return ControlPrediction(
            controls=controls,
            segment_durations=durations,
            final_time_s=self.final_time_s,
        )


def optimize_teacher_controls(
    teacher: BatchedOracleTeacher,
    *,
    x: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    final_time_s: torch.Tensor,
    dynamics: dict[str, torch.Tensor],
    supervision: FixedDTControlSupervision,
    config: TSConfig,
    normalizer: Normalizer,
    stages: tuple[tuple[ControlTrainingStage, int], ...],
    learning_rate: float,
    gradient_clip_norm: float,
    log_every: int,
    loss_components: LossComponentsFn,
) -> list[dict[str, float | int | str]]:
    """Refine controls with the production loss while keeping every clock fixed.

    ``loss_components`` is injected rather than imported: it is ``train``'s objective
    dispatch, and importing the training loop from inside ``control`` would invert the
    layering — the loop imports this package, not the other way round. Passing it also
    makes it explicit at both call sites that the teacher is refined against exactly the
    production objective, which is the whole point of the oracle.
    """
    optimizer = torch.optim.Adam((teacher.control_logits,), lr=learning_rate)
    flight_weights = torch.ones(len(x), dtype=x.dtype, device=x.device)
    history: list[dict[str, float | int | str]] = []
    global_step = 0
    for training_stage, stage_steps in stages:
        for stage_step in range(1, stage_steps + 1):
            optimizer.zero_grad()
            components = loss_components(
                teacher(),
                x[:, -1],
                target,
                weights,
                final_time_s,
                flight_weights,
                config,
                normalizer,
                dynamics,
                supervision,
                training_stage,
            )
            components.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                (teacher.control_logits,), gradient_clip_norm
            )
            optimizer.step()
            global_step += 1
            if stage_step == 1 or stage_step % log_every == 0 or stage_step == stage_steps:
                row: dict[str, float | int | str] = {
                    "step": global_step,
                    "stage": training_stage.label,
                    "stage_step": stage_step,
                    "loss": float(components.total.detach()),
                    "gradient_norm": float(gradient_norm.detach()),
                }
                row.update(
                    {
                        name: float(value.detach())
                        for name, value in components.tensors().items()
                    }
                )
                history.append(row)
                print(
                    f"teacher step {global_step:4d} {training_stage.label:>4s}: "
                    f"loss={row['loss']:.5f} grad={row['gradient_norm']:.3g}"
                )
    return history
