"""Small-cohort rollout fine-tuning after oracle-teacher imitation."""

from __future__ import annotations

import torch
from torch import nn

from config import TSConfig
from control_training_curriculum import ControlTrainingStage
from dataset import Normalizer
from fixed_dt_supervision import FixedDTControlSupervision
from train import model_forward, prediction_loss_components

from oracle_teacher.optimization import TeacherOptimizationStage


def fine_tune_model_on_rollout(
    model: nn.Module,
    *,
    x: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    final_time_s: torch.Tensor,
    dynamics: dict[str, torch.Tensor],
    supervision: FixedDTControlSupervision,
    config: TSConfig,
    normalizer: Normalizer,
    stages: tuple[TeacherOptimizationStage, ...],
    learning_rate: float,
    gradient_clip_norm: float,
    log_every: int,
) -> list[dict[str, float | int | str]]:
    """Run production loss stages without any future-aware deployment input."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    flight_weights = torch.ones(len(x), dtype=x.dtype, device=x.device)
    history: list[dict[str, float | int | str]] = []
    global_step = 0
    for stage in stages:
        training_stage = ControlTrainingStage(stage.label, stage.horizon_s, 1, None)
        for stage_step in range(1, stage.steps + 1):
            model.train()
            optimizer.zero_grad()
            prediction = model_forward(model, x, dynamics)
            components = prediction_loss_components(
                prediction,
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
                model.parameters(), gradient_clip_norm
            )
            optimizer.step()
            global_step += 1
            if stage_step == 1 or stage_step % log_every == 0 or stage_step == stage.steps:
                row: dict[str, float | int | str] = {
                    "step": global_step,
                    "stage": stage.label,
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
                    f"rollout {global_step:4d} {stage.label:>4s}: "
                    f"loss={row['loss']:.5f} grad={row['gradient_norm']:.3g}"
                )
    return history
