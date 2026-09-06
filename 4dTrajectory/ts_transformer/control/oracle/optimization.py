"""Batched direct-shooting refinement for train-only oracle teacher schedules."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from batch_contract import anchor_state
from config import TSConfig
from control.oracle.basis import DURATION_UNIFORM, BasisSchedule
from dataset import Normalizer
from fixed_dt_supervision import FixedDTControlSupervision
from prediction_outputs import ControlPrediction


# train.prediction_loss_components, injected. Typed loosely on purpose: spelling out the
# ten-argument signature here would duplicate a contract that already has one home.
LossComponentsFn = Callable[..., Any]


class BatchedOracleTeacher(BasisSchedule):
    """One independent bounded control schedule per train flight.

    The teacher's schedule IS a uniform-duration :class:`BasisSchedule`; this name and its
    positional signature stay because the teacher generator and the paired-CV ablation both
    construct it, and a schedule defined twice is two subtly different search spaces.
    """

    def __init__(
        self,
        initial_controls: torch.Tensor,
        control_lower: torch.Tensor,
        control_upper: torch.Tensor,
        final_time_s: torch.Tensor,
    ) -> None:
        super().__init__(
            initial_controls, control_lower, control_upper, final_time_s, DURATION_UNIFORM
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
    steps: int,
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
    if steps < 1:
        raise ValueError("teacher refinement steps must be positive")
    optimizer = torch.optim.Adam((teacher.control_logits,), lr=learning_rate)
    flight_weights = torch.ones(len(x), dtype=x.dtype, device=x.device)
    history: list[dict[str, float | int | str]] = []
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        components = loss_components(
            teacher(),
            anchor_state(x, len(config.channels)),
            target,
            weights,
            final_time_s,
            flight_weights,
            config,
            normalizer,
            dynamics,
            supervision,
        )
        components.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            (teacher.control_logits,), gradient_clip_norm
        )
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            row: dict[str, float | int | str] = {
                "step": step,
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
                f"teacher step {step:4d}: "
                f"loss={row['loss']:.5f} grad={row['gradient_norm']:.3g}"
            )
    return history
