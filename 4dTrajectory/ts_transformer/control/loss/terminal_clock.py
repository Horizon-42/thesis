"""Terminal-clock strategies for deterministic control training.

Dense state supervision owns its observed-clock rollout.  This module optionally adds a
second endpoint-only rollout on the model's deployable predicted clock and attaches those
endpoints to the common loss-result contract.  Tracking objectives therefore remain
clock-agnostic and contain no rollout branches.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import torch

from config import (
    CONTROL_TERMINAL_CLOCK_PREDICTED,
    CONTROL_TERMINAL_CLOCK_PREDICTED_DETACHED_TIME,
    CONTROL_TERMINAL_CLOCK_STATE_SUPERVISION,
    TSConfig,
)
from control.constraints import build_command_hook
from control.dynamics import rollout as control_rollout
from control.loss.components import ControlStateLossResult
from control.training.curriculum import ControlTrainingStage
from dataset import Normalizer
from prediction_outputs import ControlPrediction


TerminalClockStrategy = Callable[
    [
        ControlStateLossResult,
        ControlPrediction,
        dict[str, torch.Tensor],
        TSConfig,
        Normalizer,
        ControlTrainingStage,
    ],
    ControlStateLossResult,
]


def _state_supervision_terminal(
    result: ControlStateLossResult,
    prediction: ControlPrediction,
    dynamics: dict[str, torch.Tensor],
    config: TSConfig,
    normalizer: Normalizer,
    stage: ControlTrainingStage,
) -> ControlStateLossResult:
    del prediction, dynamics, config, normalizer, stage
    return result


def _predicted_clock_terminal(
    result: ControlStateLossResult,
    prediction: ControlPrediction,
    dynamics: dict[str, torch.Tensor],
    config: TSConfig,
    normalizer: Normalizer,
    stage: ControlTrainingStage,
) -> ControlStateLossResult:
    # Numeric curriculum stages supervise physical prefix boundaries, not a deployment
    # endpoint. Dual-clock terminal training begins with the full-horizon stage.
    if not stage.is_full_horizon:
        return result
    return _roll_out_terminal_endpoints(
        result,
        prediction.controls,
        prediction.segment_durations,
        dynamics,
        config,
        normalizer,
    )


def _predicted_clock_terminal_detached_time(
    result: ControlStateLossResult,
    prediction: ControlPrediction,
    dynamics: dict[str, torch.Tensor],
    config: TSConfig,
    normalizer: Normalizer,
    stage: ControlTrainingStage,
) -> ControlStateLossResult:
    """Supervise deployable endpoints without steering the total-time head.

    Factorized durations contain two independently useful quantities: a learned simplex
    partition and a learned total time.  Re-normalizing the physical durations recovers
    the partition, while multiplying by a detached total keeps the exact forward clock.
    Terminal gradients therefore still train controls and duration fractions, but the
    explicitly supervised final-time loss is the sole owner of the total-time head.
    """
    if not stage.is_full_horizon:
        return result
    fractions = prediction.segment_durations / prediction.segment_durations.sum(
        dim=1, keepdim=True
    )
    durations = fractions * prediction.final_time_s.detach().unsqueeze(1)
    return _roll_out_terminal_endpoints(
        result,
        prediction.controls,
        durations,
        dynamics,
        config,
        normalizer,
    )


def _roll_out_terminal_endpoints(
    result: ControlStateLossResult,
    controls: torch.Tensor,
    durations: torch.Tensor,
    dynamics: dict[str, torch.Tensor],
    config: TSConfig,
    normalizer: Normalizer,
) -> ControlStateLossResult:
    rollout = control_rollout.rollout_control_endpoints(
        controls,
        durations,
        dynamics,
        config,
        command_hook=build_command_hook(config, dynamics),
    )
    dtype, device = rollout.channels.dtype, rollout.channels.device
    mean = torch.as_tensor(normalizer.mean, dtype=dtype, device=device)
    scale = torch.as_tensor(normalizer.std, dtype=dtype, device=device)
    normalized_endpoints = (rollout.channels - mean) / scale
    return replace(result, normalized_terminal_end_states=normalized_endpoints)


_TERMINAL_CLOCK_STRATEGIES: dict[str, TerminalClockStrategy] = {
    CONTROL_TERMINAL_CLOCK_STATE_SUPERVISION: _state_supervision_terminal,
    CONTROL_TERMINAL_CLOCK_PREDICTED: _predicted_clock_terminal,
    CONTROL_TERMINAL_CLOCK_PREDICTED_DETACHED_TIME: (
        _predicted_clock_terminal_detached_time
    ),
}


def apply_control_terminal_clock(
    result: ControlStateLossResult,
    prediction: ControlPrediction,
    dynamics: dict[str, torch.Tensor],
    config: TSConfig,
    normalizer: Normalizer,
    stage: ControlTrainingStage,
) -> ControlStateLossResult:
    """Attach endpoints from the configured terminal clock to a dense-loss result."""
    return _TERMINAL_CLOCK_STRATEGIES[config.control_terminal_supervision_clock](
        result, prediction, dynamics, config, normalizer, stage
    )
