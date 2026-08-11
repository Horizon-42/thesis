"""Open-loop quality metrics for train-only control teacher schedules."""

from __future__ import annotations

import numpy as np
import torch

from channels import POSITION_IDX
from config import TSConfig
from control_oracle import ORACLE_OBJECTIVE_PHYSICAL_CRITERIA, evaluate_control_prediction
from dataset import FixedAnchorTrajectoryWindows
from prediction_outputs import ControlPrediction


def observed_clock_prediction(
    prediction: ControlPrediction, target_final_time_s: torch.Tensor
) -> ControlPrediction:
    """Preserve a schedule's partition while evaluating geometry on the true clock."""
    fractions = prediction.segment_durations / prediction.segment_durations.sum(
        dim=1, keepdim=True
    )
    durations = fractions * target_final_time_s.unsqueeze(1)
    return ControlPrediction(
        controls=prediction.controls,
        segment_durations=durations,
        final_time_s=target_final_time_s,
    )


def move_dynamics(
    dynamics: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in dynamics.items()}


def evaluate_schedule(
    prediction: ControlPrediction,
    dataset: FixedAnchorTrajectoryWindows,
    index: int,
    config: TSConfig,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate one known schedule on the unchanged fixed-2s reference contract."""
    _x, target, _weights, _final_time, _flight_weights, dynamics, supervision = (
        dataset.batch(np.array([index]))
    )
    dynamics_device = move_dynamics(dynamics, device)
    supervision_device = supervision.to(device)
    terminal_target = target[:, -1].to(device)
    with torch.no_grad():
        evaluation = evaluate_control_prediction(
            prediction,
            supervision=supervision_device,
            terminal_target=terminal_target,
            config=config,
            normalizer=dataset.normalizer,
            dynamics=dynamics_device,
            objective_mode=ORACLE_OBJECTIVE_PHYSICAL_CRITERIA,
        )
    valid = supervision.valid[0].numpy()
    predicted = evaluation.rollout.physical_query_states[0].cpu().numpy()[valid]
    truth = dataset.normalizer.decode(
        supervision.states[0, valid].numpy().astype(np.float64)
    )
    distance = np.linalg.norm(
        predicted[:, list(POSITION_IDX)] - truth[:, list(POSITION_IDX)], axis=1
    )
    terminal_position = dataset.normalizer.decode(
        evaluation.rollout.normalized_segment_end_states[0, -1]
        .cpu()
        .numpy()
    )[list(POSITION_IDX)]
    terminal_truth = dataset.normalizer.decode(
        terminal_target[0].cpu().numpy().astype(np.float64)
    )[list(POSITION_IDX)]
    return {
        "ade_m": float(distance.mean()),
        "fde_at_last_complete_dt_m": float(distance[-1]),
        "terminal_distance_m": float(np.linalg.norm(terminal_position - terminal_truth)),
        "physical_criteria": float(evaluation.objective.total.cpu()),
    }
