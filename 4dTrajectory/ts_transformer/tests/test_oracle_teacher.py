from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


TS_DIR = Path(__file__).resolve().parents[1]
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

from oracle_teacher.imitation import control_imitation_loss  # noqa: E402
from oracle_teacher.evaluation import observed_clock_prediction  # noqa: E402
from oracle_teacher.progressive_pretraining import (  # noqa: E402
    coarsen_schedule,
    refine_control_model,
)
from prediction_outputs import ControlOutputHead, ControlPrediction  # noqa: E402


def test_control_imitation_loss_balances_physical_control_ranges():
    lower = torch.tensor([[0.0, -1.0, 0.5]])
    upper = torch.tensor([[200_000.0, 1.0, 2.0]])
    target_controls = torch.tensor([[[100_000.0, 0.0, 1.0]]])
    predicted_controls = target_controls + torch.tensor([[[20_000.0, 0.2, 0.15]]])
    prediction = ControlPrediction(
        controls=predicted_controls,
        segment_durations=torch.tensor([[5.0]]),
        final_time_s=torch.tensor([5.0]),
    )

    loss = control_imitation_loss(
        prediction,
        target_controls,
        torch.tensor([[5.0]]),
        torch.tensor([5.0]),
        lower,
        upper,
        final_time_scale_s=600.0,
    )

    # Every channel differs by exactly 10% of its own aircraft-specific range.
    torch.testing.assert_close(loss.control, torch.tensor(0.01))
    torch.testing.assert_close(loss.duration_fraction, torch.tensor(0.0))
    torch.testing.assert_close(loss.final_time, torch.tensor(0.0))


def test_observed_clock_teacher_evaluation_preserves_partition_and_replaces_total():
    prediction = ControlPrediction(
        controls=torch.zeros(1, 2, 3),
        segment_durations=torch.tensor([[2.0, 6.0]]),
        final_time_s=torch.tensor([8.0]),
    )

    state_clock = observed_clock_prediction(prediction, torch.tensor([20.0]))

    torch.testing.assert_close(
        state_clock.segment_durations, torch.tensor([[5.0, 15.0]])
    )
    torch.testing.assert_close(state_clock.final_time_s, torch.tensor([20.0]))


def test_progressive_teacher_coarsening_preserves_time_and_weights_controls():
    controls = torch.tensor(
        [[
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
            [4.0, 4.0, 4.0],
            [8.0, 8.0, 8.0],
        ]]
    )
    durations = torch.tensor([[1.0, 3.0, 2.0, 2.0]])

    coarse_controls, coarse_durations = coarsen_schedule(
        controls, durations, 2
    )

    torch.testing.assert_close(coarse_durations, torch.tensor([[4.0, 4.0]]))
    torch.testing.assert_close(
        coarse_controls,
        torch.tensor([[[1.5, 1.5, 1.5], [6.0, 6.0, 6.0]]]),
    )


class _HeadContainer(nn.Module):
    def __init__(self, segments: int):
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.control_head = ControlOutputHead(4, segments)


def test_progressive_head_refinement_preserves_rollout_schedule_exactly():
    torch.manual_seed(7)
    coarse = _HeadContainer(2)
    fine = _HeadContainer(4)
    refine_control_model(coarse, fine)
    features = torch.randn(3, 4)
    total_time = torch.tensor([20.0, 30.0, 40.0])
    lower = torch.tensor([[0.0, -1.0, 0.5]]).expand(3, -1)
    upper = torch.tensor([[200_000.0, 1.0, 2.0]]).expand(3, -1)

    coarse_prediction = coarse.control_head(
        features, total_time, lower=lower, upper=upper
    )
    fine_prediction = fine.control_head(
        features, total_time, lower=lower, upper=upper
    )

    torch.testing.assert_close(
        fine_prediction.controls,
        coarse_prediction.controls.repeat_interleave(2, dim=1),
    )
    torch.testing.assert_close(
        fine_prediction.segment_durations,
        coarse_prediction.segment_durations.repeat_interleave(2, dim=1) / 2.0,
    )
