from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import torch


TS_DIR = Path(__file__).resolve().parents[1]
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

import control.loss.terminal_clock as terminal_clock_module  # noqa: E402
import control.dynamics.rollout as control_rollout_module  # noqa: E402
from config import (  # noqa: E402
    CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
    CONTROL_TERMINAL_CLOCK_PREDICTED,
    CONTROL_TERMINAL_CLOCK_PREDICTED_DETACHED_TIME,
    PREDICTION_CONTROL,
    TSConfig,
)
from control.loss.components import ControlStateLossResult  # noqa: E402
from control.training.curriculum import ControlTrainingStage  # noqa: E402
from dataset import Normalizer  # noqa: E402
from prediction_outputs import ControlPrediction  # noqa: E402


def _config() -> TSConfig:
    return TSConfig(
        prediction_output=PREDICTION_CONTROL,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        control_state_objective=CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
        checkpoint_selection_metric=CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
        control_terminal_supervision_clock=CONTROL_TERMINAL_CLOCK_PREDICTED,
    )


class _DurationBackend:
    """Stand-in whose endpoint position is just cumulative time, so the CLOCK is visible."""

    def endpoint_rollout(self, inputs, config, *, command_hook=None):
        del config, command_hook
        position = inputs.segment_durations_s.cumsum(dim=1)
        channels = torch.stack(
            (
                position,
                position * 0.0,
                position * 0.0,
                position * 0.0,
                position * 0.0,
                position * 0.0,
            ),
            dim=-1,
        )
        return SimpleNamespace(channels=channels, geodetic_states=channels, controls=inputs.controls)


def test_predicted_terminal_clock_attaches_deployable_endpoints_and_gradients(monkeypatch):
    monkeypatch.setattr(
        control_rollout_module,
        "control_dynamics_backend",
        lambda config: _DurationBackend(),
    )
    durations = torch.tensor([[2.0, 3.0]], requires_grad=True)
    prediction = ControlPrediction(
        controls=torch.zeros(1, 2, 3),
        segment_durations=durations,
        final_time_s=durations.sum(dim=1),
    )
    observed_endpoints = torch.zeros(1, 2, 6, dtype=torch.float64)
    result = ControlStateLossResult(
        normalized_mse=torch.zeros(1, dtype=torch.float64),
        normalized_segment_end_states=observed_endpoints,
    )
    dynamics = {
        "initial_state": torch.zeros(1, 7),
        "initial_controls": torch.zeros(1, 3),
        "aero_params": torch.zeros(1, 1),
        "frame_params": torch.zeros(1, 1),
        "max_thrust_n": torch.ones(1),
    }

    dual = terminal_clock_module.apply_control_terminal_clock(
        result,
        prediction,
        dynamics,
        _config(),
        Normalizer(mean=np.zeros(6), std=np.ones(6)),
        ControlTrainingStage("full", None, 1, None),
    )
    dual.terminal_end_states[:, -1, 0].sum().backward()

    assert dual.normalized_segment_end_states is observed_endpoints
    torch.testing.assert_close(
        dual.terminal_end_states[:, -1, 0], torch.tensor([5.0], dtype=torch.float64)
    )
    torch.testing.assert_close(durations.grad, torch.ones_like(durations))


def test_detached_time_terminal_trains_partition_but_not_total_time(monkeypatch):
    monkeypatch.setattr(
        control_rollout_module,
        "control_dynamics_backend",
        lambda config: _DurationBackend(),
    )
    total_time = torch.tensor([5.0], requires_grad=True)
    duration_logits = torch.tensor([[0.2, -0.2]], requires_grad=True)
    durations = torch.softmax(duration_logits, dim=1) * total_time.unsqueeze(1)
    prediction = ControlPrediction(
        controls=torch.zeros(1, 2, 3),
        segment_durations=durations,
        final_time_s=total_time,
    )
    result = ControlStateLossResult(
        normalized_mse=torch.zeros(1, dtype=torch.float64),
        normalized_segment_end_states=torch.zeros(1, 2, 6, dtype=torch.float64),
    )
    config = TSConfig.from_dict(
        {
            **_config().to_dict(),
            "control_terminal_supervision_clock": (
                CONTROL_TERMINAL_CLOCK_PREDICTED_DETACHED_TIME
            ),
        }
    )

    dual = terminal_clock_module.apply_control_terminal_clock(
        result,
        prediction,
        {
            "initial_state": torch.zeros(1, 7),
            "initial_controls": torch.zeros(1, 3),
            "aero_params": torch.zeros(1, 1),
            "frame_params": torch.zeros(1, 1),
            "max_thrust_n": torch.ones(1),
        },
        config,
        Normalizer(mean=np.zeros(6), std=np.ones(6)),
        ControlTrainingStage("full", None, 1, None),
    )
    dual.terminal_end_states[:, 0, 0].sum().backward()

    torch.testing.assert_close(total_time.grad, torch.zeros_like(total_time))
    assert duration_logits.grad is not None
    assert duration_logits.grad.abs().sum() > 0


def test_predicted_terminal_clock_keeps_observed_prefix_during_curriculum(monkeypatch):
    monkeypatch.setattr(
        control_rollout_module,
        "control_dynamics_backend",
        lambda config: (_ for _ in ()).throw(AssertionError("rollout must not run")),
    )
    prediction = ControlPrediction(
        controls=torch.zeros(1, 2, 3),
        segment_durations=torch.ones(1, 2),
        final_time_s=torch.tensor([2.0]),
    )
    result = ControlStateLossResult(
        normalized_mse=torch.zeros(1),
        normalized_segment_end_states=torch.zeros(1, 2, 6),
    )

    prefix = terminal_clock_module.apply_control_terminal_clock(
        result,
        prediction,
        {},
        _config(),
        Normalizer(mean=np.zeros(6), std=np.ones(6)),
        ControlTrainingStage("60s", 60.0, 1, 10),
    )

    assert prefix is result
