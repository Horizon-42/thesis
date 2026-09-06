"""The batched control-basis fit: bounds, the given duration, and per-flight best state."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from control.oracle.basis import (
    DURATION_FREE,
    DURATION_MODES,
    DURATION_UNIFORM,
    BasisSchedule,
    clip_gradients_per_flight,
    fit_basis_schedules,
    free_number_count,
)


def _schedule(duration_mode: str, *, batch: int = 2, n_segments: int = 4) -> BasisSchedule:
    lower = torch.tensor([[-0.2, -0.8, 0.2]] * batch, dtype=torch.float64)
    upper = torch.tensor([[1.0, 0.8, 2.0]] * batch, dtype=torch.float64)
    controls = torch.tensor(
        [[[0.3, 0.1, 1.0]] * n_segments] * batch, dtype=torch.float64
    )
    final_time = torch.tensor([300.0, 480.0][:batch], dtype=torch.float64)
    return BasisSchedule(controls, lower, upper, final_time, duration_mode)


@pytest.mark.parametrize("duration_mode", DURATION_MODES)
def test_forward_reproduces_the_seed_and_the_given_duration(duration_mode):
    schedule = _schedule(duration_mode)
    prediction = schedule()
    assert torch.allclose(
        prediction.controls,
        torch.tensor([[[0.3, 0.1, 1.0]] * 4] * 2, dtype=torch.float64),
        atol=1e-9,
    )
    assert torch.allclose(
        prediction.segment_durations.sum(dim=1), schedule.final_time_s, atol=1e-9
    )
    assert torch.equal(prediction.final_time_s, schedule.final_time_s)


def test_uniform_splits_evenly_and_free_starts_uniform_but_is_a_parameter():
    uniform = _schedule(DURATION_UNIFORM)()
    assert torch.allclose(
        uniform.segment_durations[0],
        torch.full((4,), 300.0 / 4, dtype=torch.float64),
        atol=1e-9,
    )
    free = _schedule(DURATION_FREE)
    assert free.duration_logits is not None
    assert torch.allclose(
        free().segment_durations[0],
        torch.full((4,), 300.0 / 4, dtype=torch.float64),
        atol=1e-9,
    )
    assert _schedule(DURATION_UNIFORM).duration_logits is None


def test_controls_stay_inside_the_box_for_any_logits():
    schedule = _schedule(DURATION_UNIFORM)
    with torch.no_grad():
        schedule.control_logits.copy_(
            torch.full_like(schedule.control_logits, 50.0) * torch.tensor([1.0, -1.0, 1.0])
        )
    controls = schedule().controls
    assert torch.all(controls >= schedule.control_lower.unsqueeze(1) - 1e-9)
    assert torch.all(controls <= schedule.control_upper.unsqueeze(1) + 1e-9)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(initial_controls=torch.zeros(2, 4)), "must be \\[B, N, 3\\]"),
        (dict(final_time_s=torch.tensor([300.0, -1.0], dtype=torch.float64)), "must be positive"),
        (dict(duration_mode="learned"), "unknown duration mode"),
    ],
)
def test_construction_refuses_a_broken_contract(kwargs, message):
    lower = torch.tensor([[-0.2, -0.8, 0.2]] * 2, dtype=torch.float64)
    upper = torch.tensor([[1.0, 0.8, 2.0]] * 2, dtype=torch.float64)
    arguments = dict(
        initial_controls=torch.full((2, 4, 3), 0.5, dtype=torch.float64),
        control_lower=lower,
        control_upper=upper,
        final_time_s=torch.tensor([300.0, 480.0], dtype=torch.float64),
        duration_mode=DURATION_UNIFORM,
    )
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        BasisSchedule(**arguments)


def test_free_number_count_matches_the_two_modes():
    assert free_number_count(8, DURATION_UNIFORM) == 24
    assert free_number_count(8, DURATION_FREE) == 32
    with pytest.raises(ValueError, match="unknown duration mode"):
        free_number_count(8, "learned")


def test_the_fit_keeps_each_flight_s_own_best_step():
    """Flight 0 is best early and flight 1 late; a batch-level best would lose one of them."""
    schedule = _schedule(DURATION_UNIFORM, batch=2, n_segments=2)
    calls = {"step": 0}
    # Objective values per flight, by call index: flight 0 dips at call 1, flight 1 at call 3.
    values = torch.tensor(
        [[5.0, 1.0, 4.0, 4.0, 4.0], [5.0, 5.0, 5.0, 2.0, 6.0]], dtype=torch.float64
    )

    def objective(prediction):
        index = calls["step"]
        calls["step"] += 1
        # Keep a live gradient path so backward() has something to do.
        anchor = prediction.controls.sum(dim=(1, 2)) * 0.0
        return values[:, index] + anchor

    fit = fit_basis_schedules(
        schedule, objective, steps=4,
        control_learning_rate=1e-3, duration_learning_rate=1e-3, gradient_clip_norm=1.0,
    )
    assert torch.allclose(fit.best_value, torch.tensor([1.0, 2.0], dtype=torch.float64))
    assert np.array_equal(fit.best_step, np.array([1, 3]))
    # The seed is step 0's value, before any update, for both flights.
    assert torch.allclose(fit.seed_value, torch.tensor([5.0, 5.0], dtype=torch.float64))
    # steps=4 -> 0.95*4 = 3.8, so only a best step of 4 counts as "still improving".
    assert np.array_equal(fit.still_improving, np.array([False, False]))


def test_the_fit_restores_the_best_parameters_not_the_last():
    schedule = _schedule(DURATION_UNIFORM, batch=1, n_segments=2)
    seen: list[torch.Tensor] = []

    def objective(prediction):
        seen.append(schedule.control_logits.detach().clone())
        # Minimised at the first call, then driven away from it.
        return torch.tensor([float(len(seen) - 1)], dtype=torch.float64) + (
            prediction.controls.sum() * 1e-6
        )

    fit_basis_schedules(
        schedule, objective, steps=3,
        control_learning_rate=1e-1, duration_learning_rate=1e-1, gradient_clip_norm=1.0,
    )
    assert torch.allclose(schedule.control_logits.detach(), seen[0], atol=1e-12)


def test_the_fit_refuses_a_non_finite_objective_and_a_wrong_shape():
    schedule = _schedule(DURATION_UNIFORM, batch=2, n_segments=2)
    with pytest.raises(FloatingPointError, match="non-finite"):
        fit_basis_schedules(
            schedule, lambda p: p.controls.sum(dim=(1, 2)) * float("nan"),
            steps=1, control_learning_rate=1e-3, duration_learning_rate=1e-3,
            gradient_clip_norm=1.0,
        )
    with pytest.raises(ValueError, match="one value per flight"):
        fit_basis_schedules(
            schedule, lambda p: p.controls.sum(),
            steps=1, control_learning_rate=1e-3, duration_learning_rate=1e-3,
            gradient_clip_norm=1.0,
        )


def test_still_improving_flags_a_flight_that_used_the_whole_budget():
    schedule = _schedule(DURATION_UNIFORM, batch=2, n_segments=2)
    calls = {"index": 0}
    values = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float64)

    def objective(prediction):
        index = calls["index"]
        calls["index"] += 1
        return values[:, index] + prediction.controls.sum(dim=(1, 2)) * 0.0

    fit = fit_basis_schedules(
        schedule, objective, steps=4,
        control_learning_rate=1e-3, duration_learning_rate=1e-3, gradient_clip_norm=1.0,
    )
    assert np.array_equal(fit.best_step, np.array([4, 0]))
    assert np.array_equal(fit.still_improving, np.array([True, False]))


def test_the_gradient_clip_is_per_flight_not_joint():
    """A joint clip would throttle flight 0 because flight 1's gradient is huge."""
    parameter = torch.nn.Parameter(torch.zeros(2, 2, 3, dtype=torch.float64))
    parameter.grad = torch.zeros_like(parameter)
    parameter.grad[0] = 1.0        # per-flight norm sqrt(6) ~ 2.449, under the cap
    parameter.grad[1] = 1000.0     # far over it
    norms = clip_gradients_per_flight([parameter], 10.0)
    assert torch.allclose(norms, torch.tensor([6.0, 6e6], dtype=torch.float64).sqrt())
    assert torch.allclose(parameter.grad[0], torch.ones(2, 3, dtype=torch.float64))
    assert float(parameter.grad[1].square().sum().sqrt()) == pytest.approx(10.0, rel=1e-9)


def test_the_clipped_share_counts_flight_steps():
    schedule = _schedule(DURATION_UNIFORM, batch=2, n_segments=2)
    # A large constant-gradient objective: both flights clip at every one of the 3 updates.
    fit = fit_basis_schedules(
        schedule, lambda p: p.controls.sum(dim=(1, 2)) * 1e6,
        steps=3, control_learning_rate=1e-6, duration_learning_rate=1e-6,
        gradient_clip_norm=1e-6,
    )
    assert fit.clipped_share == pytest.approx(1.0)
    assert fit.steps == 3
