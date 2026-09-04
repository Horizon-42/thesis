"""The rollout's command hook: identity is bit-exact, a hook is differentiable, and
backends without actuator states refuse it."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TS_DIR.parents[1]
for path in (TS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    CONTROL_DYNAMICS_FIRST_ORDER_LAG, CONTROL_DYNAMICS_POINT_MASS, PREDICTION_CONTROL, TSConfig,
)
from control.dynamics import rollout as control_rollout  # noqa: E402
from control.dynamics.hooks import RolloutStateView  # noqa: E402
from control.envelope import CONTROL_LOWER, CONTROL_UPPER  # noqa: E402
from dataset import build_series, dynamics_arrays  # noqa: E402
from synthetic import synthetic_arrivals  # noqa: E402

AIRPORT, RUNWAY = "KRDU", "05L"


def _batch(config: TSConfig, n_flights: int = 3, segments: int = 6):
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=5)
    series, _ = build_series(flights, config, airport=AIRPORT)
    anchor = config.seq_len - 1
    rows = [dynamics_arrays(item, anchor) for item in series]
    dynamics = {key: torch.from_numpy(np.stack([row[key] for row in rows])) for key in rows[0]}
    generator = torch.Generator().manual_seed(11)
    lower, upper = torch.tensor(CONTROL_LOWER, dtype=torch.float32), torch.tensor(CONTROL_UPPER, dtype=torch.float32)
    unit = torch.rand((len(series), segments, 3), generator=generator)
    controls = (lower + unit * (upper - lower) * 0.5 + 0.25 * (upper - lower)).requires_grad_(True)
    durations = torch.full((len(series), segments), 8.0)
    return dynamics, controls, durations


class _Identity:
    needs_reference = False

    def __init__(self):
        self.calls = 0

    def __call__(self, state: RolloutStateView, command: torch.Tensor, segment_index: int):
        assert state.chart.shape[-1] == 7 and state.actuators is not None
        assert state.actuators.shape == command.shape
        self.calls += 1
        return command

    def diagnostics(self):
        return {"calls": torch.tensor(float(self.calls))}


class _StateDependentBank:
    needs_reference = False

    """Add a bank proportional to the cross-track position: a hook that READS the state."""

    def __call__(self, state: RolloutStateView, command: torch.Tensor, segment_index: int):
        bank = 1e-4 * state.chart[:, 1]                    # from n (metres) → radians
        return torch.stack([command[:, 0], command[:, 1] + bank, command[:, 2]], dim=-1)

    def diagnostics(self):
        return {}


def _lag_config() -> TSConfig:
    return TSConfig(prediction_output=PREDICTION_CONTROL, control_dynamics_model=CONTROL_DYNAMICS_FIRST_ORDER_LAG,
                    control_dynamics_backend="scaled-transport-chart-velocity", seq_len=8, n_segments=6)


def test_identity_hook_is_bit_exact_on_endpoint_and_dense_rollouts():
    config = _lag_config()
    dynamics, controls, durations = _batch(config)
    plain = control_rollout.rollout_control_endpoints(controls, durations, dynamics, config)
    hook = _Identity()
    hooked = control_rollout.rollout_control_endpoints(controls, durations, dynamics, config, command_hook=hook)
    assert torch.equal(plain.channels, hooked.channels)
    assert torch.equal(plain.geodetic_states, hooked.geodetic_states)
    # The effective schedule is reported in the rollout's dtype (float64).
    assert torch.equal(hooked.controls, controls.to(hooked.controls.dtype))
    assert torch.equal(plain.controls, controls.to(plain.controls.dtype))
    assert hook.calls == controls.shape[1]
    offsets = torch.arange(1, 7, dtype=torch.float64)[None].expand(len(controls), -1) * 5.0
    valid = torch.ones_like(offsets, dtype=torch.bool)
    dense_plain = control_rollout.rollout_control_dense(controls, durations, dynamics, offsets, valid, config)
    dense_hooked = control_rollout.rollout_control_dense(controls, durations, dynamics, offsets, valid, config, command_hook=_Identity())
    assert torch.equal(dense_plain.query_channels, dense_hooked.query_channels)
    assert torch.equal(dense_plain.segment_end_channels, dense_hooked.segment_end_channels)
    assert torch.equal(dense_hooked.controls, controls.to(dense_hooked.controls.dtype))


def test_a_state_reading_hook_changes_the_path_and_stays_differentiable():
    config = _lag_config()
    dynamics, controls, durations = _batch(config)
    hooked = control_rollout.rollout_control_endpoints(controls, durations, dynamics, config, command_hook=_StateDependentBank())
    plain = control_rollout.rollout_control_endpoints(controls, durations, dynamics, config)
    assert not torch.allclose(hooked.channels, plain.channels)
    # The effective schedule differs from the commands from the second segment on (the
    # first segment starts at the anchor, whose n is what it is), and it carries a graph.
    assert not torch.allclose(hooked.controls, controls.to(hooked.controls.dtype))
    hooked.channels[..., :3].square().sum().backward()
    assert controls.grad is not None and torch.count_nonzero(controls.grad) > 0
    # The dense rollout reproduces the same endpoints from the settled schedule.
    controls2 = controls.detach().clone().requires_grad_(True)
    offsets = torch.cumsum(durations, dim=1).to(torch.float64)
    valid = torch.ones_like(offsets, dtype=torch.bool)
    dense = control_rollout.rollout_control_dense(controls2, durations, dynamics, offsets, valid, config, command_hook=_StateDependentBank())
    hooked2 = control_rollout.rollout_control_endpoints(controls2, durations, dynamics, config, command_hook=_StateDependentBank())
    assert torch.allclose(dense.controls, hooked2.controls)
    assert torch.allclose(dense.segment_end_channels[..., :3], hooked2.channels[..., :3], atol=1.0)


def test_point_mass_backends_refuse_a_hook():
    config = TSConfig(prediction_output=PREDICTION_CONTROL, control_dynamics_model=CONTROL_DYNAMICS_POINT_MASS,
                      seq_len=8, n_segments=6)
    dynamics, controls, durations = _batch(config)
    with pytest.raises(NotImplementedError, match="first-order-lag"):
        control_rollout.rollout_control_endpoints(controls, durations, dynamics, config, command_hook=_Identity())
    plain = control_rollout.rollout_control_endpoints(controls, durations, dynamics, config)
    assert torch.equal(plain.controls, controls.to(plain.controls.dtype))
