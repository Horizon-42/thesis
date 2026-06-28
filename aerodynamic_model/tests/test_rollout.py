import unittest

from aerodynamic_model.common import GeodeticState, LoadFactorControl  # noqa: E402
from aerodynamic_model.rollout import (  # noqa: E402
    RolloutSample,
    rollout_piecewise_constant,
)


def _state(t_marker: float = 0.0) -> GeodeticState:
    # A neutral placeholder state; only `altitude` is mutated by the fake steppers.
    return GeodeticState(latitude=0.0, longitude=0.0, altitude=t_marker,
                         V=100.0, psi=0.0, gamma=0.0, m=60000.0)


class _CountingSimulator:
    """Stepper that drops altitude by the control's load_factor each step."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def step(self, state: GeodeticState, control, dt: float) -> GeodeticState:
        self.calls.append(dt)
        nxt = _state(state.altitude - control.load_factor)
        return nxt


class _DivergingSimulator:
    """Raises after `fail_after` steps to mimic leaving the flight envelope."""

    def __init__(self, fail_after: int) -> None:
        self.fail_after = fail_after
        self.count = 0

    def step(self, state: GeodeticState, control, dt: float) -> GeodeticState:
        self.count += 1
        if self.count > self.fail_after:
            raise ValueError("below ground")
        return _state(state.altitude - 1.0)


def _controls(n: int) -> list[LoadFactorControl]:
    return [LoadFactorControl(thrust=1000.0 * (i + 1), bank_rad=0.0, load_factor=1.0)
            for i in range(n)]


class TestRolloutPiecewiseConstant(unittest.TestCase):
    def test_every_step_emitted_with_clamped_substep(self):
        sim = _CountingSimulator()
        # 2 segments over 4 s, integrator_dt 1 s, emit every step.
        samples = rollout_piecewise_constant(
            sim, _state(), _controls(2), 4.0, integrator_dt=1.0)
        times = [round(s.t, 6) for s in samples]
        self.assertEqual(times, [0.0, 1.0, 2.0, 3.0, 4.0])
        # initial sample + one per step.
        self.assertEqual(len(samples), 5)
        self.assertEqual(sim.calls, [1.0, 1.0, 1.0, 1.0])
        # last sample lands exactly at total_duration.
        self.assertAlmostEqual(samples[-1].t, 4.0)

    def test_segment_index_and_control_tracked(self):
        sim = _CountingSimulator()
        controls = _controls(2)
        samples = rollout_piecewise_constant(
            sim, _state(), controls, 4.0, integrator_dt=1.0)
        # initial sample is attributed to segment 0 / controls[0].
        self.assertEqual(samples[0].segment_index, 0)
        self.assertIs(samples[0].control, controls[0])
        # second segment's samples carry segment_index 1 / controls[1].
        seg1 = [s for s in samples if s.t > 2.0 + 1e-9]
        self.assertTrue(seg1)
        self.assertTrue(all(s.segment_index == 1 for s in seg1))
        self.assertTrue(all(s.control is controls[1] for s in seg1))

    def test_substep_clamped_to_segment_boundary(self):
        sim = _CountingSimulator()
        # 1 segment over 2.5 s with 1 s steps -> 1.0, 1.0, 0.5 (clamped).
        rollout_piecewise_constant(sim, _state(), _controls(1), 2.5, integrator_dt=1.0)
        self.assertEqual([round(c, 6) for c in sim.calls], [1.0, 1.0, 0.5])

    def test_output_dt_decouples_sampling_from_integration(self):
        sim = _CountingSimulator()
        # 1 segment over 4 s, integrate at 1 s, sample at 2 s + boundary.
        samples = rollout_piecewise_constant(
            sim, _state(), _controls(1), 4.0, integrator_dt=1.0, output_dt=2.0)
        times = [round(s.t, 6) for s in samples]
        self.assertEqual(times, [0.0, 2.0, 4.0])
        # integration still happened at the fine grid.
        self.assertEqual(sim.calls, [1.0, 1.0, 1.0, 1.0])

    def test_truncate_returns_partial_and_logs(self):
        sim = _DivergingSimulator(fail_after=2)
        logged: list[tuple[float, str]] = []
        samples = rollout_piecewise_constant(
            sim, _state(), _controls(1), 5.0, integrator_dt=1.0,
            truncate_on_envelope_exit=True,
            on_truncate=lambda t, exc: logged.append((t, str(exc))))
        # initial + 2 good steps, then truncated before the failing 3rd.
        self.assertEqual(len(samples), 3)
        self.assertEqual(len(logged), 1)
        self.assertAlmostEqual(logged[0][0], 2.0)

    def test_envelope_exit_raises_by_default(self):
        sim = _DivergingSimulator(fail_after=1)
        with self.assertRaises(ValueError):
            rollout_piecewise_constant(
                sim, _state(), _controls(1), 5.0, integrator_dt=1.0)

    def test_empty_controls_rejected(self):
        with self.assertRaises(ValueError):
            rollout_piecewise_constant(_CountingSimulator(), _state(), [], 5.0,
                                       integrator_dt=1.0)

    def test_returns_rollout_samples(self):
        sim = _CountingSimulator()
        samples = rollout_piecewise_constant(
            sim, _state(), _controls(1), 1.0, integrator_dt=1.0)
        self.assertTrue(all(isinstance(s, RolloutSample) for s in samples))


if __name__ == "__main__":
    unittest.main()
