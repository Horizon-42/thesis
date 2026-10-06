"""A solve's record: the optimizer's plan beside its replay through the real simulator.

The replay (:func:`rollout_controls`) integrates the optimizer's piecewise-constant controls through
the same dynamics the live simulator uses; it is the trajectory ``evaluation`` grades. The plan is
the NLP's own dense node states. Both go out as :class:`StateSample` rows of one
:class:`ScenarioOptimization`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from aerodynamic_model.casadi_simulator import CasadiSimulator
from aerodynamic_model.common import GeodeticState, LoadFactorControl
from aerodynamic_model.rollout import RolloutSample, rollout_piecewise_constant
from collocation.components import altitude_floor_m
from evaluation_export import STATE_DECIMALS
from optimization_run_config import DEFAULT_ROLLOUT_DT_S

# The replay ground guard sits this far BELOW the NLP's altitude floor. The guard exists
# to truncate DIVERGED replays (tens of metres to kilometres below the floor); but
# min-time plans deliberately RIDE the floor, and a faithful replay oscillates
# centimetres around it (measured: a 3.9 cm dip on a floor-riding HS solve whose
# unguarded replay landed 0.7 m from the target — a zero-margin guard cut that same
# replay 10 km short and failed 97% of an unconstrained batch). 5 m is two orders above
# that noise and far below any real divergence; the terminal evaluation is applied only
# to the final threshold event and is unaffected by where this mid-flight guard sits.
ROLLOUT_GUARD_MARGIN_M = 5.0


def rollout_guard_altitude_m(target_altitude_m: float) -> float:
    """The replay truncation altitude for a solve flying to ``target_altitude_m``:
    the NLP's own floor minus :data:`ROLLOUT_GUARD_MARGIN_M`."""
    return altitude_floor_m(target_altitude_m) - ROLLOUT_GUARD_MARGIN_M


def _quantized_sample(sample: "StateSample") -> dict[str, float]:
    """One ``StateSample`` as a JSON row at the record contract's serialized precision."""
    return {
        key: round(value, STATE_DECIMALS[key]) if key in STATE_DECIMALS else value
        for key, value in asdict(sample).items()
    }


@dataclass
class StateSample:
    t: float
    lat: float
    lon: float
    alt: float
    V: float
    psi: float
    gamma: float
    m: float

    @classmethod
    def from_state(cls, t: float, state: GeodeticState) -> "StateSample":
        return cls(
            t=t,
            lat=state.latitude,
            lon=state.longitude,
            alt=state.altitude,
            V=state.V,
            psi=state.psi,
            gamma=state.gamma,
            m=state.m,
        )


@dataclass
class ScenarioOptimization:
    source: dict[str, Any]
    final_time_s: float
    optimizer_states: list[StateSample]
    simulator_states: list[StateSample]
    # The neutral evaluation-input record (evaluation_export.evaluation_record) —
    # written to its own *_eval.json by the batch, NOT part of the states file.
    evaluation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        # Both arrays go out at the record contract's serialized precision (0.56 mm worst
        # case, 30% smaller — see evaluation_export.STATE_DECIMALS). `simulator_states` is
        # also what the eval record's `states_ref` resolves to, so the two views of the
        # rollout have to be written the same way or they would disagree in the last digits.
        return {
            "source": self.source,
            # `t` is deliberately NOT quantized (see evaluation_export.STATE_DECIMALS), so
            # this header and the eval record's `final_time_s` — both read off the same
            # rollout — agree exactly without either side rounding.
            "final_time_s": self.final_time_s,
            "optimizer_states": [_quantized_sample(s) for s in self.optimizer_states],
            "simulator_states": [_quantized_sample(s) for s in self.simulator_states],
        }


def node_states_to_samples(
    node_state: Any, times: list[float], mass: float
) -> list[StateSample]:
    """Reshape the optimizer's node states into timed samples.

    ``times`` aligns 1:1 with ``node_state`` and comes from the optimizer's own
    ``last_dense_state_times_s`` (prefixed with the caller's t=0 initial state).
    Multiphase solves have per-phase node spacing — free per-phase durations AND
    per-phase auto substep counts — so spreading the nodes evenly over the horizon
    time-warped every constrained plan export.
    """
    samples: list[StateSample] = []
    for values, t in zip(node_state, times, strict=True):
        lat, lon, alt, V, psi, gamma = (float(v) for v in values)
        samples.append(StateSample(t=float(t), lat=lat, lon=lon, alt=alt, V=V,
                                   psi=psi, gamma=gamma, m=mass))
    return samples


class _GroundCheckedSimulator:
    """``CasadiSimulator`` plus a ground guard (the raw simulator has NO envelope checks).

    A replay stepping below ``min_altitude_m`` raises, so the shared rollout TRUNCATES
    (its envelope handling) instead of recording subterranean samples — a diverged
    replay used to record kilometres below sea level. Callers pass
    :func:`rollout_guard_altitude_m` (the NLP's floor minus a divergence margin), NOT
    the floor itself: plans ride the floor, and a zero-margin guard truncates faithful
    replays on centimetre-scale integration noise.
    """

    def __init__(self, simulator: CasadiSimulator, min_altitude_m: float) -> None:
        self._simulator = simulator
        self._min_altitude_m = float(min_altitude_m)

    def step(self, state: GeodeticState, control: Any, dt: float) -> GeodeticState:
        next_state = self._simulator.step(state, control, dt)
        if next_state.altitude < self._min_altitude_m:
            raise ValueError(
                f"altitude {next_state.altitude:.1f} m below the trajectory floor "
                f"{self._min_altitude_m:.1f} m"
            )
        return next_state


def rollout_controls(
    initial_state: GeodeticState,
    node_control: Any,
    final_time: float,
    aircraft: Any,
    *,
    dt: float = DEFAULT_ROLLOUT_DT_S,
    segment_durations: Any = None,
    min_altitude_m: float,
) -> list[RolloutSample]:
    """Roll the piecewise-constant optimizer controls through the REAL simulator.

    This produces the "simulator real states": the optimizer's own controls integrated
    through the actual dynamics, which differ from the optimizer's node states (the plan).

    Thin adapter over ``aerodynamic_model.rollout_piecewise_constant`` — builds the
    load-factor controls + the simulator and runs the shared rollout (truncating where the
    replay leaves the envelope; ``require_usable_rollout`` refuses an empty one). Returns the RAW rollout
    samples: each carries its state AND the control active at that time, which is what
    the evaluation export needs (aligned state/control lists). ``segment_durations``
    (one per control) drives the multiphase non-uniform schedule; ``None`` = equal segments.
    ``min_altitude_m`` truncates a replay that descends below it — REQUIRED: solve
    replays pass ``rollout_guard_altitude_m(target)``; a target-less replay states
    ``0.0`` (sea level) explicitly. It used to default to 0.0, which never fires for
    an elevated-airport target — a caller that forgot it silently recorded diverged
    replays kilometres below the field as valid rollouts.
    """
    controls = [
        LoadFactorControl(thrust=float(row[0]), bank_rad=float(row[1]),
                          load_factor=float(row[2]))
        for row in node_control
    ]
    sim = _GroundCheckedSimulator(CasadiSimulator(aircraft, dt), min_altitude_m)
    return rollout_piecewise_constant(
        sim, initial_state, controls, final_time,
        integrator_dt=dt,
        segment_durations=list(segment_durations) if segment_durations is not None else None,
        truncate_on_envelope_exit=True,
    )


def require_usable_rollout(samples: list[RolloutSample]) -> list[RolloutSample]:
    """A rollout truncated before its first full step (envelope exit at t=0) has
    no usable trajectory — fail the scenario loudly instead of exporting a
    degenerate one-sample "solved" record (zero horizontal extent, which nothing
    downstream can arc-length match)."""
    if len(samples) < 2:
        raise ValueError(
            "control rollout exited the flight envelope at its first step — "
            "no usable trajectory"
        )
    return samples
