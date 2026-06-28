"""Teaching scaffold: optimize flight scenarios, dump optimizer-vs-simulator states.

For each :class:`FlightScenario` (from the ``flight_scenarios`` package) this:

  1. solves the trajectory optimization from the scenario's initial state to its target,
  2. rolls the optimizer's piecewise-constant controls forward through the **real**
     simulator (the same dynamics the live sim uses).

It writes **one JSON per scenario** with both state sequences:

    { source, final_time_s,
      optimizer_states: [ {t, lat, lon, alt, V, psi, gamma, m}, … ],   # the NLP's plan
      simulator_states: [ {t, lat, lon, alt, V, psi, gamma, m}, … ] }  # the real rollout

The two sequences are NOT identical — the optimizer's node states are the idealized plan,
the simulator states are what the dynamics actually do under those controls. The gap is the
point of the exercise (and what the CZML builder in ``aeroviz-4d/python`` visualizes).

This is a **teaching scaffold**: the loop / IO / serialization / CLI are wired; the two core
steps — running the optimizer (TODO ①) and the forward rollout (TODO ②) — are documented
TODOs. See ``flight_scenarios/README.md`` and the comments below.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_OPT_DIR = Path(__file__).resolve().parent
if str(_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPT_DIR))

from flight_scenarios import FlightScenario, load_scenarios  # noqa: E402
from aerodynamic_model.common import GeodeticState, LoadFactorControl  # noqa: E402
from aerodynamic_model.casadi_simulator import CasadiSimulator  # noqa: E402
from aerodynamic_model.rollout import rollout_piecewise_constant  # noqa: E402
from casadi_direct_collocation_optimizer import CasadiDirectCollocationOptimizer  # noqa: E402

# Optimizer + rollout defaults (override on the CLI).
DEFAULT_N_SEGMENTS = 10
DEFAULT_DT = 1.0                 # optimizer dt (API parity; state mesh is auto-selected)
DEFAULT_MAX_DURATION_S = 1000.0  # free-time upper bound; the solver minimises T below this
DEFAULT_ROLLOUT_DT_S = 0.5       # forward-integration step for the simulator rollout


# ── Output records (plumbing) ─────────────────────────────────────────────────

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "final_time_s": self.final_time_s,
            "optimizer_states": [asdict(s) for s in self.optimizer_states],
            "simulator_states": [asdict(s) for s in self.simulator_states],
        }


# ── The core: optimize one scenario into two state sequences ──────────────────

def optimize_scenario(
    scenario: FlightScenario,
    *,
    n_segments: int = DEFAULT_N_SEGMENTS,
    dt: float = DEFAULT_DT,
    max_duration: float = DEFAULT_MAX_DURATION_S,
    rollout_dt_s: float = DEFAULT_ROLLOUT_DT_S,
) -> ScenarioOptimization:
    """Optimize ``scenario`` and return its optimizer + simulator state sequences."""
    initial = scenario.initial
    target = scenario.target
    aircraft = scenario.aircraft
    if target is None:
        raise ValueError(
            "scenario has no target state; build scenarios with flight_scenarios (its "
            "build_scenario populates target) before optimizing."
        )

    # ── TODO ① — run the optimizer (initial -> target) ────────────────────────────
    # Construct the direct-collocation optimizer and solve the free-time NLP:
    #
    optimizer = CasadiDirectCollocationOptimizer(n_segments, dt, max_duration, aircraft)
    final_time, node_control, node_state = optimizer.optimize_free_time(
        initial, target, max_duration)
    #   • node_state  rows are [lat, lon, alt, V, psi, gamma]   — the optimizer's plan
    #   • node_control rows are [thrust_N, bank_rad, load_factor]
    #
    # Then build the two sequences (the helpers below do the work) and return:
    #
    optimizer_states = _node_states_to_samples(node_state, final_time, initial.m)
    simulator_states = simulate_controls(initial, node_control, final_time, aircraft,
                                             dt=rollout_dt_s)
    return ScenarioOptimization(scenario.source, float(final_time),
                                   optimizer_states, simulator_states)


def _node_states_to_samples(
    node_state: Any, final_time: float, mass: float
) -> list[StateSample]:
    """Reshape the optimizer's node states into timed samples (provided plumbing).

    The boundary nodes are evenly spaced in time over ``[0, final_time]``.
    """
    states = list(node_state)
    count = len(states)
    samples: list[StateSample] = []
    for index, values in enumerate(states):
        lat, lon, alt, V, psi, gamma = (float(v) for v in values)
        t = (index / (count - 1)) * final_time if count > 1 else 0.0
        samples.append(StateSample(t=t, lat=lat, lon=lon, alt=alt, V=V, psi=psi, gamma=gamma, m=mass))
    return samples


def simulate_controls(
    initial_state: GeodeticState,
    node_control: Any,
    final_time: float,
    aircraft: Any,
    *,
    dt: float = DEFAULT_ROLLOUT_DT_S,
) -> list[StateSample]:
    """Roll the piecewise-constant optimizer controls through the REAL simulator.

    This produces the "simulator real states": the optimizer's own controls integrated
    through the actual dynamics, which differ from the optimizer's node states (the plan).

    Thin adapter over ``aerodynamic_model.rollout_piecewise_constant`` — builds the
    load-factor controls + the simulator, runs the shared rollout (truncating silently if
    the replay leaves the envelope — this is a viz aid), and maps each neutral
    ``(t, state)`` sample onto a serializable :class:`StateSample`.
    """
    controls = [
        LoadFactorControl(thrust=float(row[0]), bank_rad=float(row[1]),
                          load_factor=float(row[2]))
        for row in node_control
    ]
    sim = CasadiSimulator(aircraft, dt)
    samples = rollout_piecewise_constant(
        sim, initial_state, controls, final_time,
        integrator_dt=dt, truncate_on_envelope_exit=True,
    )
    return [StateSample.from_state(s.t, s.state) for s in samples]

# ── Batch + IO (wired) ────────────────────────────────────────────────────────

def optimize_scenarios(
    scenarios: list[FlightScenario],
    *,
    output_dir: str | Path,
    n_segments: int = DEFAULT_N_SEGMENTS,
    dt: float = DEFAULT_DT,
    max_duration: float = DEFAULT_MAX_DURATION_S,
    rollout_dt_s: float = DEFAULT_ROLLOUT_DT_S,
) -> list[Path]:
    """Optimize each scenario and write one ``*_states.json`` per scenario."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, scenario in enumerate(scenarios):
        result = optimize_scenario(
            scenario,
            n_segments=n_segments,
            dt=dt,
            max_duration=max_duration,
            rollout_dt_s=rollout_dt_s,
        )
        path = out / _scenario_filename(scenario, index)
        path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        written.append(path)
        print(
            f"✓ {path.name}: optimizer {len(result.optimizer_states)} states, "
            f"simulator {len(result.simulator_states)} states, T={result.final_time_s:.1f}s"
        )
    return written


def _scenario_filename(scenario: FlightScenario, index: int) -> str:
    flight_id = scenario.source.get("id") or f"scenario{index}"
    runway = scenario.source.get("runway")
    base = f"{flight_id}_{runway}" if runway else str(flight_id)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return f"{safe}_states.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize flight scenarios -> state JSON files")
    parser.add_argument("--scenarios", required=True, help="Scenario JSON from flight_scenarios")
    parser.add_argument("--output-dir", required=True, help="Where to write the *_states.json files")
    parser.add_argument("--n-segments", type=int, default=DEFAULT_N_SEGMENTS)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION_S)
    parser.add_argument("--rollout-dt", type=float, default=DEFAULT_ROLLOUT_DT_S)
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenarios)
    paths = optimize_scenarios(
        scenarios,
        output_dir=args.output_dir,
        n_segments=args.n_segments,
        dt=args.dt,
        max_duration=args.max_duration,
        rollout_dt_s=args.rollout_dt,
    )
    print(f"✓ wrote {len(paths)} state file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
