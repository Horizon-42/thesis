"""Where the truth lives in each longitudinal coordinate (design §2.4). Read-only.

For every flight of a control prediction directory: the observed track from its anchor,
through the package's own lagged-command inversion (``commanded_controls``, τ = the
simple-v3 constants) sampled at N uniform midpoints exactly as ``segment_controls`` samples
it — once in thrust fraction, once in specific force — plus the specific-force interval the
thrust clamp admits at each of those states, ``[(-0.2·T_max - D)/W, (T_max - D)/W]``.
It prints the percentiles of all four, how often each teacher leaves its box, and the share
of the n_x teacher a candidate box would cut.

    conda activate aeroviz
    python 4dTrajectory/ts_transformer/docs/specific_force_teacher_distribution.py \
        4dTrajectory/outputs/KRDU/experiments/b1_quantile_20260907/B1_point_matched_pred_val
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "4dTrajectory"))
from aerodynamic_model.torch_dynamics import GRAVITY_MPS2  # noqa: E402
from aircraft.aero_params import aero_params_for_aircraft  # noqa: E402
from flight_scenarios.scenario import aircraft_for_code, aircraft_provider_of  # noqa: E402
from ts_transformer.config import (  # noqa: E402
    CONTROL_SPECIFIC_FORCE,
    CONTROL_THRUST_FRACTION,
    TSConfig,
    recipe_settings,
)
from ts_transformer.outputs.dynamics.inverse import _drag_force, commanded_controls  # noqa: E402
from ts_transformer.outputs.envelope import (  # noqa: E402
    MAX_THRUST_FRACTION,
    MIN_THRUST_FRACTION,
    SPECIFIC_FORCE_CONTRACT,
)

# simple-v3's (thrust, bank, load) time constants — the recipe the readout arm trained under.
TAU = np.array(TSConfig(**recipe_settings("simple-v3", keep_name=True)).control_time_constants_s)
N = 32
PERCENTILES = (0.1, 0.5, 1, 5, 50, 95, 99, 99.5, 99.9)


def flight_rows(pred_dir: Path, row: dict) -> tuple[np.ndarray, ...]:
    ev = json.loads((pred_dir / row["eval_file"]).read_text())
    st = json.loads((pred_dir / row["states_file"]).read_text())
    aircraft = aircraft_for_code(ev["source"]["dynamics_typecode"],
                                 provider=aircraft_provider_of(ev["source"]["dynamics_source"]))
    aero = aero_params_for_aircraft(aircraft)
    aero_row = np.array([aero.S, aero.Cl_max, aero.Cd0, aero.k, aero.stall_threshold, aero.k_stall])
    max_thrust = aircraft.engine.max_thrust_total_n
    observed = [s for s in st["observed_states"] if s["t"] >= 0.0]
    times = np.array([s["t"] for s in observed])
    states = np.array([[s["lat"], s["lon"], s["alt"], s["V"], s["psi"], s["gamma"], s["m"]]
                       for s in observed])
    midpoints = (np.arange(N) + 0.5) * (times[-1] / N)
    kwargs = dict(aero_params=aero_row, max_thrust_n=max_thrust, time_constants_s=TAU)
    delta = commanded_controls(states, times, parameterization=CONTROL_THRUST_FRACTION, **kwargs)
    n_x = commanded_controls(states, times, parameterization=CONTROL_SPECIFIC_FORCE, **kwargs)
    drag = _drag_force(states[:, 2], np.maximum(states[:, 3], 1e-3), states[:, 6], delta[:, 2], aero_row)
    weight = states[:, 6] * GRAVITY_MPS2
    floor = (MIN_THRUST_FRACTION * max_thrust - drag) / weight
    ceiling = (MAX_THRUST_FRACTION * max_thrust - drag) / weight
    sample = lambda values: np.interp(midpoints, times, values)  # noqa: E731
    return sample(delta[:, 0]), sample(n_x[:, 0]), sample(floor), sample(ceiling)


def main(pred_dir: Path) -> None:
    summary = json.loads((pred_dir / "summary.json").read_text())
    columns = list(zip(*(flight_rows(pred_dir, row) for row in summary["results"]
                         if row["status"] == "solved")))
    delta, n_x, floor, ceiling = (np.concatenate(column) for column in columns)
    print(f"flights {len(columns[0])}, segments {len(delta)} (N = {N})")
    for name, values in (("delta teacher", delta), ("n_x teacher (g)", n_x),
                         ("feasible floor", floor), ("feasible ceiling", ceiling)):
        print(f"{name:>17}: " + "  ".join(
            f"p{q:g} {v:+.4f}" for q, v in zip(PERCENTILES, np.percentile(values, PERCENTILES))
        ))
    print(f"delta teacher outside [{MIN_THRUST_FRACTION}, {MAX_THRUST_FRACTION}]: "
          f"{100 * np.mean((delta < MIN_THRUST_FRACTION) | (delta > MAX_THRUST_FRACTION)):.2f} %")
    print(f"n_x teacher outside its own feasible interval: "
          f"{100 * np.mean((n_x < floor) | (n_x > ceiling)):.2f} %")
    lower, upper = SPECIFIC_FORCE_CONTRACT.lower[0], SPECIFIC_FORCE_CONTRACT.upper[0]
    print(f"states whose feasible floor lies below the box floor {lower}: "
          f"{100 * np.mean(floor < lower):.2f} %")
    print(f"n_x teacher outside the box [{lower}, {upper}]: "
          f"{100 * np.mean((n_x < lower) | (n_x > upper)):.2f} % "
          f"(below {100 * np.mean(n_x < lower):.2f} %)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
