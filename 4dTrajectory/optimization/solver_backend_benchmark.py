"""Benchmark the NLP solver backends (ipopt vs sqpmethod) on the same problems.

The direct-collocation optimiser solves a nonlinear program (NLP).  By default
it uses IPOPT (interior point) — robust, but warm-starts poorly.  CasADi also
ships an SQP method (``sqpmethod`` + an active-set QP), which can be faster on
benign / warm-started solves but is typically less robust on hard cold starts.

This script measures, for each backend on the SAME problems:
  * cold solve wall-time (median of a few repeats, after a warm-up),
  * whether it converged,
  * terminal accuracy (node → target).

Two problem flavours:
  * feasible  – an on-manifold target (forward-propagated nominal control); both
                backends should converge,
  * aggressive – a steep near-stall straight-in (the RW05L-style case that
                stresses the solver); tests robustness, not just speed.

Run:
    python -m solver_backend_benchmark
    python -m solver_backend_benchmark --aircraft A320 --horizon 90 --repeats 3
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
import time
from pathlib import Path

import casadi as ca
import numpy as np

from geokit import kt_to_ms

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_OPT_DIR = Path(__file__).resolve().parent
if str(_OPT_DIR) not in sys.path:
    sys.path.insert(0, str(_OPT_DIR))

from aircraft.aircraft_sets import A320, C172  # noqa: E402
from aerodynamic_model.common import GeodeticState  # noqa: E402
from aerodynamic_model.casadi_simulator import (  # noqa: E402
    aero_params_for_aircraft, make_geodetic_step_integrator,
)
from casadi_direct_collocation_optimizer import (  # noqa: E402
    CasadiDirectCollocationOptimizer, _SOLVER_BACKENDS,
)

_AIRCRAFT = {"A320": A320, "C172": C172}
_R = 6_371_000.0


def _horiz_m(lat, lon, ref):
    return _R * math.hypot(
        math.radians(lat - ref.latitude),
        math.radians(lon - ref.longitude) * math.cos(math.radians(ref.latitude)),
    )


def _feasible_target(aircraft, init, horizon):
    step = make_geodetic_step_integrator(transport="approx")["step_func"]
    ap = aero_params_for_aircraft(aircraft)
    aero = ca.DM([ap.S, ap.Cl_max, ap.Cd0, ap.k, ap.stall_threshold, ap.k_stall])
    u = ca.DM([aircraft.approach.thrust_guess_n, 0.0, 1.0])
    x = ca.DM([init.latitude, init.longitude, init.altitude,
               init.V, init.psi, init.gamma, init.m])
    for _ in range(int(horizon / 0.05)):
        x = step(x_geo=x, u=u, aero_params=aero, dt=0.05)["x_geo_next"]
    a = np.array(x).reshape(-1)
    return GeodeticState(a[0], a[1], a[2], a[3], a[4], a[5], aircraft.mass.max_takeoff_kg)


def _time_solve(fn, repeats):
    """Warm up once (discarded), then return (median_seconds, last_result)."""
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            fn()
        except ValueError:
            pass
    times = []
    result = None
    for _ in range(repeats):
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            try:
                result = ("ok", fn())
            except ValueError as exc:
                result = ("fail", str(exc).split(": ")[-1][:28])
            times.append(time.perf_counter() - t0)
    return float(np.median(times)), result


def _row(label, backend, secs, result, target):
    if result[0] == "ok":
        _, controls, states = result[1]
        miss = _horiz_m(states[-1][0], states[-1][1], target)
        status = f"converged  miss={miss:6.2f} m"
    else:
        status = f"FAILED ({result[1]})"
    print(f"  {label:<22}{backend:<11}{secs * 1000:9.0f} ms   {status}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aircraft", choices=sorted(_AIRCRAFT), default="A320")
    parser.add_argument("--n-segments", type=int, default=10)
    parser.add_argument("--horizon", type=float, default=90.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    aircraft = _AIRCRAFT[args.aircraft]
    init = GeodeticState(35.60, -78.50, 1500.0,
                         kt_to_ms(aircraft.approach.reference_speed_kt) + 12.0,
                         math.radians(40.0), math.radians(-3.0), aircraft.mass.max_takeoff_kg)
    feasible = _feasible_target(aircraft, init, args.horizon)
    # Aggressive: a steep straight-in to a near-stall terminal speed.
    aggressive = GeodeticState(
        feasible.latitude, feasible.longitude, init.altitude - 1300.0,
        kt_to_ms(aircraft.approach.reference_speed_kt), init.psi, math.radians(-4.0),
        aircraft.mass.max_takeoff_kg,
    )

    print(f"\nSolver-backend benchmark — aircraft={args.aircraft}, N={args.n_segments}, "
          f"horizon={args.horizon:.0f}s, repeats={args.repeats}")
    print(f"  {'problem':<22}{'backend':<11}{'solve':>9}      result")
    print("  " + "-" * 64)

    for backend in _SOLVER_BACKENDS:
        opt = CasadiDirectCollocationOptimizer(
            n_segments=args.n_segments, dt=0.2, max_duration=args.horizon * 1.6,
            aircraft=aircraft, solver_backend=backend,
        )
        secs, res = _time_solve(
            lambda o=opt: o.optimize_trajectory(init, feasible, duration=args.horizon),
            args.repeats)
        _row("feasible (fixed-T)", backend, secs, res, feasible)
        secs, res = _time_solve(
            lambda o=opt: o.optimize_free_time(init, feasible, args.horizon * 1.6),
            args.repeats)
        _row("feasible (free-T)", backend, secs, res, feasible)
        secs, res = _time_solve(
            lambda o=opt: o.optimize_trajectory(init, aggressive, duration=args.horizon),
            args.repeats)
        _row("aggressive (fixed-T)", backend, secs, res, aggressive)
        print("  " + "-" * 64)

    print("\nNote: every solve is COLD (fresh initial guess) — this is the current")
    print("frontend usage.  SQP's warm-start advantage (feeding the previous solution)")
    print("is NOT exercised here; with warm starts sqpmethod would look better still.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
