"""dynamics_comparison.py — reusable core for the 4-way point-mass dynamics study.

Flies ONE trajectory (one start state + one *constant* control) forward for a
given horizon and integrates it four different ways, so the differences are
MODEL/frame differences, not discretisation noise (all share the same RK4 step):

  A  fixed local-tangent ENU, anchored at ``anchor_geo``   (make_dynamics_model)
       - one ENU tangent plane for the whole flight; the flat point-mass RHS is
         integrated entirely in ENU and converted back only to record samples.
  B  per-step re-anchored ENU             (make_geo_step_from_enu_integrator)
       - rebuilds the tangent frame every step.  Most faithful discrete
         integrator, so it is the REFERENCE.
  C  full geodetic RHS, WITH transport    (make_geodetic_step_integrator(True))
       - one continuous RHS in (lat, lon, h); curvature + frame-rotation folded
         in.  Matches B to ~mm: validates the RHS.
  D  geodetic RHS, NO transport           (make_geodetic_step_integrator(False))
       - drops the transport terms; isolates how much they matter.

This is the parameterised engine shared by:
  * ``dynamics_comparison_30km.py``      — the documented 30 km study (HTML doc), and
  * ``aeroviz_backend`` /dynamics-comparison/run — the interactive Compare mode.

Both call ``compare_dynamics`` then sample/serialise the full paths their own way.
"""

from __future__ import annotations

import math
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerodynamic_model.casadi_simulator import (  # noqa: E402
    make_dynamics_model,
    rk4_step_expr,
    geodetic_state_to_enu_expr,
    enu_state_to_geodetic_expr,
    make_geo_step_from_enu_integrator,
    make_geodetic_step_integrator,
)

# Mean-earth radius for the small-angle ground-distance metric (matches the
# 30 km study so the two callers report comparable numbers).
_R = 6_371_000.0

SYSTEM_KEYS = ("A", "B", "C", "D")
#: The systems whose error is measured against the reference B.
COMPARED_KEYS = ("A", "C", "D")
REFERENCE_KEY = "B"


def horizontal_error_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Small-angle ground distance between two geodetic points (metres)."""
    return _R * math.hypot(
        math.radians(lat1 - lat2),
        math.radians(lon1 - lon2) * math.cos(math.radians(lat2)),
    )


def heading_error_deg(psi1_rad: float, psi2_rad: float) -> float:
    """Shortest-arc heading difference (degrees), inputs in radians."""
    d = math.degrees(psi1_rad - psi2_rad) % 360.0
    return min(d, 360.0 - d)


# The CasADi integrators are graph-builds (not cheap to recompile every call),
# and they are stateless (all aircraft/aero data is passed at call time), so
# build each once and reuse across calls.  The backend serves on a threading
# HTTP server, so guard the lazy init with a lock (double-checked) — otherwise
# two concurrent first requests both rebuild the graphs.
_INTEGRATORS: dict | None = None
_INTEGRATORS_LOCK = threading.Lock()


def _integrators() -> dict:
    global _INTEGRATORS
    if _INTEGRATORS is None:
        with _INTEGRATORS_LOCK:
            if _INTEGRATORS is None:
                _INTEGRATORS = {
                    "flat_rhs": make_dynamics_model()["rhs_func"],
                    "reanchored": make_geo_step_from_enu_integrator()["step_func"],
                    "geodetic_transport": make_geodetic_step_integrator(include_transport=True)["step_func"],
                    "geodetic_no_transport": make_geodetic_step_integrator(include_transport=False)["step_func"],
                }
    return _INTEGRATORS


@dataclass
class DynamicsComparison:
    """Full (un-decimated) per-step rollout of the four systems.

    ``paths[key]`` is a list of geodetic states ``[lat, lon, alt, V, psi, gamma, m]``
    (lat/lon in degrees, psi/gamma in radians); index ``i`` is shared across keys
    and aligned with ``times_s[i]`` and ``dist_m[i]`` (cumulative ground distance
    along the reference B).
    """

    times_s: list[float]
    dist_m: list[float]
    paths: dict[str, list[list[float]]]

    @property
    def n_samples(self) -> int:
        return len(self.times_s)


def compare_dynamics(
    *,
    start: dict,
    control: tuple[float, float, float],
    aero_params,
    duration_s: float,
    dt_s: float,
    anchor_geo: tuple[float, float, float] | None = None,
    max_range_m: float | None = None,
    stop_below_ground: bool = True,
) -> DynamicsComparison:
    """Integrate the four systems from ``start`` under a constant ``control``.

    Parameters
    ----------
    start
        ``{lat, lon, alt, V, psi, gamma, mass}`` — lat/lon in degrees,
        psi/gamma in radians.
    control
        ``(thrust_N, bank_rad, load_factor)`` — held constant for the whole run.
    aero_params
        ``[S, Cl_max, Cd0, k, stall_threshold, k_stall]`` (see ``aero_params_for_aircraft``).
    duration_s, dt_s
        Horizon and the shared RK4 step.
    anchor_geo
        ``(lat, lon, alt)`` tangent anchor for system A.  Defaults to the start
        point (so the four systems share a common origin and diverge as they fly
        away from it).  The 30 km study passes the target instead.
    max_range_m
        Optional early stop once the reference has flown this ground distance.
    stop_below_ground
        Stop if the reference descends below 0 m (a free-flight safety net).
    """
    integ = _integrators()
    flat_rhs = integ["flat_rhs"]
    reanchored = integ["reanchored"]
    geo_step_funcs = {
        "B": reanchored,
        "C": integ["geodetic_transport"],
        "D": integ["geodetic_no_transport"],
    }

    u = ca.DM([control[0], control[1], control[2]])
    aero = ca.DM(list(aero_params))
    dt = float(dt_s)
    n_steps = max(1, int(round(duration_s / dt)))

    x0 = [
        float(start["lat"]),
        float(start["lon"]),
        float(start["alt"]),
        float(start["V"]),
        float(start["psi"]),
        float(start["gamma"]),
        float(start["mass"]),
    ]

    if anchor_geo is None:
        anchor_geo = (x0[0], x0[1], 0.0)
    ref_geo = ca.DM([anchor_geo[0], anchor_geo[1], anchor_geo[2]])

    def geo_to_enu(xg):  # 7-vec geodetic (deg) -> 7-vec ENU in the fixed frame
        return ca.vertcat(*geodetic_state_to_enu_expr(ca.DM(xg), ref_geo), xg[6])

    def enu_to_geo(xe):  # 7-vec ENU -> 7-vec geodetic (deg)
        arr = np.array(ca.vertcat(*enu_state_to_geodetic_expr(xe, ref_geo), xe[6])).ravel()
        return [float(v) for v in arr]

    def geo_step(which, x):  # one geodetic step for B / C / D
        f = geo_step_funcs[which]
        arr = np.array(f(x_geo=ca.DM(x), u=u, aero_params=aero, dt=dt)["x_geo_next"]).ravel()
        return [float(v) for v in arr]

    paths: dict[str, list[list[float]]] = {k: [list(x0)] for k in SYSTEM_KEYS}
    state = {k: list(x0) for k in "BCD"}
    enu_A = geo_to_enu(x0)  # system A lives in the fixed ENU frame
    dist = [0.0]
    times = [0.0]

    for step in range(n_steps):
        # Compute the next state of every system FIRST, then decide whether to
        # record it.  This lets us reject a step that leaves the envelope —
        # non-finite (a divergent near-singular rollout) or below ground for ANY
        # system — without recording a sub-surface / NaN sample, so the last
        # recorded sample is always on the surface and JSON-clean.
        enu_A_next = rk4_step_expr(flat_rhs, enu_A, u, aero, dt)  # integrate in ENU
        next_states = {"A": enu_to_geo(enu_A_next)}  # convert only to record
        for k in "BCD":
            next_states[k] = geo_step(k, state[k])

        if not all(math.isfinite(v) for s in next_states.values() for v in s):
            break  # divergent rollout produced NaN/inf — truncate (viz aid)
        if stop_below_ground and any(s[2] < 0.0 for s in next_states.values()):
            break  # next step dips below ground — stop on the last surface sample

        enu_A = enu_A_next
        paths["A"].append(next_states["A"])
        for k in "BCD":
            state[k] = next_states[k]
            paths[k].append(next_states[k])
        b, bprev = paths["B"][-1], paths["B"][-2]
        dist.append(dist[-1] + horizontal_error_m(b[0], b[1], bprev[0], bprev[1]))
        times.append((step + 1) * dt)
        if max_range_m is not None and dist[-1] >= max_range_m:
            break

    return DynamicsComparison(times_s=times, dist_m=dist, paths=paths)


def even_sample_indices(n: int, max_samples: int) -> list[int]:
    """Indices into ``0..n-1`` decimated to AT MOST ``max_samples`` points,
    evenly spaced and always including the first (0) and last (n-1)."""
    if n <= 0:
        return []
    if max_samples <= 1 or n <= max_samples:
        return [0] if max_samples <= 1 else list(range(n))
    # Endpoint-inclusive even spacing.  Rounding can collide near the ends, so
    # dedupe via a set; the count is <= max_samples by construction.
    last = n - 1
    return sorted({round(i * last / (max_samples - 1)) for i in range(max_samples)})


def error_series(
    comparison: DynamicsComparison,
    indices: list[int],
) -> dict[str, dict[str, list[float]]]:
    """Per-sample error of A/C/D vs the reference B at the given indices.

    Returns ``{key: {"horiz", "alt", "head", "speed", "fpa"}}`` for keys in
    ``COMPARED_KEYS`` (B is the zero reference and is omitted).  ``head`` and
    ``horiz`` are magnitudes; ``alt``, ``speed`` and ``fpa`` (flight-path-angle,
    state index 5) are signed.
    """
    B = comparison.paths[REFERENCE_KEY]
    out = {k: {"horiz": [], "alt": [], "head": [], "speed": [], "fpa": []} for k in COMPARED_KEYS}
    for i in indices:
        ref = B[i]
        for k in COMPARED_KEYS:
            p = comparison.paths[k][i]
            out[k]["horiz"].append(horizontal_error_m(p[0], p[1], ref[0], ref[1]))
            out[k]["alt"].append(p[2] - ref[2])
            out[k]["head"].append(heading_error_deg(p[4], ref[4]))
            out[k]["speed"].append(p[3] - ref[3])
            out[k]["fpa"].append(math.degrees(p[5] - ref[5]))
    return out
