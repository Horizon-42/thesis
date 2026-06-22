"""30 km comparison of four point-mass dynamics formulations.

We fly ONE trajectory (one start state + one constant control) for ~30 km and
integrate it four different ways, then measure how far each lands from the
physically faithful reference at every point along the path:

  (A) fixed local-tangent ENU, anchored at the TARGET  (make_local_enu_step_integrator)
        - one ENU tangent plane for the whole 30 km, ref_geo = target.
  (B) per-step re-anchored ENU                          (make_geo_step_from_enu_integrator)
        - rebuilds the tangent frame every step (ref = current point).  This is
          the most faithful discrete integrator, so it is the REFERENCE.
  (C) full geodetic RHS, WITH transport                 (make_geodetic_step_integrator(True))
        - one continuous RHS in (lat, lon, h); curvature + frame-rotation
          (transport) folded in.  Should match (B) to ~mm: validates the RHS.
  (D) geodetic RHS, NO transport                        (make_geodetic_step_integrator(False))
        - same but drops the transport terms; isolates how much they matter.

All four are integrated with the SAME RK4 step (dt below) from the SAME start,
so the differences are MODEL/frame differences, not discretisation noise.

Error metrics vs the reference (B), as a function of distance flown:
  - horizontal position error (m), altitude error (m), heading error (deg).

Writes ``dynamics_comparison_30km_data.json`` (consumed by the HTML doc) and
prints a summary table.

Run:  python -m dynamics_comparison_30km
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import casadi as ca
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerodynamic_model.aircraft_sets import A320  # noqa: E402
from aerodynamic_model.casadi_simulator import (  # noqa: E402
    aero_params_for_aircraft,
    make_dynamics_model,
    rk4_step_expr,
    geodetic_state_to_enu_expr,
    enu_state_to_geodetic_expr,
    make_geo_step_from_enu_integrator,
    make_geodetic_step_integrator,
)

_R = 6_371_000.0


def _horiz_m(lat1, lon1, lat2, lon2):
    return _R * math.hypot(
        math.radians(lat1 - lat2),
        math.radians(lon1 - lon2) * math.cos(math.radians(lat2)),
    )


def _heading_err_deg(psi1, psi2):
    d = math.degrees(psi1 - psi2) % 360.0
    return min(d, 360.0 - d)


def main() -> int:
    aircraft = A320
    ap = aero_params_for_aircraft(aircraft)
    aero = ca.DM([ap.S, ap.Cl_max, ap.Cd0, ap.k, ap.stall_threshold, ap.k_stall])

    # ── Scenario (documented in the HTML) ────────────────────────────────
    # Target on the ground; start ~30 km to the south-west, heading NE toward
    # it, in a gentle steady descent.  Constant control for the whole run.
    target = dict(lat=35.60, lon=-78.50, alt=200.0)
    bearing_deg = 45.0                      # fly toward NE (psi from East = 45 deg => NE)
    range_km = 30.0
    # Place the start ``range_km`` to the SW of the target along that bearing.
    d = range_km * 1000.0
    dlat = (d * math.sin(math.radians(bearing_deg))) / _R
    dlon = (d * math.cos(math.radians(bearing_deg))) / (_R * math.cos(math.radians(target["lat"])))
    start = dict(
        lat=target["lat"] - math.degrees(dlat),
        lon=target["lon"] - math.degrees(dlon),
        alt=2300.0,
    )
    V0 = 130.0                              # m/s
    psi0 = math.radians(bearing_deg)        # heading from East, +ve toward North
    gamma0 = math.radians(-2.0)
    control = dict(thrust_n=70_000.0, bank_deg=0.0, load_factor=1.0)
    u = ca.DM([control["thrust_n"], math.radians(control["bank_deg"]), control["load_factor"]])
    dt = 0.02
    duration = 250.0
    n_steps = int(duration / dt)

    x0 = [start["lat"], start["lon"], start["alt"], V0, psi0, gamma0, aircraft.mass_kg]

    # ── Integrators (all take lat/lon in DEGREES externally) ─────────────
    # (A) fixed local-tangent ENU: a fixed frame integrates ENTIRELY in ENU,
    # converting geodetic <-> ENU only at the boundary.  So we convert the start
    # into the target-anchored frame ONCE, step the flat point-mass RHS in ENU,
    # and convert back only to record each sample (NOT every step).
    flat_rhs = make_dynamics_model()["rhs_func"]
    reanchored = make_geo_step_from_enu_integrator()["step_func"]
    rhs_tr = make_geodetic_step_integrator(include_transport=True)["step_func"]
    rhs_no = make_geodetic_step_integrator(include_transport=False)["step_func"]
    ref_geo = ca.DM([target["lat"], target["lon"], 0.0])  # fixed anchor = target

    def geo_to_enu(xg):   # 7-vec geodetic (deg) -> 7-vec ENU in the fixed frame
        return ca.vertcat(*geodetic_state_to_enu_expr(ca.DM(xg), ref_geo), xg[6])

    def enu_to_geo(xe):   # 7-vec ENU -> 7-vec geodetic (deg)
        return list(np.array(ca.vertcat(*enu_state_to_geodetic_expr(xe, ref_geo), xe[6])).ravel())

    def geo_step(which, x):   # one geodetic step for B / C / D
        f = {"B": reanchored, "C": rhs_tr, "D": rhs_no}[which]
        return list(np.array(f(x_geo=ca.DM(x), u=u, aero_params=aero, dt=dt)["x_geo_next"]).ravel())

    paths = {k: [list(x0)] for k in "ABCD"}
    state = {k: list(x0) for k in "BCD"}
    enu_A = geo_to_enu(x0)             # system A's state lives in the fixed ENU frame
    dist = [0.0]                       # cumulative ground distance along reference B
    for _ in range(n_steps):
        enu_A = rk4_step_expr(flat_rhs, enu_A, u, aero, dt)   # integrate in ENU
        paths["A"].append(enu_to_geo(enu_A))                 # convert only to record
        for k in "BCD":
            state[k] = geo_step(k, state[k])
            paths[k].append(list(state[k]))
        b, bprev = paths["B"][-1], paths["B"][-2]
        dist.append(dist[-1] + _horiz_m(b[0], b[1], bprev[0], bprev[1]))
        if dist[-1] >= range_km * 1000.0:   # stop once we have flown 30 km
            break
    B = paths["B"]

    # Sample ~every 0.5 km for a compact series.
    series = {"distance_km": [], "A": {"horiz": [], "alt": [], "head": []},
              "C": {"horiz": [], "alt": [], "head": []},
              "D": {"horiz": [], "alt": [], "head": []}}
    next_mark = 0.0
    for i in range(len(B)):
        if dist[i] + 1e-9 < next_mark:
            continue
        next_mark += 500.0
        series["distance_km"].append(round(dist[i] / 1000.0, 4))
        for k in ("A", "C", "D"):
            p = paths[k][i]
            series[k]["horiz"].append(round(_horiz_m(p[0], p[1], B[i][0], B[i][1]), 4))
            series[k]["alt"].append(round(p[2] - B[i][2], 4))
            series[k]["head"].append(round(_heading_err_deg(p[4], B[i][4]), 5))

    meta = {
        "aircraft": "A320", "range_km": range_km, "dt_s": dt, "duration_s": duration,
        "V0_mps": V0, "bearing_deg": bearing_deg, "gamma0_deg": -2.0,
        "control": control, "start": start, "target": target,
        "reference": "B = per-step re-anchored ENU (most faithful)",
    }
    out = {"meta": meta, "series": series}
    out_path = Path(__file__).resolve().parent / "dynamics_comparison_30km_data.json"
    out_path.write_text(json.dumps(out, indent=1))

    # ── Summary ──────────────────────────────────────────────────────────
    def final(k, field):
        return series[k][field][-1]
    print(f"\n30 km dynamics comparison — A320, reference = (B) per-step re-anchored ENU")
    print(f"start {start['lat']:.4f},{start['lon']:.4f},{start['alt']:.0f}m  ->  "
          f"target {target['lat']},{target['lon']},{target['alt']}m   "
          f"V0={V0} psi={bearing_deg}deg gamma=-2deg  thrust={control['thrust_n']:.0f}N\n")
    print(f"  {'system':<40}{'horiz err':>12}{'alt err':>12}{'head err':>12}")
    print("  " + "-" * 76)
    names = {"A": "A fixed local-ENU @ target",
             "C": "C full geodetic RHS (+transport)",
             "D": "D geodetic RHS (no transport)"}
    for k in ("A", "C", "D"):
        print(f"  {names[k]:<40}{final(k,'horiz'):>10.2f} m{final(k,'alt'):>10.2f} m{final(k,'head'):>9.3f} deg")
    print(f"\n  (at {range_km:.0f} km. C should be ~0: validates the RHS = re-anchored.)")
    print(f"  data -> {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
