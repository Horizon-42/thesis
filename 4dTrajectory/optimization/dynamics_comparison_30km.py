"""30 km comparison of four point-mass dynamics formulations.

We fly ONE trajectory (one start state + one constant control) for ~30 km and
integrate it four different ways, then measure how far each lands from the
physically faithful reference at every point along the path:

  (A) fixed local-tangent ENU, anchored at the TARGET  (make_local_enu_step_integrator)
        - one ENU tangent plane for the whole 30 km, ref_geo = target.
  (B) per-step re-anchored ENU                          (make_geo_step_from_enu_integrator)
        - rebuilds the tangent frame every step (ref = current point).  This is
          the most faithful discrete integrator, so it is the REFERENCE.
  (C) geodetic RHS, APPROX transport                    (make_geodetic_step_integrator("approx"))
        - one continuous RHS in (lat, lon, h); curvature + frame-rotation
          (transport) folded in.  Should match (B) to ~mm: validates the RHS.
  (D) geodetic RHS, NO transport                        (make_geodetic_step_integrator("none"))
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from geokit import SPHERE_RADIUS_M as _R  # noqa: E402
from aircraft.aircraft_sets import A320  # noqa: E402
from aircraft.aero_params import aero_params_for_aircraft  # noqa: E402
from dynamics_comparison import (  # noqa: E402
    compare_dynamics,
    horizontal_error_m as _horiz_m,
    heading_error_deg as _heading_err_deg,
)


def main() -> int:
    aircraft = A320
    ap = aero_params_for_aircraft(aircraft)

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
    dt = 0.02
    duration = 250.0

    # System A is the fixed local-tangent ENU anchored at the TARGET (this study's
    # documented setup; see dynamics_comparison.py for the shared engine).
    comparison = compare_dynamics(
        start=dict(lat=start["lat"], lon=start["lon"], alt=start["alt"],
                   V=V0, psi=psi0, gamma=gamma0, mass=aircraft.mass_kg),
        control=(control["thrust_n"], math.radians(control["bank_deg"]), control["load_factor"]),
        aero_params=[ap.S, ap.Cl_max, ap.Cd0, ap.k, ap.stall_threshold, ap.k_stall],
        duration_s=duration,
        dt_s=dt,
        anchor_geo=(target["lat"], target["lon"], 0.0),
        max_range_m=range_km * 1000.0,
        stop_below_ground=False,
    )
    paths = comparison.paths
    dist = comparison.dist_m
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
