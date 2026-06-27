"""Compare the two DISCRETE flight-dynamics systems over a 5 km horizon.

Why this matters
----------------
The frontend playback integrates with the re-anchored ENU stepper
``make_geo_step_from_enu_integrator``: every RK4 step it freezes a fresh
local tangent plane, integrates the *flat* ENU dynamics, then re-projects
to geodetic.  The new direct-collocation optimiser instead integrates one
continuous geodetic RHS (``make_geodetic_dynamics_model``) that folds the
coordinate transformation into the WGS84 curvature factors plus the
transport-rate terms on ``psi``/``gamma``.

These are two *different discrete operators*.  This script propagates the
SAME initial state with the SAME control through both and reports the
endpoint divergence over a terminal-area horizon, so we can show:

1. The geodetic RHS **with transport** reproduces the re-anchored stepper
   to truncation error (sub-centimetre over 5 km) -- i.e. they are two
   consistent discretisations of the same continuous system, so the
   optimiser and the playback integrator land on the same target without
   any multiple-shooting polish.

2. The geodetic RHS **without transport** drifts by the transport-scale
   term (metres horizontally, ~0.04-0.05 deg in psi/gamma over 5 km),
   which is exactly the meridian-convergence + curvature-pitch effect the
   re-anchored stepper captures implicitly by re-projecting the velocity
   each step.

Run
---
    python -m geodetic_vs_reanchored_error                  # A320 defaults
    python -m geodetic_vs_reanchored_error --aircraft C172 --max-range-km 5
    python -m geodetic_vs_reanchored_error --csv out.csv

Outputs a per-distance and per-heading summary to stdout; ``--csv`` dumps
the full grid for plotting.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np

# Make ``aerodynamic_model`` importable when run as a script or module.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerodynamic_model.aircraft_sets import A320, C172  # noqa: E402
from aerodynamic_model.casadi_simulator import (  # noqa: E402
    AeroParams,
    make_geo_step_from_enu_integrator,
    make_geodetic_step_integrator,
)

_AIRCRAFT = {"A320": A320, "C172": C172}
_DEFAULT_DISTANCES_KM = (1.0, 2.0, 3.0, 4.0, 5.0)
# Headings measured from East (psi = 0 is due East, +ve toward North),
# matching the dynamics-model convention.
_DEFAULT_HEADINGS_DEG = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)


# --------------------------------------------------------------------------
# Steppers
# --------------------------------------------------------------------------

@dataclass
class Steppers:
    reanchored: ca.Function       # make_geo_step_from_enu_integrator
    geodetic_transport: ca.Function
    geodetic_no_transport: ca.Function
    aero_params: ca.DM


def build_steppers(aircraft) -> Steppers:
    ap = AeroParams(S=aircraft.wing_area_m2)
    aero = ca.DM([ap.S, ap.Cl_max, ap.Cd0, ap.k, ap.stall_threshold, ap.k_stall])
    return Steppers(
        reanchored=make_geo_step_from_enu_integrator()["step_func"],
        geodetic_transport=make_geodetic_step_integrator(transport="approx")["step_func"],
        geodetic_no_transport=make_geodetic_step_integrator(transport="none")["step_func"],
        aero_params=aero,
    )


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def _horizontal_m(a: np.ndarray, b: np.ndarray) -> float:
    """Great-circle-ish horizontal distance between two geodetic points
    (lat/lon in degrees), accurate at the metre level over a few km."""
    R = 6_371_000.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    mid = 0.5 * (lat1 + lat2)
    return R * math.hypot(dlat, dlon * math.cos(mid))


def _distance_from_origin_m(state: np.ndarray, origin: np.ndarray) -> float:
    return _horizontal_m(state, origin)


def _wrap_deg(angle_deg: float) -> float:
    """Wrap an angle difference into (-180, 180] so heading values near
    the +x axis (which atan2 may report as either ~0 or ~360) compare
    cleanly."""
    return (angle_deg + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------
# Core comparison
# --------------------------------------------------------------------------

@dataclass
class Divergence:
    distance_km: float
    heading_deg: float
    horiz_transport_m: float
    horiz_no_transport_m: float
    alt_transport_m: float
    alt_no_transport_m: float
    psi_transport_deg: float
    psi_no_transport_deg: float
    gamma_transport_deg: float
    gamma_no_transport_deg: float


def _state(lat, lon, alt, V, psi, gamma, m) -> ca.DM:
    return ca.DM([lat, lon, alt, V, psi, gamma, m])


def compare_heading(
    steppers: Steppers,
    aircraft,
    heading_deg: float,
    distances_km,
    dt: float,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float,
) -> list[Divergence]:
    """Propagate all three steppers in lockstep from a common state and
    record the divergence whenever the re-anchored stepper crosses each
    distance milestone."""
    V0 = aircraft.terminal_speed_kt * 0.51444 + 10.0
    psi0 = math.radians(heading_deg)
    gamma0 = 0.0
    u = ca.DM([aircraft.approach_thrust_guess_n, 0.0, 1.0])
    aero = steppers.aero_params

    x0 = _state(ref_lat, ref_lon, ref_alt, V0, psi0, gamma0, aircraft.mass_kg)
    origin = np.array(x0).reshape(-1)[:2]

    x_re = x0
    x_gt = x0
    x_gnt = x0

    targets = sorted(distances_km)
    next_idx = 0
    results: list[Divergence] = []

    # Enough time to cover the largest distance with margin.
    max_time = 1.3 * (max(distances_km) * 1000.0) / V0
    n_steps = int(round(max_time / dt))

    for _ in range(n_steps):
        x_re = steppers.reanchored(x_geo=x_re, u=u, aero_params=aero, dt=dt)["x_geo_next"]
        x_gt = steppers.geodetic_transport(x_geo=x_gt, u=u, aero_params=aero, dt=dt)["x_geo_next"]
        x_gnt = steppers.geodetic_no_transport(x_geo=x_gnt, u=u, aero_params=aero, dt=dt)["x_geo_next"]

        re = np.array(x_re).reshape(-1)
        gt = np.array(x_gt).reshape(-1)
        gnt = np.array(x_gnt).reshape(-1)

        dist_m = _distance_from_origin_m(re, origin)
        while next_idx < len(targets) and dist_m >= targets[next_idx] * 1000.0:
            results.append(Divergence(
                distance_km=targets[next_idx],
                heading_deg=heading_deg,
                horiz_transport_m=_horizontal_m(gt, re),
                horiz_no_transport_m=_horizontal_m(gnt, re),
                alt_transport_m=gt[2] - re[2],
                alt_no_transport_m=gnt[2] - re[2],
                psi_transport_deg=_wrap_deg(math.degrees(gt[4] - re[4])),
                psi_no_transport_deg=_wrap_deg(math.degrees(gnt[4] - re[4])),
                gamma_transport_deg=_wrap_deg(math.degrees(gt[5] - re[5])),
                gamma_no_transport_deg=_wrap_deg(math.degrees(gnt[5] - re[5])),
            ))
            next_idx += 1
        if next_idx >= len(targets):
            break

    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _print_summary(rows: list[Divergence], aircraft_code: str, ref_lat: float) -> None:
    print()
    print(f"Discrete dynamics comparison  —  aircraft={aircraft_code}, ref_lat={ref_lat:.3f}°")
    print("Endpoint divergence of each geodetic RK4 vs the re-anchored ENU RK4 (playback).")
    print()

    # Aggregate by distance (max over headings) to show the worst case.
    by_dist: dict[float, list[Divergence]] = {}
    for r in rows:
        by_dist.setdefault(r.distance_km, []).append(r)

    print("Worst-case over all headings, by distance:")
    print(f"  {'dist':>5} | {'GEO+transport vs re-anchored':^34} | {'GEO no-transport vs re-anchored':^34}")
    print(f"  {'[km]':>5} | {'horiz[m]':>9} {'alt[m]':>8} {'dpsi[°]':>7} {'dgam[°]':>7} | "
          f"{'horiz[m]':>9} {'alt[m]':>8} {'dpsi[°]':>7} {'dgam[°]':>7}")
    print("  " + "-" * 88)
    for dist in sorted(by_dist):
        g = by_dist[dist]
        ht = max(abs(r.horiz_transport_m) for r in g)
        at = max(abs(r.alt_transport_m) for r in g)
        pt = max(abs(r.psi_transport_deg) for r in g)
        mt = max(abs(r.gamma_transport_deg) for r in g)
        hn = max(abs(r.horiz_no_transport_m) for r in g)
        an = max(abs(r.alt_no_transport_m) for r in g)
        pn = max(abs(r.psi_no_transport_deg) for r in g)
        mn = max(abs(r.gamma_no_transport_deg) for r in g)
        print(f"  {dist:5.1f} | {ht:9.4f} {at:8.4f} {pt:7.4f} {mt:7.4f} | "
              f"{hn:9.4f} {an:8.4f} {pn:7.4f} {mn:7.4f}")

    # Per-heading breakdown at the longest distance.
    longest = max(by_dist)
    print()
    print(f"Per-heading breakdown at {longest:.1f} km (heading measured from East):")
    print(f"  {'hdg[°]':>6} | {'GEO+transport horiz[m]':>22} | {'GEO no-transport horiz[m]':>26}")
    print("  " + "-" * 60)
    for r in sorted(by_dist[longest], key=lambda x: x.heading_deg):
        print(f"  {r.heading_deg:6.0f} | {abs(r.horiz_transport_m):22.4f} | {abs(r.horiz_no_transport_m):26.4f}")

    # Headline numbers.
    all_ht = max(abs(r.horiz_transport_m) for r in rows)
    all_hn = max(abs(r.horiz_no_transport_m) for r in rows)
    print()
    print("Conclusion:")
    print(f"  • GEO+transport tracks the playback integrator to {all_ht:.4f} m over the whole grid")
    print("    → optimiser and playback share one discrete system; no polish required.")
    print(f"  • GEO no-transport drifts up to {all_hn:.4f} m → that gap IS the transport rate")
    print("    (meridian convergence + curvature pitch) the playback stepper captures.")


def _write_csv(rows: list[Divergence], path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "distance_km", "heading_deg",
            "horiz_transport_m", "horiz_no_transport_m",
            "alt_transport_m", "alt_no_transport_m",
            "psi_transport_deg", "psi_no_transport_deg",
            "gamma_transport_deg", "gamma_no_transport_deg",
        ])
        for r in rows:
            writer.writerow([
                r.distance_km, r.heading_deg,
                r.horiz_transport_m, r.horiz_no_transport_m,
                r.alt_transport_m, r.alt_no_transport_m,
                r.psi_transport_deg, r.psi_no_transport_deg,
                r.gamma_transport_deg, r.gamma_no_transport_deg,
            ])
    print(f"\nWrote full grid to {path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aircraft", choices=sorted(_AIRCRAFT), default="A320")
    parser.add_argument("--reference-lat", type=float, default=51.1138)
    parser.add_argument("--reference-lon", type=float, default=-114.0203)
    parser.add_argument("--reference-alt", type=float, default=1000.0)
    parser.add_argument("--max-range-km", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.05, help="RK4 step (s) for all three integrators")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    aircraft = _AIRCRAFT[args.aircraft]
    distances = tuple(d for d in _DEFAULT_DISTANCES_KM if d <= args.max_range_km) or (args.max_range_km,)
    steppers = build_steppers(aircraft)

    rows: list[Divergence] = []
    for heading in _DEFAULT_HEADINGS_DEG:
        rows.extend(compare_heading(
            steppers, aircraft, heading, distances, args.dt,
            args.reference_lat, args.reference_lon, args.reference_alt,
        ))

    _print_summary(rows, args.aircraft, args.reference_lat)
    if args.csv is not None:
        _write_csv(rows, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
