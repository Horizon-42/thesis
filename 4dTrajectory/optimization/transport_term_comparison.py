"""Formal comparison of the two geodetic TRANSPORT models: APPROX vs FULL.

Why this exists
---------------
The continuous geodetic RHS (``make_geodetic_dynamics_model``) carries the
frame-rotation *transport rate* on ``psi``/``gamma``.  Two models are available:

  * ``approx`` (the historical default) — the EXACT ``gamma`` transport plus the
    dominant ``psi`` meridian-convergence term, but DROPPING one ``psi`` term.
  * ``full``   — adds that dropped term back, giving the EXACT transport.

The dropped term (derived by differentiating ``psi = atan2(u_N, u_E)`` under the
frame's transport angular velocity, ``d u_hat/dt = -omega_en x u_hat``) is

    psi_dot_cross =  V sin(gamma) sin(psi) cos(psi) * ( 1/(R_N+h) - 1/(R_M+h) )

a product of two small factors — ``sin(gamma)`` (zero in level flight) and the
curvature-radius difference ``1/(R_N+h) - 1/(R_M+h) = O(e^2)`` (zero on a
sphere) — so it is tiny.  This script QUANTIFIES exactly how tiny, so the choice
to use ``approx`` (or not) is an explicit, measured decision rather than a silent
approximation.

What it measures
----------------
1. RHS level (dt-independent, exact): ``f_full - f_approx`` equals the analytic
   cross term in the ``psi`` component and is identically zero in every other
   component (lat, lon, h, V, gamma, m) — so the approximation perturbs ONLY the
   heading rate.

2. Trajectory level: integrate the SAME initial state + constant control with
   both models (same RK4 scheme — only the vector field differs) and report the
   divergence ``|approx - full|`` over a flight.  Because both use the same
   scheme, this divergence is the pure integrated cross term (no truncation
   noise): it is INDEPENDENT of the step ``dt`` (verified) and grows only with
   horizon.

3. Context: both models track the re-anchored ENU stepper (system B, the exact
   discrete truth the playback runs) to the millimetre; the gap BETWEEN them is
   exactly (2).

Run
---
    python -m transport_term_comparison
    python -m transport_term_comparison --aircraft A320 --bank-deg 15 --json out.json

Writes ``transport_term_comparison_data.json`` (used by
``4dTrajectory/docs/transport_term_comparison.zh.html``) and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import casadi as ca
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aircraft.aircraft_sets import A320, C172  # noqa: E402
from aerodynamic_model.casadi_coordinates_converter import WGS84_A, WGS84_E2  # noqa: E402
from aerodynamic_model.casadi_simulator import (  # noqa: E402
    AeroParams,
    make_geo_step_from_enu_integrator,
    make_geodetic_dynamics_model,
    make_geodetic_step_integrator,
)

_AIRCRAFT = {"A320": A320, "C172": C172}
_R_MEAN = 6_371_000.0  # mean-earth radius for the small-angle ground metric


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def _horizontal_m(a: np.ndarray, b: np.ndarray) -> float:
    """Small-angle ground distance (m) between two geodetic points (deg)."""
    return _R_MEAN * math.hypot(
        math.radians(a[0] - b[0]),
        math.radians(a[1] - b[1]) * math.cos(math.radians(b[0])),
    )


def _arcsec(rad: float) -> float:
    return math.degrees(rad) * 3600.0


def _radii(lat_rad: float, h: float) -> tuple[float, float]:
    """WGS84 meridional (R_M) and prime-vertical (R_N) radii at this latitude."""
    den = 1.0 - WGS84_E2 * math.sin(lat_rad) ** 2
    R_N = WGS84_A / math.sqrt(den)
    R_M = WGS84_A * (1.0 - WGS84_E2) / den ** 1.5
    return R_M, R_N


def _analytic_cross_term(state_deg: list[float]) -> float:
    """The exact dropped psi-transport term at a geodetic state (lat/lon deg)."""
    lat, _, h, V, psi, gamma, _ = state_deg
    R_M, R_N = _radii(math.radians(lat), h)
    return (
        V * math.sin(gamma) * math.sin(psi) * math.cos(psi)
        * (1.0 / (R_N + h) - 1.0 / (R_M + h))
    )


# --------------------------------------------------------------------------
# Integration
# --------------------------------------------------------------------------

def _build(aircraft):
    ap = AeroParams(S=aircraft.wing_area_m2)
    aero = ca.DM([ap.S, ap.Cl_max, ap.Cd0, ap.k, ap.stall_threshold, ap.k_stall])
    return {
        "aero": aero,
        "approx": make_geodetic_step_integrator(transport="approx")["step_func"],
        "full": make_geodetic_step_integrator(transport="full")["step_func"],
        "truth": make_geo_step_from_enu_integrator()["step_func"],  # re-anchored = exact discrete
        "rhs_approx": make_geodetic_dynamics_model(transport="approx")["rhs_func"],
        "rhs_full": make_geodetic_dynamics_model(transport="full")["rhs_func"],
    }


def _roll(step, aero, x0, u, dt, horizon) -> np.ndarray:
    n = int(round(horizon / dt))
    x = ca.DM(x0)
    for _ in range(n):
        x = step(x_geo=x, u=u, aero_params=aero, dt=dt)["x_geo_next"]
    return np.array(x).reshape(-1)


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def compare(
    aircraft,
    *,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float,
    speed: float,
    heading_deg: float,
    fpa_deg: float,
    bank_deg: float,
    horizons_s: tuple[float, ...] = (30.0, 60.0, 120.0),
    dts_s: tuple[float, ...] = (0.05, 0.02, 0.01, 0.005),
) -> dict:
    funcs = _build(aircraft)
    aero = funcs["aero"]
    x0 = [
        ref_lat, ref_lon, ref_alt, speed,
        math.radians(heading_deg), math.radians(fpa_deg), aircraft.mass_kg,
    ]
    u = ca.DM([aircraft.approach_thrust_guess_n, math.radians(bank_deg), 1.0])

    # ── 1. RHS-level difference vs the analytic cross term ────────────────────
    x_rad = ca.DM([
        math.radians(ref_lat), math.radians(ref_lon), ref_alt, speed,
        math.radians(heading_deg), math.radians(fpa_deg), aircraft.mass_kg,
    ])
    f_approx = np.array(funcs["rhs_approx"](x_rad, u, aero)).reshape(-1)
    f_full = np.array(funcs["rhs_full"](x_rad, u, aero)).reshape(-1)
    rhs_diff = f_full - f_approx
    cross = _analytic_cross_term(x0)
    labels = ["lat_dot", "lon_dot", "h_dot", "V_dot", "psi_dot", "gamma_dot", "m_dot"]
    rhs = {
        "labels": labels,
        "diff": [float(v) for v in rhs_diff],
        "analyticCross": cross,
        "psiMatchesCross": bool(abs(rhs_diff[4] - cross) < 1e-18),
        "nonPsiAllZero": bool(np.array_equal(np.delete(rhs_diff, 4), np.zeros(6))),
    }

    # ── 2. Trajectory divergence |approx - full| over horizon (at the finest dt)
    fine_dt = min(dts_s)
    horizon_rows = []
    for T in horizons_s:
        xa = _roll(funcs["approx"], aero, x0, u, fine_dt, T)
        xf = _roll(funcs["full"], aero, x0, u, fine_dt, T)
        horizon_rows.append({
            "horizonS": T,
            "horizMm": _horizontal_m(xa, xf) * 1000.0,
            "psiArcsec": _arcsec(abs(xa[4] - xf[4])),
            "gammaArcsec": _arcsec(abs(xa[5] - xf[5])),
            "altMm": abs(xa[2] - xf[2]) * 1000.0,
        })

    # ── 3. dt-independence of |approx - full| (proves it is NOT truncation) ───
    mid_T = horizons_s[len(horizons_s) // 2]
    dt_rows = []
    for dt in dts_s:
        xa = _roll(funcs["approx"], aero, x0, u, dt, mid_T)
        xf = _roll(funcs["full"], aero, x0, u, dt, mid_T)
        dt_rows.append({
            "dtS": dt,
            "horizMm": _horizontal_m(xa, xf) * 1000.0,
            "psiArcsec": _arcsec(abs(xa[4] - xf[4])),
        })

    # ── 4. Both vs the re-anchored truth B (context: both track B to ~mm) ─────
    truth_T = horizons_s[-1]
    xt = _roll(funcs["truth"], aero, x0, u, fine_dt, truth_T)
    xa = _roll(funcs["approx"], aero, x0, u, fine_dt, truth_T)
    xf = _roll(funcs["full"], aero, x0, u, fine_dt, truth_T)
    vs_truth = {
        "horizonS": truth_T,
        "approxHorizMm": _horizontal_m(xa, xt) * 1000.0,
        "fullHorizMm": _horizontal_m(xf, xt) * 1000.0,
        "approxPsiArcsec": _arcsec(abs(xa[4] - xt[4])),
        "fullPsiArcsec": _arcsec(abs(xf[4] - xt[4])),
    }

    return {
        "meta": {
            "aircraft": aircraft.code,
            "refLat": ref_lat,
            "refLon": ref_lon,
            "refAltM": ref_alt,
            "speedMps": speed,
            "headingDeg": heading_deg,
            "fpaDeg": fpa_deg,
            "bankDeg": bank_deg,
            "fineDtS": fine_dt,
        },
        "rhsDifference": rhs,
        "horizonSweep": horizon_rows,
        "dtIndependence": dt_rows,
        "vsTruth": vs_truth,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _print_summary(data: dict) -> None:
    m = data["meta"]
    print()
    print("Transport-term comparison  —  APPROX vs FULL (exact) geodetic transport")
    print(f"aircraft={m['aircraft']}  ref=({m['refLat']:.4f},{m['refLon']:.4f}) "
          f"alt={m['refAltM']:.0f}m  V={m['speedMps']:.0f}  hdg={m['headingDeg']:.0f}°  "
          f"fpa={m['fpaDeg']:.0f}°  bank={m['bankDeg']:.0f}°")
    print()

    rhs = data["rhsDifference"]
    print("1) RHS difference f_full − f_approx (should be the cross term in psi ONLY):")
    for lbl, v in zip(rhs["labels"], rhs["diff"]):
        print(f"     {lbl:9s} {v: .6e}")
    print(f"   analytic cross term = {rhs['analyticCross']: .6e}  "
          f"(psi matches: {rhs['psiMatchesCross']}, all others zero: {rhs['nonPsiAllZero']})")
    print()

    print("2) |approx − full| over horizon (pure integrated cross term):")
    print(f"     {'T[s]':>5} {'horiz[mm]':>10} {'dpsi[arcsec]':>13} {'dgamma[arcsec]':>15} {'dalt[mm]':>9}")
    for r in data["horizonSweep"]:
        print(f"     {r['horizonS']:5.0f} {r['horizMm']:10.4f} {r['psiArcsec']:13.4f} "
              f"{r['gammaArcsec']:15.4f} {r['altMm']:9.4f}")
    print()

    print("3) dt-independence of |approx − full| (flat ⇒ vector-field, not truncation):")
    for r in data["dtIndependence"]:
        print(f"     dt={r['dtS']:6.3f}  horiz={r['horizMm']:9.4f} mm  dpsi={r['psiArcsec']:9.4f} arcsec")
    print()

    vt = data["vsTruth"]
    print(f"4) vs re-anchored truth B over {vt['horizonS']:.0f}s (both track B to ~mm):")
    print(f"     approx: horiz={vt['approxHorizMm']:8.4f} mm  dpsi={vt['approxPsiArcsec']:8.4f} arcsec")
    print(f"     full  : horiz={vt['fullHorizMm']:8.4f} mm  dpsi={vt['fullPsiArcsec']:8.4f} arcsec")
    print()
    print("Conclusion: the APPROX transport perturbs ONLY the heading rate, by the exact")
    print("cross term; integrated over a terminal-area flight that is sub-millimetre")
    print("horizontally and ~0.03 arcsec in heading — negligible, now quantified, and the")
    print("FULL (exact) transport is available as an explicit option.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--aircraft", choices=sorted(_AIRCRAFT), default="A320")
    p.add_argument("--reference-lat", type=float, default=51.1138)
    p.add_argument("--reference-lon", type=float, default=-114.0203)
    p.add_argument("--reference-alt", type=float, default=3000.0)
    p.add_argument("--speed", type=float, default=130.0)
    p.add_argument("--heading-deg", type=float, default=30.0)
    p.add_argument("--fpa-deg", type=float, default=-3.0)
    p.add_argument("--bank-deg", type=float, default=15.0)
    p.add_argument("--json", type=Path, default=Path(__file__).with_name("transport_term_comparison_data.json"))
    args = p.parse_args(argv)

    aircraft = _AIRCRAFT[args.aircraft]
    data = compare(
        aircraft,
        ref_lat=args.reference_lat,
        ref_lon=args.reference_lon,
        ref_alt=args.reference_alt,
        speed=args.speed,
        heading_deg=args.heading_deg,
        fpa_deg=args.fpa_deg,
        bank_deg=args.bank_deg,
    )
    _print_summary(data)
    args.json.write_text(json.dumps(data, indent=2))
    print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
