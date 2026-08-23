"""Quantify the modeling error introduced by using a single FIXED ENU frame
over a 5 km final-approach horizon.

Why this matters
----------------
The existing ``aerodynamic_model.casadi_simulator`` exposes a *continuous*
ENU RHS via ``make_dynamics_model()``, but the geodetic wrapper
``make_geo_step_from_enu_integrator()`` re-anchors the ENU frame at every
RK4 step.  That re-anchoring is a discrete jump operator and cannot be
embedded into the defect equations of a direct-collocation transcription,
which assume one continuous ODE on the whole horizon.

A practical fix for direct collocation is to freeze the ENU frame at the
runway threshold (or any single point) for the whole final-approach
window.  Earth curvature then becomes a *modeling* error that this script
quantifies.

Three independent geometric error sources at distance ``d`` from the
anchor (R = WGS84 mean radius ≈ 6 378 137 m):

* **Curvature drop**     ``Δh ≈ d² / (2R)``           — a flat-ENU "u=0"
  point sits below the true ellipsoid by this much.
* **Up-axis tilt**       ``α   ≈ d / R``              — the angle between
  the anchor's "up" and the local "up" at the displaced point.  A
  level horizontal velocity ``V`` in the anchor frame therefore picks
  up an apparent vertical component ``V·sin α`` at the displaced point.
* **Meridian convergence** ``γ_c ≈ Δλ · sin φ``      — the rotation
  between anchor-frame north and local-frame north.

We sample a polar grid up to 5 km and report all three.  A second test
propagates a kinematic point mass with the same initial state through
(a) a straight Cartesian integration in the fixed ENU frame, and
(b) the existing re-anchored stepper, and reports the geodetic
divergence at the end of the horizon.  That second number is what the
direct-collocation transcription will actually inherit.

Run
---
    python -m fixed_enu_frame_error                      # CYYC defaults
    python -m fixed_enu_frame_error --reference-lat 49.19 \\
        --reference-lon -123.18 --reference-alt 4 --max-range-km 5

Outputs a per-radius summary table to stdout; pass ``--csv path`` to
also dump the full grid for plotting.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from geokit import EARTH_RADIUS_MEAN_M as _EARTH_RADIUS_M
from geokit import haversine_m

_AERODYNAMIC_MODEL_DIR = Path(__file__).resolve().parents[2] / "aerodynamic_model"
if str(_AERODYNAMIC_MODEL_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_AERODYNAMIC_MODEL_DIR.parent))

from aerodynamic_model.coordinates_convertor import (
    CoordinateConverter,
    ENUCoordinate,
    GeodeticCoordinate,
)

# WGS84 mean radius (m), single-sourced from geokit (EARTH_RADIUS_MEAN_M). The
# closed-form curvature scales use a single R rather than picking between N (prime
# vertical) and M (meridian) — the difference is < 0.3 % at typical airport
# latitudes and not the dominant error term over 5 km.

# Reference horizontal speed used to translate the up-axis tilt (a pure
# geometric angle) into an apparent vertical velocity component (m/s).
# 80 m/s ≈ A320 final-approach speed (145 kt ≈ 74.6 m/s).
_REFERENCE_SPEED_MPS = 80.0


@dataclass(frozen=True)
class GridPoint:
    radius_m: float
    bearing_deg: float
    altitude_m: float

    @property
    def east_m(self) -> float:
        return self.radius_m * math.sin(math.radians(self.bearing_deg))

    @property
    def north_m(self) -> float:
        return self.radius_m * math.cos(math.radians(self.bearing_deg))


@dataclass(frozen=True)
class PointError:
    radius_m: float
    bearing_deg: float
    altitude_m: float
    curvature_drop_m: float
    up_tilt_deg: float
    meridian_convergence_deg: float
    apparent_vertical_speed_mps: float
    horizontal_position_drift_m: float


def _enumerate_grid(
    radii_m: Iterable[float],
    bearings_deg: Iterable[float],
    altitudes_m: Iterable[float],
) -> list[GridPoint]:
    return [
        GridPoint(r, b, h)
        for r in radii_m
        for b in bearings_deg
        for h in altitudes_m
    ]


def _local_axes_in_ecef(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Return the 3x3 matrix whose rows are (e_hat, n_hat, u_hat) at the
    point expressed in ECEF.  This is identical to the rotation that
    ``CoordinateConverter`` builds inline — we duplicate it here as a
    matrix so we can compose two such frames with one matmul."""
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_lam, cos_lam = math.sin(lam), math.cos(lam)
    e_hat = (-sin_lam, cos_lam, 0.0)
    n_hat = (-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi)
    u_hat = (cos_phi * cos_lam, cos_phi * sin_lam, sin_phi)
    return np.array([e_hat, n_hat, u_hat])


def compute_point_error(
    grid_point: GridPoint,
    reference: GeodeticCoordinate,
    reference_speed_mps: float,
) -> PointError:
    """For one polar-grid sample, compute the four geometric error metrics
    that a fixed-ENU dynamics model inherits over the 5 km horizon."""

    # Step 1 — promote (east, north, up=alt) in the fixed (anchor) frame
    # into a true geodetic point.  This is the *exact* point the anchor
    # frame believes the aircraft is at.
    enu_pt = ENUCoordinate(grid_point.east_m, grid_point.north_m, grid_point.altitude_m)
    geo_pt = CoordinateConverter.enu_to_geodetic(enu_pt, reference)

    # Step 2 — curvature drop: where would a "u=0" point in the fixed
    # frame land in geodetic altitude?  The difference vs. the anchor's
    # altitude is the systematic vertical bias of treating the tangent
    # plane as the ellipsoid surface.
    enu_ground = ENUCoordinate(grid_point.east_m, grid_point.north_m, 0.0)
    geo_ground = CoordinateConverter.enu_to_geodetic(enu_ground, reference)
    curvature_drop_m = reference.altitude - geo_ground.altitude

    # Step 3 — up-axis tilt: angle between the anchor's up axis and the
    # local up axis at the displaced point, expressed in the anchor
    # frame.  Compose anchor's e/n/u basis with the displaced point's
    # e/n/u basis through ECEF.
    anchor_axes = _local_axes_in_ecef(reference.latitude, reference.longitude)
    local_axes = _local_axes_in_ecef(geo_pt.latitude, geo_pt.longitude)
    # Rotation that maps a vector expressed in the displaced point's
    # local ENU into the anchor ENU.
    R_local_to_anchor = anchor_axes @ local_axes.T

    # The local up axis (0, 0, 1) re-expressed in anchor ENU:
    local_up_in_anchor = R_local_to_anchor[:, 2]
    # The angle between local up and anchor up = arccos of the z-component.
    up_tilt_rad = math.acos(max(-1.0, min(1.0, float(local_up_in_anchor[2]))))
    up_tilt_deg = math.degrees(up_tilt_rad)

    # Step 4 — meridian convergence: the local north axis re-expressed
    # in anchor ENU has a small east component.  The angle between
    # local-north and anchor-north (after both are projected onto the
    # anchor's horizontal plane) is the convergence.
    local_north_in_anchor = R_local_to_anchor[:, 1]
    convergence_rad = math.atan2(local_north_in_anchor[0], local_north_in_anchor[1])
    meridian_convergence_deg = math.degrees(convergence_rad)

    # Step 5 — translate the tilt into an apparent vertical-velocity
    # component picked up by a horizontal velocity at the displaced
    # point.  A velocity (V, 0, 0) in local ENU (heading east) becomes:
    apparent_vertical_speed_mps = reference_speed_mps * math.sin(up_tilt_rad)

    # Step 6 — sanity check the horizontal position: a flat-Earth
    # interpretation that converts ENU back to (lat, lon) via simple
    # metric-per-degree scaling (no ECEF round-trip) would land at a
    # slightly different geodetic point.  Quantify that drift in metres
    # along the surface.
    flat_dlat_deg = grid_point.north_m / 111_320.0
    flat_dlon_deg = grid_point.east_m / (
        111_320.0 * max(0.1, math.cos(math.radians(reference.latitude)))
    )
    flat_geo = GeodeticCoordinate(
        latitude=reference.latitude + flat_dlat_deg,
        longitude=reference.longitude + flat_dlon_deg,
        altitude=reference.altitude + grid_point.altitude_m,
    )
    horizontal_position_drift_m = _haversine_distance_m(geo_pt, flat_geo)

    return PointError(
        radius_m=grid_point.radius_m,
        bearing_deg=grid_point.bearing_deg,
        altitude_m=grid_point.altitude_m,
        curvature_drop_m=curvature_drop_m,
        up_tilt_deg=up_tilt_deg,
        meridian_convergence_deg=meridian_convergence_deg,
        apparent_vertical_speed_mps=apparent_vertical_speed_mps,
        horizontal_position_drift_m=horizontal_position_drift_m,
    )


def _haversine_distance_m(a: GeodeticCoordinate, b: GeodeticCoordinate) -> float:
    return haversine_m(a.latitude, a.longitude, b.latitude, b.longitude, radius_m=_EARTH_RADIUS_M)


# --------------------------------------------------------------------------
# Dynamics-level comparison
# --------------------------------------------------------------------------
#
# Even with the geometric biases above, the question that actually
# matters for direct collocation is:
#
#   Given the same kinematic state and the same nominal control
#   sequence, how far apart are the endpoints predicted by
#       (a) integration in a single fixed ENU frame, and
#       (b) the existing re-anchored stepper?
#
# We isolate this to the *frame* effect by using a kinematic point mass
# (constant V, ψ, γ — no aerodynamics) and propagating it through both
# methods.  Any divergence is purely the cost of freezing the ENU frame.


@dataclass(frozen=True)
class KinematicEndpointDelta:
    horizontal_position_m: float
    altitude_m: float
    heading_deg: float
    flight_path_deg: float


def _rk4_fixed_enu(
    east0: float,
    north0: float,
    up0: float,
    V: float,
    psi: float,
    gamma: float,
    dt: float,
    n_steps: int,
) -> tuple[float, float, float]:
    """Integrate dx/dt = V cos γ cos ψ in a single fixed ENU frame.

    Note: ψ here is measured from east (psi=0 -> east) to match the
    convention used in ``casadi_simulator.make_dynamics_model``.
    """
    east, north, up = east0, north0, up0
    de = V * math.cos(gamma) * math.cos(psi)
    dn = V * math.cos(gamma) * math.sin(psi)
    du = V * math.sin(gamma)
    for _ in range(n_steps):
        # Kinematics with constant V/ψ/γ collapse to a straight line —
        # an analytical step would suffice, but RK4 keeps the structure
        # identical to the optimiser so the comparison is honest.
        east += de * dt
        north += dn * dt
        up += du * dt
    return east, north, up


def _step_reanchored(
    state: tuple[float, float, float, float, float, float],
    dt: float,
) -> tuple[float, float, float, float, float, float]:
    """One re-anchored kinematic step: integrate dx in the local ENU
    plane, convert (east, north, up) -> geodetic, then re-derive V/ψ/γ
    against the new anchor.

    state = (lat, lon, alt, V, psi, gamma)
    """
    lat, lon, alt, V, psi, gamma = state
    ref = GeodeticCoordinate(lat, lon, 0.0)
    de = V * math.cos(gamma) * math.cos(psi) * dt
    dn = V * math.cos(gamma) * math.sin(psi) * dt
    du = V * math.sin(gamma) * dt
    new_enu = ENUCoordinate(de, dn, alt + du)
    new_geo = CoordinateConverter.enu_to_geodetic(new_enu, ref)

    # Re-express the velocity vector in the new local ENU frame so that
    # V/ψ/γ stay consistent with the new anchor.  This duplicates the
    # logic from ``geodetic_simulator.GeodeticSimulator.step``.
    ve_old = V * math.cos(gamma) * math.cos(psi)
    vn_old = V * math.cos(gamma) * math.sin(psi)
    vu_old = V * math.sin(gamma)

    old_axes = _local_axes_in_ecef(ref.latitude, ref.longitude)
    new_axes = _local_axes_in_ecef(new_geo.latitude, new_geo.longitude)
    # ENU(old) -> ECEF -> ENU(new)
    v_ecef = old_axes.T @ np.array([ve_old, vn_old, vu_old])
    v_new = new_axes @ v_ecef

    V_new = float(np.linalg.norm(v_new))
    horiz = math.hypot(float(v_new[0]), float(v_new[1]))
    psi_new = math.atan2(float(v_new[1]), float(v_new[0]))
    gamma_new = math.atan2(float(v_new[2]), horiz)

    return (
        new_geo.latitude,
        new_geo.longitude,
        new_geo.altitude,
        V_new,
        psi_new,
        gamma_new,
    )


def kinematic_divergence(
    reference: GeodeticCoordinate,
    initial_speed_mps: float,
    initial_heading_deg: float,
    initial_flight_path_deg: float,
    horizon_m: float,
    dt_s: float,
) -> KinematicEndpointDelta:
    """Propagate the same kinematic state through both methods and
    return the geodetic discrepancy at the end of ``horizon_m``."""
    V = initial_speed_mps
    psi = math.radians(initial_heading_deg)
    gamma = math.radians(initial_flight_path_deg)
    total_time_s = horizon_m / max(1e-6, V * math.cos(gamma))
    n_steps = max(1, int(round(total_time_s / dt_s)))
    dt = total_time_s / n_steps

    # Method A — fixed-ENU straight integration anchored at the runway.
    # ENU "up" is the offset from ``reference``, so the aircraft starts
    # at u=0 (i.e. at the anchor altitude) and ``enu_to_geodetic`` will
    # add the anchor altitude back during conversion.
    e_end, n_end, u_end = _rk4_fixed_enu(
        0.0, 0.0, 0.0, V, psi, gamma, dt, n_steps
    )
    fixed_geo = CoordinateConverter.enu_to_geodetic(
        ENUCoordinate(e_end, n_end, u_end), reference
    )
    fixed_heading_deg = math.degrees(psi)
    fixed_flight_path_deg = math.degrees(gamma)

    # Method B — re-anchored stepper (matches the current simulator).
    state = (
        reference.latitude,
        reference.longitude,
        reference.altitude,
        V,
        psi,
        gamma,
    )
    for _ in range(n_steps):
        state = _step_reanchored(state, dt)
    reanchored_geo = GeodeticCoordinate(state[0], state[1], state[2])
    reanchored_heading_deg = math.degrees(state[4])
    reanchored_flight_path_deg = math.degrees(state[5])

    delta_xy_m = _haversine_distance_m(fixed_geo, reanchored_geo)
    delta_alt_m = fixed_geo.altitude - reanchored_geo.altitude
    delta_psi_deg = _wrap180(fixed_heading_deg - reanchored_heading_deg)
    delta_gamma_deg = fixed_flight_path_deg - reanchored_flight_path_deg
    return KinematicEndpointDelta(
        horizontal_position_m=delta_xy_m,
        altitude_m=delta_alt_m,
        heading_deg=delta_psi_deg,
        flight_path_deg=delta_gamma_deg,
    )


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _summarise_by_radius(errors: list[PointError]) -> list[dict]:
    by_radius: dict[float, list[PointError]] = {}
    for err in errors:
        by_radius.setdefault(err.radius_m, []).append(err)

    rows = []
    for radius_m in sorted(by_radius):
        bucket = by_radius[radius_m]
        rows.append({
            "radius_km": radius_m / 1000.0,
            "curvature_drop_m": max(abs(e.curvature_drop_m) for e in bucket),
            "up_tilt_deg": max(e.up_tilt_deg for e in bucket),
            "meridian_convergence_deg": max(
                abs(e.meridian_convergence_deg) for e in bucket
            ),
            "apparent_vertical_speed_mps": max(
                abs(e.apparent_vertical_speed_mps) for e in bucket
            ),
            "horizontal_position_drift_m": max(
                e.horizontal_position_drift_m for e in bucket
            ),
        })
    return rows


def _print_summary(
    reference: GeodeticCoordinate,
    rows: list[dict],
    delta: KinematicEndpointDelta,
    horizon_km: float,
    speed_mps: float,
    heading_deg: float,
) -> None:
    print(
        f"Anchor: lat={reference.latitude:.4f}°, lon={reference.longitude:.4f}°, "
        f"alt={reference.altitude:.1f} m  (R={_EARTH_RADIUS_M/1000:.1f} km)"
    )
    print()
    print("Per-radius worst-case geometric error (over all bearings/altitudes):")
    header = (
        f"{'r [km]':>7} | {'Δh_curv [m]':>11} | {'up-tilt [°]':>11} | "
        f"{'meridian γc [°]':>15} | {'V_app⊥ [m/s] @ V=80':>20} | {'pos drift [m]':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['radius_km']:>7.2f} | "
            f"{row['curvature_drop_m']:>11.3f} | "
            f"{row['up_tilt_deg']:>11.4f} | "
            f"{row['meridian_convergence_deg']:>15.4f} | "
            f"{row['apparent_vertical_speed_mps']:>20.4f} | "
            f"{row['horizontal_position_drift_m']:>13.3f}"
        )

    print()
    print(
        f"Kinematic-divergence test  (V={speed_mps:.1f} m/s, ψ={heading_deg:.1f}°, "
        f"γ=0°, horizon={horizon_km:.1f} km):"
    )
    print(
        "  fixed-ENU vs. re-anchored stepper, same control sequence, "
        "constant-velocity point mass"
    )
    print(f"    Δhorizontal position     : {delta.horizontal_position_m:>10.3f} m")
    print(f"    Δaltitude (fixed - true) : {delta.altitude_m:>10.3f} m")
    print(f"    Δheading                 : {delta.heading_deg:>10.4f} °")
    print(f"    Δflight-path angle       : {delta.flight_path_deg:>10.4f} °")


def _write_csv(path: Path, errors: list[PointError]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "radius_m",
            "bearing_deg",
            "altitude_m",
            "curvature_drop_m",
            "up_tilt_deg",
            "meridian_convergence_deg",
            "apparent_vertical_speed_mps",
            "horizontal_position_drift_m",
        ])
        for err in errors:
            writer.writerow([
                err.radius_m,
                err.bearing_deg,
                err.altitude_m,
                err.curvature_drop_m,
                err.up_tilt_deg,
                err.meridian_convergence_deg,
                err.apparent_vertical_speed_mps,
                err.horizontal_position_drift_m,
            ])


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-lat", type=float, default=51.1138,
                        help="Anchor latitude in degrees (default CYYC)")
    parser.add_argument("--reference-lon", type=float, default=-114.0203,
                        help="Anchor longitude in degrees (default CYYC)")
    parser.add_argument("--reference-alt", type=float, default=1099.0,
                        help="Anchor altitude (m, WGS84)")
    parser.add_argument("--max-range-km", type=float, default=5.0,
                        help="Maximum horizontal range to test (km)")
    parser.add_argument("--speed-mps", type=float, default=_REFERENCE_SPEED_MPS,
                        help="Reference horizontal speed for v_⊥ scaling (m/s)")
    parser.add_argument("--heading-deg", type=float, default=0.0,
                        help="Heading for the kinematic divergence test, "
                             "measured from east (ψ=0 -> east)")
    parser.add_argument("--dt-s", type=float, default=0.5,
                        help="Integration step for the dynamics test")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional path to dump full grid as CSV")
    args = parser.parse_args(argv)

    reference = GeodeticCoordinate(
        latitude=args.reference_lat,
        longitude=args.reference_lon,
        altitude=args.reference_alt,
    )

    # Honour --max-range-km for the grid too (it used to cap only the kinematic half):
    # keep the fixed inner rings below the cap and end exactly at the cap, no duplicates.
    radii_m = [r * 1000.0 for r in (0.5, 1.0, 2.0, 3.0, 4.0) if r < args.max_range_km]
    radii_m.append(args.max_range_km * 1000.0)
    bearings_deg = list(range(0, 360, 45))
    altitudes_m = [0.0, 500.0, 1500.0]
    grid = _enumerate_grid(radii_m, bearings_deg, altitudes_m)

    errors = [compute_point_error(p, reference, args.speed_mps) for p in grid]
    rows = _summarise_by_radius(errors)

    delta = kinematic_divergence(
        reference=reference,
        initial_speed_mps=args.speed_mps,
        initial_heading_deg=args.heading_deg,
        initial_flight_path_deg=0.0,
        horizon_m=args.max_range_km * 1000.0,
        dt_s=args.dt_s,
    )

    _print_summary(
        reference=reference,
        rows=rows,
        delta=delta,
        horizon_km=args.max_range_km,
        speed_mps=args.speed_mps,
        heading_deg=args.heading_deg,
    )

    if args.csv is not None:
        _write_csv(args.csv, errors)
        print(f"\nFull grid written to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
