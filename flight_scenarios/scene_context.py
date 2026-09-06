"""The traffic scene around an ego flight at its anchor time (scene design §五 P2.b).

Given the harvest, its scene index, the ego's runway target and the anchor time t₀,
``scene_context`` returns the OTHER aircraft airborne in the window before t₀ — each
with only what was observable then — and a few scalars about the runway's recent use.

The leakage line, stricter than the design's §3.2 (which listed "which runway it will
land on" as a neighbour category): for an aircraft still airborne at t₀, its landing
time AND its eventual runway / outcome are the future. What a neighbour contributes is
what its samples up to t₀ show — position and velocity in the ego's threshold chart,
height above the ego's threshold (HAE minus the threshold's HAE elevation: a relative
height, no datum conversion), distance to that threshold, an ETA from its current
ground speed, whether it is established on the ego runway's final (the on-final
membership: inside the full-scale cone floored at 500 m, aligned within 30°, upstream)
and its order ahead of the ego by ETA. Landed aircraft (landing ≤ t₀, the past) enter
only the scalars: time since the last landing on the ego's runway, landings in the last
30 min and the ego runway's share of them. Each neighbour ALSO carries a
``future_label`` (its outcome, runway, landing time from the roster) for auxiliary
supervision — a label, never a feature; ``scene/features`` must not read it.

Window ``WINDOW_S`` before t₀, radius ``RADIUS_M`` from the ego's threshold at the
neighbour's last sample, the ``N_MAX`` nearest to the ego kept (nearest first) — all
stated in the returned ``SceneContext``. Track files are read through
``harvest.store.read_track_view`` (the derived, altitude-repaired view) behind an LRU
cache sized for one airport's cohort pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
import math
from typing import Any, Sequence

import numpy as np

from final_approach.frame import RunwayFrame, TrackPoint
from trajectory_data_process.harvest.store import HarvestPaths, read_track_view
from trajectory_data_process.scene_index import IndexEntry, SceneIndex, parse_utc_s

from .fas_geometry import course_halfwidth_m, fas_course_geometry

WINDOW_S = 120.0
RADIUS_M = 40_000.0
N_MAX = 16
RECENT_LANDINGS_S = 1_800.0
SPEED_FLOOR_MPS = 1.0
# Mirrors of 4dTrajectory/ts_transformer/final_approach_geometry.py's membership rule
# (MEMBERSHIP_K, MEMBERSHIP_FLOOR_M, ALIGNMENT_MAX_DEG): that package is not importable
# from here; tests/test_scene_context.py on the ts side pins the two to each other.
MEMBERSHIP_K = 1.0
MEMBERSHIP_FLOOR_M = 500.0
ALIGNMENT_MAX_DEG = 30.0
_FAS = fas_course_geometry()
_TRACK_CACHE_SIZE = 4_096


def ego_frame(runway: str, target: dict[str, Any]) -> RunwayFrame:
    """The ego runway's threshold frame from an arrivals manifest ``runway_targets`` row
    (HAE elevation: track altitudes are HAE, so heights come out relative)."""
    return RunwayFrame(ident=runway, lat=float(target["lat"]), lon=float(target["lon"]),
                       elevation_m=float(target["elevation_hae_m"]), course_deg=float(target["course_deg"]))


def runway_axes(frame: RunwayFrame, lat: float, lon: float, alt_hae_m: float) -> tuple[float, float, float]:
    """``(d, xt, height)``: the ts chart's runway axes — ``d`` upstream of the threshold
    (positive before it), ``xt`` right of the inbound course — from the frame's
    ``(along, cross, height)``."""
    p = frame.project(TrackPoint(lat, lon, alt_hae_m))
    return -p.along_m, p.cross_m, p.height_m


def on_final(d: float, xt: float, heading_error_rad: float) -> bool:
    """The on-final membership at one sample: upstream, inside the floored full-scale
    cone, heading within ``ALIGNMENT_MAX_DEG`` of the inbound course."""
    halfwidth = max(MEMBERSHIP_K * course_halfwidth_m(max(d, 0.0), _FAS), MEMBERSHIP_FLOOR_M)
    return d > 0.0 and abs(xt) <= halfwidth and math.cos(heading_error_rad) >= math.cos(math.radians(ALIGNMENT_MAX_DEG))


@dataclass(frozen=True)
class Observed:
    """What a neighbour's samples up to t₀ show, in the ego's threshold chart."""
    t_rel_s: np.ndarray        # [M] sample times relative to t₀ (≤ 0), oldest first
    e_m: np.ndarray            # [M] chart east
    n_m: np.ndarray            # [M] chart north
    height_m: np.ndarray       # [M] above the ego's threshold (HAE − HAE)
    d_m: float                 # at the last sample
    xt_m: float
    ground_speed_mps: float    # from the last two samples (position-derived, like the ego's)
    heading_rad: float         # math-ENU
    distance_to_threshold_m: float
    eta_s: float               # distance / ground speed (the closing rate is not assumed)
    established: bool          # on the ego runway's final at the last sample
    age_s: float               # t₀ − last sample time


@dataclass(frozen=True)
class FutureLabel:
    """The roster's word on what the neighbour went on to do — supervision only."""
    outcome: str
    runway: str | None
    landing_utc_s: float | None


@dataclass(frozen=True)
class Neighbour:
    flight_key: str
    icao24: str
    distance_to_ego_m: float
    observed: Observed
    future_label: FutureLabel


@dataclass(frozen=True)
class Scalars:
    since_last_landing_same_runway_s: float | None
    landings_recent: int
    landings_recent_same_runway: int
    same_runway_share_recent: float | None
    airborne_in_radius: int
    established_on_ego_final: int
    ahead_by_eta: int
    lead_eta_s: float | None            # the nearest ETA ahead of the ego's (established or not)
    lead_gap_s: float | None            # ego ETA − that lead's ETA
    hour_utc: float
    weekday: int


@dataclass(frozen=True)
class SceneContext:
    airport: str
    ego_flight_key: str
    t0_utc_s: float
    ego_eta_s: float
    neighbours: tuple[Neighbour, ...]    # nearest to the ego first, ≤ N_MAX
    scalars: Scalars
    window_s: float = WINDOW_S
    radius_m: float = RADIUS_M
    n_max: int = N_MAX
    candidates_in_window: int = 0        # airborne in the window before the radius / N_MAX cut
    in_radius: int = 0


@dataclass
class _Reader:
    paths: HarvestPaths

    def __post_init__(self) -> None:
        self._view = lru_cache(maxsize=_TRACK_CACHE_SIZE)(lambda relative: read_track_view(self.paths, relative))

    def samples(self, entry: IndexEntry) -> tuple[np.ndarray, np.ndarray]:
        """``(absolute_utc_s [M], samples [M, 4] as t, lon, lat, alt_hae)`` of a track."""
        track = self._view(entry.file)
        samples = np.asarray(track["samples"], dtype=np.float64)
        return parse_utc_s(track["start_time_utc"]) + samples[:, 0], samples


def _observed(frame: RunwayFrame, course_rad: float, utc: np.ndarray, samples: np.ndarray,
              t0: float, window_s: float) -> Observed | None:
    keep = (utc <= t0) & (utc >= t0 - window_s)
    if keep.sum() < 2:
        return None
    utc, samples = utc[keep], samples[keep]
    axes = np.array([runway_axes(frame, s[2], s[1], s[3]) for s in samples])     # d, xt, h
    d, xt, height = axes[:, 0], axes[:, 1], axes[:, 2]
    e, n = -d * math.cos(course_rad) + xt * math.sin(course_rad), -d * math.sin(course_rad) - xt * math.cos(course_rad)
    dt = utc[-1] - utc[-2]
    ve, vn = (e[-1] - e[-2]) / max(dt, 1e-3), (n[-1] - n[-2]) / max(dt, 1e-3)
    speed = max(math.hypot(ve, vn), SPEED_FLOOR_MPS)
    heading = math.atan2(vn, ve)
    distance = math.hypot(e[-1], n[-1])
    return Observed(
        t_rel_s=utc - t0, e_m=e, n_m=n, height_m=height, d_m=float(d[-1]), xt_m=float(xt[-1]),
        ground_speed_mps=speed, heading_rad=heading, distance_to_threshold_m=distance,
        eta_s=distance / speed, established=on_final(float(d[-1]), float(xt[-1]), heading - course_rad),
        age_s=float(t0 - utc[-1]),
    )


def scene_context(
    paths: HarvestPaths,
    index: SceneIndex,
    *,
    ego_flight_key: str,
    ego_runway: str,
    ego_target: dict[str, Any],
    t0_utc_s: float,
    ego_lat: float,
    ego_lon: float,
    ego_alt_hae_m: float,
    ego_ground_speed_mps: float,
    window_s: float = WINDOW_S,
    radius_m: float = RADIUS_M,
    n_max: int = N_MAX,
    reader: "_Reader | None" = None,
) -> SceneContext:
    """The scene at t₀: see the module docstring for what enters and what does not."""
    reader = reader or _Reader(paths)
    frame = ego_frame(ego_runway, ego_target)
    course_rad = math.radians(90.0 - float(ego_target["course_deg"]))      # compass → math-ENU
    d_ego, xt_ego, _ = runway_axes(frame, ego_lat, ego_lon, ego_alt_hae_m)
    e_ego, n_ego = -d_ego * math.cos(course_rad) + xt_ego * math.sin(course_rad), -d_ego * math.sin(course_rad) - xt_ego * math.cos(course_rad)
    ego_eta = math.hypot(e_ego, n_ego) / max(ego_ground_speed_mps, SPEED_FLOOR_MPS)

    candidates = [e for e in index.airborne_at(t0_utc_s, window_s) if e.flight_key != ego_flight_key]
    neighbours: list[Neighbour] = []
    for entry in candidates:
        utc, samples = reader.samples(entry)
        observed = _observed(frame, course_rad, utc, samples, t0_utc_s, window_s)
        if observed is None or observed.distance_to_threshold_m > radius_m:
            continue
        neighbours.append(Neighbour(
            flight_key=entry.flight_key, icao24=entry.icao24,
            distance_to_ego_m=math.hypot(observed.e_m[-1] - e_ego, observed.n_m[-1] - n_ego),
            observed=observed,
            future_label=FutureLabel(entry.outcome, entry.runway, entry.landing_utc_s),
        ))
    in_radius = len(neighbours)
    neighbours.sort(key=lambda nb: nb.distance_to_ego_m)
    neighbours = neighbours[:n_max]

    recent = index.landings_before(t0_utc_s, since_s=RECENT_LANDINGS_S)
    same = [t for t, runway, _ in index.landings_before(t0_utc_s) if runway == ego_runway]
    ahead = [nb.observed.eta_s for nb in neighbours if nb.observed.eta_s < ego_eta]
    when = datetime.fromtimestamp(t0_utc_s, tz=timezone.utc)
    recent_same = sum(runway == ego_runway for _, runway, _ in recent)
    scalars = Scalars(
        since_last_landing_same_runway_s=(t0_utc_s - same[-1]) if same else None,
        landings_recent=len(recent), landings_recent_same_runway=recent_same,
        same_runway_share_recent=(recent_same / len(recent)) if recent else None,
        airborne_in_radius=in_radius,
        established_on_ego_final=sum(nb.observed.established for nb in neighbours),
        ahead_by_eta=len(ahead),
        lead_eta_s=max(ahead) if ahead else None,
        lead_gap_s=(ego_eta - max(ahead)) if ahead else None,
        hour_utc=when.hour + when.minute / 60.0, weekday=when.weekday(),
    )
    return SceneContext(
        airport=index.airport, ego_flight_key=ego_flight_key, t0_utc_s=t0_utc_s, ego_eta_s=ego_eta,
        neighbours=tuple(neighbours), scalars=scalars, window_s=window_s, radius_m=radius_m, n_max=n_max,
        candidates_in_window=len(candidates), in_radius=in_radius,
    )
