"""A window — one commanded scenario in its airport's recorded traffic — and the judge of a flown track.

The judge runs on the REPLAY (what ``evaluation`` grades), at every check step and at every threshold
crossing in the window (the wake rule applies when a leader is over its threshold). It sees the
commanded aircraft and the recorded aircraft within :data:`NEAR_M` of it: every minimum the rules apply
is shorter than half that (the largest, 8 NM wake = 14.8 km), and an aircraft between two others on one
final is near both, so the cut changes no loss that involves the commanded aircraft (design §5.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from geokit import NM_M

from flight_scenarios import FlightScenario
from trajectory_data_process.harvest.utc import parse_iso_utc_s

from . import rules
from .frame import TargetFrame
from .runways import Runway, established, relative, runway_frame
from .scene import RecordedFlight, Traffic

#: The judge sees the recorded aircraft within this horizontal distance of the commanded one.
NEAR_M = 30_000.0
#: The step (s) over which a recorded aircraft's track (its direction of motion) is measured.
_TRACK_STEP_S = 1.0


@dataclass(frozen=True)
class Window:
    """One commanded scenario and the recorded aircraft that can share its airspace, in one frame.
    Times are seconds from the commanded start (``t0_utc_s``)."""

    flight_key: str
    t0_utc_s: float
    frame: TargetFrame
    runways: dict[str, Runway]
    rules: rules.Separation
    runway: str
    category: str | None
    recorded: tuple[RecordedFlight, ...]
    categories: tuple[str | None, ...]
    uncategorised_types: tuple[str, ...]       # types the CWT tables do not list (judged on radar alone)


def make_window(scenario: FlightScenario, traffic: Traffic, *, procedure_root: str | Path,
                horizon_s: float) -> Window:
    """``scenario``'s window: every recorded arrival in the air within ``horizon_s`` of its start.
    The distance minima become times at the commanded aircraft's target speed (MD9)."""
    source = scenario.source
    key = source["flight_key"]
    own = traffic.flight(key)
    t0 = parse_iso_utc_s(source["entry_time_utc"])
    frame = TargetFrame(scenario.target)
    recorded = tuple(traffic.airborne(t0, t0 + horizon_s, exclude=key))
    types = [own.typecode] + [f.typecode for f in recorded]
    categories = [rules.category(t) for t in types]
    return Window(
        flight_key=key, t0_utc_s=t0, frame=frame,
        runways=runway_frame(traffic.runway_targets, frame, traffic.airport, procedure_root),
        rules=rules.separation(traffic.runway_targets, scenario.target.V),
        runway=source["runway"], category=categories[0],
        recorded=recorded, categories=tuple(categories[1:]),
        uncategorised_types=tuple(sorted({t for t, c in zip(types, categories) if t is not None and c is None})),
    )


@dataclass(frozen=True)
class FlownTrack:
    """The commanded aircraft's replay: times from the window start, MSL positions, compass track."""

    t_s: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    alt_m: np.ndarray
    track_deg: np.ndarray

    @classmethod
    def from_samples(cls, samples) -> "FlownTrack":
        """From ``scenario_replay.StateSample`` rows (ψ is math-ENU: 0 = east, counter-clockwise)."""
        return cls(*(np.array([getattr(s, name) for s in samples]) for name in ("t", "lat", "lon", "alt")),
                   track_deg=(90.0 - np.degrees([s.psi for s in samples])) % 360.0)

    def at(self, t_s: float) -> tuple[float, float, float, float]:
        """``(lat, lon, alt)`` interpolated, and the track of the next sample."""
        track = float(self.track_deg[min(np.searchsorted(self.t_s, t_s), len(self.t_s) - 1)])
        return (float(np.interp(t_s, self.t_s, self.lat_deg)), float(np.interp(t_s, self.t_s, self.lon_deg)),
                float(np.interp(t_s, self.t_s, self.alt_m)), track)


@dataclass(frozen=True)
class Conflict:
    """A loss of separation between the commanded aircraft and recorded aircraft ``other``."""

    t_s: float
    other: int                  # index into ``Window.recorded``
    kind: str                   # the judge's kind (rules.IN_TRAIL, ...)
    required_m: float
    distance_m: float
    vertical_m: float
    above: bool                 # the commanded aircraft is above the other one
    responsible: bool           # the commanded aircraft answers for it (two-tier D93)


@dataclass(frozen=True)
class Check:
    times_s: np.ndarray         # every instant judged
    conflicts: tuple[Conflict, ...]
    background: int             # losses between two recorded aircraft near the commanded one


def check_times(window: Window, end_s: float, step_s: float) -> np.ndarray:
    """The UTC multiples of ``step_s`` in ``[0, end]`` (MD6: one grid for every aircraft, as the
    two-tier scene step), the window's start and end, and every threshold crossing inside it (the wake
    rule's instants)."""
    first = math.ceil(window.t0_utc_s / step_s) * step_s - window.t0_utc_s
    landings = [f.end_utc_s - window.t0_utc_s for f in window.recorded]
    return np.unique(np.concatenate([np.arange(first, end_s, step_s), [0.0, end_s],
                                     [t for t in landings if 0.0 <= t <= end_s]]))


def _recorded_state(window: Window, k: int, t_s: float) -> tuple[float, float, float, float]:
    """``(n, e, alt, track_deg)`` of recorded aircraft ``k`` at ``t_s``."""
    flight = window.recorded[k]
    t = window.t0_utc_s + t_s
    lat, lon, alt = flight.at(np.array([t, max(t - _TRACK_STEP_S, flight.start_utc_s),
                                        min(t + _TRACK_STEP_S, flight.end_utc_s)]))
    n, e = window.frame.to_ne(lat, lon)
    return float(n[0]), float(e[0]), float(alt[0]), math.degrees(math.atan2(e[2] - e[1], n[2] - n[1])) % 360.0


def check(window: Window, flown: FlownTrack, *, reading: str, step_s: float) -> Check:
    """Judge ``flown`` against the recorded traffic at every check instant (module docstring)."""
    end_s = float(flown.t_s[-1])
    times = check_times(window, end_s, step_s)
    conflicts, background = [], 0
    for t in times:
        lat, lon, alt, track = flown.at(t)
        n0, e0 = (float(v) for v in window.frame.to_ne(lat, lon))
        rows = [(n0, e0, alt, track, window.runway, window.category)]
        index = [-1]                                            # -1: the commanded aircraft
        over = [0] if t == end_s else []
        for k, flight in enumerate(window.recorded):
            if not flight.start_utc_s <= window.t0_utc_s + t <= flight.end_utc_s:
                continue
            n, e, h, trk = _recorded_state(window, k, t)
            if math.hypot(n - n0, e - e0) > NEAR_M:
                continue
            if window.t0_utc_s + t == flight.end_utc_s:
                over.append(len(rows))
            rows.append((n, e, h, trk, flight.runway, window.categories[k]))
            index.append(k)
        for loss in rules.judge(_scene(window, rows), window.rules, reading, over):
            if 0 not in (loss.i, loss.j):
                background += 1
                continue
            j = loss.j if loss.i == 0 else loss.i
            conflicts.append(Conflict(float(t), index[j], loss.kind, loss.required_m, loss.distance_m,
                                      loss.vertical_m, rows[0][2] >= rows[j][2], 0 in loss.responsible))
    return Check(times, tuple(conflicts), background)


def _scene(window: Window, rows) -> rules.Scene:
    """The judge's view of the aircraft ``rows`` = ``(n, e, alt, track_deg, runway, category)``."""
    runways = tuple(r[4] for r in rows)
    measured = [relative(window.runways[r[4]], r[0], r[1], r[3]) for r in rows]
    before, right, off = (np.array([float(m[i]) for m in measured]) for i in range(3))
    return rules.Scene(
        e_m=np.array([r[1] for r in rows]), n_m=np.array([r[0] for r in rows]),
        height_m=np.array([r[2] for r in rows]), runway=runways,
        along_m=np.array([window.rules.along_nm[r] * NM_M for r in runways]) - before,
        track_minus_course_deg=off, right_of_course_m=right,
        established=np.array([bool(established(window.runways[r], b, rr, o))
                              for r, b, rr, o in zip(runways, before, right, off)]),
        category=tuple(r[5] for r in rows),
    )
