"""One flight's per-step signals in the airport frame — what the labeller reads, and what the
prior will read as its state input.

Built from the ts data plane's `FlightSeries` (read-time repair, MSL datum, 2 s resampling, cut
at the observed threshold crossing), so the labelled population and its preprocessing are the
models' own. The channels are turned back into physical states by `data.channels.states_from_channels`
(the exact inverse of the build) and then projected into the airport frame.

Stored raw: smoothing is a reading parameter of the spec, applied by the labeller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ts_transformer.data.channels import states_from_channels
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.words import compass_from_math_rad, wrap180

#: v2 (2026-09-24): ``typecode`` is the flight's own ICAO type (``None`` when its identity is
#: unresolved); v1 carried the type the dynamics flew, an A320 for every type without dynamics.
SIGNALS_SCHEMA = "ts-instruction-signals-v2"

#: The per-row arrays, in storage order.
ROW_FIELDS = ("time_s", "e_m", "n_m", "altitude_m", "track_deg", "ground_speed_mps", "vertical_rate_mps")


@dataclass
class FlightSignals:
    dataset_id: str
    airport: str
    runway: str
    typecode: str | None          # the flight's ICAO type; None = identity unresolved
    time_s: np.ndarray            # [N] seconds from the built flight's first row, uniform step
    e_m: np.ndarray               # [N] airport frame, metres
    n_m: np.ndarray
    altitude_m: np.ndarray        # [N] geometric MSL
    track_deg: np.ndarray         # [N] compass, true north, UNWRAPPED along time
    ground_speed_mps: np.ndarray  # [N]
    vertical_rate_mps: np.ndarray  # [N] climbing positive

    @property
    def n_rows(self) -> int:
        return len(self.time_s)

    def horizontal_distance_m(self) -> np.ndarray:
        """Cumulative horizontal path from the first row (trapezoid on ground speed)."""
        dt = np.diff(self.time_s)
        steps = 0.5 * (self.ground_speed_mps[1:] + self.ground_speed_mps[:-1]) * dt
        return np.concatenate(([0.0], np.cumsum(steps)))


def signals_from_series(series: Any, geometry: AirportGeometry) -> FlightSignals:
    """``series`` is a `data.dataset.FlightSeries` (not imported: `dataset` imports torch, and
    this group stays torch-free)."""
    if series.airport != geometry.code:
        raise ValueError(f"{series.dataset_id} lands at {series.airport}, not {geometry.code}")
    runway = str(series.scenario.source["runway"]).upper()
    candidate = geometry.candidates[geometry.candidate_index(runway)]
    target = series.scenario.target
    target_e, target_n = geometry.frame.horizontal_from_latlon(target.latitude, target.longitude)
    course_error = float(wrap180(float(compass_from_math_rad(target.psi)) - candidate.course_deg))
    if math.hypot(target_e - candidate.threshold_e_m, target_n - candidate.threshold_n_m) > 1.0 or abs(course_error) > 0.01:
        raise ValueError(f"{series.dataset_id}: the flight's own target is not candidate {runway}'s threshold and course")
    states = states_from_channels(series.times, series.values, series.frame, mass_kg=series.scenario.initial.m)
    lat = np.array([state.latitude for _, state in states])
    lon = np.array([state.longitude for _, state in states])
    altitude = np.array([state.altitude for _, state in states])
    speed = np.array([state.V for _, state in states])
    psi = np.array([state.psi for _, state in states])
    gamma = np.array([state.gamma for _, state in states])
    e, n = geometry.frame.horizontal_from_latlon(lat, lon)
    track = np.degrees(np.unwrap(np.radians(compass_from_math_rad(psi))))
    return FlightSignals(
        dataset_id=series.dataset_id,
        airport=series.airport,
        runway=runway,
        typecode=series.scenario.source["resolved_typecode"],
        time_s=np.asarray(series.times, dtype=np.float64),
        e_m=np.asarray(e, dtype=np.float64),
        n_m=np.asarray(n, dtype=np.float64),
        altitude_m=altitude,
        track_deg=track,
        ground_speed_mps=speed * np.cos(gamma),
        vertical_rate_mps=speed * np.sin(gamma),
    )


def pack_signals(items: Sequence[FlightSignals]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Flights → concatenated row arrays (+ ``offsets``) and per-flight metadata."""
    lengths = np.array([item.n_rows for item in items], dtype=np.int64)
    arrays = {"offsets": np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)}
    for name in ROW_FIELDS:
        arrays[name] = (np.concatenate([getattr(item, name) for item in items]) if items
                        else np.zeros(0)).astype(np.float64)
    meta = [{"dataset_id": item.dataset_id, "airport": item.airport, "runway": item.runway,
             "typecode": item.typecode} for item in items]
    return arrays, meta


def unpack_signals(arrays: dict[str, np.ndarray], meta: Sequence[dict[str, Any]]) -> list[FlightSignals]:
    offsets = arrays["offsets"]
    if len(offsets) != len(meta) + 1:
        raise ValueError(f"{len(meta)} flights but {len(offsets) - 1} row ranges")
    flights = []
    for index, info in enumerate(meta):
        rows = slice(int(offsets[index]), int(offsets[index + 1]))
        flights.append(FlightSignals(**info, **{name: np.asarray(arrays[name][rows]) for name in ROW_FIELDS}))
    return flights
