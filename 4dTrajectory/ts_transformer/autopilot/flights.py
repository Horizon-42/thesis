"""A labelled flight's inputs, rebuilt from the data plane (executor design §2.4), and where the executor starts it.

The artefact stores each flight's signals, not its dynamics. The aircraft's aero row, installed
thrust and mass come from the data plane's own `FlightSeries`,
rebuilt here with the call that built the signals (`data.dataset.build_series` under the default
`TSConfig`, from the airport's live arrival manifest),
its first row where the stored signals have it (`stored_rows`: an artefact's rows are on the UTC
steps since 2026-10-02, an older one's at each flight's first sample — both rebuild row for row).
The rebuilt flight must reproduce the stored signals row for row and name the same runway and
aircraft (`instructions.signals.signals_from_series`), and the build configuration must be the one the
signals recorded, or it is refused: an executor flown from another flight — or another airframe — than
the one the sentence was read off answers nothing. The flight is identified by what it is, never by the
bytes of the manifest it came from (vocabulary §7.2 #4, D21: a manifest rewritten over the same flights opens).

THE START STATE (vocabulary D77; the user's choices of 2026-10-05). The executor starts a flight at one of its observed
2 s rows (a sentence's first row, or a closed loop's first predicted step) and reads no sample after it: the position and
the MSL height of the row, and the velocity by the executor spec's start rule (`params.START_RULES`,
`start_velocity`) — the slope of a least-squares line through the observed rows of the rule's window before the row, the
row included. A window is cut at the flight's row 0 (no row before it exists); row 0 itself, alone in its window, takes
the line through rows 0 and 1 (the user's choice: every row has a value, and none reads past a closed loop's first
predicted step, 16 s later). The observed positions are the airport frame's, so the slope is turned into metres per
second on the ground at the row's latitude and height (WGS84). No wind: the airspeed is the ground speed and the vertical
rate together, as the speed law reads them (vocabulary §5.6). Every start is this one: a replay's from its sentence's
first row, a closed loop's (the closed-loop reading, a speaker's, the live executor) from its first predicted step. The
observed rows a closed-loop sentence stores take their track, ground speed and vertical rate from the same rule
(`observed_rows`, D77, D82), the track in [0°, 360°).

THE FRAME (vocabulary D81). The dynamics integrate in a chart whose origin is the airport reference, at the airport
elevation E (`AirportGeometry.frame`), the same for every flight of an airport: the executor holds nothing of the landed
runway or of a procedure.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from geokit import METRES_PER_DEG_LAT, wgs84_curvature_radii
from ts_transformer.autopilot.params import START_RULES
from ts_transformer.config import TSConfig
from ts_transformer.data.dataset import FlightSeries, RowStart, build_series, load_flight_dicts
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import load_candidates
from ts_transformer.instructions.signals import ROW_FIELDS, FlightSignals, signals_from_series
from ts_transformer.repo_layout import arrival_manifest_path
from trajectory_data_process.harvest.utc import parse_iso_utc_s

#: How closely a rebuilt flight must reproduce the stored signals (the same code on the same
#: manifest reproduces them exactly; the tolerance absorbs only floating-point reassociation).
SIGNAL_TOLERANCE = 1e-6


def rebuild_series(directory: Path, signals: Sequence[FlightSignals]) -> list[FlightSeries]:
    """The `FlightSeries` behind each of ``signals``, in the same order, each checked to be the flight the
    signals were read from (`require_same_flight`)."""
    record = json.loads((directory / "signals.json").read_text(encoding="utf-8"))
    config = TSConfig()
    changed = {name: (value, getattr(config, name)) for name, value in record["config"].items()
               if getattr(config, name) != value}
    if changed:
        raise ValueError(f"the build configuration moved since the signals were read: {changed}")
    geometries = load_candidates(directory)
    built: dict[str, FlightSeries] = {}
    for airport in sorted({item.airport for item in signals}):
        manifest = arrival_manifest_path(airport)
        keys = {item.dataset_id for item in signals if item.airport == airport}
        flights = load_flight_dicts([manifest], include_flight_keys=keys, verbose=False)
        series, _report = build_series(flights, config, row_start=stored_rows(signals))
        built.update({item.dataset_id: item for item in series})
    missing = [item.dataset_id for item in signals if item.dataset_id not in built]
    if missing:
        raise ValueError(f"{len(missing)} flight(s) did not rebuild, e.g. {missing[:3]}")
    for item in signals:
        require_same_flight(built[item.dataset_id], item, geometries[item.airport])
    return [built[item.dataset_id] for item in signals]


def stored_rows(signals: Sequence[FlightSignals]) -> RowStart:
    """The first row where ``signals`` has it: its row 0's UTC (``entry_time_utc``) after the flight's first kept sample
    (the arrival record's ``entry_time_utc``)."""
    row_zero = {item.dataset_id: parse_iso_utc_s(item.entry_time_utc) for item in signals}

    def start(dataset_id: str, source: Mapping[str, Any]) -> float:
        return row_zero[dataset_id] - parse_iso_utc_s(source["entry_time_utc"])
    return start


def require_same_flight(series: FlightSeries, signals: FlightSignals, geometry: AirportGeometry) -> None:
    """The rebuilt flight reproduces the stored signals: the same runway, aircraft and clock, every row of
    every field."""
    again = signals_from_series(series, geometry)
    for name in ("runway", "typecode", "entry_time_utc", "landing_time_utc"):
        if getattr(again, name) != getattr(signals, name):
            raise ValueError(f"{signals.dataset_id}: the rebuilt flight's {name} is {getattr(again, name)!r}, "
                             f"the stored signals' {getattr(signals, name)!r}")
    for name in ROW_FIELDS:
        stored, rebuilt = getattr(signals, name), getattr(again, name)
        if stored.shape != rebuilt.shape or not np.allclose(stored, rebuilt, rtol=0.0, atol=SIGNAL_TOLERANCE):
            raise ValueError(f"{signals.dataset_id}: the rebuilt flight's {name} differs from the stored signals")


def start_velocity(signals: FlightSignals, rows: Sequence[int], rule: str, geometry: AirportGeometry) -> np.ndarray:
    """``[len(rows), 3]``: the velocity east, north and up, m/s on the ground, at each of the observed flight's 2 s
    ``rows`` (from its row 0) by the start rule ``rule`` (module docstring): the slope of the least-squares line through
    the rows at or before the row within the rule's window, cut at row 0 (row 0: rows 0 and 1); the data plane's own
    velocity for ``centred-fit-15s``."""
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) and not (0 <= rows.min() and rows.max() < signals.n_rows):
        raise ValueError(f"{signals.dataset_id}: rows {rows.min()}..{rows.max()} outside its {signals.n_rows} rows")
    if rule not in START_RULES:
        raise ValueError(f"start rule {rule!r} is none of {tuple(START_RULES)}")
    window_s = START_RULES[rule]
    if window_s is None:                                  # the data plane's centred fit (A33's comparison only)
        track = np.radians(signals.track_deg[rows])
        ground = signals.ground_speed_mps[rows]
        return np.column_stack([ground * np.sin(track), ground * np.cos(track), signals.vertical_rate_mps[rows]])
    time_s = signals.time_s
    last = np.maximum(rows, 1)
    first = np.minimum(np.searchsorted(time_s, time_s[rows] - window_s - 1e-9, side="left"), last - 1)
    width = int((last - first).max(initial=0)) + 1
    at = first[:, None] + np.arange(width)[None, :]
    inside = at <= last[:, None]
    at = np.minimum(at, last[:, None])
    t = np.where(inside, time_s[at] - time_s[last][:, None], 0.0)
    counted = inside.sum(axis=1)
    t_mean = t.sum(axis=1) / counted
    spread = np.where(inside, t - t_mean[:, None], 0.0)

    def slope(values: np.ndarray) -> np.ndarray:
        x = np.where(inside, values[at], 0.0)
        x_mean = x.sum(axis=1) / counted
        return (spread * (x - x_mean[:, None])).sum(axis=1) / (spread * spread).sum(axis=1)

    east, north, up = slope(signals.e_m), slope(signals.n_m), slope(signals.altitude_m)
    to_east, to_north = ground_scale(signals, rows, geometry)
    return np.column_stack([east * to_east, north * to_north, up])


def ground_scale(signals: FlightSignals, rows: Sequence[int], geometry: AirportGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Metres on the ground per metre of the airport frame, east and north, at each of the observed flight's ``rows``
    (`AirportENUFrame`: an equirectangular chart; the WGS84 radii of curvature at the row's latitude and height, D87)."""
    rows = np.asarray(rows, dtype=np.int64)
    frame = geometry.frame
    lat_deg = frame.lat0 + signals.n_m[rows] / METRES_PER_DEG_LAT
    radii = np.array([wgs84_curvature_radii(float(lat)) for lat in lat_deg]).reshape(-1, 2)
    height = signals.altitude_m[rows]
    to_east = np.radians(1.0 / frame.m_per_deg_lon) * (radii[:, 1] + height) * np.cos(np.radians(lat_deg))
    to_north = np.radians(1.0 / METRES_PER_DEG_LAT) * (radii[:, 0] + height)
    return to_east, to_north


def observed_rows(signals: FlightSignals, rows: Sequence[int], rule: str, geometry: AirportGeometry) -> np.ndarray:
    """The observed flight's `instructions.artefact.STATE_COLUMNS` at its 2 s ``rows`` (what a closed-loop sentence holds
    before its first predicted step): the position and MSL height of each row, and the track (compass, [0°, 360°)),
    ground speed and vertical rate by the start rule (`start_velocity`, D77, D82)."""
    rows = np.asarray(rows, dtype=np.int64)
    velocity = start_velocity(signals, rows, rule, geometry)
    track = np.remainder(np.degrees(np.arctan2(velocity[:, 0], velocity[:, 1])), 360.0)
    return np.column_stack([signals.e_m[rows], signals.n_m[rows], signals.altitude_m[rows], track,
                            np.hypot(velocity[:, 0], velocity[:, 1]), velocity[:, 2]])


def start_state(signals: FlightSignals, row: int, rule: str, geometry: AirportGeometry, mass_kg: float) -> np.ndarray:
    """The geodetic state ``(lat, lon, alt, V, ψ, γ, m)`` (`aerodynamic_model.torch_dynamics.STATE_NAMES`, ψ in the math
    convention) the executor starts the flight at, at its 2 s row ``row`` (module docstring)."""
    east, north, up = start_velocity(signals, [row], rule, geometry)[0]
    lat, lon = geometry.frame.latlon_from_horizontal(float(signals.e_m[row]), float(signals.n_m[row]))
    ground = math.hypot(east, north)
    return np.array([lat, lon, float(signals.altitude_m[row]), math.hypot(ground, up), math.atan2(north, east),
                     math.atan2(up, ground), mass_kg], dtype=np.float64)


def frame_params(geometry: AirportGeometry) -> np.ndarray:
    """The chart the dynamics integrate in (`aerodynamic_model.torch_transport_chart_dynamics`, ``[lat0, lon0, alt0,
    heading]``): the airport reference at E, unrotated (module docstring, D81)."""
    return np.array([geometry.frame.lat0, geometry.frame.lon0, geometry.elevation_m, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class FlightInputs:
    """The batch's physical context, float64, one row per flight."""

    initial_state: torch.Tensor     # [B,7] geodetic, at the row the flight is flown from
    aero_params: torch.Tensor       # [B,6]
    frame_params: torch.Tensor      # [B,4] the chart the rollout integrates in: the airport's (D81)
    max_thrust_n: torch.Tensor      # [B]


def flight_inputs(series: Sequence[FlightSeries], signals: Sequence[FlightSignals], anchors: Sequence[int],
                  geometries: Sequence[AirportGeometry], rule: str, *, device: torch.device) -> FlightInputs:
    """Each flight's physical context, flown from its own 2 s row ``anchors[i]`` of its observed flight ``signals[i]``
    (from its row 0): its sentence's first row (a replay), or a closed loop's first predicted step; the start state by
    the start rule ``rule`` (`start_state`), the airframe of its rebuilt series, the airport's chart."""
    if not len(series) == len(signals) == len(anchors) == len(geometries):
        raise ValueError(f"{len(series)} flights, {len(signals)} observed flights, {len(anchors)} anchors and "
                         f"{len(geometries)} airports")
    states, aero, thrust = [], [], []
    for item, flight, anchor, geometry in zip(series, signals, anchors, geometries):
        if item.dataset_id != flight.dataset_id:
            raise ValueError(f"the series of {item.dataset_id} given with the observed flight {flight.dataset_id}")
        aircraft, airframe = item.scenario.dynamics("a rollout")
        states.append(start_state(flight, int(anchor), rule, geometry, float(item.scenario.initial.m)))
        aero.append([airframe.S, airframe.Cl_max, airframe.Cd0, airframe.k, airframe.stall_threshold, airframe.k_stall])
        thrust.append(float(aircraft.engine.max_thrust_total_n))

    def stack(rows: Any) -> torch.Tensor:
        return torch.as_tensor(np.asarray(rows, dtype=np.float64), dtype=torch.float64, device=device)

    return FlightInputs(initial_state=stack(states), aero_params=stack(aero),
                        frame_params=stack([frame_params(g) for g in geometries]), max_thrust_n=stack(thrust))
