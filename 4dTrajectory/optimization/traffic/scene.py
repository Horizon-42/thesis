"""The recorded traffic of an airport: every arrival of the manifest, flying its record.

The arrivals come through ``flight_scenarios.build.load_model_arrivals`` — the loader the scenarios
came through — so a recorded altitude and a commanded scenario share one datum (MSL). A flight is in
the air from its first sample (its entry into the arrival slice) to its last (its threshold crossing);
between samples its position is the linear interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aircraft.identity import get_default_identity_resolver
from flight_scenarios.identity import flight_key
from trajectory_data_process.harvest.utc import parse_iso_utc_s


@dataclass(frozen=True)
class RecordedFlight:
    """One recorded arrival: UTC sample times (s) and its MSL track."""

    flight_key: str
    runway: str
    typecode: str | None          # None: the identity resolver found no ICAO type
    t_utc_s: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    alt_m: np.ndarray

    @property
    def start_utc_s(self) -> float:
        return float(self.t_utc_s[0])

    @property
    def end_utc_s(self) -> float:
        return float(self.t_utc_s[-1])

    def at(self, t_utc_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(lat, lon, alt)`` at the given UTC times, linear between samples (inside the span only)."""
        return (np.interp(t_utc_s, self.t_utc_s, self.lat_deg),
                np.interp(t_utc_s, self.t_utc_s, self.lon_deg),
                np.interp(t_utc_s, self.t_utc_s, self.alt_m))


def recorded_flight(flight: dict) -> RecordedFlight:
    """A manifest arrival (as ``load_model_arrivals`` returns it) as a :class:`RecordedFlight`."""
    rows = np.asarray(flight["waypoints"], dtype=float)          # [offset_s, lon, lat, alt]
    identity = get_default_identity_resolver().resolve(declared_type=flight["type"], icao24=flight["icao24"])
    return RecordedFlight(
        flight_key=flight_key(flight, 0),      # the key the scenarios carry (build_scenario)
        runway=flight["runway"],
        typecode=identity.typecode,
        t_utc_s=parse_iso_utc_s(flight["entry_time_utc"]) + rows[:, 0],
        lat_deg=rows[:, 2],
        lon_deg=rows[:, 1],
        alt_m=rows[:, 3],
    )


@dataclass(frozen=True)
class Traffic:
    """An airport's recorded arrivals and the runway targets they land on (``runway_targets``:
    runway -> ``{lat, lon, course_deg, ...}``, the manifest's)."""

    airport: str
    flights: tuple[RecordedFlight, ...]
    runway_targets: dict

    def flight(self, flight_key: str) -> RecordedFlight:
        """The recorded flight ``flight_key`` (a commanded scenario's own record)."""
        found = [f for f in self.flights if f.flight_key == flight_key]
        if len(found) != 1:
            raise KeyError(f"{len(found)} recorded flight(s) with flight_key {flight_key!r} at {self.airport}")
        return found[0]

    def airborne(self, start_utc_s: float, end_utc_s: float, *, exclude: str) -> list[RecordedFlight]:
        """The flights in the air at some time in ``[start, end]``, less the flight ``exclude``."""
        return [f for f in self.flights
                if f.start_utc_s <= end_utc_s and f.end_utc_s >= start_utc_s and f.flight_key != exclude]


def traffic_from_arrivals(flights: list[dict]) -> Traffic:
    """The traffic of an airport's arrivals as ``load_model_arrivals`` returns them (the live harvest
    root, never ``harvest-heldout``): MSL, one airport. Writes nothing."""
    airports = {f["arr_airport"] for f in flights}
    if len(airports) != 1:
        raise ValueError(f"one airport per manifest, found {sorted(airports)}")
    targets = {}
    for f in flights:
        targets.setdefault(f["runway"], f["runway_target"])
    return Traffic(airports.pop(), tuple(recorded_flight(f) for f in flights), targets)
