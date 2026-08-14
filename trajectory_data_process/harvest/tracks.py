"""OpenSky state vectors -> one contiguous flight track.

This replaces ``trajectory_data_process/trajectory.py``'s segmentation, which the audit
of the shipped artifacts showed producing three distinct defects. Each fix here is at
the root, not a filter bolted downstream.

DEFECT 1 -- A TRACK GLUED TO A LATER PASS (measured: 6 tracks)
--------------------------------------------------------------
An aircraft that lands keeps transmitting from the gate, so its state vectors are
CONTINUOUS across the turnaround and the old >900 s gap rule never fired. The spatial
crop then removed the middle (the aircraft was outside the radius) and left two chunks
glued together: KSTL AAL2717 carried a **6598 s** hole with two stray samples after it,
and because the runway anchor was chosen as "closest sample to the threshold", the
flight's recorded ``landing_time_utc`` came from the WRONG pass -- and landing time is
part of the flight's identity.

Fixed twice over: a sustained ON-GROUND run now ends the flight (a landing rollout is
where an arrival ends, which is the physical truth the gap rule was approximating), and
after cropping only the FINAL CONTIGUOUS run survives, so a spatial filter can never
again invent a discontinuity.

DEFECT 2 -- ONE APPROACH SPLIT IN TWO
-------------------------------------
The old rule split a track whenever the departure/arrival metadata "changed", with
"complete" defined as *either* field being non-null. So ``(None, 'KRDU')`` ->
``('KATL', 'KRDU')`` -- the same flight, with the origin estimate merely filled in
mid-track -- was read as a different flight and cut in half. Here both fields must be
known on both sides before a change counts. (Probed on a sample hour this never fired,
so it was latent rather than active; it is fixed because a latent cut through a final
approach is not something to leave armed.)

DEFECT 3 -- THE DATUM
---------------------
Altitudes stay exactly as the sensor reported them: geometric = height above the WGS84
ellipsoid (HAE). Nothing is converted here. The harvest is the faithful record; the
datum choice belongs to the consumer, and ``harvest.airports.Runway.frame`` makes it an
explicit argument.

Samples without a geometric altitude are dropped (measured: 22 of 79,747 airborne rows,
0.03% -- and 18 of 23 on-ground rows, which is why the ground rule above matters more
than this one).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any, Iterator, Literal, Sequence

from geokit import FT_M, haversine_km

# Altitude units of the incoming rows. There is no default and no sniffing, because the
# two real sources disagree and the error is silent: ``fetch_history_dataframe`` already
# converts to metres, while the archived ``history_rows/**.jsonl`` responses hold the
# feet that OpenSky sent. Guessing wrong scales every altitude by 3.28 -- which does not
# crash, it just turns a 3 deg approach into a 9.9 deg one (observed, on this very file).
AltitudeUnits = Literal["m", "ft"]

_UNIT_SCALE: dict[str, float] = {"m": 1.0, "ft": FT_M}

# A time hole this large means the receivers lost the aircraft for long enough that
# what follows is a different flight, not a gap in this one.
DEFAULT_MAX_GAP_S = 900.0
# Consecutive seconds on the ground that end a flight. Long enough that a bounced
# on-ground bit during the flare cannot cut a landing short, far shorter than any
# turnaround.
DEFAULT_GROUND_SPLIT_S = 60.0
# Samples the harvest keeps around the field. Must exceed the assignment window's
# outer edge (5 km) with room for the approach that leads into it.
DEFAULT_CROP_RADIUS_KM = 30.0
DEFAULT_MIN_SAMPLES = 10
# OpenSky's documented stale-vector exclusion is ``time - lastcontact <= 15``.
# Position is an independent state-vector component, so the same bound is applied to
# ``lastposupdate``.  This is source integrity, not an approach-performance gate.
MAX_SOURCE_AGE_S = 15.0
# OpenSky timestamps are sub-second while historical state rows are second-granularity.
# Permit only the quantisation-sized future offset, never an arbitrary clock reversal.
MAX_SOURCE_FUTURE_S = 1.0
# A larger gap between actual position updates is a receiver-coverage discontinuity.
# It must not be bridged by a fit or emitted as continuous model input.
MAX_POSITION_COVERAGE_GAP_S = 15.0

SOURCE_INTEGRITY_SCHEMA = "opensky-source-timing-v1"


@dataclass(frozen=True)
class SourceIntegrity:
    """Audit of the source-state cleanup that produced one track.

    Horizontal samples use ``lastposupdate`` as their physical time.  OpenSky exposes
    no corresponding geometric-altitude timestamp, so when altitude changes while the
    position timestamp is held we retain the state snapshot closest to the position
    update and count the asynchronous group explicitly.
    """

    schema_version: str
    input_rows: int
    metadata_missing_rows: int
    stale_last_contact_rows: int
    stale_position_rows: int
    future_timestamp_rows: int
    inconsistent_position_groups: int
    geoaltitude_async_groups: int
    held_rows_removed: int
    coverage_gap_count: int
    retained_rows: int
    position_time_basis: str = "lastposupdate"
    geoaltitude_alignment: str = "nearest_state_snapshot_to_lastposupdate"

    def to_dict(self) -> dict[str, int | str]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


def source_integrity_from_dict(value: Any) -> SourceIntegrity:
    """Strictly restore the regenerable source-timing audit contract."""
    if not isinstance(value, dict):
        raise ValueError("source_integrity must be an object")
    names = {item.name for item in fields(SourceIntegrity)}
    if set(value) != names:
        raise ValueError("source_integrity fields do not match the current schema")
    try:
        integrity = SourceIntegrity(**value)
    except TypeError as error:
        raise ValueError("invalid source_integrity fields") from error
    if integrity.schema_version != SOURCE_INTEGRITY_SCHEMA:
        raise ValueError(
            f"source_integrity schema must be {SOURCE_INTEGRITY_SCHEMA}"
        )
    for item in fields(integrity):
        field_value = getattr(integrity, item.name)
        if item.name.endswith(("_rows", "_groups", "_count")):
            if (
                isinstance(field_value, bool)
                or not isinstance(field_value, int)
                or field_value < 0
            ):
                raise ValueError(
                    f"source_integrity.{item.name} must be a non-negative integer"
                )
    if integrity.retained_rows > integrity.input_rows:
        raise ValueError("source_integrity retained_rows exceeds input_rows")
    if integrity.position_time_basis != "lastposupdate" or (
        integrity.geoaltitude_alignment
        != "nearest_state_snapshot_to_lastposupdate"
    ):
        raise ValueError("source_integrity timing policy is unsupported")
    return integrity


@dataclass(frozen=True)
class Sample:
    """One state vector, including source timing used for integrity checks.

    Before source cleanup, ``time_s`` is the state-vector row time.  A reconstructed
    track rewrites it to ``last_position_update_s`` and collapses held snapshots, so
    downstream fits and model inputs use the physical position-update time.
    """

    time_s: float
    lat: float
    lon: float
    alt_hae_m: float
    on_ground: bool
    reported_ground_speed_m_s: float | None = None
    last_position_update_s: float | None = None
    last_contact_s: float | None = None


@dataclass(frozen=True)
class Track:
    """One contiguous flight of one aircraft, time-ordered.

    Contiguity is a guarantee, not a hope: ``reconstruct_tracks`` is the only
    constructor and it splits on every discontinuity it can detect. Downstream code
    (notably ``final_approach``, which reads the sign of along-track progression as
    direction of travel) relies on that.
    """

    icao24: str
    callsign: str | None
    samples: tuple[Sample, ...]
    source_integrity: SourceIntegrity | None = None

    @property
    def start_s(self) -> float:
        return self.samples[0].time_s

    @property
    def end_s(self) -> float:
        return self.samples[-1].time_s

    @property
    def max_gap_s(self) -> float:
        """Largest hole between consecutive samples -- 0 for a clean 1 Hz track."""
        return max(
            (b.time_s - a.time_s for a, b in zip(self.samples, self.samples[1:])),
            default=0.0,
        )


def reconstruct_tracks(
    rows: Sequence[dict[str, Any]],
    *,
    airport_lat: float,
    airport_lon: float,
    altitude_units: AltitudeUnits,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    ground_split_s: float = DEFAULT_GROUND_SPLIT_S,
    crop_radius_km: float = DEFAULT_CROP_RADIUS_KM,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> list[Track]:
    """Turn raw state-vector rows into contiguous tracks around one airport.

    ``rows`` are OpenSky history records; order does not matter (they are sorted here).

    ``altitude_units`` is REQUIRED -- pass ``"m"`` for rows from
    ``fetch_history_dataframe`` (it converts) and ``"ft"`` for rows read back from the
    archived history-row JSONL (it does not). See ``AltitudeUnits``.
    """
    if altitude_units not in _UNIT_SCALE:
        raise ValueError(f"altitude_units must be 'm' or 'ft', got {altitude_units!r}")
    scale = _UNIT_SCALE[altitude_units]

    by_aircraft: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if _to_sample(row, scale) is None:
            continue
        by_aircraft.setdefault(str(row["icao24"]).lower().strip(), []).append(row)

    tracks: list[Track] = []
    for icao24, aircraft_rows in by_aircraft.items():
        aircraft_rows.sort(key=lambda r: _time_s(r))
        for segment in _split(aircraft_rows, max_gap_s=max_gap_s, ground_split_s=ground_split_s):
            samples = [s for s in (_to_sample(r, scale) for r in segment) if s is not None]
            source_timed, integrity = source_timed_final_block(samples)
            cropped = _final_run_within_radius(
                source_timed, airport_lat, airport_lon, crop_radius_km
            )
            if len(cropped) < min_samples:
                continue
            integrity = replace(integrity, retained_rows=len(cropped))
            tracks.append(
                Track(
                    icao24=icao24,
                    callsign=_callsign(segment),
                    samples=tuple(cropped),
                    source_integrity=integrity,
                )
            )
    return tracks


def source_timed_final_block(
    samples: Sequence[Sample],
) -> tuple[list[Sample], SourceIntegrity]:
    """Return the final fresh, contiguous position-update block and its audit.

    The function is linear. It creates no interpolated points and never lets a
    state-row timestamp masquerade as a new aircraft position.
    """
    missing = stale_contact = stale_position = future = 0
    usable: list[Sample] = []
    for sample in samples:
        position_time = sample.last_position_update_s
        contact_time = sample.last_contact_s
        if position_time is None or contact_time is None:
            missing += 1
            continue
        position_age = sample.time_s - position_time
        contact_age = sample.time_s - contact_time
        if position_age < -MAX_SOURCE_FUTURE_S or contact_age < -MAX_SOURCE_FUTURE_S:
            future += 1
            continue
        if contact_age > MAX_SOURCE_AGE_S:
            stale_contact += 1
            continue
        if position_age > MAX_SOURCE_AGE_S:
            stale_position += 1
            continue
        usable.append(sample)

    groups: list[tuple[int, list[Sample]]] = []
    for sample in usable:
        assert sample.last_position_update_s is not None
        key = round(sample.last_position_update_s * 1000.0)
        if groups and groups[-1][0] == key:
            groups[-1][1].append(sample)
        else:
            groups.append((key, [sample]))

    canonical: list[Sample] = []
    inconsistent_position_groups = 0
    geoaltitude_async_groups = 0
    held_rows_removed = 0
    for _key, group in groups:
        first = group[0]
        if any(
            not math.isclose(sample.lat, first.lat, rel_tol=0.0, abs_tol=1e-7)
            or not math.isclose(sample.lon, first.lon, rel_tol=0.0, abs_tol=1e-7)
            for sample in group[1:]
        ):
            inconsistent_position_groups += 1
            held_rows_removed += len(group)
            continue
        if max(sample.alt_hae_m for sample in group) - min(
            sample.alt_hae_m for sample in group
        ) > 0.05:
            geoaltitude_async_groups += 1
        selected = min(
            group,
            key=lambda sample: (
                abs(sample.time_s - float(sample.last_position_update_s)),
                sample.time_s,
            ),
        )
        held_rows_removed += len(group) - 1
        assert selected.last_position_update_s is not None
        canonical.append(
            Sample(
                time_s=selected.last_position_update_s,
                lat=selected.lat,
                lon=selected.lon,
                alt_hae_m=selected.alt_hae_m,
                on_ground=selected.on_ground,
                reported_ground_speed_m_s=selected.reported_ground_speed_m_s,
                last_position_update_s=selected.last_position_update_s,
                last_contact_s=selected.last_contact_s,
            )
        )
    blocks: list[list[Sample]] = []
    for sample in canonical:
        gap = sample.time_s - blocks[-1][-1].time_s if blocks else None
        if not blocks or gap is None or gap <= 0.0 or gap > MAX_POSITION_COVERAGE_GAP_S:
            blocks.append([sample])
        else:
            blocks[-1].append(sample)
    selected_block = blocks[-1] if blocks else []
    integrity = SourceIntegrity(
        schema_version=SOURCE_INTEGRITY_SCHEMA,
        input_rows=len(samples),
        metadata_missing_rows=missing,
        stale_last_contact_rows=stale_contact,
        stale_position_rows=stale_position,
        future_timestamp_rows=future,
        inconsistent_position_groups=inconsistent_position_groups,
        geoaltitude_async_groups=geoaltitude_async_groups,
        held_rows_removed=held_rows_removed,
        coverage_gap_count=max(0, len(blocks) - 1),
        retained_rows=len(selected_block),
    )
    return selected_block, integrity


def _time_s(row: dict[str, Any]) -> float:
    value = row.get("timestamp", row.get("time"))
    if isinstance(value, (int, float)):
        return float(value)
    import pandas as pd  # local: only the raw-row path needs pandas timestamp parsing

    ts = pd.Timestamp(value)
    return float((ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")).timestamp())


def _to_sample(row: dict[str, Any], scale: float) -> Sample | None:
    lat, lon = row.get("latitude", row.get("lat")), row.get("longitude", row.get("lon"))
    alt = row.get("geoaltitude")
    if lat is None or lon is None or alt is None:
        return None
    try:
        time_s = _time_s(row)
        latitude = float(lat)
        longitude = float(lon)
        altitude = float(alt) * scale
        if not all(
            math.isfinite(value)
            for value in (time_s, latitude, longitude, altitude)
        ):
            return None
        return Sample(
            time_s=time_s,
            lat=latitude,
            lon=longitude,
            alt_hae_m=altitude,
            on_ground=bool(row.get("onground") or row.get("on_ground") or False),
            reported_ground_speed_m_s=_optional_float(row.get("velocity")),
            last_position_update_s=_optional_time_s(row.get("lastposupdate")),
            last_contact_s=_optional_time_s(row.get("lastcontact")),
        )
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_time_s(value: Any) -> float | None:
    number = _optional_float(value)
    if number is not None:
        return number
    if value is None:
        return None
    try:
        import pandas as pd

        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        result = float(timestamp.timestamp())
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _metadata(row: dict[str, Any]) -> tuple[str | None, str | None]:
    def clean(value: Any) -> str | None:
        text = str(value).strip() if value is not None else ""
        return text or None

    return clean(row.get("estdepartureairport")), clean(row.get("estarrivalairport"))


def _split(
    rows: Sequence[dict[str, Any]], *, max_gap_s: float, ground_split_s: float
) -> Iterator[list[dict[str, Any]]]:
    """Cut one aircraft's rows wherever the flight demonstrably ended."""
    start = 0
    ground_since: float | None = None
    ground_cut_done = False
    for i in range(1, len(rows)):
        previous, current = rows[i - 1], rows[i]
        cut = _time_s(current) - _time_s(previous) > max_gap_s

        previous_ground = bool(previous.get("onground"))
        current_ground = bool(current.get("onground"))

        if previous_ground:
            # A sustained rollout/taxi ends the ARRIVAL. Cut once, ground_split_s into
            # the run, so the landing itself stays attached to the approach that flew it.
            ground_since = _time_s(previous) if ground_since is None else ground_since
            if not ground_cut_done and _time_s(previous) - ground_since >= ground_split_s:
                cut = True
                ground_cut_done = True
        else:
            ground_since = None
            ground_cut_done = False

        # Leaving the ground starts a DEPARTURE. Without this the taxi run stays glued
        # to whatever takes off next -- which is how one track came to span a 6598 s hole.
        if previous_ground and not current_ground:
            cut = True

        # Both fields must be known on both sides before a metadata change counts --
        # otherwise merely filling in the origin estimate cuts a live approach in half.
        before, after = _metadata(previous), _metadata(current)
        if all(before) and all(after) and before != after:
            cut = True

        if cut:
            yield list(rows[start:i])
            start = i
    yield list(rows[start:])


def _final_run_within_radius(
    samples: Sequence[Sample], airport_lat: float, airport_lon: float, radius_km: float
) -> list[Sample]:
    """The LAST contiguous stretch inside ``radius_km`` of the field.

    A plain radius filter keeps every in-range sample regardless of what was removed
    between them, which is how one flight's approach ended up glued to a later pass of
    the same aircraft. Taking only the final run makes that unrepresentable: whatever
    survives was contiguous in the source.
    """
    inside = [
        i
        for i, s in enumerate(samples)
        if haversine_km(s.lat, s.lon, airport_lat, airport_lon) <= radius_km
    ]
    if not inside:
        return []
    start = inside[-1]
    for a, b in zip(reversed(inside[:-1]), reversed(inside[1:])):
        if b - a != 1:
            break
        start = a
    return list(samples[start : inside[-1] + 1])


def _callsign(rows: Sequence[dict[str, Any]]) -> str | None:
    for row in rows:
        value = row.get("callsign")
        text = str(value).strip() if value is not None else ""
        if text:
            return text
    return None
