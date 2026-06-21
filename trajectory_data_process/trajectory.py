"""Canonical trajectory model for the package.

A :class:`Trajectory` is the single intermediate structure every stage works on.
It is built once from OpenSky history-DB rows and then consumed by both the CZML
export and the training-dataset builders, so trajectory points are parsed in
exactly one place.

OpenSky history rows carry barometric *and* geometric altitude in the same row,
so each :class:`TrajectoryPoint` keeps both. Geometric altitude is the value used
for visualization; barometric altitude is retained for reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timezone
from typing import Iterator

import pandas as pd


# Canonical column names after normalization. Anything the history query returns
# with a table prefix (e.g. ``FlightsData4.estarrivalairport``) is reduced to the
# bare name below.
CORE_COLUMNS = ("time", "icao24", "lat", "lon")

# ``traffic`` renames the raw OpenSky columns to its own vocabulary before handing
# back a dataframe. Map both the raw names and traffic's names to one internal set.
COLUMN_ALIASES = {
    "timestamp": "time",
    "latitude": "lat",
    "longitude": "lon",
    "altitude": "baroaltitude",
    "track": "heading",
    "groundspeed": "velocity",
    "vertical_rate": "vertrate",
}


@dataclass(frozen=True)
class TrajectoryPoint:
    """One sampled state along a trajectory (time-ordered)."""

    time: int  # unix seconds, UTC
    lat: float
    lon: float
    geo_altitude_m: float | None
    baro_altitude_m: float | None
    heading_deg: float | None
    on_ground: bool


@dataclass
class Trajectory:
    """A single aircraft track segment with stable flight metadata."""

    icao24: str
    callsign: str | None
    dep_airport: str | None
    arr_airport: str | None
    points: list[TrajectoryPoint]

    @property
    def start_time(self) -> int:
        return self.points[0].time

    @property
    def end_time(self) -> int:
        return self.points[-1].time

    @property
    def geo_altitude_coverage(self) -> float:
        """Fraction of points that carry geometric altitude (0.0–1.0)."""
        if not self.points:
            return 0.0
        with_geo = sum(1 for p in self.points if p.geo_altitude_m is not None)
        return with_geo / len(self.points)


def canonical_column(column: object) -> str:
    """Map a traffic/pyopensky column label to its internal canonical name."""
    text = str(column).strip().strip('"').lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return COLUMN_ALIASES.get(text, text)


def normalize_history_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, de-duplicated view with canonical column names."""
    out = df.rename(columns={c: canonical_column(c) for c in df.columns})
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out["icao24"] = out["icao24"].astype(str).str.lower().str.strip()
    out = out.sort_values(["icao24", "time"], kind="stable")
    dedupe = [c for c in ("icao24", "time", "lat", "lon") if c in out.columns]
    return out.drop_duplicates(subset=dedupe, keep="first").reset_index(drop=True)


def build_trajectories_from_history(
    df: pd.DataFrame,
    *,
    max_gap_sec: int = 900,
    require_geo_altitude: bool = True,
    min_points: int = 2,
) -> list[Trajectory]:
    """Build trajectories from OpenSky history rows.

    Rows are grouped per aircraft and split into separate trajectories on a time
    gap larger than ``max_gap_sec`` or a change of departure/arrival airport.

    With ``require_geo_altitude`` (the default) the dataframe must contain a
    ``geoaltitude`` column and points without a geometric altitude are dropped;
    trajectories left with fewer than ``min_points`` are discarded.
    """
    if df.empty:
        return []

    missing = [c for c in CORE_COLUMNS if c not in {canonical_column(c2) for c2 in df.columns}]
    if missing:
        raise ValueError(f"History rows missing required columns: {missing}")

    normalized = normalize_history_dataframe(df)
    if require_geo_altitude and "geoaltitude" not in normalized.columns:
        raise ValueError(
            "History rows do not contain a 'geoaltitude' column; geometric altitude "
            "is required. Include geoaltitude in the history query or pass "
            "require_geo_altitude=False."
        )

    trajectories: list[Trajectory] = []
    for segment in _iter_segments(normalized, max_gap_sec=max_gap_sec):
        points = _segment_points(segment, require_geo_altitude=require_geo_altitude)
        if len(points) < min_points:
            continue
        meta = _segment_metadata(segment)
        trajectories.append(
            Trajectory(
                icao24=meta["icao24"],
                callsign=meta["callsign"],
                dep_airport=meta["dep_airport"],
                arr_airport=meta["arr_airport"],
                points=points,
            )
        )
    return trajectories


def _iter_segments(df: pd.DataFrame, *, max_gap_sec: int) -> Iterator[pd.DataFrame]:
    """Split per-aircraft rows into track segments on time gaps / metadata change."""
    for _icao24, group in df.groupby("icao24", sort=False):
        ordered = group.sort_values("time", kind="stable").reset_index(drop=True)
        start = 0
        prev_time = ordered.at[0, "time"]
        prev_key = _metadata_key(ordered.iloc[0])
        for idx in range(1, len(ordered)):
            cur_time = ordered.at[idx, "time"]
            cur_key = _metadata_key(ordered.iloc[idx])
            gap_sec = (cur_time - prev_time).total_seconds()
            metadata_changed = _complete(prev_key) and _complete(cur_key) and cur_key != prev_key
            if gap_sec > max_gap_sec or metadata_changed:
                yield ordered.iloc[start:idx].copy()
                start = idx
            prev_time = cur_time
            prev_key = cur_key
        yield ordered.iloc[start:].copy()


def _segment_points(segment: pd.DataFrame, *, require_geo_altitude: bool) -> list[TrajectoryPoint]:
    points: list[TrajectoryPoint] = []
    for row in segment.to_dict(orient="records"):
        lat = _finite(row.get("lat"))
        lon = _finite(row.get("lon"))
        if lat is None or lon is None:
            continue
        geo = _finite(row.get("geoaltitude"))
        if require_geo_altitude and geo is None:
            continue
        points.append(
            TrajectoryPoint(
                time=_unix_seconds(row["time"]),
                lat=lat,
                lon=lon,
                geo_altitude_m=geo,
                baro_altitude_m=_finite(row.get("baroaltitude")),
                heading_deg=_finite(row.get("heading")),
                on_ground=_as_bool(row.get("onground")),
            )
        )
    return points


def _segment_metadata(segment: pd.DataFrame) -> dict[str, str | None]:
    first = segment.iloc[0]
    return {
        "icao24": str(first["icao24"]).lower().strip(),
        "callsign": _first_text(segment.get("callsign")),
        "dep_airport": _first_text(segment.get("estdepartureairport")),
        "arr_airport": _first_text(segment.get("estarrivalairport")),
    }


def _metadata_key(row: pd.Series) -> tuple[str | None, str | None]:
    return _clean(row.get("estdepartureairport")), _clean(row.get("estarrivalairport"))


def _complete(key: tuple[str | None, str | None]) -> bool:
    return bool(key[0] or key[1])


def _unix_seconds(value: object) -> int:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize(timezone.utc) if ts.tzinfo is None else ts.tz_convert(timezone.utc)
    return int(ts.timestamp())


def _finite(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_bool(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _clean(value: object) -> str | None:
    text = _first_text(pd.Series([value]))
    return text.upper() if text else None


def _first_text(series: pd.Series | None) -> str | None:
    if series is None:
        return None
    for value in series:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return None
