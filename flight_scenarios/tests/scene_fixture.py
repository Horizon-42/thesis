"""A tiny synthetic harvest (tracks roster + track files) for the scene data plane's tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

from trajectory_data_process.harvest.store import TRACK_SCHEMA_VERSION, HarvestPaths

AIRPORT = "KRDU"
RUNWAY = "05L"
TARGET = {"lat": 35.8744489, "lon": -78.8019636, "elevation_hae_m": 79.8, "elevation_msl_m": 111.8,
          "course_deg": 45.0, "threshold_crossing_height_m": 17.5, "published_glidepath_deg": 3.0}
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
M_PER_DEG_LAT = 111_319.4908
PSI = math.radians(90.0 - TARGET["course_deg"])        # inbound course, math-ENU


def chart_to_latlon(e_m: float, n_m: float) -> tuple[float, float]:
    lat = TARGET["lat"] + n_m / M_PER_DEG_LAT
    lon = TARGET["lon"] + e_m / (M_PER_DEG_LAT * math.cos(math.radians(TARGET["lat"])))
    return lat, lon


def axes_to_chart(d_m: float, xt_m: float) -> tuple[float, float]:
    return -d_m * math.cos(PSI) + xt_m * math.sin(PSI), -d_m * math.sin(PSI) - xt_m * math.cos(PSI)


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def straight_samples(start_utc: float, end_utc: float, d_start: float, d_end: float, xt: float, alt: float,
                     step_s: float = 5.0) -> list[list[float]]:
    """Samples flying along the runway axis from ``d_start`` to ``d_end`` (upstream metres)."""
    n = int(round((end_utc - start_utc) / step_s)) + 1
    rows = []
    for k in range(n):
        f = k / max(n - 1, 1)
        e, nn = axes_to_chart(d_start + (d_end - d_start) * f, xt)
        lat, lon = chart_to_latlon(e, nn)
        rows.append([round(k * step_s, 3), lon, lat, alt])
    return rows


def write_harvest(root: Path, tracks: list[dict]) -> HarvestPaths:
    """``tracks``: dicts with callsign, icao24, outcome, runway, landing_utc (or None),
    start_utc, samples. Writes the files and the v2 roster."""
    paths = HarvestPaths(root=root, code=AIRPORT)
    rows = []
    for i, t in enumerate(tracks):
        landing = t.get("landing_utc")
        stamp = datetime.fromtimestamp(landing if landing else t["start_utc"], tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        key = f"{t['callsign']}_{t['runway'] or 'not_landing'}_{t['icao24']}_{stamp}"
        relative = f"{t['outcome']}/{key}.json"
        record = {
            "flight_key": key, "icao24": t["icao24"], "callsign": t["callsign"], "outcome": t["outcome"],
            "runway": t["runway"], "landing_time_utc": _iso(landing)[:19] + "Z" if landing else None,
            "landing_sample_index": len(t["samples"]) - 1 if landing else None,
            "start_time_utc": _iso(t["start_utc"]), "duration_s": float(t["samples"][-1][0]),
            "max_sample_gap_s": 5.0, "altitude_source": "opensky_history_geoaltitude_m", "altitude_datum": "hae",
            "assignment": {"outcome": t["outcome"], "runway": t["runway"]}, "observed_threshold_event": None,
            "samples": t["samples"], "reported_ground_speeds_m_s": [None] * len(t["samples"]),
        }
        (paths.tracks / relative).parent.mkdir(parents=True, exist_ok=True)
        (paths.tracks / relative).write_text(json.dumps(record), encoding="utf-8")
        rows.append({"flight_key": key, "file": relative, "outcome": t["outcome"], "runway": t["runway"],
                     "icao24": t["icao24"], "callsign": t["callsign"], "landing_time_utc": record["landing_time_utc"],
                     "landing_sample_index": record["landing_sample_index"], "event_status": "unavailable"})
    paths.manifest.write_text(json.dumps({
        "schema_version": TRACK_SCHEMA_VERSION, "airport": AIRPORT, "source_integrity_complete": True,
        "records": rows,
    }), encoding="utf-8")
    return paths


def standard_scene(root: Path) -> tuple[HarvestPaths, dict[str, str]]:
    """The ego 15 km out on the centreline at T0 plus six other aircraft; returns the
    paths and the flight keys by role."""
    tracks = [
        # the ego: on the centreline, 25 → 14 km over the 120 s before T0 (and on to landing)
        dict(callsign="EGO1", icao24="e00001", outcome="assigned", runway=RUNWAY, landing_utc=T0 + 200.0,
             start_utc=T0 - 130.0, samples=straight_samples(T0 - 130.0, T0 + 200.0, 25_500.0, 0.0, 0.0, 900.0)),
        # A: established on the final 6 km out at T0, inbound at 70 m/s
        dict(callsign="AHEAD", icao24="a00001", outcome="assigned", runway=RUNWAY, landing_utc=T0 + 90.0,
             start_utc=T0 - 100.0, samples=straight_samples(T0 - 100.0, T0 + 90.0, 13_000.0, 0.0, 30.0, 400.0)),
        # B: only after T0 — the future; must not appear
        dict(callsign="LATER", icao24="b00001", outcome="assigned", runway=RUNWAY, landing_utc=T0 + 400.0,
             start_utc=T0 + 10.0, samples=straight_samples(T0 + 10.0, T0 + 400.0, 20_000.0, 0.0, 0.0, 700.0)),
        # C: on the downwind 20 km out, 8 km right of the course, flying OUTBOUND
        dict(callsign="DWIND", icao24="c00001", outcome="assigned", runway=RUNWAY, landing_utc=T0 + 500.0,
             start_utc=T0 - 90.0, samples=straight_samples(T0 - 90.0, T0 + 30.0, 12_000.0, 24_000.0, 8_000.0, 1_200.0)),
        # D: 60 km out — outside the radius
        dict(callsign="FAR", icao24="d00001", outcome="not_landing", runway=None, landing_utc=None,
             start_utc=T0 - 60.0, samples=straight_samples(T0 - 60.0, T0 + 60.0, 62_000.0, 55_000.0, 0.0, 3_000.0)),
        # E: landed on the ego's runway 300 s before T0 — the past
        dict(callsign="LANDED", icao24="e00002", outcome="assigned", runway=RUNWAY, landing_utc=T0 - 300.0,
             start_utc=T0 - 500.0, samples=straight_samples(T0 - 500.0, T0 - 300.0, 14_000.0, 0.0, 0.0, 500.0)),
        # F: landed on the other runway 1000 s before T0
        dict(callsign="OTHER", icao24="f00001", outcome="assigned", runway="23R", landing_utc=T0 - 1_000.0,
             start_utc=T0 - 1_200.0, samples=straight_samples(T0 - 1_200.0, T0 - 1_000.0, 14_000.0, 0.0, 0.0, 500.0)),
    ]
    paths = write_harvest(root, tracks)
    roster = json.loads(paths.manifest.read_text())["records"]
    keys = {row["callsign"]: row["flight_key"] for row in roster}
    return paths, keys
