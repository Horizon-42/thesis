#!/usr/bin/env python3
"""
Standalone OpenSky downloader for airport trajectories.

This module lives OUTSIDE aeroviz-4d on purpose.
It downloads real OpenSky data and converts it to the JSON format expected by
`aeroviz-4d/python/generate_czml.py`.

Main capabilities
-----------------
1) Live mode (no credentials required):
    - Queries `/api/flights/all` for recent flights
    - Filters flights related to target airport
    - Requests `/api/tracks/all` for each candidate
2) Historical mode (OAuth client credentials required):
   - Queries `/api/flights/arrival` and `/api/flights/departure`
   - Requests `/api/tracks/all` for each flight in the interval
3) Writes:
   - raw OpenSky payloads (debug)
   - normalized CZML input JSON for aeroviz-4d
4) Optional: run `generate_czml.py` directly.

Usage examples
--------------
# Live data around CYYC (anonymous)
python fetch_cylw_opensky.py --mode live --to-czml

# Historical data (requires OAuth client_id/client_secret)
python fetch_cylw_opensky.py \
  --mode historical \
  --begin "2026-04-05T00:00:00Z" \
  --end   "2026-04-06T00:00:00Z" \
  --client-id "$OPENSKY_CLIENT_ID" \
  --client-secret "$OPENSKY_CLIENT_SECRET" \
  --to-czml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


API_ROOT = "https://opensky-network.org/api"
TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/"
    "protocol/openid-connect/token"
)

# Built-in airport hints: (lat, lon, elevation_m)
AIRPORT_HINTS: dict[str, tuple[float, float, float]] = {
    "CYYC": (51.118822, -114.009933, 3557.0 * 0.3048),
    "CYLW": (49.9561, -119.3778, 1421.0 * 0.3048),
}


@dataclass
class OpenSkyClient:
    client_id: str | None = None
    client_secret: str | None = None
    timeout_sec: int = 30

    _token: str | None = None
    _token_expiry: float = 0.0

    @property
    def has_oauth(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _refresh_token(self) -> None:
        if not self.has_oauth:
            raise RuntimeError("OAuth credentials missing")

        payload = urlencode(
            {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }
        ).encode("utf-8")

        req = Request(TOKEN_URL, data=payload, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urlopen(req, timeout=self.timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        self._token = data["access_token"]
        expires_in = int(data.get("expires_in", 1800))
        # Refresh slightly early.
        self._token_expiry = time.time() + max(0, expires_in - 30)

    def _auth_headers(self) -> dict[str, str]:
        if not self.has_oauth:
            return {}
        if not self._token or time.time() >= self._token_expiry:
            self._refresh_token()
        return {"Authorization": f"Bearer {self._token}"}

    def get(self, endpoint: str, params: dict[str, Any] | None = None, authenticated: bool = False) -> Any:
        query = urlencode({k: v for k, v in (params or {}).items() if v is not None}, doseq=True)
        url = f"{API_ROOT}{endpoint}" + (f"?{query}" if query else "")
        req = Request(url, method="GET")
        req.add_header("Accept", "application/json")

        if authenticated:
            for k, v in self._auth_headers().items():
                req.add_header(k, v)

        try:
            with urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} for {url}: {body[:300]}") from e
        except URLError as e:
            raise RuntimeError(f"Network error for {url}: {e}") from e


def parse_time_to_unix(value: str) -> int:
    # Accept unix seconds or ISO timestamp.
    if value.isdigit():
        return int(value)
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def sanitize_callsign(value: str | None, fallback: str) -> str:
    text = (value or "").strip()
    return text if text else fallback.upper()


def default_outputs_root(script_path: Path) -> Path:
    return script_path.parent / "outputs"


def default_aeroviz_root(script_path: Path) -> Path:
    # sibling: thesis/opensky_cylw + thesis/aeroviz-4d
    return script_path.parent.parent / "aeroviz-4d"


def resolve_airport_profile(airport: str, aeroviz_root: Path) -> tuple[float, float, float]:
    airport = airport.upper()

    csv_path = aeroviz_root / "public" / "data" / "airports.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ident = (row.get("ident") or "").upper()
                gps_code = (row.get("gps_code") or "").upper()
                icao_code = (row.get("icao_code") or "").upper()
                if airport in {ident, gps_code, icao_code}:
                    lat = row.get("latitude_deg")
                    lon = row.get("longitude_deg")
                    if lat and lon:
                        elev_ft = row.get("elevation_ft")
                        elev_m = (float(elev_ft) * 0.3048) if elev_ft not in (None, "") else 0.0
                        return float(lat), float(lon), elev_m

    if airport in AIRPORT_HINTS:
        return AIRPORT_HINTS[airport]

    raise RuntimeError(
        f"Cannot resolve center for airport {airport}. "
        f"Provide airports.csv under {aeroviz_root / 'public' / 'data'} or add a hint."
    )


def fetch_live_tracks(
    client: OpenSkyClient,
    *,
    bbox: tuple[float, float, float, float],
    max_tracks: int,
) -> list[dict[str, Any]]:
    lamin, lamax, lomin, lomax = bbox
    states_payload = client.get(
        "/states/all",
        params={"lamin": lamin, "lamax": lamax, "lomin": lomin, "lomax": lomax},
        authenticated=False,
    )

    states = states_payload.get("states") or []
    seen: set[str] = set()
    tracks: list[dict[str, Any]] = []

    for row in states:
        if not row or len(row) < 2:
            continue
        icao24 = (row[0] or "").lower().strip()
        if not icao24 or icao24 in seen:
            continue
        seen.add(icao24)

        try:
            track = client.get(
                "/tracks/all",
                params={"icao24": icao24, "time": 0},
                authenticated=False,
            )
            if track and track.get("path"):
                tracks.append(track)
        except RuntimeError:
            # Skip individual failures; keep the rest.
            continue

        if len(tracks) >= max_tracks:
            break

    return tracks


def fetch_recent_airport_tracks_anonymous(
    client: OpenSkyClient,
    *,
    airport: str,
    window_hours: int,
    max_tracks: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    now = int(time.time())
    begin = now - max(1, window_hours) * 3600

    flights_all = client.get(
        "/flights/all",
        params={"begin": begin, "end": now},
        authenticated=False,
    ) or []

    airport = airport.upper()
    arrivals: list[dict[str, Any]] = []
    for flight in flights_all:
        if (flight.get("estArrivalAirport") or "").upper() == airport:
            arrivals.append(flight)

    arrivals.sort(key=lambda f: -(int(f.get("lastSeen") or 0)))
    # Landing visualization: use arrivals only.
    related: list[dict[str, Any]] = arrivals

    tracks: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for flight in related:
        icao24 = (flight.get("icao24") or "").lower().strip()
        if not icao24:
            continue
        t_ref = int(flight.get("lastSeen") or flight.get("firstSeen") or 0)
        if t_ref <= 0:
            continue
        key = (icao24, t_ref)
        if key in seen:
            continue
        seen.add(key)

        try:
            track = client.get(
                "/tracks/all",
                params={"icao24": icao24, "time": t_ref},
                authenticated=False,
            )
            if track and track.get("path"):
                tracks.append(track)
        except RuntimeError:
            continue

        if len(tracks) >= max_tracks:
            break

    return flights_all, related, tracks


def fetch_historical_tracks(
    client: OpenSkyClient,
    *,
    airport: str,
    begin: int,
    end: int,
    max_tracks: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    arrivals = client.get(
        "/flights/arrival",
        params={"airport": airport.upper(), "begin": begin, "end": end},
        authenticated=True,
    ) or []
    departures: list[dict[str, Any]] = []
    candidates = arrivals
    tracks: list[dict[str, Any]] = []

    # Retrieve track around flight timestamp.
    for flight in candidates:
        icao24 = (flight.get("icao24") or "").lower()
        if not icao24:
            continue
        t_ref = int(flight.get("lastSeen") or flight.get("firstSeen") or 0)
        if t_ref <= 0:
            continue

        try:
            track = client.get(
                "/tracks/all",
                params={"icao24": icao24, "time": t_ref},
                authenticated=True,
            )
            if track and track.get("path"):
                tracks.append(track)
        except RuntimeError:
            continue

        if len(tracks) >= max_tracks:
            break

    return arrivals, departures, tracks


def track_to_czml_flight(
    track: dict[str, Any],
    *,
    airport_lat: float,
    airport_lon: float,
    airport_elev_m: float,
    match_radius_km: float,
    require_landing: bool,
    landing_radius_km: float,
    max_end_distance_km: float,
    altitude_mode: str,
    min_ground_samples: int,
    max_altitude_bias_m: float,
    approach_alt_buffer_m: float,
    approach_window_min: int,
    include_ground: bool,
) -> dict[str, Any] | None:
    path = track.get("path") or []
    if not path:
        return None

    parsed: list[tuple[int, float, float, float, bool, float]] = []
    for wp in path:
        if not wp or len(wp) < 6:
            continue
        t, lat, lon, alt_m, _trk, on_ground = wp
        if t is None or lat is None or lon is None or alt_m is None:
            continue
        alt = float(alt_m)
        if math.isnan(alt):
            continue
        dist_km = haversine_km(float(lat), float(lon), airport_lat, airport_lon)

        parsed.append((int(t), float(lon), float(lat), alt, bool(on_ground), dist_km))

    if len(parsed) < 2:
        return None

    parsed.sort(key=lambda x: x[0])

    bias_m = 0.0
    bias_applied = False
    ground_samples = 0
    if altitude_mode == "touchdown-bias":
        touchdown_alts = [
            alt
            for _t, _lon, _lat, alt, gnd, d in parsed
            if gnd and d <= landing_radius_km
        ]
        ground_samples = len(touchdown_alts)
        if ground_samples >= min_ground_samples:
            touchdown_alts_sorted = sorted(touchdown_alts)
            median_alt = touchdown_alts_sorted[ground_samples // 2]
            candidate_bias = airport_elev_m - median_alt
            if abs(candidate_bias) <= max_altitude_bias_m:
                bias_m = candidate_bias
                bias_applied = True

    if bias_applied:
        parsed = [
            (t, lon, lat, alt + bias_m, gnd, d)
            for t, lon, lat, alt, gnd, d in parsed
        ]

    # Airport relevance: at least one point near target airport.
    min_dist = min(
        d
        for _t, _lon, _lat, _alt, _gnd, d in parsed
    )
    if min_dist > match_radius_km:
        return None

    closest_idx = min(range(len(parsed)), key=lambda i: parsed[i][5])
    closest_dist_km = parsed[closest_idx][5]
    if closest_dist_km > max_end_distance_km:
        return None

    landing_reference_t: int | None = None

    if require_landing:
        touchdown_times = [
            t
            for t, _lon, _lat, _alt, gnd, d in parsed
            if gnd and d <= landing_radius_km
        ]

        landing_ok = False

        if touchdown_times:
            first_touchdown_t = min(touchdown_times)
            airborne_before = any((not gnd) and t < first_touchdown_t for t, _lon, _lat, _alt, gnd, _d in parsed)
            landing_ok = airborne_before
            if landing_ok:
                landing_reference_t = first_touchdown_t

        if not landing_ok:
            low_alt_threshold = airport_elev_m + approach_alt_buffer_m
            near_low_points = [
                (t, alt)
                for t, _lon, _lat, alt, _gnd, d in parsed
                if d <= landing_radius_km and alt <= low_alt_threshold
            ]
            if near_low_points:
                first_near_low_t = min(t for t, _alt in near_low_points)
                had_higher_before = any(
                    t < first_near_low_t and alt >= low_alt_threshold + 250.0
                    for t, _lon, _lat, alt, _gnd, _d in parsed
                )
                landing_ok = had_higher_before
                if landing_ok:
                    landing_reference_t = parsed[closest_idx][0]

        if not landing_ok:
            return None

    if landing_reference_t is None:
        # Fallback: use closest point to airport as approach anchor.
        landing_reference_t = parsed[closest_idx][0]

    # Keep only the final approach window before landing/closest approach.
    windowed = parsed
    if approach_window_min > 0:
        window_sec = approach_window_min * 60
        window_start_t = landing_reference_t - window_sec
        windowed = [row for row in parsed if window_start_t <= row[0] <= landing_reference_t]

        # If sampling is sparse, keep a bounded local segment near closest approach.
        if len(windowed) < 2:
            local_start = max(0, closest_idx - 40)
            local_rows = parsed[local_start : closest_idx + 1]
            windowed = [
                row for row in local_rows
                if (landing_reference_t - window_sec) <= row[0] <= landing_reference_t
            ]

        if len(windowed) < 2:
            return None

    # Convert to CZML waypoints and optionally remove ground points.
    waypoints_abs: list[tuple[int, float, float, float]] = []
    for t, lon, lat, alt, gnd, _dist_km in windowed:
        if not include_ground and gnd:
            continue
        waypoints_abs.append((t, lon, lat, alt))

    if len(waypoints_abs) < 2:
        return None

    waypoints_abs.sort(key=lambda x: x[0])
    t0 = waypoints_abs[0][0]
    rel = [[t - t0, lon, lat, alt] for t, lon, lat, alt in waypoints_abs]

    icao24 = (track.get("icao24") or "unknown").lower()
    callsign = sanitize_callsign(track.get("callsign"), fallback=icao24)
    flight_id = callsign.replace(" ", "")[:16] or icao24

    return {
        "id": flight_id,
        "callsign": callsign,
        "type": "UNK",
        "altitude_source": "opensky_tracks_all_baro_altitude_m",
        "altitude_correction_mode": altitude_mode,
        "altitude_bias_m": round(bias_m, 3),
        "altitude_bias_applied": bias_applied,
        "altitude_ground_samples": ground_samples,
        "waypoints": rel,
    }


def convert_tracks_to_czml_input(
    tracks: list[dict[str, Any]],
    *,
    airport_lat: float,
    airport_lon: float,
    airport_elev_m: float,
    match_radius_km: float,
    require_landing: bool,
    landing_radius_km: float,
    max_end_distance_km: float,
    altitude_mode: str,
    min_ground_samples: int,
    max_altitude_bias_m: float,
    approach_alt_buffer_m: float,
    approach_window_min: int,
    include_ground: bool,
    limit_flights: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    for track in tracks:
        item = track_to_czml_flight(
            track,
            airport_lat=airport_lat,
            airport_lon=airport_lon,
            airport_elev_m=airport_elev_m,
            match_radius_km=match_radius_km,
            require_landing=require_landing,
            landing_radius_km=landing_radius_km,
            max_end_distance_km=max_end_distance_km,
            altitude_mode=altitude_mode,
            min_ground_samples=min_ground_samples,
            max_altitude_bias_m=max_altitude_bias_m,
            approach_alt_buffer_m=approach_alt_buffer_m,
            approach_window_min=approach_window_min,
            include_ground=include_ground,
        )
        if not item:
            continue

        # Avoid duplicate IDs.
        base = item["id"]
        if base in used_ids:
            suffix = 2
            while f"{base}_{suffix}" in used_ids:
                suffix += 1
            item["id"] = f"{base}_{suffix}"
        used_ids.add(item["id"])

        out.append(item)
        if len(out) >= limit_flights:
            break

    return out


def run_generate_czml(
    *,
    aeroviz_root: Path,
    czml_input_path: Path,
    czml_output_path: Path,
) -> None:
    generator = aeroviz_root / "python" / "generate_czml.py"
    if not generator.exists():
        raise RuntimeError(f"Cannot find generator script: {generator}")

    cmd = [
        sys.executable,
        str(generator),
        "--input",
        str(czml_input_path),
        "--output",
        str(czml_output_path),
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download recent airport trajectories from OpenSky")

    parser.add_argument("--mode", choices=["auto", "live", "historical"], default="auto")
    parser.add_argument("--airport", default="CYYC")

    parser.add_argument("--begin", default=None, help="Historical begin (Unix or ISO, e.g. 2026-04-05T00:00:00Z)")
    parser.add_argument("--end", default=None, help="Historical end (Unix or ISO)")

    parser.add_argument("--client-id", default=os.getenv("OPENSKY_CLIENT_ID"))
    parser.add_argument("--client-secret", default=os.getenv("OPENSKY_CLIENT_SECRET"))

    parser.add_argument("--bbox-lat-pad", type=float, default=0.30, help="Live mode latitude half-span around selected airport")
    parser.add_argument("--bbox-lon-pad", type=float, default=0.45, help="Live mode longitude half-span around selected airport")

    parser.add_argument("--match-radius-km", type=float, default=35.0, help="Track is accepted only if it comes within this distance of airport")
    parser.add_argument("--landing-radius-km", type=float, default=15.0, help="Touchdown (on_ground) must appear within this distance when landing is required")
    parser.add_argument("--max-end-distance-km", type=float, default=2.5, help="Final approach anchor point must be within this distance of airport")
    parser.add_argument(
        "--altitude-mode",
        choices=["raw", "touchdown-bias"],
        default="raw",
        help="Altitude handling: raw keeps OpenSky barometric altitude; touchdown-bias applies a constant offset estimated from near-runway on-ground points",
    )
    parser.add_argument("--min-ground-samples", type=int, default=2, help="Minimum near-runway on-ground points needed for touchdown-bias estimation")
    parser.add_argument("--max-altitude-bias-m", type=float, default=400.0, help="Reject touchdown-bias estimates whose absolute value exceeds this threshold")
    parser.add_argument("--approach-alt-buffer-m", type=float, default=450.0, help="If on_ground is missing, accept near-airport low-altitude segment below airport_elevation + this buffer")
    parser.add_argument("--approach-window-min", type=int, default=20, help="Keep only this many minutes before landing/closest-approach")
    parser.add_argument("--radius-km", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--allow-partial", action="store_true", help="Allow tracks that do not contain landing/touchdown near airport")
    parser.add_argument("--live-window-hours", type=int, default=12, help="Look back this many hours for /flights/all in live mode")
    parser.add_argument("--max-tracks", type=int, default=80)
    parser.add_argument("--max-flights", type=int, default=16)
    parser.add_argument("--min-flights", type=int, default=3, help="If strict landing filter yields fewer than this count, fill with partial approach tracks")
    parser.add_argument("--exclude-ground", action="store_true", help="Exclude on-ground points from exported trajectory")

    parser.add_argument("--output-root", default=None, help="Folder for outputs (default: ./outputs next to script)")
    parser.add_argument("--to-czml", action="store_true", help="Run aeroviz-4d/python/generate_czml.py after download")
    parser.add_argument("--aeroviz-root", default=None, help="Path to aeroviz-4d folder")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()

    aeroviz_root = Path(args.aeroviz_root) if args.aeroviz_root else default_aeroviz_root(script_path)
    airport_lat, airport_lon, airport_elev_m = resolve_airport_profile(args.airport, aeroviz_root)

    output_root = Path(args.output_root) if args.output_root else default_outputs_root(script_path)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    airport_tag = args.airport.lower()
    raw_path = output_root / f"{airport_tag}_raw_{timestamp}.json"
    czml_input_path = output_root / f"{airport_tag}_czml_input_{timestamp}.json"

    client = OpenSkyClient(
        client_id=args.client_id,
        client_secret=args.client_secret,
    )

    mode = args.mode
    if mode == "auto":
        mode = "historical" if client.has_oauth else "live"

    payload: dict[str, Any] = {
        "mode": mode,
        "airport": args.airport.upper(),
        "airport_center": {"lat": airport_lat, "lon": airport_lon},
        "airport_elevation_m": airport_elev_m,
        "altitude_mode": args.altitude_mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if mode == "historical":
        if not client.has_oauth:
            raise RuntimeError(
                "Historical mode requires OAuth credentials. Set OPENSKY_CLIENT_ID/OPENSKY_CLIENT_SECRET "
                "or pass --client-id/--client-secret."
            )

        if args.begin and args.end:
            begin = parse_time_to_unix(args.begin)
            end = parse_time_to_unix(args.end)
        else:
            # Default to previous UTC day [00:00, 24:00) for stable historical availability.
            now = datetime.now(timezone.utc)
            day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(days=1)
            begin = int(day_start.timestamp())
            end = int((day_start + timedelta(days=1)).timestamp())

        arrivals, departures, tracks = fetch_historical_tracks(
            client,
            airport=args.airport,
            begin=begin,
            end=end,
            max_tracks=args.max_tracks,
        )
        payload.update(
            {
                "begin": begin,
                "end": end,
                "arrivals_count": len(arrivals),
                "departures_count": len(departures),
                "arrivals": arrivals,
                "departures": departures,
                "tracks_count": len(tracks),
                "tracks": tracks,
            }
        )
    else:
        flights_all, related_flights, tracks = fetch_recent_airport_tracks_anonymous(
            client,
            airport=args.airport,
            window_hours=args.live_window_hours,
            max_tracks=args.max_tracks,
        )

        bbox = (
            airport_lat - args.bbox_lat_pad,
            airport_lat + args.bbox_lat_pad,
            airport_lon - args.bbox_lon_pad,
            airport_lon + args.bbox_lon_pad,
        )

        # Fallback for sparse flights/all results.
        if not tracks:
            tracks = fetch_live_tracks(
                client,
                bbox=bbox,
                max_tracks=args.max_tracks,
            )
        elif len(tracks) < args.max_tracks:
            extra_tracks = fetch_live_tracks(
                client,
                bbox=bbox,
                max_tracks=args.max_tracks - len(tracks),
            )
            seen_keys = {
                ((t.get("icao24") or "").lower().strip(), int(t.get("startTime") or 0))
                for t in tracks
            }
            for track in extra_tracks:
                key = ((track.get("icao24") or "").lower().strip(), int(track.get("startTime") or 0))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                tracks.append(track)
                if len(tracks) >= args.max_tracks:
                    break

        payload.update(
            {
                "flights_all_count": len(flights_all),
                "related_flights_count": len(related_flights),
                "bbox": {
                    "lamin": bbox[0],
                    "lamax": bbox[1],
                    "lomin": bbox[2],
                    "lomax": bbox[3],
                },
                "tracks_count": len(tracks),
                "tracks": tracks,
            }
        )

    match_radius_km = args.radius_km if args.radius_km is not None else args.match_radius_km

    flights = convert_tracks_to_czml_input(
        payload.get("tracks", []),
        airport_lat=airport_lat,
        airport_lon=airport_lon,
        airport_elev_m=airport_elev_m,
        match_radius_km=match_radius_km,
        require_landing=not args.allow_partial,
        landing_radius_km=args.landing_radius_km,
        max_end_distance_km=args.max_end_distance_km,
        altitude_mode=args.altitude_mode,
        min_ground_samples=args.min_ground_samples,
        max_altitude_bias_m=args.max_altitude_bias_m,
        approach_alt_buffer_m=args.approach_alt_buffer_m,
        approach_window_min=args.approach_window_min,
        include_ground=not args.exclude_ground,
        limit_flights=args.max_flights,
    )

    if (not args.allow_partial) and len(flights) < max(1, args.min_flights):
        relaxed = convert_tracks_to_czml_input(
            payload.get("tracks", []),
            airport_lat=airport_lat,
            airport_lon=airport_lon,
            airport_elev_m=airport_elev_m,
            match_radius_km=match_radius_km,
            require_landing=False,
            landing_radius_km=args.landing_radius_km,
            max_end_distance_km=args.max_end_distance_km,
            altitude_mode=args.altitude_mode,
            min_ground_samples=args.min_ground_samples,
            max_altitude_bias_m=args.max_altitude_bias_m,
            approach_alt_buffer_m=args.approach_alt_buffer_m,
            approach_window_min=args.approach_window_min,
            include_ground=not args.exclude_ground,
            limit_flights=args.max_flights,
        )

        used_ids = {item["id"] for item in flights}
        for item in relaxed:
            if item["id"] in used_ids:
                continue
            flights.append(item)
            used_ids.add(item["id"])
            if len(flights) >= args.max_flights:
                break

    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    czml_input_path.write_text(json.dumps(flights, indent=2), encoding="utf-8")

    print(f"[OpenSky] mode={mode}")
    print(f"[OpenSky] tracks downloaded: {len(payload.get('tracks', []))}")
    print(f"[OpenSky] flights exported for CZML: {len(flights)}")
    print(f"[OpenSky] raw output: {raw_path}")
    print(f"[OpenSky] czml input: {czml_input_path}")

    if args.to_czml:
        czml_output_path = aeroviz_root / "public" / "data" / "trajectories.czml"
        run_generate_czml(
            aeroviz_root=aeroviz_root,
            czml_input_path=czml_input_path,
            czml_output_path=czml_output_path,
        )
        print(f"[OpenSky] generated CZML: {czml_output_path}")


if __name__ == "__main__":
    main()
