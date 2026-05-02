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

Usage examples
--------------
# Live data around CYYC (anonymous)
python fetch_cylw_opensky.py --mode live

# Historical data (requires OAuth client_id/client_secret)
python fetch_cylw_opensky.py \
  --mode historical \
  --begin "2026-04-05T00:00:00Z" \
  --end   "2026-04-06T00:00:00Z" \
  --client-id "$OPENSKY_CLIENT_ID" \
    --client-secret "$OPENSKY_CLIENT_SECRET"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from trajectory_normalization import (
    convert_tracks_to_czml_input,
    convert_tracks_to_raw_czml_input,
)


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
    """OpenSky API 客户端，封装匿名 GET、OAuth token 获取和认证请求。"""

    client_id: str | None = None
    client_secret: str | None = None
    timeout_sec: int = 30

    _token: str | None = None
    _token_expiry: float = 0.0
    _token_lock: Lock = field(default_factory=Lock)

    @property
    def has_oauth(self) -> bool:
        """判断当前客户端是否具备历史接口所需的 OAuth 凭据。"""
        return bool(self.client_id and self.client_secret)

    def _refresh_token(self) -> None:
        """使用 client credentials flow 刷新 OpenSky OAuth access token。"""
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
        # 提前刷新，避免请求发送时 token 刚好过期。
        self._token_expiry = time.time() + max(0, expires_in - 30)

    def _auth_headers(self) -> dict[str, str]:
        """返回认证请求头；必要时用锁保护 token 刷新，避免并发重复刷新。"""
        if not self.has_oauth:
            return {}
        if not self._token or time.time() >= self._token_expiry:
            with self._token_lock:
                if not self._token or time.time() >= self._token_expiry:
                    self._refresh_token()
        return {"Authorization": f"Bearer {self._token}"}

    def get(self, endpoint: str, params: dict[str, Any] | None = None, authenticated: bool = False) -> Any:
        """执行 OpenSky GET 请求并解析 JSON 响应。"""
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
            # OpenSky 的 /flights/* 和 /tracks 在无结果时可能返回空 404；
            # 这里把它视为 None，让调用方用 `or []` 继续处理。
            if e.code == 404 and body.strip() in ("", "[]", "{}"):
                return None
            raise RuntimeError(f"HTTP {e.code} for {url}: {body[:300]}") from e
        except URLError as e:
            raise RuntimeError(f"Network error for {url}: {e}") from e


def load_credentials_file(path: Path) -> tuple[str | None, str | None]:
    """从 JSON 凭据文件读取 (client_id, client_secret)，兼容 camelCase 和 snake_case 键名。"""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"Failed to read credentials file {path}: {e}") from e
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Credentials file {path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"Credentials file {path} must contain a JSON object")
    client_id = data.get("clientId") or data.get("client_id")
    client_secret = data.get("clientSecret") or data.get("client_secret")
    return (client_id, client_secret)


def parse_time_to_unix(value: str) -> int:
    """将 Unix 秒字符串或 ISO 时间字符串解析为 Unix 秒。"""
    if value.isdigit():
        return int(value)
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def default_outputs_root(script_path: Path) -> Path:
    """返回脚本旁边的默认输出目录。"""
    return script_path.parent / "outputs"


def default_aeroviz_root(script_path: Path) -> Path:
    """返回与 opensky_data_query 同级的默认 aeroviz-4d 目录。"""
    return script_path.parent.parent / "aeroviz-4d"


def resolve_airport_profile(airport: str, aeroviz_root: Path) -> tuple[float, float, float]:
    """解析机场中心点和标高，优先读取 AeroViz airports.csv，缺失时使用内置提示。"""
    airport = airport.upper()

    csv_path = aeroviz_root / "public" / "data" / "common" / "airports.csv"
    if not csv_path.exists():
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
                        # airports.csv 标高单位为英尺，下游归一化统一使用米。
                        elev_ft = row.get("elevation_ft")
                        elev_m = (float(elev_ft) * 0.3048) if elev_ft not in (None, "") else 0.0
                        return float(lat), float(lon), elev_m

    if airport in AIRPORT_HINTS:
        return AIRPORT_HINTS[airport]

    raise RuntimeError(
        f"Cannot resolve center for airport {airport}. "
        f"Provide airports.csv under {aeroviz_root / 'public' / 'data' / 'common'} or add a hint."
    )


def fetch_live_tracks(
    client: OpenSkyClient,
    *,
    bbox: tuple[float, float, float, float],
    max_tracks: int,
) -> list[dict[str, Any]]:
    """匿名 live 兜底路径：按机场附近 bbox 查询当前状态，再逐个拉取 track。"""
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
        # /states/all 返回数组字段；这里只需要 icao24 去请求完整轨迹。
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
            # 单架飞机失败不终止整个批次，保证匿名 live 模式尽可能返回可用样本。
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
    """匿名 live 主路径：查询近期航班列表，筛选目标机场到达航班并拉取轨迹。"""
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
    # 当前可视化关注落地进近，因此匿名航班列表只选目标机场 arrivals。
    related: list[dict[str, Any]] = arrivals

    tracks: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for flight in related:
        # OpenSky track 查询需要 icao24 和参考时间；使用 lastSeen 更贴近落地段。
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
            # 某个候选航班 track 不可用时跳过，继续处理其他候选。
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
    track_workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """认证 historical 路径：查询指定时间窗的到达航班，并并发拉取每条轨迹。"""
    arrivals = client.get(
        "/flights/arrival",
        params={"airport": airport.upper(), "begin": begin, "end": end},
        authenticated=True,
    ) or []
    departures: list[dict[str, Any]] = []
    candidates = arrivals
    tracks_with_index: list[tuple[int, dict[str, Any]]] = []

    print(
        f"[OpenSky] historical arrivals found: {len(candidates)} "
        f"({datetime.fromtimestamp(begin, timezone.utc).isoformat()} to "
        f"{datetime.fromtimestamp(end, timezone.utc).isoformat()})",
        flush=True,
    )
    if not candidates:
        return arrivals, departures, []

    def fetch_one(index: int, flight: dict[str, Any]) -> tuple[int, dict[str, Any] | None]:
        """拉取单个历史航班的 track，并保留原始顺序 index 供最终排序。"""
        icao24 = (flight.get("icao24") or "").lower()
        if not icao24:
            return index, None
        t_ref = int(flight.get("lastSeen") or flight.get("firstSeen") or 0)
        if t_ref <= 0:
            return index, None

        try:
            track = client.get(
                "/tracks/all",
                params={"icao24": icao24, "time": t_ref},
                authenticated=True,
            )
            if track and track.get("path"):
                return index, track
        except RuntimeError:
            return index, None

        return index, None

    worker_count = max(1, min(track_workers, len(candidates), max_tracks))
    if worker_count == 1:
        # 串行路径便于调试，也可用于规避 OpenSky 限流或网络不稳定问题。
        for index, flight in enumerate(candidates, start=1):
            _idx, track = fetch_one(index, flight)
            if track:
                tracks_with_index.append((index, track))
            print(
                f"[OpenSky] historical tracks checked: {index}/{len(candidates)} "
                f"downloaded={len(tracks_with_index)}/{max_tracks}",
                flush=True,
            )
            if len(tracks_with_index) >= max_tracks:
                break

        tracks_with_index.sort(key=lambda item: item[0])
        return arrivals, departures, [track for _, track in tracks_with_index]

    print(
        f"[OpenSky] fetching historical tracks with {worker_count} workers...",
        flush=True,
    )
    # 并发下载只用于 track 请求；最终仍按候选航班顺序排序，避免输出顺序随线程完成时间变化。
    executor = ThreadPoolExecutor(max_workers=worker_count)
    futures = {
        executor.submit(fetch_one, index, flight): index
        for index, flight in enumerate(candidates, start=1)
    }
    checked = 0
    try:
        for future in as_completed(futures):
            checked += 1
            index, track = future.result()
            if track:
                tracks_with_index.append((index, track))
            print(
                f"[OpenSky] historical tracks checked: {checked}/{len(candidates)} "
                f"downloaded={len(tracks_with_index)}/{max_tracks}",
                flush=True,
            )
            if len(tracks_with_index) >= max_tracks:
                # 达到目标数量后取消尚未开始的请求，减少不必要的 API 调用。
                for pending in futures:
                    if not pending.done():
                        pending.cancel()
                break
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    tracks_with_index.sort(key=lambda item: item[0])
    tracks = [track for _, track in tracks_with_index[:max_tracks]]
    return arrivals, departures, tracks


def parse_args() -> argparse.Namespace:
    """解析 OpenSky 下载与轨迹转换参数。"""
    parser = argparse.ArgumentParser(description="Download recent airport trajectories from OpenSky")

    parser.add_argument("--mode", choices=["auto", "live", "historical"], default="auto")
    parser.add_argument("--airport", default="CYYC")

    parser.add_argument("--begin", default=None, help="Historical begin (Unix or ISO, e.g. 2026-04-05T00:00:00Z)")
    parser.add_argument("--end", default=None, help="Historical end (Unix or ISO)")

    parser.add_argument("--client-id", default=os.getenv("OPENSKY_CLIENT_ID"))
    parser.add_argument("--client-secret", default=os.getenv("OPENSKY_CLIENT_SECRET"))
    parser.add_argument(
        "--credentials-file",
        default=None,
        help=(
            "Path to a JSON file with OpenSky OAuth credentials. "
            "Accepts keys {clientId, clientSecret} or {client_id, client_secret}. "
            "Defaults to credentials.json next to this script. "
            "Used only when --client-id/--client-secret (and env vars) are absent."
        ),
    )

    parser.add_argument("--bbox-lat-pad", type=float, default=0.30, help="Live mode latitude half-span around selected airport")
    parser.add_argument("--bbox-lon-pad", type=float, default=0.45, help="Live mode longitude half-span around selected airport")

    parser.add_argument("--match-radius-km", type=float, default=35.0, help="Track is accepted only if it comes within this distance of airport")
    parser.add_argument("--landing-radius-km", type=float, default=15.0, help="Touchdown (on_ground) must appear within this distance when landing is required")
    parser.add_argument("--max-end-distance-km", type=float, default=2.5, help="Final approach anchor point must be within this distance of airport")
    parser.add_argument(
        "--altitude-mode",
        choices=["raw", "touchdown-bias", "approach-bias", "auto-bias"],
        default="raw",
        help=(
            "Altitude handling: raw keeps OpenSky barometric altitude; "
            "touchdown-bias estimates a constant offset from near-runway on-ground points; "
            "approach-bias estimates from low-altitude samples near runway; "
            "auto-bias tries touchdown-bias first, then approach-bias"
        ),
    )
    parser.add_argument("--min-ground-samples", type=int, default=2, help="Minimum near-runway on-ground points needed for touchdown-bias estimation")
    parser.add_argument("--max-altitude-bias-m", type=float, default=400.0, help="Reject touchdown-bias estimates whose absolute value exceeds this threshold")
    parser.add_argument("--approach-alt-buffer-m", type=float, default=450.0, help="If on_ground is missing, accept near-airport low-altitude segment below airport_elevation + this buffer")
    parser.add_argument("--approach-window-min", type=int, default=20, help="Keep only this many minutes before landing/closest-approach")
    parser.add_argument("--radius-km", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--allow-partial", action="store_true", help="Allow tracks that do not contain landing/touchdown near airport")
    parser.add_argument("--live-window-hours", type=int, default=12, help="Look back this many hours for /flights/all in live mode")
    parser.add_argument("--max-tracks", type=int, default=80)
    parser.add_argument(
        "--track-workers",
        type=int,
        default=4,
        help=(
            "Concurrent /tracks/all requests in historical mode. "
            "Use 1 for the old serial behavior. Default: 4."
        ),
    )
    parser.add_argument("--max-flights", type=int, default=16)
    parser.add_argument("--min-flights", type=int, default=3, help="If strict landing filter yields fewer than this count, fill with partial approach tracks")
    parser.add_argument("--exclude-ground", action="store_true", help="Exclude on-ground points from exported trajectory")
    parser.add_argument(
        "--disable-normalization",
        action="store_true",
        help="Bypass normalization/filtering and export raw OpenSky tracks directly to CZML input schema",
    )
    parser.add_argument(
        "--input-raw-json",
        default=None,
        help="Existing *_raw_*.json file; bypass network fetch and run conversion/normalization from its tracks",
    )

    parser.add_argument("--output-root", default=None, help="Folder for outputs (default: ./outputs next to script)")
    parser.add_argument("--aeroviz-root", default=None, help="Path to aeroviz-4d folder")

    return parser.parse_args()


def main() -> None:
    """执行下载或离线复用流程，写出 raw payload 和 CZML-input JSON。"""
    args = parse_args()
    script_path = Path(__file__).resolve()

    aeroviz_root = Path(args.aeroviz_root) if args.aeroviz_root else default_aeroviz_root(script_path)

    source_raw_path: Path | None = None
    source_raw_payload: dict[str, Any] | None = None
    selected_airport = args.airport.upper()

    if args.input_raw_json:
        # 离线模式复用既有 raw JSON，只重新执行 normalization/conversion。
        source_raw_path = Path(args.input_raw_json)
        if not source_raw_path.exists():
            raise RuntimeError(f"Input raw JSON not found: {source_raw_path}")

        loaded = json.loads(source_raw_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise RuntimeError("Input raw JSON must be an object with key 'tracks'")
        if not isinstance(loaded.get("tracks"), list):
            raise RuntimeError("Input raw JSON is missing a valid 'tracks' array")

        source_raw_payload = loaded
        selected_airport = str(loaded.get("airport") or selected_airport).upper()

    airport_lat, airport_lon, airport_elev_m = resolve_airport_profile(selected_airport, aeroviz_root)

    output_root = Path(args.output_root) if args.output_root else default_outputs_root(script_path)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    airport_tag = selected_airport.lower()
    airport_output_root = output_root / airport_tag
    airport_output_root.mkdir(parents=True, exist_ok=True)
    raw_path = airport_output_root / f"{airport_tag}_raw_{timestamp}.json"
    czml_input_path = airport_output_root / f"{airport_tag}_czml_input_{timestamp}.json"

    mode = args.mode
    if source_raw_payload is not None:
        mode = str(source_raw_payload.get("mode") or "offline")
    else:
        client_id = args.client_id
        client_secret = args.client_secret
        if not (client_id and client_secret):
            # 优先级：显式参数或环境变量 > credentials-file > 脚本旁 credentials.json。
            credentials_path = (
                Path(args.credentials_file)
                if args.credentials_file
                else script_path.parent / "credentials.json"
            )
            if credentials_path.exists():
                file_id, file_secret = load_credentials_file(credentials_path)
                client_id = client_id or file_id
                client_secret = client_secret or file_secret
                if file_id or file_secret:
                    print(f"[OpenSky] Loaded OAuth credentials from {credentials_path}")
            elif args.credentials_file:
                raise RuntimeError(f"Credentials file not found: {credentials_path}")

        client = OpenSkyClient(
            client_id=client_id,
            client_secret=client_secret,
        )

        if mode == "auto":
            # 有 OAuth 时默认使用 historical，否则退回匿名 live。
            mode = "historical" if client.has_oauth else "live"

    payload: dict[str, Any] = {
        # raw payload 保留足够上下文，便于后续离线复现同一次转换。
        "mode": mode,
        "airport": selected_airport,
        "airport_center": {"lat": airport_lat, "lon": airport_lon},
        "airport_elevation_m": airport_elev_m,
        "altitude_mode": args.altitude_mode,
        "normalization_enabled": (not args.disable_normalization),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if source_raw_payload is not None:
        # 保留原 raw 文件中的下载元数据，只更新本次转换相关字段。
        tracks = source_raw_payload.get("tracks") or []
        payload.update(
            {
                "source_raw_json": str(source_raw_path),
                "tracks_count": len(tracks),
                "tracks": tracks,
            }
        )
        for key in (
            "begin",
            "end",
            "flights_all_count",
            "related_flights_count",
            "arrivals_count",
            "departures_count",
            "bbox",
        ):
            if key in source_raw_payload:
                payload[key] = source_raw_payload[key]
    elif mode == "historical":
        if not client.has_oauth:
            raise RuntimeError(
                "Historical mode requires OAuth credentials. Set OPENSKY_CLIENT_ID/OPENSKY_CLIENT_SECRET "
                "or pass --client-id/--client-secret."
            )

        if args.begin and args.end:
            begin = parse_time_to_unix(args.begin)
            end = parse_time_to_unix(args.end)
        else:
            # 默认上一完整 UTC 日，避免“今天”尚未完全入库导致历史接口结果不稳定。
            now = datetime.now(timezone.utc)
            day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(days=1)
            begin = int(day_start.timestamp())
            end = int((day_start + timedelta(days=1)).timestamp())

        arrivals, departures, tracks = fetch_historical_tracks(
            client,
            airport=selected_airport,
            begin=begin,
            end=end,
            max_tracks=args.max_tracks,
            track_workers=args.track_workers,
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
            airport=selected_airport,
            window_hours=args.live_window_hours,
            max_tracks=args.max_tracks,
        )

        bbox = (
            airport_lat - args.bbox_lat_pad,
            airport_lat + args.bbox_lat_pad,
            airport_lon - args.bbox_lon_pad,
            airport_lon + args.bbox_lon_pad,
        )

        # /flights/all 结果稀疏时，用 bbox states 查询补充仍在附近的飞机。
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
                # bbox 补充可能与航班列表路径重复，按 icao24/startTime 去重。
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

    if args.disable_normalization:
        # 原始模式只转换字段结构，不做机场过滤、进近窗口截取或高度偏置。
        flights = convert_tracks_to_raw_czml_input(
            payload.get("tracks", []),
            include_ground=not args.exclude_ground,
            limit_flights=args.max_flights,
        )
    else:
        # 标准模式输出更适合 AeroViz 展示的最终进近轨迹。
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
            # 严格落地过滤样本过少时，用宽松进近过滤补足，保证可视化至少有数据可看。
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
    if source_raw_path is not None:
        print(f"[OpenSky] source raw input: {source_raw_path}")


if __name__ == "__main__":
    main()
