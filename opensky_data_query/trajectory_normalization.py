#!/usr/bin/env python3
"""
Pure normalization utilities for OpenSky trajectory data.

This module converts raw OpenSky tracks into the intermediate flight JSON schema
consumed by aeroviz-4d/python/generate_czml.py.

Design goal: keep this module free of network, filesystem, and subprocess logic.
"""

from __future__ import annotations

import math
from typing import Any


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两组经纬度之间的大圆距离，单位为公里。"""
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def sanitize_callsign(value: str | None, fallback: str) -> str:
    """清理航班呼号；呼号缺失时使用稳定的备用标识。"""
    text = (value or "").strip()
    return text if text else fallback.upper()


def track_to_raw_czml_flight(
    track: dict[str, Any],
    *,
    include_ground: bool,
) -> dict[str, Any] | None:
    """将一条 OpenSky track 直接转换为 CZML 输入记录，不做进近过滤和高度修正。"""
    path = track.get("path") or []
    if not path:
        return None

    parsed: list[tuple[int, float, float, float, bool]] = []
    for wp in path:
        # OpenSky path 点格式为 [time, lat, lon, baro_altitude, true_track, on_ground]。
        # 这里丢弃字段不足、坐标缺失或高度无效的点，避免后续生成非法 CZML。
        if not wp or len(wp) < 6:
            continue
        t, lat, lon, alt_m, _trk, on_ground = wp
        if t is None or lat is None or lon is None or alt_m is None:
            continue
        alt = float(alt_m)
        if math.isnan(alt):
            continue
        parsed.append((int(t), float(lon), float(lat), alt, bool(on_ground)))

    if not parsed:
        return None

    parsed.sort(key=lambda x: x[0])

    waypoints_abs: list[tuple[int, float, float, float]] = []
    for t, lon, lat, alt, gnd in parsed:
        if not include_ground and gnd:
            continue
        waypoints_abs.append((t, lon, lat, alt))

    if not waypoints_abs:
        return None

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
        "altitude_correction_mode": "raw-pass-through",
        "altitude_bias_m": 0.0,
        "altitude_bias_applied": False,
        "altitude_bias_source": "none",
        "altitude_bias_scope": "none",
        "waypoints": rel,
    }


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
    """将一条 OpenSky track 转换为经过机场相关性、进近窗口和高度处理后的 CZML 输入记录。"""
    path = track.get("path") or []
    if not path:
        return None

    parsed: list[tuple[int, float, float, float, bool, float]] = []
    for wp in path:
        # 标准化字段顺序为 (time, lon, lat, altitude_m, on_ground, airport_distance_km)，
        # 后续过滤逻辑都基于这个内部结构工作。
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

    # 剔除明显不可用的轨迹，例如所有高度都为 0 m 的陈旧 track。
    if max(row[3] for row in parsed) <= 1.0:
        return None

    def apply_uniform_altitude_bias(
        rows: list[tuple[int, float, float, float, bool, float]],
        bias: float,
    ) -> list[tuple[int, float, float, float, bool, float]]:
        """对整条轨迹应用同一个高度偏置，保持垂直剖面形状不变。"""
        if abs(bias) <= 1e-12:
            return rows

        shifted = [
            (t, lon, lat, alt + bias, gnd, d)
            for t, lon, lat, alt, gnd, d in rows
        ]

        # 不变量：所有航点必须应用完全相同的高度修正量。
        for src, dst in zip(rows, shifted):
            delta = dst[3] - src[3]
            if abs(delta - bias) > 1e-9:
                raise RuntimeError("Non-uniform altitude correction detected")
        return shifted

    def median(values: list[float]) -> float:
        """返回数值列表的中位数，用于降低异常高度样本对偏置估计的影响。"""
        ordered = sorted(values)
        n = len(ordered)
        mid = n // 2
        if n % 2 == 1:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    bias_m = 0.0
    bias_applied = False
    bias_source = "none"
    bias_scope = "none"
    ground_samples = 0
    approach_samples = 0

    def try_apply_bias(candidate_bias: float, source: str) -> None:
        """在候选偏置没有超过安全阈值时记录本次高度修正方案。"""
        nonlocal bias_m, bias_applied, bias_source
        if abs(candidate_bias) <= max_altitude_bias_m:
            bias_m = candidate_bias
            bias_applied = True
            bias_source = source

    if altitude_mode in {"touchdown-bias", "auto-bias"}:
        # 首选真实落地点证据：机场附近 on_ground 点的中位高度应接近机场标高。
        touchdown_alts = [
            alt
            for _t, _lon, _lat, alt, gnd, d in parsed
            if gnd and d <= landing_radius_km
        ]
        ground_samples = len(touchdown_alts)
        if ground_samples >= min_ground_samples:
            candidate_bias = airport_elev_m - median(touchdown_alts)
            try_apply_bias(candidate_bias, source="touchdown")

    if (not bias_applied) and altitude_mode in {"approach-bias", "auto-bias"}:
        # 如果 OpenSky 没有提供可靠的 on_ground 点，则用机场附近低高度样本保守估计偏置。
        low_alt_threshold = airport_elev_m + approach_alt_buffer_m
        near_low_alts = [
            alt
            for _t, _lon, _lat, alt, _gnd, d in parsed
            if d <= landing_radius_km and alt <= low_alt_threshold
        ]
        approach_samples = len(near_low_alts)
        if approach_samples >= min_ground_samples:
            # 没有落地点证据时只允许向上修正，避免把进近轨迹压到地形或跑道以下。
            reference_alt = min(near_low_alts)
            candidate_bias = airport_elev_m - reference_alt
            if candidate_bias >= 0.0:
                try_apply_bias(candidate_bias, source="approach")

    if bias_applied:
        parsed = apply_uniform_altitude_bias(parsed, bias_m)
        bias_scope = "uniform-all-waypoints"

    # 机场相关性：轨迹至少要有一个点进入目标机场匹配半径。
    min_dist = min(
        d
        for _t, _lon, _lat, _alt, _gnd, d in parsed
    )
    if min_dist > match_radius_km:
        return None

    closest_idx = min(range(len(parsed)), key=lambda i: parsed[i][5])
    closest_dist_km = parsed[closest_idx][5]
    # 最终进近锚点必须足够接近机场，避免把只是路过附近的航班纳入展示。
    if closest_dist_km > max_end_distance_km:
        return None

    landing_reference_t: int | None = None

    if require_landing:
        # 严格模式优先要求机场附近出现 on_ground，并确认此前存在空中点，避免纯地面滑行轨迹。
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
            # 对缺失 on_ground 的数据，允许用“先高后低并接近机场”的模式识别进近结束点。
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
        # 宽松模式没有明确落地参考时，用距离机场最近的点作为进近窗口锚点。
        landing_reference_t = parsed[closest_idx][0]

    # 只保留落地或最近点之前的最后进近窗口，减少巡航和远端航段对可视化的干扰。
    windowed = parsed
    if approach_window_min > 0:
        window_sec = approach_window_min * 60
        window_start_t = landing_reference_t - window_sec
        windowed = [row for row in parsed if window_start_t <= row[0] <= landing_reference_t]

        # 采样过稀时，从最近点附近截取一个有界局部片段，尽量保留可绘制的进近段。
        if len(windowed) < 2:
            local_start = max(0, closest_idx - 40)
            local_rows = parsed[local_start: closest_idx + 1]
            windowed = [
                row for row in local_rows
                if (landing_reference_t - window_sec) <= row[0] <= landing_reference_t
            ]

        if len(windowed) < 2:
            return None

    # 转换为生成器需要的航点结构，并按配置决定是否保留地面点。
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
        "altitude_bias_source": bias_source,
        "altitude_bias_scope": bias_scope,
        "altitude_ground_samples": ground_samples,
        "altitude_approach_samples": approach_samples,
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
    """批量转换 OpenSky tracks，输出去重后的标准化 CZML 输入航班列表。"""
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

        # CZML entity id 必须唯一；同一呼号出现多次时追加递增后缀。
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


def convert_tracks_to_raw_czml_input(
    tracks: list[dict[str, Any]],
    *,
    include_ground: bool,
    limit_flights: int,
) -> list[dict[str, Any]]:
    """批量转换 OpenSky tracks，输出未标准化的 CZML 输入航班列表。"""
    out: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    for track in tracks:
        item = track_to_raw_czml_flight(
            track,
            include_ground=include_ground,
        )
        if not item:
            continue

        # CZML entity id 必须唯一；同一呼号出现多次时追加递增后缀。
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
