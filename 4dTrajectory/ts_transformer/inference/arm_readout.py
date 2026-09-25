"""A prediction directory read as an arm of a paired comparison: every scored flight with its predicted endpoint
against the assigned threshold (and its parallel sibling) and the time-free path metrics attached, plus the
Markdown table the readouts print. Shared by the frame-ablation readout (`docs/compare_frame_arms.py`, which it
came from, 2026-09-25) and the residual readouts (`experiments.wind_residual_readout`,
`experiments.straight_in_residual_readout`) — a runner imports this module, never a `docs/` script (L20)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from flight_scenarios.identity import summary_row_key
from flight_scenarios.runway_target import find_threshold
import ts_transformer.geometry.geometric_metrics as gm

# Parallel siblings whose separation a runway-blind frame would average across.
SIBLINGS = {
    "KSJC": {"12L": "12R", "12R": "12L", "30L": "30R", "30R": "30L"},
    "KRDU": {"05L": "05R", "05R": "05L", "23L": "23R", "23R": "23L"},
}


def flight_key(row: dict) -> str:
    return summary_row_key(row)


def _world_en(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    east, north = gm.chart_en(lat, lon, lat0, lon0)
    return float(east), float(north)


def endpoint_geometry(eval_record: dict, states: dict, row: dict) -> dict:
    """The predicted endpoint against the ASSIGNED threshold and its parallel sibling."""
    target = eval_record["target_state"]
    last = states["predicted_states"][-1]
    lat0, lon0, psi = float(target["lat"]), float(target["lon"]), float(target["psi"])
    east, north = _world_en(float(last["lat"]), float(last["lon"]), lat0, lon0)
    cosine, sine = math.cos(psi), math.sin(psi)
    # The first predicted step against the kinematic extrapolation of the anchor state:
    # a model that is not anchored to the aircraft shows up HERE, as a jump at t = dt
    # (the KRDU NW translation, docs/2026-09-03_krdu_nw_endpoint_bias.md).
    anchor = eval_record["initial_state"]
    first = states["predicted_states"][1]
    a_e, a_n = _world_en(float(anchor["lat"]), float(anchor["lon"]), lat0, lon0)
    p_e, p_n = _world_en(float(first["lat"]), float(first["lon"]), lat0, lon0)
    ground_speed = float(anchor["V"]) * math.cos(float(anchor["gamma"]))
    dt = float(first["t"])
    off_e = p_e - (a_e + ground_speed * dt * math.cos(float(anchor["psi"])))
    off_n = p_n - (a_n + ground_speed * dt * math.sin(float(anchor["psi"])))
    result = {
        "first_step_along_m": off_e * cosine + off_n * sine,
        "first_step_lateral_m": off_e * sine - off_n * cosine,
        "first_step_offset_m": math.hypot(off_e, off_n),
        # Same convention as approach_difficulty: positive to the RIGHT of the inbound
        # course; along positive PAST the threshold.
        "endpoint_cross_track_m": east * sine - north * cosine,
        "endpoint_along_track_m": east * cosine + north * sine,
        "endpoint_height_m": float(last["alt"]) - float(target["alt"]),
        "closer_to_sibling": None,
    }
    airport, runway = row.get("arr_airport"), row.get("runway")
    sibling = SIBLINGS.get(airport, {}).get(runway)
    if sibling is not None:
        other = find_threshold(airport, sibling)
        if other is not None:
            e2, n2 = _world_en(
                float(last["lat"]), float(last["lon"]), float(other["lat"]), float(other["lon"])
            )
            # Distance to each centreline (cross-track magnitude), not to the point.
            own = abs(result["endpoint_cross_track_m"])
            theirs = abs(e2 * sine - n2 * cosine)
            result["closer_to_sibling"] = bool(theirs < own)
            result["sibling_cross_track_m"] = e2 * sine - n2 * cosine
    return result


def load_arm(pred_dir: Path, *, geometry_truth: str = gm.GEOMETRY_TRUTH_CLOSED) -> dict[str, dict]:
    """Every scored flight of one prediction directory, keyed by flight_key, with the
    endpoint geometry and the time-free path metrics attached."""
    summary = json.loads((pred_dir / "summary.json").read_text())
    rows: dict[str, dict] = {}
    for row in summary.get("results", []):
        if row.get("ade_m") is None or row.get("route_tortuosity") is None:
            continue
        row = dict(row)
        eval_record = json.loads((pred_dir / row["eval_file"]).read_text())
        states = json.loads((pred_dir / row["states_file"]).read_text())
        row.update(endpoint_geometry(eval_record, states, row))
        row.update(gm.record_geometry(eval_record, states, row, geometry_truth=geometry_truth))
        rows[flight_key(row)] = row
    return rows


def fmt(value: float, digits: int = 0) -> str:
    return f"{value:.{digits}f}"


def print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
