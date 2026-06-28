"""Build comparison CZML(s) for optimized scenarios.

Two modes:

* **Single** (``--state-file``) — one ``*_states.json`` from
  ``4dTrajectory/optimization/scenario_optimization.py`` → one CZML with three coloured,
  time-dynamic trajectories:
    • reference  (white)  — the observed ADS-B track, copied from the airport's trajectories.czml
    • optimizer  (cyan)   — the optimizer's plan        (optimizer_states)
    • simulator  (orange) — the real forward rollout    (simulator_states)

* **Batch** (``--summary``) — the run's ``summary.json`` → **one combined CZML per runway**.
  Every flight on one map: solved flights get the three paths above (entity ids namespaced by
  flight id); **unsolved flights get their reference only, in dark red (FAILED_COLOR)** so the
  success/failure spread is visible at a glance. The reference is found by flight id.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_layout import airport_data_path
from generate_czml import build_document_packet, build_position_property

# A fixed display epoch (state times are offsets in seconds from it), matching the
# convention generate_czml uses. The relative motion is what matters, not the wall clock.
EPOCH = datetime(2026, 4, 1, 8, 0, 0, tzinfo=timezone.utc)

# RGBA colours (0-255) for the three trajectories.
REFERENCE_COLOR = (235, 235, 235, 200)   # observed ADS-B (white)
OPTIMIZER_COLOR = (0, 200, 255, 220)     # optimizer plan (cyan)
SIMULATOR_COLOR = (255, 140, 0, 220)     # simulator rollout (orange)
FAILED_COLOR = (200, 60, 60, 200)        # unsolved scenario — reference only, flagged dark red

_TRAIL_TIME_S = 100000  # keep the whole path drawn


# ── state sequence -> CZML geometry ───────────────────────────────────────────

def _states_to_waypoints(states: list[dict[str, Any]]) -> list[tuple[float, float, float, float]]:
    """Convert a list of state dicts (``{t, lat, lon, alt, …}``) to CZML waypoints.

    ``build_position_property`` wants ``(offset_sec, lon, lat, alt_m)`` tuples — note the
    order: time, then **lon before lat** (GeoJSON/CZML convention), then metres.
    """
    return [(s["t"], s["lon"], s["lat"], s["alt"]) for s in states]


# ── copy the matching reference flight from the ADS-B CZML ─────────────────────

def _reference_entity_from_adsb(
    adsb_czml: list[dict[str, Any]],
    flight_id: str | None,
    color_rgba: tuple[int, int, int, int],
    *,
    entity_id: str = "scenario-reference",
    name: str | None = None,
) -> dict[str, Any] | None:
    """Find the observed flight in ``adsb_czml`` and copy it as the reference trajectory.

    ``adsb_czml`` is the loaded ``trajectories.czml`` (element 0 is the ``"document"``
    packet; the rest are flight entities whose ``id`` is the flight id). Find the entity
    whose ``id`` matches ``flight_id``, **deep-copy** it (so we don't mutate the source),
    recolour its ``path`` material to ``color_rgba``, and re-id/-name it (``entity_id`` /
    ``name``) so several references can coexist in one combined CZML.
    """
    for packet in adsb_czml:
        if packet.get("id") == flight_id:
            entity = copy.deepcopy(packet)
            entity["id"] = entity_id
            entity["name"] = name or f"Reference {flight_id}"
            entity["path"]["material"]["solidColor"]["color"]["rgba"] = list(color_rgba)
            return entity
    return None        # no matching flight in the ADS-B CZML


# ── Wired: entity assembly + document + combined CZML ─────────────────────────

def _build_trajectory_entity(
    entity_id: str,
    name: str,
    states: list[dict[str, Any]],
    color_rgba: tuple[int, int, int, int],
) -> dict[str, Any]:
    """A coloured, time-dynamic path entity from a state sequence (uses TODO ①)."""
    waypoints = _states_to_waypoints(states)
    return {
        "id": entity_id,
        "name": name,
        "position": build_position_property(EPOCH, waypoints),
        "point": {"pixelSize": 9, "color": {"rgba": list(color_rgba)}},
        "path": {
            "show": True,
            "leadTime": 0,
            "trailTime": _TRAIL_TIME_S,
            "width": 3,
            "material": {"solidColor": {"color": {"rgba": list(color_rgba)}}},
        },
        "label": {
            "text": name,
            "font": "12px sans-serif",
            "fillColor": {"rgba": list(color_rgba)},
            "outlineColor": {"rgba": [0, 0, 0, 255]},
            "outlineWidth": 2,
            "style": "FILL_AND_OUTLINE",
            "verticalOrigin": "BOTTOM",
            "pixelOffset": {"cartesian2": [0, -28]},
        },
    }


def build_comparison_czml(
    state_data: dict[str, Any],
    adsb_czml: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assemble the combined CZML: [document, reference, optimizer, simulator]."""
    optimizer_states = state_data["optimizer_states"]
    simulator_states = state_data["simulator_states"]
    flight_id = state_data.get("source", {}).get("id")

    max_t = max(
        float(state_data.get("final_time_s", 0.0)),
        _last_time(optimizer_states),
        _last_time(simulator_states),
    )
    end_dt = EPOCH.fromtimestamp(EPOCH.timestamp() + max_t, tz=timezone.utc)
    document = build_document_packet(EPOCH, end_dt, multiplier=30)

    packets = [document]
    reference = _reference_entity_from_adsb(adsb_czml, flight_id, REFERENCE_COLOR)
    if reference is not None:
        packets.append(reference)
    packets.append(_build_trajectory_entity("scenario-optimizer", "Optimizer", optimizer_states, OPTIMIZER_COLOR))
    packets.append(_build_trajectory_entity("scenario-simulator", "Simulator", simulator_states, SIMULATOR_COLOR))
    return packets


def _last_time(states: list[dict[str, Any]]) -> float:
    return float(states[-1]["t"]) if states else 0.0


# ── Batch: one combined comparison CZML per runway (from a summary.json) ───────

def build_runway_comparison(
    results: list[dict[str, Any]],
    states_dir: str | Path,
    adsb_czml: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combined CZML for one runway: every flight on one map.

    Each result (a row from ``scenario_optimization``'s ``summary.json``) becomes:
      • solved → reference (white) + optimizer (cyan) + simulator (orange), entity ids
        namespaced by flight id so they don't collide;
      • failed → reference only, in the dark-red FAILED_COLOR, labelled "(unsolved)".

    The reference is found in ``adsb_czml`` by the flight id (see ``_reference_entity_from_adsb``).
    """
    states_dir = Path(states_dir)
    entities: list[dict[str, Any]] = []
    max_t = 0.0
    for result in results:
        flight_id = result.get("id")
        solved = result.get("status") == "solved" and result.get("states_file")
        if solved:
            state_data = json.loads((states_dir / result["states_file"]).read_text(encoding="utf-8"))
            optimizer_states = state_data["optimizer_states"]
            simulator_states = state_data["simulator_states"]
            max_t = max(max_t, float(state_data.get("final_time_s", 0.0)),
                        _last_time(optimizer_states), _last_time(simulator_states))
            reference = _reference_entity_from_adsb(
                adsb_czml, flight_id, REFERENCE_COLOR,
                entity_id=f"ref-{flight_id}", name=f"Ref {flight_id}",
            )
            if reference is not None:
                entities.append(reference)
            entities.append(_build_trajectory_entity(f"opt-{flight_id}", f"Opt {flight_id}", optimizer_states, OPTIMIZER_COLOR))
            entities.append(_build_trajectory_entity(f"sim-{flight_id}", f"Sim {flight_id}", simulator_states, SIMULATOR_COLOR))
        else:
            reference = _reference_entity_from_adsb(
                adsb_czml, flight_id, FAILED_COLOR,
                entity_id=f"ref-{flight_id}", name=f"Ref {flight_id} (unsolved)",
            )
            if reference is not None:
                entities.append(reference)

    end_dt = EPOCH.fromtimestamp(EPOCH.timestamp() + max(max_t, 1.0), tz=timezone.utc)
    document = build_document_packet(EPOCH, end_dt, multiplier=30)
    return [document] + entities


def group_results_by_runway(
    summary: dict[str, Any], fallback_airport: str | None = None
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group a summary's results by ``(airport, runway)`` (one combined CZML per group)."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in summary.get("results", []):
        airport = result.get("arr_airport") or fallback_airport or "UNK"
        runway = result.get("runway") or "unknown"
        grouped[(airport, runway)].append(result)
    return grouped


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_adsb(airport: str, override: str | None) -> list[dict[str, Any]]:
    path = Path(override) if override else airport_data_path(airport, "trajectories.czml")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build comparison CZML(s) for optimized scenarios")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--state-file", help="A single *_states.json -> one comparison CZML (with --output)")
    mode.add_argument("--summary", help="summary.json from scenario_optimization -> one combined CZML per runway (with --output-dir)")
    parser.add_argument("--states-dir", default=None, help="Dir with the *_states.json (batch mode; defaults to the summary's dir)")
    parser.add_argument("--airport", default="KRDU", help="Airport ICAO; locates the ADS-B trajectories.czml (batch: fallback when a result has no arr_airport)")
    parser.add_argument("--adsb-czml", default=None, help="ADS-B CZML path override (single mode)")
    parser.add_argument("--output", default=None, help="Output CZML path (single mode)")
    parser.add_argument("--output-dir", default=None, help="Output dir for per-runway CZMLs (batch mode)")
    args = parser.parse_args()

    if args.state_file:
        if not args.output:
            parser.error("--state-file requires --output")
        state_data = json.loads(Path(args.state_file).read_text(encoding="utf-8"))
        czml = build_comparison_czml(state_data, _load_adsb(args.airport, args.adsb_czml))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(czml, indent=2), encoding="utf-8")
        print(f"✓ wrote comparison CZML ({len(czml)} packets) -> {output}")
        return

    # Batch: one combined CZML per (airport, runway), driven by the summary.
    if not args.output_dir:
        parser.error("--summary requires --output-dir")
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    states_dir = Path(args.states_dir) if args.states_dir else summary_path.parent
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adsb_cache: dict[str, list[dict[str, Any]]] = {}

    def adsb_for(airport: str) -> list[dict[str, Any]]:
        if airport not in adsb_cache:
            adsb_cache[airport] = _load_adsb(airport, None)
        return adsb_cache[airport]

    groups = group_results_by_runway(summary, fallback_airport=args.airport)
    for (airport, runway), results in sorted(groups.items()):
        czml = build_runway_comparison(results, states_dir, adsb_for(airport))
        out_path = out_dir / f"comparison_{airport}_{runway}.czml"
        out_path.write_text(json.dumps(czml, indent=2), encoding="utf-8")
        failed = sum(1 for r in results if r.get("status") != "solved")
        print(f"✓ {out_path.name}: {len(results)} flight(s), {failed} unsolved (red) -> {len(czml)} packets")

    rate = summary.get("failure_rate")
    rate_str = f"{rate:.1%}" if isinstance(rate, (int, float)) else "n/a"
    print(f"✓ wrote {len(groups)} runway CZML(s) to {out_dir}; overall failure rate {rate_str}")


if __name__ == "__main__":
    main()
