"""Teaching scaffold: build a 3-way comparison CZML for an optimized scenario.

Reads a ``*_states.json`` produced by ``4dTrajectory/optimization/scenario_optimization.py``
(``{source, final_time_s, optimizer_states[], simulator_states[]}``) and writes **one
combined CZML** with three coloured, time-dynamic trajectories so they can be compared in
the frontend on Cesium's clock:

  • reference  (white)  — the observed ADS-B track, copied from the airport's trajectories.czml
  • optimizer  (cyan)   — the optimizer's plan        (optimizer_states)
  • simulator  (orange) — the real forward rollout    (simulator_states)

This is a **teaching scaffold**: the loading / document packet / entity styling / IO / CLI
are wired; the two core steps — converting a state sequence to CZML geometry (TODO ①) and
copying the matching reference flight out of the ADS-B CZML (TODO ②) — are documented TODOs.
"""

from __future__ import annotations

import argparse
import copy
import json
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

_TRAIL_TIME_S = 100000  # keep the whole path drawn


# ── TODO ① — state sequence -> CZML geometry ──────────────────────────────────

def _states_to_waypoints(states: list[dict[str, Any]]) -> list[tuple[float, float, float, float]]:
    """Convert a list of state dicts (``{t, lat, lon, alt, …}``) to CZML waypoints.

    ``build_position_property`` wants ``(offset_sec, lon, lat, alt_m)`` tuples — note the
    order: time, then **lon before lat** (GeoJSON/CZML convention), then metres.

    # TODO ①: return [(s["t"], s["lon"], s["lat"], s["alt"]) for s in states]
    """
    raise NotImplementedError(
        "TODO ①: convert the state dicts to (t, lon, lat, alt) waypoints — see the comment."
    )


# ── TODO ② — copy the matching reference flight from the ADS-B CZML ────────────

def _reference_entity_from_adsb(
    adsb_czml: list[dict[str, Any]],
    flight_id: str | None,
    color_rgba: tuple[int, int, int, int],
) -> dict[str, Any] | None:
    """Find the observed flight in ``adsb_czml`` and copy it as the reference trajectory.

    ``adsb_czml`` is the loaded ``trajectories.czml`` (element 0 is the ``"document"``
    packet; the rest are flight entities whose ``id`` is the flight id). Find the entity
    whose ``id`` matches ``flight_id``, **deep-copy** it (so we don't mutate the source),
    recolour its ``path`` material to ``color_rgba``, and re-id/-name it as the reference.

    # TODO ②:
    #   for packet in adsb_czml:
    #       if packet.get("id") == flight_id:
    #           entity = copy.deepcopy(packet)
    #           entity["id"] = "scenario-reference"
    #           entity["name"] = f"Reference {flight_id}"
    #           entity["path"]["material"]["solidColor"]["color"]["rgba"] = list(color_rgba)
    #           return entity
    #   return None        # no matching flight in the ADS-B CZML
    """
    raise NotImplementedError(
        "TODO ②: find + deep-copy + recolour the matching ADS-B flight — see the comment."
    )


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


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 3-way comparison CZML for an optimized scenario")
    parser.add_argument("--state-file", required=True, help="A *_states.json from scenario_optimization.py")
    parser.add_argument("--airport", default="KRDU", help="Airport ICAO (locates the ADS-B trajectories.czml)")
    parser.add_argument("--adsb-czml", default=None, help="ADS-B CZML path (defaults to the airport's trajectories.czml)")
    parser.add_argument("--output", required=True, help="Output combined CZML path")
    args = parser.parse_args()

    state_data = json.loads(Path(args.state_file).read_text(encoding="utf-8"))
    adsb_path = Path(args.adsb_czml) if args.adsb_czml else airport_data_path(args.airport, "trajectories.czml")
    adsb_czml = json.loads(Path(adsb_path).read_text(encoding="utf-8"))

    czml = build_comparison_czml(state_data, adsb_czml)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(czml, indent=2), encoding="utf-8")
    print(f"✓ wrote comparison CZML ({len(czml)} packets) -> {output}")


if __name__ == "__main__":
    main()
