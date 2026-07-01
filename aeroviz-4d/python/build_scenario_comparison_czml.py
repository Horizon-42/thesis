"""Build comparison CZML(s) for optimized scenarios.

Two modes:

* **Single** (``--state-file``) — one ``*_states.json`` from
  ``4dTrajectory/optimization/scenario_optimization.py`` → one CZML with three coloured,
  time-dynamic trajectories:
    • reference  (white)  — the observed ADS-B track, copied from the airport's trajectories.czml
    • optimizer  (orange) — the optimizer's plan        (optimizer_states)  ["Optimize states"]
    • simulator  (blue)   — the real forward rollout    (simulator_states)  ["Optimize results"]

* **Batch** (``--summary``) — the run's ``summary.json`` → **one combined CZML per runway**
  plus a single ``comparison_index.json``. Every flight on one map: solved flights get the
  three paths above; **unsolved flights get their reference only, in dark red (FAILED_COLOR)**.
  Each entity has a globally-unique id ``{kind}-{flightId}_{runway}`` and a ``properties`` bag
  (``group``/``flightId``/``kind``/``runway``/``airport``/``status``) so the frontend can
  group and **randomly sample** trajectories. Entities default to ``show=false`` (override with
  ``--start-visible``); the frontend reads ``comparison_index.json`` (one record per group, with
  its ``initialState`` and the CZML file + entity ids it owns), samples a subset, and reveals
  only those. The reference is found by flight id in the airport's ``trajectories.czml``.
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

# RGBA colours (0-255) for the three trajectories. These must stay in sync with the frontend
# legend / path colours in aeroviz-4d/src/utils/trajectoryRenderModel.ts (COMPARISON_KIND_COLORS):
#   optimizer → "Optimize states" (orange),  simulator → "Optimize results" (blue).
REFERENCE_COLOR = (235, 235, 235, 200)   # observed ADS-B (white)
OPTIMIZER_COLOR = (255, 140, 0, 220)     # optimizer plan — "Optimize states" (orange)
SIMULATOR_COLOR = (40, 120, 255, 220)    # simulator rollout — "Optimize results" (blue)
FAILED_COLOR = (200, 60, 60, 200)        # unsolved scenario — reference only, flagged dark red

# Trailing-tail length (seconds) for the optimizer/simulator paths: the tail fades behind the
# moving aircraft as playback advances, so the head (current position) is distinguishable from the
# tail. Matches generate_czml.py's observed-track trail (300 s) — which the reference inherits — so
# all three comparison trajectories fade identically.
_TRAIL_TIME_S = 300


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
    properties: dict[str, Any] | None = None,
    show: bool = True,
) -> dict[str, Any] | None:
    """Find the observed flight in ``adsb_czml`` and copy it as the reference trajectory.

    ``adsb_czml`` is the loaded ``trajectories.czml`` (element 0 is the ``"document"``
    packet; the rest are flight entities whose ``id`` is the flight id). Find the entity
    whose ``id`` matches ``flight_id``, **deep-copy** it (so we don't mutate the source),
    recolour its ``path`` material to ``color_rgba``, and re-id/-name it (``entity_id`` /
    ``name``) so several references can coexist in one combined CZML.

    ``properties`` attaches a CZML custom-property bag (so the frontend can group/sample)
    and ``show`` sets the entity-level visibility (``False`` ⇒ hidden until revealed).
    """
    for packet in adsb_czml:
        if packet.get("id") == flight_id:
            entity = copy.deepcopy(packet)
            entity["id"] = entity_id
            entity["name"] = name or f"Reference {flight_id}"
            entity["show"] = show
            entity["path"]["material"]["solidColor"]["color"]["rgba"] = list(color_rgba)
            if properties is not None:
                entity["properties"] = properties
            return entity
    return None        # no matching flight in the ADS-B CZML


# ── Wired: entity assembly + document + combined CZML ─────────────────────────

def _build_trajectory_entity(
    entity_id: str,
    name: str,
    states: list[dict[str, Any]],
    color_rgba: tuple[int, int, int, int],
    *,
    properties: dict[str, Any] | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """A coloured, time-dynamic path entity from a state sequence.

    ``properties`` attaches a CZML custom-property bag (so the frontend can group/sample
    by ``group``/``kind``/…) and ``show`` sets entity-level visibility (``False`` ⇒ the
    whole entity is hidden until the frontend reveals it).
    """
    waypoints = _states_to_waypoints(states)
    entity: dict[str, Any] = {
        "id": entity_id,
        "name": name,
        "show": show,
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
    if properties is not None:
        entity["properties"] = properties
    return entity


def build_comparison_czml(
    state_data: dict[str, Any],
    adsb_czml: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assemble the combined CZML: [document, reference, optimizer, simulator]."""
    optimizer_states = state_data["optimizer_states"]
    simulator_states = state_data["simulator_states"]
    flight_id = state_data.get("source", {}).get("id")

    reference = _reference_entity_from_adsb(adsb_czml, flight_id, REFERENCE_COLOR)
    optimizer = _build_trajectory_entity("scenario-optimizer", "Optimizer", optimizer_states, OPTIMIZER_COLOR)
    simulator = _build_trajectory_entity("scenario-simulator", "Simulator", simulator_states, SIMULATOR_COLOR)
    entities = [e for e in (reference, optimizer, simulator) if e is not None]

    # Clock spans the longest of the trajectories actually present — reference included, so the
    # observed track (usually the longest) is never truncated.
    max_t = max((_entity_last_time(e) for e in entities), default=0.0)
    end_dt = EPOCH.fromtimestamp(EPOCH.timestamp() + max(max_t, 1.0), tz=timezone.utc)
    document = build_document_packet(EPOCH, end_dt, multiplier=30)
    return [document] + entities


def _last_time(states: list[dict[str, Any]]) -> float:
    return float(states[-1]["t"]) if states else 0.0


def _entity_last_time(entity: dict[str, Any] | None) -> float:
    """Last time offset (sec) of a built entity's time-sampled position; 0 if it has none.

    The document clock must span the LONGEST trajectory in the file. The reference (observed)
    tracks routinely run far longer than the optimizer/simulator — and on a runway with no
    solved scenarios they are the ONLY trajectories — so the clock has to be derived from every
    entity's position, not just the optimizer/simulator states. (All entities share `EPOCH`, so
    their offsets are directly comparable.)
    """
    if entity is None:
        return 0.0
    cd = entity.get("position", {}).get("cartographicDegrees", [])
    return float(cd[-4]) if len(cd) >= 4 else 0.0


# ── Batch: one combined comparison CZML per runway (from a summary.json) ───────

_INITIAL_STATE_KEYS = ("lat", "lon", "alt", "V", "psi", "gamma")


def _initial_state(states: list[dict[str, Any]]) -> dict[str, float] | None:
    """The first sample's kinematic state (used as a group's ``initialState`` in the index)."""
    if not states:
        return None
    first = states[0]
    return {key: first[key] for key in _INITIAL_STATE_KEYS}


def _traj_properties(
    group: str, flight_id: str | None, kind: str, runway: str, airport: str, status: str
) -> dict[str, Any]:
    """CZML custom-property bag the frontend uses to group / sample / colour-key entities.

    ``group`` is the per-flight key (unique within the run); ``kind`` is one of
    ``reference`` / ``optimizer`` / ``simulator``.
    """
    return {
        "group": group, "flightId": flight_id, "kind": kind,
        "runway": runway, "airport": airport, "status": status,
    }


def build_runway_comparison(
    results: list[dict[str, Any]],
    states_dir: str | Path,
    adsb_czml: list[dict[str, Any]],
    *,
    airport: str = "UNK",
    start_hidden: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Combined CZML for one runway **plus** its index records (every flight on one map).

    Each result (a row from ``scenario_optimization``'s ``summary.json``) becomes:
      • solved → reference (white) + optimizer (orange) + simulator (blue);
      • failed → reference only, in the dark-red FAILED_COLOR, labelled "(unsolved)".

    Every entity gets a globally-unique id ``{kind}-{flightId}_{runway}`` (so duplicate
    ``(flightId, runway)`` rows — which collide on the *same* states file — can't produce
    colliding CZML ids) and a ``properties`` bag (``group``/``kind``/…). Entities are
    ``show=False`` when ``start_hidden`` so the frontend renders only the groups it samples.

    Returns ``(czml_packets, index_records)``. Each index record describes one group:
    its id, flight id, runway, status, initial state, and the entity ids that belong to it.
    """
    states_dir = Path(states_dir)
    show = not start_hidden

    # Collapse to one row per group (= ``flightId_runway``), preferring a solved row. A flight
    # can appear in the summary as both a failed attempt and a solved one (their states
    # filenames collide on overwrite); the solved result is the one to show, regardless of row
    # order. One-per-group also subsumes the duplicate-states-file dedup (group ↔ file is 1:1).
    best: dict[str, dict[str, Any]] = {}
    for result in results:
        group = f"{result.get('id')}_{result.get('runway') or 'unknown'}"
        is_solved = result.get("status") == "solved" and result.get("states_file")
        current = best.get(group)
        if current is None or (is_solved and current.get("status") != "solved"):
            best[group] = result

    entities: list[dict[str, Any]] = []
    index_records: list[dict[str, Any]] = []

    for group, result in best.items():
        flight_id = result.get("id")
        runway = result.get("runway") or "unknown"
        solved = result.get("status") == "solved" and result.get("states_file")
        entity_ids: list[str] = []

        if solved:
            states_file = result["states_file"]
            state_data = json.loads((states_dir / states_file).read_text(encoding="utf-8"))
            optimizer_states = state_data["optimizer_states"]
            simulator_states = state_data["simulator_states"]

            reference = _reference_entity_from_adsb(
                adsb_czml, flight_id, REFERENCE_COLOR,
                entity_id=f"ref-{group}", name=f"Ref {flight_id}",
                properties=_traj_properties(group, flight_id, "reference", runway, airport, "solved"),
                show=show,
            )
            if reference is not None:
                entities.append(reference)
                entity_ids.append(f"ref-{group}")
            entities.append(_build_trajectory_entity(
                f"opt-{group}", f"Opt {flight_id}", optimizer_states, OPTIMIZER_COLOR,
                properties=_traj_properties(group, flight_id, "optimizer", runway, airport, "solved"),
                show=show))
            entity_ids.append(f"opt-{group}")
            entities.append(_build_trajectory_entity(
                f"sim-{group}", f"Sim {flight_id}", simulator_states, SIMULATOR_COLOR,
                properties=_traj_properties(group, flight_id, "simulator", runway, airport, "solved"),
                show=show))
            entity_ids.append(f"sim-{group}")

            index_records.append({
                "group": group, "flightId": flight_id, "runway": runway, "airport": airport,
                "status": "solved", "finalTimeS": float(state_data.get("final_time_s", 0.0)),
                "initialState": _initial_state(optimizer_states or simulator_states),
                "entities": entity_ids,
            })
        else:
            reference = _reference_entity_from_adsb(
                adsb_czml, flight_id, FAILED_COLOR,
                entity_id=f"ref-{group}", name=f"Ref {flight_id} (unsolved)",
                properties=_traj_properties(group, flight_id, "reference", runway, airport, "failed"),
                show=show,
            )
            if reference is not None:
                entities.append(reference)
                entity_ids.append(f"ref-{group}")
            index_records.append({
                "group": group, "flightId": flight_id, "runway": runway, "airport": airport,
                "status": "failed", "finalTimeS": None, "initialState": None,
                "entities": entity_ids,
            })

    # Span the clock over the LONGEST trajectory actually in the file — references included.
    # (Deriving it only from solved opt/sim states collapsed the clock to ~1 s on runways with
    #  no solved scenarios, freezing every reference track at its start point.)
    max_t = max((_entity_last_time(e) for e in entities), default=0.0)
    end_dt = EPOCH.fromtimestamp(EPOCH.timestamp() + max(max_t, 1.0), tz=timezone.utc)
    document = build_document_packet(EPOCH, end_dt, multiplier=30)
    return [document] + entities, index_records


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


def _upsert_category(
    manifest_path: Path, *, key: str, label: str, directory: str, group_count: int
) -> int:
    """Add/replace one category in the shared ``categories.json`` manifest.

    Each ``--category`` run writes its CZMLs into its own subdir and records itself here, so
    the frontend can offer a selector listing exactly the optimization categories that exist
    (e.g. ADS-B target / runway target / …, with/without constraints). Returns the new total.
    """
    manifest: dict[str, Any] = {"categories": []}
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("categories"), list):
            manifest = loaded
    kept = [c for c in manifest["categories"] if c.get("key") != key]
    kept.append({"key": key, "label": label, "dir": directory, "groups": group_count})
    manifest["categories"] = sorted(kept, key=lambda c: c["key"])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return len(manifest["categories"])


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
    parser.add_argument(
        "--start-visible", action="store_true",
        help="batch mode: emit entities with show=true (default hidden, so the frontend "
             "reveals only the groups it samples from comparison_index.json)",
    )
    parser.add_argument(
        "--category", default=None,
        help="optimization-category key (e.g. asdb / runway / runwayConstrained). When set, "
             "the output-dir is treated as that category's subdir and a categories.json manifest "
             "is written to its parent so the frontend can offer a category selector.",
    )
    parser.add_argument(
        "--category-label", default=None,
        help="display label for --category (defaults to the key)",
    )
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
    index: dict[str, Any] = {
        "epoch": EPOCH.isoformat(),
        "startHidden": not args.start_visible,
        "category": args.category,
        "groups": [],
    }
    for (airport, runway), results in sorted(groups.items()):
        czml, records = build_runway_comparison(
            results, states_dir, adsb_for(airport),
            airport=airport, start_hidden=not args.start_visible,
        )
        out_path = out_dir / f"comparison_{airport}_{runway}.czml"
        out_path.write_text(json.dumps(czml, indent=2), encoding="utf-8")
        for record in records:
            record["czml"] = out_path.name        # which CZML file this group lives in
        index["groups"].extend(records)
        failed = sum(1 for r in records if r["status"] != "solved")
        print(f"✓ {out_path.name}: {len(records)} group(s), {failed} unsolved (red) -> {len(czml)} packets")

    index_path = out_dir / "comparison_index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    rate = summary.get("failure_rate")
    rate_str = f"{rate:.1%}" if isinstance(rate, (int, float)) else "n/a"
    print(f"✓ wrote {len(groups)} runway CZML(s) + index ({len(index['groups'])} groups) "
          f"to {out_dir}; overall failure rate {rate_str}")

    if args.category:
        total = _upsert_category(
            out_dir.parent / "categories.json",
            key=args.category, label=args.category_label or args.category,
            directory=out_dir.name, group_count=len(index["groups"]),
        )
        print(f"✓ registered category {args.category!r} -> {out_dir.parent / 'categories.json'} "
              f"({total} categor{'y' if total == 1 else 'ies'})")


if __name__ == "__main__":
    main()
