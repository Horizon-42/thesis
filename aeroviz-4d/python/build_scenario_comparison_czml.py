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
  With ``--evaluation-report`` (the evaluation package's report JSON), solved flights whose
  final state FAILED the evaluation gates render their reference in **yellow
  (OFF_TARGET_COLOR)** with status ``offTarget``, and the index's ``optimization`` block
  carries the report's batch metrics (successRate / avgStateErrorM / avgTimeS).
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

# Record-filename suffixes. MUST match 4dTrajectory/optimization/evaluation_export.py, which
# is where they are defined and where both writers (the optimizer batch and ts_transformer)
# stamp `flight_key` into the names. Mirrored rather than imported ON PURPOSE: this package is
# standalone frontend tooling with its own pytest rootdir, and evaluation_export pulls in
# aerodynamic_model — importing it would make the CZML builder depend on the whole modeling
# tree for two string constants. _group_key's tests pin the round-trip, so a drift here fails
# loudly rather than silently regrouping flights.
STATES_SUFFIX = "_states.json"
EVAL_SUFFIX = "_eval.json"

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
# Solved but FAILED the evaluation gates ("off target"): the OPTIMIZATION RESULT (simulator)
# path renders bright yellow — it is the trajectory that missed (or truncated at the ground
# guard mid-flight), so the marking must be on IT, not just the reference — and the reference
# renders a darker amber so the pair reads as one flagged group.
OFF_TARGET_COLOR = (255, 205, 40, 235)       # the simulator/result path
OFF_TARGET_REF_COLOR = (150, 118, 25, 200)   # the observed reference (dark amber)
PREDICTION_COLOR = (170, 90, 230, 225)       # learned prediction — "Predicted" (purple)

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

_INITIAL_STATE_KEYS = ("lat", "lon", "alt", "V", "psi", "gamma", "m")


def _initial_state(states: list[dict[str, Any]]) -> dict[str, float] | None:
    """The first sample's kinematic state (used as a group's ``initialState`` in the index).

    Includes ``m`` (the optimizer's aircraft mass, in kg) so the frontend flight list can show
    it; mass is (near-)constant over an approach, so the initial sample is representative.
    """
    if not states:
        return None
    first = states[0]
    return {key: first[key] for key in _INITIAL_STATE_KEYS}


def scenario_initial_map(scenario_paths: list[str | Path]) -> dict[tuple[str, str], dict[str, float]]:
    """Map ``(flightId, runway)`` → the scenario's initial state, from one or more scenario files.

    The ``FlightScenario`` initial state (``V``/``m``/…) is derived from the observed track and the
    resolved aircraft **before** optimization, so it exists for EVERY flight — solved or failed —
    and is the single source the optimizer's own initial state was built from. This lets the flight
    list show V + mass for failed optimizations too, consistent with the solved ones. The initial
    state is target-independent, so multiple scenario files (track-end / threshold) agree; later
    files fill any gaps in earlier ones.
    """
    out: dict[tuple[str, str], dict[str, float]] = {}
    for path in scenario_paths:
        scenarios = json.loads(Path(path).read_text(encoding="utf-8"))
        for scenario in scenarios:
            source = scenario.get("source", {})
            key = (source.get("id"), source.get("runway") or "unknown")
            out.setdefault(key, scenario["initial"])
    return out


# The two states-file schemas this builder understands, each mapping to the entity
# prefixes the frontend derives `kind` from (kindOfEntityId in useComparisonTrajectoryLayer).
#
#   optimizer (4dTrajectory/optimization)     opt- the NLP's plan, sim- the true-dynamics replay
#   predicted (4dTrajectory/ts_transformer)   pred- the learned forecast
#
# A learned predictor has no plan/replay split — it emits one trajectory and no controls —
# so it gets its own kind rather than borrowing `optimizer`, which would make the legend
# ("Optimize states") lie about what is drawn.
_OPTIMIZER_SCHEMA = ("optimizer_states", "simulator_states")
_PREDICTION_SCHEMA = ("predicted_states",)


def states_schema(state_data: dict[str, Any]) -> str:
    """``"optimizer"`` or ``"predicted"``, from which state keys a states file carries."""
    if all(key in state_data for key in _OPTIMIZER_SCHEMA):
        return "optimizer"
    if all(key in state_data for key in _PREDICTION_SCHEMA):
        return "predicted"
    raise KeyError(
        f"states file has neither the optimizer schema {_OPTIMIZER_SCHEMA} nor the "
        f"prediction schema {_PREDICTION_SCHEMA}; got keys {sorted(state_data)}"
    )


def _group_key(result: dict[str, Any]) -> str:
    """The flight identity a comparison group is keyed by.

    Taken from the record filename, whose stem IS ``flight_scenarios.identity.flight_key``
    (``callsign_runway_icao24_landingTime``) — the same identity that keys the train/val/test
    split. ``eval_file`` is the fallback because a FAILED row has no ``states_file`` but
    always has an eval record; both names share the stem, so solved and failed rows of one
    flight group together.

    It is emphatically NOT ``id_runway``: ``id`` is the callsign (a copy of it, despite the
    name) and repeats daily, so the same callsign landing on the same runway on two different
    days collided and one flight silently overwrote the other. Measured on the KRDU harvest,
    ``id_runway`` yields 778 distinct keys for 996 arrivals — 22% of the batch never drawn.
    The raw data carries no unique flight id at all (OpenSky stores state vectors by
    icao24 + time; an "arrival" is a segment this project derives), which is why the
    identity has to include the landing time.
    """
    for name in (result.get("states_file"), result.get("eval_file")):
        if name:
            stem = Path(name).name
            for suffix in (STATES_SUFFIX, EVAL_SUFFIX):
                if stem.endswith(suffix):
                    return stem[: -len(suffix)]
            return Path(stem).stem
    # No record files at all: nothing to draw but the reference. Fall back to the coarse key
    # rather than dropping the row — collisions here cost a duplicate reference, not a flight.
    return f"{result.get('id')}_{result.get('runway') or 'unknown'}"


def _traj_properties(
    group: str, flight_id: str | None, kind: str, runway: str, airport: str, status: str
) -> dict[str, Any]:
    """CZML custom-property bag the frontend uses to group / sample / colour-key entities.

    ``group`` is the per-flight key (unique within the run); ``kind`` is one of
    ``reference`` / ``optimizer`` / ``simulator`` / ``predicted``.
    """
    return {
        "group": group, "flightId": flight_id, "kind": kind,
        "runway": runway, "airport": airport, "status": status,
    }


def _flight_facts(
    initial_state: dict[str, float] | None,
    scenario_initial: dict[str, float] | None,
) -> tuple[float | None, float | None]:
    """``(initialVMps, massKg)`` for a group — the optimizer's own initial state when solved,
    else the scenario's (so failed optimizations still get V + mass)."""
    source = initial_state if initial_state else scenario_initial
    if not source:
        return None, None
    v = source.get("V")
    m = source.get("m")
    return (float(v) if v is not None else None), (float(m) if m is not None else None)


def build_runway_comparison(
    results: list[dict[str, Any]],
    states_dir: str | Path,
    adsb_czml: list[dict[str, Any]],
    *,
    airport: str = "UNK",
    start_hidden: bool = True,
    scenario_initial: dict[tuple[str, str], dict[str, float]] | None = None,
    verdicts: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Combined CZML for one runway **plus** its index records (every flight on one map).

    Each result (a row from ``scenario_optimization``'s ``summary.json``) becomes:
      • solved (+ inside the gates) → reference (white) + optimizer (orange) + simulator (blue);
      • solved but OFF TARGET → the same three paths, reference in the yellow
        OFF_TARGET_COLOR, labelled "(off target)", status ``offTarget``;
      • failed → reference only, in the dark-red FAILED_COLOR, labelled "(unsolved)".

    ``verdicts`` maps an eval-record filename (the summary row's ``eval_file``) to its
    evaluation-report row (``{"solved", "success", "lateral_m", "vertical_m", …}`` — see
    ``evaluation.metrics.evaluate_batch``). A solved flight whose row says ``success: false``
    missed the regulation gates; its final lateral/vertical deviations are copied onto the
    index record (``lateralErrM``/``verticalErrM``). ``None`` (no report) keeps every solved
    flight plain "solved".

    Every entity gets a globally-unique id ``{kind}-{flightId}_{runway}`` (so duplicate
    ``(flightId, runway)`` rows — which collide on the *same* states file — can't produce
    colliding CZML ids) and a ``properties`` bag (``group``/``kind``/…). Entities are
    ``show=False`` when ``start_hidden`` so the frontend renders only the groups it samples.

    Returns ``(czml_packets, index_records)``. Each index record describes one group:
    its id, flight id, runway, status, initial state, and the entity ids that belong to it.
    """
    states_dir = Path(states_dir)
    show = not start_hidden

    # Collapse to one row per group, preferring a solved row. A flight can appear in the
    # summary as both a failed attempt and a solved one; the solved result is the one to show,
    # regardless of row order. Both rows carry the same record filenames, so they land in the
    # same group and the dedup still does its job.
    best: dict[str, dict[str, Any]] = {}
    for result in results:
        group = _group_key(result)
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
            schema = states_schema(state_data)
            if schema == "optimizer":
                optimizer_states = state_data["optimizer_states"]
                simulator_states = state_data["simulator_states"]
                predicted_states = None
            else:
                optimizer_states = simulator_states = None
                predicted_states = state_data["predicted_states"]

            # The evaluation verdict for this flight (joined by the summary row's eval_file):
            # solved-but-outside-the-gates renders as "off target" (yellow reference).
            verdict = (verdicts or {}).get(result.get("eval_file") or "")
            off_target = (
                verdict is not None and verdict.get("solved") and not verdict.get("success")
            )
            status = "offTarget" if off_target else "solved"
            # The off-target COLOURING exists to make the few results that missed their
            # target stand out among mostly-successful ones. For a learned prediction that
            # is backwards: a forecast essentially never lands inside the 106.75 m lateral
            # gate (that limit is FAA containment for a planned/flown approach, not a
            # forecast-accuracy target), so ~100% of a prediction batch would go yellow —
            # the marking would carry no information and would erase the kind's own colour.
            # The `status` property stays accurate either way, and the per-flight deviation
            # is still surfaced by the index's lateralErrM/verticalErrM and the evaluation
            # report, which say far more than a binary colour.
            mark_off_target = off_target and schema == "optimizer"
            ref_color = OFF_TARGET_REF_COLOR if mark_off_target else REFERENCE_COLOR
            sim_color = OFF_TARGET_COLOR if mark_off_target else SIMULATOR_COLOR
            ref_name = f"Ref {flight_id} (off target)" if mark_off_target else f"Ref {flight_id}"

            reference = _reference_entity_from_adsb(
                adsb_czml, flight_id, ref_color,
                entity_id=f"ref-{group}", name=ref_name,
                properties=_traj_properties(group, flight_id, "reference", runway, airport, status),
                show=show,
            )
            if reference is not None:
                entities.append(reference)
                entity_ids.append(f"ref-{group}")
            if schema == "optimizer":
                entities.append(_build_trajectory_entity(
                    f"opt-{group}", f"Opt {flight_id}", optimizer_states, OPTIMIZER_COLOR,
                    properties=_traj_properties(group, flight_id, "optimizer", runway, airport, status),
                    show=show))
                entity_ids.append(f"opt-{group}")
                entities.append(_build_trajectory_entity(
                    f"sim-{group}",
                    f"Sim {flight_id} (off target)" if mark_off_target else f"Sim {flight_id}",
                    simulator_states, sim_color,
                    properties=_traj_properties(group, flight_id, "simulator", runway, airport, status),
                    show=show))
                entity_ids.append(f"sim-{group}")
            else:
                # One path, not two: a learned predictor has no plan-vs-replay split. The
                # off-target colour still applies — it marks the trajectory that missed the
                # gates, and here that is the prediction itself.
                entities.append(_build_trajectory_entity(
                    f"pred-{group}", f"Pred {flight_id}",
                    predicted_states, PREDICTION_COLOR,
                    properties=_traj_properties(group, flight_id, "predicted", runway, airport, status),
                    show=show))
                entity_ids.append(f"pred-{group}")

            initial_state = _initial_state(optimizer_states or simulator_states or predicted_states)
            scen = (scenario_initial or {}).get((flight_id, runway))
            initial_v, mass_kg = _flight_facts(initial_state, scen)
            record = {
                "group": group, "flightId": flight_id, "runway": runway, "airport": airport,
                "status": status, "finalTimeS": float(state_data.get("final_time_s", 0.0)),
                "initialState": initial_state, "initialVMps": initial_v, "massKg": mass_kg,
                "entities": entity_ids,
            }
            if verdict is not None:
                # Final-state deviations from the evaluation (shown by the flight list).
                record["lateralErrM"] = verdict.get("lateral_m")
                record["verticalErrM"] = verdict.get("vertical_m")
            index_records.append(record)
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
            # A failed optimization has no states, but the scenario still carries the resolved
            # aircraft mass + observed V — surface them so the flight list shows them (and flags red).
            scen = (scenario_initial or {}).get((flight_id, runway))
            initial_v, mass_kg = _flight_facts(None, scen)
            index_records.append({
                "group": group, "flightId": flight_id, "runway": runway, "airport": airport,
                "status": "failed", "finalTimeS": None, "initialState": None,
                "initialVMps": initial_v, "massKg": mass_kg,
                "entities": entity_ids,
            })

    # Span the clock over the LONGEST trajectory actually in the file — references included.
    # (Deriving it only from solved opt/sim states collapsed the clock to ~1 s on runways with
    #  no solved scenarios, freezing every reference track at its start point.)
    max_t = max((_entity_last_time(e) for e in entities), default=0.0)
    end_dt = EPOCH.fromtimestamp(EPOCH.timestamp() + max(max_t, 1.0), tz=timezone.utc)
    document = build_document_packet(EPOCH, end_dt, multiplier=30)
    return [document] + entities, index_records


def clear_stale_outputs(out_dir: Path, *, keep_report: bool) -> list[Path]:
    """Delete a previous build's files from the category dir before writing this one.

    A runway present in an EARLIER batch but absent from this summary would leave its
    stale ``comparison_*.czml`` behind (dormant — the rewritten index stops referencing
    it — but unbounded growth). And when this run publishes NO evaluation report
    (``keep_report=False``), a previously published ``evaluation_report.json`` must go
    too: the frontend's Details window fetches it by path and would show outdated
    metrics next to the fresh index. Returns the deleted paths.
    """
    stale = sorted(out_dir.glob("comparison_*.czml"))
    if not keep_report:
        stale += [p for p in [out_dir / "evaluation_report.json"] if p.exists()]
    for path in stale:
        path.unlink()
    if stale:
        print(f"… cleared {len(stale)} file(s) from a previous build in {out_dir}")
    return stale


def publish_evaluation_report(evaluation_report: dict[str, Any], out_dir: Path) -> Path:
    """Publish the evaluation report VERBATIM into the category dir the frontend serves.

    The frontend's detailed evaluation view visualizes this copy directly — every number
    in it was computed by ``python -m evaluation`` (the one backend exit); the frontend
    never recomputes a metric.
    """
    path = out_dir / "evaluation_report.json"
    path.write_text(json.dumps(evaluation_report, indent=2), encoding="utf-8")
    return path


def load_verdicts(evaluation_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-trajectory verdict rows of an evaluation report, keyed by eval-record filename.

    The key is the report row's ``file`` (``<identity>_eval.json``), which the optimization
    summary rows carry as ``eval_file`` — the unambiguous join between the two artifacts.
    """
    return {
        row["file"]: row
        for row in evaluation_report.get("trajectories", [])
        if row.get("file")
    }


def optimization_stats(
    summary: dict[str, Any], evaluation_report: dict[str, Any] | None
) -> dict[str, Any]:
    """The index's ``optimization`` block: solve stats from the run's summary.json, and —
    when an evaluation report is given — the evaluation's batch metrics (success rate,
    mean final lateral deviation, mean flight time). Nothing is recomputed here."""
    solved = summary.get("solved")
    total = summary.get("total")
    stats: dict[str, Any] = {
        "total": total,
        "solved": solved,
        "failed": summary.get("failed"),
        "solveRate": (solved / total) if isinstance(solved, (int, float)) and total else None,
    }
    if evaluation_report is not None:
        lateral = evaluation_report.get("lateral_m") or {}
        times = evaluation_report.get("final_time_s") or {}
        stats.update({
            "successful": evaluation_report.get("successful"),
            "successRate": evaluation_report.get("success_rate"),
            "avgStateErrorM": lateral.get("mean"),
            "avgTimeS": times.get("mean"),
        })
    return stats


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
    manifest_path: Path, *, key: str, label: str, directory: str, group_count: int,
    constrained: bool,
) -> int:
    """Add/replace one category in the shared ``categories.json`` manifest.

    Each ``--category`` run writes its CZMLs into its own subdir and records itself here, so
    the frontend can offer a selector listing exactly the optimization categories that exist
    (e.g. ADS-B target / runway target / …, with/without constraints). ``constrained`` is
    stamped as an explicit manifest field — the frontend keys constraint-scoped behavior
    (the Observe procedure auto-open) off it, never off the key/dir spelling. Returns the
    new total.
    """
    manifest: dict[str, Any] = {"categories": []}
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("categories"), list):
            manifest = loaded
    kept = [c for c in manifest["categories"] if c.get("key") != key]
    kept.append({"key": key, "label": label, "dir": directory, "groups": group_count,
                 "constrained": constrained})
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
        help="optimization-category key (e.g. asdb / runway / runway_cons). When set, "
             "the output-dir is treated as that category's subdir and a categories.json manifest "
             "is written to its parent so the frontend can offer a category selector.",
    )
    parser.add_argument(
        "--category-label", default=None,
        help="display label for --category (defaults to the key)",
    )
    parser.add_argument(
        "--constrained", action="store_true",
        help="mark --category as a constrained-optimization category (its solves enforce "
             "the runway's RNAV procedure); stamped into the manifest as an explicit field "
             "so the frontend never derives it from the key/dir spelling",
    )
    parser.add_argument(
        "--scenarios", default=None,
        help="batch mode: comma-separated *_scenarios.json path(s). Their per-flight initial "
             "state (V + mass) is added to every index record — including FAILED optimizations "
             "(which have no states), so the flight list shows V + mass for those too.",
    )
    parser.add_argument(
        "--evaluation-report", default=None,
        help="batch mode: the evaluation package's report JSON for this run. Solved flights "
             "whose final state failed the gates render with a YELLOW reference (status "
             "offTarget), and the index's optimization block carries the batch metrics "
             "(successRate / avgStateErrorM / avgTimeS).",
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
    clear_stale_outputs(out_dir, keep_report=args.evaluation_report is not None)

    adsb_cache: dict[str, list[dict[str, Any]]] = {}

    def adsb_for(airport: str) -> list[dict[str, Any]]:
        if airport not in adsb_cache:
            adsb_cache[airport] = _load_adsb(airport, None)
        return adsb_cache[airport]

    scenario_initial = (
        scenario_initial_map([p.strip() for p in args.scenarios.split(",") if p.strip()])
        if args.scenarios else None
    )
    evaluation_report = (
        json.loads(Path(args.evaluation_report).read_text(encoding="utf-8"))
        if args.evaluation_report else None
    )
    verdicts = load_verdicts(evaluation_report) if evaluation_report is not None else None

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
            scenario_initial=scenario_initial, verdicts=verdicts,
        )
        out_path = out_dir / f"comparison_{airport}_{runway}.czml"
        out_path.write_text(json.dumps(czml, indent=2), encoding="utf-8")
        for record in records:
            record["czml"] = out_path.name        # which CZML file this group lives in
        index["groups"].extend(records)
        failed = sum(1 for r in records if r["status"] == "failed")
        off_target = sum(1 for r in records if r["status"] == "offTarget")
        print(f"✓ {out_path.name}: {len(records)} group(s), {failed} unsolved (red), "
              f"{off_target} off-target (yellow) -> {len(czml)} packets")

    # Solve stats from the run's summary.json + (when given) the evaluation report's batch
    # metrics — nothing recomputed here.
    index["optimization"] = optimization_stats(summary, evaluation_report)
    if evaluation_report is not None:
        report_path = publish_evaluation_report(evaluation_report, out_dir)
        print(f"✓ published evaluation report -> {report_path}")

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
            constrained=args.constrained,
        )
        print(f"✓ registered category {args.category!r} -> {out_dir.parent / 'categories.json'} "
              f"({total} categor{'y' if total == 1 else 'ies'})")


if __name__ == "__main__":
    main()
