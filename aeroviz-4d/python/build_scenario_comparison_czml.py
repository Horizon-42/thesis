"""Build comparison CZML(s) for optimized scenarios.

Two modes:

* **Single** (``--state-file``) — one ``*_states.json`` from
  ``4dTrajectory/optimization/scenario_optimization.py`` → one CZML with three coloured,
  time-dynamic trajectories:
    • reference  (white)  — the observed ADS-B track, copied from the airport's trajectories.czml
    • optimizer  (orange) — the optimizer's plan        (optimizer_states)  ["Optimize states"]
    • simulator  (blue)   — the real forward rollout    (simulator_states)  ["Optimize results"]

* **Batch** (``--summary``) — the run's ``summary.json`` → **one result CZML per runway**
  plus a single ``comparison_index.json``. Optimizer/simulator/prediction paths live in
  those files; reference ids point to the airport's one canonical ``trajectories.czml``
  datasource, so the observed positions are not copied into every category. Solved flights
  get result paths plus that logical reference; unsolved flights get the reference only.
  With ``--evaluation-report`` (the evaluation package's report JSON), solved flights whose
  final state FAILED the evaluation gates render their reference in **yellow
  (OFF_TARGET_COLOR)** with status ``offTarget``, and the index's ``optimization`` block
  carries the report's batch metrics (successRate / avgStateErrorM / avgTimeS).
  A ts_transformer summary additionally publishes its existing ADE/FDE aggregates in
  the index's ``prediction`` block; the frontend displays those values without recomputing
  model accuracy.
  Each entity has a globally-unique id ``{kind}-{flightId}_{runway}`` and a ``properties`` bag
  (``group``/``flightId``/``kind``/``runway``/``airport``/``status``) so the frontend can
  group and **randomly sample** trajectories. Entities default to ``show=false`` (override with
  ``--start-visible``); the frontend reads ``comparison_index.json`` (one record per group, with
  its ``initialState`` and the CZML file + entity ids it owns), samples a subset, and reveals
  only those. The reference is resolved by flight key in the canonical observed datasource.
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATASET_SPLITS = ("train", "val", "test")

from data_layout import airport_data_path
from flight_identity import flight_key
from generate_czml import build_document_packet, build_position_property

# Record-filename suffixes. MUST match 4dTrajectory/optimization/evaluation_export.py, which
# is where they are defined and where both writers (the optimizer batch and ts_transformer)
# stamp `flight_key` into the names. Mirrored rather than imported ON PURPOSE: this package is
# standalone frontend tooling with its own pytest rootdir, and evaluation_export pulls in
# aerodynamic_model — importing it would make the CZML builder depend on the whole modeling
# tree for two string constants. _group_key's tests pin the round-trip, so a drift here fails
# loudly rather than silently regrouping flights. (flight_key itself is mirrored the same
# way, in this package's flight_identity module — see its docstring for the shared pin.)
STATES_SUFFIX = "_states.json"
EVAL_SUFFIX = "_eval.json"

# A fixed display epoch (state times are offsets in seconds from it), matching the
# convention generate_czml uses. The relative motion is what matters, not the wall clock.
EPOCH = datetime(2026, 4, 1, 8, 0, 0, tzinfo=timezone.utc)


def _write_json_atomic(path: Path, value: Any, *, pretty: bool = False) -> None:
    """Stream JSON to a sibling temp file, then publish it as one atomic artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(
                value,
                output,
                indent=2 if pretty else None,
                separators=None if pretty else (",", ":"),
            )
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise

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
# The lookback window the predictor was CONDITIONED on: observed samples, so the same hue as
# the forecast they lead into but faded, reading as one continuous track that goes from "given"
# to "predicted". Without it the purple line starts in mid-air at the anchor with nothing
# joining it to the beginning of the approach.
LOOKBACK_COLOR = (170, 90, 230, 85)          # model input window — "Lookback" (faded purple)

# Trailing-tail length (seconds) for the optimizer/simulator paths: the tail fades behind the
# moving aircraft as playback advances, so the head (current position) is distinguishable from the
# tail. Matches generate_czml.py's observed-track trail (300 s) — which the reference inherits — so
# all three comparison trajectories fade identically.
_TRAIL_TIME_S = 300


# ── state sequence -> CZML geometry ───────────────────────────────────────────

def _states_to_waypoints(
    states: list[dict[str, Any]], hae_minus_msl_m: float
) -> list[tuple[float, float, float, float]]:
    """Convert a list of state dicts (``{t, lat, lon, alt, …}``) to CZML waypoints.

    ``build_position_property`` wants ``(offset_sec, lon, lat, alt_m)`` tuples — note the
    order: time, then **lon before lat** (GeoJSON/CZML convention), then metres.

    Record ``alt`` is MSL (the modeling plane's datum, converted on ingest); Cesium reads
    CZML altitude as height above the WGS84 ellipsoid, so the undulation is added back
    HERE — the single point every record-derived entity (opt-/sim-/pred-) flows through.
    The observed reference bypasses this function (deep-copied from ``trajectories.czml``,
    already ellipsoidal). Records are MSL by contract: pre-datum-fix (HAE-era) artifacts
    are discarded wholesale, never fed back in.
    """
    if not states:
        return []
    return [
        (s["t"], s["lon"], s["lat"], float(s["alt"]) + hae_minus_msl_m)
        for s in states
    ]


def _record_offset(record: dict[str, Any]) -> float:
    source = record.get("source") or {}
    value = source.get("hae_minus_msl_m")
    if value is None:
        raise ValueError(
            "record lacks source.hae_minus_msl_m; regenerate legacy vertical-datum artifact"
        )
    return float(value)


def _time_shifted(states: list[dict[str, Any]], offset_s: float) -> list[dict[str, Any]]:
    """The same states with every ``t`` moved by ``offset_s`` seconds."""
    return [{**state, "t": state["t"] + offset_s} for state in states]


def _lookback_states(observed_states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The observed samples the predictor was shown: everything up to and including the anchor.

    A prediction record rebases time so ``t = 0`` IS the anchor, which makes the lookback the
    negative-``t`` half of ``observed_states``. The anchor sample itself (``t == 0``) belongs to
    both halves — it is literally the same state object in the record — so keeping it here joins
    the faded input segment to the prediction with no gap.
    """
    return [state for state in observed_states if state["t"] <= 0.0]


# ── copy the matching reference flight from the ADS-B CZML ─────────────────────

def _reference_entity_from_adsb(
    adsb_czml: list[dict[str, Any]],
    flight_identity: str | None,
    color_rgba: tuple[int, int, int, int],
    *,
    entity_id: str = "scenario-reference",
    name: str | None = None,
    properties: dict[str, Any] | None = None,
    show: bool = True,
) -> dict[str, Any] | None:
    """Find the observed flight in ``adsb_czml`` and copy it as the reference trajectory.

    ``adsb_czml`` is the loaded ``trajectories.czml`` (element 0 is the ``"document"``
    packet; the rest are flight entities whose ``id`` IS the flight_key
    ``id_runway_icao24_landingTime``). ``flight_identity`` must be that same key — the
    comparison group key / record-filename stem, NOT the bare callsign: a callsign lookup
    resolved duplicated callsigns to whichever namesake came first, silently drawing the
    wrong flight as the white reference line. Find the matching entity, **deep-copy** it
    (so we don't mutate the source), recolour its ``path`` material to ``color_rgba``,
    and re-id/-name it (``entity_id`` / ``name``) so several references can coexist in
    one combined CZML.

    ``properties`` attaches a CZML custom-property bag (so the frontend can group/sample)
    and ``show`` sets the entity-level visibility (``False`` ⇒ hidden until revealed).
    """
    for packet in adsb_czml:
        if packet.get("id") == flight_identity:
            entity = copy.deepcopy(packet)
            entity["id"] = entity_id
            entity["name"] = name or f"Reference {packet.get('name') or flight_identity}"
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
    hae_minus_msl_m: float,
    properties: dict[str, Any] | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """A coloured, time-dynamic path entity from a state sequence.

    ``properties`` attaches a CZML custom-property bag (so the frontend can group/sample
    by ``group``/``kind``/…) and ``show`` sets entity-level visibility (``False`` ⇒ the
    whole entity is hidden until the frontend reveals it).
    """
    waypoints = _states_to_waypoints(states, hae_minus_msl_m)
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
    # The observed entity is looked up by flight_key (the ADS-B CZML's entity id), derived
    # from the record's own source dict — same fields both writers stamped.
    identity = flight_key(state_data.get("source", {}), 0)
    offset = _record_offset(state_data)

    reference = _reference_entity_from_adsb(adsb_czml, identity, REFERENCE_COLOR)
    optimizer = _build_trajectory_entity(
        "scenario-optimizer", "Optimizer", optimizer_states, OPTIMIZER_COLOR,
        hae_minus_msl_m=offset,
    )
    simulator = _build_trajectory_entity(
        "scenario-simulator", "Simulator", simulator_states, SIMULATOR_COLOR,
        hae_minus_msl_m=offset,
    )
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


def scenario_initial_map(scenario_paths: list[str | Path]) -> dict[str, dict[str, float]]:
    """Map ``flight_key`` (= the comparison group key) → the scenario's initial state.

    The ``FlightScenario`` initial state (``V``/``m``/…) is derived from the observed track and the
    resolved aircraft **before** optimization, so it exists for EVERY flight — solved or failed —
    and is the single source the optimizer's own initial state was built from. This lets the flight
    list show V + mass for failed optimizations too, consistent with the solved ones. The initial
    state is target-independent, so multiple scenario files (track-end / threshold) agree; later
    files fill any gaps in earlier ones.

    Keyed by ``flight_key`` — the same identity the record-filename stems carry — NOT
    ``(id, runway)``: the same callsign lands on the same runway on different days, and the
    tuple key served one flight's V/mass for every namesake.
    """
    out: dict[str, dict[str, float]] = {}
    for path in scenario_paths:
        scenarios = json.loads(Path(path).read_text(encoding="utf-8"))
        for index, scenario in enumerate(scenarios):
            key = flight_key(scenario.get("source", {}), index)
            out.setdefault(key, scenario["initial"])
    return out


# The two states-file schemas this builder understands, each mapping to the entity
# prefixes the frontend derives `kind` from (kindOfEntityId in useComparisonTrajectoryLayer).
#
#   optimizer (4dTrajectory/optimization)     opt- the NLP's plan, sim- the true-dynamics replay
#   predicted (4dTrajectory/ts_transformer)   pred- the learned forecast, look- its input window
#
# A learned predictor has no plan/replay split — it emits one trajectory and no controls —
# so it gets its own kind rather than borrowing `optimizer`, which would make the legend
# ("Optimize states") lie about what is drawn.
#
# `observed_states` is part of the prediction schema, not an optional extra: it is the whole
# observed track (negative t before the anchor) and it is the ONLY source for the lookback the
# model was conditioned on. Requiring it here fails loudly on a record that cannot be drawn
# completely, instead of silently emitting a forecast that begins in mid-air.
_OPTIMIZER_SCHEMA = ("optimizer_states", "simulator_states")
_PREDICTION_SCHEMA = ("predicted_states", "observed_states")


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
    # No record files at all: nothing to draw but the reference. Reconstruct the identity
    # from the row itself — summary rows carry the same id/runway/icao24/landing_time_utc
    # fields flight_key reads — so this matches the stem the files would have had (and the
    # observed layer's entity id, which the reference lookup needs).
    return flight_key(result, 0)


def _traj_properties(
    group: str, flight_id: str | None, kind: str, runway: str, airport: str, status: str
) -> dict[str, Any]:
    """CZML custom-property bag the frontend uses to group / sample / colour-key entities.

    ``group`` is the per-flight key (unique within the run); ``kind`` is one of
    ``reference`` / ``optimizer`` / ``simulator`` / ``predicted`` / ``lookback``.
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
    include_reference_entities: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Combined CZML for one runway **plus** its index records (every flight on one map).

    Each result (a row from ``scenario_optimization``'s ``summary.json``) becomes:
      • solved (+ inside the gates) → reference (white) + optimizer (orange) + simulator (blue);
      • solved but OFF TARGET → the same three paths, reference in the yellow
        OFF_TARGET_COLOR, labelled "(off target)", status ``offTarget``;
      • failed → reference only, in the dark-red FAILED_COLOR, labelled "(unsolved)".

    ``verdicts`` maps an eval-record filename (the summary row's ``eval_file``) to its
    evaluation-report row (``{"solved", "verdict", "lateral_m", "vertical_m", …}`` — see
    ``evaluation.metrics.evaluate_batch``). Only an explicit ``verdict: "fail"`` becomes
    off-target; ``indeterminate`` remains distinct. Final deviations are copied onto the
    index record (``lateralErrM``/``verticalErrM``). ``None`` (no report) keeps every solved
    flight plain "solved".

    Every entity gets a globally-unique id ``{kind}-{group}`` where ``group`` is the
    flight_key stem of the record filename, and a ``properties`` bag (``group``/``kind``/…).
    The reference is looked up in ``adsb_czml`` by that same ``group`` — the observed
    layer's entity ids ARE flight_keys — so a duplicated callsign can no longer resolve
    to the wrong namesake's track. Entities are ``show=False`` when ``start_hidden`` so
    the frontend renders only the groups it samples.

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
                predicted_states = lookback_states = None
            else:
                optimizer_states = simulator_states = None
                # A prediction record rebases its own time so t=0 is the ANCHOR — the last
                # observed sample the model was shown, typically `seq_len` samples into the
                # approach. The reference copied from the ADS-B CZML still starts at t=0 = the
                # start of the track, so writing the prediction's times through unshifted drew
                # it `anchorTimeS` seconds EARLY on the shared clock: measured on KRDU 05L, the
                # forecast's first sample (bit-identical to the reference's t=118 s sample) was
                # plotted at t=0, 12.0 km from where the reference then was. Shifting both the
                # forecast and its lookback back onto the observed timeline puts every entity
                # of a group on one clock again.
                anchor_time_s = float(state_data["source"]["anchorTimeS"])
                predicted_states = _time_shifted(state_data["predicted_states"], anchor_time_s)
                lookback_states = _time_shifted(
                    _lookback_states(state_data["observed_states"]), anchor_time_s)

            # The evaluation verdict for this flight (joined by the summary row's eval_file):
            # solved-but-outside-the-gates renders as "off target" (yellow reference).
            verdict = (verdicts or {}).get(result.get("eval_file") or "")
            terminal_verdict = verdict.get("verdict") if verdict is not None else None
            off_target = verdict is not None and verdict.get("solved") \
                and terminal_verdict == "fail"
            indeterminate = verdict is not None and verdict.get("solved") \
                and terminal_verdict == "indeterminate"
            status = "offTarget" if off_target else (
                "indeterminate" if indeterminate else "solved"
            )
            offset = _record_offset(state_data)
            # Terminal-verdict colouring is reserved for optimizer results. Learned
            # predictions keep their kind colour so prediction error and operational
            # terminal assessment remain distinct visual concepts.
            # The `status` property stays accurate either way, and the per-flight deviation
            # is still surfaced by the index's lateralErrM/verticalErrM and the evaluation
            # report, which say far more than a binary colour.
            mark_off_target = off_target and schema == "optimizer"
            ref_color = OFF_TARGET_REF_COLOR if mark_off_target else REFERENCE_COLOR
            sim_color = OFF_TARGET_COLOR if mark_off_target else SIMULATOR_COLOR
            ref_name = f"Ref {flight_id} (off target)" if mark_off_target else f"Ref {flight_id}"

            if include_reference_entities:
                reference = _reference_entity_from_adsb(
                    adsb_czml, group, ref_color,
                    entity_id=f"ref-{group}", name=ref_name,
                    properties=_traj_properties(
                        group, flight_id, "reference", runway, airport, status
                    ),
                    show=show,
                )
                if reference is not None:
                    entities.append(reference)
                    entity_ids.append(f"ref-{group}")
            else:
                # Logical entity id: the frontend resolves this to the same flight in
                # the canonical observed datasource instead of storing its CZML again.
                entity_ids.append(f"ref-{group}")
            if schema == "optimizer":
                entities.append(_build_trajectory_entity(
                    f"opt-{group}", f"Opt {flight_id}", optimizer_states, OPTIMIZER_COLOR,
                    hae_minus_msl_m=offset,
                    properties=_traj_properties(group, flight_id, "optimizer", runway, airport, status),
                    show=show))
                entity_ids.append(f"opt-{group}")
                entities.append(_build_trajectory_entity(
                    f"sim-{group}",
                    f"Sim {flight_id} (off target)" if mark_off_target else f"Sim {flight_id}",
                    simulator_states, sim_color,
                    hae_minus_msl_m=offset,
                    properties=_traj_properties(group, flight_id, "simulator", runway, airport, status),
                    show=show))
                entity_ids.append(f"sim-{group}")
            else:
                # Two entities for one continuous track: the faded lookback the model was
                # conditioned on, then the forecast it produced. Splitting them (rather than
                # concatenating into one path) is what lets the input segment carry its own,
                # lower-alpha colour — the point of drawing it at all is that a viewer can see
                # where "given" ends and "predicted" begins.
                entities.append(_build_trajectory_entity(
                    f"look-{group}", f"Lookback {flight_id}",
                    lookback_states, LOOKBACK_COLOR,
                    hae_minus_msl_m=offset,
                    properties=_traj_properties(group, flight_id, "lookback", runway, airport, status),
                    show=show))
                entity_ids.append(f"look-{group}")
                # One path, not two: a learned predictor has no plan-vs-replay split. The
                # off-target colour still applies — it marks the trajectory that missed the
                # gates, and here that is the prediction itself.
                entities.append(_build_trajectory_entity(
                    f"pred-{group}", f"Pred {flight_id}",
                    predicted_states, PREDICTION_COLOR,
                    hae_minus_msl_m=offset,
                    properties=_traj_properties(group, flight_id, "predicted", runway, airport, status),
                    show=show))
                entity_ids.append(f"pred-{group}")

            initial_state = _initial_state(optimizer_states or simulator_states or predicted_states)
            scen = (scenario_initial or {}).get(group)
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
                record["terminalVerdict"] = terminal_verdict
            index_records.append(record)
        else:
            if include_reference_entities:
                reference = _reference_entity_from_adsb(
                    adsb_czml, group, FAILED_COLOR,
                    entity_id=f"ref-{group}", name=f"Ref {flight_id} (unsolved)",
                    properties=_traj_properties(
                        group, flight_id, "reference", runway, airport, "failed"
                    ),
                    show=show,
                )
                if reference is not None:
                    entities.append(reference)
                    entity_ids.append(f"ref-{group}")
            else:
                entity_ids.append(f"ref-{group}")
            # A failed optimization has no states, but the scenario still carries the resolved
            # aircraft mass + observed V — surface them so the flight list shows them (and flags red).
            scen = (scenario_initial or {}).get(group)
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


def prune_unreferenced_outputs(out_dir: Path, keep_names: set[str]) -> list[Path]:
    """Delete old generations only after the new index has committed.

    Physical batch artifacts are immutable and generation-named. The index is the single
    commit point: until its atomic replace succeeds, the previous index and every file it
    references remain untouched. Once committed, files outside ``keep_names`` are dormant
    and may be removed without exposing a partial batch.
    """
    stale = [
        path
        for pattern in ("comparison_*.czml", "evaluation_report*.json")
        for path in sorted(out_dir.glob(pattern))
        if path.name not in keep_names
    ]
    for path in stale:
        path.unlink()
    if stale:
        print(f"… cleared {len(stale)} file(s) from a previous build in {out_dir}")
    return stale


def publish_evaluation_report(
    evaluation_report: dict[str, Any],
    out_dir: Path,
    *,
    filename: str,
) -> Path:
    """Publish the evaluation report VERBATIM into the category dir the frontend serves.

    The frontend's detailed evaluation view visualizes this copy directly — every number
    in it was computed by ``python -m evaluation`` (the one backend exit); the frontend
    never recomputes a metric.
    """
    path = out_dir / filename
    _write_json_atomic(path, evaluation_report, pretty=True)
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


def prediction_accuracy_stats(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Frontend accuracy/clock/kinematic summary for a ts_transformer batch.

    The predictor already computes these aggregates against the observed overlap and writes
    them to ``summary.json.accuracy``. Publication only changes field casing; it must never
    recompute trajectory accuracy from rendered CZML.
    """
    mode = summary.get("mode")
    if not isinstance(mode, str) or not mode.startswith("tsTransformer:"):
        return None

    accuracy = summary.get("accuracy")
    if not isinstance(accuracy, dict):
        raise ValueError("ts_transformer summary is missing its accuracy block")

    def spread(source_key: str) -> dict[str, float] | None:
        source = accuracy.get(source_key)
        # `accuracy_block` omits error spreads when no forecast overlaps its observed
        # reference. That is a valid empty result; the UI renders unavailable metrics.
        if source is None:
            return None
        if not isinstance(source, dict):
            raise ValueError(
                f"ts_transformer summary accuracy.{source_key} is malformed"
            )
        mean_value = source.get("mean")
        p95_value = source.get("p95")
        if not isinstance(mean_value, (int, float)) or not isinstance(
            p95_value, (int, float)
        ):
            raise ValueError(
                f"ts_transformer summary accuracy.{source_key} requires mean and p95"
            )
        result = {"mean": float(mean_value), "p95": float(p95_value)}
        for source_name, output_name in (("median", "median"), ("max", "max")):
            value = source.get(source_name)
            if isinstance(value, (int, float)):
                result[output_name] = float(value)
        return result

    def final_time() -> dict[str, float] | None:
        source = accuracy.get("final_time_s")
        if source is None:
            return None
        if not isinstance(source, dict):
            raise ValueError("ts_transformer summary accuracy.final_time_s is malformed")
        mapping = (("mae", "mae"), ("p95_abs", "p95Abs"),
                   ("mean_signed", "meanSigned"))
        result = {
            output_name: float(source[source_name])
            for source_name, output_name in mapping
            if isinstance(source.get(source_name), (int, float))
        }
        return result or None

    raw_names = {
        "position_velocity_rmse_mps": "positionVelocityRmseMps",
        "heading_consistency_p95_deg": "headingConsistencyP95Deg",
        "turn_rate_p95_deg_s": "turnRateP95DegS",
        "acceleration_p95_mps2": "accelerationP95Mps2",
        "jerk_p95_mps3": "jerkP95Mps3",
    }

    def raw_role(role: str) -> dict[str, dict[str, float]] | None:
        raw = accuracy.get("raw_kinematics")
        if not isinstance(raw, dict):
            return None
        source = raw.get(role)
        if not isinstance(source, dict):
            return None
        result: dict[str, dict[str, float]] = {}
        for source_name, output_name in raw_names.items():
            values = source.get(source_name)
            if not isinstance(values, dict):
                continue
            converted = {
                key: float(values[key])
                for key in ("median", "mean", "p95", "max")
                if isinstance(values.get(key), (int, float))
            }
            if converted:
                result[output_name] = converted
        return result or None

    def raw_delta() -> dict[str, float] | None:
        raw = accuracy.get("raw_kinematics")
        source = raw.get("delta") if isinstance(raw, dict) else None
        if not isinstance(source, dict):
            return None
        result = {
            output_name: float(source[source_name])
            for source_name, output_name in raw_names.items()
            if isinstance(source.get(source_name), (int, float))
        }
        return result or None

    return {
        "flights": accuracy.get("flights"),
        "flightsWithoutOverlap": accuracy.get("flights_without_overlap"),
        "finalTimeS": final_time(),
        "adeM": spread("ade_m"),
        "fdeM": spread("fde_m"),
        "crossTrackP95M": spread("cross_track_p95_m"),
        "altitudeP95M": spread("altitude_p95_m"),
        "rawKinematics": {
            "predicted": raw_role("predicted"),
            "observedBaseline": raw_role("observed_baseline"),
            "delta": raw_delta(),
        },
    }


def evaluation_batch_stats(report: dict[str, Any]) -> dict[str, Any]:
    """Copy every batch-level evaluation statistic into the committed index.

    The full, per-flight report remains an immutable sibling artifact.  This block deliberately
    excludes ``trajectories`` so the Observe summary can render all aggregate statistics from the
    index it already fetches without loading the details payload.
    """
    field_names = (
        "schema_version",
        "total",
        "measured",
        "solved",
        "solve_rate",
        "successful",
        "success_rate",
        "failed",
        "indeterminate",
        "verdict_counts",
        "lateral_m",
        "vertical_m",
        "final_time_s",
        "observed",
    )
    return {
        {
            "solve_rate": "solveRate",
            "success_rate": "successRate",
            "schema_version": "schemaVersion",
            "verdict_counts": "verdictCounts",
            "lateral_m": "lateralM",
            "vertical_m": "verticalM",
            "final_time_s": "finalTimeS",
        }.get(name, name): report[name]
        for name in field_names
        if name in report
    }


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


def publish_comparison_batch(
    *,
    summary: dict[str, Any],
    states_dir: Path,
    out_dir: Path,
    airport: str,
    category: str | None,
    start_hidden: bool,
    scenario_initial: dict[str, dict[str, Any]] | None,
    evaluation_report: dict[str, Any] | None,
    generation: str | None = None,
) -> dict[str, Any]:
    """Build and atomically publish one complete comparison generation.

    Every CZML and the report receive an immutable generation suffix. They are
    written first; ``comparison_index.json`` is atomically replaced last and is therefore
    the batch commit point. A failed build removes only its unpublished generation. Old
    files are pruned only after commit, so a reader can never observe an old index whose
    files were pre-deleted or an index that mixes two generations.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    generation = generation or uuid.uuid4().hex[:16]
    safe_generation_chars = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    )
    if not generation or any(ch not in safe_generation_chars for ch in generation):
        raise ValueError(f"invalid comparison generation {generation!r}")
    if evaluation_report is None:
        raise ValueError(
            "comparison-v2-generation requires an evaluation report; "
            "run evaluation before publishing comparison data"
        )

    verdicts = load_verdicts(evaluation_report)
    groups = group_results_by_runway(summary, fallback_airport=airport)
    index: dict[str, Any] = {
        "schemaVersion": "comparison-v2-generation",
        "generation": generation,
        "epoch": EPOCH.isoformat(),
        "startHidden": start_hidden,
        "category": category,
        "referenceSource": "canonicalObserved",
        "groups": [],
    }
    if summary.get("split") in DATASET_SPLITS:
        index["datasetSplit"] = summary["split"]
    created: list[Path] = []
    try:
        for (group_airport, runway), results in sorted(groups.items()):
            czml, records = build_runway_comparison(
                results,
                states_dir,
                [],
                airport=group_airport,
                start_hidden=start_hidden,
                scenario_initial=scenario_initial,
                verdicts=verdicts,
                include_reference_entities=False,
            )
            out_path = (
                out_dir / f"comparison_{group_airport}_{runway}_{generation}.czml"
            )
            _write_json_atomic(out_path, czml)
            created.append(out_path)
            for record in records:
                record["czml"] = out_path.name
            index["groups"].extend(records)
            failed = sum(1 for record in records if record["status"] == "failed")
            off_target = sum(
                1 for record in records if record["status"] == "offTarget"
            )
            print(
                f"✓ staged {out_path.name}: {len(records)} group(s), "
                f"{failed} unsolved (red), {off_target} off-target (yellow) "
                f"-> {len(czml)} packets"
            )

        index["optimization"] = optimization_stats(summary, evaluation_report)
        index["evaluation"] = evaluation_batch_stats(evaluation_report)
        prediction = prediction_accuracy_stats(summary)
        if prediction is not None:
            index["prediction"] = prediction
        report_name = f"evaluation_report_{generation}.json"
        report_path = publish_evaluation_report(
            evaluation_report, out_dir, filename=report_name
        )
        created.append(report_path)
        index["evaluationReport"] = report_name

        index_path = out_dir / "comparison_index.json"
        _write_json_atomic(index_path, index, pretty=True)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise

    keep_names = {
        group["czml"]
        for group in index["groups"]
        if isinstance(group.get("czml"), str)
    }
    keep_names.add(index["evaluationReport"])
    try:
        prune_unreferenced_outputs(out_dir, keep_names)
    except OSError as exc:
        # Garbage collection is not part of the commit. The new index is already complete;
        # leaving dormant old files is safer than reporting the valid publication as failed.
        print(f"⚠ comparison generation committed; stale-file cleanup failed: {exc}")
    return index


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_adsb(airport: str, override: str | None) -> list[dict[str, Any]]:
    path = Path(override) if override else airport_data_path(airport, "trajectories.czml")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _upsert_category(
    manifest_path: Path, *, key: str, label: str, directory: str, group_count: int,
    constrained: bool, dataset_split: str | None = None,
    result_source: str | None = None,
    experiment: dict[str, Any] | None = None,
) -> int:
    """Add/replace one category in the shared ``categories.json`` manifest.

    Each ``--category`` run writes its CZMLs into its own subdir and records itself here, so
    the frontend can offer a selector listing exactly the optimization categories that exist
    (e.g. fitted ADS-B crossing / runway target / …). ``constrained`` is
    stamped as an explicit manifest field — the frontend keys constraint-scoped behavior
    (the Observe procedure auto-open) off it, never off the key/dir spelling. Returns the
    new total.
    """
    manifest: dict[str, Any] = {"categories": []}
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("categories"), list):
            manifest = loaded
    # The old mode was both semantically wrong (raw receiver cutoff) and misspelled
    # ``asdb``. Once the replacement is published, hide legacy aliases from the picker
    # without deleting their on-disk output directories.
    replaced = {key}
    if key == "fitted_adsb":
        replaced.update({"adsb", "asdb", "adsb_cons", "asdb_cons"})
    kept = [c for c in manifest["categories"] if c.get("key") not in replaced]
    entry = {"key": key, "label": label, "dir": directory, "groups": group_count,
             "constrained": constrained}
    if dataset_split is not None:
        entry["datasetSplit"] = dataset_split
    if result_source is not None:
        entry["resultSource"] = result_source
    if experiment is not None:
        entry["experiment"] = experiment
    kept.append(entry)
    manifest["categories"] = sorted(kept, key=lambda c: c["key"])
    _write_json_atomic(manifest_path, manifest, pretty=True)
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
        help="optimization-category key (e.g. fitted_adsb / runway / runway_cons). When set, "
             "the output-dir is treated as that category's subdir and a categories.json manifest "
             "is written to its parent so the frontend can offer a category selector.",
    )
    parser.add_argument(
        "--category-label", default=None,
        help="display label for --category (defaults to the key)",
    )
    parser.add_argument(
        "--dataset-split", choices=DATASET_SPLITS, default=None,
        help="dataset split represented by this prediction category; stored explicitly in "
             "the frontend manifest",
    )
    parser.add_argument(
        "--result-source", choices=("prediction", "experiment"), default=None,
        help="optional Observe result-source classification; legacy categories default to "
             "prediction, while checkpoint sweeps use experiment",
    )
    parser.add_argument(
        "--experiment-id", default=None,
        help="stable repository-relative checkpoint identity (required for experiment source)",
    )
    parser.add_argument(
        "--experiment-group", default=None,
        help="campaign/collection label used to group experiment checkpoints in the frontend",
    )
    parser.add_argument(
        "--experiment-checkpoint", default=None,
        help="repository-relative checkpoint path shown in experiment metadata",
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
        _write_json_atomic(output, czml)
        print(f"✓ wrote comparison CZML ({len(czml)} packets) -> {output}")
        return

    # Batch: one combined CZML per (airport, runway), driven by the summary.
    if not args.output_dir:
        parser.error("--summary requires --output-dir")
    if not args.evaluation_report:
        parser.error("--summary batch requires --evaluation-report")
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if args.dataset_split is not None and summary.get("split") != args.dataset_split:
        parser.error(
            f"--dataset-split {args.dataset_split!r} does not match summary split "
            f"{summary.get('split')!r}"
        )
    if args.result_source == "experiment" and (
        not args.experiment_id or not args.experiment_group or not args.experiment_checkpoint
    ):
        parser.error(
            "--result-source experiment requires --experiment-id, --experiment-group and "
            "--experiment-checkpoint"
        )
    states_dir = Path(args.states_dir) if args.states_dir else summary_path.parent
    out_dir = Path(args.output_dir)

    scenario_initial = (
        scenario_initial_map([p.strip() for p in args.scenarios.split(",") if p.strip()])
        if args.scenarios else None
    )
    evaluation_report = (
        json.loads(Path(args.evaluation_report).read_text(encoding="utf-8"))
        if args.evaluation_report else None
    )
    index = publish_comparison_batch(
        summary=summary,
        states_dir=states_dir,
        out_dir=out_dir,
        airport=args.airport,
        category=args.category,
        start_hidden=not args.start_visible,
        scenario_initial=scenario_initial,
        evaluation_report=evaluation_report,
    )
    groups = group_results_by_runway(summary, fallback_airport=args.airport)
    rate = summary.get("failure_rate")
    rate_str = f"{rate:.1%}" if isinstance(rate, (int, float)) else "n/a"
    print(f"✓ wrote {len(groups)} runway CZML(s) + index ({len(index['groups'])} groups) "
          f"to {out_dir}; overall failure rate {rate_str}")

    if args.category:
        config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
        experiment = None
        if args.result_source == "experiment":
            experiment = {
                "id": args.experiment_id,
                "group": args.experiment_group,
                "checkpoint": args.experiment_checkpoint,
                "model": config.get("model"),
                "predictionOutput": config.get("prediction_output", "state"),
                "horizonMode": config.get("horizon_mode", "normalized"),
                "seed": config.get("seed"),
            }
        total = _upsert_category(
            out_dir.parent / "categories.json",
            key=args.category, label=args.category_label or args.category,
            directory=out_dir.name, group_count=len(index["groups"]),
            constrained=args.constrained,
            dataset_split=args.dataset_split,
            result_source=args.result_source,
            experiment=experiment,
        )
        print(f"✓ registered category {args.category!r} -> {out_dir.parent / 'categories.json'} "
              f"({total} categor{'y' if total == 1 else 'ies'})")


if __name__ == "__main__":
    main()
