"""The scene samples of one split as the scene prior trains on and reads them (multi-aircraft design §2.3, §2.5, §6.5 step 3)
— shared by the scene training runner and the readouts, not a runner itself.

Per airport, from the instruction artefact and the arrivals manifest the signals were read from (its runway targets give
the separation rules; refused when the manifest moved): every flight of the split is in the scene — with a sentence (its
`prior.data.flight_record` rows: inputs, the words said before each step, targets, what is asked) or background (the
labeller refused it: the same inputs over all its rows, its words "none", nothing asked; one with more rows than the
model's positions hold is left out and counted — a handful of tracks hours long, 2 of the training days' 340 background
flights; a flight with a sentence that long is refused). The flights chain into
segments and each segment is cut into samples by §2.3's rule (`prior.scene.SceneIndex.segments`, `prior.scene.samples`);
a sample's steps run from its earliest flight's first step (a flight carried in from before the cut starts at its row
0: context) to its last loss step (`prior.scene_data.scene_sample`). Beside each sample, what its edge features read
(`inference.scene_edges.SceneRows`): every placed row's time, position, velocity, and — for a flight with a sentence — the
runway in force before the step (its ``in_force``, the words said at an earlier row) with its position on that runway's
approach clock and rate along it; the CWT category (`runway_schedule.wake_category`; none without a type — counted).
The edge features themselves are computed when a batch is formed (`edges`): about 1.6 GB for the training days if kept.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from geokit import NM_M

from ts_transformer.experiments.traffic_census import APPROACH_SPEED_MPS
from ts_transformer.inference.runway_schedule import Separation, faa_separation, wake_category
from ts_transformer.inference.scene_edges import SceneRows, scene_edges
from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway
from ts_transformer.instructions.artefact import arrival_manifest_sha256s, load_candidates, load_sentences, load_signals
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import RUNWAY
from ts_transformer.io_utils import file_sha256
from ts_transformer.prior.data import STATIC_FEATURES, flight_record, own_context, step_inputs
from ts_transformer.prior.scene import Landings, Presence, SceneIndex, hang, presence, samples
from ts_transformer.prior.scene_data import Node, SceneSample, scene_sample
from ts_transformer.repo_layout import arrival_manifest_path


@dataclass(frozen=True)
class Built:
    """One sample: what the prior reads, what its edge features read, and its airport's separation rules."""

    sample: SceneSample
    rows: SceneRows
    separation: Separation


def edges(built: Built) -> np.ndarray:
    """The sample's edge features, ``[T, A, A, E]`` (`inference.scene_edges`)."""
    return scene_edges(built.rows, built.separation)


@dataclass(frozen=True)
class _Flight:
    presence: Presence
    signals: FlightSignals
    node_rows: dict[str, np.ndarray]     # a Node's arrays over the flight's rows
    runway: list[str | None]             # per row: the runway in force before the step
    along_m: np.ndarray                  # per row (NaN without a runway in force)
    along_rate_mps: np.ndarray
    category: str | None


def _flight(signals: FlightSignals, grid: np.ndarray | None, capture_row: int | None, geometry: AirportGeometry,
            separation: Separation, landings: Landings | None, airport: int) -> _Flight:
    rows = signals.n_rows if grid is None else len(grid)
    seen = presence(signals, None if grid is None else rows, geometry)
    if grid is None:
        context = own_context(signals, landings) if landings is not None else None
        features, relative = step_inputs(signals, rows, geometry, context)
        zeros = np.zeros((rows, 6))
        arrays = {"features": features, "relative": relative, "static": np.zeros(len(STATIC_FEATURES), dtype=np.float32),
                  "in_force": zeros.astype(np.int64), "since": zeros.astype(np.float32),
                  "targets": zeros.astype(np.int64), "asked": zeros.astype(bool)}
        runway: list[str | None] = [None] * rows
    else:
        record = flight_record(signals, grid, geometry, landings, airport, capture_row)
        arrays = {name: getattr(record, name) for name in ("features", "relative", "static", "in_force", "since",
                                                           "targets", "asked")}
        runway = [None if k == 0 else geometry.candidates[k - 1].ident for k in record.in_force[:, RUNWAY]]
    along, rate = np.full(rows, np.nan), np.full(rows, np.nan)
    for ident in {r for r in runway if r is not None}:
        on = np.array([r == ident for r in runway])
        candidate = geometry.candidates[geometry.candidate_index(ident)]
        relative_rows = relative_to_runway(signals.e_m[:rows][on], signals.n_m[:rows][on], signals.track_deg[:rows][on],
                                           signals.altitude_m[:rows][on], candidate)
        along[on] = separation.along_nm[ident] * NM_M - relative_rows.before_threshold_m
        rate[on] = signals.ground_speed_mps[:rows][on] * np.cos(np.radians(relative_rows.track_minus_course_deg))
    category = None if signals.typecode is None else wake_category(signals.typecode)
    return _Flight(seen, signals, arrays, runway, along, rate, category)


def _sample(members: Sequence[_Flight], first_s: float, steps: int, loss_from: int, loss_to: int, step_s: float,
            airport: int, separation: Separation) -> Built:
    nodes, starts = [], []
    for f in members:
        start = int(round((float(hang(f.presence.start_s, step_s)) - first_s) / step_s))
        starts.append(start)
        nodes.append(Node(f.presence.dataset_id, f.presence.speaking, start, **f.node_rows))
    sample = scene_sample(airport, nodes, steps, loss_from, loss_to)
    count = len(members)
    shape = (count, steps)
    time_s = np.full(shape, np.nan)
    columns = {name: np.full(shape, np.nan) for name in ("e", "n", "h", "gs", "track", "vs", "along", "rate")}
    runway: list[list[str | None]] = [[None] * steps for _ in range(count)]
    for a, (f, start) in enumerate(zip(members, starts)):
        kept = min(len(f.runway), steps - start)
        span = slice(start, start + kept)
        s = f.signals
        time_s[a, span] = f.presence.times_s[:kept]
        for name, values in (("e", s.e_m), ("n", s.n_m), ("h", s.altitude_m), ("gs", s.ground_speed_mps),
                             ("track", s.track_deg), ("vs", s.vertical_rate_mps), ("along", f.along_m),
                             ("rate", f.along_rate_mps)):
            columns[name][a, span] = values[:kept]
        runway[a][start: start + kept] = f.runway[:kept]
    rows = SceneRows(time_s, columns["e"], columns["n"], columns["h"], columns["gs"], columns["track"], columns["vs"],
                     columns["along"], columns["rate"], runway, [f.category for f in members])
    return Built(sample, rows, separation)


def airport_separation(airport: str, recorded_sha256: Mapping[str, str]) -> Separation:
    """The airport's separation rules from the arrivals manifest the artefact's signals were read from (refused when it
    moved since)."""
    manifest = arrival_manifest_path(airport)
    if file_sha256(manifest) != recorded_sha256[airport]:
        raise ValueError(f"{manifest} is not the manifest the artefact's signals were read from "
                         f"(sha256 {recorded_sha256[airport][:12]} recorded)")
    return faa_separation(json.loads(manifest.read_text(encoding="utf-8"))["runway_targets"],
                          speed_mps=APPROACH_SPEED_MPS)


def build_split(directory: Path, split: str, spec: VocabularySpec, airports: Sequence[str],
                landings: Mapping[str, Landings] | None, max_rows: int) -> tuple[list[Built], dict[str, Any]]:
    """Every sample of ``split`` at ``airports`` (the model's, in its order: a sample's airport is its index there), and
    what was built: flights with a sentence and background, the background left out for more rows than ``max_rows``
    (the model's positions), those without a type, samples, the most aircraft in one. ``landings``: each airport's
    landing context, None for a variant without it (`prior.data.airport_landings`)."""
    geometries = load_candidates(directory)
    signals = load_signals(directory, split)
    sentences = load_sentences(directory, split, spec)
    offsets = sentences["offsets"]
    spoken = {int(i): k for k, i in enumerate(sentences["signal_index"])}
    recorded = arrival_manifest_sha256s(directory)
    step_s = spec.step_s
    built: list[Built] = []
    counts: Counter = Counter(dict.fromkeys(("with_a_sentence", "background", "background_left_out_for_its_length",
                                             "without_a_type", "samples", "most_aircraft"), 0))
    for code in sorted({f.airport for f in signals}):
        airport = airports.index(code)
        geometry, separation = geometries[code], airport_separation(code, recorded)
        context = None if landings is None else landings[code]
        flights = []
        for i, s in enumerate(signals):
            if s.airport != code:
                continue
            k = spoken.get(i)
            grid = None if k is None else sentences["words"][offsets[k]: offsets[k + 1]]
            rows = s.n_rows if grid is None else len(grid)
            if rows > max_rows:
                if grid is not None:
                    raise ValueError(f"{s.dataset_id}: a sentence of {rows} rows; the model's positions hold {max_rows}")
                counts["background_left_out_for_its_length"] += 1
                continue
            flights.append(_flight(s, grid, None if k is None else int(sentences["capture_row"][k]), geometry,
                                   separation, context, airport))
            counts["with_a_sentence" if k is not None else "background"] += 1
            counts["without_a_type"] += s.typecode is None
        by_key = {f.presence.dataset_id: f for f in flights}
        for segment in SceneIndex([f.presence for f in flights]).segments(step_s):
            cuts = samples(segment, step_s)
            for number, cut in enumerate(cuts):
                members = [by_key[p.dataset_id] for p in cut.flights]
                first_s = min(float(hang(f.presence.start_s, step_s)) for f in members)
                end_s = cut.loss_end_s + (step_s if number == len(cuts) - 1 else 0.0)    # the last sample: its end included
                steps = int(round((end_s - first_s) / step_s))
                loss_from = int(round((cut.loss_start_s - first_s) / step_s))
                built.append(_sample(members, first_s, steps, loss_from, steps, step_s, airport, separation))
                counts["samples"] += 1
                counts["most_aircraft"] = max(counts["most_aircraft"], len(members))
    return built, dict(counts)
