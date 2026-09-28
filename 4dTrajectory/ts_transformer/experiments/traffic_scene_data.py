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
0: context) to its last loss step (`prior.scene_data.SceneSample`). Each flight's rows are held once, however many
samples it is in, and laid out when a batch is formed — and so is what its edge features read
(`inference.scene_edges.SceneRows`, `Built.rows`): every placed row's time and position, and — for a flight with a
sentence — the runway in force before the step (its ``in_force``, the words said at an earlier row) with its position on
that runway's approach clock; the CWT category (`runway_schedule.wake_category`; none without a type — counted). The
edge features themselves are computed as a batch is formed (`edges`). A scene prior records what decides them
(`edge_source_sha256`) and is read only by the same code.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from geokit import NM_M

from ts_transformer.autopilot.spec import logic
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
from ts_transformer.prior.scene_data import Node, SceneSample
from ts_transformer.repo_layout import CWT_SUPPLEMENT, CWT_TABLE, arrival_manifest_path

PACKAGE = Path(__file__).resolve().parents[1]
#: What decides a scene prior's edge features, as package paths: the features and the separation rules they read
#: (`runway_schedule`), the clock position (`instructions.airport.relative_to_runway`), the runway in force
#: (`prior.data.sentence_steps`), the steps and row times (`prior.scene.hang`, `presence`), what the samples hand them and
#: how a batch lays them out — and the CWT tables the categories are read from (`EDGE_TABLES`). A scene prior's checkpoint
#: records their hash (`edge_source_sha256`) and `prior_train.load_prior` reads it only on the same: a code change in any
#: of these files — the scheduler's in `runway_schedule` too — refuses every scene prior, as the executor's hash refuses
#: its specs.
EDGE_SOURCES = ("inference/scene_edges.py", "inference/runway_schedule.py", "instructions/airport.py", "prior/data.py",
                "prior/scene.py", "prior/scene_data.py", "experiments/traffic_scene_data.py")
EDGE_TABLES = (CWT_TABLE, CWT_SUPPLEMENT)


def edge_source_sha256() -> str:
    """sha256 over `EDGE_SOURCES` (path and `autopilot.spec.logic`, in order: wording is free, code is not) and
    `EDGE_TABLES` (name and bytes)."""
    digest = hashlib.sha256()
    for path in EDGE_SOURCES:
        digest.update(path.encode("utf-8") + b"\0" + logic((PACKAGE / path).read_text(encoding="utf-8")).encode("utf-8")
                      + b"\0")
    for table in EDGE_TABLES:
        digest.update(table.name.encode("utf-8") + b"\0" + table.read_bytes() + b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class FlightRows:
    """One flight as the samples place it, held once: its presence, a `Node`'s arrays over its rows, its positions, the
    runway in force before each step with its position on that runway's approach clock (NaN without one) and its CWT
    category."""

    presence: Presence
    node_rows: dict[str, np.ndarray]
    e_m: np.ndarray
    n_m: np.ndarray
    height_m: np.ndarray
    runway: list[str | None]
    along_m: np.ndarray
    category: str | None


@dataclass(frozen=True)
class Built:
    """One sample: what the prior reads, its flights (in the sample's order) and its airport's separation rules."""

    sample: SceneSample
    flights: tuple[FlightRows, ...]
    separation: Separation

    @property
    def rows(self) -> SceneRows:
        """What the sample's edge features read, laid out on its steps."""
        shape = (len(self.flights), self.sample.steps)
        columns = {name: np.full(shape, np.nan) for name in ("time", "e", "n", "h", "along")}
        runway: list[list[str | None]] = [[None] * self.sample.steps for _ in self.flights]
        for a, (node, f) in enumerate(zip(self.sample.nodes, self.flights)):
            kept = self.sample.kept(node)
            span = slice(node.first_step, node.first_step + kept)
            for name, values in (("time", f.presence.times_s), ("e", f.e_m), ("n", f.n_m), ("h", f.height_m),
                                 ("along", f.along_m)):
                columns[name][a, span] = values[:kept]
            runway[a][span] = f.runway[:kept]
        return SceneRows(columns["time"], columns["e"], columns["n"], columns["h"], columns["along"], runway,
                         [f.category for f in self.flights])


def edges(built: Built) -> np.ndarray:
    """The sample's edge features, ``[T, A, A, E]`` (`inference.scene_edges`)."""
    return scene_edges(built.rows, built.separation)


def _flight(signals: FlightSignals, grid: np.ndarray | None, capture_row: int | None, geometry: AirportGeometry,
            separation: Separation, landings: Landings | None, airport: int) -> FlightRows:
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
    along = np.full(rows, np.nan)
    for ident in {r for r in runway if r is not None}:
        on = np.array([r == ident for r in runway])
        candidate = geometry.candidates[geometry.candidate_index(ident)]
        # the distance before the threshold reads the positions only (the track argument is not read here)
        relative_rows = relative_to_runway(signals.e_m[:rows][on], signals.n_m[:rows][on], signals.track_deg[:rows][on],
                                           signals.altitude_m[:rows][on], candidate)
        along[on] = separation.along_nm[ident] * NM_M - relative_rows.before_threshold_m
    category = None if signals.typecode is None else wake_category(signals.typecode)
    return FlightRows(seen, arrays, signals.e_m[:rows], signals.n_m[:rows], signals.altitude_m[:rows], runway, along,
                      category)


def _sample(members: Sequence[FlightRows], first_s: float, steps: int, loss_from: int, loss_to: int, step_s: float,
            airport: int, separation: Separation) -> Built:
    nodes = tuple(Node(f.presence.dataset_id, f.presence.speaking,
                       int(round((float(hang(f.presence.start_s, step_s)) - first_s) / step_s)), **f.node_rows)
                  for f in members)
    return Built(SceneSample(airport, nodes, steps, loss_from, loss_to), tuple(members), separation)


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
        if len(by_key) != len(flights):
            raise ValueError(f"{code}: {len(flights) - len(by_key)} flights share a dataset id with another")
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
