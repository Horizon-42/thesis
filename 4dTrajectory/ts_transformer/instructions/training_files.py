"""The files of the frontend's Training module: the sets an airport's index lists, the overlays drawn over them, and the
checks every writer and reader of them shares (the runners `instruction_training_export`, `executor_training_export`,
`prior_training_export`, `prior_generation_training_export`, and the backend's live executor).

Not a runner, and torch-free: everything here raises `ValueError` (a set that is not listed: `NotListed`, one of
them), never `SystemExit` — a server thread would let that escape its handler and drop the request
unanswered. A runner lets it propagate, with its traceback.

**A set** — a read-back set (`KIND_READBACK`): ``<airport>/training/<set-id>/sample.json`` (`SAMPLE_SCHEMA`), val
flights; a window set (`KIND_TRAFFIC`, `window_training_export`): ``<airport>/training/<set-id>/traffic.json``
(`TRAFFIC_SCHEMA`), multi-aircraft windows of `TRAFFIC_SPLIT` — each listed in ``<airport>/training/index.json``
(`INDEX_SCHEMA`). **An overlay** — another model's output on a set's own flights —
``<airport>/training/<overlay-id>/<file>``, listed in ``<airport>/training/overlays.json`` (`OVERLAYS_SCHEMA`) with the
set it is drawn over, and bound to that set by what it shares with it (`BaseSet.block`, `require_set_datum`) — never by
the set file's bytes. A set or an overlay is never overwritten, and a manifest is
rewritten only when it is still what the run read at its start (`require_index_unchanged`,
`require_overlays_unchanged`): another export that wrote it
meanwhile would otherwise lose its entry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ts_transformer.instructions import display
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import arrival_manifest_sha256s
from ts_transformer.instructions.spec import READING_RULE, VocabularySpec
from ts_transformer.instructions.words import COLUMNS, UNCHANGED
from ts_transformer.io_utils import utc_now

#: MIRROR of `aeroviz-4d/src/data/trainingSample.ts` (`TRAINING_INDEX_SCHEMA`, `TRAINING_SAMPLE_SCHEMA`,
#: `TRAINING_READABLE_SET_KIND`); the reader refuses anything else by name, so these move together. A name changes
#: with its file's shape, on both sides, in the same change. Sample v7 (2026-09-24) is `instruction-v3`'s shape: a
#: heading word is the band θ ± the heading tolerance over the rows it is judged on (from its row plus the lead to the
#: next heading word's, `display.HeadingBand`) with each row's verdict — no turn region, turn end, hold funnel, split
#: part or inserted intercept any more; the capture turn is its rows and the labeller's check; the vocabulary carries
#: the lead. (v6 was `instruction-v2`'s: turns bounded by rate, judged holds; a flight's ``typecode`` its own ICAO
#: type or null, as in v7.) The index keeps its v1 shape: sets of every vocabulary sit in it.
INDEX_SCHEMA = "aeroviz-training-index-v1"
SAMPLE_SCHEMA = "aeroviz-training-sample-v7"
KIND_READBACK = "vocabulary-readback"
INDEX_FILE = "index.json"
SAMPLE_FILE = "sample.json"
#: MIRROR of `aeroviz-4d/src/data/trainingTraffic.ts` (`TRAINING_TRAFFIC_SCHEMA`, `TRAINING_TRAFFIC_SET_KIND`): a window
#: set (the Training module §2.9) — multi-aircraft windows, the flights the prior commands in them as a read-back set's
#: flights, every other aircraft of a window as the judge replays it, and the windows as recorded.
TRAFFIC_SCHEMA = "aeroviz-training-traffic-v1"
KIND_TRAFFIC = "traffic-windows"
TRAFFIC_FILE = "traffic.json"

#: MIRROR of `aeroviz-4d/src/data/trainingOverlays.ts` (`TRAINING_OVERLAYS_SCHEMA`, `TRAINING_OVERLAY_KINDS`): the
#: manifest of what is drawn OVER this airport's sets — another model's output on a set's own flights, each in a
#: file of its own schema (`executor_training_export`: the executor's replay; `prior_training_export`: the prior's
#: predictions; `prior_generation_training_export`: the prior's own sentences, flown — from the set's own starts, or with
#: ``--augment-seed`` from augmented ones, a kind of its own). The index lists sets and keeps
#: its shape; this file lists overlays, each naming the set it is drawn over. A kind the reader does not know rejects
#: that entry alone. v2 (2026-09-28): an entry no longer names its set's sample by the file's sha256 — a set re-exported
#: with the same flights keeps its overlays; what an overlay shares with its set is in its payload (`BaseSet.block`).
OVERLAYS_SCHEMA = "aeroviz-training-overlays-v2"
OVERLAYS_FILE = "overlays.json"
KIND_EXECUTOR = "executor-replay"
KIND_PRIOR = "prior-prediction"
KIND_GENERATION = "prior-generation"
KIND_AUGMENTED_GENERATION = "prior-generation-augmented"
KIND_WINDOW_GENERATION = "window-generation"
OVERLAY_KINDS = (KIND_EXECUTOR, KIND_PRIOR, KIND_GENERATION, KIND_AUGMENTED_GENERATION, KIND_WINDOW_GENERATION)

#: Only validation flights are drawn into a read-back set: train is what a prior will be fitted on, and test stays shut.
SPLIT = "val"
#: A window set's windows are the formal window readouts' (`traffic_window_generation`): the prior's internal selection
#: split, so val stays unread until the multi-aircraft stage is settled (multi-aircraft design §1.3; user 2026-09-30).
TRAFFIC_SPLIT = "select"
#: Each set kind's file schema and split.
SET_KINDS = {KIND_READBACK: (SAMPLE_SCHEMA, SPLIT), KIND_TRAFFIC: (TRAFFIC_SCHEMA, TRAFFIC_SPLIT)}

#: How far apart two readings of one flight's HAE − MSL may be, each read off a pair of heights written to 0.01 m
#: (`rounded(..., 2)`): 0.01 m a reading. MIRROR of `TRAINING_DATUM_TOLERANCE_M` in the frontend's overlay reader.
DATUM_TOLERANCE_M = 0.02

#: Why a word was issued — every `Instruction.kind` the labeller (`instructions/labeller/*`) writes into a kept
#: sentence. MIRROR of `TRAINING_WORD_KINDS` in the frontend reader, which names each one; a kind outside this list
#: stops the export by name rather than reaching a view unnamed.
WORD_KINDS = ("initial", "per-step", "clear", "target", "step", "angle", "unspecified")


class NotListed(ValueError):
    """The set asked for is not in its airport's index (a server answers this as "not found")."""


def rounded(values: Any, digits: int) -> list[float]:
    """Numbers at display precision, flattened: what every Training file writes."""
    return [round(float(value), digits) for value in np.asarray(values, dtype=np.float64).ravel()]


def words_in_force(grid: np.ndarray) -> np.ndarray:
    """``[rows, columns]``: the word in force at each step — the value at the last step its column was said at or
    before (a sentence's step 0 says every column)."""
    rows = grid.shape[0]
    last_said = np.maximum.accumulate(np.where(grid != UNCHANGED, np.arange(rows)[:, None], 0), axis=0)
    return grid[last_said, np.arange(len(COLUMNS))[None, :]]


def band_payload(band: display.HeadingBand) -> dict[str, Any]:
    """A heading word's band as the reader takes it: its judged rows, θ ± the tolerance on the chart's branch, and each
    row's verdict — the set's words and the executor overlay's write it the same way."""
    return {"firstRow": band.first_row, "stopRow": band.stop_row, "targetOnTrackDeg": round(band.target_on_track_deg, 3),
            "bandDeg": rounded(band.band_deg, 3), "inside": [int(bool(value)) for value in band.inside]}


# ---- the stored sentence
#: What a re-read sentence must give back of its stored one, besides its words grid.
SENTENCE_FIELDS = ("runway_index", "capture_row", "join_row", "unspecified_row")


def stored_sentence(sentences: dict[str, np.ndarray], index: int) -> dict[str, Any]:
    """Sentence ``index`` of an artefact's split (`artefact.load_sentences`): its words grid and `SENTENCE_FIELDS`."""
    offsets = sentences["offsets"]
    return {"words": sentences["words"][offsets[index]: offsets[index + 1]],
            **{name: int(sentences[name][index]) for name in SENTENCE_FIELDS}}


def require_stored_sentence(dataset_id: str, reading: Any, stored: dict[str, Any]) -> None:
    """The one test that a flight read again (a labeller `Reading`) gives its stored sentence: every word at its step
    and column, and every one of `SENTENCE_FIELDS`. Refused by name, with the first difference."""
    if reading.words.shape != stored["words"].shape:
        raise ValueError(f"{dataset_id}: re-read {reading.words.shape[0]} steps, the artefact holds "
                         f"{stored['words'].shape[0]}")
    differ = np.argwhere(reading.words != stored["words"])
    if len(differ):
        row, column = (int(v) for v in differ[0])
        raise ValueError(f"{dataset_id}: re-read {COLUMNS[column]} word {int(reading.words[row, column])} at step {row}, "
                         f"the artefact holds {int(stored['words'][row, column])} ({len(differ)} cells differ)")
    for name in SENTENCE_FIELDS:
        if getattr(reading, name) != stored[name]:
            raise ValueError(f"{dataset_id}: re-read {name} {getattr(reading, name)}, the artefact holds {stored[name]}")


# ---- the index and its sets
def index_sets(payload: dict[str, Any], path: Path, airport: str) -> list[dict[str, Any]]:
    """The sets an airport's index lists; refused when it is another schema's or another airport's."""
    if payload["schema"] != INDEX_SCHEMA:
        raise ValueError(f"{path} is a {payload['schema']} file, not {INDEX_SCHEMA}")
    if payload["airport"] != airport:
        raise ValueError(f"{path} is {payload['airport']}'s index, not {airport}'s")
    return payload["sets"]


def listed_set(payload: dict[str, Any], path: Path, airport: str, set_id: str) -> dict[str, Any]:
    """The index entry of set ``set_id`` (`NotListed` when the index does not list it)."""
    listed = [item for item in index_sets(payload, path, airport) if item["id"] == set_id]
    if not listed:
        raise NotListed(f"{path} lists no set {set_id}")
    if len(listed) > 1:
        raise ValueError(f"{path} lists set {set_id} {len(listed)} times")
    return listed[0]


def check_set(entry: dict[str, Any], payload: dict[str, Any], file: Path, airport: str, kind: str) -> None:
    """A set the Training exporters write and the frontend reads, of ``kind`` (`SET_KINDS`): of this reading rule, its
    file under the kind's schema, the set and airport it is listed as, drawn from the kind's split. Anything else is
    refused by name — an overlay drawn over (or a flight flown from) a set the frontend refuses would never be seen."""
    if (entry["kind"], entry["readingRule"]) != (kind, READING_RULE):
        raise ValueError(f"set {entry['id']} at {airport} is a {entry['kind']} set of {entry['readingRule']}, not a "
                         f"{kind} set of {READING_RULE}")
    schema, split = SET_KINDS[kind]
    if payload["schema"] != schema:
        raise ValueError(f"{file} is a {payload['schema']} file, not {schema}: re-export the set first")
    found = (payload["setId"], payload["airport"], payload["cohort"]["split"], payload["vocabulary"]["readingRule"])
    if found != (entry["id"], airport, split, READING_RULE):
        raise ValueError(f"{file} holds set {found[0]} at {found[1]}, split {found[2]}, read under {found[3]}; expected "
                         f"{entry['id']} at {airport}, split {split}, {READING_RULE}")


def read_index(training: Path, airport: str, set_id: str) -> list[dict[str, Any]]:
    """The airport's sets as they stand (no index yet: none), for a run that will add ``set_id``; refused when the
    index already lists it — an export is never overwritten."""
    path = training / INDEX_FILE
    if not path.exists():
        return []
    sets = index_sets(json.loads(path.read_text(encoding="utf-8")), path, airport)
    if any(item["id"] == set_id for item in sets):
        raise ValueError(f"{path} already lists set {set_id}; an export is never overwritten")
    return sets


def candidates_sha256(geometry: AirportGeometry) -> str:
    """The runway pointer's classes ARE the airport's candidates (vocabulary design §4.1), and the
    landing rule reads every runway end of the airport (§2.2), so the identity of both is the sha of
    the airport geometry — candidates and runway ends — a sample's ``candidatesSha256`` and the index's
    ``runwaySha256``."""
    canonical = json.dumps(geometry.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def airport_frame(geometry: AirportGeometry) -> dict[str, Any]:
    """The airport frame a sample's metres are in (its ``airportFrame``)."""
    return {"code": geometry.code, "lat": geometry.frame.lat0, "lon": geometry.frame.lon0,
            "elevationM": geometry.frame.alt0}


@dataclass(frozen=True)
class BaseSet:
    """An exported set an overlay is drawn over: its index entry and its file (a read-back set's sample, a window set's
    windows — both carry the vocabulary, the candidates and the frame)."""

    airport: str
    entry: dict[str, Any]
    sample: dict[str, Any]

    @property
    def block(self) -> dict[str, Any]:
        """What an overlay records of its set — what the two share besides their flights: the set id, the vocabulary
        spec, the candidate runways (a runway index points into them) and the airport frame (the metres). The frontend
        matches each against the sample it loaded, and the flights one by one; never the sample file's bytes or its
        time of writing, so a set exported again with the same flights keeps its overlays."""
        return {"setId": self.entry["id"], "specSha256": self.sample["vocabulary"]["specSha256"],
                "candidatesSha256": self.sample["candidatesSha256"], "airportFrame": self.sample["airportFrame"]}


def open_base_set(training: Path, airport: str, set_id: str, kind: str, spec: VocabularySpec,
                  geometry: AirportGeometry) -> BaseSet:
    """Set ``set_id`` of ``kind`` for an overlay of a model of ``spec`` flown on ``geometry`` (the overlay's own
    artefact's airport): `check_set`'s set, of that spec, whose candidates and frame are that geometry's."""
    path = training / INDEX_FILE
    entry = listed_set(json.loads(path.read_text(encoding="utf-8")), path, airport, set_id)
    file = training / entry["file"]
    sample = json.loads(file.read_text(encoding="utf-8"))
    check_set(entry, sample, file, airport, kind)
    if (entry["vocabularySha256"], sample["vocabulary"]["specSha256"]) != (spec.sha256, spec.sha256):
        raise ValueError(f"set {set_id} at {airport} is of spec {entry['vocabularySha256'][:12]} (its sample "
                         f"{sample['vocabulary']['specSha256'][:12]}), not {spec.sha256[:12]}")
    if sample["candidatesSha256"] != candidates_sha256(geometry):
        raise ValueError(f"set {set_id} at {airport} has candidates {sample['candidatesSha256'][:12]}, the overlay's "
                         f"artefact {candidates_sha256(geometry)[:12]}")
    if sample["airportFrame"] != airport_frame(geometry):
        raise ValueError(f"set {set_id} at {airport} is in the frame {sample['airportFrame']}, the overlay's artefact "
                         f"in {airport_frame(geometry)}")
    return BaseSet(airport, entry, sample)


def base_flights(base: BaseSet, flights: list[Any], sentences: dict[str, np.ndarray]) -> list[tuple[Any, int]]:
    """Each of the set's flights in the artefact — its signals and its sentence's position in ``sentences`` — in the
    set's order; refused unless the stored sentence is the set's own: every word at its step and column, and the
    same number of steps."""
    by_id = {flight.dataset_id: i for i, flight in enumerate(flights)}
    stored_at = {int(index): k for k, index in enumerate(sentences["signal_index"])}
    out = []
    for item in base.sample["flights"]:
        i = by_id.get(item["datasetId"])
        if i is None or i not in stored_at:
            raise ValueError(f"{item['datasetId']} of set {base.entry['id']} has no labelled sentence in the artefact")
        k = stored_at[i]
        grid = stored_sentence(sentences, k)["words"]
        cells = [(int(r), int(c), int(grid[r, c])) for r, c in zip(*np.nonzero(grid != UNCHANGED))]
        events = [(event["row"], event["column"], event["value"]) for event in item["words"]["events"]]
        if len(grid) != item["rows"] or cells != events:
            raise ValueError(f"{item['datasetId']}: the artefact's sentence ({len(grid)} steps) is not the one set "
                             f"{base.entry['id']} shows ({item['rows']} steps)")
        out.append((flights[i], k))
    return out


def set_flight_datum_m(item: dict[str, Any]) -> float:
    """A set flight's HAE − MSL (``item``: one of its sample's flights): its first row's ellipsoid height less its reported
    height — the one runway offset its exporter added (`runway_hae_minus_msl_m`), to `DATUM_TOLERANCE_M` / 2."""
    signals = item["signals"]
    return float(signals["altitudeHaeM"][0]) - float(signals["raw"]["altitudeM"][0])


def require_set_datum(base: BaseSet, flights: list[Any], hae_minus_msl_m: dict[str, float]) -> None:
    """Each of the set's flights (``flights``: `base_flights`' signals, in the set's order) has the HAE − MSL an overlay
    that draws heights adds to it (``hae_minus_msl_m``: `runway_hae_minus_msl_m` of the overlay's artefact, by runway) —
    or the overlay's tracks would stand beside the set's on another datum; refused by name."""
    for item, flight in zip(base.sample["flights"], flights, strict=True):
        found, own = set_flight_datum_m(item), hae_minus_msl_m[flight.runway]
        if abs(found - own) > DATUM_TOLERANCE_M:
            raise ValueError(f"{item['datasetId']}: set {base.entry['id']} draws it {found:+.2f} m HAE − MSL, the overlay "
                             f"{own:+.2f} m")


# ---- the overlays
def read_overlays(training: Path, airport: str, overlay_id: str) -> list[dict[str, Any]]:
    """The airport's overlays as they stand (none yet: an empty list); refused when the manifest is another
    schema's or another airport's, or already lists this overlay — an overlay is never overwritten."""
    path = training / OVERLAYS_FILE
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema"] != OVERLAYS_SCHEMA:
        raise ValueError(f"{path} is a {payload['schema']} file, not {OVERLAYS_SCHEMA}")
    if payload["airport"] != airport:
        raise ValueError(f"{path} is {payload['airport']}'s overlays, not {airport}'s")
    if any(item["id"] == overlay_id for item in payload["overlays"]):
        raise ValueError(f"{path} already lists overlay {overlay_id}; an overlay is never overwritten")
    return payload["overlays"]


def overlay_entry(overlay_id: str, kind: str, base: BaseSet, title: str, file_name: str, flights: int,
                  source: dict[str, Any]) -> dict[str, Any]:
    """One overlay as the manifest lists it: what it is, the set it is drawn over, and where its file is."""
    if kind not in OVERLAY_KINDS:
        raise ValueError(f"unknown overlay kind {kind!r}")
    return {"id": overlay_id, "kind": kind, "base": base.entry["id"], "title": title,
            "file": f"{overlay_id}/{file_name}", "flights": flights, "source": source}


# ---- the height Cesium draws in
def runway_hae_minus_msl_m(artefact: Path, airport: str, manifest: Path) -> dict[str, float]:
    """Each runway's HAE − MSL offset at ``airport``, metres by ident: what the data plane subtracted from the reported
    heights of the flights assigned to that runway (`flight_scenarios.datum.flight_to_msl`, from the arrival manifest's
    ``runway_targets``). A flight's MSL height plus its OWN runway's offset is the height its aircraft reported — the
    ellipsoid height Cesium draws in — so every track drawn for a flight (observed, replayed, generated, flown live)
    adds that one number. A geoid model's local value does not give the reported height back: at KRDU 05L EGM96 says
    −33.53 m where −32.0 m was subtracted. ``manifest``: the airport's arrival manifest (`repo_layout.
    arrival_manifest_path`), refused unless it is the one ``artefact``'s signals were read from (its recorded sha256,
    as `autopilot.flights.rebuild_series` checks it)."""
    recorded = arrival_manifest_sha256s(artefact)[airport]
    data = manifest.read_bytes()
    if hashlib.sha256(data).hexdigest() != recorded:
        raise ValueError(f"{manifest} is not the arrival manifest {artefact.name}'s signals were read from "
                         f"(sha256 {recorded[:12]} recorded)")
    targets = json.loads(data)["runway_targets"]
    return {ident: float(target["hae_minus_msl_m"]) for ident, target in targets.items()}


# ---- writing
def serialise(payload: dict[str, Any]) -> str:
    """A Training file as it is written: without indentation (its per-step arrays are long, and one number per line
    triples its size) and refusing NaN. Called while every airport is built, so a payload that cannot be written stops
    the run before the first airport is."""
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def require_index_unchanged(training: Path, airport: str, set_id: str, existing: list[dict[str, Any]]) -> None:
    """The index is still what ``existing`` read at the start of the run: another export that wrote it meanwhile
    would lose its set to ``[*existing, new]``. A run checks EVERY airport before it writes any, and each write checks
    again."""
    if read_index(training, airport, set_id) != existing:
        raise ValueError(f"{training / INDEX_FILE} changed since this run read it; run the export again")


def require_overlays_unchanged(training: Path, airport: str, overlay_id: str, existing: list[dict[str, Any]]) -> None:
    """The overlays manifest is still what ``existing`` read at the start of the run (`require_index_unchanged`'s
    rule, for overlays)."""
    if read_overlays(training, airport, overlay_id) != existing:
        raise ValueError(f"{training / OVERLAYS_FILE} changed since this run read it; run the export again")


def write_set(training: Path, airport: str, entry: dict[str, Any], text: str, existing: list[dict[str, Any]]) -> Path:
    """A set's sample (``text``, `serialise`'s) in a directory of its own (refused if it exists), then the index with
    it added (`require_index_unchanged`)."""
    require_index_unchanged(training, airport, entry["id"], existing)
    out = training / entry["file"]
    out.parent.mkdir(parents=True)
    _write_text_atomic(out, text)
    _write_text_atomic(training / INDEX_FILE, serialise({"schema": INDEX_SCHEMA, "writtenUtc": utc_now(),
                                                        "airport": airport, "sets": [*existing, entry]}))
    return out


def write_overlay(training: Path, airport: str, entry: dict[str, Any], text: str, existing: list[dict[str, Any]]) -> Path:
    """An overlay's file (``text``, `serialise`'s) in a directory of its own (refused if it exists), then the overlays
    manifest with it added (`require_overlays_unchanged`)."""
    require_overlays_unchanged(training, airport, entry["id"], existing)
    out = training / entry["file"]
    out.parent.mkdir()
    _write_text_atomic(out, text)
    _write_text_atomic(training / OVERLAYS_FILE, serialise({"schema": OVERLAYS_SCHEMA, "writtenUtc": utc_now(),
                                                           "airport": airport, "overlays": [*existing, entry]}))
    return out
