"""The files of the frontend's Training module: the sets an airport's index lists, the overlays drawn over them, and the
checks every writer and reader of them shares (the runners `instruction_training_export`, `executor_training_export`,
`prior_training_export`, and the backend's live executor).

Not a runner, and torch-free: everything here raises `ValueError` (a set that is not listed: `NotListed`, one of
them), never `SystemExit` — a server thread would let that escape its handler and drop the request
unanswered. A runner's ``main`` turns them into ``parser.error``.

**A set** (`KIND_READBACK`): ``<airport>/training/<set-id>/sample.json`` (`SAMPLE_SCHEMA`), listed in
``<airport>/training/index.json`` (`INDEX_SCHEMA`). **An overlay** — another model's output on a set's own flights —
``<airport>/training/<overlay-id>/<file>``, listed in ``<airport>/training/overlays.json`` (`OVERLAYS_SCHEMA`) with the
set it is drawn over and that set's sample sha256. A set or an overlay is never overwritten, and a manifest is
rewritten only when it is still what the run read at its start (`require_unchanged`): another export that wrote it
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

#: MIRROR of `aeroviz-4d/src/data/trainingOverlays.ts` (`TRAINING_OVERLAYS_SCHEMA`, `TRAINING_OVERLAY_KINDS`): the
#: manifest of what is drawn OVER this airport's sets — another model's output on a set's own flights, each in a
#: file of its own schema (`executor_training_export`: the executor's replay; `prior_training_export`: the prior's
#: predictions; `prior_generation_training_export`: the prior's own sentences, flown). The index lists sets and keeps
#: its shape; this file lists overlays, each naming the set it is drawn over and that set's sample by its sha256, so a
#: set re-exported under the same id is not mistaken for it. A kind the reader does not know rejects that entry alone.
OVERLAYS_SCHEMA = "aeroviz-training-overlays-v1"
OVERLAYS_FILE = "overlays.json"
KIND_EXECUTOR = "executor-replay"
KIND_PRIOR = "prior-prediction"
KIND_GENERATION = "prior-generation"
OVERLAY_KINDS = (KIND_EXECUTOR, KIND_PRIOR, KIND_GENERATION)

#: Only validation flights are drawn: train is what a prior will be fitted on, and test stays shut.
SPLIT = "val"

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
    if len(listed) != 1:
        raise NotListed(f"{path} lists no set {set_id}")
    return listed[0]


def check_readback(entry: dict[str, Any], sample: dict[str, Any], file: Path, airport: str) -> None:
    """A set the Training exporters write and the frontend reads: a read-back of this reading rule, its sample under
    `SAMPLE_SCHEMA`, the set and airport it is listed as, drawn from `SPLIT`. Anything else is refused by name — an
    overlay drawn over (or a flight flown from) a set the frontend refuses would never be seen."""
    if (entry["kind"], entry["readingRule"]) != (KIND_READBACK, READING_RULE):
        raise ValueError(f"set {entry['id']} at {airport} is a {entry['kind']} set of {entry['readingRule']}, not a "
                         f"{KIND_READBACK} set of {READING_RULE}")
    if sample["schema"] != SAMPLE_SCHEMA:
        raise ValueError(f"{file} is a {sample['schema']} file, not {SAMPLE_SCHEMA}: re-export the set first")
    found = (sample["setId"], sample["airport"], sample["cohort"]["split"])
    if found != (entry["id"], airport, SPLIT):
        raise ValueError(f"{file} holds set {found[0]} at {found[1]}, split {found[2]}; expected {entry['id']} at "
                         f"{airport}, split {SPLIT}")


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


@dataclass(frozen=True)
class BaseSet:
    """An exported set an overlay is drawn over: its index entry, its sample, and the sample file's sha256."""

    airport: str
    entry: dict[str, Any]
    sample: dict[str, Any]
    sha256: str

    @property
    def block(self) -> dict[str, Any]:
        """What an overlay records of its set: the frontend matches the set id, the sample's time of writing and
        the spec against the sample it loaded; `check-publication` matches the sha256 of the file on disk."""
        return {"setId": self.entry["id"], "sampleWrittenUtc": self.sample["writtenUtc"], "sampleSha256": self.sha256,
                "specSha256": self.sample["vocabulary"]["specSha256"]}


def open_base_set(training: Path, airport: str, set_id: str, spec: VocabularySpec) -> BaseSet:
    """Set ``set_id`` for an overlay of a model of ``spec``: `check_readback`'s set, of that spec."""
    path = training / INDEX_FILE
    entry = listed_set(json.loads(path.read_text(encoding="utf-8")), path, airport, set_id)
    file = training / entry["file"]
    raw = file.read_bytes()
    sample = json.loads(raw)
    check_readback(entry, sample, file, airport)
    if (entry["vocabularySha256"], sample["vocabulary"]["specSha256"]) != (spec.sha256, spec.sha256):
        raise ValueError(f"set {set_id} at {airport} is of spec {entry['vocabularySha256'][:12]} (its sample "
                         f"{sample['vocabulary']['specSha256'][:12]}), not {spec.sha256[:12]}")
    return BaseSet(airport, entry, sample, hashlib.sha256(raw).hexdigest())


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
    return {"id": overlay_id, "kind": kind, "base": base.entry["id"], "baseSampleSha256": base.sha256, "title": title,
            "file": f"{overlay_id}/{file_name}", "flights": flights, "source": source}


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


def require_unchanged(training: Path, airport: str, new_id: str, existing: list[dict[str, Any]], *,
                      manifest: str) -> None:
    """The manifest (`INDEX_FILE` or `OVERLAYS_FILE`) is still what ``existing`` read at the start of the run:
    another export that wrote it meanwhile would lose its entry to ``[*existing, new]``. A run checks EVERY airport
    before it writes any, and each write checks again."""
    now = read_index(training, airport, new_id) if manifest == INDEX_FILE else read_overlays(training, airport, new_id)
    if now != existing:
        raise ValueError(f"{training / manifest} changed since this run read it; run the export again")


def write_set(training: Path, airport: str, entry: dict[str, Any], text: str, existing: list[dict[str, Any]]) -> Path:
    """A set's sample (``text``, `serialise`'s) in a directory of its own (refused if it exists), then the index with
    it added (`require_unchanged`)."""
    require_unchanged(training, airport, entry["id"], existing, manifest=INDEX_FILE)
    out = training / entry["file"]
    out.parent.mkdir(parents=True)
    _write_text_atomic(out, text)
    _write_text_atomic(training / INDEX_FILE, serialise({"schema": INDEX_SCHEMA, "writtenUtc": utc_now(),
                                                        "airport": airport, "sets": [*existing, entry]}))
    return out


def write_overlay(training: Path, airport: str, entry: dict[str, Any], text: str, existing: list[dict[str, Any]]) -> Path:
    """An overlay's file (``text``, `serialise`'s) in a directory of its own (refused if it exists), then the overlays
    manifest with it added (`require_unchanged`)."""
    require_unchanged(training, airport, entry["id"], existing, manifest=OVERLAYS_FILE)
    out = training / entry["file"]
    out.parent.mkdir()
    _write_text_atomic(out, text)
    _write_text_atomic(training / OVERLAYS_FILE, serialise({"schema": OVERLAYS_SCHEMA, "writtenUtc": utc_now(),
                                                           "airport": airport, "overlays": [*existing, entry]}))
    return out
