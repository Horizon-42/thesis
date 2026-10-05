"""The files of the frontend's Training view of stage B (prior design §12 B6, outline §6): the index of an airport's
prior sets and a set's sample, which the export (`experiments/prior_training_export.py`) writes and the backend's live
executor (`aeroviz_backend/autopilot_segment/`) and the frontend read.

Not a runner, and torch-free: everything here raises `ValueError` (a set that is not listed: `NotListed`, one of them),
never `SystemExit` — a server thread would let that escape its handler and drop the request unanswered.

**Beside the other sets (outline §6 item 3).** A prior set is ``<airport>/training/<set-id>/sample.json``
(`SAMPLE_SCHEMA`), listed in ``<airport>/training/index_prior_v2.json`` (`INDEX_FILE`, `INDEX_SCHEMA`) — an index of its
own beside stage A's ``index_v4.json`` and the instruction-v3 view's ``index.json``, which this code never reads or
writes. A set or the index entry of one is never overwritten, and the index is rewritten only while it is still what the
run read at its start (`require_index_unchanged`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ts_transformer.instructions.spec import READING_RULE
from ts_transformer.io_utils import utc_now

#: MIRROR of the frontend's reader of the prior's sets (B6); the reader refuses anything else by name, so these move
#: together. A name changes with its file's shape, on both sides, in the same change. Index v1 / sample v1 (B6,
#: 2026-10-05): the flights of one free-generation readout — the observed track, the open-loop sentence, the closed-loop
#: sentence at the prior's Δ and the sentences the prior said, flown again, with the words the procedure masks blocked.
#: Index v2 / sample v2 (B10, outline D109): a set's source names the claim of the val read it was exported under
#: (``validationClaim``: null for every set but the base's one validation readout, whose flights are of val); the
#: readers give val to that set alone.
#: Sample v3 (B13, D127): each prior sentence's flown track written unrounded (the live segment is checked against it
#: within the executor's bound).
INDEX_SCHEMA = "aeroviz-training-prior-index-v2"
INDEX_FILE = "index_prior_v2.json"
SAMPLE_SCHEMA = "aeroviz-training-prior-sample-v3"
SAMPLE_FILE = "sample.json"
SET_KIND = "prior-free-generation"
#: The reader of the val days whose claim a set's ``validationClaim`` names (`checkpoint.claim_validation_read`): the
#: base's one validation free generation. The export writes it, the backend's live service checks it; the frontend's
#: `TRAINING_PRIOR_CLAIM_READER` is its MIRROR.
CLAIM_READER = "prior_free_generation"


class NotListed(ValueError):
    """The set asked for is not in its airport's index (a server answers this as "not found")."""


def index_sets(payload: dict[str, Any], path: Path, airport: str) -> list[dict[str, Any]]:
    """The sets an airport's index lists; refused when it is another schema's or another airport's."""
    if payload["schema"] != INDEX_SCHEMA:
        raise ValueError(f"{path} is a {payload['schema']} file, not {INDEX_SCHEMA}")
    if payload["airport"] != airport:
        raise ValueError(f"{path} is {payload['airport']}'s index, not {airport}'s")
    return payload["sets"]


def listed_set(training: Path, airport: str, set_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Set ``set_id``'s index entry and its sample, refused unless the sample is this schema's, this kind's and this
    reading's, and is the set and airport it is listed as (`NotListed` when the index does not list it)."""
    path = training / INDEX_FILE
    if not path.exists():
        raise NotListed(f"{path} does not exist")
    listed = [item for item in index_sets(json.loads(path.read_text(encoding="utf-8")), path, airport)
              if item["id"] == set_id]
    if not listed:
        raise NotListed(f"{path} lists no set {set_id}")
    if len(listed) > 1:
        raise ValueError(f"{path} lists set {set_id} {len(listed)} times")
    entry = listed[0]
    file = training / entry["file"]
    sample = json.loads(file.read_text(encoding="utf-8"))
    if sample["schema"] != SAMPLE_SCHEMA or (entry["kind"], entry["readingRule"]) != (SET_KIND, READING_RULE):
        raise ValueError(f"set {set_id} at {airport} is a {entry['kind']} set of {entry['readingRule']} in a "
                         f"{sample['schema']} file, not a {SET_KIND} set of {READING_RULE} in a {SAMPLE_SCHEMA} file")
    if (sample["setId"], sample["airport"], sample["readingRule"]) != (set_id, airport, READING_RULE):
        raise ValueError(f"{file} holds set {sample['setId']} at {sample['airport']}, not {set_id} at {airport}")
    return entry, sample


def read_index(training: Path, airport: str, set_id: str) -> list[dict[str, Any]]:
    """The airport's prior sets as they stand (no index yet: none), for a run that will add ``set_id``; refused when the
    index already lists it — an export is never overwritten."""
    path = training / INDEX_FILE
    if not path.exists():
        return []
    sets = index_sets(json.loads(path.read_text(encoding="utf-8")), path, airport)
    if any(item["id"] == set_id for item in sets):
        raise ValueError(f"{path} already lists set {set_id}; an export is never overwritten")
    return sets


def serialise(payload: dict[str, Any]) -> str:
    """A Training file as it is written: without indentation and refusing NaN."""
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def require_index_unchanged(training: Path, airport: str, set_id: str, existing: list[dict[str, Any]]) -> None:
    """The index is still what ``existing`` read at the start of the run: another export that wrote it meanwhile would
    lose its set."""
    if read_index(training, airport, set_id) != existing:
        raise ValueError(f"{training / INDEX_FILE} changed since this run read it; run the export again")


def require_writable(training: Path, airport: str, entry: dict[str, Any], existing: list[dict[str, Any]]) -> None:
    """The set can be written: the index is still what the run read (`require_index_unchanged`) and the set's directory
    does not exist. A run asks it of every airport before it writes any, so a refusal leaves no airport written."""
    require_index_unchanged(training, airport, entry["id"], existing)
    directory = (training / entry["file"]).parent
    if directory.exists():
        raise ValueError(f"{directory} exists; an export is never overwritten")


def write_set(training: Path, airport: str, entry: dict[str, Any], text: str, existing: list[dict[str, Any]]) -> Path:
    """A set's sample (``text``, `serialise`'s) in a directory of its own, then the index with it added — refused
    unless `require_writable`."""
    require_writable(training, airport, entry, existing)
    out = training / entry["file"]
    out.parent.mkdir(parents=True)
    _write_text_atomic(out, text)
    _write_text_atomic(training / INDEX_FILE, serialise({"schema": INDEX_SCHEMA, "writtenUtc": utc_now(),
                                                        "airport": airport, "sets": [*existing, entry]}))
    return out
