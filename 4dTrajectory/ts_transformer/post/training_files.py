"""The files of the frontend's Training view of stage C (post-training §8 C11, outline §6): the index of an airport's
window sets and a set's sample, which the export (`experiments/post_training_export.py`) writes and the backend's live
executor (`aeroviz_backend/autopilot_segment/`) and the frontend read.

Not a runner, and torch-free: everything here raises `ValueError` (a set that is not listed: `NotListed`, one of them),
never `SystemExit` — a server thread would let that escape its handler and drop the request unanswered.

**Beside the other sets (outline §6 item 3).** A window set is ``<airport>/training/<set-id>/sample.json``
(`SAMPLE_SCHEMA`), listed in ``<airport>/training/index_post_v1.json`` (`INDEX_FILE`, `INDEX_SCHEMA`) — an index of its
own beside stage A's ``index_v4.json`` and stage B's ``index_prior_v2.json``, which this code never reads or writes. A
set or the index entry of one is never overwritten, and the index is rewritten only while it is still what the run read
at its start (`require_index_unchanged`). The same rules as stage A's and stage B's files (`instructions.training_files`,
`prior.training_files`), with this stage's names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ts_transformer.instructions.spec import READING_RULE
from ts_transformer.instructions.training_files import serialise
from ts_transformer.io_utils import utc_now

#: MIRROR of the frontend's reader of the window sets (C11); the reader refuses anything else by name, so these move
#: together. A name changes with its file's shape, on both sides, in the same change. Index v1 / sample v1 (C11,
#: 2026-10-05): windows of recorded traffic — the commanded flight's observed track, open-loop and closed-loop sentence
#: (stage A's head), the other aircraft on their records, and for each round of a post-training campaign the sentence
#: its model said for the commanded aircraft, flown, with the window's end (its outcome, a loss of separation and its
#: other aircraft, the reward). Sample v2 (prior D127, followed for windows, 2026-10-06): a round's flown track is written
#: unrounded — the live executor's answer is checked against it within the executor's bound.
INDEX_SCHEMA = "aeroviz-training-window-index-v1"
INDEX_FILE = "index_post_v1.json"
SAMPLE_SCHEMA = "aeroviz-training-window-sample-v2"
SAMPLE_FILE = "sample.json"
SET_KIND = "post-training-windows"


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
    """The airport's window sets as they stand (no index yet: none), for a run that will add ``set_id``; refused when
    the index already lists it — an export is never overwritten."""
    path = training / INDEX_FILE
    if not path.exists():
        return []
    sets = index_sets(json.loads(path.read_text(encoding="utf-8")), path, airport)
    if any(item["id"] == set_id for item in sets):
        raise ValueError(f"{path} already lists set {set_id}; an export is never overwritten")
    return sets


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def require_writable(training: Path, airport: str, entry: dict[str, Any], existing: list[dict[str, Any]]) -> None:
    """The set can be written: the index is still what ``existing`` read at the start of the run (another export that
    wrote it meanwhile would lose its set) and the set's directory does not exist. A run asks it of every airport before
    it writes any, so a refusal leaves no airport written."""
    if read_index(training, airport, entry["id"]) != existing:
        raise ValueError(f"{training / INDEX_FILE} changed since this run read it; run the export again")
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
