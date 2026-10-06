"""The files of the frontend's Training view of stage A (vocabulary §12.1 A23, outline §6): the index of an airport's
stage-A sets and a set's sample, and what the export (`experiments/training_export.py`) and the backend's live executor
(`aeroviz_backend/autopilot_segment/`) share of them.

Not a runner, and torch-free: everything here raises `ValueError` (a set that is not listed: `NotListed`, one of them),
never `SystemExit` — a server thread would let that escape its handler and drop the request unanswered.

**Beside the old sets (outline §6 item 3).** A stage-A set is ``<airport>/training/<set-id>/sample.json``
(`SAMPLE_SCHEMA`), listed in ``<airport>/training/index_v5.json`` (`INDEX_FILE`, `INDEX_SCHEMA`) — a NEW index beside
the older ones (``index_v4.json``, the instruction-v3 view's ``index.json``), which this code never reads or writes, so a
checkout of the older view keeps its sets until they are deleted.

**One writer for the three stages (outline §6.2 item 7).** `TrainingFiles` reads and writes the index and the sets of
every stage, given the stage's constants: stage A's are `FILES`, stage B's and C's are in `prior.training_files` and
`post.training_files`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ts_transformer.instructions import envelope
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import READ_SPLITS
from ts_transformer.instructions.spec import READING_RULE
from ts_transformer.instructions.words import wrap180
from ts_transformer.io_utils import utc_now

#: MIRROR of `aeroviz-4d/src/data/trainingSample.ts` (`TRAINING_INDEX_SCHEMA`, `TRAINING_INDEX_FILE`,
#: `TRAINING_SAMPLE_SCHEMA`, `TRAINING_SET_KIND`); the reader refuses anything else by name, so these move together. A
#: name changes with its file's shape, on both sides, in the same change. Index v2 / sample v9 (A23, 2026-10-04): the
#: stage-A view — five columns, the open-loop sentence on the 2 s rows and the closed-loop sentences at Δ = 2, 4, 8 s with
#: their correction words, flown states, judge's outcome and decision-altitude check. Sample v10 (A32, 2026-10-05): the
#: same shape from the closed-loop format v8 — the observed rows by the start rule (D77), a flight ending where the judge
#: ends it (D79), the flights drawn from the closed-loop files and checked against the stored outcome (D86) — so that a
#: view of this code never reads a v9 set. Index v3 (``index_v5.json``) / sample v11 (D135, 2026-10-06): a flown sentence
#: is the one block of every stage (`experiments.training_export.flown_sentence`), its track unrounded; the sets are
#: listed in a new index beside v2's, which the view before D135 reads until the old sets are deleted.
INDEX_SCHEMA = "aeroviz-training-index-v3"
INDEX_FILE = "index_v5.json"
SAMPLE_SCHEMA = "aeroviz-training-sample-v11"
SAMPLE_FILE = "sample.json"
SET_KIND = "closed-loop-readback"
#: The splits a set draws from (outline §6 item 4, D85): the splits a readout reads.
SPLITS = READ_SPLITS


class NotListed(ValueError):
    """The set asked for is not in its airport's index (a server answers this as "not found"); every stage's."""


def rounded(values: Any, digits: int) -> list[float]:
    """Numbers at display precision, flattened: what every Training file writes."""
    return [round(float(value), digits) for value in np.asarray(values, dtype=np.float64).ravel()]


def unrounded(values: Any) -> list[float]:
    """Numbers as they are, flattened (prior D127: a value that a later step computes from is not rounded for display —
    the live executor's answer is checked against a flown track within the executor's bound)."""
    return [float(value) for value in np.asarray(values, dtype=np.float64).ravel()]


def nullable(values: Any, digits: int) -> list[float | None]:
    """`rounded`, a NaN written as null (an error with no observed reference past the end of the observed path, D44)."""
    return [None if not np.isfinite(value) else round(float(value), digits)
            for value in np.asarray(values, dtype=np.float64).ravel()]


# ---- the envelopes the views draw (they compute none)
def heading_bands(track_deg: np.ndarray, words: Sequence[tuple[int, float]], lead_rows: int, end_row: int,
                  tolerance_deg: float) -> list[dict[str, Any]]:
    """Each heading word ``(row, target track)`` as `envelope.heading_words_inside` judges it, for drawing: its judged
    rows ``[firstRow, stopRow)``, its target track, the band's half width and each judged row's verdict."""
    rows = envelope.heading_word_rows([row for row, _ in words], lead_rows, end_row)
    out = []
    for (row, target), (first, stop) in zip(words, rows):
        inside = np.abs(wrap180(np.asarray(track_deg)[first:stop] - target)) <= tolerance_deg
        out.append({"row": int(row), "firstRow": int(first), "stopRow": int(stop), "targetDeg": round(float(target), 3),
                    "toleranceDeg": float(tolerance_deg), "inside": [int(v) for v in inside]})
    return out


def tube_payload(word_row: int, end_row: int, level_m: float | None, low_m: np.ndarray, high_m: np.ndarray,
                 elevation_m: float, check: dict[str, Any]) -> dict[str, Any]:
    """An altitude word's tube over its rows ``[row, end)`` (`labeller.vertical.tube_bounds`, heights above E) as MSL
    heights, beside the flight's MSL track, with its check (`tube_checks`)."""
    return {"row": int(word_row), "endRow": int(end_row), "levelM": None if level_m is None else round(level_m, 1),
            "lowMslM": rounded(np.asarray(low_m) + elevation_m, 1), "highMslM": rounded(np.asarray(high_m) + elevation_m, 1),
            "rows": int(check["rows"]), "inside": int(check["inside"]), "contained": bool(check["contained"])}


def speed_payload(check: dict[str, Any], target_mps: float, tolerance_mps: float) -> dict[str, Any]:
    """A speed word's span (`labeller.speed.span_checks`): its rows, its target and band, where it arrived, its checks."""
    return {"row": int(check["row"]), "endRow": int(check["row"] + check["rows"]), "targetMps": float(target_mps),
            "toleranceMps": float(tolerance_mps), "arrivalRow": int(check["row"] + check["arrival_rows"]),
            "transitionOk": bool(check["transition_ok"]), "accelOk": bool(check["accel_ok"]),
            "contained": bool(check["contained"])}


def candidates_sha256(geometry: AirportGeometry) -> str:
    """The identity of an airport's candidates and runway ends (the runway word points into them): written beside a
    set, information for its reader."""
    canonical = json.dumps(geometry.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---- the index and its sets: one definition for the three stages (outline §6.2 item 7)
def serialise(payload: dict[str, Any]) -> str:
    """A Training file as it is written: without indentation and refusing NaN."""
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class TrainingFiles:
    """One stage's Training files — its index ``<airport>/training/<index_file>`` (``index_schema``) and its sets
    ``<airport>/training/<set-id>/sample.json`` (``sample_schema``, of ``set_kind``) — read and written by the one
    definition the three stages share (outline §6.2 item 7); each stage gives only its constants (stage A: `FILES`;
    `prior.training_files.FILES`; `post.training_files.FILES`). A set or its index entry is never overwritten, and the
    index is rewritten only while it is still what the run read at its start (`require_index_unchanged`)."""

    index_schema: str
    index_file: str
    sample_schema: str
    set_kind: str

    def index_sets(self, payload: dict[str, Any], path: Path, airport: str) -> list[dict[str, Any]]:
        """The sets an airport's index lists; refused when it is another schema's or another airport's."""
        if payload["schema"] != self.index_schema:
            raise ValueError(f"{path} is a {payload['schema']} file, not {self.index_schema}")
        if payload["airport"] != airport:
            raise ValueError(f"{path} is {payload['airport']}'s index, not {airport}'s")
        return payload["sets"]

    def listed_set(self, training: Path, airport: str, set_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Set ``set_id``'s index entry and its sample, refused unless the sample is this schema's, this kind's and this
        reading's, and is the set and airport it is listed as (`NotListed` when the index does not list it)."""
        path = training / self.index_file
        if not path.exists():
            raise NotListed(f"{path} does not exist")
        listed = [item for item in self.index_sets(json.loads(path.read_text(encoding="utf-8")), path, airport)
                  if item["id"] == set_id]
        if not listed:
            raise NotListed(f"{path} lists no set {set_id}")
        if len(listed) > 1:
            raise ValueError(f"{path} lists set {set_id} {len(listed)} times")
        entry = listed[0]
        file = training / entry["file"]
        sample = json.loads(file.read_text(encoding="utf-8"))
        if sample["schema"] != self.sample_schema or (entry["kind"], entry["readingRule"]) != (self.set_kind,
                                                                                              READING_RULE):
            raise ValueError(f"set {set_id} at {airport} is a {entry['kind']} set of {entry['readingRule']} in a "
                             f"{sample['schema']} file, not a {self.set_kind} set of {READING_RULE} in a "
                             f"{self.sample_schema} file")
        if (sample["setId"], sample["airport"], sample["readingRule"]) != (set_id, airport, READING_RULE):
            raise ValueError(f"{file} holds set {sample['setId']} at {sample['airport']}, not {set_id} at {airport}")
        return entry, sample

    def read_index(self, training: Path, airport: str, set_id: str) -> list[dict[str, Any]]:
        """The airport's sets of this stage as they stand (no index yet: none), for a run that will add ``set_id``;
        refused when the index already lists it — an export is never overwritten."""
        path = training / self.index_file
        if not path.exists():
            return []
        sets = self.index_sets(json.loads(path.read_text(encoding="utf-8")), path, airport)
        if any(item["id"] == set_id for item in sets):
            raise ValueError(f"{path} already lists set {set_id}; an export is never overwritten")
        return sets

    def require_index_unchanged(self, training: Path, airport: str, set_id: str, existing: list[dict[str, Any]]
                                ) -> None:
        """The index is still what ``existing`` read at the start of the run: another export that wrote it meanwhile
        would lose its set."""
        if self.read_index(training, airport, set_id) != existing:
            raise ValueError(f"{training / self.index_file} changed since this run read it; run the export again")

    def require_writable(self, training: Path, airport: str, entry: dict[str, Any], existing: list[dict[str, Any]]
                         ) -> None:
        """The set can be written: the index is still what the run read (`require_index_unchanged`) and the set's
        directory does not exist. A run asks it of every airport before it writes any, so a refusal leaves no airport
        written."""
        self.require_index_unchanged(training, airport, entry["id"], existing)
        directory = (training / entry["file"]).parent
        if directory.exists():
            raise ValueError(f"{directory} exists; an export is never overwritten")

    def write_set(self, training: Path, airport: str, entry: dict[str, Any], text: str, existing: list[dict[str, Any]]
                  ) -> Path:
        """A set's sample (``text``, `serialise`'s) in a directory of its own, then the index with it added — refused
        unless `require_writable`."""
        self.require_writable(training, airport, entry, existing)
        out = training / entry["file"]
        out.parent.mkdir(parents=True)
        _write_text_atomic(out, text)
        _write_text_atomic(training / self.index_file, serialise({"schema": self.index_schema, "writtenUtc": utc_now(),
                                                                  "airport": airport, "sets": [*existing, entry]}))
        return out


#: Stage A's Training files.
FILES = TrainingFiles(index_schema=INDEX_SCHEMA, index_file=INDEX_FILE, sample_schema=SAMPLE_SCHEMA, set_kind=SET_KIND)
