"""The sentence artefact on disk (framework document §3): one directory, written once.

``signals_<split>.npz`` + ``signals.json``   the per-step signals, where they came from, and the day split
                                             (`data.day_split`) that dealt them — test days are never here
``spec.json`` + ``measurements.json``        the vocabulary spec (with its sha) and the numbers behind it
``spec_from.json``                           only when the spec is another artefact's, kept unchanged (`keep_spec`):
                                             where it came from
``candidates.json``                          every airport's candidate runways and runway ends (its geometry)
``sentences_<split>.npz`` + ``labels.json``  the sentences, and every flight's outcome
``conformance/``                             the labeller's reference sample and its passed records
                                             (`instructions.conformance`)
``closed_loop/``                             the closed-loop sentences of every split and row interval, flown by an
                                             executor spec (`write_closed_loop`), with ``summary.json`` and their
                                             reference sample and passed records (``conformance/``,
                                             `autopilot.closed_loop`)

A reader checks the spec's sha against the sentences it opens and refuses a mismatch — the
sentence files carry the sha they were read with. The LABELLER is not named by its source here
(vocabulary §7.2, D21): which code wrote the sentences is recorded as information, and a runner that
labels, or reads sentences for a replay, asks the conformance for a passed record of today's code.
The arrival manifests' sha256s are recorded as information too; a flight rebuilt from the harvest
is compared row by row with its stored signals (`autopilot.flights.require_same_flight`). Nothing
here is ever overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ts_transformer.data.day_split import DEVELOPMENT_SPLITS, DaySplit, landing_day
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.labeller.interval import interval_rows, on_interval_rows
from ts_transformer.instructions.labeller.read import Reading
from ts_transformer.instructions.signals import SIGNALS_SCHEMA, FlightSignals, pack_signals, unpack_signals
from ts_transformer.instructions.spec import SPEC_SCHEMA, VocabularySpec
from ts_transformer.io_utils import utc_now, write_bytes_atomic, write_json_atomic

#: v3 (instruction-v4): five columns; per sentence the runway, the capture row, the "unspecified" row and the go-around
#: rows (each go-around's row and the row the runway is said again); no clearance row and no labeller source hash.
#: v4 (instruction-v5, D58): the altitude words are levels above the airport elevation E, not MSL.
SENTENCES_SCHEMA = "ts-instruction-sentences-v4"
CANDIDATES_SCHEMA = "ts-instruction-candidates-v2"
#: The splits an artefact holds: the development operating days (the internal selection set is its
#: own split, so no reader carves it out of train by another rule).
SPLITS = DEVELOPMENT_SPLITS


def _fresh(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"{path} exists; an instruction artefact is never overwritten")
    return path


def _require_split_days(days: DaySplit, split: str, flights: Sequence[FlightSignals]) -> None:
    """Every flight lands on a day of ``split``; a test day's flight is refused by name (`SealedDay`)."""
    for flight in flights:
        found = days.development_split(landing_day(flight.landing_time_utc))
        if found != split:
            raise ValueError(f"{flight.dataset_id} lands on a {found} day, not a {split} day")


def write_signals(directory: Path, items: dict[str, Sequence[FlightSignals]], record: dict[str, Any],
                  days: DaySplit) -> None:
    """``items``: split → flights, every flight on a day ``days`` deals to that split — all checked before the
    first file is written."""
    for split, flights in items.items():
        if split not in SPLITS:
            raise ValueError(f"an instruction artefact holds the splits {SPLITS}, not {split!r}")
        _require_split_days(days, split, flights)
    splits = {}
    for split, flights in items.items():
        arrays, meta = pack_signals(flights)
        np.savez_compressed(_fresh(directory / f"signals_{split}.npz"), **arrays)
        splits[split] = {"flights": meta}
    write_json_atomic(_fresh(directory / "signals.json"),
                      {"schema": SIGNALS_SCHEMA, **record, "day_split": days.to_dict(), "splits": splits})


def _signals_record(directory: Path) -> dict[str, Any]:
    record = json.loads((directory / "signals.json").read_text(encoding="utf-8"))
    if record["schema"] != SIGNALS_SCHEMA:
        raise ValueError(f"{directory / 'signals.json'} is not a {SIGNALS_SCHEMA} file")
    return record


def load_day_split(directory: Path) -> DaySplit:
    """The day split that dealt the artefact's flights."""
    return DaySplit.from_dict(_signals_record(directory)["day_split"])


def load_signals(directory: Path, split: str) -> list[FlightSignals]:
    """One split's flights; each is checked to land on a day of that split."""
    if split not in SPLITS:
        raise ValueError(f"an instruction artefact holds the splits {SPLITS}, not {split!r}")
    record = _signals_record(directory)
    with np.load(directory / f"signals_{split}.npz") as arrays:
        flights = unpack_signals({name: arrays[name] for name in arrays.files}, record["splits"][split]["flights"])
    _require_split_days(DaySplit.from_dict(record["day_split"]), split, flights)
    return flights


def signals_flights(directory: Path, split: str) -> list[dict[str, Any]]:
    """One split's flight records — identity, airport, runway, type, entry and landing times — in signal order, without
    the signal arrays (a sentence's ``signal_index`` indexes this list)."""
    if split not in SPLITS:
        raise ValueError(f"an instruction artefact holds the splits {SPLITS}, not {split!r}")
    return _signals_record(directory)["splits"][split]["flights"]


def write_candidates(directory: Path, geometries: dict[str, AirportGeometry]) -> None:
    write_json_atomic(_fresh(directory / "candidates.json"),
                      {"schema": CANDIDATES_SCHEMA,
                       "airports": {code: geometry.to_dict() for code, geometry in sorted(geometries.items())}})


def load_candidates(directory: Path) -> dict[str, AirportGeometry]:
    record = json.loads((directory / "candidates.json").read_text(encoding="utf-8"))
    if "schema" not in record or record["schema"] != CANDIDATES_SCHEMA:
        raise ValueError(f"{directory / 'candidates.json'} is not a {CANDIDATES_SCHEMA} file")
    return {code: AirportGeometry.from_dict(data) for code, data in record["airports"].items()}


def write_spec(directory: Path, spec: VocabularySpec, measurements: dict[str, Any], source: dict[str, Any]) -> None:
    """``source``: the git state the spec was measured at and the labeller code that measured it (information)."""
    write_json_atomic(_fresh(directory / "spec.json"),
                      {"schema": SPEC_SCHEMA, "sha256": spec.sha256, "spec": spec.to_dict(), "source": source})
    write_json_atomic(_fresh(directory / "measurements.json"), {"spec_sha256": spec.sha256, **measurements})


def keep_spec(source: Path, directory: Path, provenance: dict[str, Any]) -> VocabularySpec:
    """``source``'s spec kept for ``directory`` (a spec is the vocabulary's format, not the data's: new rows under the same
    words keep it, never measure it again): its ``spec.json`` and ``measurements.json`` copied byte for byte — the spec
    and the git state that measured it, on ``source``'s train signals — and ``spec_from.json`` saying so
    (``provenance``: the caller's name for ``source`` and the git state that kept it).
    Refused unless the two artefacts' signals were built under one configuration."""
    spec = load_spec(source)
    built = {path: _signals_record(path)["config"] for path in (source, directory)}
    if built[source] != built[directory]:
        raise ValueError(f"{source}'s signals were built under {built[source]}, {directory}'s under {built[directory]}")
    copies = {name: _fresh(directory / name) for name in ("spec.json", "measurements.json")}
    record = _fresh(directory / "spec_from.json")         # all three refused before any is written
    for name, path in copies.items():
        write_bytes_atomic(path, (source / name).read_bytes())
    write_json_atomic(record, {**provenance, "spec_sha256": spec.sha256, "written_utc": utc_now()})
    return spec


def _spec_record(directory: Path) -> dict[str, Any]:
    record = json.loads((directory / "spec.json").read_text(encoding="utf-8"))
    if record["schema"] != SPEC_SCHEMA:
        raise ValueError(f"{directory / 'spec.json'} is not a {SPEC_SCHEMA} file")
    return record


def load_spec(directory: Path) -> VocabularySpec:
    record = _spec_record(directory)
    spec = VocabularySpec.from_dict(record["spec"])
    if spec.sha256 != record["sha256"]:
        raise ValueError(f"{directory / 'spec.json'}: the stored sha does not match its spec")
    return spec


def write_sentences(directory: Path, split: str, spec: VocabularySpec, readings: Sequence[Reading],
                    signal_index: Sequence[int], labeller_code_sha256: str) -> None:
    """The labelled flights' sentences, in the order given; ``signal_index`` is each one's
    position in ``signals_<split>.npz``. A sentence's words line up row for row with the first
    ``len(words)`` rows of its signals (the rows before the threshold crossing). Sentence ``k``'s
    go-arounds are ``go_around_row[go_around_offsets[k]: go_around_offsets[k + 1]]`` (and the rows the
    runway is said again, ``runway_again_row``, alike). ``labeller_code_sha256`` names the code that
    read them (`instructions.conformance.labeller_code_sha256`): information, never checked."""
    lengths = np.array([len(r.words) for r in readings], dtype=np.int64)
    counts = np.array([len(r.go_around_rows) for r in readings], dtype=np.int64)
    instructions = [(f, i.column, i.value, i.row) for f, r in enumerate(readings) for i in r.instructions]
    table = np.array(instructions, dtype=np.int64).reshape(-1, 4)
    np.savez_compressed(
        _fresh(directory / f"sentences_{split}.npz"),
        schema=np.array(SENTENCES_SCHEMA),
        spec_sha256=np.array(spec.sha256),
        labeller_code_sha256=np.array(labeller_code_sha256),
        offsets=np.concatenate(([0], np.cumsum(lengths))).astype(np.int64),
        words=np.concatenate([r.words for r in readings]).astype(np.int16),
        signal_index=np.asarray(signal_index, dtype=np.int64),
        runway_index=np.array([r.runway_index for r in readings], dtype=np.int64),
        capture_row=np.array([r.capture_row for r in readings], dtype=np.int64),
        unspecified_row=np.array([r.unspecified_row for r in readings], dtype=np.int64),
        go_around_offsets=np.concatenate(([0], np.cumsum(counts))).astype(np.int64),
        go_around_row=np.array([row for r in readings for row in r.go_around_rows], dtype=np.int64),
        runway_again_row=np.array([row for r in readings for row in r.runway_again_rows], dtype=np.int64),
        instruction_flight=table[:, 0], instruction_column=table[:, 1],
        instruction_value=table[:, 2], instruction_row=table[:, 3],
    )


def load_sentences(directory: Path, split: str, spec: VocabularySpec) -> dict[str, np.ndarray]:
    """One split's sentences, refused unless they are this schema's and were read with ``spec``."""
    with np.load(directory / f"sentences_{split}.npz") as arrays:
        data = {name: arrays[name] for name in arrays.files}
    if str(data["schema"]) != SENTENCES_SCHEMA:
        raise ValueError(f"sentences_{split}.npz is not a {SENTENCES_SCHEMA} file")
    if str(data["spec_sha256"]) != spec.sha256:
        raise ValueError(f"sentences_{split}.npz was read with spec {str(data['spec_sha256'])[:12]}, "
                         f"not {spec.sha256[:12]}")
    return data


#: The closed-loop sentences (vocabulary §4.9, D32; written by `experiments/instruction_closed_loop.py` from
#: `autopilot.closed_loop`): per split and row interval, the words said from the first predicted step, which are
#: corrections, the states on every row (observed before that step, flown from it) and the errors against the observed
#: path.
#: v2 (A10, D42): `observed_row` and `timed_out` (the flight ends when the executor is done, not at the open-loop
#: sentence's last row).
#: v3 (A12, D44–D46): the observed words from the 2 s reading, `observed_row` a 2 s row, `matched_row` the matched
#: point's observed time; `vertical_m` NaN past the end of the observed path.
#: v4 (A15, D51): the states on every 2 s row, the Δ rows marked (`on_interval`).
#: v5 (A20, D58): the altitude words are levels above the airport elevation E (the states stay MSL).
CLOSED_LOOP_SCHEMA = "ts-instruction-closed-loop-v5"
#: Every array a closed-loop file holds.
CLOSED_LOOP_FIELDS = {"schema", "spec_sha256", "executor_params_sha256", "row_interval_s", "start_row", "signal_index",
                      "first_row", "offsets", "state_offsets", "words", "correction", "states", "on_interval",
                      "lateral_m", "vertical_m", "uncorrectable", "observed_row", "matched_row", "timed_out"}
#: The directory inside the artefact that holds them, written once (`closed_loop_path`).
CLOSED_LOOP_DIRECTORY = "closed_loop"


def closed_loop_path(directory: Path, split: str, row_interval_s: float) -> Path:
    return directory / CLOSED_LOOP_DIRECTORY / f"{split}_{row_interval_s:g}s.npz"


def write_closed_loop(path: Path, spec: VocabularySpec, *, executor_params_sha256: str, row_interval_s: float,
                      start_row: int, signal_index: Sequence[int], first_row: Sequence[int],
                      grids: Sequence[np.ndarray], corrections: Sequence[np.ndarray], states: Sequence[np.ndarray],
                      on_interval: Sequence[np.ndarray], lateral_m: Sequence[np.ndarray], vertical_m: Sequence[np.ndarray],
                      uncorrectable: Sequence[np.ndarray], observed_row: Sequence[np.ndarray],
                      matched_row: Sequence[np.ndarray], timed_out: Sequence[bool]) -> None:
    """One split's closed-loop sentences at one row interval, in the order given. Sentence ``k``'s words are
    ``words[offsets[k]: offsets[k + 1]]`` (from its first predicted step, `start_row` rows after its first row on the
    interval's grid, which is its signals' 2 s row ``first_row[k]``), its states ``states[state_offsets[k]:
    state_offsets[k + 1]]`` (every 2 s row from its first row to its last said row: observed before ``start_row``, flown
    from it, D51) and ``on_interval`` which of them are its Δ rows; ``uncorrectable``
    ``[rows, 2]`` the rows where the reading makes no heading / angle correction (D34); ``observed_row`` the last 2 s
    row of the open-loop reading (from ``first_row[k]``) whose words each row has said, ``matched_row`` the matched
    point's observed time there (2 s rows, D42, D45); ``vertical_m`` NaN past the end of the observed path (D44);
    ``timed_out`` whether each flight was done at its time limit."""
    if not grids:
        raise ValueError(f"no closed-loop sentence for {path.name}: nothing to write")
    lengths = np.array([len(grid) for grid in grids], dtype=np.int64)
    state_lengths = np.array([len(rows) for rows in states], dtype=np.int64)
    every = interval_rows(row_interval_s, spec.step_s)
    marks = [np.asarray(rows, dtype=bool) for rows in on_interval]
    if (not np.array_equal(state_lengths, (lengths + start_row - 1) * every + 1)
            or not all(np.array_equal(rows, on_interval_rows(len(rows), every)) for rows in marks)):
        raise ValueError("every sentence's states cover its 2 s rows from its first row to its last said row, its Δ rows "
                         "marked: the rows before the first predicted step and one Δ row for each word row")
    np.savez_compressed(
        _fresh(path),
        schema=np.array(CLOSED_LOOP_SCHEMA), spec_sha256=np.array(spec.sha256),
        executor_params_sha256=np.array(executor_params_sha256), row_interval_s=np.array(row_interval_s),
        start_row=np.array(start_row), signal_index=np.asarray(signal_index, dtype=np.int64),
        first_row=np.asarray(first_row, dtype=np.int64),
        offsets=np.concatenate(([0], np.cumsum(lengths))).astype(np.int64),
        state_offsets=np.concatenate(([0], np.cumsum(state_lengths))).astype(np.int64),
        words=np.concatenate(grids).astype(np.int16), correction=np.concatenate(corrections).astype(bool),
        states=np.concatenate(states).astype(np.float64), on_interval=np.concatenate(marks),
        lateral_m=np.concatenate(lateral_m).astype(np.float64),
        vertical_m=np.concatenate(vertical_m).astype(np.float64),
        uncorrectable=np.concatenate(uncorrectable).astype(bool),
        observed_row=np.concatenate(observed_row).astype(np.int64),
        matched_row=np.concatenate(matched_row).astype(np.float64), timed_out=np.asarray(timed_out, dtype=bool))


def load_closed_loop(path: Path, spec: VocabularySpec) -> dict[str, np.ndarray]:
    """One split's closed-loop sentences at one row interval, refused unless they are this schema's and were read with
    ``spec``."""
    with np.load(path) as arrays:
        data = {name: arrays[name] for name in arrays.files}
    if set(data) != CLOSED_LOOP_FIELDS or str(data["schema"]) != CLOSED_LOOP_SCHEMA:
        raise ValueError(f"{path} is not a {CLOSED_LOOP_SCHEMA} file (fields {sorted(data)})")
    if str(data["spec_sha256"]) != spec.sha256:
        raise ValueError(f"{path} was read with spec {str(data['spec_sha256'])[:12]}, not {spec.sha256[:12]}")
    return data
