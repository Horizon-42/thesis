"""The labeller checked by what it reads, not by its source (vocabulary §7.2 #2, D21; the executor's own rule, C33).

When `instruction_labels` writes an artefact it also writes ``conformance/``: a REFERENCE — a fixed sample of the train
split's flights (`draw`: a seeded permutation, `PER_AIRPORT` labelled flights and `REFUSED_PER_AIRPORT` refused ones of
every airport) with their signals and what the labeller made of each: the sentence (words, runway, capture row,
"unspecified" row, go-around rows, stratum — D70) or the refusal's reason — and each labelled sentence also on its UTC Δ grid at every
row interval of `INTERVALS_S` (`labeller.interval.on_utc_grid`, the rule of D45 and D46; D49), or why that grid refuses
it. The CHECK (`check`, runner `instruction_conformance`) reads the reference's flights again with the code on disk,
under the artefact's own spec and candidates, and requires the same sentences word for word, on the 2 s rows and on
every Δ grid, and the same refusals. A check that passes writes ``passed-<code>.json`` (`write_passed`), from a
clean checkout only; `require_conforming_labeller` asks for one before a runner labels or reads sentences for a replay.

The code is named by `labeller_code_sha256` — the logic (`io_utils.logic`: no comment, docstring or layout) of the
modules that read a flight into a sentence (`LABELLER_MODULES`) and the crossing interpolation the landing rule shares
with the harvest — which only names the record ("this code was checked"). Torch-free and free of the repository's
paths: the caller hands the git state in.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from final_approach import crossing
from ts_transformer.instructions.artefact import load_candidates, load_spec
from ts_transformer.instructions.labeller.interval import on_utc_grid
from ts_transformer.instructions.labeller.read import Reading, read_flight
from ts_transformer.instructions.labeller.records import Refused
from ts_transformer.instructions.readout import stratum
from ts_transformer.instructions.signals import FlightSignals, pack_signals, unpack_signals
from ts_transformer.instructions.words import Words
from ts_transformer.io_utils import file_sha256, logic_sha256, utc_now, write_json_atomic

#: v3 (A20, D58): the sentences' altitude words are levels above the airport elevation E.
#: v4 (A27, D70): a labelled flight's record holds its stratum.
REFERENCE_SCHEMA = "ts-instruction-conformance-reference-v4"
PASSED_SCHEMA = "ts-instruction-conformance-passed-v1"
DIRECTORY = "conformance"
#: The reference's flights: the train split, every airport alike (A3: 250 labelled train flights, seed 1337, at five
#: airports), and a few refusals of each so that a refusal that moves is found too.
SPLIT, SEED, PER_AIRPORT, REFUSED_PER_AIRPORT = "train", 1337, 50, 10
#: The row intervals whose Δ grid the reference holds beside the 2 s rows (D49: the ablation's coarser intervals, D25).
INTERVALS_S = (4.0, 8.0)
#: The modules that decide a sentence, relative to the package — what the code's name covers: since A27 the readout too,
#: whose `stratum` the sentence file stores (D70). The spec's measurement, the artefact, the figures and this module are
#: left out: they do not change a sentence.
LABELLER_MODULES = ("spec.py", "words.py", "grammar.py", "airport.py", "signals.py", "piecewise.py", "envelope.py",
                    "readout.py", "labeller/*.py")
#: Modules outside the package whose code decides a sentence: the crossing interpolation of the landing rule.
LABELLER_EXTERNAL_MODULES = (crossing,)


def labeller_code_files() -> list[tuple[str, Path]]:
    """``(label, file)`` of `LABELLER_MODULES` (package path order) then `LABELLER_EXTERNAL_MODULES` (module name)."""
    package = Path(__file__).resolve().parent
    paths = sorted({path for pattern in LABELLER_MODULES for path in package.glob(pattern)})
    return ([(path.relative_to(package).as_posix(), path) for path in paths]
            + [(module.__name__, Path(module.__file__).resolve()) for module in LABELLER_EXTERNAL_MODULES])


def labeller_code_sha256() -> str:
    return logic_sha256(labeller_code_files())


def labelled_record(reading: Reading) -> dict[str, Any]:
    """A labelled flight's record: the sentence's rows beside its words."""
    return {"dataset_id": reading.dataset_id, "status": "labelled", "runway_index": reading.runway_index,
            "stratum": stratum(reading), "capture_row": reading.capture_row, "unspecified_row": reading.unspecified_row,
            "go_around_rows": list(reading.go_around_rows), "runway_again_rows": list(reading.runway_again_rows),
            "approaches": [[a.first, a.end, a.runway_index, a.capture_row, a.unspecified_row]
                           for a in reading.approaches]}


def refused_record(dataset_id: str, refusal: Refused) -> dict[str, Any]:
    return {"dataset_id": dataset_id, "status": "refused", "reason": refusal.reason}


def sentences_of(reading: Reading, flight: FlightSignals, geometry: Any,
                 words: Words) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """A labelled flight's sentence by row interval (``"2"`` its 2 s rows, then each of `INTERVALS_S` on its UTC Δ grid),
    and the reason of every Δ grid that refuses it (D49), whose sentence is then no row."""
    step_s = words.spec.step_s
    courses = [candidate.course_deg for candidate in geometry.candidates]
    grids, refusals = {f"{step_s:g}": reading.words}, {}
    for interval in INTERVALS_S:
        try:
            grids[f"{interval:g}"] = on_utc_grid(reading.words, flight.entry_time_utc, interval, step_s,
                                                 reading.held_height_m, words, courses)[1]
        except Refused as refusal:
            refusals[f"{interval:g}"] = refusal.reason
            grids[f"{interval:g}"] = np.zeros((0, len(reading.words[0])), dtype=reading.words.dtype)
    return grids, refusals


def interval_names(words: Words) -> list[str]:
    """The reference's row intervals by name: the 2 s rows, then `INTERVALS_S`."""
    return [f"{words.spec.step_s:g}", *(f"{interval:g}" for interval in INTERVALS_S)]


def outcome(flight: FlightSignals, geometry: Any, spec: Any,
            words: Words) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """What the labeller makes of one flight: a record (status, reason or the sentence's rows, and which Δ grids refuse
    it) and its word grids by row interval (`sentences_of`)."""
    try:
        reading = read_flight(flight, geometry, spec, words)
    except Refused as refusal:
        return refused_record(flight.dataset_id, refusal), None
    grids, refusals = sentences_of(reading, flight, geometry, words)
    return {**labelled_record(reading), "interval_refusals": refusals}, grids


def draw(flights: Sequence[FlightSignals], statuses: Sequence[str]) -> list[int]:
    """The reference's flights among ``flights`` (each one's labeller status alongside): in a permutation seeded by
    `SEED`, the first `PER_AIRPORT` labelled and the first `REFUSED_PER_AIRPORT` refused of each airport, in signal
    order."""
    wanted: dict[tuple[str, str], int] = {}
    for flight in flights:
        wanted[(flight.airport, "labelled")] = PER_AIRPORT
        wanted[(flight.airport, "refused")] = REFUSED_PER_AIRPORT
    chosen = []
    for index in np.random.default_rng(SEED).permutation(len(flights)).tolist():
        key = (flights[index].airport, statuses[index])
        if wanted[key]:
            wanted[key] -= 1
            chosen.append(index)
    return sorted(chosen)


def write_reference(directory: Path, flights: Sequence[FlightSignals], records: Sequence[dict[str, Any]],
                    readings: Sequence[Reading | None], *, code_sha256: str, git: dict[str, Any]) -> Path:
    """Write ``<directory>/conformance/`` (a new directory): the drawn flights' signals, records and word grids by row
    interval (``records`` / ``readings``: every flight's record and its reading, None where it was refused, in the order
    of ``flights``; the Δ grids from `sentences_of`)."""
    target = directory / DIRECTORY
    if target.exists():
        raise FileExistsError(f"{target} exists: an artefact's reference is written once")
    spec = load_spec(directory)
    words, geometries = Words(spec), load_candidates(directory)
    chosen = draw(flights, [record["status"] for record in records])
    outcomes, grids = [], {name: [] for name in interval_names(words)}
    for i in chosen:
        if readings[i] is None:
            outcomes.append(records[i])
            continue
        sentences, refusals = sentences_of(readings[i], flights[i], geometries[flights[i].airport], words)
        outcomes.append({**records[i], "interval_refusals": refusals})
        for name, rows in grids.items():
            rows.append(sentences[name])
    arrays, meta = pack_signals([flights[i] for i in chosen])
    staging = directory / f".{DIRECTORY}.writing-{os.getpid()}"
    staging.mkdir()
    word_arrays = {}
    for name, rows in grids.items():
        word_arrays[f"words_{name}s"] = (np.concatenate(rows) if rows else np.zeros((0, 5))).astype(np.int16)
        word_arrays[f"word_offsets_{name}s"] = np.concatenate(([0], np.cumsum([len(r) for r in rows]))).astype(np.int64)
    with (staging / "reference.npz").open("xb") as handle:
        np.savez_compressed(handle, **{f"signal_{name}": value for name, value in arrays.items()}, **word_arrays)
    payload = {"schema": REFERENCE_SCHEMA, "written_utc": utc_now(), "git": git, "python": platform.python_version(),
               "spec_sha256": spec.sha256, "split": SPLIT, "seed": SEED, "per_airport": PER_AIRPORT,
               "refused_per_airport": REFUSED_PER_AIRPORT, "intervals": interval_names(words),
               "labeller_code_sha256": code_sha256, "flights": meta, "outcomes": outcomes}
    (staging / "reference.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    staging.rename(target)
    return target


def reference_sha256(directory: Path) -> str:
    """sha256 over the reference's two files, by their bytes (the reference is data, not code)."""
    digest = hashlib.sha256()
    for name in ("reference.json", "reference.npz"):
        digest.update(name.encode("utf-8") + b"\0" + file_sha256(directory / DIRECTORY / name).encode("utf-8") + b"\0")
    return digest.hexdigest()


@dataclass
class Checked:
    """A check: how many flights were read again, and every one that differs (dataset id → what)."""

    flights: int
    code_sha256: str
    git: dict[str, Any]
    mismatches: dict[str, list[str]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.mismatches


def check(directory: Path, *, git: dict[str, Any]) -> Checked:
    """Read every reference flight again with the code on disk and compare (module docstring)."""
    code = labeller_code_sha256()
    payload = json.loads((directory / DIRECTORY / "reference.json").read_text(encoding="utf-8"))
    if payload["schema"] != REFERENCE_SCHEMA:
        raise ValueError(f"{directory / DIRECTORY / 'reference.json'} is not a {REFERENCE_SCHEMA} file")
    spec = load_spec(directory)
    if payload["spec_sha256"] != spec.sha256:
        raise ValueError(f"the reference was read with spec {payload['spec_sha256'][:12]}, the artefact holds "
                         f"{spec.sha256[:12]}")
    words, geometries = Words(spec), load_candidates(directory)
    if payload["intervals"] != interval_names(words):
        raise ValueError(f"the reference holds the row intervals {payload['intervals']}, this code reads "
                         f"{interval_names(words)}")
    with np.load(directory / DIRECTORY / "reference.npz") as data:
        arrays = {name: data[name] for name in data.files}
    flights = unpack_signals({name[len("signal_"):]: value for name, value in arrays.items() if name.startswith("signal_")},
                             payload["flights"])
    checked = Checked(flights=len(flights), code_sha256=code, git=git)
    labelled = 0
    for flight, expected in zip(flights, payload["outcomes"], strict=True):
        record, grids = outcome(flight, geometries[flight.airport], spec, words)
        if record["status"] != expected["status"]:
            said = record["reason"] if record["status"] == "refused" else "a sentence"
            problems = [f"{record['status']} ({said}), the reference {expected['status']}"]
        else:
            problems = [f"{key}: {record[key]!r}, the reference {value!r}" for key, value in expected.items()
                        if record[key] != value]
        if expected["status"] == "labelled":
            for name in payload["intervals"]:
                offsets = arrays[f"word_offsets_{name}s"]
                stored = arrays[f"words_{name}s"][offsets[labelled]: offsets[labelled + 1]]
                grid = None if grids is None else grids[name]
                if grid is None or grid.shape != stored.shape or not np.array_equal(grid, stored):
                    problems.append(f"the words differ ({name} s rows)")
            labelled += 1
        if problems:
            checked.mismatches[flight.dataset_id] = problems
    return checked


def passed_path(directory: Path, code_sha256: str) -> Path:
    return directory / DIRECTORY / f"passed-{code_sha256[:12]}.json"


def write_passed(directory: Path, checked: Checked, *, git: dict[str, Any]) -> Path:
    """The record that the code on disk reads the reference as it was read (refused unless the check passed, from a
    clean checkout, with the code and the commit the check read with)."""
    if not checked.passed:
        raise ValueError(f"{len(checked.mismatches)} reference flights read otherwise: no passed record")
    if checked.git["dirty"] or git != checked.git or labeller_code_sha256() != checked.code_sha256:
        raise RuntimeError("a passed record is written from a clean checkout only, for the code the check read with")
    path = passed_path(directory, checked.code_sha256)
    write_json_atomic(path, {"schema": PASSED_SCHEMA, "written_utc": utc_now(), "git": git,
                             "python": platform.python_version(), "labeller_code_sha256": checked.code_sha256,
                             "reference_sha256": reference_sha256(directory), "flights": checked.flights})
    return path


def require_conforming_labeller(directory: Path) -> None:
    """Refused unless the labeller code on disk has read ``directory``'s reference as it was read: a passed record of
    this code against this reference."""
    code = labeller_code_sha256()
    path = passed_path(directory, code)
    command = f"python run_ts.py instruction_conformance --dir {directory} (Python {platform.python_version()})"
    if not path.exists():
        raise ValueError(f"the labeller code on disk ({code[:12]}) has not been checked against {directory.name}'s "
                         f"reference: {command}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record["schema"] != PASSED_SCHEMA or record["labeller_code_sha256"] != code
            or record["reference_sha256"] != reference_sha256(directory)):
        raise ValueError(f"{path} is not a passed record of this code against {directory.name}'s reference: {command}")
