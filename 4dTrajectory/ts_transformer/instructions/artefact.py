"""The sentence artefact on disk (framework document §3): one directory, written once.

``signals_<split>.npz`` + ``signals.json``   the per-step signals, where they came from, and the day split
                                             (`data.day_split`) that dealt them — test days are never here
``spec.json`` + ``measurements.json``        the vocabulary spec (with its sha) and the numbers behind it
``candidates.json``                          every airport's candidate runways and runway ends (its geometry)
``sentences_<split>.npz`` + ``labels.json``  the sentences, and every flight's outcome

A reader checks the spec's sha against the sentences it opens and refuses a mismatch — the
sentence files carry the sha they were read with. The spec also records the LABELLER's source
hash (`LABELLER_MODULES`): labelling refuses to run with a labeller other than the one
that measured the spec, and a sentence file refuses to load under a spec measured by another.
Nothing here is ever overwritten.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from final_approach import crossing
from ts_transformer.data.day_split import DEVELOPMENT_SPLITS, DaySplit, landing_day
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.labeller.read import Reading
from ts_transformer.instructions.signals import SIGNALS_SCHEMA, FlightSignals, pack_signals, unpack_signals
from ts_transformer.instructions.spec import SPEC_SCHEMA, VocabularySpec
from ts_transformer.io_utils import write_json_atomic

SENTENCES_SCHEMA = "ts-instruction-sentences-v2"
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


def write_candidates(directory: Path, geometries: dict[str, AirportGeometry]) -> None:
    write_json_atomic(_fresh(directory / "candidates.json"),
                      {"schema": CANDIDATES_SCHEMA,
                       "airports": {code: geometry.to_dict() for code, geometry in sorted(geometries.items())}})


def load_candidates(directory: Path) -> dict[str, AirportGeometry]:
    record = json.loads((directory / "candidates.json").read_text(encoding="utf-8"))
    if "schema" not in record or record["schema"] != CANDIDATES_SCHEMA:
        raise ValueError(f"{directory / 'candidates.json'} is not a {CANDIDATES_SCHEMA} file")
    return {code: AirportGeometry.from_dict(data) for code, data in record["airports"].items()}


#: The modules that decide a sentence and the spec — what the labeller's source hash covers.
#: The artefact, readout, figure and display modules are left out: changing how a sentence is
#: shown does not change the sentence.
LABELLER_MODULES = ("spec.py", "words.py", "airport.py", "signals.py", "piecewise.py", "envelope.py", "measure.py",
                    "labeller/*.py")
#: Modules outside the package whose code decides a sentence: the crossing interpolation the
#: landing rule shares with the harvest and the evaluator.
LABELLER_EXTERNAL_MODULES = (crossing,)


def labeller_source_sha256() -> str:
    """sha256 over `LABELLER_MODULES` (relative path and bytes, in path order) and then
    `LABELLER_EXTERNAL_MODULES` (module name and bytes): the code that reads a flight into a
    sentence and measures the spec."""
    package = Path(__file__).resolve().parent
    paths = sorted({path for pattern in LABELLER_MODULES for path in package.glob(pattern)})
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(package).as_posix().encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    for module in LABELLER_EXTERNAL_MODULES:
        digest.update(module.__name__.encode("utf-8") + b"\0" + Path(module.__file__).read_bytes() + b"\0")
    return digest.hexdigest()


def write_spec(directory: Path, spec: VocabularySpec, measurements: dict[str, Any], source: dict[str, Any]) -> None:
    """``source``: ``labeller_source_sha256`` and the git state the spec was measured at."""
    write_json_atomic(_fresh(directory / "spec.json"),
                      {"schema": SPEC_SCHEMA, "sha256": spec.sha256, "spec": spec.to_dict(), "source": source})
    write_json_atomic(_fresh(directory / "measurements.json"), {"spec_sha256": spec.sha256, **measurements})


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


def spec_labeller_source(directory: Path) -> str:
    """The labeller source hash the spec was measured with."""
    return _spec_record(directory)["source"]["labeller_source_sha256"]


def require_current_labeller(directory: Path) -> None:
    """Refuse to label with code other than the labeller that measured the spec."""
    recorded, current = spec_labeller_source(directory), labeller_source_sha256()
    if recorded != current:
        raise ValueError(f"{directory / 'spec.json'} was measured by labeller {recorded[:12]}; this code is "
                         f"{current[:12]} — measure a new spec in a new directory")


def write_sentences(directory: Path, split: str, spec: VocabularySpec, readings: Sequence[Reading],
                    signal_index: Sequence[int]) -> None:
    """The labelled flights' sentences, in the order given; ``signal_index`` is each one's
    position in ``signals_<split>.npz``. A sentence's words line up row for row with the first
    ``len(words)`` rows of its signals (the rows before the threshold crossing)."""
    lengths = np.array([len(r.words) for r in readings], dtype=np.int64)
    instructions = [(f, i.column, i.value, i.row) for f, r in enumerate(readings) for i in r.instructions]
    table = np.array(instructions, dtype=np.int64).reshape(-1, 4)
    np.savez_compressed(
        _fresh(directory / f"sentences_{split}.npz"),
        schema=np.array(SENTENCES_SCHEMA),
        spec_sha256=np.array(spec.sha256),
        labeller_source_sha256=np.array(labeller_source_sha256()),
        offsets=np.concatenate(([0], np.cumsum(lengths))).astype(np.int64),
        words=np.concatenate([r.words for r in readings]).astype(np.int16),
        signal_index=np.asarray(signal_index, dtype=np.int64),
        runway_index=np.array([r.runway_index for r in readings], dtype=np.int64),
        capture_row=np.array([r.capture_row for r in readings], dtype=np.int64),
        join_row=np.array([r.join_row for r in readings], dtype=np.int64),
        unspecified_row=np.array([r.unspecified_row for r in readings], dtype=np.int64),
        instruction_flight=table[:, 0], instruction_column=table[:, 1],
        instruction_value=table[:, 2], instruction_row=table[:, 3],
    )


def load_sentences(directory: Path, split: str, spec: VocabularySpec) -> dict[str, np.ndarray]:
    with np.load(directory / f"sentences_{split}.npz") as arrays:
        data = {name: arrays[name] for name in arrays.files}
    if str(data["schema"]) != SENTENCES_SCHEMA:
        raise ValueError(f"sentences_{split}.npz is not a {SENTENCES_SCHEMA} file")
    if str(data["spec_sha256"]) != spec.sha256:
        raise ValueError(f"sentences_{split}.npz was read with spec {str(data['spec_sha256'])[:12]}, "
                         f"not {spec.sha256[:12]}")
    if str(data["labeller_source_sha256"]) != spec_labeller_source(directory):
        raise ValueError(f"sentences_{split}.npz was read by labeller {str(data['labeller_source_sha256'])[:12]}, "
                         f"not the one that measured the spec")
    return data
