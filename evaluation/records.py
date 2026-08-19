"""Validated trajectory-evaluation records and strict JSON input.

Every derived record must identify its subject explicitly.  All numeric values
used by the evaluator are checked at this boundary so IEEE ``NaN``/infinity
cannot turn unordered comparisons into false passes or leak into reports.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Literal, get_args

STATE_KEYS = ("lat", "lon", "alt", "V", "psi", "gamma", "m")

# The subject vocabulary, defined once: the type annotation and the runtime check
# below are the same tuple, so a new subject cannot be accepted by one and not the
# other.
Subject = Literal["optimized", "predicted", "observed"]
SUBJECTS = get_args(Subject)


@dataclass(frozen=True)
class TrajectoryRecord:
    source: dict[str, Any]
    initial_state: dict[str, float]
    target_state: dict[str, float] | None
    final_time_s: float | None
    states: list[dict[str, float]]
    controls: list[dict[str, float]]
    reference_file: str | None = None
    reason: str | None = None
    path: Path | None = field(default=None, compare=False)

    @property
    def solved(self) -> bool:
        return bool(self.states)

    @property
    def airport(self) -> str:
        """The arrival airport ICAO code, upper-cased.

        Required on every record: it selects the runway data a verdict is measured
        against, and producers that cannot name it (``evaluation_export`` copies
        whatever the scenario carried) would otherwise be graded against nothing.
        """
        code = self.source.get("arr_airport")
        if not isinstance(code, str):
            raise ValueError(
                f"record {self.path or self.source.get('id')!r} requires "
                "source.arr_airport"
            )
        return code.upper()


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{path} must be finite, got {value!r}")
    return numeric


def _state(value: Any, path: str, *, timed: bool) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    for key in (("t", *STATE_KEYS) if timed else STATE_KEYS):
        if key not in value:
            raise ValueError(f"{path} missing key {key!r}")
        _number(value[key], f"{path}.{key}")


def record_from_dict(data: dict[str, Any], *, path: Path | None = None) -> TrajectoryRecord:
    """Build a validated record from an already parsed JSON object."""
    where = str(path) if path is not None else "<record>"
    if not isinstance(data, dict):
        raise ValueError(f"{where}: record must be an object")
    source = data.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"{where}: source must be an object")
    subject = source.get("subject")
    if subject not in SUBJECTS:
        raise ValueError(
            f"{where}: source.subject must be one of {SUBJECTS}, got {subject!r}"
        )

    try:
        initial = data["initial_state"]
        target = data["target_state"]
        states = data["states"]
        controls = data["controls"]
    except KeyError as exc:
        raise ValueError(f"{where}: missing required field {exc.args[0]!r}") from exc
    _state(initial, f"{where}: initial_state", timed=False)
    if target is not None:
        _state(target, f"{where}: target_state", timed=False)
    if not isinstance(states, list):
        raise ValueError(f"{where}: states must be an array")
    if not isinstance(controls, list):
        raise ValueError(f"{where}: controls must be an array")
    for index, sample in enumerate(states):
        _state(sample, f"{where}: states[{index}]", timed=True)
    for index, control in enumerate(controls):
        if not isinstance(control, dict):
            raise ValueError(f"{where}: controls[{index}] must be an object")
        for key, value in control.items():
            _number(value, f"{where}: controls[{index}].{key}")

    if controls and len(controls) != len(states):
        raise ValueError(
            f"{where}: controls ({len(controls)}) must align 1:1 with states ({len(states)})"
        )
    final_time = data.get("final_time_s")
    if states:
        if target is None:
            raise ValueError(f"{where}: a solved record requires target_state")
        if final_time is None:
            raise ValueError(f"{where}: a solved record requires final_time_s")
        final_numeric = _number(final_time, f"{where}: final_time_s")
        if abs(final_numeric - float(states[-1]["t"])) > 1e-6:
            raise ValueError(
                f"{where}: final_time_s ({final_time}) must equal the last state "
                f"sample's t ({states[-1]['t']})"
            )
    # An unsolved record may still carry the target it was asked for (validated above);
    # what it may not carry is a flight time, since nothing was flown.
    elif final_time is not None:
        raise ValueError(f"{where}: an unsolved record requires final_time_s == null")

    reference_file = data.get("reference_file")
    if reference_file is not None and not isinstance(reference_file, str):
        raise ValueError(f"{where}: reference_file must be a string")
    return TrajectoryRecord(
        source=dict(source),
        initial_state=initial,
        target_state=target,
        final_time_s=final_time,
        states=states,
        controls=controls,
        reference_file=reference_file,
        reason=data.get("reason"),
        path=path,
    )


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {token!r}")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_record(path: str | Path) -> TrajectoryRecord:
    """Read and validate one record, including any ``states_ref`` source."""
    p = Path(path)
    data = _load_json(p)
    states_ref = data.get("states_ref")
    if states_ref is not None:
        if not isinstance(states_ref, dict):
            raise ValueError(f"{p}: states_ref must be an object")
        file_name, key = states_ref.get("file"), states_ref.get("key")
        if not isinstance(file_name, str) or not isinstance(key, str):
            raise ValueError(f"{p}: states_ref requires string file and key")
        source_path = p.parent / file_name
        source = _load_json(source_path)
        states = source.get(key)
        if not isinstance(states, list):
            raise ValueError(f"{p}: {source_path} has no state list {key!r}")
        start = states_ref.get("start_index", 0)
        stop = states_ref.get("stop_index")
        effective_stop = len(states) if stop is None else stop
        if (
            isinstance(start, bool) or not isinstance(start, int) or start < 0
            or start >= len(states) or isinstance(effective_stop, bool)
            or not isinstance(effective_stop, int) or effective_stop <= start
            or effective_stop > len(states)
        ):
            raise ValueError(f"{p}: invalid states_ref slice [{start!r}, {stop!r}]")
        data = dict(data)
        data["states"] = states[start:effective_stop]
    return record_from_dict(data, path=p)


def record_files(path: str | Path) -> list[Path]:
    """The record files at ``path``: one loose record, or a batch summary's roster.

    Split out of :func:`load_records` so a caller can walk a batch without holding it: the
    roster is a few hundred bytes per flight, the records it names are ~1 MB each.
    """
    p = Path(path)
    if not p.is_dir():
        return [p]
    manifest = p / "summary.json"
    if not manifest.exists():
        raise ValueError(
            f"{p} has no summary.json manifest; pass a record file for a loose record"
        )
    rows = _load_json(manifest).get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{manifest} lists no records (empty batch)")
    files: list[Path] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("eval_file"), str):
            raise ValueError(f"{manifest}: results[{index}].eval_file must be a string")
        files.append(p / row["eval_file"])
    return sorted(files)


def roster_context_keys(path: str | Path) -> list[tuple[str, str]] | None:
    """``(airport, runway)`` pairs from a batch roster, or ``None`` for a loose record.

    ``summary_row`` carries both, so the assessment contexts a batch needs can be resolved
    from the roster alone — which is what lets the records themselves be streamed instead
    of materialized just to collect their airport codes.
    """
    p = Path(path)
    if not p.is_dir():
        return None
    rows = _load_json(p / "summary.json").get("results")
    if not isinstance(rows, list):
        return None
    keys: list[tuple[str, str]] = []
    for row in rows:
        airport, runway = row.get("arr_airport"), row.get("runway")
        if not isinstance(airport, str) or not isinstance(runway, str):
            # A roster written before summary_row carried them: fall back to the records.
            return None
        keys.append((airport.upper(), runway))
    return keys


def iter_records(path: str | Path) -> Iterator[TrajectoryRecord]:
    """Yield the rostered records one at a time, holding only one at a time.

    A 2,000-flight batch is ~1 MB of resolved state per record (``load_record`` follows
    ``states_ref`` into the full ``*_states.json``), so materializing the list peaked at
    ~15 GB on the largest airport. Every consumer here reduces record-by-record.
    """
    for file in record_files(path):
        yield load_record(file)


def load_records(path: str | Path) -> list[TrajectoryRecord]:
    """Load one record file or the records rostered by a batch summary.

    Kept for callers that genuinely need random access (``evaluation.visualize`` samples
    track overlays); the batch metrics path streams via :func:`iter_records`.
    """
    return [load_record(file) for file in record_files(path)]
