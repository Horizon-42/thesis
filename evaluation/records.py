"""The evaluation input contract + file IO.

One JSON file per trajectory:

    {
      "source":        { ... provenance (id, runway, ...) ... },
      "initial_state": {"lat", "lon", "alt", "V", "psi", "gamma", "m"},
      "target_state":  {same keys} | null (unsolved records only),
      "final_time_s":  <float> | null,
      "states":   [ {"t", "lat", "lon", "alt", "V", "psi", "gamma", "m"}, ... ],
      "controls": [ {<control fields, e.g. thrust/bank_rad/load_factor>}, ... ],
      "reference_file": "references/<id>_reference_eval.json",   # optional pointer
      "reason":   "<solver failure>"        # unsolved records only
    }

``controls`` aligns 1:1 with ``states``: ``controls[i]`` is the control ACTIVE at
``states[i].t`` (zero-order hold — the sparse piecewise-constant optimizer controls
sampled onto the state grid; the terminal entry repeats the last segment's control).
A REFERENCE record (an observed ADS-B track expressed in this same contract) has
no control data: non-empty ``states`` with ``controls == []``. An UNSOLVED
configuration keeps its boundary conditions but has ``states == controls == []``
and ``final_time_s == null`` — that is how the batch solve rate is computed from
the file set alone.

``reference_file`` (optional) points at the record's observed counterpart — a
reference record as above — resolved RELATIVE to this record's own directory
(see ``reference.py`` for the comparison it enables).

Units: metres / seconds / kg; lat/lon in degrees; psi/gamma in radians with the
model heading convention (psi = 0 East, CCW); alt in metres MSL.

Validation happens here at the file boundary: list alignment, the BOUNDARY
samples' keys, the target keys, and ``final_time_s == states[-1].t``. Interior
samples' positional keys are exercised by the downstream reads themselves (a
missing key fails loudly with ``KeyError`` there) and are not re-checked here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STATE_KEYS = ("lat", "lon", "alt", "V", "psi", "gamma", "m")


@dataclass(frozen=True)
class TrajectoryRecord:
    """One trajectory's evaluation input.

    The dict/list fields hold the module-docstring schema verbatim (keys, units,
    alignment and empty-list semantics are all specified there, once):
    ``initial_state``/``target_state`` are ``STATE_KEYS -> float`` dicts,
    ``states`` samples add ``"t"``, and ``controls`` aligns 1:1 with ``states``
    (or is empty). ``path`` is where the record was loaded from (None when built
    in memory); it locates the file in reports and resolves ``reference_file``.
    """

    source: dict[str, Any]           # provenance, free-form (id, runway, icao24, …)
    initial_state: dict[str, float]
    target_state: dict[str, float] | None   # None only on unsolved records
    final_time_s: float | None
    states: list[dict[str, float]]
    controls: list[dict[str, float]]
    reference_file: str | None = None
    reason: str | None = None
    path: Path | None = field(default=None, compare=False)

    @property
    def solved(self) -> bool:
        return bool(self.states)


def record_from_dict(data: dict[str, Any], *, path: Path | None = None) -> TrajectoryRecord:
    """Build a validated record from parsed JSON (the one boundary check).

    ``data`` is one record in the module-docstring schema (parsed, not a path).
    """
    where = str(path) if path is not None else "<record>"
    states = data["states"]
    controls = data["controls"]
    # Aligned 1:1, or empty — a reference record (observed track) has no control data.
    if controls and len(controls) != len(states):
        raise ValueError(
            f"{where}: controls ({len(controls)}) must align 1:1 with states ({len(states)})"
        )
    target = data["target_state"]
    if states:
        if target is None:
            raise ValueError(f"{where}: a solved record requires target_state")
        if data["final_time_s"] is None:
            raise ValueError(f"{where}: a solved record requires final_time_s")
        for key in ("t", *STATE_KEYS):
            if key not in states[0] or key not in states[-1]:
                raise ValueError(f"{where}: state samples missing key {key!r}")
        for key in STATE_KEYS:
            if key not in target:
                raise ValueError(f"{where}: target_state missing key {key!r}")
        # One flight-time notion: the metrics read states[-1].t, the reference
        # delta reads final_time_s — the contract requires them to agree.
        if abs(float(data["final_time_s"]) - float(states[-1]["t"])) > 1e-6:
            raise ValueError(
                f"{where}: final_time_s ({data['final_time_s']}) must equal the "
                f"last state sample's t ({states[-1]['t']})"
            )
    return TrajectoryRecord(
        source=data.get("source", {}),
        initial_state=data["initial_state"],
        target_state=target,
        final_time_s=data["final_time_s"],
        states=states,
        controls=controls,
        reference_file=data.get("reference_file"),
        reason=data.get("reason"),
        path=path,
    )


def load_record(path: str | Path) -> TrajectoryRecord:
    """Read + validate one record file."""
    p = Path(path)
    return record_from_dict(json.loads(p.read_text(encoding="utf-8")), path=p)


def load_records(path: str | Path, *, pattern: str = "*_eval.json") -> list[TrajectoryRecord]:
    """One record file, or every ``pattern`` file under a directory (sorted by name)."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob(pattern))
        if not files:
            raise ValueError(f"no {pattern} files under {p}")
        return [load_record(f) for f in files]
    return [load_record(p)]
