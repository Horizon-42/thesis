"""Post-process an optimizer solve into the neutral evaluation-input record.

The ``evaluation`` package (repo root) judges a trajectory from a file: initial /
target state plus a state list with a 1:1 ALIGNED control list. The optimizer's
native output is sparse (N piecewise-constant control segments), but the
true-dynamics rollout (``aerodynamic_model.rollout_piecewise_constant``) already
carries the ACTIVE control on every sample — so this module only MAPS those
samples onto the record schema; it never re-derives the control schedule and
never re-integrates (one source).

``final_time_s`` is the LAST rollout sample's time — for a rollout truncated by
an envelope exit that is shorter than the planned horizon, and the final state
then (correctly) fails the evaluation gates.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from aerodynamic_model.common import GeodeticState
from aerodynamic_model.rollout import RolloutSample

# Record-filename contract — the single source for every writer and glob (the optimizer
# batch in scenario_optimization.py AND ts_transformer/export.py; this module is
# casadi-free, which is what lets the torch env import it).
# NOTE: ``*_reference_eval.json`` also matches the ``*_eval.json`` glob — reference
# records survive the stale-record sweeps only because they live under REFERENCES_DIR
# and those globs are non-recursive. (evaluation/records.py's CLI default pattern
# mirrors EVAL_SUFFIX but is owned by that package's public interface.)
STATES_SUFFIX = "_states.json"
EVAL_SUFFIX = "_eval.json"
REFERENCE_EVAL_SUFFIX = "_reference_eval.json"
REFERENCES_DIR = "references"
# One observed track, shared by every reference record that quotes it. The fitted-ADS-B and
# runway-threshold datasets reference the SAME flights and differ only in `target_state`, so
# writing the track twice duplicated ~67 KB per flight per dataset for a ~200-byte
# difference. The record keeps its own identity and boundary states and points at the track
# through the contract's existing `states_ref` indirection (`evaluation.records.load_record`
# resolves it), so nothing downstream reads a reference differently.
OBSERVED_TRACKS_DIR = "observed_tracks"
OBSERVED_TRACK_SUFFIX = "_track.json"
OBSERVED_TRACK_KEY = "states"

# The reference cache contract, beside the record shapes it anchors (this module is the
# casadi-free single source both the optimizer batch and the pipeline runner import — the
# runner used to restate it as a comment-pinned mirror). v3 replaced each record's inline
# state array with a `states_ref` into the shared observed-track store above, so a v2
# manifest no longer describes what is on disk and must not be reused.
REFERENCE_CACHE_SCHEMA = "optimization-references-v3-shared-tracks"


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes — the integrity primitive of the reference cache."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observed_track_path(reference_path: Path) -> Path:
    """The shared track file a reference record quotes, from the record's own path.

    The store is a SIBLING of the per-target reference directories (see
    ``OBSERVED_TRACKS_DIR``), so the mapping is pure path arithmetic — shared by the
    writer (the optimizer batch) and every validator of the cache contract.
    """
    stem = reference_path.name.removesuffix(REFERENCE_EVAL_SUFFIX)
    return reference_path.parent.parent / OBSERVED_TRACKS_DIR / (
        stem + OBSERVED_TRACK_SUFFIX
    )


# ── Serialized precision ──────────────────────────────────────────────────────
#
# Records are JSON, so every number costs its decimal text. Written at full float repr a
# single state row is 185 bytes — `"lat":35.766821578167715` carries 17 significant digits
# for a quantity whose SOURCE resolution (ADS-B position) is metres and whose gates are
# metre-scale. Rounding to the precision below makes the timed arrays 30% smaller with a
# worst-case position shift of 0.56 mm (measured over a KRDU batch), four orders below the
# tightest thing anything downstream compares (`evaluation.arrival`'s 1 cm target check).
#
# It applies ONLY to the timed state/control ARRAYS. `initial_state` and `target_state` are
# left at full precision deliberately: those ARE what the 1 cm check measures, and a 1 cm
# budget is not somewhere to spend half a rounding step.
#
# `t` is deliberately NOT in the table. It is the one field carrying hard contracts —
# `final_time_s == states[-1]["t"]` to 1e-6, and strictly increasing offsets — and
# `ts_transformer` exports normalized clocks whose relative offsets are tiny by
# construction (a regression test pins a 3.7e-13 s horizon surviving export). Quantizing
# time would collapse those to ties for a further 7 percentage points of file size; not
# worth putting an ordering invariant on a budget.
STATE_DECIMALS = {
    "lat": 8,      # 1.1 mm
    "lon": 8,      # 1.1 mm
    "alt": 3,      # 1 mm
    "V": 4,        # 0.1 mm/s
    "psi": 9,      # 2e-9 rad = 0.05 mm over the 25 km ring
    "gamma": 9,
    "m": 3,        # 1 g
}
TIME_KEY = "t"
CONTROL_DECIMALS = {"thrust": 6, "bank_rad": 9, "load_factor": 9}


def _quantize(row: dict[str, float], decimals: dict[str, int]) -> dict[str, float]:
    """One row at the serialized precision above; unknown keys pass through untouched."""
    return {
        key: (round(value, decimals[key]) if key in decimals else value)
        for key, value in row.items()
    }


def timed_state_row(t: float, state: GeodeticState) -> dict[str, float]:
    """One timed state row of a record's ``states`` array, at serialized precision."""
    return _quantize({"t": float(t), **state_dict(state)}, STATE_DECIMALS)


def control_row(control: Any) -> dict[str, float]:
    """One row of a record's ``controls`` array, at serialized precision."""
    return _quantize(
        {key: float(value) for key, value in asdict(control).items()}, CONTROL_DECIMALS
    )


def state_dict(state: GeodeticState) -> dict[str, float]:
    """A ``GeodeticState`` in the evaluation contract's key naming."""
    return {
        "lat": state.latitude,
        "lon": state.longitude,
        "alt": state.altitude,
        "V": state.V,
        "psi": state.psi,
        "gamma": state.gamma,
        "m": state.m,
    }


def evaluation_record(
    initial: GeodeticState,
    target: GeodeticState,
    samples: Sequence[RolloutSample],
    source: dict[str, Any],
    *,
    subject: str,
) -> dict[str, Any]:
    """The solved-trajectory record: rollout states + their 1:1 aligned controls."""
    states = [timed_state_row(s.t, s.state) for s in samples]
    return {
        "source": {**source, "subject": _subject(subject)},
        "initial_state": state_dict(initial),
        "target_state": state_dict(target),
        # Read back OFF the serialized array, never recomputed from the sample: the record
        # contract requires `final_time_s == states[-1]["t"]` to 1e-6, and the array is
        # quantized (see STATE_DECIMALS). Deriving it here is what makes that structural.
        "final_time_s": states[-1]["t"],
        "states": states,
        "controls": [control_row(s.control) for s in samples],
    }


def reference_evaluation_record(
    initial: GeodeticState,
    target: GeodeticState,
    timed_states: Sequence[tuple[float, GeodeticState]],
    source: dict[str, Any],
    *,
    subject: str,
    track_ref: str | None = None,
) -> dict[str, Any]:
    """An OBSERVED track as an evaluation record (the comparison reference).

    ``timed_states`` is the track with derived kinematics (t rebased to 0 —
    ``flight_scenarios.state_samples_from_track``). Observed data carries no
    control inputs, so ``controls`` is EMPTY — the contract's marker for a
    reference record (vs the aligned 1:1 list of a solver record).

    ``track_ref`` (a path relative to the record) puts the state array in a shared file
    instead of inline; the record then carries ``states_ref``, exactly as a solved record
    does for its ``*_states.json``. Same states either way — see ``OBSERVED_TRACKS_DIR``.
    """
    states = observed_track_states(timed_states)
    record = {
        "source": {**source, "subject": _subject(subject)},
        "initial_state": state_dict(initial),
        "target_state": state_dict(target),
        # Off the serialized array, for the same reason as `evaluation_record` above — and
        # here it must match the SHARED track file the record points at, not this list.
        "final_time_s": states[-1]["t"],
        "states": [],
        "controls": [],
    }
    if track_ref is None:
        record["states"] = states
    else:
        record["states_ref"] = {"file": track_ref, "key": OBSERVED_TRACK_KEY}
    return record


def observed_track_states(
    timed_states: Sequence[tuple[float, GeodeticState]],
) -> list[dict[str, Any]]:
    """The timed observed states in the record contract's state shape."""
    return [timed_state_row(t, s) for t, s in timed_states]


def observed_track_document(
    timed_states: Sequence[tuple[float, GeodeticState]],
) -> dict[str, Any]:
    """The shared observed-track file a reference record's ``states_ref`` points into."""
    return {OBSERVED_TRACK_KEY: observed_track_states(timed_states)}


def failed_evaluation_record(
    initial: GeodeticState,
    target: GeodeticState | None,
    source: dict[str, Any],
    reason: str,
    *,
    subject: str,
) -> dict[str, Any]:
    """The unsolved-configuration record: boundary conditions kept, empty lists.

    Empty ``states``/``controls`` are the contract's marker for "no solution" —
    the evaluation batch computes the solve rate from these.
    """
    return {
        "source": {**source, "subject": _subject(subject)},
        "initial_state": state_dict(initial),
        "target_state": state_dict(target) if target is not None else None,
        "final_time_s": None,
        "states": [],
        "controls": [],
        "reason": reason,
    }


def _subject(value: str) -> str:
    if value not in ("optimized", "predicted", "observed"):
        raise ValueError(f"invalid evaluation subject {value!r}")
    return value


def summary_row(
    source: dict[str, Any],
    *,
    status: str,
    states_file: str | None,
    eval_file: str | None,
    final_time_s: float | None,
    reason: str | None,
) -> dict[str, Any]:
    """One ``summary.json`` roster row: the flight's identity + outcome.

    The manifest-only read side (``evaluation/records.py`` resolves ``results[].eval_file``)
    and the batch metrics group on these fields, so the row shape lives here — beside the
    record shapes — and both writers (the optimizer batch and ``ts_transformer``) build
    their rows through it. Writers may add columns on top; they must not rename these.
    """
    return {
        "id": source.get("id"),
        "callsign": source.get("callsign"),
        "icao24": source.get("icao24"),
        # Part of the flight's identity, not an extra: id is the callsign and repeats
        # daily, so without the landing time a row cannot name WHICH flight it is.
        "landing_time_utc": source.get("landing_time_utc"),
        "arr_airport": source.get("arr_airport"),
        "runway": source.get("runway"),
        "target_source": source.get("target_source"),
        "status": status,
        "states_file": states_file,
        "eval_file": eval_file,
        "final_time_s": final_time_s,
        "reason": reason,
    }
