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

from dataclasses import asdict
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
) -> dict[str, Any]:
    """The solved-trajectory record: rollout states + their 1:1 aligned controls."""
    return {
        "source": dict(source),
        "initial_state": state_dict(initial),
        "target_state": state_dict(target),
        "final_time_s": float(samples[-1].t),
        "states": [{"t": float(s.t), **state_dict(s.state)} for s in samples],
        "controls": [
            {key: float(value) for key, value in asdict(s.control).items()}
            for s in samples
        ],
    }


def reference_evaluation_record(
    initial: GeodeticState,
    target: GeodeticState,
    timed_states: Sequence[tuple[float, GeodeticState]],
    source: dict[str, Any],
) -> dict[str, Any]:
    """An OBSERVED track as an evaluation record (the comparison reference).

    ``timed_states`` is the track with derived kinematics (t rebased to 0 —
    ``flight_scenarios.state_samples_from_track``). Observed data carries no
    control inputs, so ``controls`` is EMPTY — the contract's marker for a
    reference record (vs the aligned 1:1 list of a solver record).
    """
    return {
        "source": dict(source),
        "initial_state": state_dict(initial),
        "target_state": state_dict(target),
        "final_time_s": float(timed_states[-1][0]),
        "states": [{"t": float(t), **state_dict(s)} for t, s in timed_states],
        "controls": [],
    }


def failed_evaluation_record(
    initial: GeodeticState,
    target: GeodeticState | None,
    source: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    """The unsolved-configuration record: boundary conditions kept, empty lists.

    Empty ``states``/``controls`` are the contract's marker for "no solution" —
    the evaluation batch computes the solve rate from these.
    """
    return {
        "source": dict(source),
        "initial_state": state_dict(initial),
        "target_state": state_dict(target) if target is not None else None,
        "final_time_s": None,
        "states": [],
        "controls": [],
        "reason": reason,
    }


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
