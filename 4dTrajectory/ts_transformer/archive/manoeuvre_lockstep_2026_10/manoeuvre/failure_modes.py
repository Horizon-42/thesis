"""How a closed-loop flight that did not cross the threshold failed (two-tier v3 §5.2, A2).

Every quantity is read in the runway's course frame from the flown record and the truth
record of one lockstep flight (`experiments/manoeuvre_lockstep.py --write-records`): the
inbound final approach course ``(cos ψ, sin ψ)`` from the target state's ψ (math-ENU, as
`data/approach_difficulty` reads it), ``to_go`` positive BEFORE the threshold along it,
``cross`` positive to the RIGHT of it (`approach_difficulty.anchor_cross_track_m`'s sign).
"Established at a row" is the package's one rule (`ESTABLISHED_CROSS_TRACK_M`,
`ESTABLISHED_TRACK_TOLERANCE_DEG`, ahead of the threshold), on the same chart track heading
`approach_difficulty` reads; "heading aligned" is the heading half of it alone, so a turn's
timing can be read before the aircraft is on the course. Note the two meanings of "established"
in one artefact: a lockstep row's reference verdict (`reference.established`) is "crossed the
threshold on the final" — what selects the flights read here — while the rows and mode names
below use the centreline rule.

A turn onto the course is an ONSET of heading alignment (an aligned row after an unaligned
one; row 0 counts when it is aligned); the final turn is the last onset, and the turn delay is
the flown path's last onset against the truth's. The failure modes, judged in this order on
the flown path (a flight takes the first that fits):

    established-short   was established at some row and the budget ended still before the
                        threshold: aligned, did not get there
    overshoot           never established and crossed the extended centreline at least once:
                        cut through the course
    no-turn             never turned onto the course after the first row and ended unaligned
                        (a downwind flown to the budget's end)
    passed-abeam        ended past the threshold's abeam line without crossing on the final
    parallel-offset     heading aligned at the end but off the course by the established
                        cross-track bound or more: turned, never captured
    other               none of the above (a turn still in progress at the budget, say)

The per-flight row carries the numbers behind the verdict (where it started and ended, when each
side first aligned its heading and first met the established rule, the final turn's onset and
the turn delay against the truth, the centreline crossings ahead of the threshold, the speed
and height error at the end), so a class can be re-read without re-flying anything.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from aerodynamic_model.common import GeodeticState
from ts_transformer.data.approach_difficulty import ESTABLISHED_CROSS_TRACK_M, ESTABLISHED_TRACK_TOLERANCE_DEG, course_frame_rows
from ts_transformer.data.channels import channels_from_states
from ts_transformer.data.coordinate_frames import ENUFrame

FAILURE_MODES = ("established-short", "overshoot", "no-turn", "passed-abeam", "parallel-offset", "other")


def course_geometry(states: Sequence[Mapping[str, float]], target: Mapping[str, float]) -> dict[str, np.ndarray]:
    """One path's rows in the course frame: ``t``, ``to_go_m``, ``cross_m``, ``heading_error_deg``
    (the chart track ``atan2(ndot, edot)`` against the course, folded to [0, 180]),
    ``established`` (bool per row), ``V`` and ``alt``. ``states`` and ``target`` are record state
    dicts (t, lat, lon, alt, V, psi, gamma, m); the projection is the data build's own
    (`channels_from_states`)."""
    if not states:
        raise ValueError("a path needs at least one state")
    frame = ENUFrame(lat0=float(target["lat"]), lon0=float(target["lon"]), alt0=float(target["alt"]))
    samples = [(float(s["t"]), GeodeticState(latitude=float(s["lat"]), longitude=float(s["lon"]), altitude=float(s["alt"]),
                                             V=float(s["V"]), psi=float(s["psi"]), gamma=float(s["gamma"]), m=float(s["m"])))
               for s in states]
    times, rows = channels_from_states(samples, frame)
    frame = course_frame_rows(rows[:, 0], rows[:, 1], rows[:, 3], rows[:, 4], float(target["psi"]))
    return {
        "t": times, "to_go_m": frame["to_go_m"], "cross_m": frame["cross_m"],
        "heading_error_deg": np.abs(frame["relative_course_deg"]), "established": frame["established"],
        "V": np.array([float(s["V"]) for s in states]), "alt": np.array([float(s["alt"]) for s in states]),
    }


def _first_time(times: np.ndarray, mask: np.ndarray) -> float | None:
    hits = np.flatnonzero(mask)
    return float(times[hits[0]]) if len(hits) else None


def alignment_onsets(mask: np.ndarray) -> np.ndarray:
    """The rows where heading alignment BEGINS: an aligned row after an unaligned one, and row 0
    when it is aligned."""
    return np.flatnonzero(mask & ~np.concatenate(([False], mask[:-1])))


def _centreline_crossings(cross: np.ndarray, to_go: np.ndarray) -> int:
    """Sign changes of the cross-track over the rows AHEAD of the threshold (behind its abeam
    line "cutting through the course" means nothing)."""
    ahead = cross[to_go > 0.0]
    signs = np.sign(ahead[ahead != 0.0])
    return int(np.count_nonzero(np.diff(signs)))


def failure_mode(*, ever_established: bool, to_go_end_m: float, crossings: int, heading_aligned_end: bool,
                 turned: bool, cross_end_m: float) -> str:
    """The first mode of `FAILURE_MODES` that fits (the module docstring's order); ``turned`` =
    an alignment onset after row 0."""
    if ever_established and to_go_end_m > 0.0:
        return "established-short"
    if not ever_established and crossings >= 1:
        return "overshoot"
    if not turned and not heading_aligned_end:
        return "no-turn"
    if to_go_end_m <= 0.0:
        return "passed-abeam"
    if heading_aligned_end and abs(cross_end_m) >= ESTABLISHED_CROSS_TRACK_M:
        return "parallel-offset"
    return "other"


def flight_failure(flown: Sequence[Mapping[str, float]], truth: Sequence[Mapping[str, float]],
                   target: Mapping[str, float]) -> dict[str, Any]:
    """One flight's failure row: the flown path against the truth from the same anchor (both
    on the record clock, t = 0 at the anchor). ``first_aligned_s`` is each side's first
    heading-aligned row (§5.2), ``first_established_s`` its first row under the full rule,
    ``final_turn_s`` the last alignment onset. The truth's speed and height at the flown end are
    interpolated on the truth's clock, held at its last row past its end."""
    f, g = course_geometry(flown, target), course_geometry(truth, target)
    aligned_f = f["heading_error_deg"] <= ESTABLISHED_TRACK_TOLERANCE_DEG
    aligned_g = g["heading_error_deg"] <= ESTABLISHED_TRACK_TOLERANCE_DEG
    t_end = float(f["t"][-1])
    onsets_f, onsets_g = alignment_onsets(aligned_f), alignment_onsets(aligned_g)
    final_turn = {"flown": float(f["t"][onsets_f[-1]]) if len(onsets_f) else None,
                  "truth": float(g["t"][onsets_g[-1]]) if len(onsets_g) else None}
    turn_delay = None if final_turn["flown"] is None or final_turn["truth"] is None else final_turn["flown"] - final_turn["truth"]
    ever_established = bool(f["established"].any())
    crossings = _centreline_crossings(f["cross_m"], f["to_go_m"])
    mode = failure_mode(
        ever_established=ever_established, to_go_end_m=float(f["to_go_m"][-1]), crossings=crossings,
        heading_aligned_end=bool(aligned_f[-1]), turned=bool((onsets_f > 0).any()), cross_end_m=float(f["cross_m"][-1]),
    )
    return {
        "mode": mode,
        "start": {"to_go_m": float(f["to_go_m"][0]), "cross_m": float(f["cross_m"][0])},
        "end": {"t_s": t_end, "to_go_m": float(f["to_go_m"][-1]), "cross_m": float(f["cross_m"][-1]),
                "side": "right" if f["cross_m"][-1] > 0.0 else "left", "heading_error_deg": float(f["heading_error_deg"][-1])},
        "ever_established": ever_established,
        "first_aligned_s": {"flown": _first_time(f["t"], aligned_f), "truth": _first_time(g["t"], aligned_g)},
        "first_established_s": {"flown": _first_time(f["t"], f["established"]), "truth": _first_time(g["t"], g["established"])},
        "final_turn_s": final_turn, "turn_delay_s": turn_delay,
        "centreline_crossings": crossings,
        "end_errors": {"speed_mps": float(f["V"][-1] - np.interp(t_end, g["t"], g["V"])),
                       "altitude_m": float(f["alt"][-1] - np.interp(t_end, g["t"], g["alt"]))},
        "truth_end": {"t_s": float(g["t"][-1]), "to_go_m": float(g["to_go_m"][-1])},
    }


def _p50(values: Sequence[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return float(np.median(kept)) if kept else None


def summarise(rows: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per mode: count, share of the rows, the p50 of every number behind the verdict, and
    the flight ids (in row order) — the table A2 reads."""
    out: dict[str, dict[str, Any]] = {}
    for mode in FAILURE_MODES:
        chosen = {key: row for key, row in rows.items() if row["mode"] == mode}
        out[mode] = {
            "n": len(chosen), "share": len(chosen) / len(rows) if rows else 0.0,
            "ever_established_share": (float(np.mean([r["ever_established"] for r in chosen.values()])) if chosen else None),
            "to_go_start_p50_m": _p50([r["start"]["to_go_m"] for r in chosen.values()]),
            "to_go_end_p50_m": _p50([r["end"]["to_go_m"] for r in chosen.values()]),
            "cross_end_abs_p50_m": _p50([abs(r["end"]["cross_m"]) for r in chosen.values()]),
            "right_side_share": (float(np.mean([r["end"]["side"] == "right" for r in chosen.values()])) if chosen else None),
            "turn_delay_p50_s": _p50([r["turn_delay_s"] for r in chosen.values()]),
            "speed_error_p50_mps": _p50([r["end_errors"]["speed_mps"] for r in chosen.values()]),
            "altitude_error_p50_m": _p50([r["end_errors"]["altitude_m"] for r in chosen.values()]),
            "flights": list(chosen),
        }
    return out


__all__ = ["FAILURE_MODES", "alignment_onsets", "course_geometry", "failure_mode", "flight_failure", "summarise"]
