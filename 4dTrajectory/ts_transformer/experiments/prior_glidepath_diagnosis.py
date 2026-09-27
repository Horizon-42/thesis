"""Why the labelled words' replays sink below the glidepath lower edge: the executor against the observed aircraft, word by
word (prior readouts §12).

The second stage's step 0 (`prior_procedure_check`) stopped 2.5 % of the labelled words' replays below the glidepath
lower edge (`prior.procedure`). This runner flies the same kind of seeded sample and measures where the executor's height
leaves the observed aircraft's (`flight_series`). The executor's state at the start of each cycle is compared with the
observed aircraft at the row the word clock matched to that very state (`autopilot.executor.fly`: `clock.now(cycle,
state)`; `autopilot.sentence.TrackClock`: the observed point nearest it) — the nearest 2 s row, NOT interpolated: up to
1 s of flight apart, about ±4 m of height on a 3° final. Both are read against the published glidepath of the runway the
executor was flying to (the words of the cycle that brought it there, as `glidepath_stops` reads a step's end state):

- **stops** — at each stopped replay's stopping state: the executor and the observed aircraft against the glidepath, and
  the observed aircraft classed (`observed_class`: would the same check have stopped it, else how far below the
  glidepath);
- **height lost** — over the stopped replays up to the stop, how far the executor fell below the observed aircraft in
  each cycle (positive: it lost height against it), summed by the words the executor flew in that cycle: the altitude
  column's kind (a level, or "descend to land"), the angle class, and whether the lateral law had captured the centreline;
- **after the capture** — on each replay's longest run (at least `MIN_RUN_S`) of "descend to land" with one descent
  class after the capture: the executor's and the observed aircraft's mean path angle over it (height lost ÷ distance to
  go covered, each at least `MIN_RUN_M`), how far from the glidepath the observed aircraft began it, and the height the
  executor lost against it per minute;
- **at the FAF** — each replay's first state inside the FAF reached on a captured cycle: executor − observed, and each
  against the glidepath, apart for the replays that flew "descend to land" with the shallowest descent class before it;
- **what-ifs** — the same sample flown with a line or two of the vertical law changed (`WHAT_IFS`: the aim inside the
  word's tube, and where the reach of the landing is read from), each measured like the law as it is (every reading above) and on the replay gate's flights from row 0
  (`replay_words`: landed and the words inside their envelopes, the altitude column's failures apart). The changes live
  in this process only (`law_changed`): the executor's source and spec stay as measured.

"After the capture" and "at the FAF" read each replay as the executor flew it to its end, past a stop (the executor is
what is diagnosed; the stop is the check's). Writes ``--out`` (a new directory, from a clean tree): ``diagnosis.json``.
Development splits only (`SPLITS`).
"""

from __future__ import annotations

import argparse
import inspect
import math
import textwrap
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay, vertical
from ts_transformer.autopilot.executor import Flown
from ts_transformer.autopilot.judge import CROSSINGS, flown_track, outcome_of
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.sentence import row_at
from ts_transformer.experiments.prior_free_generation import (
    flight_rows, fly_reference, glidepath_stops, in_force, reference_grid,
)
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import SPLITS, load_candidates
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import ALTITUDE, ANGLE, ANGLE_LEVEL, RUNWAY, Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.prior.procedure import (
    GLIDEPATH_BELOW_M, RunwayProcedure, below_floor, published_procedures, track_tolerance_m,
)
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: v3 (2026-09-27, executor v11 milestone 3): "the law" is the v11 law (level below the glidepath after the capture), the
#: what-ifs re-expressed against it (`toward_below_glidepath` is the law up to v10). v2 (the same day): several
#: what-ifs, each read in full and on the replay gate's flights, against the v10 law.
DIAGNOSIS_SCHEMA = "ts-prior-glidepath-diagnosis-v3"
#: The shortest run of one "descend to land" class after the capture whose angles are read: its duration, s, and the
#: distance to go each of the executor and the observed aircraft covered over it, m.
MIN_RUN_S = 20.0
MIN_RUN_M = 500.0
#: The observed aircraft "on the glidepath" at a stop: at most this far below it, m.
OBSERVED_ON_GLIDEPATH_M = 30.0
#: The shallowest descent class.
SHALLOWEST = ANGLE_LEVEL + 1
#: The line of `vertical.Vertical.rate` every aim what-if changes: the aim inside the tube — after the capture level below
#: the published glidepath (``below_glidepath``, executor v11), else toward the crossing point (``toward``) — kept between
#: the class's edges.
AIM_IN_TUBE = "in_tube = torch.minimum(torch.maximum(torch.where(below_glidepath, torch.zeros_like(toward), toward),"
_BELOW_GLIDEPATH = "below_glidepath"
#: The line that decides whether the landing can still be reached from the tube (from its lower edge, at the steepest
#: class's lower edge), and the same read from the aircraft's own height.
REACH = "in_reach = (lower - to_go_m * math.tan(self.steepest_low_rad) <= admitted_high)"
OWN_REACH = "in_reach = (height - to_go_m * math.tan(self.steepest_low_rad) <= admitted_high)"
_CENTRE = "torch.minimum(nominal, on_line)"
#: What each what-if makes of the law, as ``(line, becomes)`` pairs, each line held exactly once; the aim changes after
#: the capture only (before it every what-if is the law):
WHAT_IFS = {
    # the law up to spec v10: toward the crossing point below the glidepath too (the v11 law joins it from below)
    "toward_below_glidepath": ((AIM_IN_TUBE, "in_tube = torch.minimum(torch.maximum(toward,"),),
    # the class's nominal angle, never steeper than the line to the crossing point
    "class_centre_in_tube": ((AIM_IN_TUBE, "in_tube = torch.minimum(torch.maximum("
                                           f"torch.where(line_captured, {_CENTRE}, toward),"),),
    # below it, level; on or above it, the class's nominal angle (never steeper than the line to the crossing point)
    "join_from_below_centre": ((AIM_IN_TUBE, "in_tube = torch.minimum(torch.maximum("
                                             f"torch.where({_BELOW_GLIDEPATH}, torch.zeros_like(toward), "
                                             f"torch.where(line_captured, {_CENTRE}, toward)),"),),
    # the class centre, the reach read from the aircraft's own height (it leaves the tube for the crossing point only
    # when it could no longer get down to it at the steepest class's lower edge; the tube's lower edge lags it)
    "class_centre_own_reach": ((AIM_IN_TUBE, "in_tube = torch.minimum(torch.maximum("
                                             f"torch.where(line_captured, {_CENTRE}, toward),"),
                               (REACH, OWN_REACH)),
    # below the glidepath level, on or above it the class centre, the reach from its own height
    "join_from_below_centre_own_reach": ((AIM_IN_TUBE, "in_tube = torch.minimum(torch.maximum("
                                                       f"torch.where({_BELOW_GLIDEPATH}, torch.zeros_like(toward), "
                                                       f"torch.where(line_captured, {_CENTRE}, toward)),"),
                                         (REACH, OWN_REACH)),
}


@contextmanager
def law_changed(name: str) -> Iterator[None]:
    """The vertical law with what-if ``name``'s lines replaced (`WHAT_IFS`), in this process, restored on exit (compiled at
    the law's own line numbers). Refused unless the law holds each line exactly once: the what-if is the current law less
    those lines."""
    lines, first = inspect.getsourcelines(vertical.Vertical.rate)
    source = textwrap.dedent("".join(lines))
    for line, becomes in WHAT_IFS[name]:
        if source.count(line) != 1:
            raise RuntimeError(f"vertical.Vertical.rate holds {source.count(line)} copies of {line!r}, not 1")
        source = source.replace(line, becomes)
    namespace = dict(vars(vertical))
    exec(compile("\n" * (first - 1) + source, vertical.__file__, "exec"), namespace)
    original = vertical.Vertical.rate
    vertical.Vertical.rate = namespace["rate"]
    try:
        yield
    finally:
        vertical.Vertical.rate = original


def glidepath_m(final: RunwayProcedure, d_m: np.ndarray) -> np.ndarray:
    """The published glidepath's height ``d_m`` before the threshold, m MSL (the lower edge is it less
    `GLIDEPATH_BELOW_M`, `RunwayProcedure.floor_m`)."""
    return final.crossing_m + np.asarray(d_m, dtype=np.float64) * final.glidepath_tan


def flight_series(flown: Flown, index: int, grid: np.ndarray, signal: FlightSignals, geometry: AirportGeometry,
                  finals: Sequence[RunwayProcedure], words: Words) -> dict[str, np.ndarray]:
    """Replay ``index``, one entry per cycle ``c`` up to the one its executor was done at:

    - ``*_x``: the executor's state at the start of cycle ``c`` (``states[c]``); ``*_o``: the observed aircraft at the
      row the word clock matched to that state (the observed rows from `N_LOOK` on — ``grid`` is the sentence from the
      first predicted step, `reference_grid`);
    - ``d_*``, ``gp_*``, ``faf_d``: each position's distance to go, the glidepath and the FAF under ``into``, the runway
      pointed at by the words of the cycle that brought the executor to that state (cycle ``c − 1``'s; cycle 0's at the
      start);
    - ``land``, ``angle``, ``runway``, ``captured``: what the executor flew in cycle ``c`` — the words heard at its
      step's first cycle (`fly`: every word is heard once a step) — and the lateral capture after the cycle's law."""
    step_rows = round(words.spec.step_s / flown.cycle_s)
    cycles = int(flown.done_cycle[index]) + 1
    track = flown_track(flown.states[index, :cycles].cpu().numpy(), geometry)
    force = in_force(grid)
    heard = np.minimum(row_at(flown.sentence_s[index, :cycles].cpu().numpy(), words.spec.step_s), len(force) - 1)
    flying = force[heard[(np.arange(cycles) // step_rows) * step_rows]]
    into = flying[np.maximum(np.arange(cycles) - 1, 0), RUNWAY]
    observed = slice(N_LOOK, N_LOOK + len(grid))
    e_o, n_o, h_o = (np.asarray(values[observed], dtype=np.float64)[heard]
                     for values in (signal.e_m, signal.n_m, signal.altitude_m))
    d_x, d_o, gp_x, gp_o, faf = (np.empty(cycles) for _ in range(5))
    for pointer in np.unique(into):
        at, final = into == pointer, finals[pointer]
        d_x[at], d_o[at] = final.axes(track["e"][at], track["n"][at])[0], final.axes(e_o[at], n_o[at])[0]
        gp_x[at], gp_o[at], faf[at] = glidepath_m(final, d_x[at]), glidepath_m(final, d_o[at]), final.faf_d_m
    return {"h_x": track["height"], "e_o": e_o, "n_o": n_o, "h_o": h_o, "d_x": d_x, "d_o": d_o, "gp_x": gp_x,
            "gp_o": gp_o, "faf_d": faf, "into": into, "land": flying[:, ALTITUDE] == words.altitude_land,
            "angle": flying[:, ANGLE], "runway": flying[:, RUNWAY],
            "captured": flown.modes["captured"][index, :cycles].cpu().numpy().astype(bool)}


def observed_class(one: dict[str, np.ndarray], state: int, final: RunwayProcedure, spec: VocabularySpec) -> str:
    """The observed aircraft at ``state``: stopped itself if the flown-track check (`below_floor`) would stop it there,
    else on the glidepath (at most `OBSERVED_ON_GLIDEPATH_M` below it), above the lower edge, or below the lower edge."""
    at = slice(state, state + 1)
    if below_floor(final, one["e_o"][at], one["n_o"][at], one["h_o"][at], spec)[0][0]:
        return "stopped itself"
    below = float(one["gp_o"][state] - one["h_o"][state])
    if below <= OBSERVED_ON_GLIDEPATH_M:
        return "on the glidepath"
    return "above the lower edge" if below <= GLIDEPATH_BELOW_M else "below the lower edge"


def read_stop(one: dict[str, np.ndarray], state: int, row: int, force: np.ndarray, finals: Sequence[RunwayProcedure],
              spec: VocabularySpec, words: Words) -> dict[str, Any]:
    """A stopped replay at its stopping ``state`` (`glidepath_stops`: the end of the step that flew sentence ``row``,
    whose words in force are ``force[row]``): the executor and the observed aircraft against the glidepath, their
    difference, the observed aircraft's class, the words and the capture of the stopping step. Refused where this
    runner's reading of the stop is not the check's: a state the word clock never matched (the flight's last), another
    runway, or an executor not beyond the stop line."""
    stop_line = GLIDEPATH_BELOW_M + track_tolerance_m(spec)
    if state >= len(one["h_x"]):
        raise ValueError(f"the stop at state {state} is past the last state the word clock matched")
    if int(one["into"][state]) != int(force[row, RUNWAY]):
        raise ValueError(f"the stop at state {state} reads runway {int(one['into'][state])}, the check "
                         f"{int(force[row, RUNWAY])}")
    executor_vs = float(one["h_x"][state] - one["gp_x"][state])
    if executor_vs >= -stop_line:
        raise ValueError(f"the stop at state {state} is {-executor_vs:.1f} m below the glidepath, not beyond the "
                         f"stop line {stop_line:.1f} m")
    final = finals[int(force[row, RUNWAY])]
    return {"runway": final.ident, "d_m": float(one["d_x"][state]), "executor_vs_glidepath_m": executor_vs,
            "observed_vs_glidepath_m": float(one["h_o"][state] - one["gp_o"][state]),
            "executor_minus_observed_m": float(one["h_x"][state] - one["h_o"][state]),
            "start_executor_minus_observed_m": float(one["h_x"][0] - one["h_o"][0]),
            "observed": observed_class(one, state, final, spec),
            "altitude": "descend to land" if force[row, ALTITUDE] == words.altitude_land else "level",
            "angle_class": int(force[row, ANGLE]), "captured": bool(one["captured"][state - 1])}


def height_lost(series: Sequence[dict[str, np.ndarray]], cycle_s: float) -> list[dict[str, Any]]:
    """How far the executor fell below the observed aircraft in each cycle (the change of observed − executor height from
    the cycle's start to the next's; positive: it lost height against it), summed over ``series`` by what it flew in
    that cycle (altitude kind, angle class, lateral capture), most lost first."""
    metres: dict[tuple[bool, int, bool], float] = defaultdict(float)
    cycles: Counter = Counter()
    for one in series:
        lost = -np.diff(one["h_x"] - one["h_o"])
        for key, value in zip(zip(one["land"][:-1].tolist(), one["angle"][:-1].tolist(),
                                  one["captured"][:-1].tolist()), lost.tolist()):
            metres[key] += value
            cycles[key] += 1
    return [{"altitude": "descend to land" if land else "level", "angle_class": angle, "captured": captured,
             "executor_lost_m": metres[(land, angle, captured)], "seconds": cycles[(land, angle, captured)] * cycle_s}
            for land, angle, captured in sorted(metres, key=metres.__getitem__, reverse=True)]


def longest_run(mask: np.ndarray) -> tuple[int, int] | None:
    """The first and last index of ``mask``'s longest run of True (the earliest of equal ones); None without any."""
    edges = np.diff(np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    if not len(starts):
        return None
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def run_angles(one: dict[str, np.ndarray], angle_class: int, cycle_s: float) -> dict[str, float] | None:
    """On the longest run of cycles ``a..b`` flown under "descend to land" with ``angle_class`` after the capture, to the
    runway that brought the executor to each cycle's state (so every state of the run is read under one final), from
    state ``a`` to state ``b`` (flown by cycles ``a..b − 1``, all in the run): the executor's and the observed aircraft's
    mean path angle, deg (descending positive), the observed aircraft against the glidepath at the start, m, and the
    height the executor lost against it, m/min (positive: it fell below); None for a run shorter than `MIN_RUN_S` or
    `MIN_RUN_M`."""
    run = longest_run(one["land"] & (one["angle"] == angle_class) & one["captured"] & (one["runway"] == one["into"]))
    if run is None or (run[1] - run[0]) * cycle_s < MIN_RUN_S:
        return None
    a, b = run
    covered_x, covered_o = one["d_x"][a] - one["d_x"][b], one["d_o"][a] - one["d_o"][b]
    if min(covered_x, covered_o) < MIN_RUN_M:
        return None
    gap = one["h_x"] - one["h_o"]
    return {"executor_deg": math.degrees(math.atan2(one["h_x"][a] - one["h_x"][b], covered_x)),
            "observed_deg": math.degrees(math.atan2(one["h_o"][a] - one["h_o"][b], covered_o)),
            "observed_vs_glidepath_m": float(one["h_o"][a] - one["gp_o"][a]),
            "executor_lost_m_per_min": float(gap[a] - gap[b]) / ((b - a) * cycle_s) * 60.0}


def at_faf(one: dict[str, np.ndarray]) -> dict[str, Any] | None:
    """The first state inside the FAF reached on a captured cycle: executor − observed and each against the glidepath,
    m, and whether "descend to land" with the shallowest class was flown before it; None if never."""
    captured_into = np.concatenate([[False], one["captured"][:-1]])
    inside = np.flatnonzero(captured_into & (one["d_x"] > 0.0) & (one["d_x"] <= one["faf_d"]))
    if not len(inside):
        return None
    s = int(inside[0])
    return {"executor_minus_observed_m": float(one["h_x"][s] - one["h_o"][s]),
            "executor_vs_glidepath_m": float(one["h_x"][s] - one["gp_x"][s]),
            "observed_vs_glidepath_m": float(one["h_o"][s] - one["gp_o"][s]),
            "shallowest_before": bool((one["land"][:s] & (one["angle"][:s] == SHALLOWEST)).any())}


def after_capture(angle_class: int, read: Sequence[dict[str, float]], words: Words) -> dict[str, Any]:
    """One descent class's `run_angles` over the replays whose run was read."""
    return {"angle_class": angle_class, "angle_deg": list(words.angle_bounds(angle_class)),
            "nominal_deg": words.angle_deg(angle_class), "replays": len(read),
            **{name: _spread([r[name] for r in read])
               for name in ("executor_deg", "observed_deg", "observed_vs_glidepath_m", "executor_lost_m_per_min")}}


def _spread(values: Sequence[float]) -> dict[str, float] | None:
    if not len(values):
        return None
    return {"n": len(values), **{f"p{q}": float(np.percentile(values, q)) for q in (10, 50, 90)},
            "mean": float(np.mean(values))}


def fly_sample(batch: replay.Batch, words: Words, params: ExecutorParams,
               procedures: dict[str, tuple[RunwayProcedure, ...]], chunk: int) -> Iterator[tuple[Any, ...]]:
    """``batch`` flown on its labelled words from the first predicted step, ``chunk`` flights at a time, as the stage-0
    check flies it: each chunk's flights, flown record, sentences, finals, glidepath stops and outcome rows."""
    cpu = torch.device("cpu")
    for start in range(0, len(batch.readings), chunk):
        part = replay.subset(batch, list(range(start, min(start + chunk, len(batch.readings)))))
        flown = fly_reference(part, words, params, device=cpu)
        grids = [reference_grid(reading.words) for reading in part.readings]
        finals = [procedures[geometry.code] for geometry in part.geometries]
        stops = glidepath_stops(flown, grids, part.geometries, finals, words)
        rows = flight_rows(part, flown, grids, words, "labelled", [None] * len(grids), None)
        yield part, flown, grids, finals, stops, rows
        print(f"  flown {min(start + chunk, len(batch.readings))}/{len(batch.readings)}", flush=True)


def outcome_record(part: replay.Batch, flown: Flown, j: int, row: dict[str, Any], stopped: bool,
                   spec: VocabularySpec) -> dict[str, Any]:
    """One replay's stratum, outcome (the judge's, `flight_rows`: the one it has without the stop), whether the edge
    stopped it, and its height over the threshold where it crossed (None without a crossing; the runway the judge
    read, `flight_rows`' last pointer)."""
    outcome = outcome_of(flown, j, part.geometries[j], row["last_runway"], spec)
    if outcome.outcome != row["outcome"]:
        raise ValueError(f"{row['dataset_id']}: outcome {outcome.outcome}, flight_rows {row['outcome']}")
    return {"stratum": row["stratum"], "outcome": row["outcome"], "stopped": stopped,
            "crossing_height_m": outcome.crossing["height_m"] if outcome.outcome in CROSSINGS else None}


def outcome_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stops and outcomes over ``records``, and the height over the threshold at each kind of crossing."""
    stopped = [r for r in records if r["stopped"]]
    return {"replays": len(records), "stopped": len(stopped), "stopped_share": len(stopped) / len(records),
            "outcomes": dict(Counter(r["outcome"] for r in records)),
            "landed_not_stopped_share": sum(r["outcome"] == "landed" and not r["stopped"] for r in records) / len(records),
            "stopped_by_stratum": dict(Counter(r["stratum"] for r in stopped)),
            "replays_by_stratum": dict(Counter(r["stratum"] for r in records)),
            "crossing_height_m": {kind: _spread([r["crossing_height_m"] for r in records if r["outcome"] == kind])
                                  for kind in CROSSINGS}}


def diagnose(batch: replay.Batch, words: Words, params: ExecutorParams,
             procedures: dict[str, tuple[RunwayProcedure, ...]], chunk: int) -> dict[str, Any]:
    """The current executor's replays measured against the observed aircraft (module docstring)."""
    step_rows = round(words.spec.step_s / params.cycle_s)
    records, stopped_series, stopped_rows, faf, runs = [], [], [], [], defaultdict(list)
    for part, flown, grids, finals, stops, rows in fly_sample(batch, words, params, procedures, chunk):
        for j, row in enumerate(rows):
            one = flight_series(flown, j, grids[j], part.signals[j], part.geometries[j], finals[j], words)
            stopped = bool(stops.step[j] >= 0)
            records.append(outcome_record(part, flown, j, row, stopped, words.spec))
            faf.append(at_faf(one))
            for angle_class in range(SHALLOWEST, words.n_descent + 1):
                runs[angle_class].append(run_angles(one, angle_class, params.cycle_s))
            if not stopped:
                continue
            state = (int(stops.step[j]) + 1) * step_rows                  # `glidepath_stops`' stopping state
            stopped_rows.append({"dataset_id": row["dataset_id"], "airport": row["airport"], "stratum": row["stratum"],
                                 **read_stop(one, state, int(stops.row[j]), in_force(grids[j]), finals[j],
                                             words.spec, words)})
            stopped_series.append({name: values[: state + 1] for name, values in one.items()})
    reached = [r for r in faf if r is not None]
    stopped_at = [r["stopped"] for r, f in zip(records, faf) if f is not None]

    def faf_group(shallowest: bool) -> dict[str, Any]:
        group = [(f, s) for f, s in zip(reached, stopped_at) if f["shallowest_before"] == shallowest]
        return {"replays": len(group), "stopped": sum(s for _, s in group),
                "executor_minus_observed_m": _spread([f["executor_minus_observed_m"] for f, _ in group])}

    return {
        "summary": outcome_summary(records),
        "stops": {
            "stop_line_below_glidepath_m": GLIDEPATH_BELOW_M + track_tolerance_m(words.spec),
            "observed": dict(Counter(r["observed"] for r in stopped_rows)),
            "captured": sum(r["captured"] for r in stopped_rows),
            **{name: _spread([r[name] for r in stopped_rows])
               for name in ("executor_vs_glidepath_m", "observed_vs_glidepath_m", "executor_minus_observed_m",
                            "start_executor_minus_observed_m", "d_m")}},
        "height_lost": height_lost(stopped_series, params.cycle_s),
        "after_capture": [after_capture(angle_class, [r for r in runs[angle_class] if r is not None], words)
                          for angle_class in sorted(runs)],
        "at_faf": {"replays": len(reached),
                   **{name: _spread([f[name] for f in reached])
                      for name in ("executor_minus_observed_m", "executor_vs_glidepath_m", "observed_vs_glidepath_m")},
                   "shallowest_before": faf_group(True), "not_shallowest_before": faf_group(False)},
        "stopped_replays": stopped_rows,
    }


def replay_words(batch: replay.Batch, words: Words, params: ExecutorParams, chunk: int) -> dict[str, Any]:
    """The replay gate's flights (`replay.fly_batch`: every sentence from its row 0, judged): `replay.summary` — landed,
    the words inside their envelopes, the failures by check (the altitude column's apart)."""
    verdicts = []
    for start in range(0, len(batch.readings), chunk):
        part = replay.subset(batch, list(range(start, min(start + chunk, len(batch.readings)))))
        verdicts += replay.fly_batch(part, params, words, device=torch.device("cpu"))[1]
    return replay.summary(verdicts)


def what_if(name: str, batch: replay.Batch, words: Words, params: ExecutorParams,
            procedures: dict[str, tuple[RunwayProcedure, ...]], chunk: int) -> dict[str, Any]:
    """The same sample flown under what-if ``name``: read as the law is (`diagnose`), and on the replay gate's flights."""
    with law_changed(name):
        return {"change": [{"replaced": line, "by": becomes} for line, becomes in WHAT_IFS[name]],
                **diagnose(batch, words, params, procedures, chunk),
                "replay": replay_words(batch, words, params, chunk)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--per-airport", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--what-ifs", nargs="+", choices=sorted(WHAT_IFS), default=sorted(WHAT_IFS))
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    instructions, executor_dir, out = map(resolved, (args.instructions, args.executor, args.out))
    if out.exists():
        parser.error(f"{out} exists; a diagnosis is never overwritten")
    git = git_state()
    if git["dirty"]:
        parser.error("the diagnosis runs from a clean tree")
    started = time.perf_counter()
    params, record, words = replay.open_executor(executor_dir, instructions)
    procedures = published_procedures(load_candidates(instructions))
    batch = replay.draw(instructions, args.split, words.spec, words, per_airport=args.per_airport, seed=args.seed)
    print("the executor as measured:", flush=True)
    current = {**diagnose(batch, words, params, procedures, args.chunk),
               "replay": replay_words(batch, words, params, args.chunk)}
    variants = {}
    for name in args.what_ifs:
        print(f"the what-if {name}:", flush=True)
        variants[name] = what_if(name, batch, words, params, procedures, args.chunk)
    out.mkdir(parents=True)
    write_json_atomic(out / "diagnosis.json", {
        "schema": DIAGNOSIS_SCHEMA, "written_utc": utc_now(), "git": git, "split": args.split,
        "instructions": str(instructions), "executor": {"directory": str(executor_dir), "sha256": record["sha256"]},
        "constants": {"min_run_s": MIN_RUN_S, "min_run_m": MIN_RUN_M, "observed_on_glidepath_m": OBSERVED_ON_GLIDEPATH_M,
                      "glidepath_below_m": GLIDEPATH_BELOW_M, "track_tolerance_m": track_tolerance_m(words.spec),
                      "observed_row": "the 2 s row the word clock matched to the executor's state, not interpolated"},
        "replay_sample": batch.drawn, "current": current, "what_ifs": variants,
        "elapsed_s": time.perf_counter() - started})
    for name, read in (("as measured", current), *variants.items()):
        summary, gate = read["summary"], read["replay"]
        angles = "  ".join(f"class {a['angle_class']}: {a['executor_deg']['p50']:.2f}° vs {a['observed_deg']['p50']:.2f}°"
                           for a in read["after_capture"] if a["executor_deg"] is not None)
        print(f"{name}: stopped {summary['stopped']}/{summary['replays']} ({summary['stopped_share']:.2%}), outcomes "
              f"{summary['outcomes']}; replay gate landed {gate['landed_share']:.4f}, words {gate['words_inside_share']:.4f}; "
              f"after the capture, executor vs observed p50 — {angles}")
    print(f"at the stops, the observed aircraft: {current['stops']['observed']}  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
