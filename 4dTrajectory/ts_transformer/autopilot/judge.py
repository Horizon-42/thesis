"""Did the executor fly the sentence? (vocabulary §5.8; executor design §8) — three layers, read off what it flew.

Layer 1, every cycle: which limit bound, and by how much the rate the laws wanted differs from the rate the dynamics
gave (the rate each limit costs: bank → track rate, load factor and the path-angle rate limit → path-angle rate, thrust
and the stall floor → speed rate).

Layer 3, the flight, read first because it decides where the flight ends. The runway in force R and the go-around state
G are read per cycle (`executor.Flown.runway`, mode ``go_around``): a state row is judged under the R and G of the cycle
that ended at it (row 0 under the first cycle's). A threshold plane crossed is bracketed and interpolated as the harvest
does (`final_approach.crossing.bracket_fraction`). An APPROACH CROSSING is a crossing of R's plane with G false, LINED
UP (the track within the vocabulary's lined-up angle of R's course) and within the landing screen's lateral limit
(`instructions.airport.landing_cross_limit_m`: 1,000 m and half the spacing to a parallel). The events, the earliest of
which is the outcome (at one row, in `EVENT_ORDER`, the table's order):

- ``dynamics_failure``: a non-finite state, no airspeed, or a cycle the dynamics' own stall cut-off bound;
- ``ground_contact``: below R's threshold elevation while before R's threshold;
- ``crossed_too_high``: an approach crossing higher than the landing screen's height (100 m) above the threshold;
- ``crossed_off_runway``: an approach crossing at most that high, outside the runway limit (`runway_lateral_limit_m`:
  the final approach segment's full-scale half-width at the threshold, 106.7 m, never beyond half the spacing to a
  parallel);
- ``unstable_at_minimums``: an approach crossing at most that high, inside the runway limit, after a failed decision-
  altitude check — or with no DA point (the aircraft crossed above the DA);
- ``landed``: such a crossing after a DA check that passed;
- ``crossed_other_runway``: G false, the threshold plane of a candidate other than R crossed lined up with it, inside that
  runway's own limit, at any height (the flight ends there; what it flew after is not the flight's);
- ``timeout``: none of these within the flight's time limit.

A crossing that is not lined up (abeam a threshold on a downwind) is not an event, and while G is true no crossing is one,
of R or of another candidate (D33): the flight flies on.

THE DECISION-ALTITUDE CHECK (D3, D38). The approach of an approach crossing is the run of cycles before it with R the
crossed runway and G false; its DA POINT is the first state row of that run where the aircraft, on R's final (before the
threshold, lined up, within the landing screen's lateral limit), descends through the decision altitude (the row before
it above, this one at or below; `runway_data.VerticalPath.decision_height_m`). There the lateral offset must lie inside
the FAS cone at that distance (`flight_scenarios.fas_geometry`, FAA Order 8260.58D Formula 3-1-1) and the height within
the evaluation module's vertical bound (`evaluation.thresholds.RNAV_TERMINAL_VERTICAL_BOUND_M`, ±22 m, ICAO Doc 9613) of
the published glidepath, the straight-line reference (`instructions.airport.glidepath_height_m`). Neither is a parameter:
one definition with the evaluation (D38). Quality beyond that is the evaluation module's.

Layer 2, every word: the sentence's words checked against what was FLOWN, with the labeller's own checks. The flown
track is read at the data's 2 s rows (the labeller's smoothing, no landing cut: the flight is read up to where it
ended) and judged from the flown row each word was heard at: a heading word over its rows to the end of the flight
(`envelope.heading_words_inside`, §3.3), altitude and angle words in their tubes on the height above E (`labeller.vertical.tube_checks`, D58),
speed words in their spans (`labeller.speed.span_checks`). A word whose row the flight never reached is counted as
``not_reached`` and not judged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from evaluation.thresholds import RNAV_TERMINAL_VERTICAL_BOUND_M
from final_approach.crossing import bracket_fraction
from flight_scenarios.fas_geometry import course_halfwidth_m, fas_course_geometry
from ts_transformer.autopilot.executor import LIMITS, Flown
from ts_transformer.autopilot.frame import ALT, GAMMA, LAT, LON, MASS, PSI, SPEED
from ts_transformer.autopilot.runway_data import VerticalPath
from ts_transformer.autopilot.sentence import row_at
from ts_transformer.instructions import envelope
from ts_transformer.instructions.airport import (
    AirportGeometry, RunwayRelative, curvature_radius_m, glidepath_height_m, landing_cross_limit_m, relative_to_runway,
)
from ts_transformer.instructions.labeller.read import Smoothed, smooth
from ts_transformer.instructions.labeller.records import Instruction
from ts_transformer.instructions.labeller.speed import span_checks
from ts_transformer.instructions.labeller.vertical import tube_checks
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import HEADING, Words, compass_from_math_rad

OUTCOMES = ("landed", "unstable_at_minimums", "crossed_too_high", "crossed_off_runway", "crossed_other_runway",
            "ground_contact", "timeout", "dynamics_failure")
#: Which event is the outcome when two happen at the same row: the order of the design's table (§5.8).
EVENT_ORDER = ("dynamics_failure", "ground_contact", "crossed_too_high", "crossed_off_runway", "unstable_at_minimums",
               "landed", "crossed_other_runway")
CROSSINGS = ("landed", "unstable_at_minimums", "crossed_too_high", "crossed_off_runway", "crossed_other_runway")
#: Which wanted rate (track, path angle, speed) a limit costs.
LIMIT_RATE = {"bank_cap": 0, "bank_rate": 0, "load_factor": 1, "path_rate_limited": 1, "stall_floor": 2,
              "thrust_max": 2, "thrust_min": 2, "stall": 2}
assert set(LIMIT_RATE) == set(LIMITS)


@dataclass
class Verdict:
    outcome: str
    #: the state row the outcome is read at (a crossing's first row past the plane)
    end_row: int
    #: at the interpolated threshold crossing, for a crossing outcome: metres right of the centreline and above the
    #: threshold, the (fractional) state row, the candidate crossed (``runway_index``) and, for an approach crossing at
    #: most the landing screen's height, its decision-altitude check (``decision``, None without a DA point)
    crossing: dict[str, Any] | None
    #: layer 1: per limit, the cycles it bound and the mean |wanted − given| of the rate it costs then
    limits: dict[str, dict[str, float]]
    #: layer 2 (None when fewer than two flown rows are left to read)
    words: dict[str, Any] | None
    flown_rows: int

    @property
    def flew_the_sentence(self) -> bool:
        """Landed (a DA check that passed included), every word said to it inside its envelope (a word after the
        landing was never said: `not_reached`), no dynamics failure."""
        return self.outcome == "landed" and self.words is not None and self.words["all_contained"]


def flown_track(states: np.ndarray, geometry: AirportGeometry) -> dict[str, np.ndarray]:
    """A flown state sequence ``[T, 7]`` read in the airport frame, as the words read a flight."""
    e, n = geometry.frame.horizontal_from_latlon(states[:, LAT], states[:, LON])
    speed, gamma = states[:, SPEED], states[:, GAMMA]
    return {"e": e, "n": n, "height": states[:, ALT], "speed": speed, "gamma": gamma, "mass": states[:, MASS],
            "track": np.degrees(np.unwrap(np.radians(compass_from_math_rad(states[:, PSI])))),
            "ground_speed": speed * np.cos(gamma), "vertical_rate": speed * np.sin(gamma)}


def runway_lateral_limit_m(geometry: AirportGeometry, index: int, spec: VocabularySpec) -> float:
    """How far off candidate ``index``'s centreline an approach crossing may lie and still be ON the runway: its final
    approach segment's full-scale course half-width at the threshold (FAA Order 8260.58D Formula 3-1-1,
    `flight_scenarios.fas_geometry`; 350 ft = 106.7 m on every candidate here), and never beyond the harvest's limit
    (`landing_cross_limit_m`: 1,000 m, half the spacing to a parallel)."""
    return min(fas_course_geometry(geometry.candidates[index].length_m).course_width_m,
               landing_cross_limit_m(geometry, index, spec.landing_cross_limit_m, spec.parallel_course_delta_deg))


def _crossing(relative: RunwayRelative, row: int, runway_index: int) -> dict[str, Any]:
    """The crossing of ``relative``'s threshold plane between state rows ``row − 1`` and ``row``, interpolated."""
    before, right, height = relative.before_threshold_m, relative.right_of_course_m, relative.height_above_threshold_m
    fraction = bracket_fraction(-float(before[row - 1]), -float(before[row]))
    return {"cross_m": float(right[row - 1] + fraction * (right[row] - right[row - 1])),
            "height_m": float(height[row - 1] + fraction * (height[row] - height[row - 1])),
            "at_row": float(row - 1 + fraction), "runway_index": runway_index}


def decision_check(relative: RunwayRelative, rows: range, geometry: AirportGeometry, index: int, path: VerticalPath,
                   spec: VocabularySpec) -> dict[str, Any] | None:
    """The decision-altitude check on candidate ``index`` over the state ``rows`` of one approach (module docstring):
    None when the aircraft never descends through the DA on its final there."""
    before, right = relative.before_threshold_m, relative.right_of_course_m
    height, off = relative.height_above_threshold_m, relative.track_minus_course_deg
    limit = landing_cross_limit_m(geometry, index, spec.landing_cross_limit_m, spec.parallel_course_delta_deg)
    candidate = geometry.candidates[index]
    for row in rows:
        if row == 0 or not (before[row] > 0.0 and abs(off[row]) <= spec.lined_up_deg and abs(right[row]) <= limit
                            and height[row - 1] > path.decision_height_m >= height[row]):
            continue
        cone = float(course_halfwidth_m(float(before[row]), fas_course_geometry(candidate.length_m)))
        glidepath = float(glidepath_height_m(before[row], path.crossing_height_m, path.glidepath_deg,
                                             curvature_radius_m(geometry.frame.lat0, candidate.course_deg)))
        lateral_ok = abs(float(right[row])) <= cone
        vertical_ok = abs(float(height[row]) - glidepath) <= RNAV_TERMINAL_VERTICAL_BOUND_M
        return {"row": row, "right_m": float(right[row]), "cone_half_width_m": cone,
                "above_glidepath_m": float(height[row]) - glidepath, "lateral_ok": lateral_ok,
                "vertical_ok": vertical_ok, "passed": lateral_ok and vertical_ok}
    return None


def _outcome(states: np.ndarray, track: dict[str, np.ndarray], runway_cycle: np.ndarray, go_around_cycle: np.ndarray,
             stalled: np.ndarray, geometry: AirportGeometry, paths: Sequence[VerticalPath], spec: VocabularySpec,
             ) -> tuple[str, int, dict[str, Any] | None]:
    """``runway_cycle`` / ``go_around_cycle``: R and G during each cycle (one fewer than the state rows); ``stalled``
    per state row: the dynamics' stall cut-off having bound in the cycle that ended at the row. The events (module
    docstring), the earliest the outcome."""
    rows = len(states)
    runway_row = np.concatenate((runway_cycle[:1], runway_cycle))[:rows]
    go_around_row = np.concatenate((go_around_cycle[:1], go_around_cycle))[:rows]
    relatives = [relative_to_runway(track["e"], track["n"], track["track"], track["height"], candidate)
                 for candidate in geometry.candidates]
    events: list[tuple[int, str, dict[str, Any] | None]] = []
    bad = np.nonzero(~np.isfinite(states).all(axis=1) | (states[:, SPEED] <= 0.0) | stalled)[0]
    if len(bad):
        events.append((int(bad[0]), "dynamics_failure", None))
    before_r = np.array([relatives[k].before_threshold_m[r] for r, k in enumerate(runway_row)])
    height_r = np.array([relatives[k].height_above_threshold_m[r] for r, k in enumerate(runway_row)])
    ground = np.nonzero((before_r > 0.0) & (height_r < 0.0))[0]
    if len(ground):
        events.append((int(ground[0]), "ground_contact", None))
    for index, relative in enumerate(relatives):
        before = relative.before_threshold_m
        landing_limit = landing_cross_limit_m(geometry, index, spec.landing_cross_limit_m, spec.parallel_course_delta_deg)
        on_runway_m = runway_lateral_limit_m(geometry, index, spec)
        for row in (np.nonzero((before[:-1] > 0.0) & (before[1:] <= 0.0))[0] + 1).tolist():
            if go_around_row[row]:
                continue                                   # D33: no crossing is an event while G is true
            crossing = _crossing(relative, row, index)
            lined_up = abs(float(relative.track_minus_course_deg[row])) <= spec.lined_up_deg
            if runway_row[row] != index:
                if lined_up and abs(crossing["cross_m"]) <= on_runway_m:
                    events.append((row, "crossed_other_runway", crossing))
                    break
                continue
            if not lined_up or abs(crossing["cross_m"]) > landing_limit:
                continue                                   # not an approach crossing: no event
            if crossing["height_m"] > spec.landing_max_height_m:
                events.append((row, "crossed_too_high", crossing))
            elif abs(crossing["cross_m"]) > on_runway_m:
                events.append((row, "crossed_off_runway", crossing))
            else:
                start = row - 1                            # the approach: the cycles before it under R, G false
                while start > 0 and runway_cycle[start - 1] == index and not go_around_cycle[start - 1]:
                    start -= 1
                decision = decision_check(relative, range(start + 1, row + 1), geometry, index, paths[index], spec)
                crossing["decision"] = decision
                passed = decision is not None and decision["passed"]
                events.append((row, "landed" if passed else "unstable_at_minimums", crossing))
            break
    if not events:
        return "timeout", rows - 1, None
    row, kind, crossing = min(events, key=lambda event: (event[0], EVENT_ORDER.index(event[1])))
    return kind, row, crossing


def _limits(flown: Flown, index: int, states: np.ndarray, end_row: int) -> dict[str, dict[str, float]]:
    cycles = max(end_row, 1)
    dt = flown.cycle_s
    track = np.degrees(np.unwrap(np.radians(compass_from_math_rad(states[: cycles + 1, PSI]))))
    given = np.column_stack((np.diff(track), np.diff(states[: cycles + 1, GAMMA]),
                             np.diff(states[: cycles + 1, SPEED]))) / dt
    wanted = flown.wanted[index, :cycles].cpu().numpy()
    out = {}
    for name in LIMITS:
        bound = flown.limits[name][index, :cycles].cpu().numpy()
        gap = np.abs(wanted[bound, LIMIT_RATE[name]] - given[bound, LIMIT_RATE[name]])
        out[name] = {"cycles": int(bound.sum()), "wanted_minus_given": float(gap.mean()) if len(gap) else 0.0}
    out["cycles"] = {"cycles": cycles, "wanted_minus_given": 0.0}
    return out


def flown_signals(track: dict[str, np.ndarray], end_row: int, reference: FlightSignals, step_rows: int,
                  step_s: float) -> FlightSignals:
    """What was flown up to ``end_row``, at the data's step (every ``step_rows`` cycles, ``step_s`` apart). Its clock and
    day are the observed flight's (the flight starts at the same entry; its landing time names the day, not the
    replay's own landing)."""
    rows = np.arange(0, end_row + 1, step_rows)
    return FlightSignals(dataset_id=reference.dataset_id, airport=reference.airport, runway=reference.runway,
                         typecode=reference.typecode, entry_time_utc=reference.entry_time_utc,
                         landing_time_utc=reference.landing_time_utc, time_s=np.arange(len(rows)) * step_s,
                         e_m=track["e"][rows], n_m=track["n"][rows], altitude_m=track["height"][rows],
                         track_deg=track["track"][rows], ground_speed_mps=track["ground_speed"][rows],
                         vertical_rate_mps=track["vertical_rate"][rows])


@dataclass(frozen=True)
class Said:
    """How the sentence's words reached the executor: ``cycles``, one per word, the cycle it was heard at
    (``n_cycles`` for a word the flight never reached); ``moved``, the words heard, at the flown rows (the data's
    step) they were heard at — each sentence row on its own flown row (D57), so no two words of a column share one."""
    cycles: list[int]
    n_cycles: int
    moved: list[Instruction]


def words_said(flown: Flown, index: int, instructions: Sequence[Instruction], sentence_step_s: float,
               spec: VocabularySpec, end_row: int) -> Said:
    """Each word (its ``row`` a sentence row, ``sentence_step_s`` apart) at the flown row where the executor heard it,
    as the executor looked it up (`sentence.Sentences.at`, one reading of a sentence time, `sentence.row_at`): the first
    cycle that starts a sentence row whose sentence time's row reaches the word's; the flown row is that cycle on the
    data's step. Only the cycles before the outcome's state row ``end_row`` count (an outcome that does not stop the
    executor — another runway, a stall — leaves cycles after it that are not the flight's)."""
    last = min(int(flown.done_cycle[index]) + 1, max(end_row, 1))
    sentence_rows = int(round(sentence_step_s / flown.cycle_s))
    data_rows = int(round(spec.step_s / flown.cycle_s))
    step_start_rows = row_at(flown.sentence_s[index, :last].cpu().numpy(), sentence_step_s)[::sentence_rows]
    cycles = [min(int(np.searchsorted(step_start_rows, word.row)) * sentence_rows, last) for word in instructions]
    moved = [replace(word, row=cycle // data_rows) for word, cycle in zip(instructions, cycles) if cycle < last]
    return Said(cycles, last, moved)


def read_flown(flown: Flown, index: int, outcome: str, end_row: int, geometry: AirportGeometry,
               observed: FlightSignals, spec: VocabularySpec) -> Smoothed | None:
    """The flown track at the data's step, smoothed as the labeller smooths, on the rows before a crossing — where the
    labeller ends a sentence — and before a failed state; None when fewer than two rows are left."""
    last = int(flown.done_cycle[index]) + 1
    track = flown_track(flown.states[index, : last + 1].cpu().numpy(), geometry)
    read_to = end_row - 1 if outcome in (*CROSSINGS, "dynamics_failure") else end_row
    signals = flown_signals(track, read_to, observed, int(round(spec.step_s / flown.cycle_s)), spec.step_s)
    return smooth(signals, spec) if signals.n_rows >= 2 else None


@dataclass
class Outcome:
    """Layers 1 and 3 of a verdict — how the flight ended, where, and the limits."""
    outcome: str
    end_row: int
    crossing: dict[str, Any] | None
    limits: dict[str, dict[str, float]]


def outcome_of(flown: Flown, index: int, geometry: AirportGeometry, paths: Sequence[VerticalPath],
               spec: VocabularySpec) -> Outcome:
    """Flight ``index`` of ``flown``: its outcome, the state row it is read at, the crossing and the limits; ``paths``
    its airport's candidates' vertical paths (`runway_data.published_vertical_paths`)."""
    last = int(flown.done_cycle[index]) + 1
    states = flown.states[index, : last + 1].cpu().numpy()
    track = flown_track(states, geometry)
    stalled = np.concatenate(([False], flown.limits["stall"][index, :last].cpu().numpy()))
    outcome, end_row, crossing = _outcome(states, track, flown.runway[index, :last].cpu().numpy(),
                                          flown.modes["go_around"][index, :last].cpu().numpy(), stalled, geometry,
                                          paths, spec)
    return Outcome(outcome, end_row, crossing, _limits(flown, index, states, end_row))


def judge(flown: Flown, index: int, geometry: AirportGeometry, paths: Sequence[VerticalPath],
          instructions: Sequence[Instruction], sentence_step_s: float, observed: FlightSignals, spec: VocabularySpec,
          words: Words) -> Verdict:
    """Flight ``index`` of ``flown``: its outcome, its limits, and its words — ``instructions`` the sentence the
    executor flew (rows ``sentence_step_s`` apart; a heading word's ``info["target_deg"]`` the compass track it says
    under the runway in force), ``observed`` the flight's observed signals (its clock and identity)."""
    ended = outcome_of(flown, index, geometry, paths, spec)
    outcome, end_row, crossing, limits = ended.outcome, ended.end_row, ended.crossing, ended.limits
    smoothed = read_flown(flown, index, outcome, end_row, geometry, observed, spec)
    if smoothed is None:
        return Verdict(outcome, end_row, crossing, limits, None, flown_rows=end_row + 1)
    rows = len(smoothed.track_deg)
    said = words_said(flown, index, instructions, sentence_step_s, spec, end_row)
    reached = [word for word in said.moved if word.row < rows]
    never = len(instructions) - len(reached)
    headings = envelope.heading_words_inside(
        smoothed.track_deg, [(word.row, float(word.info["target_deg"])) for word in reached if word.column == HEADING],
        spec.rows_exact(spec.heading_lead_s), rows, spec.heading_tolerance_deg)
    vertical = tube_checks(reached, smoothed.distance_m, smoothed.altitude_m - geometry.elevation_m, spec, words)  # D58
    speed = span_checks(reached, smoothed.ground_speed_mps, spec, words)
    contained = (all(h["inside"] == h["rows"] for h in headings) and all(v["contained"] for v in vertical)
                 and all(v["contained"] for v in speed))
    return Verdict(outcome, end_row, crossing, limits, flown_rows=end_row + 1, words={
        "not_reached": never, "heading": headings,
        "vertical": vertical, "speed": speed, "all_contained": bool(contained)})
