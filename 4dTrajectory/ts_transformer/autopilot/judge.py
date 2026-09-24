"""Did the executor fly the sentence? (executor design §8) — three layers, read off what it flew.

Layer 1, every cycle (§8.1): which limit bound, and by how much the rate the laws wanted differs from
the rate the dynamics gave (the rate each limit costs: bank → track rate, load factor and the
path-angle rate limit → path-angle rate, thrust and the stall floor → speed rate).

Layer 3, the flight (§8.3), read first because it decides where the flight ends. The events, the first
of which is the outcome (at one row, in `EVENT_ORDER`):

- ``dynamics_failure``: a non-finite state, no airspeed, or a cycle the dynamics' own stall cut-off bound
  (§8.1: the stall floor should never let it);
- ``ground_contact``: below the pointed threshold's elevation while still before it;
- the pointed threshold plane crossed, bracketed and interpolated as the harvest does
  (`instructions.labeller.read.landing_passages`, `final_approach.crossing.bracket_fraction`):
  ``landed`` when the crossing meets the harvest's landing condition and the executor had captured the
  line, ``crossed_without_capture`` when it meets it uncaptured, ``crossed_off_runway`` when a captured
  crossing misses it (a crossing neither captured nor landing is a downwind abeam, not an event);
- ``timeout``: none of these within the flight's time limit.

Layer 2, every word (§8.2): the sentence's words checked against what was FLOWN, with the labeller's own
checks. The flown track is read at the sentence's 2 s rows through `read.admit` — the gate observed
flights pass, smoothing and the landing cut included — and judged from each word's own row:

- a heading word (vocabulary design §10.1): said at a flown row, it says where the track is `heading_lead_s`
  later, so from that row plus the lead to the next heading word's row plus the lead every row of the flown track
  lies within the heading tolerance of it (`envelope.heading_words_inside`, the labeller's own check), up to the
  clearance the executor was told or its capture, whichever came first (the capture turn is judged as its own); a
  word whose rows the lead carries past that is not judged;
- the heading word in force while the executor intercepted the line on its own at more than the heading tolerance
  from it (`lateral`: cleared on a heading that cannot reach the line) failed;
- the clearance: the executor's capture turn, from its first row to where the track is on the course (within
  the corridor's course tolerance; no turn when it already is), with the same turn check toward the course
  (§2.2: monotone, rate and bank inside §2.3's range); and once the
  flight is in the corridor (`envelope.corridor`: position AND course — the labeller's capture is the first
  row of the run inside it), every later row to the landing stays inside;
- altitude and angle words: `labeller.vertical.tube_checks`; speed words: `labeller.speed.span_checks`.

A word whose row the flown track never reaches is counted as ``not_reached`` and not judged: the sentence
is time-indexed, so a word after the flight's end was never said to the executor — a flight that landed
first flew what it was told, and one that did not land fails layer 3 anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from final_approach.crossing import bracket_fraction
from ts_transformer.autopilot.executor import LIMITS, Flown
from ts_transformer.autopilot.frame import ALT, GAMMA, LAT, LON, MASS, PSI, SPEED
from ts_transformer.instructions import envelope
from ts_transformer.instructions.airport import AirportGeometry, landing_cross_limit_m, relative_to_runway
from ts_transformer.instructions.labeller.lateral import turn_check
from ts_transformer.instructions.labeller.read import Reading, admit, landing_passages
from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.labeller.speed import span_checks
from ts_transformer.instructions.labeller.vertical import tube_checks
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import ALTITUDE, ANGLE, APPROACH, HEADING, Words, compass_from_math_rad, wrap180
from ts_transformer.instructions.words import SPEED as SPEED_WORD

OUTCOMES = ("landed", "crossed_without_capture", "crossed_off_runway", "ground_contact", "timeout",
            "dynamics_failure")
#: Which event is the outcome when two happen at the same row: a failure of the dynamics first (nothing
#: after it is flight), then the ground, then the crossing.
EVENT_ORDER = ("dynamics_failure", "ground_contact", "landed", "crossed_without_capture", "crossed_off_runway")
CROSSINGS = ("landed", "crossed_without_capture", "crossed_off_runway")
#: Which wanted rate (track, path angle, speed) a limit costs.
LIMIT_RATE = {"bank_cap": 0, "bank_rate": 0, "load_factor": 1, "path_rate_limited": 1, "stall_floor": 2,
              "thrust_max": 2, "thrust_min": 2, "stall": 2}
assert set(LIMIT_RATE) == set(LIMITS)


@dataclass
class Verdict:
    outcome: str
    #: the state row the outcome is read at (a crossing's first row past the plane)
    end_row: int
    #: at the interpolated threshold crossing, for a crossing outcome: metres right of the centreline and
    #: above the threshold, and the (fractional) state row it happened at
    crossing: dict[str, float] | None
    #: layer 1: per limit, the cycles it bound and the mean |wanted − given| of the rate it costs then
    limits: dict[str, dict[str, float]]
    #: layer 2 (None when the flown track does not pass the labeller's gate: ``refused`` says why)
    words: dict[str, Any] | None
    flown_rows: int
    refused: str | None = None

    @property
    def flew_the_sentence(self) -> bool:
        """§8.3: landed on the pointed runway, every word said to it inside its envelope (a word after the
        landing was never said: `not_reached`), no dynamics failure."""
        return self.outcome == "landed" and self.words is not None and self.words["all_contained"]


def flown_track(states: np.ndarray, geometry: AirportGeometry) -> dict[str, np.ndarray]:
    """A flown state sequence ``[T, 7]`` read in the airport frame, as the words read a flight."""
    e, n = geometry.frame.horizontal_from_latlon(states[:, LAT], states[:, LON])
    speed, gamma = states[:, SPEED], states[:, GAMMA]
    return {"e": e, "n": n, "height": states[:, ALT], "speed": speed, "gamma": gamma, "mass": states[:, MASS],
            "track": np.degrees(np.unwrap(np.radians(compass_from_math_rad(states[:, PSI])))),
            "ground_speed": speed * np.cos(gamma), "vertical_rate": speed * np.sin(gamma)}


def _outcome(states: np.ndarray, track: dict[str, np.ndarray], captured: np.ndarray, stalled: np.ndarray,
             geometry: AirportGeometry, runway_index: int,
             spec: VocabularySpec) -> tuple[str, int, dict[str, float] | None]:
    """``captured`` and ``stalled`` per state row: the law's capture, and the dynamics' stall cut-off having
    bound in the cycle that ended at the row."""
    candidate = geometry.candidates[runway_index]
    relative = relative_to_runway(track["e"], track["n"], track["track"], track["height"], candidate)
    before, right, height = relative.before_threshold_m, relative.right_of_course_m, relative.height_above_threshold_m
    events: list[tuple[int, str]] = []
    bad = np.nonzero(~np.isfinite(states).all(axis=1) | (states[:, SPEED] <= 0.0) | stalled)[0]
    if len(bad):
        events.append((int(bad[0]), "dynamics_failure"))
    ground = np.nonzero((before > 0.0) & (height < 0.0))[0]
    if len(ground):
        events.append((int(ground[0]), "ground_contact"))
    landings = set(landing_passages(relative, landing_cross_limit_m(geometry, runway_index, spec.landing_cross_limit_m,
                                                                     spec.parallel_course_delta_deg), spec))
    crossing = None
    for row in np.nonzero((before[:-1] > 0.0) & (before[1:] <= 0.0))[0] + 1:
        row = int(row)
        if row in landings or captured[row]:
            fraction = bracket_fraction(-float(before[row - 1]), -float(before[row]))
            crossing = {"cross_m": float(right[row - 1] + fraction * (right[row] - right[row - 1])),
                        "height_m": float(height[row - 1] + fraction * (height[row] - height[row - 1])),
                        "at_row": float(row - 1 + fraction)}
            kind = ("landed" if captured[row] else "crossed_without_capture") if row in landings else "crossed_off_runway"
            events.append((row, kind))
            break
    if not events:
        return "timeout", len(states) - 1, None
    row, kind = min(events, key=lambda event: (event[0], EVENT_ORDER.index(event[1])))
    return kind, row, crossing if kind in CROSSINGS else None


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


def flown_signals(track: dict[str, np.ndarray], end_row: int, reference: FlightSignals, step_rows: int) -> FlightSignals:
    """What was flown up to ``end_row``, at the sentence's step (every ``step_rows`` cycles)."""
    rows = np.arange(0, end_row + 1, step_rows)
    return FlightSignals(dataset_id=reference.dataset_id, airport=reference.airport, runway=reference.runway,
                         typecode=reference.typecode, time_s=rows * (reference.time_s[1] - reference.time_s[0]) / step_rows,
                         e_m=track["e"][rows], n_m=track["n"][rows], altitude_m=track["height"][rows],
                         track_deg=track["track"][rows], ground_speed_mps=track["ground_speed"][rows],
                         vertical_rate_mps=track["vertical_rate"][rows])


def said_at(instructions: list[Instruction], flown_rows: list[int]) -> tuple[list[Instruction], int]:
    """The sentence's words at the flown rows the executor was told them at (``info["sentence_row"]``: the row
    the observed aircraft was told it at), and how many were superseded before they flew: on a clock that runs
    ahead of the sentence (the executor cut a corner the observed aircraft flew round), two altitude, angle or
    speed words can fall on one flown row, and only the later one is flown. Heading words are kept (each is judged
    on its own rows, `envelope.heading_words_inside`)."""
    moved = [replace(word, row=row, info={**word.info, "sentence_row": word.row})
             for word, row in zip(instructions, flown_rows)]
    last = {}
    for number, word in enumerate(moved):
        last[(word.column, word.row)] = number
    kept = [word for number, word in enumerate(moved)
            if word.column not in (ALTITUDE, ANGLE, SPEED_WORD) or last[(word.column, word.row)] == number]
    return kept, len(moved) - len(kept)


@dataclass
class Outcome:
    """Layer 1 of a verdict alone — how the flight ended, where, and the limits — for a reading whose words have
    no envelope to be judged against yet (vocabulary design §10.1's comparison)."""
    outcome: str
    end_row: int
    crossing: dict[str, float] | None
    limits: dict[str, dict[str, float]]


def outcome_of(flown: Flown, index: int, geometry: AirportGeometry, runway_index: int, spec: VocabularySpec) -> Outcome:
    """Flight ``index`` of ``flown``: its outcome, the state row it is read at, the crossing and the limits."""
    last = int(flown.done_cycle[index]) + 1
    states = flown.states[index, : last + 1].cpu().numpy()
    track = flown_track(states, geometry)

    def per_row(values: np.ndarray) -> np.ndarray:
        return np.concatenate(([False], values[index, :last].cpu().numpy()))

    outcome, end_row, crossing = _outcome(states, track, per_row(flown.modes["captured"]),
                                          per_row(flown.limits["stall"]), geometry, runway_index, spec)
    return Outcome(outcome, end_row, crossing, _limits(flown, index, states, end_row))


def judge(flown: Flown, index: int, geometry: AirportGeometry, runway_index: int, reading: Reading,
          observed: FlightSignals, spec: VocabularySpec, words: Words) -> Verdict:
    """Flight ``index`` of ``flown``: its outcome, its limits, and its words (``reading`` is the
    labeller's reading of the observed flight, whose words the executor flew)."""
    last = int(flown.done_cycle[index]) + 1
    states = flown.states[index, : last + 1].cpu().numpy()
    track = flown_track(states, geometry)
    ended = outcome_of(flown, index, geometry, runway_index, spec)
    outcome, end_row, crossing, limits = ended.outcome, ended.end_row, ended.crossing, ended.limits
    step_rows = int(round(spec.step_s / flown.cycle_s))
    # the words are read on the rows before the crossing, where the labeller ends a sentence
    read_to = end_row - 1 if outcome in CROSSINGS else end_row
    try:
        flight = admit(flown_signals(track, read_to, observed, step_rows), geometry, spec)
    except Refused as refusal:
        return Verdict(outcome, end_row, crossing, limits, None, flown_rows=end_row + 1, refused=refusal.reason)
    rows = flight.signals.n_rows
    # each word at the flown row where the executor was told it: the first cycle whose sentence time reached the
    # word's step (on the time clock, the word's own row)
    # (a word the clock never reached was never said: it is past the flight's rows)
    sentence = flown.sentence_s[index, :last].cpu().numpy()
    said_cycles = [int(np.searchsorted(sentence, word.row * spec.step_s - 1e-9)) for word in reading.instructions]
    said = [(word, cycle // step_rows) for word, cycle in zip(reading.instructions, said_cycles) if cycle < len(sentence)]
    moved, superseded = said_at([word for word, _ in said], [row for _, row in said])
    reached = [word for word in moved if word.row < rows]
    never = len(reading.instructions) - len(said) + len(moved) - len(reached)

    def first_row(mode: str) -> int | None:
        # up to the outcome's row: after a crossing without capture the executor flies on, and a later mode
        # belongs to no judged row
        cycles = np.nonzero(flown.modes[mode][index, :end_row].cpu().numpy())[0]
        return min(rows - 1, int(cycles[0] + 1) // step_rows) if len(cycles) else None

    capture_row, tracking_row = first_row("captured"), first_row("tracking")
    smoothed, relative = flight.smoothed, flight.relative
    cleared_at = [word.row for word in reached if word.column == APPROACH and word.kind == "clear"]
    heading_end = min([rows, *cleared_at, *([] if capture_row is None else [capture_row])])
    headings = envelope.heading_words_inside(
        smoothed.track_deg, [(word.row, float(word.info["target_deg"])) for word in reached if word.column == HEADING],
        spec.rows_exact(spec.heading_lead_s), heading_end, spec.heading_tolerance_deg)
    capture_turn = None
    if capture_row is not None:
        # the capture turn: from the capture to where the track is on the course (within the corridor's course
        # tolerance) — none when it already is
        course = float(smoothed.track_deg[capture_row]) + float(wrap180(geometry.candidates[runway_index].course_deg
                                                                         - smoothed.track_deg[capture_row]))
        on_course = np.abs(smoothed.track_deg[capture_row:] - course) <= spec.corridor_course_tolerance_deg
        end = capture_row + int(np.argmax(on_course)) if on_course.any() else rows - 1
        check = turn_check(smoothed.track_deg, smoothed.ground_speed_mps, capture_row, end, course, spec)
        capture_turn = {"rows": end - capture_row, "progress_ok": check["progress_ok"], "rate_ok": check["rate_ok"]}
    inside = envelope.corridor(relative.right_of_course_m, relative.track_minus_course_deg, relative.before_threshold_m,
                               spec.corridor_half_width_m, spec.corridor_widening_deg,
                               spec.corridor_course_tolerance_deg)
    from_row = rows if tracking_row is None else tracking_row
    entered = np.nonzero(inside[from_row:])[0]
    corridor = inside[from_row + int(entered[0]):] if len(entered) else np.zeros(0, dtype=bool)
    cleared = any(i.column == APPROACH for i in reached if i.kind == "clear")
    vertical = tube_checks(reached, smoothed.distance_m, smoothed.altitude_m, spec, words)
    speed = span_checks(reached, smoothed.ground_speed_mps, spec, words)
    clearance_ok = (capture_turn is not None and capture_turn["progress_ok"] and capture_turn["rate_ok"]
                    and len(corridor) > 0 and bool(corridor.all()))
    # the executor left the heading word in force to intercept the line on its own (lateral law): that word failed
    off_word = int(flown.modes["intercepting_off_word"][index, :end_row].sum())
    contained = (off_word == 0 and all(h["inside"] == h["rows"] for h in headings)
                 and (not cleared or clearance_ok)
                 and all(v["contained"] for v in vertical) and all(v["contained"] for v in speed))
    return Verdict(outcome, end_row, crossing, limits, flown_rows=end_row + 1, words={
        "not_reached": never, "superseded_before_flown": superseded,
        "heading": headings, "capture_turn": capture_turn, "intercepting_off_word_cycles": off_word,
        "aim_left_tube_cycles": int(flown.modes["aim_left_tube"][index, :end_row].sum()),
        "corridor": {"cleared": cleared, "entered": bool(len(corridor)), "rows": int(len(corridor)),
                     "inside": int(corridor.sum())},
        "vertical": vertical, "speed": speed, "all_contained": bool(contained)})
