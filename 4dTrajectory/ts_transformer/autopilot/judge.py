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

- a heading word — a split turn's parts together, as the labeller judges them: its turn from the (first)
  word's row to where the flown track enters the (last) word's band, with the labeller's own turn check
  (`labeller.lateral._turn_check`; the turn's way and size are the labeller's record of the observed turn,
  so an orbit stays a 360° turn), then its hold's rows in the funnel (`read.span_funnel` over a
  `lateral.HeadingSpan` of the FLOWN track; after a turn, `hold_funnel`, its mirror with the turn's way
  given), up to the next heading word or where the executor's capture
  turn begins — as the labeller ends a hold at the capture turn; holds the labeller would not judge (after
  a turn under `turn_rate_min_from_deg`, or whose slowest turn does not finish) are not judged here either.
  A turn the executor's capture takes over before its band is reached (the clearance comes with it, and the
  vocabulary lets the capture begin at once, §2.2) is judged as the labeller judges an intercept the
  capture cuts: to its furthest progress (`labeller.lateral._intercept_end`), not "reached"; a turn cut
  before it progressed at all, or said at or after the capture (a word acting early — a sensitivity probe, or a
  generated sentence), is ``superseded`` by the capture, not judged;
- the clearance: the executor's capture turn, from its first row to where it hands over to the line, with
  the same turn check toward the course (§2.2: monotone, rate and bank inside §2.3's range); and once the
  flight is in the corridor (`envelope.corridor`: position AND course — the labeller's capture is the first
  row of the run inside it), every later row to the landing stays inside;
- altitude and angle words: `labeller.vertical.tube_checks`; speed words: `labeller.speed.span_checks`.

A word whose row the flown track never reaches is counted as ``not_reached`` and not judged: the sentence
is time-indexed, so a word after the flight's end was never said to the executor — a flight that landed
first flew what it was told, and one that did not land fails layer 3 anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from final_approach.crossing import bracket_fraction
from ts_transformer.autopilot.executor import LIMITS, Flown
from ts_transformer.autopilot.frame import ALT, GAMMA, LAT, LON, MASS, PSI, SPEED
from ts_transformer.instructions import envelope
from ts_transformer.instructions.airport import AirportGeometry, landing_cross_limit_m, relative_to_runway
from ts_transformer.instructions.labeller.lateral import HeadingSpan, _intercept_end, _turn_check
from ts_transformer.instructions.labeller.read import Admitted, Reading, admit, landing_passages, span_funnel
from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.labeller.speed import span_checks
from ts_transformer.instructions.labeller.vertical import tube_checks
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import APPROACH, HEADING, Words, compass_from_math_rad, wrap180

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


def _turn_groups(heading: list[Instruction]) -> list[list[Instruction]]:
    """Heading words grouped into the turns they fly: a split turn's parts are ONE turn (the labeller
    judges it whole, its next part issued before the previous one's band is reached)."""
    groups: list[list[Instruction]] = []
    for word in heading:
        continues = word.kind.endswith("-split") and word.info["part"] > 1
        if continues:
            groups[-1].append(word)
        else:
            groups.append([word])
    return groups


def hold_funnel(flight: Admitted, span: HeadingSpan, target_deg: float, turn_deg: float,
                spec: VocabularySpec) -> tuple[envelope.TurnEnds, envelope.HoldFunnel]:
    """MIRROR of `instructions.labeller.read.span_funnel` for a span reached through a turn, with the turn's
    signed size from the word's row GIVEN (``turn_deg``) where the labeller takes the shorter way from the
    track there: the executor may lag a split turn by more than 180° less its part when the last part is
    said, and the vocabulary measures a word from the word in force (§2.3). Equal to the labeller's for a
    turn under 180° (checked in `tests/test_autopilot.py`)."""
    row = span.word.row
    track, speed = flight.smoothed.track_deg, flight.smoothed.ground_speed_mps
    positions = np.column_stack((flight.signals.e_m, flight.signals.n_m))
    ends = envelope.turn_ends(float(track[row]), turn_deg, speed[row:], spec.step_s, spec.turn_rate_min_deg_s,
                              spec.turn_rate_max_deg_s, spec.turn_bank_max_deg, spec.heading_tolerance_deg,
                              spec.turn_start_delay_max_s)
    late = math.ceil(spec.turn_start_delay_max_s / spec.step_s)
    origin = max(row, span.hold_start - 1 - late)
    length = float(np.hypot(*np.diff(positions[origin: span.hold_end + 1], axis=0).T).sum())
    return ends, envelope.hold_funnel(positions[row] + ends.corners, target_deg, spec.heading_tolerance_deg, length)


def _heading_words(flight: Admitted, instructions: list[Instruction], turns: list[dict[str, Any]],
                   capture_row: int | None, spec: VocabularySpec, words: Words) -> list[dict[str, Any]]:
    """One result per turn (a single word, or a split turn's parts together), from its first word's row:
    the turn to where the flown track enters the last word's band, then the hold to the next heading word or
    the executor's capture (``capture_row``: None when it never captured; the holds then run to the flown
    track's end). ``turns`` are the labeller's turn records of the observed flight: their signed ``turn_deg``
    fixes the turn's way and size."""
    smoothed = flight.smoothed
    track, speed = smoothed.track_deg, smoothed.ground_speed_mps
    every = _turn_groups(sorted((i for i in instructions if i.column == HEADING), key=lambda item: item.row))
    # a heading word said at or after the executor's capture never flew: the capture had taken over
    groups = [g for g in every if capture_row is None or g[0].row < capture_row or g[0].kind == "initial"]
    last_end = flight.signals.n_rows - 1 if capture_row is None else capture_row
    ends = [group[0].row for group in groups[1:]] + [last_end]
    results = []
    for number, (group, end) in enumerate(zip(groups, ends)):
        first, last = group[0], group[-1]
        target = words.heading_deg(last.value)
        # "turn": None for the word flown from entry or a turn the capture superseded; "hold": None when no
        # row is held, a reason when the labeller would not judge it, else its rows and those in the funnel
        result: dict[str, Any] = {"row": first.row, "kind": first.kind, "words": len(group), "turn": None,
                                  "superseded": False, "hold": None}
        if first.kind == "initial":
            arrival, turn, target_unwrapped = first.row, None, target
        else:
            (record,) = [r for r in turns if r["departure_row"] == first.row and first.kind.startswith(r["kind"])]
            # the turn said so far: all of it, or the parts said (the sentence may end mid split turn)
            said = record["turn_deg"] * len(group) / record["parts"]
            shorter = float(wrap180(target - track[first.row]))
            total = shorter + 360.0 * round((said - shorter) / 360.0)
            target_unwrapped = float(track[first.row]) + total
            inside = np.abs(track[first.row: end + 1] - target_unwrapped) <= spec.heading_tolerance_deg
            arrival = first.row + int(np.argmax(inside)) if inside.any() else None
            cut = arrival is None and capture_row is not None and number == len(groups) - 1
            if cut:
                stop = _intercept_end(track, first.row, end, target_unwrapped, spec.heading_tolerance_deg)
            else:
                stop = end if arrival is None else arrival
            if cut and stop == first.row:
                result["superseded"] = True
                results.append(result)
                continue
            check = _turn_check(track, speed, first.row, stop, target_unwrapped, spec)
            turn = {"rate_min_applies": check["rate_min_applies"]}
            result["turn"] = {"reached": arrival is not None or cut, "progress_ok": check["progress_ok"],
                              "rate_ok": check["rate_ok"]}
        if arrival is None or end <= arrival:
            results.append(result)
            continue
        span = HeadingSpan(word=last, turn=turn, hold_start=arrival, hold_end=end)
        if turn is not None and not turn["rate_min_applies"]:
            result["hold"] = "not judged: after a turn under turn_rate_min_from_deg"
        else:
            if turn is None:
                ends_of_turn, funnel = span_funnel(flight, span, target, spec)
            else:
                ends_of_turn, funnel = hold_funnel(flight, span, target, target_unwrapped - float(track[last.row]),
                                                   spec)
            if ends_of_turn is not None and not ends_of_turn.finished:
                result["hold"] = "not judged: slowest turn unfinished"
            else:
                positions = np.column_stack((flight.signals.e_m, flight.signals.n_m))[arrival: end + 1]
                inside = envelope.inside_convex(positions, funnel.outline)
                result["hold"] = {"rows": int(len(inside)), "inside": int(inside.sum())}
        results.append(result)
    return results + [{"row": group[0].row, "kind": group[0].kind, "words": len(group), "turn": None,
                       "superseded": True, "hold": None} for group in every[len(groups):]]


def judge(flown: Flown, index: int, geometry: AirportGeometry, runway_index: int, reading: Reading,
          observed: FlightSignals, spec: VocabularySpec, words: Words) -> Verdict:
    """Flight ``index`` of ``flown``: its outcome, its limits, and its words (``reading`` is the
    labeller's reading of the observed flight, whose words the executor flew)."""
    last = int(flown.done_cycle[index]) + 1
    states = flown.states[index, : last + 1].cpu().numpy()
    track = flown_track(states, geometry)

    def per_row(values: np.ndarray) -> np.ndarray:
        return np.concatenate(([False], values[index, :last].cpu().numpy()))

    captured = per_row(flown.modes["captured"])
    outcome, end_row, crossing = _outcome(states, track, captured, per_row(flown.limits["stall"]), geometry,
                                          runway_index, spec)
    limits = _limits(flown, index, states, end_row)
    step_rows = int(round(spec.step_s / flown.cycle_s))
    # the words are read on the rows before the crossing, where the labeller ends a sentence
    read_to = end_row - 1 if outcome in CROSSINGS else end_row
    try:
        flight = admit(flown_signals(track, read_to, observed, step_rows), geometry, spec)
    except Refused as refusal:
        return Verdict(outcome, end_row, crossing, limits, None, flown_rows=end_row + 1, refused=refusal.reason)
    rows = flight.signals.n_rows
    reached = [i for i in reading.instructions if i.row < rows]

    def first_row(mode: str) -> int | None:
        # up to the outcome's row: after a crossing without capture the executor flies on, and a later mode
        # belongs to no judged row
        cycles = np.nonzero(flown.modes[mode][index, :end_row].cpu().numpy())[0]
        return min(rows - 1, int(cycles[0] + 1) // step_rows) if len(cycles) else None

    capture_row, tracking_row = first_row("captured"), first_row("tracking")
    headings = _heading_words(flight, reached, reading.checks["turns"], capture_row, spec, words)
    smoothed, relative = flight.smoothed, flight.relative
    capture_turn = None
    if capture_row is not None:
        end = rows - 1 if tracking_row is None else tracking_row
        course = float(smoothed.track_deg[capture_row]) + float(wrap180(geometry.candidates[runway_index].course_deg
                                                                         - smoothed.track_deg[capture_row]))
        check = _turn_check(smoothed.track_deg, smoothed.ground_speed_mps, capture_row, end, course, spec)
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
    holds = [h["hold"] for h in headings if isinstance(h["hold"], dict)]
    clearance_ok = (capture_turn is not None and capture_turn["progress_ok"] and capture_turn["rate_ok"]
                    and len(corridor) > 0 and bool(corridor.all()))
    contained = (all(h["turn"] is None or all(h["turn"].values()) for h in headings)
                 and all(h["inside"] == h["rows"] for h in holds)
                 and (not cleared or clearance_ok)
                 and all(v["contained"] for v in vertical) and all(v["contained"] for v in speed))
    return Verdict(outcome, end_row, crossing, limits, flown_rows=end_row + 1, words={
        "not_reached": len(reading.instructions) - len(reached),
        "heading": headings, "capture_turn": capture_turn,
        "corridor": {"cleared": cleared, "entered": bool(len(corridor)), "rows": int(len(corridor)),
                     "inside": int(corridor.sum())},
        "vertical": vertical, "speed": speed, "all_contained": bool(contained)})
