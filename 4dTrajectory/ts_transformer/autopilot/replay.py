"""A batch of labelled flights flown from their sentences (executor design §11; design §14.2 A6): who is flown, their
inputs, the flight and the verdicts — shared by the spec's conformance reference and the replay readout.

Who is flown (§11), by `group_of`: an identified type that publishes an approach speed ("unspecified" is
flown at it) is flown — on its own dynamics (`OWN`: the identified type is the dynamics type) or on a
stand-in's (`STAND_IN`: the airframe `aircraft/performance_index.json` substitutes for it, reported, never
gated: its errors are the stand-in's aerodynamics, and its "unspecified" speed is its type's as published,
unscaled — the flight's mass is the stand-in's, `flight_approach_ias_mps`); a flight with no identified
type, with no aircraft dynamics (the index excludes its type, or nothing models it; since 2026-09-24 never
an A320 in its place, C31), or whose type publishes no approach speed, is counted, not flown. The sample is a seeded
permutation of a split's labelled flights, read in order until each airport holds ``per_airport`` flights of
the asked groups (0: every one) — the pool, the count read and the exclusions are returned with it.

Every flight is re-read with the labeller and must reproduce its stored sentence (the words grid and the
runway), the export's rule: the executor flies the reading of the flight, not a grid that merely looks like it.

THE ROW INTERVAL (design §4.8, D11, D25). A batch is flown at one row interval Δ (the data's 2 s, or 4 or 8 s): each
sentence is put on the Δ grid (`instructions.labeller.interval`: the rows on UTC multiples of Δ) and flown from its first
Δ row — the flight's state there, the observed rows from there on (the word clock's, the judge's reference) — with
`Sentence` holding the grid and its words as instructions on the Δ rows. At Δ = 2 s the first row is row 0 and the
sentence is the labelled one.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_transformer.autopilot.executor import Flown, fly
from ts_transformer.autopilot.flights import FlightInputs, flight_inputs, rebuild_series
from ts_transformer.autopilot.frame import AirportCharts
from ts_transformer.autopilot.executor import GO_AROUND_EXTRA_S
from ts_transformer.autopilot.judge import Outcome, Verdict, flown_track, judge
from ts_transformer.autopilot.lateral import Runways
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.runway_data import VerticalPath, published_vertical_paths
from ts_transformer.autopilot.sentence import DistanceClock, Sentences, TimeClock, TrackClock
from ts_transformer.autopilot.spec import load_spec, require_conforming_executor
from ts_transformer.autopilot.speed import approach_speed_ias_mps
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.instructions import artefact
from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway
from ts_transformer.instructions.artefact import load_candidates, load_sentences, load_signals
from ts_transformer.instructions.conformance import require_conforming_labeller
from ts_transformer.instructions.labeller.interval import first_interval_row, later_utc, on_interval
from ts_transformer.instructions.labeller.read import Reading, read_flight
from ts_transformer.instructions.labeller.records import Instruction, Refused
from ts_transformer.instructions.signals import ROW_FIELDS, FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import HEADING, RUNWAY, RUNWAY_GO_AROUND, UNCHANGED, Words

#: Rebuilt at a time while drawing the sample (a rebuild opens the flights' tracks).
DRAW_CHUNK = 200
OWN, STAND_IN = "own dynamics", "stand-in dynamics"


@dataclass(frozen=True)
class Sentence:
    """What one flight is flown on: its sentence on the batch's row interval (``grid`` ``[N, 5]``), its words as
    instructions on those rows (a heading word's ``info["target_deg"]`` the track it says under the runway in force at
    its row), and the 2 s row of the observed flight its row 0 is (``first_row``)."""
    grid: np.ndarray
    instructions: list[Instruction]
    first_row: int

    @property
    def go_arounds(self) -> int:
        return int((self.grid[:, RUNWAY] == RUNWAY_GO_AROUND).sum())


def sentence_on_interval(reading: Reading, signals: FlightSignals, interval_s: float, geometry: AirportGeometry,
                         words: Words) -> Sentence:
    """A labelled flight's sentence on the row interval (module docstring): `on_interval` on its UTC grid, read at the
    heights the labeller checked its words at; its words as instructions on the new rows."""
    step_s = words.spec.step_s
    first = first_interval_row(signals.entry_time_utc, interval_s, step_s)
    courses = [candidate.course_deg for candidate in geometry.candidates]
    grid = on_interval(reading.words, first, interval_s, step_s, reading.held_altitude_m, words, courses)
    return Sentence(grid=grid, instructions=instructions_of(grid, geometry, words), first_row=first)


def instructions_of(grid: np.ndarray, geometry: AirportGeometry, words: Words) -> list[Instruction]:
    """A sentence's words as instructions on its rows (what the judge reads): a heading word's ``target_deg`` the track
    it says under the runway in force at its row (the runway column comes first in a row)."""
    courses = [candidate.course_deg for candidate in geometry.candidates]
    runway = grid[np.maximum.accumulate(np.where(grid[:, RUNWAY] >= 0, np.arange(len(grid)), 0)), RUNWAY]
    instructions = []
    for row, column in zip(*np.nonzero(grid != UNCHANGED)):
        value = int(grid[row, column])
        info = {"target_deg": words.heading_track_deg(value, courses[runway[row]])} if column == HEADING else {}
        instructions.append(Instruction(int(column), value, int(row), "said", info))
    return instructions


def from_row(signals: FlightSignals, row: int) -> FlightSignals:
    """The observed flight from row ``row`` on: its clock starting there (row 0 at that row's UTC second)."""
    if row == 0:
        return signals
    entry = later_utc(signals.entry_time_utc, float(signals.time_s[row] - signals.time_s[0]))
    return replace(signals, entry_time_utc=entry,
                   **{name: getattr(signals, name)[row:] for name in ROW_FIELDS if name != "time_s"},
                   time_s=signals.time_s[row:] - signals.time_s[row])


@dataclass
class Batch:
    indices: list[int]              # each flight's place in the artefact's signals of the split
    signals: list[FlightSignals]    # the observed flights, from each sentence's first row on
    series: list[FlightSeries]
    readings: list[Reading]         # the labeller's readings (the data's step)
    sentences: list[Sentence]       # what is flown, on ``row_interval_s``
    row_interval_s: float
    geometries: list[AirportGeometry]
    vertical_paths: list[tuple[VerticalPath, ...]]  # each flight's candidates' vertical paths (`published_vertical_paths`)
    approach_ias_mps: list[float]
    groups: list[str]               # OWN or STAND_IN, per flight
    drawn: dict[str, Any]           # the sample's description: pool, read, exclusions, per airport

    def inputs(self, device: torch.device) -> FlightInputs:
        return flight_inputs(self.series, [s.first_row for s in self.sentences], device=device)


def open_executor(executor_dir: Path, instructions_dir: Path) -> tuple[ExecutorParams, dict[str, Any], Words]:
    """An executor spec and the instruction artefact it flies (`open_spec`), refused unless the executor code on disk
    flies the spec's reference tracks within the bounds (`spec.require_conforming_executor`, executor design §12.3)."""
    opened = open_spec(executor_dir, instructions_dir)
    require_conforming_executor(executor_dir)
    return opened


def open_spec(executor_dir: Path, instructions_dir: Path) -> tuple[ExecutorParams, dict[str, Any], Words]:
    """An executor spec and the instruction artefact it flies, refused unless the spec was measured against this
    artefact's vocabulary and the labeller code on disk reads the artefact's reference as it was read (the replay
    re-reads every flight, and the judge reads with it, `instructions.conformance`) — whatever executor code is on
    disk: what flies the spec's reference tracks again (`autopilot.conformance`) opens it so."""
    params, record = load_spec(executor_dir)
    spec = artefact.load_spec(instructions_dir)
    if record["vocabulary_spec_sha256"] != spec.sha256:
        raise ValueError(f"the executor spec was measured against vocabulary {record['vocabulary_spec_sha256'][:12]}, "
                         f"{instructions_dir} holds {spec.sha256[:12]}")
    require_conforming_labeller(instructions_dir)
    params.check(spec, spec.step_s)
    return params, record, Words(spec)


def group_of(series: FlightSeries) -> str:
    """`OWN` or `STAND_IN` for a flight that can be flown, else why it cannot (§11)."""
    source = series.scenario.source
    identified = source["resolved_typecode"]
    if identified is None:
        return "no identified type"
    if not series.scenario.has_dynamics:
        # the performance index excludes the type, or nothing models it: the signals keep the flight
        # (`all-flights`), but there is no airframe to fly it on (C31)
        return "no aircraft dynamics"
    if math.isnan(approach_speed_ias_mps(identified, None)):
        return "type publishes no approach speed"
    return OWN if identified == source["dynamics_typecode"] else STAND_IN


def flight_approach_ias_mps(series: FlightSeries, group: str) -> float:
    """The "unspecified" speed a flown flight is given: its type's at its mass on its own dynamics, as
    published (unscaled) on a stand-in's."""
    mass = float(series.scenario.initial.m) if group == OWN else None
    return approach_speed_ias_mps(series.scenario.source["resolved_typecode"], mass)


@dataclass
class Drawn:
    """A split's sample before any sentence: the flights, their rebuilt series and groups, and its description."""
    indices: list[int]              # into the artefact's signals of the split
    signals: list[FlightSignals]
    series: list[FlightSeries]
    groups: list[str]
    geometries: dict[str, AirportGeometry]
    vertical_paths: dict[str, tuple[VerticalPath, ...]]
    description: dict[str, Any]


def draw_flights(directory: Path, split: str, candidates: list[int], *, per_airport: int, seed: int,
                 groups: tuple[str, ...] = (OWN,)) -> Drawn:
    """Of the split's signals at ``candidates``, the first ``per_airport`` of ``groups`` per airport (0: every
    one), in a seeded permutation."""
    geometries = load_candidates(directory)
    signals = load_signals(directory, split)
    order = [int(i) for i in np.random.default_rng(seed).permutation(sorted(candidates))]
    wanted = {airport: per_airport or len(order) for airport in geometries}
    taken: list[tuple[int, FlightSeries, str]] = []
    excluded: Counter = Counter()
    read = 0
    for start in range(0, len(order), DRAW_CHUNK):
        chunk = [i for i in order[start: start + DRAW_CHUNK] if wanted[signals[i].airport] > 0]
        if not chunk:
            if not any(wanted.values()):
                break
            continue
        for i, series in zip(chunk, rebuild_series(directory, [signals[i] for i in chunk])):
            airport = signals[i].airport
            if wanted[airport] == 0:
                continue
            read += 1
            group = group_of(series)
            if group in groups:
                taken.append((i, series, group))
                wanted[airport] -= 1
            else:
                excluded[group] += 1
        if not any(wanted.values()):
            break
    short = {airport: need for airport, need in wanted.items() if need and per_airport}
    if short:
        raise ValueError(f"the {split} split holds too few eligible flights: {short} short")
    paths = {code: published_vertical_paths(geometry) for code, geometry in geometries.items()}
    return Drawn(indices=[i for i, _, _ in taken], signals=[signals[i] for i, _, _ in taken],
                 series=[s for _, s, _ in taken], groups=[g for _, _, g in taken], geometries=geometries,
                 vertical_paths=paths,
                 description={"split": split, "seed": seed, "per_airport": per_airport or "every labelled flight",
                              "groups": list(groups), "pool": len(order), "read": read,
                              "excluded": dict(excluded.most_common()), "flights": len(taken),
                              "by_group": dict(Counter(g for _, _, g in taken)),
                              "threshold_crossing_heights_m": {
                                  code: dict(zip((c.ident for c in geometry.candidates),
                                                 (path.crossing_height_m for path in paths[code])))
                                  for code, geometry in geometries.items()}})


def batch_of(drawn: Drawn, keep: list[int], readings: list[Reading], row_interval_s: float, words: Words) -> Batch:
    """The flights of ``drawn`` at ``keep`` flown from ``readings`` (one per kept flight, in that order), each sentence
    on ``row_interval_s`` (`sentence_on_interval`); a flight whose sentence the row interval refuses is left out and
    counted by reason in the description (``refused_on_interval``)."""
    flights, sentences, kept_readings = [], [], []
    refused: Counter = Counter()
    for i, reading in zip(keep, readings):
        try:
            sentence = sentence_on_interval(reading, drawn.signals[i], row_interval_s,
                                            drawn.geometries[drawn.signals[i].airport], words)
        except Refused as refusal:
            refused[refusal.reason] += 1
            continue
        flights.append(i)
        sentences.append(sentence)
        kept_readings.append(reading)
    return Batch(indices=[drawn.indices[i] for i in flights],
                 signals=[from_row(drawn.signals[i], s.first_row) for i, s in zip(flights, sentences)],
                 series=[drawn.series[i] for i in flights], readings=kept_readings, sentences=sentences,
                 row_interval_s=row_interval_s, geometries=[drawn.geometries[drawn.signals[i].airport] for i in flights],
                 vertical_paths=[drawn.vertical_paths[drawn.signals[i].airport] for i in flights],
                 approach_ias_mps=[flight_approach_ias_mps(drawn.series[i], drawn.groups[i]) for i in flights],
                 groups=[drawn.groups[i] for i in flights],
                 drawn={**drawn.description, "row_interval_s": row_interval_s,
                        "refused_on_interval": dict(refused.most_common())})


def draw(directory: Path, split: str, spec: VocabularySpec, words: Words, *, per_airport: int, seed: int,
         groups: tuple[str, ...] = (OWN,), row_interval_s: float) -> Batch:
    """The split's first ``per_airport`` labelled flights of ``groups`` per airport (0: every one), in a
    seeded permutation, each re-read and checked against its stored sentence, flown on ``row_interval_s``."""
    drawn, readings = draw_readings(directory, split, spec, words, per_airport=per_airport, seed=seed, groups=groups)
    return batch_of(drawn, list(range(len(readings))), readings, row_interval_s, words)


def draw_readings(directory: Path, split: str, spec: VocabularySpec, words: Words, *, per_airport: int, seed: int,
                  groups: tuple[str, ...] = (OWN,)) -> tuple[Drawn, list[Reading]]:
    """`draw`'s flights and their readings, each re-read and checked against its stored sentence, before any row
    interval."""
    sentences = load_sentences(directory, split, spec)
    stored = {int(index): k for k, index in enumerate(sentences["signal_index"])}
    drawn = draw_flights(directory, split, list(stored), per_airport=per_airport, seed=seed, groups=groups)
    readings = []
    for i, flight in zip(drawn.indices, drawn.signals):
        reading = read_flight(flight, drawn.geometries[flight.airport], spec, words)
        k = stored[i]
        grid = sentences["words"][sentences["offsets"][k]: sentences["offsets"][k + 1]]
        if not np.array_equal(reading.words, grid) or reading.runway_index != int(sentences["runway_index"][k]):
            raise ValueError(f"{flight.dataset_id}: the re-read sentence differs from the stored one")
        readings.append(reading)
    return drawn, readings


def subset(batch: Batch, indices: list[int]) -> Batch:
    """The flights at ``indices``, in that order (the sample's description is the whole batch's)."""
    return Batch(indices=[batch.indices[i] for i in indices], signals=[batch.signals[i] for i in indices],
                 series=[batch.series[i] for i in indices],
                 readings=[batch.readings[i] for i in indices], sentences=[batch.sentences[i] for i in indices],
                 row_interval_s=batch.row_interval_s, geometries=[batch.geometries[i] for i in indices],
                 vertical_paths=[batch.vertical_paths[i] for i in indices],
                 approach_ias_mps=[batch.approach_ias_mps[i] for i in indices], groups=[batch.groups[i] for i in indices],
                 drawn=batch.drawn)


def observed_rows(signals: FlightSignals, sentence: Sentence, row_interval_s: float, step_s: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """The observed positions at the data's rows (``step_s`` apart, ``signals`` from the sentence's first row on) over
    the sentence's span: what the distance and track clocks read, at every row interval alike — so that in the ablation
    only the sentence's interval changes, never the clock (design §4.8)."""
    rows = len(sentence.grid) * int(round(row_interval_s / step_s))
    return signals.e_m[:rows], signals.n_m[:rows]


def word_clock(batch: Batch, params: ExecutorParams, step_s: float,
               device: torch.device) -> TimeClock | DistanceClock | TrackClock:
    """The clock the batch's truth sentences are said on (`ExecutorParams.word_clock`, §11): the distance and track
    clocks read each observed flight at the data's rows (``step_s``, `observed_rows`); a sentence time in seconds is
    looked up on the sentence's own rows (`Sentences.at`)."""
    if params.word_clock == "time":
        return TimeClock(params.cycle_s)
    rows = [observed_rows(f, s, batch.row_interval_s, step_s) for f, s in zip(batch.signals, batch.sentences)]
    clock = DistanceClock if params.word_clock == "distance" else TrackClock
    return clock.of([e for e, _ in rows], [n for _, n in rows], step_s, params.cycle_s, device=device)


def time_limits_s(batch: Batch, params: ExecutorParams) -> list[float]:
    """Each flight's time limit before its go-arounds: its sentence's rows × the row interval × the timeout factor
    (design §5.8: the remaining observed time × 1.5)."""
    return [len(s.grid) * batch.row_interval_s * params.timeout_factor for s in batch.sentences]


def reserve_s(batch: Batch) -> float:
    """The time the batch's go-arounds may add (`executor.GO_AROUND_EXTRA_S` each, its most go-arounds)."""
    return GO_AROUND_EXTRA_S * max(s.go_arounds for s in batch.sentences)


def fly_sentences(batch: Batch, params: ExecutorParams, words: Words, *, device: torch.device) -> Flown:
    """Fly every flight's sentence from its first row."""
    f64 = torch.float64
    sentences = Sentences([s.grid for s in batch.sentences], words, step_s=batch.row_interval_s, device=device)
    return fly(batch.inputs(device), sentences, word_clock(batch, params, words.spec.step_s, device),
               Runways.of(batch.geometries, words.spec, dtype=f64, device=device),
               AirportCharts.of(batch.geometries, dtype=f64, device=device),
               torch.tensor(batch.approach_ias_mps, dtype=f64, device=device), params, words,
               time_limit_s=torch.tensor(time_limits_s(batch, params), dtype=f64, device=device),
               reserve_s=reserve_s(batch))


def judge_batch(batch: Batch, flown: Flown, words: Words) -> list[Verdict]:
    return [judge(flown, j, batch.geometries[j], batch.vertical_paths[j], batch.sentences[j].instructions,
                  batch.row_interval_s, batch.signals[j], words.spec, words) for j in range(len(batch.sentences))]


def fly_batch(batch: Batch, params: ExecutorParams, words: Words, *,
              device: torch.device) -> tuple[Flown, list[Verdict]]:
    """Fly every flight's sentence from its first row and judge it."""
    flown = fly_sentences(batch, params, words, device=device)
    return flown, judge_batch(batch, flown, words)


#: The columns layer 2 judges (the runway column has no envelope).
JUDGED = ("heading", "altitude", "speed")


def word_results(verdict: Verdict) -> tuple[list[tuple[str, bool]], int] | None:
    """Every word the judge judged, ``(column, inside its envelope)`` (altitude and angle words share their tube), and
    how many heading words it did not judge (the lead carried their rows past the flight's end); None when fewer than
    two flown rows were left (nothing was judged)."""
    if verdict.words is None:
        return None
    judged: list[tuple[str, bool]] = []
    not_judged = 0
    for h in verdict.words["heading"]:
        if h["rows"] == 0:
            not_judged += 1
            continue
        judged.append(("heading", h["inside"] == h["rows"]))
    judged += [("altitude", bool(v["contained"])) for v in verdict.words["vertical"]]
    judged += [("speed", bool(v["contained"])) for v in verdict.words["speed"]]
    return judged, not_judged


def skipped_by_clock(headings: list[dict[str, int]]) -> list[bool]:
    """For each of a verdict's heading words, whether the clock skipped it: a later heading word was told on its flown
    row (the track or distance clock passed two sentence rows within a step, `sentence.TRACK_MAX_ROWS_PER_CYCLE`) and
    that row's last word is judged — so this one, never flown, is judged on no rows because of the clock, not because
    its lead ran past the flight's end."""
    last = {h["row"]: h for h in headings}
    return [h["rows"] == 0 and last[h["row"]] is not h and last[h["row"]]["rows"] > 0 for h in headings]


def told_with_skipped(headings: list[dict[str, int]]) -> list[bool]:
    """For each of a verdict's heading words, whether it is judged and was told on the flown row of a word the clock
    skipped (`skipped_by_clock`): it arrives two steps' worth of turn at once."""
    skipped_rows = {h["row"] for h, skipped in zip(headings, skipped_by_clock(headings)) if skipped}
    return [h["rows"] > 0 and h["row"] in skipped_rows for h in headings]


def clock_pairs(verdict: Verdict) -> dict[str, int]:
    """The judged heading words told with a word the clock skipped (`told_with_skipped`), and how many are inside."""
    headings = [] if verdict.words is None else verdict.words["heading"]
    told = [h for h, together in zip(headings, told_with_skipped(headings)) if together]
    return {"judged": len(told), "inside": sum(h["inside"] == h["rows"] for h in told)}


def summary(verdicts: list[Verdict]) -> dict[str, Any]:
    """The batch's headline numbers: outcomes, flown as said, the words inside their envelopes per column (per word
    judged; the words not judged beside it), and the word checks that failed. The heading words told with one the
    clock skipped (`told_with_skipped`) are counted apart as well: what the clock did to the sentence, not the
    executor. No criterion is read here (design D7)."""
    outcomes = Counter(v.outcome for v in verdicts)
    counted = [word_results(v) for v in verdicts]
    judged = [(column, ok) for c in counted if c is not None for column, ok in c[0]]
    words_failed: Counter = Counter()
    pairs = [clock_pairs(v) for v in verdicts]
    for v in verdicts:
        if v.words is None:
            words_failed["fewer than two flown rows (nothing judged)"] += 1
            continue
        headings = v.words["heading"]
        for h, skipped, together in zip(headings, skipped_by_clock(headings), told_with_skipped(headings)):
            outside = 0 < h["rows"] and h["inside"] < h["rows"]
            words_failed["heading word skipped by the clock (told with the next, not judged)"] += skipped
            words_failed["heading word past the flight's end (not judged)"] += h["rows"] == 0 and not skipped
            words_failed["track off its heading word a lead later"] += outside and not together
            words_failed["track off its heading word a lead later, told with a skipped word"] += outside and together
        words_failed["superseded before flown (not judged)"] += v.words["superseded_before_flown"]
        words_failed["altitude word outside its tube"] += sum(not x["contained"] for x in v.words["vertical"])
        words_failed["speed word outside its band"] += sum(not x["contained"] for x in v.words["speed"])
    decisions = [v.crossing["decision"] for v in verdicts if v.crossing is not None and "decision" in v.crossing]
    n = len(verdicts)
    return {"flights": n, "outcomes": dict(outcomes.most_common()),
            "landed_share": outcomes["landed"] / n,
            "flew_the_sentence_share": sum(v.flew_the_sentence for v in verdicts) / n,
            "words_judged": len(judged), "words_inside_share": sum(ok for _, ok in judged) / len(judged) if judged else None,
            "words_inside_share_by_column": {
                column: (sum(ok for c, ok in judged if c == column) / sum(c == column for c, _ in judged)
                         if any(c == column for c, _ in judged) else None) for column in JUDGED},
            "heading_words_not_judged": sum(c[1] for c in counted if c is not None),
            "heading_words_told_with_a_skipped_word": {"judged": sum(p["judged"] for p in pairs),
                                                       "inside": sum(p["inside"] for p in pairs)},
            "flights_with_unjudged_words": sum(c is None for c in counted),
            "decision_checks": {"approach_crossings_low": len(decisions),
                                "no_da_point": sum(d is None for d in decisions),
                                "passed": sum(d is not None and d["passed"] for d in decisions),
                                "lateral_failed": sum(d is not None and not d["lateral_ok"] for d in decisions),
                                "vertical_failed": sum(d is not None and not d["vertical_ok"] for d in decisions)},
            "word_failures": {k: int(c) for k, c in words_failed.most_common() if c}}


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {f"p{q}": float(np.percentile(values, q)) for q in (5, 25, 50, 75, 95)} | {"n": len(values)}


def observed_landing_s(observed: FlightSignals, rows: int, runway_index: int, geometry: AirportGeometry) -> float:
    """When the observed flight (from its sentence's first row on) reached the landed threshold: the sentence's last row
    of the data's step (``rows`` of them from the first), carried on at that row's ground speed over the distance still
    to go."""
    last = rows - 1
    candidate = geometry.candidates[runway_index]
    relative = relative_to_runway(observed.e_m[last: last + 1], observed.n_m[last: last + 1],
                                  observed.track_deg[last: last + 1], observed.altitude_m[last: last + 1], candidate)
    return float(observed.time_s[last] + relative.before_threshold_m[0] / observed.ground_speed_mps[last])


def flight_alignment(batch: Batch, flown: Flown,
                     verdicts: list[Outcome] | list[Verdict]) -> list[dict[str, float | None]]:
    """How each flown track differs from the observed one (reported, no gate): the mean horizontal and vertical distance
    at the data's rows both tracks reach (time-aligned from the sentence's first row), and for a landed flight its
    landing time minus the observed one — its interpolated crossing against the sentence's last row carried to the
    threshold at that row's ground speed (None for a flight that did not land)."""
    out = []
    for j, verdict in enumerate(verdicts):
        observed, reading, sentence = batch.signals[j], batch.readings[j], batch.sentences[j]
        step_rows = int(round((observed.time_s[1] - observed.time_s[0]) / flown.cycle_s))
        track = flown_track(flown.states[j, : verdict.end_row + 1].cpu().numpy(), batch.geometries[j])
        sentence_rows = len(reading.words) - sentence.first_row
        rows = min(sentence_rows, verdict.end_row // step_rows + 1)
        flown_rows = np.arange(rows) * step_rows
        out.append({
            "mean_horizontal_distance_m": float(np.mean(np.hypot(track["e"][flown_rows] - observed.e_m[:rows],
                                                                 track["n"][flown_rows] - observed.n_m[:rows]))),
            "mean_vertical_distance_m": float(np.mean(np.abs(track["height"][flown_rows] - observed.altitude_m[:rows]))),
            "landing_time_minus_observed_s": (verdict.crossing["at_row"] * flown.cycle_s - observed_landing_s(
                observed, sentence_rows, reading.runway_index, batch.geometries[j])
                if verdict.outcome == "landed" else None)})
    return out


def alignment(batch: Batch, flown: Flown, verdicts: list[Verdict]) -> dict[str, Any]:
    """`flight_alignment` over the batch, as percentiles (the landing time over the landed flights)."""
    flights = flight_alignment(batch, flown, verdicts)
    return {name: _percentiles([f[name] for f in flights if f[name] is not None])
            for name in ("mean_horizontal_distance_m", "mean_vertical_distance_m", "landing_time_minus_observed_s")}
