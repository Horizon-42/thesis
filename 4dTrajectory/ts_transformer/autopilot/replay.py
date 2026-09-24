"""A batch of labelled flights flown from their sentences (executor design §11): who is flown, their inputs,
the flight and the verdicts — shared by the spec's measurements, the sensitivity check and the replay gate.

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
runway), the export's rule: the executor flies the reading of the flight, kinds and split parts included,
not a grid that merely looks like it.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.cli import DEFAULT_CIFP, DEFAULT_CONFIG
from trajectory_data_process.harvest.airports import load_airport
from ts_transformer.autopilot.executor import Flown, fly
from ts_transformer.autopilot.flights import FlightInputs, flight_inputs, rebuild_series
from ts_transformer.autopilot.frame import AirportCharts
from ts_transformer.autopilot.judge import Outcome, Verdict, flown_track, judge
from ts_transformer.autopilot.lateral import Runways
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.sentence import DistanceClock, Sentences, TimeClock, TrackClock
from ts_transformer.autopilot.spec import load_spec, require_current_executor
from ts_transformer.autopilot.speed import approach_speed_ias_mps
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.instructions import artefact
from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway
from ts_transformer.instructions.artefact import load_candidates, load_sentences, load_signals
from ts_transformer.instructions.labeller.read import Reading, read_flight
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import Words

#: Rebuilt at a time while drawing the sample (a rebuild opens the flights' tracks).
DRAW_CHUNK = 200
OWN, STAND_IN = "own dynamics", "stand-in dynamics"


@dataclass
class Batch:
    signals: list[FlightSignals]
    series: list[FlightSeries]
    readings: list[Reading]
    geometries: list[AirportGeometry]
    crossing_heights: list[tuple[float, ...]]   # each flight's candidates' published TCH (`published_crossing_heights`)
    approach_ias_mps: list[float]
    groups: list[str]               # OWN or STAND_IN, per flight
    drawn: dict[str, Any]           # the sample's description: pool, read, exclusions, per airport

    def inputs(self, device: torch.device) -> FlightInputs:
        return flight_inputs(self.series, device=device)


def open_executor(executor_dir: Path, instructions_dir: Path) -> tuple[ExecutorParams, dict[str, Any], Words]:
    """An executor spec and the instruction artefact it flies, refused unless the spec was measured by this
    executor code against this artefact's vocabulary, and the artefact's labeller is this code (the judge
    reads with it)."""
    params, record = load_spec(executor_dir)
    require_current_executor(record)
    spec = artefact.load_spec(instructions_dir)
    if record["vocabulary_spec_sha256"] != spec.sha256:
        raise ValueError(f"the executor spec was measured against vocabulary {record['vocabulary_spec_sha256'][:12]}, "
                         f"{instructions_dir} holds {spec.sha256[:12]}")
    artefact.require_current_labeller(instructions_dir)
    if record["source"]["labeller_source_sha256"] != artefact.labeller_source_sha256():
        raise ValueError("the executor spec was measured on readings of other labeller code "
                         f"({record['source']['labeller_source_sha256'][:12]}, now "
                         f"{artefact.labeller_source_sha256()[:12]})")
    params.check(spec)
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
    crossing_heights: dict[str, tuple[float, ...]]
    description: dict[str, Any]


def published_crossing_heights(geometry: AirportGeometry) -> tuple[float, ...]:
    """Each candidate runway's published threshold crossing height (TCH, m above the threshold), in the candidates'
    order: the harvest's runway data for the airport (`trajectory_data_process.harvest.airports.load_airport`, the
    FAA CIFP's vertical path) — the evaluation's own reference plane. A candidate that publishes none is refused:
    "descend to land" has no crossing point there."""
    runways = {runway.ident: runway for runway in load_airport(geometry.code, config_file=DEFAULT_CONFIG,
                                                                 cifp_file=DEFAULT_CIFP).runways}
    heights = []
    for candidate in geometry.candidates:
        height = runways[candidate.ident].threshold_crossing_height_m
        if height is None:
            raise ValueError(f"{geometry.code} {candidate.ident} publishes no threshold crossing height")
        heights.append(float(height))
    return tuple(heights)


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
    return Drawn(indices=[i for i, _, _ in taken], signals=[signals[i] for i, _, _ in taken],
                 series=[s for _, s, _ in taken], groups=[g for _, _, g in taken], geometries=geometries,
                 crossing_heights={code: published_crossing_heights(geometry) for code, geometry in geometries.items()},
                 description={"split": split, "seed": seed, "per_airport": per_airport or "every labelled flight",
                              "groups": list(groups), "pool": len(order), "read": read,
                              "excluded": dict(excluded.most_common()), "flights": len(taken),
                              "by_group": dict(Counter(g for _, _, g in taken))})


def batch_of(drawn: Drawn, keep: list[int], readings: list[Reading]) -> Batch:
    """The flights of ``drawn`` at ``keep`` flown from ``readings`` (one per kept flight, in that order)."""
    return Batch(signals=[drawn.signals[i] for i in keep], series=[drawn.series[i] for i in keep], readings=readings,
                 geometries=[drawn.geometries[drawn.signals[i].airport] for i in keep],
                 crossing_heights=[drawn.crossing_heights[drawn.signals[i].airport] for i in keep],
                 approach_ias_mps=[flight_approach_ias_mps(drawn.series[i], drawn.groups[i]) for i in keep],
                 groups=[drawn.groups[i] for i in keep], drawn=drawn.description)


def draw(directory: Path, split: str, spec: VocabularySpec, words: Words, *, per_airport: int, seed: int,
         groups: tuple[str, ...] = (OWN,)) -> Batch:
    """The split's first ``per_airport`` labelled flights of ``groups`` per airport (0: every one), in a
    seeded permutation, each re-read and checked against its stored sentence."""
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
    return batch_of(drawn, list(range(len(readings))), readings)


def subset(batch: Batch, indices: list[int]) -> Batch:
    """The flights at ``indices``, in that order (the sample's description is the whole batch's)."""
    return Batch(signals=[batch.signals[i] for i in indices], series=[batch.series[i] for i in indices],
                 readings=[batch.readings[i] for i in indices], geometries=[batch.geometries[i] for i in indices],
                 crossing_heights=[batch.crossing_heights[i] for i in indices],
                 approach_ias_mps=[batch.approach_ias_mps[i] for i in indices], groups=[batch.groups[i] for i in indices],
                 drawn=batch.drawn)


def word_clock(batch: Batch, params: ExecutorParams, step_s: float,
               device: torch.device) -> TimeClock | DistanceClock | TrackClock:
    """The clock the batch's truth sentences are said on (`ExecutorParams.word_clock`, §11): the distance clock
    reads each observed flight's path at its sentence's rows."""
    if params.word_clock == "time":
        return TimeClock(params.cycle_s)
    rows = [len(r.words) for r in batch.readings]
    e_m, n_m = [f.e_m[:n] for f, n in zip(batch.signals, rows)], [f.n_m[:n] for f, n in zip(batch.signals, rows)]
    clock = DistanceClock if params.word_clock == "distance" else TrackClock
    return clock.of(e_m, n_m, step_s, params.cycle_s, device=device)


def fly_sentences(batch: Batch, params: ExecutorParams, words: Words, *, device: torch.device) -> Flown:
    """Fly every flight's sentence from its row 0."""
    spec = words.spec
    f64 = torch.float64
    limits = torch.tensor([len(r.words) * spec.step_s * params.timeout_factor for r in batch.readings], dtype=f64,
                          device=device)
    sentences = Sentences([r.words for r in batch.readings], words, device=device)
    return fly(batch.inputs(device), sentences, word_clock(batch, params, spec.step_s, device),
               Runways.of(batch.geometries, batch.crossing_heights, dtype=f64, device=device),
               AirportCharts.of(batch.geometries, dtype=f64, device=device),
               torch.tensor(batch.approach_ias_mps, dtype=f64, device=device), params, words, time_limit_s=limits)


def fly_batch(batch: Batch, params: ExecutorParams, words: Words, *,
              device: torch.device) -> tuple[Flown, list[Verdict]]:
    """Fly every flight's sentence from its row 0 and judge it."""
    flown = fly_sentences(batch, params, words, device=device)
    verdicts = [judge(flown, j, batch.geometries[j], batch.readings[j].runway_index, batch.readings[j],
                      batch.signals[j], words.spec, words) for j in range(len(batch.readings))]
    return flown, verdicts


def word_results(verdict: Verdict) -> tuple[list[tuple[str, bool]], int] | None:
    """Every word the judge judged, ``(column, inside its envelope)``, and how many heading words it did not
    judge (the lead carried their rows past the clearance or the capture); None when the flown track did not
    pass the labeller's gate (nothing was judged)."""
    if verdict.words is None:
        return None
    judged: list[tuple[str, bool]] = []
    not_judged = 0
    if verdict.words["intercepting_off_word_cycles"]:
        judged.append(("heading", False))                   # the word the executor left to intercept on its own
    for h in verdict.words["heading"]:
        if h["rows"] == 0:
            not_judged += 1
            continue
        judged.append(("heading", h["inside"] == h["rows"]))
    corridor, capture = verdict.words["corridor"], verdict.words["capture_turn"]
    if corridor["cleared"]:
        judged.append(("approach", capture is not None and capture["progress_ok"] and capture["rate_ok"]
                       and corridor["entered"] and corridor["inside"] == corridor["rows"]))
    judged += [("altitude", bool(v["contained"])) for v in verdict.words["vertical"]]
    judged += [("speed", bool(v["contained"])) for v in verdict.words["speed"]]
    return judged, not_judged


def skipped_by_clock(headings: list[dict[str, int]]) -> list[bool]:
    """For each of a verdict's heading words, whether the clock skipped it: a later heading word was told on its flown
    row (the track or distance clock passed two sentence rows within a step, `sentence.TRACK_MAX_ROWS_PER_CYCLE`) and
    that row's last word is judged — so this one, never flown, is judged on no rows because of the clock, not because
    its lead ran past the clearance or the capture."""
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
    """The batch's headline numbers: outcomes, flown as said, the words inside their envelopes (per word
    judged; the words not judged beside it), and the word checks that failed. The heading words told with one the
    clock skipped (`told_with_skipped`) are counted apart as well: what the clock did to the sentence, not the
    executor."""
    outcomes = Counter(v.outcome for v in verdicts)
    counted = [word_results(v) for v in verdicts]
    judged = [ok for c in counted if c is not None for _, ok in c[0]]
    words_failed: Counter = Counter()
    pairs = [clock_pairs(v) for v in verdicts]
    for v in verdicts:
        if v.words is None:
            words_failed["flown track refused by the labeller's gate"] += 1
            continue
        headings = v.words["heading"]
        for h, skipped, together in zip(headings, skipped_by_clock(headings), told_with_skipped(headings)):
            outside = 0 < h["rows"] and h["inside"] < h["rows"]
            words_failed["heading word skipped by the clock (told with the next, not judged)"] += skipped
            words_failed["heading word past the clearance or capture (not judged)"] += h["rows"] == 0 and not skipped
            words_failed["track off its heading word a lead later"] += outside and not together
            words_failed["track off its heading word a lead later, told with a skipped word"] += outside and together
        words_failed["left its heading word to intercept on its own"] += v.words["intercepting_off_word_cycles"] > 0
        words_failed["landing aim left the word's tube"] += v.words["aim_left_tube_cycles"] > 0
        words_failed["superseded before flown (not judged)"] += v.words["superseded_before_flown"]
        corridor, capture = v.words["corridor"], v.words["capture_turn"]
        words_failed["cleared, never captured"] += corridor["cleared"] and capture is None
        words_failed["capture turn outside its envelope"] += capture is not None and not (
            capture["progress_ok"] and capture["rate_ok"])
        words_failed["cleared, corridor never entered"] += corridor["cleared"] and not corridor["entered"]
        words_failed["cleared, corridor left after entry"] += corridor["cleared"] and corridor["inside"] < corridor["rows"]
        words_failed["altitude word outside its tube"] += sum(not x["contained"] for x in v.words["vertical"])
        words_failed["speed word outside its band"] += sum(not x["contained"] for x in v.words["speed"])
    n = len(verdicts)
    return {"flights": n, "outcomes": dict(outcomes.most_common()),
            "landed_share": outcomes["landed"] / n,
            "flew_the_sentence_share": sum(v.flew_the_sentence for v in verdicts) / n,
            "words_judged": len(judged), "words_inside_share": sum(judged) / len(judged) if judged else None,
            "heading_words_not_judged": sum(c[1] for c in counted if c is not None),
            "heading_words_told_with_a_skipped_word": {"judged": sum(p["judged"] for p in pairs),
                                                       "inside": sum(p["inside"] for p in pairs)},
            "flights_with_unjudged_words": sum(c is None for c in counted),
            "word_failures": {k: int(c) for k, c in words_failed.most_common() if c}}


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {f"p{q}": float(np.percentile(values, q)) for q in (5, 25, 50, 75, 95)} | {"n": len(values)}


def observed_landing_s(observed: FlightSignals, reading: Reading, geometry: AirportGeometry) -> float:
    """When the observed flight reached the pointed threshold: its sentence's last row, carried on at that row's
    ground speed over the distance still to go."""
    last = len(reading.words) - 1
    candidate = geometry.candidates[reading.runway_index]
    relative = relative_to_runway(observed.e_m[last: last + 1], observed.n_m[last: last + 1],
                                  observed.track_deg[last: last + 1], observed.altitude_m[last: last + 1], candidate)
    return float(observed.time_s[last] + relative.before_threshold_m[0] / observed.ground_speed_mps[last])


def flight_alignment(batch: Batch, flown: Flown,
                     verdicts: list[Outcome] | list[Verdict]) -> list[dict[str, float | None]]:
    """How each flown track differs from the observed one (§11, reported, no gate): the mean horizontal and
    vertical distance at the sentence's rows both tracks reach (time-aligned from row 0), and for a landed
    flight its landing time minus the observed one — its interpolated crossing against the sentence's last
    row carried to the threshold at that row's ground speed (the data plane ends a flight at its landing, so
    the last row is the last one before it; None for a flight that did not land)."""
    out = []
    for j, verdict in enumerate(verdicts):
        observed, reading = batch.signals[j], batch.readings[j]
        step_rows = int(round((observed.time_s[1] - observed.time_s[0]) / flown.cycle_s))
        track = flown_track(flown.states[j, : verdict.end_row + 1].cpu().numpy(), batch.geometries[j])
        rows = min(len(reading.words), verdict.end_row // step_rows + 1)
        flown_rows = np.arange(rows) * step_rows
        out.append({
            "mean_horizontal_distance_m": float(np.mean(np.hypot(track["e"][flown_rows] - observed.e_m[:rows],
                                                                 track["n"][flown_rows] - observed.n_m[:rows]))),
            "mean_vertical_distance_m": float(np.mean(np.abs(track["height"][flown_rows] - observed.altitude_m[:rows]))),
            "landing_time_minus_observed_s": (verdict.crossing["at_row"] * flown.cycle_s - observed_landing_s(
                observed, reading, batch.geometries[j]) if verdict.outcome == "landed" else None)})
    return out


def alignment(batch: Batch, flown: Flown, verdicts: list[Verdict]) -> dict[str, Any]:
    """`flight_alignment` over the batch, as percentiles (the landing time over the landed flights)."""
    flights = flight_alignment(batch, flown, verdicts)
    return {name: _percentiles([f[name] for f in flights if f[name] is not None])
            for name in ("mean_horizontal_distance_m", "mean_vertical_distance_m", "landing_time_minus_observed_s")}
