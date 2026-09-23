"""A batch of labelled flights flown from their sentences (executor design §11): who is flown, their inputs,
the flight and the verdicts — shared by the spec's measurements, the sensitivity check and the replay gate.

Who is flown (§11): flights whose dynamics are their own type's (`scenario.source`: the identified type is
the dynamics type) and whose type publishes an approach speed ("unspecified" is flown at it); a flight
flown on a stand-in's aerodynamics, or with no identified type, is counted, not flown. The sample is a
seeded permutation of a split's labelled flights, read in order until each airport holds ``per_airport``
eligible flights — the pool, the count read and the exclusions are returned with it.

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

from ts_transformer.autopilot.executor import Flown, fly
from ts_transformer.autopilot.flights import FlightInputs, flight_inputs, rebuild_series
from ts_transformer.autopilot.frame import AirportCharts
from ts_transformer.autopilot.judge import Verdict, flown_track, judge
from ts_transformer.autopilot.lateral import Runways
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.sentence import words_in_force
from ts_transformer.autopilot.spec import load_spec, require_current_executor
from ts_transformer.autopilot.speed import approach_speed_ias_mps
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.instructions import artefact
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import load_candidates, load_sentences, load_signals
from ts_transformer.instructions.labeller.read import Reading, read_flight
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import Words

#: Rebuilt at a time while drawing the sample (a rebuild opens the flights' tracks).
DRAW_CHUNK = 200


@dataclass
class Batch:
    signals: list[FlightSignals]
    series: list[FlightSeries]
    readings: list[Reading]
    geometries: list[AirportGeometry]
    approach_ias_mps: list[float]
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
    params.check(spec)
    return params, record, Words(spec)


def exclusion(series: FlightSeries) -> str | None:
    """Why a flight is not flown (§11), or None."""
    source = series.scenario.source
    identified = source["resolved_typecode"]
    if identified is None:
        return "no identified type"
    if identified != source["dynamics_typecode"]:
        return "flown on a stand-in's dynamics"
    if math.isnan(approach_speed_ias_mps(identified, float(series.scenario.initial.m))):
        return "type publishes no approach speed"
    return None


def draw(directory: Path, split: str, spec: VocabularySpec, words: Words, *, per_airport: int, seed: int) -> Batch:
    """The split's first ``per_airport`` eligible labelled flights per airport, in a seeded permutation."""
    geometries = load_candidates(directory)
    signals = load_signals(directory, split)
    sentences = load_sentences(directory, split, spec)
    stored = {int(index): k for k, index in enumerate(sentences["signal_index"])}
    order = [int(i) for i in np.random.default_rng(seed).permutation(sorted(stored))]
    wanted = {airport: per_airport for airport in geometries}
    taken: list[tuple[int, FlightSeries]] = []
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
            reason = exclusion(series)
            if reason is None:
                taken.append((i, series))
                wanted[airport] -= 1
            else:
                excluded[reason] += 1
        if not any(wanted.values()):
            break
    short = {airport: need for airport, need in wanted.items() if need}
    if short:
        raise ValueError(f"the {split} split holds too few eligible flights: {short} short")
    readings = []
    for i, _series in taken:
        flight = signals[i]
        reading = read_flight(flight, geometries[flight.airport], spec, words)
        k = stored[i]
        grid = sentences["words"][sentences["offsets"][k]: sentences["offsets"][k + 1]]
        if not np.array_equal(reading.words, grid) or reading.runway_index != int(sentences["runway_index"][k]):
            raise ValueError(f"{flight.dataset_id}: the re-read sentence differs from the stored one")
        readings.append(reading)
    return Batch(signals=[signals[i] for i, _ in taken], series=[s for _, s in taken], readings=readings,
                 geometries=[geometries[signals[i].airport] for i, _ in taken],
                 approach_ias_mps=[approach_speed_ias_mps(s.scenario.source["resolved_typecode"], float(s.scenario.initial.m))
                                   for _, s in taken],
                 drawn={"split": split, "seed": seed, "per_airport": per_airport, "pool": len(order), "read": read,
                        "excluded": dict(excluded.most_common()), "flights": len(taken)})


def fly_batch(batch: Batch, params: ExecutorParams, words: Words, *,
              device: torch.device) -> tuple[Flown, list[Verdict]]:
    """Fly every flight's sentence from its row 0 and judge it."""
    spec = words.spec
    f64 = torch.float64
    limits = torch.tensor([len(r.words) * spec.step_s * params.timeout_factor for r in batch.readings], dtype=f64,
                          device=device)
    cycles = int(math.ceil(float(limits.max()) / params.cycle_s))
    force = words_in_force([r.words for r in batch.readings], words, params.delays, cycle_s=params.cycle_s,
                           cycles=cycles, device=device)
    flown = fly(batch.inputs(device), force, Runways.of(batch.geometries, dtype=f64, device=device),
                AirportCharts.of(batch.geometries, dtype=f64, device=device),
                torch.tensor(batch.approach_ias_mps, dtype=f64, device=device), params, words, time_limit_s=limits)
    verdicts = [judge(flown, j, batch.geometries[j], batch.readings[j].runway_index, batch.readings[j],
                      batch.signals[j], spec, words) for j in range(len(batch.readings))]
    return flown, verdicts


def summary(verdicts: list[Verdict]) -> dict[str, Any]:
    """The batch's headline numbers: outcomes, flown as said, and the word checks that failed."""
    outcomes = Counter(v.outcome for v in verdicts)
    words_failed: Counter = Counter()
    for v in verdicts:
        if v.words is None:
            words_failed["flown track refused by the labeller's gate"] += 1
            continue
        for h in v.words["heading"]:
            if h["turn"] is not None:
                words_failed["turn not reached"] += not h["turn"]["reached"]
                words_failed["turn not monotone"] += not h["turn"]["progress_ok"]
                words_failed["turn rate or bank"] += not h["turn"]["rate_ok"]
            words_failed["hold outside its funnel"] += isinstance(h["hold"], dict) and h["hold"]["inside"] < h["hold"]["rows"]
        corridor = v.words["corridor"]
        words_failed["cleared, corridor never entered"] += corridor["cleared"] and not corridor["entered"]
        words_failed["cleared, corridor left after entry"] += corridor["cleared"] and corridor["inside"] < corridor["rows"]
        words_failed["altitude word outside its tube"] += sum(not x["contained"] for x in v.words["vertical"])
        words_failed["speed word outside its band"] += sum(not x["contained"] for x in v.words["speed"])
    n = len(verdicts)
    return {"flights": n, "outcomes": dict(outcomes.most_common()),
            "landed_share": outcomes["landed"] / n,
            "flew_the_sentence_share": sum(v.flew_the_sentence for v in verdicts) / n,
            "word_failures": {k: int(c) for k, c in words_failed.most_common() if c}}


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {f"p{q}": float(np.percentile(values, q)) for q in (5, 25, 50, 75, 95)} | {"n": len(values)}


def alignment(batch: Batch, flown: Flown, verdicts: list[Verdict]) -> dict[str, Any]:
    """How the flown tracks differ from the observed ones (§11, reported, no gate): per flight, the mean
    horizontal and vertical distance at the sentence's rows both tracks reach (time-aligned from row 0),
    and for the landed flights the landing time minus the observed one (the sentence ends at the observed
    crossing)."""
    horizontal, vertical, landing = [], [], []
    for j, verdict in enumerate(verdicts):
        observed, reading = batch.signals[j], batch.readings[j]
        step_rows = int(round((observed.time_s[1] - observed.time_s[0]) / flown.cycle_s))
        track = flown_track(flown.states[j, : verdict.end_row + 1].cpu().numpy(), batch.geometries[j])
        rows = min(len(reading.words), verdict.end_row // step_rows + 1)
        flown_rows = np.arange(rows) * step_rows
        horizontal.append(float(np.mean(np.hypot(track["e"][flown_rows] - observed.e_m[:rows],
                                                 track["n"][flown_rows] - observed.n_m[:rows]))))
        vertical.append(float(np.mean(np.abs(track["height"][flown_rows] - observed.altitude_m[:rows]))))
        if verdict.outcome == "landed":
            landing.append(verdict.end_row * flown.cycle_s - len(reading.words) * step_rows * flown.cycle_s)
    return {"mean_horizontal_distance_m": _percentiles(horizontal), "mean_vertical_distance_m": _percentiles(vertical),
            "landing_time_minus_observed_s": _percentiles(landing)}
