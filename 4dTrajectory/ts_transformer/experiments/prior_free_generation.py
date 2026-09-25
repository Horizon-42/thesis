"""Single-aircraft free generation (prior design §9.1): the prior speaks, the executor flies — step by step — and beside
it the executor flies the labelled words from the same row (the reference).

Each drawn flight (its own dynamics, the replay gate's group) is flown from the observed state at the prior's first
predicted step (row `prior.scene.N_LOOK`):

- ``--samples`` times with the prior speaking (`prior.generate.Speaker`): at every 2 s step it samples the step's six
  words from the positions so far — observed up to the first predicted step, then where the executor flew — the
  airport's landings before the step and the words it has said; the executor hears them at once and flies the step
  (`autopilot.executor.Executor`, two 1 s cycles; `autopilot.sentence.Spoken`);
- once with the labelled words: the words in force at the first predicted step, then the sentence as written, said on
  the executor spec's clock over the observed rows from that row (as the replay gate says them).

Every flight ends as the executor's judge reads it (`autopilot.judge.outcome_of`) — landed on the runway pointed at the
end, crossed without capture, crossed off the runway, ground contact, dynamics failure, or timeout (the observed
remaining time × the spec's timeout factor) — and the readout counts them in all, per airport and per airport ×
approach kind, with the runways pointed (first and last), the time to the end against the observed remaining time,
the words said per column, runway changes and go-arounds. Writes ``generation.json`` and ``sentences.npz`` (the words
the prior said) into a NEW directory; the val split only from a clean tree.

    python run_ts.py prior_free_generation --prior 4dTrajectory/outputs/POOLED/prior/<campaign>/<chosen run> \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v4_20260924 \\
        --executor 4dTrajectory/outputs/POOLED/executor/<spec dir> --split select --per-airport 100 \\
        --out 4dTrajectory/outputs/POOLED/prior/<campaign>/<name>
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.executor import Executor, Flown, fly
from ts_transformer.autopilot.flights import FlightInputs, flight_inputs
from ts_transformer.autopilot.frame import AirportCharts
from ts_transformer.autopilot.judge import OUTCOMES, outcome_of
from ts_transformer.autopilot.lateral import Runways
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.autopilot.sentence import DistanceClock, Sentences, Spoken, TimeClock, TrackClock
from ts_transformer.experiments.prior_train import load_prior, rosters
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import SPLITS
from ts_transformer.instructions.labeller.read import Reading
from ts_transformer.instructions.readout import STRATA, flight_record
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.words import (
    APPROACH, APPROACH_CLEARED, APPROACH_GO_AROUND, COLUMNS, RUNWAY, UNCHANGED, Words,
)
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.prior.data import VARIANTS, airport_landings
from ts_transformer.prior.generate import Speaker, rows_for
from ts_transformer.prior.model import Prior
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.repo_layout import REPO_ROOT, git_state

GENERATION_SCHEMA = "ts-prior-free-generation-v1"


def limits_s(batch: replay.Batch, params: ExecutorParams, step_s: float) -> list[float]:
    """Each flight's time limit from the first predicted step: its sentence's remaining rows × the step × the timeout
    factor (the replay gate's convention, `replay.fly_sentences`)."""
    return [(len(r.words) - N_LOOK) * step_s * params.timeout_factor for r in batch.readings]


def observed_remaining_s(reading: Reading, step_s: float) -> float:
    """From the first predicted step to the observed landing: the sentence's last row is at the landing."""
    return (len(reading.words) - 1 - N_LOOK) * step_s


def reference_grid(grid: np.ndarray) -> np.ndarray:
    """The labelled sentence from the first predicted step: its first row every column's word in force there, then
    the sentence as written."""
    grid = np.asarray(grid, dtype=np.int64)
    out = grid[N_LOOK:].copy()
    for column in range(6):
        written = np.flatnonzero(grid[: N_LOOK + 1, column] != UNCHANGED)
        out[0, column] = grid[written[-1], column]
    return out


def _physics(batch: replay.Batch, device: torch.device) -> tuple[Runways, AirportCharts, torch.Tensor]:
    f64 = torch.float64
    return (Runways.of(batch.geometries, batch.crossing_heights, dtype=f64, device=device),
            AirportCharts.of(batch.geometries, dtype=f64, device=device),
            torch.tensor(batch.approach_ias_mps, dtype=f64, device=device))


class ClosedLoop:
    """A batch of flights flown from their first predicted step (``inputs``: the executor's state there) with the prior
    speaking, a step at a time (`step`): the speaker reads where the executor is, says the step's words, and the
    executor flies the step (its cycles). A flight the executor is done with hears nothing more and its row is frozen;
    one it has cleared or captured keeps its runway. Each flight flies until the executor is done with it or its time
    limit (``limits``, seconds). `take` re-forms the batch from some of its flights (a closed loop's branches,
    `prior_closed_loop`)."""

    def __init__(self, model: Prior, flights: Sequence[FlightSignals], geometries: Sequence[AirportGeometry],
                 inputs: FlightInputs, runways: Runways, charts: AirportCharts, approach_ias_mps: torch.Tensor,
                 limits: Sequence[float], words: Words, params: ExecutorParams, landings: Any, *,
                 generator: torch.Generator, temperature: float) -> None:
        step_s, device = words.spec.step_s, inputs.initial_state.device
        self.step_s, self.params, self.device = step_s, params, device
        self.executor = Executor(inputs, runways, charts, approach_ias_mps, params, words,
                                 time_limit_s=torch.tensor(limits, dtype=torch.float64, device=device))
        self.speaker = Speaker(model, flights, geometries, landings, words,
                               max_rows=rows_for(max(limits) + step_s, step_s), generator=generator,
                               temperature=temperature)
        self.spoken = Spoken(len(limits), words, device=device)
        self.max_steps = rows_for(max(limits), step_s) - N_LOOK
        #: the executor's state when each step was said ([B] each): cleared (since the last go-around), captured
        self.cleared: list[np.ndarray] = []
        self.captured: list[np.ndarray] = []

    @property
    def steps(self) -> int:
        return self.spoken.steps

    @property
    def running(self) -> bool:
        executor = self.executor
        return self.steps < self.max_steps and not (bool(executor.done.all()) or executor.count == executor.cycles)

    def step(self) -> None:
        executor, speaker, count = self.executor, self.speaker, len(self.executor.done)
        done = executor.done.cpu().numpy()
        if self.steps:
            now = executor.now()
            speaker.append(now.e_m.cpu().numpy(), now.n_m.cpu().numpy(), now.height_m.cpu().numpy(), frozen=done)
        cleared, captured = executor.lateral.cleared.cpu().numpy(), executor.lateral.captured.cpu().numpy()
        # a flight that is done hears nothing more; one the executor has cleared or captured keeps its runway
        said = speaker.speak(active=~done, runway_locked=executor.runway_locked.cpu().numpy())
        self.cleared.append(cleared)
        self.captured.append(captured)
        heard = torch.full((count,), self.steps * self.step_s, dtype=torch.float64, device=self.device)
        self.spoken.say(np.where(said > 0, said - 1, UNCHANGED))
        for _ in range(executor.step_rows):
            if executor.count == executor.cycles:
                break
            executor.cycle(self.spoken.at(heard), torch.full((count,), executor.count * self.params.cycle_s,
                                                             dtype=torch.float64, device=self.device))

    def take(self, index: np.ndarray) -> None:
        """Keep the flights at ``index`` (the executor's, the speaker's and the words said): a flight taken twice flies
        on as two copies of itself."""
        rows = torch.as_tensor(index, device=self.device)
        self.executor.take(rows)
        self.spoken.take(rows)
        self.speaker.take(np.asarray(index))
        self.cleared = [row[index] for row in self.cleared]
        self.captured = [row[index] for row in self.captured]


def speak_and_fly(model: Prior, flights: Sequence[FlightSignals], geometries: Sequence[AirportGeometry],
                  inputs: FlightInputs, runways: Runways, charts: AirportCharts, approach_ias_mps: torch.Tensor,
                  limits: Sequence[float], words: Words, params: ExecutorParams, landings: Any, *,
                  generator: torch.Generator, temperature: float
                  ) -> tuple[Flown, np.ndarray, dict[int, np.ndarray], Speaker]:
    """`ClosedLoop` run to its end: ``(what was flown, the words said [B, steps, 6] with UNCHANGED where a column says
    nothing, the probability the prior put on what the masks removed [B, steps] per masked column, the speaker — the
    rows it read)``."""
    loop = ClosedLoop(model, flights, geometries, inputs, runways, charts, approach_ias_mps, limits, words, params,
                      landings, generator=generator, temperature=temperature)
    while loop.running:
        loop.step()
    return (loop.executor.flown(), loop.spoken.sentences(),
            {column: np.stack(masses, axis=1) for column, masses in loop.speaker.forbidden.items()}, loop.speaker)


def fly_reference(batch: replay.Batch, words: Words, params: ExecutorParams, *, device: torch.device) -> Flown:
    """Fly every flight's labelled words from its first predicted step, on the spec's clock over the observed rows from
    there."""
    step_s = words.spec.step_s
    grids = [reference_grid(r.words) for r in batch.readings]
    if params.word_clock == "time":
        clock: TimeClock | DistanceClock | TrackClock = TimeClock(params.cycle_s)
    else:
        rows = [len(r.words) for r in batch.readings]
        e_m = [f.e_m[N_LOOK:n] for f, n in zip(batch.signals, rows)]
        n_m = [f.n_m[N_LOOK:n] for f, n in zip(batch.signals, rows)]
        clock = (DistanceClock if params.word_clock == "distance" else TrackClock).of(e_m, n_m, step_s, params.cycle_s,
                                                                                    device=device)
    runways, charts, approach = _physics(batch, device)
    return fly(flight_inputs(batch.series, device=device, anchor=N_LOOK), Sentences(grids, words, device=device), clock,
               runways, charts, approach, params, words,
               time_limit_s=torch.tensor(limits_s(batch, params, step_s), dtype=torch.float64, device=device))


def flight_rows(batch: replay.Batch, flown: Flown, grids: Sequence[np.ndarray], words: Words, source: str,
                samples: Sequence[int | None], forbidden: dict[int, np.ndarray] | None) -> list[dict[str, Any]]:
    """One row per flight: its outcome and what was said (``samples``: each flight's sample number, None for the
    labelled words; ``forbidden``: the prior's probability on what the grammar forbids, per step, over the steps
    before the flight's end)."""
    rows = []
    step_rows = round(words.spec.step_s / flown.cycle_s)
    for j, (reading, grid, sample) in enumerate(zip(batch.readings, grids, samples)):
        # the steps said up to the flight's end: the step whose cycles ended it (later ones said nothing)
        steps = min(len(grid), int(flown.done_cycle[j]) // step_rows + 1)
        grid = np.asarray(grid)[:steps]
        runway = grid[:, RUNWAY][grid[:, RUNWAY] != UNCHANGED]
        outcome = outcome_of(flown, j, batch.geometries[j], int(runway[-1]), words.spec)
        said_after_first = (grid[1:] != UNCHANGED).sum(axis=0)
        approach = grid[:, APPROACH][grid[:, APPROACH] != UNCHANGED]
        rows.append({
            "dataset_id": batch.signals[j].dataset_id, "airport": batch.signals[j].airport,
            "stratum": flight_record(reading)["stratum"], "source": source, "sample": sample,
            "outcome": outcome.outcome, "end_s": outcome.end_row * flown.cycle_s,
            "observed_remaining_s": observed_remaining_s(reading, words.spec.step_s),
            "observed_runway": reading.runway_index, "first_runway": int(runway[0]), "last_runway": int(runway[-1]),
            "runway_changes": int((np.diff(runway) != 0).sum()), "steps_said": len(grid),
            "go_arounds": int((grid[:, APPROACH] == APPROACH_GO_AROUND).sum()),
            "cleared_at_end": bool(approach[-1] == APPROACH_CLEARED),
            "forbidden_mass": ({COLUMNS[c]: float(mass[j, :steps].mean()) for c, mass in forbidden.items()}
                               if forbidden is not None else None),
            "words_after_first": {name: int(said_after_first[c]) for c, name in enumerate(COLUMNS)},
        })
    return rows


def prior_rows(model: Prior, batch: replay.Batch, words: Words, params: ExecutorParams, landings: Any,
               samples: int, *, generator: torch.Generator, temperature: float
               ) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Every flight of ``batch`` flown ``samples`` times with the prior speaking (the executor on CPU): its
    `flight_rows` and the sentences said."""
    cpu = torch.device("cpu")
    repeated = replay.subset(batch, [j for j in range(len(batch.readings)) for _ in range(samples)])
    runways, charts, approach = _physics(repeated, cpu)
    flown, said, forbidden, _ = speak_and_fly(model, repeated.signals, repeated.geometries,
                                              flight_inputs(repeated.series, device=cpu, anchor=N_LOOK), runways,
                                              charts, approach, limits_s(repeated, params, words.spec.step_s), words,
                                              params, landings, generator=generator, temperature=temperature)
    grids = [said[j] for j in range(len(said))]
    return (flight_rows(repeated, flown, grids, words, "prior", [j % samples for j in range(len(grids))], forbidden),
            grids)


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Shares of each outcome and the rest, over ``rows``."""
    count = len(rows)
    landed = [r for r in rows if r["outcome"] == "landed"]
    timeouts = [r for r in rows if r["outcome"] == "timeout"]
    delta = [r["end_s"] - r["observed_remaining_s"] for r in landed]
    return {
        "flights": count,
        "outcomes": {name: sum(r["outcome"] == name for r in rows) / count for name in OUTCOMES},
        "first_runway_observed": sum(r["first_runway"] == r["observed_runway"] for r in rows) / count,
        "landed_on_observed_runway": (sum(r["last_runway"] == r["observed_runway"] for r in landed) / len(landed)
                                      if landed else None),
        "landing_minus_observed_s": ({"p10": float(np.percentile(delta, 10)), "p50": float(np.percentile(delta, 50)),
                                      "p90": float(np.percentile(delta, 90))} if delta else None),
        "words_after_first_per_flight": {name: float(np.mean([r["words_after_first"][name] for r in rows]))
                                         for name in COLUMNS},
        "runway_changes_per_flight": float(np.mean([r["runway_changes"] for r in rows])),
        "go_around_flights": sum(r["go_arounds"] > 0 for r in rows) / count,
        "cleared_at_end": sum(r["cleared_at_end"] for r in rows) / count,
        "timeouts_cleared": (sum(r["cleared_at_end"] for r in timeouts) / len(timeouts) if timeouts else None),
        "forbidden_mass_per_step": ({column: float(np.mean([r["forbidden_mass"][column] for r in rows]))
                                     for column in rows[0]["forbidden_mass"]} if rows[0]["forbidden_mass"] else None),
    }


def grouped(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """`summarise` in all, per airport, per airport × approach kind."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for key in ("all", r["airport"], f"{r['airport']} {r['stratum']}", r["stratum"]):
            groups[key].append(r)
    return {key: summarise(value) for key, value in sorted(groups.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the chosen prior run (prior_select)")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--per-airport", type=int, default=0, help="0: every flight of the split flown on its own dynamics")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--chunk", type=int, default=64, help="flights a batch (× samples closed loops)")
    parser.add_argument("--device", default="cuda", help="the prior's; the executor flies on CPU")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    prior_dir, instructions, executor_dir, out = map(resolved, (args.prior, args.instructions, args.executor, args.out))
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    git = git_state()
    if args.split == "val" and git["dirty"]:
        parser.error("the val readout runs from a clean tree")
    started = time.perf_counter()
    params, record, words = replay.open_executor(executor_dir, instructions)
    model, _, prior_config = load_prior(prior_dir, instructions)
    if prior_config["smoke"]:
        parser.error(f"{prior_dir} is a smoke run")
    model.to(torch.device(args.device))
    landings = (airport_landings(instructions, rosters(instructions))
                if VARIANTS[model.config.variant].landing_context else None)
    batch = replay.draw(instructions, args.split, words.spec, words, per_airport=args.per_airport, seed=args.seed)
    print(f"{len(batch.readings)} {args.split} flights ({batch.drawn['excluded']} not flown), "
          f"{time.perf_counter() - started:.0f}s", flush=True)

    cpu = torch.device("cpu")
    generator = torch.Generator(device=torch.device(args.device)).manual_seed(args.seed)
    order = sorted(range(len(batch.readings)), key=lambda j: len(batch.readings[j].words))
    rows: list[dict[str, Any]] = []
    sentences: list[np.ndarray] = []
    for start in range(0, len(order), args.chunk):
        part = replay.subset(batch, order[start: start + args.chunk])
        rows += flight_rows(part, fly_reference(part, words, params, device=cpu),
                            [reference_grid(r.words) for r in part.readings], words, "labelled",
                            [None] * len(part.readings), None)
        said, grids = prior_rows(model, part, words, params, landings, args.samples, generator=generator,
                                 temperature=args.temperature)
        rows += said
        sentences += grids
        done = start + len(part.readings)
        print(f"  {done}/{len(order)} flights, {time.perf_counter() - started:.0f}s", flush=True)

    readout = {source: grouped([r for r in rows if r["source"] == source]) for source in ("labelled", "prior")}
    landed_samples = Counter(r["dataset_id"] for r in rows if r["source"] == "prior" and r["outcome"] == "landed")
    spread = Counter(landed_samples[r["dataset_id"]] for r in rows if r["source"] == "labelled")
    out.mkdir(parents=True)
    lengths = np.array([len(g) for g in sentences], dtype=np.int64)
    np.savez_compressed(out / "sentences.npz", offsets=np.concatenate(([0], np.cumsum(lengths))),
                        words=np.concatenate(sentences).astype(np.int16),
                        dataset_id=np.array([r["dataset_id"] for r in rows if r["source"] == "prior"]))
    write_json_atomic(out / "generation.json", {
        "schema": GENERATION_SCHEMA, "written_utc": utc_now(), "git": git,
        "prior": {"directory": str(prior_dir), "variant": model.config.variant},
        "executor": {"directory": str(executor_dir), "sha256": record["sha256"]},
        "instructions": str(instructions), "split": args.split, "drawn": batch.drawn, "n_look": N_LOOK,
        "samples": args.samples, "temperature": args.temperature, "seed": args.seed,
        "readout": readout, "landed_samples_per_flight": {str(k): v for k, v in sorted(spread.items())},
        "flights": rows, "elapsed_s": time.perf_counter() - started})
    for source in ("labelled", "prior"):
        print(f"{source}:")
        for key in ("all", *STRATA, *sorted({f"{r['airport']} {r['stratum']}" for r in rows})):
            if key in readout[source]:
                part = readout[source][key]
                shares = "  ".join(f"{name} {share:.3f}" for name, share in part["outcomes"].items() if share)
                print(f"  {key:22s} n={part['flights']:5d}  {shares}  first runway observed "
                      f"{part['first_runway_observed']:.3f}  cleared at the end {part['cleared_at_end']:.3f}")
        if readout[source]["all"]["forbidden_mass_per_step"]:
            print(f"  forbidden by the grammar, mean probability a step: {readout[source]['all']['forbidden_mass_per_step']}")
    print(f"landed samples per flight (of {args.samples}): {dict(sorted(spread.items()))}")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
