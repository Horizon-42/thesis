"""Single-aircraft closed-loop supervised fine-tuning of the prior (prior design §9.2): CAT-K by segments.

Each round:

1. **chains** — ``--per-airport`` flights of the train days (their own dynamics, the replay gate's group; drawn with the
   round's seed, so rounds may share flights — the readout counts how many) are flown from their first predicted step
   in ``--branches`` branches each, the prior speaking in every branch exactly as in free generation
   (`prior_free_generation.ClosedLoop`). Every ``--segment-steps`` steps the branch whose position is closest to the
   observed flight's at that row (3D metres; past the observed last row, that row) is kept and the others become
   copies of it (`ClosedLoop.take`: the executor's state, the rows the prior read, the words said) — a branch the
   executor has finished other than by crossing the threshold captured (short of the runway, out of the dynamics, out
   of time) only when every branch has. What is left at the end is the flight's chain;
2. **targets** — at every row of a chain, the labelled words re-read against the words the chain has in force
   (`prior.relabel`);
3. **one pass** over every chain of this round and the rounds before (DAgger's aggregate), from the previous round's
   weights (round 1: the ``--prior`` run), the loss of pretraining on the relabelled targets and no teacher-forced data
   (CAT-K's second stage);
4. **the readout on the select days**: free generation, ``--select-per-airport`` flights × ``--select-samples``
   sentences, the same flights and seed every round (round 0 is the prior before fine-tuning), and the teacher-forced
   negative log-likelihood per predicted step.

The round kept (``choice.json``) has the highest landed share on select; a round within `TIE_SHARE` of it counts as a
tie and the earliest such round wins. Val is not read here: the chosen round's directory is a prior run
`prior_free_generation` reads (``config.json`` + ``checkpoint.pt``), once.

Writes into ``--out`` (a new directory, from a clean tree unless ``--smoke``): ``config.json``, ``round_00/readout.json``,
``round_<k>/{chains.npz, chains.json, checkpoint.pt, config.json, readout.json}``, ``history.json``, ``choice.json``.

    python run_ts.py prior_closed_loop --prior 4dTrajectory/outputs/POOLED/prior/v3_step1_20260924/full_s1337 \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v4_20260924 \\
        --executor 4dTrajectory/outputs/POOLED/executor/v7_20260925 --out 4dTrajectory/outputs/POOLED/prior/<name>
"""

from __future__ import annotations

import argparse
import copy
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.flights import FlightInputs, flight_inputs
from ts_transformer.autopilot.frame import AirportCharts, read_state
from ts_transformer.autopilot.judge import OUTCOMES, outcome_of
from ts_transformer.autopilot.lateral import Runways, relative
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.experiments.prior_free_generation import ClosedLoop, _physics, grouped, limits_s, prior_rows
from ts_transformer.experiments.prior_train import PRIOR_CHECKPOINT_SCHEMA, load_prior, rosters
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import load_spec
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.words import COLUMNS, RUNWAY, UNCHANGED, Words
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.data import VARIANTS, Flight, Split, airport_landings, chain_record, load_split
from ts_transformer.prior.model import Prior
from ts_transformer.prior.relabel import Targets, relabel
from ts_transformer.prior.scene import N_LOOK, Landings
from ts_transformer.prior.train import FineTuneConfig, FineTuner, TrainConfig, evaluate
from ts_transformer.repo_layout import REPO_ROOT, git_state

CLOSED_LOOP_SCHEMA = "ts-prior-closed-loop-v1"
#: Two rounds' select landed shares closer than this are a tie (about two binomial standard deviations over 2,000
#: sentences at 90 %, design §9.2).
TIE_SHARE = 0.015


@dataclass(frozen=True)
class ChainConfig:
    branches: int = 4
    segment_steps: int = 10
    window: int = 5
    temperature: float = 1.0


@dataclass(frozen=True)
class Chains:
    """One round's chains, flight by flight (the batch's order)."""

    positions: list[np.ndarray]    # [N_LOOK + steps, 3] e, n, height: observed to the first predicted step, then flown
    said: list[np.ndarray]         # [steps, 6] the words said (UNCHANGED where a column says nothing)
    cleared: list[np.ndarray]      # [steps] the executor had cleared when the step was said
    captured: list[np.ndarray]     # [steps] … had captured the line
    outcomes: list[str]
    kept_m: list[np.ndarray]       # per segment while the chain flies: the kept branch's distance to the observed
    split_segments: list[int]      # segments (while the chain flies) whose branches did not all end at one place
    set_aside: list[int]           # segments in which a finished-without-crossing branch was passed over


def observed_positions(batch: replay.Batch) -> list[np.ndarray]:
    """Each flight's observed rows up to its sentence's last (the landing): ``[rows, 3]`` e, n, height."""
    return [np.column_stack((f.e_m, f.n_m, f.altitude_m))[: len(r.words)] for f, r in zip(batch.signals, batch.readings)]


def segment_end(loop: ClosedLoop) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(where [B, 3], done [B], failed [B])`` at the end of the steps flown: a flight the executor is done with is
    where it was at the end of the cycle that finished it (it is not flown further); it failed unless it was done by
    crossing its pointed runway's threshold while captured (the executor's landing event, `Executor.cycle`)."""
    executor = loop.executor
    state = executor.state.clone()
    for b in torch.nonzero(executor.done).flatten().tolist():
        state[b] = executor.states[int(executor.done_cycle[b]) + 1][b]
    now = read_state(state, executor.charts)
    e0, n0, course, _elevation, _crossing = executor.runways.pointed(executor.lateral.runway)
    before, _right, _off = relative(now, e0, n0, course)
    crossed = executor.lateral.captured & (before <= 0.0)
    done = executor.done.cpu().numpy()
    return (np.column_stack([now.e_m.cpu().numpy(), now.n_m.cpu().numpy(), now.height_m.cpu().numpy()]), done,
            done & ~crossed.cpu().numpy())


def fly_chains(model: Prior, flights: Sequence[FlightSignals], geometries: Sequence[AirportGeometry],
               inputs: FlightInputs, runways: Runways, charts: AirportCharts, approach_ias_mps: torch.Tensor,
               limits: Sequence[float], observed: Sequence[np.ndarray], words: Words, params: ExecutorParams,
               landings: Mapping[str, Landings] | None, config: ChainConfig, *, generator: torch.Generator) -> Chains:
    """Every flight flown in ``config.branches`` branches, the closest to its ``observed`` rows (``[rows, 3]``) kept
    every ``config.segment_steps`` steps (the module's step 1); the arguments are `ClosedLoop`'s, one per flight."""
    count, branches = len(flights), config.branches
    copies = np.repeat(np.arange(count), branches)
    rows = torch.as_tensor(copies, device=inputs.initial_state.device)
    loop = ClosedLoop(model, [flights[j] for j in copies], [geometries[j] for j in copies], inputs.take(rows),
                      runways.take(rows), charts.take(rows), approach_ias_mps[rows], [limits[j] for j in copies],
                      words, params, landings, generator=generator, temperature=config.temperature)
    kept: list[list[float]] = [[] for _ in range(count)]
    split, set_aside = [0] * count, [0] * count
    over = np.zeros(count, dtype=bool)                        # the kept branch was done at the last choice
    while True:
        loop.step()
        end = not loop.running
        if end or loop.steps % config.segment_steps == 0:
            row = N_LOOK + loop.steps
            here, done, failed = segment_end(loop)
            here, done, failed = here.reshape(count, branches, 3), done.reshape(count, branches), \
                failed.reshape(count, branches)
            target = np.stack([o[min(row, len(o) - 1)] for o in observed])[:, None, :]
            distance = np.linalg.norm(here - target, axis=2)
            distance[~np.isfinite(distance)] = np.inf
            passed_over = failed & ~failed.all(axis=1, keepdims=True)
            keep = np.where(passed_over, np.inf, distance).argmin(axis=1)
            for b in np.flatnonzero(~over):
                kept[b].append(float(distance[b, keep[b]]))
                split[b] += int(np.ptp(here[b], axis=0).max() > 0.0)
                set_aside[b] += int(passed_over[b].any())
            over = done[np.arange(count), keep]
            index = np.arange(count) * branches + keep
            loop.take(index if end else np.repeat(index, branches))
        if end:
            break
    flown, said = loop.executor.flown(), loop.spoken.sentences()
    step_rows = loop.executor.step_rows
    cleared, captured = np.stack(loop.cleared, axis=1), np.stack(loop.captured, axis=1)
    speaker = loop.speaker
    positions, sentences, outcomes, cleared_out, captured_out = [], [], [], [], []
    for b in range(count):
        steps = min(said.shape[1], int(flown.done_cycle[b]) // step_rows + 1)
        rows = N_LOOK + steps
        positions.append(np.column_stack((speaker.e[b, :rows], speaker.n[b, :rows], speaker.h[b, :rows])))
        sentences.append(said[b, :steps])
        cleared_out.append(cleared[b, :steps])
        captured_out.append(captured[b, :steps])
        runway = said[b, :steps, RUNWAY][said[b, :steps, RUNWAY] != UNCHANGED]
        outcomes.append(outcome_of(flown, b, geometries[b], int(runway[-1]), words.spec).outcome)
    return Chains(positions, sentences, cleared_out, captured_out, outcomes, [np.array(k) for k in kept], split,
                  set_aside)


def fly_batch_chains(model: Prior, batch: replay.Batch, words: Words, params: ExecutorParams,
                     landings: Mapping[str, Landings] | None, config: ChainConfig, *,
                     generator: torch.Generator) -> Chains:
    """`fly_chains` of a drawn batch, flown from each flight's first predicted step (the executor on CPU)."""
    cpu = torch.device("cpu")
    runways, charts, approach = _physics(batch, cpu)
    return fly_chains(model, batch.signals, batch.geometries, flight_inputs(batch.series, device=cpu, anchor=N_LOOK),
                      runways, charts, approach, limits_s(batch, params, words.spec.step_s), observed_positions(batch),
                      words, params, landings, config, generator=generator)


def chain_flights(batch: replay.Batch, chains: Chains, airports: Sequence[str], words: Words,
                  landings: Mapping[str, Landings] | None, window: int
                  ) -> tuple[list[Flight], list[Targets], dict[str, Any]]:
    """The chains as training flights (`data.chain_record`, targets from `relabel`), the targets, and what decided
    them (`relabel`'s counts over every column-step after the first, as shares, and the share of column-steps each
    column is asked to say a word)."""
    spec = words.spec
    flights, targets = [], []
    counts: Counter = Counter()
    for j, (signals, reading) in enumerate(zip(batch.signals, batch.readings)):
        position, said = chains.positions[j], chains.said[j]
        relabelled = relabel(reading.words, said, chains.cleared[j], chains.captured[j], position[N_LOOK:, 2], spec,
                             words, window=window)
        targets.append(relabelled)
        counts.update(relabelled.counts)
        flights.append(chain_record(signals, position[:, 0], position[:, 1], position[:, 2], said, relabelled.classes,
                                    relabelled.asked, batch.geometries[j],
                                    landings[signals.airport] if landings is not None else None,
                                    airports.index(signals.airport), reading.capture_row, spec.step_s))
    later = np.concatenate([t.classes[1:] for t in targets])
    asked = np.concatenate([t.asked[1:] for t in targets])
    cells = later.size
    return flights, targets, {
        "column_steps_after_the_first": cells, "shares": {reason: n / cells for reason, n in sorted(counts.items())},
        "say_share_by_column": dict(zip(COLUMNS, ((later > 0) & asked).mean(axis=0).tolist())),
        "asked_share_by_column": dict(zip(COLUMNS, asked.mean(axis=0).tolist()))}


def write_chains(path: Path, batch: replay.Batch, chains: Chains, targets: Sequence[Any]) -> None:
    rows = np.array([len(p) for p in chains.positions])
    steps = np.array([len(s) for s in chains.said])
    np.savez_compressed(
        path, dataset_id=np.array([s.dataset_id for s in batch.signals]),
        row_offsets=np.concatenate(([0], np.cumsum(rows))), step_offsets=np.concatenate(([0], np.cumsum(steps))),
        positions=np.concatenate(chains.positions), said=np.concatenate(chains.said).astype(np.int16),
        cleared=np.concatenate(chains.cleared), captured=np.concatenate(chains.captured),
        classes=np.concatenate([t.classes for t in targets]).astype(np.int16),
        asked=np.concatenate([t.asked for t in targets]), outcome=np.array(chains.outcomes))


def chain_summary(chains: Chains, counts: Mapping[str, Any], segment_steps: int) -> dict[str, Any]:
    """How close the chains stayed, how often the branches parted, how they ended, and the targets' counts."""
    longest = max(len(k) for k in chains.kept_m)
    by_segment = [[float(k[s]) for k in chains.kept_m if len(k) > s] for s in range(longest)]
    said_after_first = np.concatenate([(s[1:] != UNCHANGED).sum(axis=0)[None] for s in chains.said])
    return {
        "flights": len(chains.said),
        "outcomes": {name: chains.outcomes.count(name) / len(chains.outcomes) for name in OUTCOMES},
        "kept_distance_m_by_segment": [{"segment_end_step": (s + 1) * segment_steps, "flights": len(v),
                                        "p50": float(np.median(v)), "p90": float(np.percentile(v, 90))}
                                       for s, v in enumerate(by_segment) if len(v) >= 20],
        "segments_parted": sum(chains.split_segments) / sum(len(k) for k in chains.kept_m),
        "segments_a_failed_branch_passed_over": sum(chains.set_aside) / sum(len(k) for k in chains.kept_m),
        "words_after_first_per_flight": dict(zip(COLUMNS, said_after_first.mean(axis=0).tolist())),
        "targets": dict(counts),
    }


def select_readout(model: Prior, batch: replay.Batch, select: Split, words: Words, params: ExecutorParams,
                   landings: Mapping[str, Landings] | None, *, samples: int, seed: int, chunk: int,
                   device: torch.device) -> dict[str, Any]:
    """Free generation on the select flights (the same flights and seed every round) and the teacher-forced NLL."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    order = sorted(range(len(batch.readings)), key=lambda j: len(batch.readings[j].words))
    rows: list[dict[str, Any]] = []
    for start in range(0, len(order), chunk):
        part = replay.subset(batch, order[start: start + chunk])
        rows += prior_rows(model, part, words, params, landings, samples, generator=generator, temperature=1.0)[0]
    return {"free_generation": grouped(rows),
            "teacher_forced": evaluate(model, select, TrainConfig(), device), "flights": rows}


def choose(landed: Sequence[float]) -> int:
    """The round kept: the highest select landed share, the earliest within `TIE_SHARE` of it."""
    best = max(landed)
    return next(r for r, share in enumerate(landed) if share >= best - TIE_SHARE)


def round_config(start: Mapping[str, Any], start_dir: Path, round_number: int, chain: ChainConfig,
                 optimiser: FineTuneConfig, chains: int, git: Mapping[str, Any], smoke: bool) -> dict[str, Any]:
    """A round's ``config.json``: the start run's (what `load_prior` checks), and what the fine-tuning did."""
    return {**start, "written_utc": utc_now(), "git": dict(git), "smoke": smoke,
            "fine_tuning": {"schema": CLOSED_LOOP_SCHEMA, "from": str(start_dir), "round": round_number,
                            "chain": asdict(chain), "optimiser": asdict(optimiser), "chains_trained_on": chains}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the chosen prior run (prior_select)")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--per-airport", type=int, default=800, help="train flights an airport a round")
    parser.add_argument("--select-per-airport", type=int, default=200)
    parser.add_argument("--select-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--chunk", type=int, default=64, help="flights a batch (× branches closed loops)")
    parser.add_argument("--device", default="cuda", help="the prior's; the executor flies on CPU")
    parser.add_argument("--smoke", action="store_true", help="a dirty tree allowed; the runs are marked smoke")
    for fields in (ChainConfig, FineTuneConfig):
        for field, default in asdict(fields()).items():
            parser.add_argument(f"--{field.replace('_', '-')}", type=type(default), default=default)
    args = parser.parse_args(argv)

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    prior_dir, instructions, executor_dir, out = map(resolved, (args.prior, args.instructions, args.executor, args.out))
    if out.exists():
        parser.error(f"{out} exists; a fine-tuning run is never overwritten")
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("the tree has uncommitted changes; a fine-tuning run is made at a commit")
    chain = ChainConfig(**{f: getattr(args, f) for f in asdict(ChainConfig())})
    optimiser = FineTuneConfig(**{f: getattr(args, f) for f in asdict(FineTuneConfig())})
    started = time.perf_counter()
    device = torch.device(args.device)
    params, record, words = replay.open_executor(executor_dir, instructions)
    spec = load_spec(instructions)
    model, _, start_config = load_prior(prior_dir, instructions)
    if start_config["smoke"]:
        parser.error(f"{prior_dir} is a smoke run")
    model.to(device)
    variant = model.config.variant
    every_landing = airport_landings(instructions, rosters(instructions))
    landings = every_landing if VARIANTS[variant].landing_context else None
    select_batch = replay.draw(instructions, "select", spec, words, per_airport=args.select_per_airport,
                               seed=args.seed)
    select = load_split(instructions, "select", spec, words, variant, landings=every_landing,
                        airports=model.config.airports)
    out.mkdir(parents=True)
    write_json_atomic(out / "config.json", {
        "schema": CLOSED_LOOP_SCHEMA, "written_utc": utc_now(), "git": git, "smoke": args.smoke,
        "prior": {"directory": str(prior_dir), "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt")},
        "executor": {"directory": str(executor_dir), "sha256": record["sha256"]}, "instructions": str(instructions),
        "chain": asdict(chain), "optimiser": asdict(optimiser), "rounds": args.rounds,
        "per_airport": args.per_airport, "select": {"per_airport": args.select_per_airport,
                                                    "samples": args.select_samples, "drawn": select_batch.drawn},
        "seed": args.seed, "tie_share": TIE_SHARE, "n_look": N_LOOK})

    def log(line: str) -> None:
        print(f"{line}  [{time.perf_counter() - started:.0f}s]", flush=True)

    def read_select(round_number: int) -> dict[str, Any]:
        readout = select_readout(model, select_batch, select, words, params, landings, samples=args.select_samples,
                                 seed=args.seed, chunk=args.chunk, device=device)
        part = readout["free_generation"]["all"]
        log(f"round {round_number}: select landed {part['outcomes']['landed']:.3f}  TF NLL "
            f"{readout['teacher_forced']['nll_per_step']:.4f}  first runway observed {part['first_runway_observed']:.3f}")
        return readout

    directory = out / "round_00"
    directory.mkdir()
    readout = read_select(0)
    write_json_atomic(directory / "readout.json", readout)
    history = [{"round": 0, "select_landed": readout["free_generation"]["all"]["outcomes"]["landed"],
                "select_tf_nll": readout["teacher_forced"]["nll_per_step"],
                "select": {k: v for k, v in readout["free_generation"]["all"].items() if k != "outcomes"}}]
    tuner = FineTuner(model, optimiser, device, seed=args.seed)
    trained: list[Flight] = []
    seen: Counter = Counter()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for round_number in range(1, args.rounds + 1):
        directory = out / f"round_{round_number:02d}"
        directory.mkdir()
        batch = replay.draw(instructions, "train", spec, words, per_airport=args.per_airport,
                            seed=args.seed + round_number)
        repeats = sum(seen[s.dataset_id] > 0 for s in batch.signals)
        seen.update(s.dataset_id for s in batch.signals)
        model.eval()
        order = sorted(range(len(batch.readings)), key=lambda j: len(batch.readings[j].words))
        batch = replay.subset(batch, order)
        parts = [fly_batch_chains(model, replay.subset(batch, list(range(s, min(s + args.chunk, len(order))))),
                                  words, params, landings, chain, generator=generator)
                 for s in range(0, len(order), args.chunk)]
        chains = Chains(*[sum((getattr(p, name) for p in parts), []) for name in Chains.__dataclass_fields__])
        flights, targets, counts = chain_flights(batch, chains, model.config.airports, words, landings,
                                                 chain.window)
        write_chains(directory / "chains.npz", batch, chains, targets)
        summary = {**chain_summary(chains, counts, chain.segment_steps), "repeated_flights": repeats,
                   "drawn": batch.drawn}
        write_json_atomic(directory / "chains.json", summary)
        log(f"round {round_number}: {len(flights)} chains ({repeats} flown in an earlier round), landed "
            f"{summary['outcomes']['landed']:.3f}, segments parted {summary['segments_parted']:.3f}")
        trained += flights
        passed = tuner.one_pass(Split(trained, select.airports, select.candidates, select.runways, select.courses,
                                      select.classes, variant))
        log(f"round {round_number}: one pass over {len(trained)} chains, train NLL {passed['nll_per_step']:.4f}")
        torch.save({"schema": PRIOR_CHECKPOINT_SCHEMA, "model_config": model.config.to_dict(),
                    "train_config": start_config["train"], "state": copy.deepcopy(model.state_dict()),
                    "spec_sha256": spec.sha256}, directory / "checkpoint.pt")
        write_json_atomic(directory / "config.json", round_config(start_config, prior_dir, round_number, chain,
                                                                  optimiser, len(trained), git, args.smoke))
        readout = read_select(round_number)
        write_json_atomic(directory / "readout.json", readout)
        history.append({"round": round_number, "select_landed": readout["free_generation"]["all"]["outcomes"]["landed"],
                        "select_tf_nll": readout["teacher_forced"]["nll_per_step"], "train_pass": passed,
                        "chains": {k: v for k, v in summary.items() if k != "drawn"},
                        "select": {k: v for k, v in readout["free_generation"]["all"].items() if k != "outcomes"}})
        write_json_atomic(out / "history.json", {"rounds": history})
    kept = choose([row["select_landed"] for row in history])
    write_json_atomic(out / "choice.json", {"rule": f"highest select landed share; within {TIE_SHARE} the earliest",
                                            "landed": [row["select_landed"] for row in history], "round": kept,
                                            "directory": str(out / f"round_{kept:02d}") if kept else str(prior_dir)})
    log(f"kept round {kept} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
