"""Single-aircraft landing-reward fine-tuning of the prior (prior design §9.3): the prior speaks freely, the executor
flies, and a sentence that lands is reinforced against the others said to the same flight.

Each round:

1. **sentences** — ``--per-airport`` flights of the train days (their own dynamics, the replay gate's group; drawn with
   the round's seed, so rounds may share flights — counted) are each flown ``--samples`` times from their first predicted
   step with the prior speaking, exactly as in free generation (`prior_free_generation.speak_and_fly`);
2. **rewards** — 1 where the executor's judge has the sentence landed on a runway in the airport's landing direction at
   the time (`prior.landing_reward.landing_direction`), else 0; each sentence's advantage is its reward less its
   flight's mean. Only flights whose sentences differ are trained on (the others' advantages are all 0);
3. **one pass** over those sentences (`train.RewardTuner`: the advantage-weighted NLL of each sentence's own words,
   the pull to the frozen start model, the teacher-forced data term on the train days), from the previous round's
   weights (round 1: ``--prior``) — only the sentences of this round (they must come from the weights being trained);
4. **the readout on the select days** (`select_readout`): free generation on ``--select-per-airport`` flights ×
   ``--select-samples`` sentences, the same flights, seed and batches every round (round 0 is the start model); the
   teacher-forced NLL; and the share landed on a runway against the landing direction.

The round kept (``choice.json``): among the rounds whose select readout keeps the guards — landed on the observed runway
at most `GUARD_RUNWAY_DROP` below round 0's share, heading words per flight at most `GUARD_HEADING_GROWTH` × round 0's —
the one with the highest landed share, the earliest within `TIE_SHARE` of it. Val is not read here:
a round directory is a prior run `prior_free_generation` reads, once.

Writes into ``--out`` (a new directory, from a clean tree unless ``--smoke``): ``config.json``, ``round_00/readout.json``,
``round_<k>/{sentences.npz, sentences.json, checkpoint.pt, config.json, readout.json}``, ``history.json``,
``choice.json``.

    python run_ts.py prior_landing_reward --prior <the step-1 run> \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v4_20260924 \\
        --executor 4dTrajectory/outputs/POOLED/executor/<spec> --out 4dTrajectory/outputs/POOLED/prior/<name>
"""

from __future__ import annotations

import argparse
import copy
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.flights import flight_inputs
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.experiments.prior_free_generation import (
    _physics, flight_rows, grouped, limits_s, prior_rows, speak_and_fly, steps_said,
)
from ts_transformer.experiments.prior_train import PRIOR_CHECKPOINT_SCHEMA, load_prior, rosters
from ts_transformer.instructions.artefact import load_spec
from ts_transformer.instructions.words import UNCHANGED, Words
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.data import VARIANTS, Flight, Split, airport_landings, chain_record, load_split
from ts_transformer.prior.landing_reward import LANDED, group_advantages, landing_direction, rewards
from ts_transformer.prior.model import Prior
from ts_transformer.prior.scene import N_LOOK, Landings
from ts_transformer.prior.train import RewardConfig, RewardTuner, TrainConfig, evaluate
from ts_transformer.repo_layout import REPO_ROOT, git_state

LANDING_REWARD_SCHEMA = "ts-prior-landing-reward-v1"
#: Two rounds' select landed shares closer than this are a tie (about two binomial standard deviations over 2,000
#: sentences at 90 %).
TIE_SHARE = 0.015
#: The guards a round kept must keep (design §9.3, §10), against round 0 on the same select flights and seed.
GUARD_RUNWAY_DROP = 0.02
GUARD_HEADING_GROWTH = 1.2
#: Select flights a batch of the readout: the one generator hands each batch its samples in turn, so the batches are part
#: of the readout (64, as in every select readout of the prior's closed loop).
SELECT_CHUNK = 64


@dataclass(frozen=True)
class Sentences:
    """One round's sentences, flight-major (flight ``f``'s ``samples`` sentences at ``f·samples …``)."""

    flight: np.ndarray               # [S] the flight's index in the round's batch
    positions: list[np.ndarray]      # [N_LOOK + steps, 3] e, n, height: observed to the first predicted step, then flown
    said: list[np.ndarray]           # [steps, 6] the words said (UNCHANGED where a column says nothing)
    outcomes: list[str]
    runway: np.ndarray               # [S] the runway pointed at the end
    observed_runway: np.ndarray      # [S] the runway the flight landed on


def speak_sentences(model: Prior, batch: replay.Batch, samples: int, words: Words, params: ExecutorParams,
                    landings: Mapping[str, Landings] | None, *, generator: torch.Generator) -> Sentences:
    """Every flight of ``batch`` flown ``samples`` times with the prior speaking (the executor on CPU)."""
    cpu = torch.device("cpu")
    count = len(batch.readings)
    repeated = replay.subset(batch, [j for j in range(count) for _ in range(samples)])
    runways, charts, approach = _physics(repeated, cpu)
    flown, said, forbidden, speaker = speak_and_fly(model, repeated.signals, repeated.geometries,
                                                    flight_inputs(repeated.series, device=cpu, anchor=N_LOOK), runways,
                                                    charts, approach, limits_s(repeated, params, words.spec.step_s),
                                                    words, params, landings, generator=generator, temperature=1.0)
    grids = [said[j] for j in range(len(said))]
    rows = flight_rows(repeated, flown, grids, words, "prior", [j % samples for j in range(len(grids))], forbidden)
    step_rows = round(words.spec.step_s / flown.cycle_s)
    positions, cut = [], []
    for j in range(len(grids)):
        steps = steps_said(flown, j, len(grids[j]), step_rows)
        rows_read = N_LOOK + steps
        positions.append(np.column_stack((speaker.e[j, :rows_read], speaker.n[j, :rows_read],
                                          speaker.h[j, :rows_read])))
        cut.append(grids[j][:steps])
    return Sentences(np.repeat(np.arange(count), samples), positions, cut, [r["outcome"] for r in rows],
                     np.array([r["last_runway"] for r in rows]), np.array([r["observed_runway"] for r in rows]))


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


def join(parts: Sequence[Sentences], offsets: Sequence[int]) -> Sentences:
    """Sentences of consecutive chunks of one batch (``offsets``: each chunk's first flight)."""
    return Sentences(np.concatenate([p.flight + o for p, o in zip(parts, offsets)]),
                     [x for p in parts for x in p.positions], [x for p in parts for x in p.said],
                     [x for p in parts for x in p.outcomes], np.concatenate([p.runway for p in parts]),
                     np.concatenate([p.observed_runway for p in parts]))


def sentence_flights(batch: replay.Batch, sentences: Sentences, keep: np.ndarray, airports: Sequence[str],
                     step_s: float, landings: Mapping[str, Landings] | None) -> list[Flight]:
    """The sentences at ``keep`` as training flights: the rows the prior read (`data.chain_record`), its own words as
    the targets, every column asked."""
    flights = []
    for s in keep:
        f = int(sentences.flight[s])
        signals, said, position = batch.signals[f], sentences.said[s], sentences.positions[s]
        classes = np.where(said != UNCHANGED, said + 1, 0)
        flights.append(chain_record(signals, position[:, 0], position[:, 1], position[:, 2], said, classes,
                                    np.ones(said.shape, dtype=bool), batch.geometries[f],
                                    landings[signals.airport] if landings is not None else None,
                                    airports.index(signals.airport), batch.readings[f].capture_row, step_s))
    return flights


def reward_summary(batch: replay.Batch, sentences: Sentences, earned: np.ndarray, samples: int) -> dict[str, Any]:
    """How the round's sentences did: the reward in all and per airport, the landed ones on a runway against the
    direction, the flights with a contrast, the landed ones on the observed runway."""
    landed = np.array([o == LANDED for o in sentences.outcomes])
    by_flight = earned.reshape(-1, samples)
    per_airport: dict[str, list[float]] = defaultdict(list)
    for s, value in enumerate(earned):
        per_airport[batch.signals[int(sentences.flight[s])].airport].append(float(value))
    return {
        "sentences": len(earned), "flights": len(by_flight), "reward_mean": float(earned.mean()),
        "landed": float(landed.mean()), "landed_against_the_direction": float((landed & (earned == 0)).mean()),
        "landed_on_observed_runway": float((landed & (sentences.runway == sentences.observed_runway)).sum()
                                           / max(landed.sum(), 1)),
        "flights_with_contrast": int((by_flight.min(axis=1) != by_flight.max(axis=1)).sum()),
        "rewards_per_flight": {str(k): v for k, v in sorted(Counter(by_flight.sum(axis=1).astype(int)).items())},
        "reward_by_airport": {code: float(np.mean(v)) for code, v in sorted(per_airport.items())},
        "outcomes": dict(Counter(sentences.outcomes)),
    }


def write_sentences(path: Path, batch: replay.Batch, sentences: Sentences, earned: np.ndarray,
                    advantages: np.ndarray) -> None:
    steps = np.array([len(s) for s in sentences.said])
    np.savez_compressed(path, dataset_id=np.array([s.dataset_id for s in batch.signals]), flight=sentences.flight,
                        step_offsets=np.concatenate(([0], np.cumsum(steps))),
                        said=np.concatenate(sentences.said).astype(np.int16), outcome=np.array(sentences.outcomes),
                        runway=sentences.runway, reward=earned, advantage=advantages)


def against_the_direction(rows: Sequence[Mapping[str, Any]], directions: Mapping[str, np.ndarray]) -> float:
    """Of a free generation's sentences (`flight_rows`), the share that landed on a runway against the airport's landing
    direction (``directions``: each flight's, by dataset id)."""
    return float(np.mean([r["outcome"] == LANDED and not directions[r["dataset_id"]][r["last_runway"]] for r in rows]))


def guarded_choice(history: Sequence[Mapping[str, Any]]) -> tuple[int, list[int]]:
    """``(the round kept, the rounds excluded)``: among the rounds within the guards of round 0, the highest select
    landed share, the earliest within `TIE_SHARE` of it. A round that landed nothing on select (no runway share to
    guard) is excluded."""
    start = history[0]["select"]
    excluded = [row["round"] for row in history
                if row["select"]["landed_on_observed_runway"] is None
                or row["select"]["landed_on_observed_runway"] < start["landed_on_observed_runway"] - GUARD_RUNWAY_DROP
                or row["select"]["words_after_first_per_flight"]["heading"]
                > GUARD_HEADING_GROWTH * start["words_after_first_per_flight"]["heading"]]
    candidates = [row for row in history if row["round"] not in excluded]
    best = max(row["select_landed"] for row in candidates)
    return next(row["round"] for row in candidates if row["select_landed"] >= best - TIE_SHARE), excluded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the start: the step-1 run (prior_select's choice)")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--per-airport", type=int, default=400, help="train flights an airport a round")
    parser.add_argument("--samples", type=int, default=8, help="sentences a flight")
    parser.add_argument("--select-per-airport", type=int, default=200)
    parser.add_argument("--select-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--chunk", type=int, default=32, help="flights a batch (× samples closed loops)")
    parser.add_argument("--device", default="cuda", help="the prior's; the executor flies on CPU")
    parser.add_argument("--smoke", action="store_true", help="a dirty tree allowed; the runs are marked smoke")
    for field, default in asdict(RewardConfig()).items():
        parser.add_argument(f"--{field.replace('_', '-')}", type=type(default), default=default)
    args = parser.parse_args(argv)

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    prior_dir, instructions, executor_dir, out = map(resolved, (args.prior, args.instructions, args.executor, args.out))
    if out.exists():
        parser.error(f"{out} exists; a fine-tuning run is never overwritten")
    if args.samples < 2:
        parser.error("a flight's sentences are compared with each other: --samples ≥ 2")
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("the tree has uncommitted changes; a fine-tuning run is made at a commit")
    config = RewardConfig(**{f: getattr(args, f) for f in asdict(RewardConfig())})
    started = time.perf_counter()
    device = torch.device(args.device)
    params, record, words = replay.open_executor(executor_dir, instructions)
    spec = load_spec(instructions)
    model, _, start_config = load_prior(prior_dir, instructions)
    if start_config["smoke"]:
        parser.error(f"{prior_dir} is a smoke run")
    model.to(device)
    reference = copy.deepcopy(model)
    variant = model.config.variant
    every_landing = airport_landings(instructions, rosters(instructions))
    landings = every_landing if VARIANTS[variant].landing_context else None
    select_batch = replay.draw(instructions, "select", spec, words, per_airport=args.select_per_airport,
                               seed=args.seed)
    select = load_split(instructions, "select", spec, words, variant, landings=every_landing,
                        airports=model.config.airports)
    data = load_split(instructions, "train", spec, words, variant, landings=every_landing,
                      airports=model.config.airports)
    select_directions = {signals.dataset_id: landing_direction(signals, geometry, every_landing[signals.airport])
                         for signals, geometry in zip(select_batch.signals, select_batch.geometries)}
    out.mkdir(parents=True)
    write_json_atomic(out / "config.json", {
        "schema": LANDING_REWARD_SCHEMA, "written_utc": utc_now(), "git": git, "smoke": args.smoke,
        "prior": {"directory": str(prior_dir), "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt")},
        "executor": {"directory": str(executor_dir), "sha256": record["sha256"]}, "instructions": str(instructions),
        "optimiser": asdict(config), "rounds": args.rounds, "per_airport": args.per_airport, "samples": args.samples,
        "select": {"per_airport": args.select_per_airport, "samples": args.select_samples, "drawn": select_batch.drawn},
        "data_flights": len(data.flights), "seed": args.seed, "tie_share": TIE_SHARE,
        "guards": {"runway_drop": GUARD_RUNWAY_DROP, "heading_growth": GUARD_HEADING_GROWTH}, "n_look": N_LOOK})

    def log(line: str) -> None:
        print(f"{line}  [{time.perf_counter() - started:.0f}s]", flush=True)

    def read_select(round_number: int) -> dict[str, Any]:
        readout = select_readout(model, select_batch, select, words, params, landings, samples=args.select_samples,
                                 seed=args.seed, chunk=SELECT_CHUNK, device=device)
        readout["landed_against_the_direction"] = against_the_direction(readout["flights"], select_directions)
        part = readout["free_generation"]["all"]
        log(f"round {round_number}: select landed {part['outcomes']['landed']:.3f}  TF NLL "
            f"{readout['teacher_forced']['nll_per_step']:.4f}  landed on the observed runway "
            f"{part['landed_on_observed_runway']}  against the direction {readout['landed_against_the_direction']:.3f}  "
            f"heading words {part['words_after_first_per_flight']['heading']:.1f}")
        return readout

    def history_row(round_number: int, readout: dict[str, Any], **more: Any) -> dict[str, Any]:
        part = readout["free_generation"]["all"]
        return {"round": round_number, "select_landed": part["outcomes"]["landed"],
                "select_tf_nll": readout["teacher_forced"]["nll_per_step"],
                "select_landed_against_the_direction": readout["landed_against_the_direction"],
                "select": {k: v for k, v in part.items() if k != "outcomes"}, "select_outcomes": part["outcomes"],
                **more}

    directory = out / "round_00"
    directory.mkdir()
    readout = read_select(0)
    write_json_atomic(directory / "readout.json", readout)
    history = [history_row(0, readout)]
    tuner = RewardTuner(model, reference, config, device, seed=args.seed)
    seen: Counter = Counter()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for round_number in range(1, args.rounds + 1):
        directory = out / f"round_{round_number:02d}"
        directory.mkdir()
        batch = replay.draw(instructions, "train", spec, words, per_airport=args.per_airport,
                            seed=args.seed + round_number)
        repeats = sum(seen[s.dataset_id] > 0 for s in batch.signals)
        seen.update(s.dataset_id for s in batch.signals)
        order = sorted(range(len(batch.readings)), key=lambda j: len(batch.readings[j].words))
        batch = replay.subset(batch, order)
        model.eval()
        starts = list(range(0, len(order), args.chunk))
        sentences = join([speak_sentences(model, replay.subset(batch, list(range(s, min(s + args.chunk, len(order))))),
                                          args.samples, words, params, landings, generator=generator)
                          for s in starts], starts)
        directions = [landing_direction(signals, geometry, every_landing[signals.airport])
                      for signals, geometry in zip(batch.signals, batch.geometries)]
        earned = rewards(sentences.outcomes, sentences.runway, [directions[f] for f in sentences.flight])
        advantages = group_advantages(earned.reshape(-1, args.samples)).reshape(-1)
        keep = np.flatnonzero(np.repeat(earned.reshape(-1, args.samples).std(axis=1) > 0, args.samples))
        write_sentences(directory / "sentences.npz", batch, sentences, earned, advantages)
        summary = {**reward_summary(batch, sentences, earned, args.samples), "trained_on": len(keep),
                   "repeated_flights": repeats, "drawn": batch.drawn}
        write_json_atomic(directory / "sentences.json", summary)
        log(f"round {round_number}: {len(earned)} sentences ({repeats} flights flown in an earlier round), reward "
            f"{summary['reward_mean']:.3f}, {summary['flights_with_contrast']} flights with a contrast")
        flights = sentence_flights(batch, sentences, keep, model.config.airports, spec.step_s, landings)
        passed = tuner.one_pass(Split(flights, data.airports, data.candidates, data.runways, data.courses,
                                      data.classes, variant), advantages[keep], data)
        log(f"round {round_number}: one pass over {len(flights)} sentences, reward term {passed['reward_mean']:.4f}, "
            f"KL {passed['kl_mean']:.4f}, data NLL {passed['data_mean']:.4f}")
        torch.save({"schema": PRIOR_CHECKPOINT_SCHEMA, "model_config": model.config.to_dict(),
                    "train_config": start_config["train"], "state": copy.deepcopy(model.state_dict()),
                    "spec_sha256": spec.sha256}, directory / "checkpoint.pt")
        write_json_atomic(directory / "config.json", {
            **start_config, "written_utc": utc_now(), "git": git, "smoke": args.smoke,
            "fine_tuning": {"schema": LANDING_REWARD_SCHEMA, "from": str(prior_dir), "round": round_number,
                            "optimiser": asdict(config), "samples": args.samples}})
        readout = read_select(round_number)
        write_json_atomic(directory / "readout.json", readout)
        history.append(history_row(round_number, readout, train_pass=passed,
                                   sentences={k: v for k, v in summary.items() if k != "drawn"}))
        write_json_atomic(out / "history.json", {"rounds": history})
    kept, excluded = guarded_choice(history)
    write_json_atomic(out / "choice.json", {
        "rule": f"within the guards (landed on the observed runway ≥ round 0's − {GUARD_RUNWAY_DROP}, heading words ≤ "
                f"{GUARD_HEADING_GROWTH} × round 0's), the highest select landed share; within {TIE_SHARE} the earliest",
        "landed": [row["select_landed"] for row in history], "excluded_by_the_guards": excluded, "round": kept,
        "directory": str(out / f"round_{kept:02d}") if kept else str(prior_dir)})
    log(f"kept round {kept} (the guards excluded {excluded}) → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
