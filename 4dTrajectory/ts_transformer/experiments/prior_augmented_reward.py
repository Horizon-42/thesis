"""Post-training stage 2 (post-training design §3–§5): augmented starts, the glidepath edge, reward and base pull only.

Each round:

1. **starts** — ``--per-airport`` train-day flights (their own dynamics, the replay gate's group) drawn with the round's
   seed from a pool of `POOL_FACTOR` × that many, each moved by a fresh augmentation (`prior.augment`: rotated about the
   airport, raised, sped up) that keeps the start plausible (`prior_free_generation.augmented_starts`); a flight none of
   `AUGMENT_TRIES` draws fits gives way to the pool's next;
2. **sentences** — each start flown ``--samples`` times with the prior speaking under the grammar's and the glidepath
   lower edge's masks (`Speaker(finals=…)`, which records what each step's masks allowed), each sentence ending at the
   first flown step below the edge (`glidepath_stops`, outcome ``below_glidepath``);
3. **rewards** — as the first stage (`prior.landing_reward`: 1 for landing in the airport's landing direction — the
   source flight's, its day and time kept), so a stopped sentence earns 0; each sentence's advantage is its reward less
   its start's mean; only starts whose sentences differ are trained on;
4. **one pass** (`train.RewardTuner`, no data term): the advantage × the NLL of each sentence's own words and the pull
   to the BASE model (``--base``: the data-only model every post-training stage pulls back to), both under the masks the
   sentence was said under. The pull measures the distance exactly (`train.flight_exact_kl`) and its weight starts at
   ``--kl-weight`` and follows a KL budget (`train.KlBudget`, design §5): before round 1's pass the start's distance to
   the base on that round's trained sentences D₀ is measured, the target is D₀ + `BUDGET_DELTA`, and every update
   adjusts the weight from the smoothed distance (fast up, slowly down, never below ``--kl-weight``); a smoothed distance
   past the budget's stop ends the pass; every round records its model's distance before its pass;
5. **the select readouts**, both with the edge: the select days' real starts (``--select-per-airport`` ×
   ``--select-samples``, the same flights and seed every round) and the same flights' augmented starts (one fixed
   augmentation each, drawn with ``seed`` + `SELECT_AUGMENT_OFFSET`), and the teacher-forced NLL on the select sentences.

The round kept (``choice.json``): among the rounds within round 0's guards — real-start landed at most
`GUARD_LANDED_DROP` below, landed on the observed runway at most `GUARD_RUNWAY_DROP` below, and in every column of
`GUARD_WORD_COLUMNS` the words a flight says after its first predicted step no farther from the labelled words on the
same flights (the ratio's |ln|) than round 0 was plus ln `GUARD_HEADING_GROWTH` — the highest augmented-start landed
share, the earliest within `TIE_SHARE`. The teacher-forced NLL is recorded, not a guard (the distance to the base is
the budget's). Val is not read here. Writes into ``--out`` (a new directory, from a clean tree unless
``--smoke``): ``config.json``, ``round_00/readout.json``, ``round_<k>/{sentences.npz, sentences.json, checkpoint.pt,
config.json, readout.json}``, ``history.json``, ``choice.json``.

    python run_ts.py prior_augmented_reward --prior <the first stage's kept round> --base <the step-1 run> \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v4_20260924 \\
        --executor 4dTrajectory/outputs/POOLED/executor/<spec> --out 4dTrajectory/outputs/POOLED/prior/<name>
"""

from __future__ import annotations

import argparse
import math
import copy
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.flights import flight_inputs
from ts_transformer.autopilot.params import ExecutorParams
from ts_transformer.experiments.prior_free_generation import (
    AugmentedStarts, _physics, augmented_inputs, augmented_starts, flight_rows, glidepath_stops, grouped, limits_s,
    prior_rows, speak_and_fly, start_altitude_windows,
)
from ts_transformer.experiments.prior_landing_reward import (
    GUARD_HEADING_GROWTH, GUARD_RUNWAY_DROP, SELECT_CHUNK, TIE_SHARE, Sentences, against_the_direction,
    reward_summary, sentence_flights, write_sentences,
)
from ts_transformer.experiments.prior_train import PRIOR_CHECKPOINT_SCHEMA, load_prior, rosters
from ts_transformer.instructions.artefact import load_candidates, load_spec
from ts_transformer.instructions.words import COLUMNS, UNCHANGED, Words
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.augment import LIMITS, Augmentation, augment_signals
from ts_transformer.prior.data import VARIANTS, Split, airport_landings, load_split
from ts_transformer.prior.landing_reward import group_advantages, landing_direction, rewards
from ts_transformer.prior.model import Prior
from ts_transformer.prior.procedure import RunwayProcedure, published_procedures
from ts_transformer.prior.scene import N_LOOK, Landings
from ts_transformer.prior.train import (
    BUDGET_ERROR_CLIP, BUDGET_GAIN_DOWN, BUDGET_GAIN_UP, BUDGET_SMOOTHING, BUDGET_STOP, KlBudget, RewardConfig,
    RewardTuner, TrainConfig, evaluate,
)
from ts_transformer.repo_layout import REPO_ROOT, git_state

AUGMENTED_REWARD_SCHEMA = "ts-prior-augmented-reward-v2"
#: The pool a round draws its starts from, × the starts it needs: a flight with no plausible augmentation gives way to
#: the pool's next.
POOL_FACTOR = 1.25
#: The select days' augmented starts are drawn with the run's seed plus this (fixed across rounds).
SELECT_AUGMENT_OFFSET = 7919
#: The second stage's own guards (design §5), against round 0: the real starts' landed share, and the words said (below).
GUARD_LANDED_DROP = 0.01
#: The columns whose words a flight says after its first step are guarded against the labelled words (design §5): every
#: one but the runway, which the labels never change after the first step.
GUARD_WORD_COLUMNS = ("approach", "heading", "altitude", "angle", "speed")
#: How much farther from the labelled words a column may be than round 0, as a ratio: the first stage's heading-word
#: growth, applied to every guarded column.
GUARD_WORD_GROWTH = GUARD_HEADING_GROWTH
#: The KL budget above the start's own distance to the base (design §5): the first stage's kept round is about this far
#: from the base — the passes after it measured 0.008–0.012 on their own fresh sentences (the sampled-words estimate of
#: the same KL `flight_exact_kl` computes, v3_rl_20260925 history) — so the second stage may go as far again.
BUDGET_DELTA = 0.01


@dataclass(frozen=True)
class MaskedSentences(Sentences):
    """`Sentences` with the masks each was said under: per sentence, the masked columns' bit-packed allowed classes a
    step (`Speaker.allowed`), cut where the sentence was."""

    allowed: list[dict[int, np.ndarray]]


def speak_augmented(model: Prior, batch: replay.Batch, moves: Sequence[Augmentation], samples: int, words: Words,
                    params: ExecutorParams, landings: Mapping[str, Landings] | None,
                    procedures: Mapping[str, tuple[RunwayProcedure, ...]], *,
                    generator: torch.Generator) -> MaskedSentences:
    """Every flight of ``batch`` flown ``samples`` times from its augmented start (``moves``) with the prior speaking
    under the edge's mask, each sentence cut at its end or its stop (the executor on CPU)."""
    cpu = torch.device("cpu")
    count = len(batch.readings)
    index = [j for j in range(count) for _ in range(samples)]
    repeated = replay.subset(batch, index)
    repeated = replace(repeated, signals=[augment_signals(s, moves[j]) for s, j in zip(repeated.signals, index)])
    inputs = augmented_inputs(flight_inputs(repeated.series, device=cpu, anchor=N_LOOK), repeated.geometries,
                              [moves[j] for j in index])
    runways, charts, approach = _physics(repeated, cpu)
    finals = [procedures[g.code] for g in repeated.geometries]
    flown, said, forbidden, speaker = speak_and_fly(model, repeated.signals, repeated.geometries, inputs, runways,
                                                    charts, approach, limits_s(repeated, params, words.spec.step_s),
                                                    words, params, landings, generator=generator, temperature=1.0,
                                                    finals=finals)
    grids = [said[j] for j in range(len(said))]
    stops = glidepath_stops(flown, grids, repeated.geometries, finals, words)
    rows = flight_rows(repeated, flown, grids, words, "prior", [j % samples for j in range(len(grids))], forbidden,
                       stops)
    packed = {column: np.stack(steps, axis=1) for column, steps in speaker.allowed.items()}
    positions, cut, allowed = [], [], []
    for j, row in enumerate(rows):
        steps = row["steps_said"]
        positions.append(np.column_stack((speaker.e[j, : N_LOOK + steps], speaker.n[j, : N_LOOK + steps],
                                          speaker.h[j, : N_LOOK + steps])))
        cut.append(grids[j][:steps])
        allowed.append({column: codes[j, :steps] for column, codes in packed.items()})
    return MaskedSentences(np.repeat(np.arange(count), samples), positions, cut, [r["outcome"] for r in rows],
                           np.array([r["last_runway"] for r in rows]), np.array([r["observed_runway"] for r in rows]),
                           allowed)


def join(parts: Sequence[MaskedSentences], offsets: Sequence[int]) -> MaskedSentences:
    """Sentences of consecutive chunks of one batch (``offsets``: each chunk's first flight)."""
    return MaskedSentences(np.concatenate([p.flight + o for p, o in zip(parts, offsets)]),
                           [x for p in parts for x in p.positions], [x for p in parts for x in p.said],
                           [x for p in parts for x in p.outcomes], np.concatenate([p.runway for p in parts]),
                           np.concatenate([p.observed_runway for p in parts]), [x for p in parts for x in p.allowed])


def round_starts(pool: replay.Batch, moves: Sequence[Augmentation | None], per_airport: int
                 ) -> tuple[list[int], list[Augmentation]]:
    """The pool's first ``per_airport`` flights of each airport with a plausible augmentation, in the pool's order, and
    their augmentations; refused when an airport's pool runs short."""
    taken: Counter = Counter()
    keep: list[int] = []
    for j, (signals, move) in enumerate(zip(pool.signals, moves)):
        if move is not None and taken[signals.airport] < per_airport:
            taken[signals.airport] += 1
            keep.append(j)
    short = {code: per_airport - n for code, n in taken.items() if n < per_airport}
    missing = {s.airport for s in pool.signals} - set(taken)
    if short or missing:
        raise ValueError(f"the pool holds too few plausible starts: {short} short, {sorted(missing)} with none")
    return keep, [moves[j] for j in keep]


def augmentation_summary(batch: replay.Batch, starts: AugmentedStarts, keep: Sequence[int]) -> dict[str, Any]:
    """Per airport (design §4.3): the starts kept, the draws each took (mean, and the share redrawn at least once), and
    the source flights given up before the last one kept — no plausible draw — whose places the pool's next took."""
    out: dict[str, Any] = {}
    kept = set(keep)
    for code in sorted({s.airport for s in batch.signals}):
        members = [j for j, s in enumerate(batch.signals) if s.airport == code]
        taken = [j for j in members if j in kept]
        last = max(taken) if taken else -1
        out[code] = {"starts": len(taken),
                     "draws_per_start": float(np.mean([starts.draws[j] for j in taken])) if taken else None,
                     "redrawn_share": float(np.mean([starts.draws[j] > 1 for j in taken])) if taken else None,
                     "sources_given_up": sum(starts.moves[j] is None for j in members if j <= last)}
    return out


def select_rows(model: Prior, batch: replay.Batch, moves: Sequence[Augmentation] | None, words: Words,
                params: ExecutorParams, landings: Mapping[str, Landings] | None,
                procedures: Mapping[str, tuple[RunwayProcedure, ...]], *, samples: int, seed: int,
                device: torch.device) -> list[dict[str, Any]]:
    """Free generation with the edge on ``batch`` (from its augmented starts when ``moves`` are given), in batches of
    `SELECT_CHUNK` flights by sentence length, one generator seeded with ``seed``."""
    generator = torch.Generator(device=device).manual_seed(seed)
    order = sorted(range(len(batch.readings)), key=lambda j: len(batch.readings[j].words))
    rows: list[dict[str, Any]] = []
    for start in range(0, len(order), SELECT_CHUNK):
        chunk = order[start: start + SELECT_CHUNK]
        rows += prior_rows(model, replay.subset(batch, chunk), words, params, landings, samples, generator=generator,
                           temperature=1.0, procedures=procedures,
                           augmentations=None if moves is None else [moves[j] for j in chunk])[0]
    return rows


def labelled_words(batch: replay.Batch) -> dict[str, float]:
    """Per column, the words a flight's labelled sentence says after its first predicted step (from the second step to
    its end, not flown), over ``batch``'s flights: what the word guard compares a round with; refused when a guarded
    column's labels say nothing (the guard would have no scale)."""
    said = np.stack([(np.asarray(r.words)[N_LOOK + 1:] != UNCHANGED).sum(axis=0) for r in batch.readings])
    out = {name: float(said[:, c].mean()) for c, name in enumerate(COLUMNS)}
    silent = [c for c in GUARD_WORD_COLUMNS if out[c] == 0.0]
    if silent:
        raise ValueError(f"the labelled sentences say no {silent} words after their first step: no scale to guard by")
    return out


def word_distance(words: Mapping[str, float], labelled: Mapping[str, float]) -> dict[str, float]:
    """Per guarded column, how far a readout's words per flight are from the labelled ones: |ln(said / labelled)|."""
    return {c: abs(math.log(words[c] / labelled[c])) if words[c] > 0 else math.inf for c in GUARD_WORD_COLUMNS}


def guarded_choice(history: Sequence[Mapping[str, Any]], labelled: Mapping[str, float]) -> tuple[int, list[int]]:
    """``(the round kept, the rounds excluded)``: among the rounds within round 0's guards (module docstring;
    ``labelled``: `labelled_words` of the select flights), the highest augmented-start landed share, the earliest within
    `TIE_SHARE` of it."""
    start = history[0]
    start_words = word_distance(start["real"]["words_after_first_per_flight"], labelled)
    margin = math.log(GUARD_WORD_GROWTH)

    def words_off(row: Mapping[str, Any]) -> bool:
        words = word_distance(row["real"]["words_after_first_per_flight"], labelled)
        return any(words[c] > start_words[c] + margin for c in GUARD_WORD_COLUMNS)

    excluded = [row["round"] for row in history
                if row["real_landed"] < start["real_landed"] - GUARD_LANDED_DROP
                or row["real"]["landed_on_observed_runway"] is None
                or row["real"]["landed_on_observed_runway"] < start["real"]["landed_on_observed_runway"] - GUARD_RUNWAY_DROP
                or words_off(row)]
    candidates = [row for row in history if row["round"] not in excluded]
    best = max(row["augmented_landed"] for row in candidates)
    return next(row["round"] for row in candidates if row["augmented_landed"] >= best - TIE_SHARE), excluded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the start: the first stage's kept round")
    parser.add_argument("--base", type=Path, required=True, help="the pull's reference: the data-only step-1 run")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--per-airport", type=int, default=400, help="augmented starts an airport a round")
    parser.add_argument("--samples", type=int, default=8, help="sentences a start")
    parser.add_argument("--select-per-airport", type=int, default=200)
    parser.add_argument("--select-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--chunk", type=int, default=32, help="starts a batch (× samples closed loops)")
    parser.add_argument("--device", default="cuda", help="the prior's; the executor flies on CPU")
    parser.add_argument("--smoke", action="store_true", help="a dirty tree allowed; the runs are marked smoke")
    for field, default in asdict(RewardConfig()).items():
        if field not in ("data_weight", "kl_estimate"):   # no data term, the exact distance (design §5)
            parser.add_argument(f"--{field.replace('_', '-')}", type=type(default), default=default)
    args = parser.parse_args(argv)

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    prior_dir, base_dir, instructions, executor_dir, out = map(
        resolved, (args.prior, args.base, args.instructions, args.executor, args.out))
    if out.exists():
        parser.error(f"{out} exists; a fine-tuning run is never overwritten")
    if args.samples < 2:
        parser.error("a start's sentences are compared with each other: --samples ≥ 2")
    if args.kl_weight <= 0.0:
        parser.error("the budget adjusts the pull's weight by factors: --kl-weight must be positive")
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("the tree has uncommitted changes; a fine-tuning run is made at a commit")
    config = RewardConfig(**{f: getattr(args, f) for f in asdict(RewardConfig()) if f not in ("data_weight", "kl_estimate")},
                          data_weight=0.0, kl_estimate="exact")
    started = time.perf_counter()
    device = torch.device(args.device)
    params, record, words = replay.open_executor(executor_dir, instructions)
    spec = load_spec(instructions)
    model, _, start_config = load_prior(prior_dir, instructions)
    base, _, base_config = load_prior(base_dir, instructions)
    for directory, loaded in ((prior_dir, start_config), (base_dir, base_config)):
        if loaded["smoke"]:
            parser.error(f"{directory} is a smoke run")
    if base.config.to_dict() != model.config.to_dict():
        parser.error("the base model and the start are not one architecture")
    model.to(device)
    base.to(device)
    variant = model.config.variant
    every_landing = airport_landings(instructions, rosters(instructions))
    landings = every_landing if VARIANTS[variant].landing_context else None
    procedures = published_procedures(load_candidates(instructions))
    windows = start_altitude_windows(instructions)
    select_batch = replay.draw(instructions, "select", spec, words, per_airport=args.select_per_airport,
                               seed=args.seed)
    # the readout's flights are fixed: a flight with no plausible draw is left out and counted, not replaced
    select_starts = augmented_starts(select_batch.signals,
                                     flight_inputs(select_batch.series, device=torch.device("cpu"), anchor=N_LOOK),
                                     np.random.default_rng(args.seed + SELECT_AUGMENT_OFFSET), windows)
    select_kept = [j for j, move in enumerate(select_starts.moves) if move is not None]
    select_augmented = replay.subset(select_batch, select_kept)
    select_moves = [select_starts.moves[j] for j in select_kept]
    select = load_split(instructions, "select", spec, words, variant, landings=every_landing,
                        airports=model.config.airports)
    select_directions = {signals.dataset_id: landing_direction(signals, geometry, every_landing[signals.airport])
                         for signals, geometry in zip(select_batch.signals, select_batch.geometries)}
    labelled = labelled_words(select_batch)
    out.mkdir(parents=True)
    write_json_atomic(out / "config.json", {
        "schema": AUGMENTED_REWARD_SCHEMA, "written_utc": utc_now(), "git": git, "smoke": args.smoke,
        "prior": {"directory": str(prior_dir), "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt")},
        "base": {"directory": str(base_dir), "checkpoint_sha256": file_sha256(base_dir / "checkpoint.pt")},
        "executor": {"directory": str(executor_dir), "sha256": record["sha256"]}, "instructions": str(instructions),
        "optimiser": asdict(config), "rounds": args.rounds, "per_airport": args.per_airport, "samples": args.samples,
        "augmentation": {"limits": asdict(LIMITS), "pool_factor": POOL_FACTOR, "altitude_windows_m": windows},
        "select": {"per_airport": args.select_per_airport, "samples": args.select_samples, "drawn": select_batch.drawn,
                   "augment_seed": args.seed + SELECT_AUGMENT_OFFSET,
                   "augmented": augmentation_summary(select_batch, select_starts, select_kept),
                   "augmented_left_out": len(select_batch.signals) - len(select_kept),
                   "augmentations": [{"dataset_id": select_batch.signals[j].dataset_id, **asdict(m)}
                                     for j, m in zip(select_kept, select_moves)],
                   "labelled_words_after_first_per_flight": labelled},
        "seed": args.seed, "tie_share": TIE_SHARE,
        "budget": {"delta": BUDGET_DELTA, "gain_up": BUDGET_GAIN_UP, "gain_down": BUDGET_GAIN_DOWN,
                   "error_clip": BUDGET_ERROR_CLIP, "stop": BUDGET_STOP, "smoothing": BUDGET_SMOOTHING,
                   "start_weight_and_floor": config.kl_weight},
        "guards": {"landed_drop": GUARD_LANDED_DROP, "runway_drop": GUARD_RUNWAY_DROP,
                   "word_columns": list(GUARD_WORD_COLUMNS), "word_margin_ln": math.log(GUARD_WORD_GROWTH)},
        "n_look": N_LOOK})

    def log(line: str) -> None:
        print(f"{line}  [{time.perf_counter() - started:.0f}s]", flush=True)

    def read_select(round_number: int) -> dict[str, Any]:
        model.eval()
        real = select_rows(model, select_batch, None, words, params, landings, procedures,
                           samples=args.select_samples, seed=args.seed, device=device)
        augmented = select_rows(model, select_augmented, select_moves, words, params, landings, procedures,
                                samples=args.select_samples, seed=args.seed, device=device)
        readout = {"real": grouped(real), "augmented": grouped(augmented),
                   "real_landed_against_the_direction": against_the_direction(real, select_directions),
                   "teacher_forced": evaluate(model, select, TrainConfig(), device),
                   "flights": {"real": real, "augmented": augmented}}
        log(f"round {round_number}: select landed real {readout['real']['all']['outcomes']['landed']:.3f} "
            f"augmented {readout['augmented']['all']['outcomes']['landed']:.3f}  below the edge real "
            f"{readout['real']['all']['outcomes']['below_glidepath']:.3f} augmented "
            f"{readout['augmented']['all']['outcomes']['below_glidepath']:.3f}  TF NLL "
            f"{readout['teacher_forced']['nll_per_step']:.4f}  words off the labels "
            + " ".join(f"{c} {d:.2f}" for c, d in word_distance(readout['real']['all']['words_after_first_per_flight'],
                                                                  labelled).items()))
        return readout

    def history_row(round_number: int, readout: dict[str, Any], **more: Any) -> dict[str, Any]:
        real, augmented = readout["real"]["all"], readout["augmented"]["all"]
        return {"round": round_number, "real_landed": real["outcomes"]["landed"],
                "augmented_landed": augmented["outcomes"]["landed"],
                "select_tf_nll": readout["teacher_forced"]["nll_per_step"],
                # None: the column said no word at all (off the labels without bound)
                "words_off_the_labels": {c: (d if math.isfinite(d) else None) for c, d in
                                         word_distance(real["words_after_first_per_flight"], labelled).items()},
                "real_landed_against_the_direction": readout["real_landed_against_the_direction"],
                "real": {k: v for k, v in real.items() if k != "outcomes"}, "real_outcomes": real["outcomes"],
                "augmented": {k: v for k, v in augmented.items() if k != "outcomes"},
                "augmented_outcomes": augmented["outcomes"], **more}

    directory = out / "round_00"
    directory.mkdir()
    readout = read_select(0)
    write_json_atomic(directory / "readout.json", readout)
    history = [history_row(0, readout)]
    tuner = RewardTuner(model, base, config, device, seed=args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for round_number in range(1, args.rounds + 1):
        directory = out / f"round_{round_number:02d}"
        directory.mkdir()
        pool = replay.draw(instructions, "train", spec, words, per_airport=round(args.per_airport * POOL_FACTOR),
                           seed=args.seed + round_number)
        # the augmentations' own stream, apart from the pool's permutation (which the replay draws with the round's seed)
        pool_starts = augmented_starts(pool.signals, flight_inputs(pool.series, device=torch.device("cpu"),
                                                                   anchor=N_LOOK),
                                       np.random.default_rng([args.seed, round_number]), windows)
        keep_pool, moves = round_starts(pool, pool_starts.moves, args.per_airport)
        batch = replay.subset(pool, keep_pool)
        order = sorted(range(len(batch.readings)), key=lambda j: len(batch.readings[j].words))
        batch, moves = replay.subset(batch, order), [moves[j] for j in order]
        model.eval()
        starts = list(range(0, len(order), args.chunk))
        sentences = join([speak_augmented(model, replay.subset(batch, list(range(s, min(s + args.chunk, len(order))))),
                                          moves[s: s + args.chunk], args.samples, words, params, landings, procedures,
                                          generator=generator)
                          for s in starts], starts)
        # the augmented start keeps its source's day and time: the landing direction is the source flight's
        directions = [landing_direction(signals, geometry, every_landing[signals.airport])
                      for signals, geometry in zip(batch.signals, batch.geometries)]
        earned = rewards(sentences.outcomes, sentences.runway, [directions[f] for f in sentences.flight])
        advantages = group_advantages(earned.reshape(-1, args.samples)).reshape(-1)
        keep = np.flatnonzero(np.repeat(earned.reshape(-1, args.samples).std(axis=1) > 0, args.samples))
        write_sentences(directory / "sentences.npz", batch, sentences, earned, advantages)
        augmented = replace(batch, signals=[augment_signals(s, m) for s, m in zip(batch.signals, moves)])
        summary = {**reward_summary(augmented, sentences, earned, args.samples), "trained_on": len(keep),
                   "augmentation": augmentation_summary(pool, pool_starts, keep_pool), "drawn": pool.drawn,
                   "augmentations": [asdict(m) for m in moves]}
        write_json_atomic(directory / "sentences.json", summary)
        log(f"round {round_number}: {len(earned)} sentences, reward {summary['reward_mean']:.3f}, below the edge "
            f"{Counter(sentences.outcomes)['below_glidepath']}, {summary['flights_with_contrast']} starts with a contrast")
        flights = sentence_flights(augmented, sentences, keep, model.config.airports, spec.step_s, landings)
        split = Split(flights, select.airports, select.candidates, select.runways, select.courses, select.classes,
                      variant)
        allowed = [sentences.allowed[s] for s in keep]
        # the model's distance to the base on its own fresh sentences, before the pass
        start_distance = tuner.distance(split, allowed)
        if tuner.budget is None:
            # the budget (design §5): the start's own distance to the base on its first round's sentences, plus the delta
            tuner.budget = KlBudget(start_distance + BUDGET_DELTA, floor=config.kl_weight)
            write_json_atomic(out / "budget.json", {"start_distance": start_distance, "target": tuner.budget.target,
                                                   "measured_on": f"round {round_number}'s {len(flights)} trained sentences"})
            log(f"KL budget: the start is {start_distance:.4f} from the base, target {tuner.budget.target:.4f}")
        passed = {**tuner.one_pass(split, advantages[keep], None, allowed, start_distance),
                  "distance_at_start": start_distance}
        log(f"round {round_number}: one pass over {len(flights)} sentences, reward term {passed['reward_mean']:.4f}, "
            f"KL to the base {start_distance:.4f} at the start, {passed['kl_mean']:.4f} in the pass (max "
            f"{passed['kl_max']:.4f}, target {tuner.budget.target:.4f}), "
            f"pull weight {passed['kl_weight_start']:.4g} → {passed['kl_weight_end']:.4g}"
            + (f"; stopped at batch {passed['stopped']['batch']} (smoothed {passed['stopped']['smoothed']:.4f})"
               if passed["stopped"] else ""))
        torch.save({"schema": PRIOR_CHECKPOINT_SCHEMA, "model_config": model.config.to_dict(),
                    "train_config": start_config["train"], "state": copy.deepcopy(model.state_dict()),
                    "spec_sha256": spec.sha256}, directory / "checkpoint.pt")
        write_json_atomic(directory / "config.json", {
            **start_config, "written_utc": utc_now(), "git": git, "smoke": args.smoke,
            "fine_tuning": {"schema": AUGMENTED_REWARD_SCHEMA, "from": str(prior_dir), "base": str(base_dir),
                            "round": round_number, "optimiser": asdict(config), "samples": args.samples,
                            "kl_budget_target": tuner.budget.target, "kl_weight_end": tuner.kl_weight}})
        readout = read_select(round_number)
        write_json_atomic(directory / "readout.json", readout)
        history.append(history_row(round_number, readout, train_pass=passed,
                                   sentences={k: v for k, v in summary.items()
                                              if k not in ("drawn", "augmentations")}))
        write_json_atomic(out / "history.json", {"rounds": history})
    kept, excluded = guarded_choice(history, labelled)
    write_json_atomic(out / "choice.json", {
        "rule": f"within round 0's guards (real-start landed ≥ − {GUARD_LANDED_DROP}, landed on the observed runway ≥ − "
                f"{GUARD_RUNWAY_DROP}, each of {', '.join(GUARD_WORD_COLUMNS)}: words per flight off the labelled "
                f"≤ round 0's + ln {GUARD_WORD_GROWTH}), the highest augmented-start landed share; within "
                f"{TIE_SHARE} the earliest",
        "augmented_landed": [row["augmented_landed"] for row in history],
        "real_landed": [row["real_landed"] for row in history], "excluded_by_the_guards": excluded, "round": kept,
        "directory": str(out / f"round_{kept:02d}") if kept else str(prior_dir)})
    log(f"kept round {kept} (the guards excluded {excluded}) → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
