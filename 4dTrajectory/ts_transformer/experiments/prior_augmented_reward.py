"""Post-training stage 2 (post-training design §3–§5): real and augmented starts under the procedure's masks, the first
stage's loss.

Each round:

1. **starts** — a pool of `POOL_FACTOR` × (``--real-per-airport`` + ``--augmented-per-airport``) train-day flights per
   airport (their own dynamics, the replay gate's group) drawn with the round's seed; the pool's first
   ``--real-per-airport`` of each airport fly from their own start, and of the rest the first ``--augmented-per-airport``
   with a plausible augmentation (`prior.augment`: rotated about the airport, raised, sped up;
   `prior_free_generation.augmented_starts`) fly from it — a flight none of `AUGMENT_TRIES` draws fits gives way to the
   pool's next (`round_starts`);
2. **sentences** — each start flown ``--samples`` times with the prior speaking under the grammar's and the procedure's
   masks (`Speaker(finals=…)`, which records what each step's masks allowed), each sentence ending at the first flown
   step below the glidepath lower edge (`glidepath_stops`, outcome ``below_glidepath``); a real start's time limit is
   the executor spec's, an augmented one's `augment.TIMEOUT_FACTOR` × its source's observed remaining time;
3. **rewards** — as the first stage (`prior.landing_reward`: 1 for landing in the airport's landing direction — the
   source flight's, its day and time kept), so a stopped sentence earns 0; each sentence's advantage is its reward less
   its start's mean; only starts whose sentences differ are trained on;
4. **one pass** (`train.RewardTuner`, the first stage's recipe, design §5): the advantage × the NLL of each sentence's own
   words and ``--kl-weight`` × the pull to the BASE model (``--base``: the data-only model every post-training stage pulls
   back to), both under the masks the sentence was said under, and ``--data-weight`` × the teacher-forced NLL of a batch
   of train-day flights (their ADS-B rows and labelled words) with every update; every round records its model's
   distance to the base on its fresh sentences before the pass, the real starts' and the augmented ones' apart;
5. **the select readouts**, both under the masks: the select days' real starts (``--select-per-airport`` ×
   ``--select-samples``, the same flights and seed every round) and the same flights' augmented starts (one fixed
   augmentation each, drawn with ``seed`` + `SELECT_AUGMENT_OFFSET`) — outcomes, words said, the readouts before the
   join beside the observed tracks' — and the teacher-forced NLL on the select sentences.

The round kept (``choice.json``): among the rounds within round 0's guards — real-start landed at most
`GUARD_LANDED_DROP` below, landed on the observed runway at most `GUARD_RUNWAY_DROP` below, and, on the real starts and on
the augmented ones alike, in every column of `GUARD_WORD_COLUMNS` the words a flight says after its first predicted step
no farther from the labelled words (the ratio's |ln|; an augmented start's are its source flight's) than round 0 was plus
ln `GUARD_WORD_GROWTH` — the highest augmented-start landed share, the earliest within `TIE_SHARE`. The teacher-forced
NLL is recorded, not a guard (the data term holds it). Val is not read here. Writes into ``--out`` (a new directory, from a clean tree unless
``--smoke``): ``config.json``, ``round_00/readout.json``, ``round_<k>/{sentences.npz, sentences.json, checkpoint.pt,
config.json, readout.json}``, ``history.json``, ``choice.json``.

    python run_ts.py prior_augmented_reward --prior <the first stage's kept round> --base <the step-1 run> \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
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
    PRE_JOIN_LINES, AugmentedStarts, ProcedureMasks, _physics, augmented_inputs, augmented_starts, flight_rows,
    glidepath_stops, grouped, limits_s, prior_rows, procedure_masks, speak_and_fly, start_altitude_windows,
)
from ts_transformer.experiments.prior_landing_reward import (
    GUARD_HEADING_GROWTH, GUARD_RUNWAY_DROP, SELECT_CHUNK, TIE_SHARE, Sentences, against_the_direction,
    reward_summary, sentence_flights, write_sentences,
)
from ts_transformer.experiments.prior_train import PRIOR_CHECKPOINT_SCHEMA, load_prior, rosters
from ts_transformer.instructions.artefact import load_candidates, load_spec
from ts_transformer.instructions.words import COLUMNS, UNCHANGED, Words
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior import mva
from ts_transformer.prior.augment import LIMITS, TIMEOUT_FACTOR, Augmentation, augment_signals
from ts_transformer.prior.data import VARIANTS, Split, airport_landings, load_split
from ts_transformer.prior.landing_reward import group_advantages, landing_direction, rewards
from ts_transformer.prior.model import Prior
from ts_transformer.prior.scene import N_LOOK, Landings
from ts_transformer.prior.train import RewardConfig, RewardTuner, TrainConfig, evaluate
from ts_transformer.repo_layout import REPO_ROOT, git_state

AUGMENTED_REWARD_SCHEMA = "ts-prior-augmented-reward-v4"
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


@dataclass(frozen=True)
class MaskedSentences(Sentences):
    """`Sentences` with the masks each was said under: per sentence, the masked columns' bit-packed allowed classes a
    step (`Speaker.allowed`), cut where the sentence was."""

    allowed: list[dict[int, np.ndarray]]


def speak_starts(model: Prior, batch: replay.Batch, moves: Sequence[Augmentation | None], samples: int, words: Words,
                 params: ExecutorParams, landings: Mapping[str, Landings] | None, procedures: ProcedureMasks, *,
                 generator: torch.Generator) -> MaskedSentences:
    """Every flight of ``batch`` flown ``samples`` times from its start — its augmented one where ``moves`` gives one,
    its own where None — with the prior speaking under the procedure's masks, each sentence cut at its end or its stop
    (the executor on CPU)."""
    cpu = torch.device("cpu")
    count = len(batch.readings)
    index = [j for j in range(count) for _ in range(samples)]
    repeated = replay.subset(batch, index)
    each = [moves[flight] for flight in index]
    repeated = replace(repeated, signals=[s if m is None else augment_signals(s, m)
                                          for s, m in zip(repeated.signals, each)])
    inputs = augmented_inputs(flight_inputs(repeated.series, device=cpu, anchor=N_LOOK), repeated.geometries, each)
    runways, charts, approach = _physics(repeated, cpu)
    finals = [procedures.finals[g.code] for g in repeated.geometries]
    own = limits_s(repeated, params, words.spec.step_s, augmented=False)
    augmented = limits_s(repeated, params, words.spec.step_s, augmented=True)
    limits = [own[j] if m is None else augmented[j] for j, m in enumerate(each)]
    flown, said, forbidden, speaker = speak_and_fly(model, repeated.signals, repeated.geometries, inputs, runways,
                                                    charts, approach, limits, words, params, landings,
                                                    generator=generator, temperature=1.0, finals=finals)
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


def real_starts(pool: replay.Batch, per_airport: int) -> list[int]:
    """The pool's first ``per_airport`` flights of each airport, in the pool's order: the real starts (the pool holds
    more than that of every airport — `replay.draw` refuses one that is short)."""
    taken: Counter = Counter()
    keep: list[int] = []
    for j, signals in enumerate(pool.signals):
        if taken[signals.airport] < per_airport:
            taken[signals.airport] += 1
            keep.append(j)
    return keep


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
                params: ExecutorParams, landings: Mapping[str, Landings] | None, procedures: ProcedureMasks, *,
                samples: int, seed: int, device: torch.device) -> list[dict[str, Any]]:
    """Free generation under the procedure's masks on ``batch`` (from its augmented starts when ``moves`` are given), in
    batches of `SELECT_CHUNK` flights by sentence length, one generator seeded with ``seed``."""
    generator = torch.Generator(device=device).manual_seed(seed)
    order = sorted(range(len(batch.readings)), key=lambda j: len(batch.readings[j].words))
    rows: list[dict[str, Any]] = []
    for start in range(0, len(order), SELECT_CHUNK):
        chunk = order[start: start + SELECT_CHUNK]
        rows += prior_rows(model, replay.subset(batch, chunk), words, params, landings, samples, generator=generator,
                           temperature=1.0, procedures=procedures,
                           augmentations=None if moves is None else [moves[j] for j in chunk])[0]
    return rows


def pre_join_line(shares: Mapping[str, float]) -> str:
    """A readout's shares before the join, said / observed, for the log."""
    return " ".join(f"{key} {shares[f'said_{key}']:.3f}/{shares[f'observed_{key}']:.3f}" for key in PRE_JOIN_LINES)


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


def guarded_choice(history: Sequence[Mapping[str, Any]], labelled: Mapping[str, Mapping[str, float]]
                   ) -> tuple[int, list[int]]:
    """``(the round kept, the rounds excluded)``: among the rounds within round 0's guards (module docstring;
    ``labelled``: `labelled_words` of the select flights, per side — ``real`` and ``augmented``, an augmented start's
    being its source flight's), the highest augmented-start landed share, the earliest within `TIE_SHARE` of it."""
    start = history[0]
    margin = math.log(GUARD_WORD_GROWTH)

    def words_off(row: Mapping[str, Any]) -> bool:
        for side in ("real", "augmented"):
            start_words = word_distance(start[side]["words_after_first_per_flight"], labelled[side])
            words = word_distance(row[side]["words_after_first_per_flight"], labelled[side])
            if any(words[c] > start_words[c] + margin for c in GUARD_WORD_COLUMNS):
                return True
        return False

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
    parser.add_argument("--real-per-airport", type=int, default=200, help="real starts an airport a round")
    parser.add_argument("--augmented-per-airport", type=int, default=200, help="augmented starts an airport a round")
    parser.add_argument("--samples", type=int, default=8, help="sentences a start")
    parser.add_argument("--select-per-airport", type=int, default=200)
    parser.add_argument("--select-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--chunk", type=int, default=32, help="starts a batch (× samples closed loops)")
    parser.add_argument("--device", default="cuda", help="the prior's; the executor flies on CPU")
    parser.add_argument("--smoke", action="store_true", help="a dirty tree allowed; the runs are marked smoke")
    for field, default in asdict(RewardConfig()).items():
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
    if args.data_weight <= 0.0:
        parser.error("the second stage trains beside the data (design §5): --data-weight > 0")
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("the tree has uncommitted changes; a fine-tuning run is made at a commit")
    config = RewardConfig(**{f: getattr(args, f) for f in asdict(RewardConfig())})
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
    procedures = procedure_masks(load_candidates(instructions))
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
    data = load_split(instructions, "train", spec, words, variant, landings=every_landing,
                      airports=model.config.airports)
    select_directions = {signals.dataset_id: landing_direction(signals, geometry, every_landing[signals.airport])
                         for signals, geometry in zip(select_batch.signals, select_batch.geometries)}
    labelled = {"real": labelled_words(select_batch), "augmented": labelled_words(select_augmented)}
    out.mkdir(parents=True)
    write_json_atomic(out / "config.json", {
        "schema": AUGMENTED_REWARD_SCHEMA, "written_utc": utc_now(), "git": git, "smoke": args.smoke,
        "prior": {"directory": str(prior_dir), "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt")},
        "base": {"directory": str(base_dir), "checkpoint_sha256": file_sha256(base_dir / "checkpoint.pt")},
        "executor": {"directory": str(executor_dir), "sha256": record["sha256"]}, "instructions": str(instructions),
        "optimiser": asdict(config), "rounds": args.rounds, "real_per_airport": args.real_per_airport,
        "augmented_per_airport": args.augmented_per_airport, "samples": args.samples,
        "augmentation": {"limits": asdict(LIMITS), "pool_factor": POOL_FACTOR, "altitude_windows_m": windows,
                         "timeout_factor": TIMEOUT_FACTOR, "real_timeout_factor": params.timeout_factor},
        "mva": {"charts_date": mva.CHARTS_DATE, "chart": mva.CHART, "facility": mva.FACILITY},
        "select": {"per_airport": args.select_per_airport, "samples": args.select_samples, "drawn": select_batch.drawn,
                   "augment_seed": args.seed + SELECT_AUGMENT_OFFSET,
                   "augmented": augmentation_summary(select_batch, select_starts, select_kept),
                   "augmented_left_out": len(select_batch.signals) - len(select_kept),
                   "augmentations": [{"dataset_id": select_batch.signals[j].dataset_id, **asdict(m)}
                                     for j, m in zip(select_kept, select_moves)],
                   "labelled_words_after_first_per_flight": labelled},
        "seed": args.seed, "tie_share": TIE_SHARE,
        "data": {"split": "train", "flights": len(data.flights)},
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
            f"{readout['teacher_forced']['nll_per_step']:.4f}  words off the labels (real / augmented) "
            + " ".join(f"{c} {d:.2f}/{e:.2f}" for (c, d), e in zip(
                word_distance(readout['real']['all']['words_after_first_per_flight'], labelled['real']).items(),
                word_distance(readout['augmented']['all']['words_after_first_per_flight'],
                              labelled['augmented']).values()))
            + f"  before the join (said / observed) real {pre_join_line(readout['real']['all']['pre_join'])} "
              f"augmented {pre_join_line(readout['augmented']['all']['pre_join'])}")
        return readout

    def history_row(round_number: int, readout: dict[str, Any], **more: Any) -> dict[str, Any]:
        real, augmented = readout["real"]["all"], readout["augmented"]["all"]
        return {"round": round_number, "real_landed": real["outcomes"]["landed"],
                "augmented_landed": augmented["outcomes"]["landed"],
                "select_tf_nll": readout["teacher_forced"]["nll_per_step"],
                # None: the column said no word at all (off the labels without bound)
                "words_off_the_labels": {side: {c: (d if math.isfinite(d) else None) for c, d in word_distance(
                    readout[side]["all"]["words_after_first_per_flight"], labelled[side]).items()}
                    for side in ("real", "augmented")},
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
        per_airport = args.real_per_airport + args.augmented_per_airport
        pool = replay.draw(instructions, "train", spec, words, per_airport=round(per_airport * POOL_FACTOR),
                           seed=args.seed + round_number)
        real = real_starts(pool, args.real_per_airport)
        taken = set(real)
        rest = [j for j in range(len(pool.signals)) if j not in taken]
        candidates = replay.subset(pool, rest)
        # the augmentations' own stream, apart from the pool's permutation (which the replay draws with the round's seed)
        pool_starts = augmented_starts(candidates.signals, flight_inputs(candidates.series, device=torch.device("cpu"),
                                                                         anchor=N_LOOK),
                                       np.random.default_rng([args.seed, round_number]), windows)
        keep_candidates, augmented_moves = round_starts(candidates, pool_starts.moves, args.augmented_per_airport)
        batch = replay.subset(pool, real + [rest[j] for j in keep_candidates])
        moves: list[Augmentation | None] = [None] * len(real) + list(augmented_moves)
        order = sorted(range(len(batch.readings)), key=lambda j: len(batch.readings[j].words))
        batch, moves = replay.subset(batch, order), [moves[j] for j in order]
        model.eval()
        starts = list(range(0, len(order), args.chunk))
        sentences = join([speak_starts(model, replay.subset(batch, list(range(s, min(s + args.chunk, len(order))))),
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
        augmented = replace(batch, signals=[s if m is None else augment_signals(s, m)
                                            for s, m in zip(batch.signals, moves)])
        is_augmented = np.array([moves[f] is not None for f in sentences.flight])
        summary = {**reward_summary(augmented, sentences, earned, args.samples), "trained_on": len(keep),
                   "reward_mean_by_start": {"real": float(earned[~is_augmented].mean()),
                                            "augmented": float(earned[is_augmented].mean())},
                   "augmentation": augmentation_summary(candidates, pool_starts, keep_candidates), "drawn": pool.drawn,
                   "augmentations": [None if m is None else asdict(m) for m in moves]}
        write_json_atomic(directory / "sentences.json", summary)
        log(f"round {round_number}: {len(earned)} sentences, reward {summary['reward_mean']:.3f} (real "
            f"{summary['reward_mean_by_start']['real']:.3f} / augmented {summary['reward_mean_by_start']['augmented']:.3f}"
            f"), below the edge {Counter(sentences.outcomes)['below_glidepath']}, {summary['flights_with_contrast']} "
            f"starts with a contrast")
        flights = sentence_flights(augmented, sentences, keep, model.config.airports, spec.step_s, landings)
        split = Split(flights, data.airports, data.candidates, data.runways, data.courses, data.classes, variant)
        allowed = [sentences.allowed[s] for s in keep]
        # the model's distance to the base on the fresh sentences it is about to train on (the starts with a contrast),
        # before the pass, the real starts' and the augmented ones' apart (design §5)
        start_distance, measured_on = {}, {}
        for side, mask in (("real", ~is_augmented[keep]), ("augmented", is_augmented[keep])):
            members = np.flatnonzero(mask)
            start_distance[side] = tuner.distance(replace(split, flights=[flights[i] for i in members]),
                                                  [allowed[i] for i in members])
            measured_on[side] = len(members)
        passed = {**tuner.one_pass(split, advantages[keep], data, allowed), "distance_at_start": start_distance,
                  "distance_sentences": measured_on}
        log(f"round {round_number}: one pass over {len(flights)} sentences, reward term {passed['reward_mean']:.4f}, "
            f"KL to the base at the start real {start_distance['real']:.4f} augmented "
            f"{start_distance['augmented']:.4f}, {passed['kl_mean']:.4f} in the pass (max {passed['kl_max']:.4f}), "
            f"data NLL {passed['data_mean']:.4f}")
        torch.save({"schema": PRIOR_CHECKPOINT_SCHEMA, "model_config": model.config.to_dict(),
                    "train_config": start_config["train"], "state": copy.deepcopy(model.state_dict()),
                    "spec_sha256": spec.sha256}, directory / "checkpoint.pt")
        write_json_atomic(directory / "config.json", {
            **start_config, "written_utc": utc_now(), "git": git, "smoke": args.smoke,
            "fine_tuning": {"schema": AUGMENTED_REWARD_SCHEMA, "from": str(prior_dir), "base": str(base_dir),
                            "round": round_number, "optimiser": asdict(config), "samples": args.samples}})
        readout = read_select(round_number)
        write_json_atomic(directory / "readout.json", readout)
        history.append(history_row(round_number, readout, train_pass=passed,
                                   sentences={k: v for k, v in summary.items()
                                              if k not in ("drawn", "augmentations")}))
        write_json_atomic(out / "history.json", {"rounds": history})
    kept, excluded = guarded_choice(history, labelled)
    write_json_atomic(out / "choice.json", {
        "rule": f"within round 0's guards (real-start landed ≥ − {GUARD_LANDED_DROP}, landed on the observed runway ≥ − "
                f"{GUARD_RUNWAY_DROP}, each of {', '.join(GUARD_WORD_COLUMNS)} on the real and on the augmented "
                f"starts: words per flight off the labelled ≤ round 0's + ln {GUARD_WORD_GROWTH}), the highest "
                f"augmented-start landed share; within {TIE_SHARE} the earliest",
        "augmented_landed": [row["augmented_landed"] for row in history],
        "real_landed": [row["real_landed"] for row in history], "excluded_by_the_guards": excluded, "round": kept,
        "directory": str(out / f"round_{kept:02d}") if kept else str(prior_dir)})
    log(f"kept round {kept} (the guards excluded {excluded}) → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
