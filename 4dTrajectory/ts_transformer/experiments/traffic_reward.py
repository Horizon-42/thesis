"""Multi-aircraft M4 (design §6.2, §6.6 step 6): the post-training of "traffic" — augmented with a traffic attention
(`prior.model.with_traffic`: at zero, augmented's answers to rounding) speaking to one aircraft of a scene at a time, the
others replayed, rewarded for landing without losing separation and pulled back to base.

Each round:

1. **scenes** — a pool of `POOL_FACTOR` × (``--real-per-airport`` + ``--augmented-per-airport``) training-day flights per
   airport (their own dynamics, the replay gate's group) drawn with the round's seed; the pool's first
   ``--real-per-airport`` of each airport speak in their real scenes (`traffic_speaking.scene_of`), and of the rest, in
   the pool's order, the first ``--augmented-per-airport`` whose scene augments (`traffic_augment`: the leader moved, the
   start moved, a flight inserted, a third each, drawn until it qualifies; `traffic_free_generation.augmented_scenes`)
   speak in their augmented ones — refused when an airport runs short;
2. **sentences** — each scene spoken to ``--samples`` times in the scene loop (`traffic_free_generation.scene_sentences`:
   the vocabulary's rules, the start's procedure's masks, the two separation masks; a moved start's time limit is stage
   2's), judged in the scene under VISUAL, each sentence kept up to its judged end;
3. **rewards** — 1 for landing on a runway in the airport's landing direction (`prior.landing_reward`, against the
   scene's landings) with no loss of separation ending it first, else 0; each sentence's advantage is its reward less its
   scene's mean. A scene is trained on only when its sentences differ and it does not start in a loss the speaking
   aircraft answers for (no word of it made that one — counted);
4. **one pass** (`traffic_tuner.SceneRewardTuner`): the second stage's loss, each sentence scored in its scene, the pull
   to base reading the aircraft alone, the data term on the training days' scene samples (M2's: each flight's rows the
   ones the loop's scenes read), the traffic attention at ``--traffic-learning-rate``; the distance to base on the fresh
   sentences before the pass, the real scenes' and the augmented ones' apart;
5. **the select readouts** (every round, round 0 the start): the select days' ``--select-per-airport`` flights ×
   ``--select-samples`` sentences in their real scenes and in one fixed augmentation each (``seed`` +
   `SELECT_AUGMENT_OFFSET`), the same flights, seed and batches every round — the executor's outcomes and the words
   (`prior_free_generation.grouped`), the separation (`traffic_free_generation.summary`), the reward, and on the real
   scenes the ordering (`ordering`); the teacher-forced NLL on the select days' scene samples and the traffic attention's
   output against the residual stream it adds to (`traffic_readout`). The select flights along their records are read
   once (the separation's target line).

The round kept (``choice.json``, design §6.6 step 6 item 8): among the rounds within round 0's guards — on the real
scenes, landed (the executor's: separation aside) at most `GUARD_LANDED_DROP` below, landed on the observed runway at most
`GUARD_RUNWAY_DROP` below, lost separation at most `GUARD_LOSS_RISE` above, the median time to land and the median gap to
the landing before on the runway both at most `GUARD_ORDER_GROWTH` × the same flights' recorded ones; on the real and the
augmented scenes, the words a flight says after its first step no farther from the labelled ones than round 0 plus ln
`GUARD_WORD_GROWTH` (stage 2's) — the highest augmented-scene reward, the earliest within `TIE_SHARE`. Round 0 kept means
the start (augmented) is. Val is not read here.

Writes into ``--out`` (a new directory, from a clean tree unless ``--smoke``): ``config.json``, ``round_00/readout.json``,
``round_<k>/{sentences.npz, sentences.json, checkpoint.pt (`prior_train.TRAFFIC_CHECKPOINT_SCHEMA`), config.json,
procedure_masks.json, readout.json}``, ``history.json``, ``choice.json``.

    python run_ts.py traffic_reward \\
        --prior 4dTrajectory/outputs/POOLED/prior/v3_stage2_clip_20260926/aug_s1337/round_07 \\
        --base 4dTrajectory/outputs/POOLED/prior/v3_step1_20260924/full_s1337 \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --executor 4dTrajectory/outputs/POOLED/executor/v11_20260927 \\
        --out 4dTrajectory/outputs/POOLED/prior/<campaign>/traffic_s1337
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.experiments.prior_augmented_reward import (
    GUARD_LANDED_DROP, GUARD_WORD_COLUMNS, GUARD_WORD_GROWTH, POOL_FACTOR, SELECT_AUGMENT_OFFSET, labelled_words,
    real_starts, word_distance,
)
from ts_transformer.experiments.prior_free_generation import grouped, limits_s, start_altitude_windows
from ts_transformer.experiments.prior_landing_reward import GUARD_RUNWAY_DROP, TIE_SHARE
from ts_transformer.experiments.prior_train import (
    PRIOR_CHECKPOINT_SCHEMA, TRAFFIC_CHECKPOINT_SCHEMA, LoadedPrior, load_prior, rosters,
)
from ts_transformer.experiments.traffic_loop import LOST_SEPARATION
from ts_transformer.experiments.traffic_free_generation import (
    AIRCRAFT_STEPS, SceneSentences, augmented_scenes, recorded_rows, scene_sentences, speaking_batches, summary,
)
from ts_transformer.experiments.traffic_prior_train import scene_evaluate
from ts_transformer.experiments.traffic_scene_data import Built, airport_flights, edge_source_sha256, split_samples
from ts_transformer.experiments.traffic_speaking import Scene, scene_landings, scene_of, with_tracks
from ts_transformer.experiments.traffic_tuner import SceneRewardTuner, SceneSplit
from ts_transformer.inference.scene_edges import EDGE_FEATURES
from ts_transformer.inference.separation import VISUAL
from ts_transformer.instructions.artefact import load_candidates, load_spec
from ts_transformer.instructions.words import UNCHANGED, Words
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.augment import Augmentation
from ts_transformer.prior.data import (
    VARIANTS, Split, airport_landings, candidate_table, chain_record, column_classes, runway_names,
)
from ts_transformer.prior.landing_reward import LANDED, group_advantages, landing_direction
from ts_transformer.prior.masks import ProcedureMasks, write_masks
from ts_transformer.prior.model import Prior, with_traffic
from ts_transformer.prior.scene import N_LOOK, Landings
from ts_transformer.prior.train import RewardConfig, TrainConfig
from ts_transformer.repo_layout import REPO_ROOT, git_state

TRAFFIC_REWARD_SCHEMA = "ts-traffic-reward-v1"
RUNNER = "ts_transformer.experiments.traffic_reward"
#: The traffic attention's learning rate (design §8): pretraining's; the rest of the model trains at the second stage's.
TRAFFIC_LEARNING_RATE = 3e-4
#: The ordering guard (design §4.3): the median time to land and the median gap to the landing before on the runway, at
#: most this × the same flights' recorded ones (the words guard's margin).
GUARD_ORDER_GROWTH = 1.2
#: The separation guard (design §6.6 step 6 item 8): the real scenes' lost-separation share at most this above round 0's
#: (the landed guard's margin; about 1.4 binomial standard deviations over 2,000 select sentences at 11.5 %). The
#: recorded share × `GUARD_ORDER_GROWTH` is the target line, reported (round 0 is four times the record).
GUARD_LOSS_RISE = 0.01


@dataclasses.dataclass(frozen=True)
class Round:
    """A round's scenes: the flights (their rows moved where a start moved), their scenes, each one's moved start (None:
    its own) and kind (``real``, or the augmentation's: D, B, A), and — each flight's — the runways in the airport's
    landing direction."""

    batch: replay.Batch
    scenes: list[Scene]
    moves: list[Augmentation | None]
    kinds: list[str]
    directions: list[np.ndarray]


def directions_of(batch: replay.Batch, scenes: Sequence[Scene], every_landing: Mapping[str, Landings]
                  ) -> list[np.ndarray]:
    """Each flight's runways in the airport's landing direction at its first predicted step, against its scene's
    landings (an augmented scene moves or adds one; `traffic_speaking.scene_landings`)."""
    return [landing_direction(signals, geometry, scene_landings(every_landing[signals.airport], scene))
            for signals, geometry, scene in zip(batch.signals, batch.geometries, scenes)]


def first_per_airport(batch: replay.Batch, per_airport: int, airports: Sequence[str]) -> list[int]:
    """The batch's first ``per_airport`` flights of each of ``airports``, in its order; refused when one has fewer."""
    taken: Counter = Counter()
    keep = []
    for j, signals in enumerate(batch.signals):
        if taken[signals.airport] < per_airport:
            taken[signals.airport] += 1
            keep.append(j)
    short = {code: per_airport - taken[code] for code in airports if taken[code] < per_airport}
    if short:
        raise ValueError(f"too few augmented scenes qualify: {short} short")
    return keep


def join_rounds(parts: Sequence[Round]) -> Round:
    batch = parts[0].batch
    for part in parts[1:]:
        batch = replay.Batch(**{f.name: (getattr(batch, f.name) + getattr(part.batch, f.name)
                                         if isinstance(getattr(batch, f.name), list) else getattr(batch, f.name))
                                for f in dataclasses.fields(batch)})
    return Round(batch, [s for p in parts for s in p.scenes], [m for p in parts for m in p.moves],
                 [k for p in parts for k in p.kinds], [d for p in parts for d in p.directions])


def real_round(batch: replay.Batch, airports: Mapping[str, Any], params: Any, every_landing: Mapping[str, Landings],
               step_s: float) -> Round:
    """``batch``'s flights in their real scenes."""
    limits = limits_s(batch, params, step_s, augmented=False)
    scenes = [scene_of(airports[g.code], s.dataset_id, limit, step_s)
              for s, g, limit in zip(batch.signals, batch.geometries, limits)]
    return Round(batch, scenes, [None] * len(scenes), ["real"] * len(scenes), directions_of(batch, scenes, every_landing))


def augmented_round(batch: replay.Batch, airports: Mapping[str, Any], params: Any, every_landing: Mapping[str, Landings],
                    seed: int | Sequence[int], windows: Mapping[str, tuple[float, float]], step_s: float
                    ) -> tuple[Round, int]:
    """``batch``'s flights in augmented scenes (M3's: the scene over stage 2's longer time limit), those with no
    qualifying draw left out: ``(the round, how many were left out)``."""
    limits = limits_s(batch, params, step_s, augmented=True)
    scenes = [scene_of(airports[g.code], s.dataset_id, limit, step_s)
              for s, g, limit in zip(batch.signals, batch.geometries, limits)]
    kept, scenes, moves, draws, left_out = augmented_scenes(batch, scenes, dict(airports), seed, dict(windows), step_s)
    return Round(kept, scenes, moves, [d["kind"] for d in draws], directions_of(kept, scenes, every_landing)), left_out


def speak(model: Prior, round_: Round, samples: int, words: Words, params: Any, landings: Any,
          procedures: ProcedureMasks, *, generator: torch.Generator, budget: int, source: str) -> SceneSentences:
    """Every flight of ``round_`` spoken to ``samples`` times in its scene (`traffic_free_generation.scene_sentences`,
    in the loop's batches — `speaking_batches`), flight-major in ``round_``'s order, each row with its ``kind`` and
    ``reward``."""
    batch, step_s = round_.batch, words.spec.step_s
    own = limits_s(batch, params, step_s, augmented=False)
    moved = limits_s(batch, params, step_s, augmented=True)
    limits = [own[j] if m is None else moved[j] for j, m in enumerate(round_.moves)]
    count = len(batch.readings)
    rows: list[Any] = [None] * (count * samples)
    said: list[Any] = [None] * (count * samples)
    positions: list[Any] = [None] * (count * samples)
    allowed: list[Any] = [None] * (count * samples)
    for chunk in speaking_batches(round_.scenes, limits, [len(r.words) for r in batch.readings], samples, budget,
                                  step_s):
        here = [round_.scenes[j] for j in chunk]
        got = scene_sentences(model, replay.subset(batch, chunk), here, here, source, words, params, landings, samples,
                              generator=generator, temperature=1.0, procedure_masks=procedures,
                              moves=[round_.moves[j] for j in chunk])
        for k, j in enumerate(chunk):
            for m in range(samples):
                src, dst = k * samples + m, j * samples + m
                row = got.rows[src]
                row["kind"] = round_.kinds[j]
                row["reward"] = float(row[VISUAL]["outcome"] == LANDED and bool(round_.directions[j][row["last_runway"]]))
                rows[dst], said[dst], positions[dst], allowed[dst] = row, got.said[src], got.positions[src], \
                    got.allowed[src]
    return SceneSentences(rows, said, positions, allowed)


def landing_gap_s(scene: Scene, runway: str, landing_s: float) -> float | None:
    """The time since the last landing before ``landing_s`` on ``runway`` or a runway separated as one with it, among
    the scene's others (at their times in the scene); None without one."""
    separation = scene.airport.flights.separation
    before = [scene.track(k).presence.landing_s for k in scene.others
              if separation.one_runway(scene.track(k).presence.runway, runway)
              and scene.track(k).presence.landing_s < landing_s]
    return float(landing_s - max(before)) if before else None


def ordering(rows: Sequence[Mapping[str, Any]], round_: Round, samples: int, step_s: float) -> dict[str, Any]:
    """On real scenes (design §4.3, one aircraft commanded): the sentences that landed with no loss ending them first
    (VISUAL) — their median time from the first predicted step to the landing and median gap to the landing before on
    the runway (`landing_gap_s`: the landing at its first predicted step's time — its scene step, within a second — plus
    its flight time) — each against the same sentences' flights as recorded (a flight counted once a landed sentence;
    a gap where both have one)."""
    times, recorded_times, gaps, recorded_gaps = [], [], [], []
    for s, row in enumerate(rows):
        if row[VISUAL]["outcome"] != LANDED:
            continue
        scene = round_.scenes[s // samples]
        own = scene.track(scene.key).presence
        times.append(row["end_s"])
        recorded_times.append(row["observed_remaining_s"])
        ident = scene.airport.flights.geometry.candidates[row["last_runway"]].ident
        gap = landing_gap_s(scene, ident, scene.first_step_s + N_LOOK * step_s + row["end_s"])
        recorded = landing_gap_s(scene, own.runway, own.landing_s)
        if gap is not None and recorded is not None:
            gaps.append(gap)
            recorded_gaps.append(recorded)

    def median(values: Sequence[float]) -> float | None:
        return float(np.median(values)) if values else None

    out = {"time_to_land_s": median(times), "recorded_time_to_land_s": median(recorded_times),
           "gap_s": median(gaps), "recorded_gap_s": median(recorded_gaps), "landed_with_a_gap": len(gaps)}
    out["time_ratio"] = out["time_to_land_s"] / out["recorded_time_to_land_s"] if times else None
    out["gap_ratio"] = out["gap_s"] / out["recorded_gap_s"] if gaps else None
    return out


def side_readout(rows: Sequence[dict[str, Any]], round_: Round, samples: int, step_s: float, *,
                 real: bool) -> dict[str, Any]:
    """One side of the select readout (module docstring)."""
    # the reward over the sentences the separation summary counts: a flight already in a loss it answers for at its
    # first predicted step is left out (`traffic_free_generation.summary`)
    counted = [r for r in rows if not r["starts_in_a_loss"]]
    out: dict[str, Any] = {"free_generation": grouped(list(rows)), "separation": summary(rows)["scene"],
                           "reward": float(np.mean([r["reward"] for r in counted])),
                           "reward_by_kind": {kind: float(np.mean([r["reward"] for r in counted if r["kind"] == kind]))
                                              for kind in sorted({r["kind"] for r in counted})}}
    if real:
        out["ordering"] = ordering(rows, round_, samples, step_s)
    return out


def traffic_readout(model: Prior, built: Sequence[Built], slots: int, device: torch.device) -> dict[str, Any]:
    """The teacher-forced NLL on scene samples (`traffic_prior_train.scene_evaluate`) and, per layer, the traffic
    attention's output against the residual stream it is added to — RMS over RMS, over the aircraft-steps with another
    aircraft present (with none it adds 0 by construction)."""
    sums = [[0.0, 0.0, 0] for _ in model.layers]

    def hook(index: int) -> Any:
        def record(module: Any, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            x, present, _ = inputs
            cells = present & (present.sum(dim=1, keepdim=True) > 1)
            sums[index][0] += float((output[cells].double() ** 2).sum())
            sums[index][1] += float((x[cells].double() ** 2).sum())
            sums[index][2] += int(cells.sum())
        return record

    handles = [layer.traffic.register_forward_hook(hook(i)) for i, layer in enumerate(model.layers)]
    try:
        nll = scene_evaluate(model, built, TrainConfig(), slots, device)
    finally:
        for handle in handles:
            handle.remove()
    return {"teacher_forced": nll, "traffic_output_over_residual": [math.sqrt(o / x) if x else None for o, x, _ in sums],
            "aircraft_steps_with_another": sums[0][2]}


def sentence_split(round_: Round, spoken: SceneSentences, keep: np.ndarray, samples: int, table: Split,
                   every_landing: Mapping[str, Landings] | None, airports: Sequence[str], step_s: float) -> SceneSplit:
    """The sentences at ``keep`` as the tuner reads them: each one's rows as it read them (`data.chain_record`, its
    landing context its scene's), its own words the targets, every column asked; its scene and positions."""
    flights, scenes, positions = [], [], []
    for s in keep:
        j = int(s) // samples
        signals, said, position, scene = round_.batch.signals[j], spoken.said[s], spoken.positions[s], round_.scenes[j]
        classes = np.where(said != UNCHANGED, said + 1, 0)
        context = None if every_landing is None else scene_landings(every_landing[signals.airport], scene)
        flights.append(chain_record(signals, position[:, 0], position[:, 1], position[:, 2], said, classes,
                                    np.ones(said.shape, dtype=bool), round_.batch.geometries[j], context,
                                    airports.index(signals.airport), round_.batch.readings[j].capture_row, step_s))
        scenes.append(scene)
        positions.append(position)
    return SceneSplit(dataclasses.replace(table, flights=flights), scenes, positions)


def trained_scenes(spoken: SceneSentences, samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(each sentence's advantage — its reward less its scene's mean —, each scene's "starts in a loss the speaking
    aircraft answers for", the sentences trained on: those of the scenes whose sentences differ and that do not start in
    a loss)``."""
    by_scene = np.array([r["reward"] for r in spoken.rows]).reshape(-1, samples)
    starts_lost = np.array([any(r["starts_in_a_loss"] for r in spoken.rows[j * samples: (j + 1) * samples])
                            for j in range(len(by_scene))])
    trained = (by_scene.min(axis=1) != by_scene.max(axis=1)) & ~starts_lost
    return group_advantages(by_scene).reshape(-1), starts_lost, np.flatnonzero(np.repeat(trained, samples))


def write_sentences(path: Path, round_: Round, spoken: SceneSentences, advantages: np.ndarray) -> None:
    steps = np.array([len(s) for s in spoken.said])
    np.savez_compressed(path, dataset_id=np.array([s.dataset_id for s in round_.batch.signals]),
                        kind=np.array(round_.kinds), step_offsets=np.concatenate(([0], np.cumsum(steps))),
                        said=np.concatenate(spoken.said).astype(np.int16),
                        outcome=np.array([r[VISUAL]["outcome"] for r in spoken.rows]),
                        runway=np.array([r["last_runway"] for r in spoken.rows]),
                        reward=np.array([r["reward"] for r in spoken.rows]), advantage=advantages)


def round_summary(round_: Round, spoken: SceneSentences, samples: int) -> dict[str, Any]:
    """How a round's sentences did: the reward, the lost separation and the executor's landed share, in all and per kind;
    the scenes with a contrast, the ones starting in a loss."""
    rows = spoken.rows
    reward = np.array([r["reward"] for r in rows]).reshape(-1, samples)
    _, starts_lost, _ = trained_scenes(spoken, samples)

    def shares(members: Sequence[dict[str, Any]]) -> dict[str, float]:
        return {"sentences": len(members), "reward": float(np.mean([r["reward"] for r in members])),
                "lost_separation": float(np.mean([r[VISUAL]["outcome"] == LOST_SEPARATION for r in members])),
                "landed": float(np.mean([r["outcome"] == LANDED for r in members]))}

    return {"all": shares(rows),
            "by_kind": {kind: shares([r for r in rows if r["kind"] == kind]) for kind in sorted(set(round_.kinds))},
            "scenes": len(round_.scenes), "scenes_starting_in_a_loss": int(starts_lost.sum()),
            "scenes_with_contrast": int(((reward.min(axis=1) != reward.max(axis=1)) & ~starts_lost).sum()),
            "kinds": dict(Counter(round_.kinds))}


def ordering_failures(row: Mapping[str, Any]) -> list[str]:
    """The ordering guards (design §4.3) a round's real-scene readout fails: its median time to land and its median gap to
    the landing before on the runway at most `GUARD_ORDER_GROWTH` × the same flights' recorded ones — unreadable (no
    landing, or none with a landing before it) fails. Against the record, not round 0: the start is read against them
    before training (`main` refuses a start that fails them — no round could then be chosen)."""
    order = row["real"]["ordering"]
    return [name for name, ratio in (("time to land", order["time_ratio"]), ("landing gap", order["gap_ratio"]))
            if ratio is None or ratio > GUARD_ORDER_GROWTH]


def guarded_choice(history: Sequence[Mapping[str, Any]], labelled: Mapping[str, Mapping[str, float]]
                   ) -> tuple[int, dict[int, list[str]]]:
    """``(the round kept, each excluded round's failed guards)`` (module docstring)."""
    start = history[0]
    margin = math.log(GUARD_WORD_GROWTH)
    excluded: dict[int, list[str]] = {}
    for row in history:
        failed = []
        real = row["real"]["free_generation"]["all"]
        first = start["real"]["free_generation"]["all"]
        if real["outcomes"]["landed"] < first["outcomes"]["landed"] - GUARD_LANDED_DROP:
            failed.append("landed")
        if (real["landed_on_observed_runway"] is None
                or real["landed_on_observed_runway"] < first["landed_on_observed_runway"] - GUARD_RUNWAY_DROP):
            failed.append("observed runway")
        if row["real"]["separation"]["lost_separation"] > start["real"]["separation"]["lost_separation"] + GUARD_LOSS_RISE:
            failed.append("lost separation")
        failed += ordering_failures(row)
        for side in ("real", "augmented"):
            said = row[side]["free_generation"]["all"]["words_after_first_per_flight"]
            first_said = start[side]["free_generation"]["all"]["words_after_first_per_flight"]
            now, then = word_distance(said, labelled[side]), word_distance(first_said, labelled[side])
            failed += [f"{side} {c} words" for c in GUARD_WORD_COLUMNS if now[c] > then[c] + margin]
        if failed:
            excluded[row["round"]] = failed
    candidates = [row for row in history if row["round"] not in excluded]
    best = max(row["augmented"]["reward"] for row in candidates)
    return next(row["round"] for row in candidates if row["augmented"]["reward"] >= best - TIE_SHARE), excluded


def write_traffic_prior(directory: Path, model: Prior, start_dir: Path, start: LoadedPrior, spec_sha256: str, *,
                        git: Mapping[str, Any], smoke: bool, fine_tuning: Mapping[str, Any]) -> None:
    """``model`` as a prior run `prior_train.load_prior` reads (`prior_train.TRAFFIC_CHECKPOINT_SCHEMA`): its checkpoint
    — the start's payload with this state, the edge and traffic features, the edge code's hash and ``start`` (the
    single-aircraft prior it grew from: its directory and checkpoint sha256) — its ``config.json`` (the start's, with
    these) and the procedure's masks it speaks under (the start's; ``start`` as `load_prior` opened ``start_dir``)."""
    grown_from = {"directory": str(start_dir), "checkpoint_sha256": file_sha256(start_dir / "checkpoint.pt")}
    features = {"edge_features": list(model.edge_features), "traffic_features": list(model.traffic_features),
                "edge_source_sha256": edge_source_sha256()}
    torch.save({"schema": TRAFFIC_CHECKPOINT_SCHEMA, "model_config": model.config.to_dict(),
                "train_config": start.config["train"], "state": copy.deepcopy(model.state_dict()),
                "spec_sha256": spec_sha256, **features, "start": grown_from}, directory / "checkpoint.pt")
    write_json_atomic(directory / "config.json", {**start.config, "schema": TRAFFIC_CHECKPOINT_SCHEMA,
                                                  "written_utc": utc_now(), "git": dict(git), "smoke": smoke, **features,
                                                  "parameters": sum(p.numel() for p in model.parameters()),
                                                  "start": grown_from, "fine_tuning": dict(fine_tuning)})
    write_masks(directory, start.procedure_masks, writer=RUNNER, git=dict(git))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the start: augmented (a single-aircraft prior)")
    parser.add_argument("--base", type=Path, required=True, help="the pull's reference: base")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--real-per-airport", type=int, default=200, help="real scenes an airport a round")
    parser.add_argument("--augmented-per-airport", type=int, default=200, help="augmented scenes an airport a round")
    parser.add_argument("--samples", type=int, default=8, help="sentences a scene")
    parser.add_argument("--select-per-airport", type=int, default=200)
    parser.add_argument("--select-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--aircraft-steps", type=int, default=AIRCRAFT_STEPS, help="a loop batch's most "
                        "(`traffic_free_generation.speaking_batches`)")
    parser.add_argument("--traffic-learning-rate", type=float, default=TRAFFIC_LEARNING_RATE)
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
        parser.error("a scene's sentences are compared with each other: --samples ≥ 2")
    if args.data_weight <= 0.0:
        parser.error("the post-training trains beside the data (design §6.2): --data-weight > 0")
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("the tree has uncommitted changes; a fine-tuning run is made at a commit")
    config = RewardConfig(**{f: getattr(args, f) for f in asdict(RewardConfig())})
    started = time.perf_counter()
    device = torch.device(args.device)
    params, record, words = replay.open_executor(executor_dir, instructions)
    spec = load_spec(instructions)
    step_s = spec.step_s
    start = load_prior(prior_dir, instructions)
    single, start_payload, start_config, start_masks = start
    base, base_payload, base_config, _ = load_prior(base_dir, instructions)
    for directory, payload, loaded in ((prior_dir, start_payload, start_config), (base_dir, base_payload, base_config)):
        if loaded["smoke"] or payload["schema"] != PRIOR_CHECKPOINT_SCHEMA:
            parser.error(f"{directory} is not a single-aircraft prior's formal run")
    if base.config.to_dict() != single.config.to_dict():
        parser.error("the base model and the start are not one architecture")
    torch.manual_seed(args.seed)                        # the traffic attention's weights: at zero it reads nothing
    model = with_traffic(single, EDGE_FEATURES).to(device)
    base.to(device)
    variant, airports, slots = model.config.variant, model.config.airports, model.config.candidate_slots
    every_landing = airport_landings(instructions, rosters(instructions))
    landings = every_landing if VARIANTS[variant].landing_context else None
    geometries = load_candidates(instructions)
    table = Split([], airports, candidate_table(geometries, airports, slots), *runway_names(geometries, airports),
                  column_classes(words, slots), variant)
    windows = start_altitude_windows(instructions)
    # the training days built once: the loop's scenes and the data term's samples read each flight's same rows
    train_flights, train_counts = airport_flights(instructions, "train", spec, airports, landings, model.config.max_rows)
    train_airports = with_tracks(instructions, "train", spec, train_flights)
    data, data_counts = split_samples(train_flights, train_counts, step_s)
    data_counts["samples_asking_nothing"] = sum(not b.sample.asks for b in data)
    data = [b for b in data if b.sample.asks]
    select_flights, select_counts = airport_flights(instructions, "select", spec, airports, landings,
                                                    model.config.max_rows)
    select_airports = with_tracks(instructions, "select", spec, select_flights)
    select_built = [b for b in split_samples(select_flights, select_counts, step_s)[0] if b.sample.asks]
    select_batch = replay.draw(instructions, "select", spec, words, per_airport=args.select_per_airport, seed=args.seed)
    select_real = real_round(select_batch, select_airports, params, every_landing, step_s)
    select_augmented, select_left_out = augmented_round(select_batch, select_airports, params, every_landing,
                                                        args.seed + SELECT_AUGMENT_OFFSET, windows, step_s)
    labelled = {"real": labelled_words(select_batch), "augmented": labelled_words(select_augmented.batch)}
    recorded = summary(recorded_rows(select_batch, select_real.scenes, words))["recorded"]
    out.mkdir(parents=True)
    write_json_atomic(out / "config.json", {
        "schema": TRAFFIC_REWARD_SCHEMA, "written_utc": utc_now(), "git": git, "smoke": args.smoke,
        "prior": {"directory": str(prior_dir), "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt"),
                  "procedure_masks": list(start_masks.names)},
        "base": {"directory": str(base_dir), "checkpoint_sha256": file_sha256(base_dir / "checkpoint.pt")},
        "executor": {"directory": str(executor_dir), "sha256": record["sha256"]}, "instructions": str(instructions),
        "edge_features": list(EDGE_FEATURES), "edge_source_sha256": edge_source_sha256(),
        "optimiser": asdict(config), "traffic_learning_rate": args.traffic_learning_rate, "rounds": args.rounds,
        "real_per_airport": args.real_per_airport, "augmented_per_airport": args.augmented_per_airport,
        "samples": args.samples, "pool_factor": POOL_FACTOR, "aircraft_steps": args.aircraft_steps,
        "data": {"split": "train", "scene_samples": len(data), "built": data_counts},
        "select": {"per_airport": args.select_per_airport, "samples": args.select_samples, "drawn": select_batch.drawn,
                   "augment_seed": args.seed + SELECT_AUGMENT_OFFSET, "augmented_left_out": select_left_out,
                   "augmented_kinds": dict(Counter(select_augmented.kinds)), "scene_samples": len(select_built),
                   "labelled_words_after_first_per_flight": labelled, "recorded": recorded},
        "seed": args.seed, "tie_share": TIE_SHARE,
        "guards": {"landed_drop": GUARD_LANDED_DROP, "runway_drop": GUARD_RUNWAY_DROP, "loss_rise": GUARD_LOSS_RISE,
                   "order_growth": GUARD_ORDER_GROWTH, "word_columns": list(GUARD_WORD_COLUMNS),
                   "word_margin_ln": math.log(GUARD_WORD_GROWTH),
                   "lost_separation_target": recorded["lost_separation"] * GUARD_ORDER_GROWTH},
        "n_look": N_LOOK})

    def log(line: str) -> None:
        print(f"{line}  [{time.perf_counter() - started:.0f}s]", flush=True)

    log(f"train days: {len(data)} scene samples ({data_counts}); select: {len(select_batch.readings)} flights, "
        f"{len(select_augmented.scenes)} augmented ({dict(Counter(select_augmented.kinds))}, {select_left_out} left out); "
        f"recorded lost separation {recorded['lost_separation']:.3f}")

    def read_select(round_number: int) -> dict[str, Any]:
        model.eval()
        readout: dict[str, Any] = {}
        for side, round_ in (("real", select_real), ("augmented", select_augmented)):
            generator = torch.Generator(device=device).manual_seed(args.seed)
            spoken = speak(model, round_, args.select_samples, words, params, landings, start_masks,
                           generator=generator, budget=args.aircraft_steps, source="scene")
            readout[side] = side_readout(spoken.rows, round_, args.select_samples, step_s, real=side == "real")
            readout[side]["flights"] = spoken.rows
        readout["traffic"] = traffic_readout(model, select_built, slots, device)
        real, augmented = readout["real"], readout["augmented"]
        log(f"round {round_number}: select reward real {real['reward']:.3f} augmented {augmented['reward']:.3f} "
            f"({' '.join(f'{k} {v:.3f}' for k, v in augmented['reward_by_kind'].items())}); lost separation real "
            f"{real['separation']['lost_separation']:.3f} augmented {augmented['separation']['lost_separation']:.3f}; "
            f"landed real {real['free_generation']['all']['outcomes']['landed']:.3f}; ordering time "
            f"{real['ordering']['time_ratio']} gap {real['ordering']['gap_ratio']}; TF NLL "
            f"{readout['traffic']['teacher_forced']['nll_per_step']:.4f}; traffic / residual "
            + " ".join("-" if r is None else f"{r:.2e}" for r in readout["traffic"]["traffic_output_over_residual"]))
        return readout

    def history_row(round_number: int, readout: Mapping[str, Any], **more: Any) -> dict[str, Any]:
        return {"round": round_number,
                **{side: {k: v for k, v in readout[side].items() if k != "flights"} for side in ("real", "augmented")},
                "traffic": readout["traffic"], **more}

    directory = out / "round_00"
    directory.mkdir()
    readout = read_select(0)
    write_json_atomic(directory / "readout.json", readout)
    history = [history_row(0, readout)]
    failing = ordering_failures(history[0])
    if failing:
        raise SystemExit(f"the start fails the ordering guards {failing} against the record "
                         f"({history[0]['real']['ordering']}): no round could be chosen — not training")
    tuner = SceneRewardTuner(model, base, config, device, seed=args.seed,
                             traffic_learning_rate=args.traffic_learning_rate, step_s=step_s)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for round_number in range(1, args.rounds + 1):
        directory = out / f"round_{round_number:02d}"
        directory.mkdir()
        pool = replay.draw(instructions, "train", spec, words,
                           per_airport=round((args.real_per_airport + args.augmented_per_airport) * POOL_FACTOR),
                           seed=args.seed + round_number)
        real = real_starts(pool, args.real_per_airport)
        taken = set(real)
        candidates, left_out = augmented_round(replay.subset(pool, [j for j in range(len(pool.signals))
                                                                    if j not in taken]),
                                               train_airports, params, every_landing, [args.seed, round_number], windows,
                                               step_s)
        kept = first_per_airport(candidates.batch, args.augmented_per_airport, sorted({s.airport for s in pool.signals}))
        augmented = Round(replay.subset(candidates.batch, kept), [candidates.scenes[j] for j in kept],
                          [candidates.moves[j] for j in kept], [candidates.kinds[j] for j in kept],
                          [candidates.directions[j] for j in kept])
        round_ = join_rounds([real_round(replay.subset(pool, real), train_airports, params, every_landing, step_s),
                              augmented])
        model.eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        spoken = speak(model, round_, args.samples, words, params, landings, start_masks, generator=generator,
                       budget=args.aircraft_steps, source="train")
        earned = np.array([r["reward"] for r in spoken.rows])
        advantages, _, keep = trained_scenes(spoken, args.samples)
        write_sentences(directory / "sentences.npz", round_, spoken, advantages)
        described = {**round_summary(round_, spoken, args.samples), "trained_on": len(keep),
                     "augmented_left_out": left_out, "drawn": pool.drawn}
        write_json_atomic(directory / "sentences.json", described)
        log(f"round {round_number}: {len(earned)} sentences, reward {described['all']['reward']:.3f} ("
            + " ".join(f"{k} {v['reward']:.3f}" for k, v in described["by_kind"].items())
            + f"), lost separation {described['all']['lost_separation']:.3f}, {described['scenes_with_contrast']} scenes "
              f"with a contrast, {described['scenes_starting_in_a_loss']} starting in a loss")
        sentences = sentence_split(round_, spoken, keep, args.samples, table, landings, airports, step_s)
        allowed = [spoken.allowed[s] for s in keep]
        is_real = np.array([round_.kinds[int(s) // args.samples] == "real" for s in keep])
        start_distance, measured_on = {}, {}
        for side, mask in (("real", is_real), ("augmented", ~is_real)):
            members = np.flatnonzero(mask)
            measured_on[side] = len(members)
            start_distance[side] = (tuner.distance(sentences.subset(members), [allowed[i] for i in members])
                                    if len(members) else None)
        passed = {**tuner.one_pass(sentences, advantages[keep], data, allowed, slots=slots),
                  "distance_at_start": start_distance, "distance_sentences": measured_on}
        if device.type == "cuda":
            passed["gpu_peak_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        log(f"round {round_number}: one pass over {len(keep)} sentences ({passed['batches']} updates), reward term "
            f"{passed['reward_mean']:.4f}, KL to the base at the start {start_distance}, {passed['kl_mean']:.4f} in the "
            f"pass (max {passed['kl_max']:.4f}), data NLL {passed['data_mean']:.4f}, words outside the clip "
            f"{passed['clipped_share']:.4f}, {passed['seconds']:.0f}s"
            + (f", GPU peak {passed['gpu_peak_gb']:.2f} GB (the round's speaking and pass)" if "gpu_peak_gb" in passed
               else ""))
        write_traffic_prior(directory, model, prior_dir, start, spec.sha256, git=git, smoke=args.smoke,
                            fine_tuning={"schema": TRAFFIC_REWARD_SCHEMA, "from": str(prior_dir), "base": str(base_dir),
                                         "round": round_number, "optimiser": asdict(config),
                                         "traffic_learning_rate": args.traffic_learning_rate,
                                         "samples": args.samples})
        readout = read_select(round_number)
        write_json_atomic(directory / "readout.json", readout)
        history.append(history_row(round_number, readout, train_pass=passed,
                                   sentences={k: v for k, v in described.items() if k != "drawn"}))
        write_json_atomic(out / "history.json", {"rounds": history})
    kept_round, excluded = guarded_choice(history, labelled)
    write_json_atomic(out / "choice.json", {
        "rule": f"within round 0's guards (real scenes: landed ≥ − {GUARD_LANDED_DROP}, on the observed runway ≥ − "
                f"{GUARD_RUNWAY_DROP}, lost separation ≤ + {GUARD_LOSS_RISE}, median time to land and landing gap ≤ "
                f"{GUARD_ORDER_GROWTH} × recorded; real and augmented: each of {', '.join(GUARD_WORD_COLUMNS)} words per "
                f"flight off the labelled ≤ round 0's + ln {GUARD_WORD_GROWTH}), the highest augmented-scene reward; "
                f"within {TIE_SHARE} the earliest",
        "augmented_reward": [row["augmented"]["reward"] for row in history],
        "real_reward": [row["real"]["reward"] for row in history], "excluded_by_the_guards": excluded,
        "round": kept_round,
        "directory": str(out / f"round_{kept_round:02d}") if kept_round else str(prior_dir)})
    log(f"kept round {kept_round} (the guards excluded {excluded}) → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
