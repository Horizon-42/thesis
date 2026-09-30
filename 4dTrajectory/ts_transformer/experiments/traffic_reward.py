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
4. **``--passes`` passes** (`traffic_tuner.SceneRewardTuner`; design §6.6 step 6 item 14: each sweep a new order of
   batches, all against the model the sentences were said by): the second stage's loss, each sentence scored in its
   scene, the pull to base reading the aircraft alone, the data term on the training days' scene samples (M2's: each
   flight's rows the ones the loop's scenes read), the traffic attention at ``--traffic-learning-rate``; the distance to
   base on the fresh sentences before the pass, the real scenes' and the augmented ones' apart;
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
`GUARD_WORD_GROWTH` (stage 2's) — the augmented-scene reward, its ties read against paired standard errors (design
§6.2, §9 item 27): the select readouts speak the same draws every round, so two rounds compare sentence by sentence
and their difference's standard error is √(sentences whose reward flipped) / sentences (`paired_difference`, over the
sentences counted in both). A candidate beats round 0 by at least `TIE_STANDARD_ERRORS` of them; of the candidates,
the earliest the highest does not beat by as much; none: round 0. Round 0 kept means the start (augmented) is. Val is
not read here.

**Round by round** (design §6.6 step 6 item 11): every stream a round draws from is its own — the pool (``seed`` + the
round), the augmentations, the sentences said and the pass's batches and data ([``seed``, round, `AUGMENT_STREAM` /
`SAMPLING_STREAM` / `PASS_STREAM`]) — and each round leaves beside its weights the optimiser's state (`OPTIMISER_FILE`:
AdamW's moments, the warm-up's step), so ``--resume`` continues a run from its last finished round up to ``--rounds`` as
one invocation would have (the same streams; on the GPU its kernels' sums differ at rounding between any two runs):
refused when the run was made with another configuration, code or device (its ``config.json``, the git commit included,
apart from `RESUMABLE`: run it from a checkout fixed at one commit) or a round did not finish (its directory without
``readout.json``, written last — move it aside as ``round_<k>.aborted-<UTC>``; nothing is deleted), and when the start
failed the ordering guards. Round 0 is the first invocation's; ``choice.json`` is written over the rounds finished at the
end of each (``--resume --rounds`` the last finished round writes it alone).

**In several processes** (design §6.6 step 6 item 13): the sentences — a round's and the select readouts' — are spoken by
``--speakers`` processes (`Speakers`), forked once the data are built and before this process starts the GPU; each
rebuilds a training round from its number and speaks the loop batches dealt to it, each batch from its own stream
(`batch_seed`), so the sentences are the same whatever the number of processes (not part of the run: a resume may change
it). The pass, the distances and the teacher-forced readout stay in this process. A speaking process that fails ends
the run with its traceback.

Writes into ``--out`` (a new directory, from a clean tree unless ``--smoke``; the same one with ``--resume``):
``config.json``, ``round_00/readout.json``, ``round_<k>/{sentences.npz, sentences.json, checkpoint.pt
(`prior_train.TRAFFIC_CHECKPOINT_SCHEMA`), optimiser.pt, config.json, procedure_masks.json, readout.json}``,
``history.json``, ``choice.json``.

    python run_ts.py traffic_reward \\
        --prior 4dTrajectory/outputs/POOLED/prior/v3_stage2_clip_20260926/aug_s1337/round_07 \\
        --base 4dTrajectory/outputs/POOLED/prior/v3_step1_20260924/full_s1337 \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --executor 4dTrajectory/outputs/POOLED/executor/v11_20260927 \\
        --out 4dTrajectory/outputs/POOLED/prior/<campaign>/traffic_s1337 --rounds 1
    # … then, the same command with --resume --rounds 2, and so on
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import dataclasses
import gc
import json
import math
import multiprocessing
import os
import signal
import sys
import traceback
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.experiments.prior_augmented_reward import (
    GUARD_LANDED_DROP, GUARD_WORD_COLUMNS, GUARD_WORD_GROWTH, POOL_FACTOR, SELECT_AUGMENT_OFFSET, labelled_words,
    real_starts, word_distance,
)
from ts_transformer.experiments.prior_free_generation import grouped, limits_s, start_altitude_windows
from ts_transformer.experiments.prior_landing_reward import GUARD_RUNWAY_DROP
from ts_transformer.experiments.prior_train import (
    PRIOR_CHECKPOINT_SCHEMA, TRAFFIC_CHECKPOINT_SCHEMA, LoadedPrior, load_prior, rosters,
)
from ts_transformer.experiments.traffic_loop import LOST_SEPARATION
from ts_transformer.experiments.traffic_free_generation import (
    SceneSentences, augmented_scenes, recorded_rows, scene_sentences, speaking_batches, summary,
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
#: A round must beat round 0 on the augmented scenes by this many paired standard errors to be chosen, and the highest
#: such round the earliest by as many (design §6.2, §9 item 27: the old tie, stage 2's 0.015, knew nothing of the
#: noise).
TIE_STANDARD_ERRORS = 2.0
#: A round's own random streams beyond its pool (``seed`` + the round), each [``seed``, round, stream] (module docstring):
#: the augmentations, the sentences said, the pass's batches and data. (A seed list is padded with zeros: [seed, round]
#: is the augmentations' stream.)
AUGMENT_STREAM = 0
SAMPLING_STREAM = 1
PASS_STREAM = 2
#: The select readouts' streams (the same every round: [seed, 0, stream]), per side.
SELECT_STREAMS = {"real": 3, "augmented": 4}
#: Beside a round's weights: what its pass leaves for the next (`traffic_tuner.SceneRewardTuner.state`).
OPTIMISER_FILE = "optimiser.pt"
#: What a resumed run's ``config.json`` may differ in: when it was written, how far it was asked to go and its resumptions.
RESUMABLE = ("written_utc", "rounds", "resumed", "speakers")
#: A speaking process's loop batch, its most aircraft-steps (`traffic_free_generation.speaking_batches`): chosen for four
#: processes' pasts and blocks beside the parent's models within the 8 GB GPU (a third of M3's one-process batch; each
#: process's peak is logged, design §6.6 step 6 item 13).
SPEAKER_AIRCRAFT_STEPS = 100_000


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


def speaking_plan(round_: Round, samples: int, params: Any, budget: int, step_s: float) -> list[list[int]]:
    """``round_``'s flights in the loop's batches (`traffic_free_generation.speaking_batches`: by the size of their
    scenes, each at most ``budget`` aircraft-steps over its ``samples`` loops a flight; a moved start's time limit is
    stage 2's) — the same in every process."""
    batch = round_.batch
    own = limits_s(batch, params, step_s, augmented=False)
    moved = limits_s(batch, params, step_s, augmented=True)
    limits = [own[j] if m is None else moved[j] for j, m in enumerate(round_.moves)]
    return speaking_batches(round_.scenes, limits, [len(r.words) for r in batch.readings], samples, budget, step_s)


def batch_seed(seed: int, number: int) -> int:
    """A torch generator's seed for the loop batch ``number`` of a stream seeded ``seed``: each batch its own, so what it
    says depends neither on the batches before it nor on the process that speaks it (design §6.6 step 6 item 13)."""
    return int(np.random.SeedSequence([seed, number]).generate_state(1)[0])


def speak_batch(model: Prior, round_: Round, chunk: Sequence[int], number: int, samples: int, words: Words, params: Any,
                landings: Any, procedures: ProcedureMasks, *, seed: int, source: str) -> SceneSentences:
    """The flights of ``round_`` at ``chunk`` — its loop batch ``number`` (`speaking_plan`) — spoken to ``samples`` times
    each in their scenes (`traffic_free_generation.scene_sentences`, the batch's own stream: `batch_seed`), flight-major
    in the chunk's order, each row with its ``kind`` and ``reward``."""
    here = [round_.scenes[j] for j in chunk]
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(batch_seed(seed, number))
    got = scene_sentences(model, replay.subset(round_.batch, list(chunk)), here, here, source, words, params, landings,
                          samples, generator=generator, temperature=1.0, procedure_masks=procedures,
                          moves=[round_.moves[j] for j in chunk])
    for k, j in enumerate(chunk):
        for m in range(samples):
            row = got.rows[k * samples + m]
            row["kind"] = round_.kinds[j]
            row["reward"] = float(row[VISUAL]["outcome"] == LANDED and bool(round_.directions[j][row["last_runway"]]))
    return got


def assemble(round_: Round, samples: int, plan: Sequence[Sequence[int]], parts: Mapping[int, SceneSentences]
             ) -> SceneSentences:
    """The loop batches' sentences (``parts``, by batch number) flight-major in ``round_``'s order."""
    count = len(round_.batch.readings)
    rows: list[Any] = [None] * (count * samples)
    said: list[Any] = [None] * (count * samples)
    positions: list[Any] = [None] * (count * samples)
    allowed: list[Any] = [None] * (count * samples)
    for number, chunk in enumerate(plan):
        got = parts[number]
        for k, j in enumerate(chunk):
            for m in range(samples):
                src, dst = k * samples + m, j * samples + m
                rows[dst], said[dst], positions[dst], allowed[dst] = got.rows[src], got.said[src], got.positions[src], \
                    got.allowed[src]
    return SceneSentences(rows, said, positions, allowed)


def speak(model: Prior, round_: Round, samples: int, words: Words, params: Any, landings: Any,
          procedures: ProcedureMasks, *, seed: int, budget: int, source: str) -> SceneSentences:
    """Every flight of ``round_`` spoken to ``samples`` times in its scene, in this process: `speak_batch` over
    `speaking_plan`, assembled (`Speakers` speaks the same batches in several)."""
    plan = speaking_plan(round_, samples, params, budget, words.spec.step_s)
    return assemble(round_, samples, plan, {number: speak_batch(model, round_, chunk, number, samples, words, params,
                                                                landings, procedures, seed=seed, source=source)
                                            for number, chunk in enumerate(plan)})


@dataclasses.dataclass(frozen=True)
class Speaking:
    """What every speaking process speaks with — the vocabulary, the executor's parameters, the landing context, the
    procedure's masks and the loop batch's most aircraft-steps — and how a round is spoken (a window runner speaks its
    own: `traffic_window_reward.WindowSpeaking`): its plan of loop batches, one batch spoken, the round's fingerprint
    and the batches assembled."""

    words: Words
    params: Any
    landings: Any
    procedures: ProcedureMasks
    budget: int

    def plan(self, round_: Any, samples: int) -> list[list[int]]:
        return speaking_plan(round_, samples, self.params, self.budget, self.words.spec.step_s)

    def speak(self, model: Prior, round_: Any, chunk: Sequence[int], number: int, samples: int, *, seed: int,
              source: str) -> Any:
        return speak_batch(model, round_, chunk, number, samples, self.words, self.params, self.landings,
                           self.procedures, seed=seed, source=source)

    def fingerprint(self, round_: Any) -> Any:
        return round_fingerprint(round_)

    def assemble(self, round_: Any, samples: int, plan: Sequence[Sequence[int]], parts: Mapping[int, Any]) -> Any:
        return assemble(round_, samples, plan, parts)


def round_fingerprint(round_: Round) -> list[tuple[Any, ...]]:
    """What must be the same in a round rebuilt by a speaking process and the parent's: each flight, its kind, its moved
    start, its scene's others and the flights moved or inserted in it (their keys and first row times)."""
    return [(signals.dataset_id, kind, None if move is None else dataclasses.asdict(move), tuple(scene.others),
             tuple((rows.presence.dataset_id, float(rows.presence.times_s[0])) for rows, _ in scene.moved))
            for signals, kind, move, scene in zip(round_.batch.signals, round_.kinds, round_.moves, round_.scenes)]


#: `prctl` option: the signal a process gets when its parent dies (linux/prctl.h).
PR_SET_PDEATHSIG = 1


def _speaker(pipe: Any, parent_ends: Sequence[Any], parent_pid: int, rounds: Callable[[int], Round],
             select: Mapping[str, Round], model: Prior, device: str, speaking: Speaking) -> None:
    """A speaking process (`Speakers`): on Linux it dies with the parent (a killed parent leaves none holding the GPU), keeps
    none of the other processes' pipes, runs one thread (the processes are the parallelism) and starts the GPU here,
    after the fork; then answers each task — (kind, key, weights, (its index, the processes), samples, seed, source) —
    with the batches of the plan its index deals it, the round's fingerprint and its GPU peak, or the traceback that ended
    it (also printed)."""
    if sys.platform == "linux":                                 # the death signal is Linux's; elsewhere none is asked for
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG)")
        if os.getppid() != parent_pid:
            os._exit(1)
    for end in parent_ends:
        end.close()
    torch.set_num_threads(1)
    try:
        device_ = torch.device(device)
        model = model.to(device_).eval()
        built: dict[int, Round] = {}
        while True:
            task = pipe.recv()
            if task is None:
                return
            kind, key, state, (index, count), samples, seed, source = task
            model.load_state_dict(state)
            if kind == "select":
                round_ = select[key]
            else:
                if key not in built:
                    built.clear()
                    built[key] = rounds(key)
                round_ = built[key]
            plan = speaking.plan(round_, samples)
            parts = {n: speaking.speak(model, round_, plan[n], n, samples, seed=seed, source=source)
                     for n in range(index, len(plan), count)}
            peak = 0.0
            if device_.type == "cuda":
                peak = torch.cuda.max_memory_reserved(device_) / 1e9
                torch.cuda.empty_cache()                        # the parent's pass needs the GPU next
                torch.cuda.reset_peak_memory_stats(device_)     # after: the next task's peak starts from the emptied cache
            pipe.send(("ok", parts, speaking.fingerprint(round_), peak))
    except BaseException:
        traceback.print_exc()
        pipe.send(("failed", traceback.format_exc()))


class Speakers:
    """``count`` processes speaking a round's loop batches in parallel (design §6.6 step 6 item 13), forked from the
    parent once its data are built and BEFORE it starts the GPU: each holds the parent's data (shared, not copied — the
    caller freezes the collector's view of it first), rebuilds a training round from its number with the parent's own
    ``rounds`` (a scene holds its airport's whole data: rebuilt, not sent), reads the select rounds (``select``, by side)
    as built, and speaks the batches its index deals it (every batch its own stream: the sentences do not depend on
    ``count``). `send` starts them, `receive` assembles the round in the parent's order after checking each process
    rebuilt the parent's round; a process that fails, or is gone, ends the run with what it said."""

    def __init__(self, count: int, rounds: Callable[[int], Any], select: Mapping[str, Any], model: Prior,
                 device: str, speaking: Speaking) -> None:
        if count < 1:
            raise ValueError("at least one speaking process")
        context = multiprocessing.get_context("fork")
        self.speaking, self.pipes, self.processes = speaking, [], []
        for _ in range(count):
            parent, child = context.Pipe()
            process = context.Process(target=_speaker, args=(child, list(self.pipes), os.getpid(), rounds, select,
                                                             model, device, speaking), daemon=True)
            process.start()
            child.close()
            self.pipes.append(parent)
            self.processes.append(process)

    def _gone(self, w: int) -> SystemExit:
        """Speaking process ``w`` is gone: what it said last, or its exit code."""
        pipe, process = self.pipes[w], self.processes[w]
        try:
            said = pipe.recv() if pipe.poll() else None             # a closed pipe polls readable and has nothing
        except (EOFError, OSError):
            said = None
        if said is not None and said[0] == "failed":
            return SystemExit(f"speaking process {w} failed:\n{said[1]}")
        process.join(timeout=5)
        return SystemExit(f"speaking process {w} is gone (exit code {process.exitcode})")

    def send(self, kind: str, key: Any, model: Prior, samples: int, *, seed: int, source: str) -> None:
        """Start the processes on training round ``key`` (``kind`` "train") or select side ``key`` ("select") with
        ``model``'s weights."""
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        for w, pipe in enumerate(self.pipes):
            try:
                pipe.send((kind, key, state, (w, len(self.pipes)), samples, seed, source))
            except OSError:
                raise self._gone(w) from None

    def receive(self, round_: Any, samples: int) -> tuple[Any, list[float]]:
        """The sentences of `send`'s round (``round_``: the parent's own), assembled as `speak` does, and each
        process's GPU peak (GB reserved)."""
        fingerprint = self.speaking.fingerprint(round_)
        parts: dict[int, SceneSentences] = {}
        peaks = []
        for w, pipe in enumerate(self.pipes):
            try:
                status, *payload = pipe.recv()
            except (EOFError, OSError):
                raise self._gone(w) from None
            if status != "ok":
                raise SystemExit(f"speaking process {w} failed:\n{payload[0]}")
            got, rebuilt, peak = payload
            if rebuilt != fingerprint:
                raise SystemExit(f"speaking process {w} rebuilt another round than this process's")
            parts.update(got)
            peaks.append(peak)
        plan = self.speaking.plan(round_, samples)
        if sorted(parts) != list(range(len(plan))):
            raise SystemExit(f"the speaking processes returned batches {sorted(parts)} of {len(plan)}")
        return self.speaking.assemble(round_, samples, plan, parts), peaks

    def speak(self, kind: str, key: Any, round_: Any, model: Prior, samples: int, *, seed: int, source: str
              ) -> tuple[Any, list[float]]:
        """`send` then `receive`."""
        self.send(kind, key, model, samples, seed=seed, source=source)
        return self.receive(round_, samples)

    def close(self) -> None:
        for pipe in self.pipes:
            try:
                pipe.send(None)
            except OSError:
                pass                                            # gone already: nothing to stop
        for process in self.processes:
            process.join(timeout=60)


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


def guarded_choice(history: Sequence[Mapping[str, Any]], labelled: Mapping[str, Mapping[str, float]],
                   rewards: Sequence[Mapping[Any, float]]) -> tuple[int, dict[int, list[str]]]:
    """``(the round kept, each excluded round's failed guards)`` (module docstring); ``rewards`` each round's augmented
    select sentences' (`select_rewards`)."""
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

    def beats(k: int, j: int) -> bool:
        difference, error = paired_difference(rewards[j], rewards[k])
        return difference > 0.0 and difference >= TIE_STANDARD_ERRORS * error

    candidates = [row["round"] for row in history
                  if row["round"] > 0 and row["round"] not in excluded and beats(row["round"], 0)]
    if not candidates:
        return 0, excluded
    # the highest by its paired difference from round 0 (the sentences both count), as the candidates were found
    best = max(candidates, key=lambda k: (paired_difference(rewards[0], rewards[k])[0], -k))
    return min(k for k in candidates if not beats(best, k)), excluded


def paired_difference(first: Mapping[Any, float], then: Mapping[Any, float]) -> tuple[float, float]:
    """``(then − first, its standard error)`` over the sentences both count, read as pairs: the mean difference and
    √(sentences whose reward differs) / sentences (the rewards are 0 or 1)."""
    keys = sorted(first.keys() & then.keys())
    if not keys:
        raise ValueError("no sentence is counted in both rounds: nothing to pair")
    a, b = np.array([first[k] for k in keys]), np.array([then[k] for k in keys])
    return float(np.mean(b - a)), math.sqrt(float(np.sum(a != b))) / len(keys)


def select_rewards(out: Path, rounds: int) -> list[dict[tuple[str, str, int], float]]:
    """Each round's augmented select sentences' rewards (0 … ``rounds``, from their ``readout.json``), keyed by
    flight, kind and sample, those the reward counts (not starting in a loss the speaking aircraft answers for)."""
    rewards = []
    for number in range(rounds + 1):
        readout = json.loads((out / f"round_{number:02d}" / "readout.json").read_text(encoding="utf-8"))
        rewards.append({(r["dataset_id"], r["kind"], r["sample"]): r["reward"]
                        for r in readout["augmented"]["flights"] if not r["starts_in_a_loss"]})
    return rewards


def round_seed(seed: int, round_number: int, stream: int) -> int:
    """A torch generator's seed for one of a round's streams (module docstring)."""
    return int(np.random.SeedSequence([seed, round_number, stream]).generate_state(1)[0])


def completed_rounds(out: Path) -> int:
    """The last round the run at ``out`` finished (``readout.json``, written last); refused when a round's directory
    holds none (cut short: move it aside as ``round_<k>.aborted-<UTC>``) or the rounds are not 0 … k."""
    numbers = sorted(int(path.name[len("round_"):]) for path in out.glob("round_*")
                     if path.name[len("round_"):].isdigit())
    unfinished = [n for n in numbers if not (out / f"round_{n:02d}" / "readout.json").exists()]
    if unfinished:
        raise SystemExit(f"{out}: round(s) {unfinished} did not finish — move each aside as round_<k>.aborted-<UTC> "
                         f"and resume")
    if not numbers:
        raise SystemExit(f"{out} finished no round (round 0 is the first invocation's): move it aside and start again")
    if numbers != list(range(len(numbers))):
        raise SystemExit(f"{out} holds rounds {numbers}, not 0 … k")
    return numbers[-1]


def run_differences(stored: Mapping[str, Any], record: Mapping[str, Any]) -> list[str]:
    """The keys in which a run's stored ``config.json`` and the one this invocation would write differ, apart from
    `RESUMABLE`."""
    now = json.loads(json.dumps(record))
    return sorted(key for key in (set(stored) | set(now)) - set(RESUMABLE)
                  if key not in stored or key not in now or stored[key] != now[key])


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


def write_choice(out: Path, history: Sequence[Mapping[str, Any]], labelled: Mapping[str, Mapping[str, float]],
                 prior_dir: Path, log: Callable[[str], None]) -> None:
    """``choice.json`` over the rounds finished (`guarded_choice`; round 0 kept: the start)."""
    rewards = select_rewards(out, len(history) - 1)
    kept_round, excluded = guarded_choice(history, labelled, rewards)
    write_json_atomic(out / "choice.json", {
        "rule": f"within round 0's guards (real scenes: landed ≥ − {GUARD_LANDED_DROP}, on the observed runway ≥ − "
                f"{GUARD_RUNWAY_DROP}, lost separation ≤ + {GUARD_LOSS_RISE}, median time to land and landing gap ≤ "
                f"{GUARD_ORDER_GROWTH} × recorded; real and augmented: each of {', '.join(GUARD_WORD_COLUMNS)} words per "
                f"flight off the labelled ≤ round 0's + ln {GUARD_WORD_GROWTH}), the augmented-scene reward: beating "
                f"round 0 by ≥ {TIE_STANDARD_ERRORS} paired standard errors, of those the highest and the earliest it "
                f"does not beat by as much; none: round 0",
        "against_round_0": [dict(zip(("difference", "standard_error"), paired_difference(rewards[0], r)))
                            for r in rewards],
        "augmented_reward": [row["augmented"]["reward"] for row in history],
        "real_reward": [row["real"]["reward"] for row in history], "excluded_by_the_guards": excluded,
        "round": kept_round,
        "directory": str(out / f"round_{kept_round:02d}") if kept_round else str(prior_dir)})
    log(f"kept round {kept_round} (the guards excluded {excluded}) → {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the start: augmented (a single-aircraft prior)")
    parser.add_argument("--base", type=Path, required=True, help="the pull's reference: base")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--rounds", type=int, default=8, help="the last round to run (0: round 0 alone)")
    parser.add_argument("--resume", action="store_true",
                        help="continue the run at --out from its last finished round (module docstring)")
    parser.add_argument("--real-per-airport", type=int, default=200, help="real scenes an airport a round")
    parser.add_argument("--augmented-per-airport", type=int, default=200, help="augmented scenes an airport a round")
    parser.add_argument("--samples", type=int, default=8, help="sentences a scene")
    parser.add_argument("--select-per-airport", type=int, default=200)
    parser.add_argument("--select-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--speakers", type=int, default=4, help="speaking processes (not part of the run: the "
                        "sentences do not depend on it)")
    parser.add_argument("--aircraft-steps", type=int, default=SPEAKER_AIRCRAFT_STEPS, help="a loop batch's most "
                        "(`traffic_free_generation.speaking_batches`)")
    parser.add_argument("--traffic-learning-rate", type=float, default=TRAFFIC_LEARNING_RATE)
    parser.add_argument("--passes", type=int, default=1, help="sweeps over a round's sentences (design §6.6 step 6 "
                        "item 14)")
    parser.add_argument("--device", default="cuda", help="the prior's; the executor flies on CPU")
    parser.add_argument("--smoke", action="store_true", help="a dirty tree allowed; the runs are marked smoke")
    for field, default in asdict(RewardConfig()).items():
        parser.add_argument(f"--{field.replace('_', '-')}", type=type(default), default=default)
    args = parser.parse_args(argv)

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    prior_dir, base_dir, instructions, executor_dir, out = map(
        resolved, (args.prior, args.base, args.instructions, args.executor, args.out))
    if args.resume and not (out / "config.json").exists():
        parser.error(f"--resume: {out} holds no run")
    if not args.resume and out.exists():
        parser.error(f"{out} exists; a fine-tuning run is never overwritten (--resume continues it)")
    if args.rounds < 0:
        parser.error("--rounds is the last round to run, 0 or more")
    if args.samples < 2:
        parser.error("a scene's sentences are compared with each other: --samples ≥ 2")
    if args.passes < 1:
        parser.error("--passes is the sweeps over a round's sentences, 1 or more")
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
    model = with_traffic(single, EDGE_FEATURES)         # on the CPU until the speaking processes are forked
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
    record = {
        "schema": TRAFFIC_REWARD_SCHEMA, "written_utc": utc_now(), "git": git, "smoke": args.smoke,
        "prior": {"directory": str(prior_dir), "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt"),
                  "procedure_masks": list(start_masks.names)},
        "base": {"directory": str(base_dir), "checkpoint_sha256": file_sha256(base_dir / "checkpoint.pt")},
        "executor": {"directory": str(executor_dir), "sha256": record["sha256"]}, "instructions": str(instructions),
        "edge_features": list(EDGE_FEATURES), "edge_source_sha256": edge_source_sha256(),
        "optimiser": asdict(config), "traffic_learning_rate": args.traffic_learning_rate, "rounds": args.rounds,
        "real_per_airport": args.real_per_airport, "augmented_per_airport": args.augmented_per_airport,
        "samples": args.samples, "pool_factor": POOL_FACTOR, "aircraft_steps": args.aircraft_steps,
        "speakers": args.speakers,
        "data": {"split": "train", "scene_samples": len(data), "built": data_counts},
        "select": {"per_airport": args.select_per_airport, "samples": args.select_samples, "drawn": select_batch.drawn,
                   "augment_seed": args.seed + SELECT_AUGMENT_OFFSET, "augmented_left_out": select_left_out,
                   "augmented_kinds": dict(Counter(select_augmented.kinds)), "scene_samples": len(select_built),
                   "labelled_words_after_first_per_flight": labelled, "recorded": recorded},
        "seed": args.seed, "tie_standard_errors": TIE_STANDARD_ERRORS, "passes": args.passes,
        "guards": {"landed_drop": GUARD_LANDED_DROP, "runway_drop": GUARD_RUNWAY_DROP, "loss_rise": GUARD_LOSS_RISE,
                   "order_growth": GUARD_ORDER_GROWTH, "word_columns": list(GUARD_WORD_COLUMNS),
                   "word_margin_ln": math.log(GUARD_WORD_GROWTH),
                   "lost_separation_target": recorded["lost_separation"] * GUARD_ORDER_GROWTH},
        "streams": {"pool": "seed + round", "augmentations": [AUGMENT_STREAM], "sampling": [SAMPLING_STREAM],
                    "pass": [PASS_STREAM], "select": SELECT_STREAMS, "loop_batches": "[stream seed, batch number]"},
        "device": str(device),
        "n_look": N_LOOK, "resumed": []}
    if args.resume:
        stored = json.loads((out / "config.json").read_text(encoding="utf-8"))
        differences = run_differences(stored, record)
        if differences:
            raise SystemExit(f"{out} was run with another configuration or code: {differences} differ")
        last = completed_rounds(out)
        if args.rounds < last:
            parser.error(f"{out} has finished round {last}; --rounds {args.rounds} is behind it")
        history = [row for row in json.loads((out / "history.json").read_text(encoding="utf-8"))["rounds"]
                   if row["round"] <= last]
        if [row["round"] for row in history] != list(range(last + 1)):
            raise SystemExit(f"{out}/history.json does not hold rounds 0 … {last}")
        failing = ordering_failures(history[0])
        if failing:
            raise SystemExit(f"the start failed the ordering guards {failing} (round 0): no round could be chosen")
        record = {**stored, "rounds": args.rounds,
                  "resumed": [*stored["resumed"], {"utc": utc_now(), "from_round": last, "to": args.rounds,
                                                   "speakers": args.speakers}]}
    else:
        out.mkdir(parents=True)
        last, history = -1, []
        write_json_atomic(out / "config.json", record)

    def log(line: str) -> None:
        print(f"{line}  [{time.perf_counter() - started:.0f}s]", flush=True)

    log(f"{f'resumed after round {last}; ' if args.resume else ''}train days: {len(data)} scene samples ({data_counts}); select: {len(select_batch.readings)} flights, "
        f"{len(select_augmented.scenes)} augmented ({dict(Counter(select_augmented.kinds))}, {select_left_out} left out); "
        f"recorded lost separation {recorded['lost_separation']:.3f}")

    def free_gpu() -> None:
        """The speaking processes need the GPU next: this process gives back what its cache holds."""
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def read_select(round_number: int) -> dict[str, Any]:
        model.eval()
        readout: dict[str, Any] = {}
        for side, round_ in (("real", select_real), ("augmented", select_augmented)):
            free_gpu()
            spoken, peaks = speakers.speak("select", side, round_, model, args.select_samples,
                                           seed=round_seed(args.seed, 0, SELECT_STREAMS[side]), source="scene")
            readout[side] = side_readout(spoken.rows, round_, args.select_samples, step_s, real=side == "real")
            readout[side]["speaking_gpu_peak_gb"] = peaks
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

    if last > 0:
        resumed = load_prior(out / f"round_{last:02d}", instructions)
        grown_from = {"directory": str(prior_dir), "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt")}
        if resumed.payload["start"] != grown_from:
            raise SystemExit(f"{out}/round_{last:02d} grew from {resumed.payload['start']}, not {grown_from}")
        model = resumed.model

    def train_round(round_number: int) -> tuple[Round, int, dict[str, Any]]:
        """Round ``round_number``'s scenes (module docstring, item 1) — its pool, real and augmented scenes from its own
        seeds, the same in every process: the round, the flights with no qualifying augmentation, the pool drawn."""
        pool = replay.draw(instructions, "train", spec, words,
                           per_airport=round((args.real_per_airport + args.augmented_per_airport) * POOL_FACTOR),
                           seed=args.seed + round_number)
        real = real_starts(pool, args.real_per_airport)
        taken = set(real)
        candidates, left_out = augmented_round(replay.subset(pool, [j for j in range(len(pool.signals))
                                                                    if j not in taken]),
                                               train_airports, params, every_landing,
                                               [args.seed, round_number, AUGMENT_STREAM], windows, step_s)
        kept = first_per_airport(candidates.batch, args.augmented_per_airport, sorted({s.airport for s in pool.signals}))
        augmented = Round(replay.subset(candidates.batch, kept), [candidates.scenes[j] for j in kept],
                          [candidates.moves[j] for j in kept], [candidates.kinds[j] for j in kept],
                          [candidates.directions[j] for j in kept])
        return (join_rounds([real_round(replay.subset(pool, real), train_airports, params, every_landing, step_s),
                             augmented]), left_out, pool.drawn)

    if args.resume and args.rounds == last:                 # the choice alone: nothing to speak
        write_json_atomic(out / "config.json", record)
        write_choice(out, history, labelled, prior_dir, log)
        return 0
    # forked before this process starts the GPU (item 13); the model is the template each process loads weights into; the
    # collector's view of the data frozen first, so that no process's collector writes to (and so copies) the shared heap
    gc.collect()
    gc.freeze()
    speakers = Speakers(args.speakers, lambda n: train_round(n)[0],
                        {"real": select_real, "augmented": select_augmented}, model, args.device,
                        Speaking(words, params, landings, start_masks, args.aircraft_steps))
    model.to(device)
    base.to(device)
    if last < 0:
        directory = out / "round_00"
        directory.mkdir()
        readout = read_select(0)
        history = [history_row(0, readout)]
        write_json_atomic(out / "history.json", {"rounds": history})
        write_json_atomic(directory / "readout.json", readout)                       # last: the round finished
        failing = ordering_failures(history[0])
        if failing:
            raise SystemExit(f"the start fails the ordering guards {failing} against the record "
                             f"({history[0]['real']['ordering']}): no round could be chosen — not training")
        last = 0
    tuner = SceneRewardTuner(model, base, config, device, seed=args.seed,
                             traffic_learning_rate=args.traffic_learning_rate, step_s=step_s)
    if last > 0:
        # on the CPU as the one-run keeps it: `load_state_dict` moves the moments to each parameter's device, not the step
        tuner.load_state(torch.load(out / f"round_{last:02d}" / OPTIMISER_FILE, map_location="cpu", weights_only=True))
    if args.resume:
        write_json_atomic(out / "config.json", record)                         # the state read: this resumption runs
    for round_number in range(last + 1, args.rounds + 1):
        directory = out / f"round_{round_number:02d}"
        directory.mkdir()
        model.eval()
        free_gpu()
        speakers.send("train", round_number, model, args.samples,
                      seed=round_seed(args.seed, round_number, SAMPLING_STREAM), source="train")
        round_, left_out, drawn = train_round(round_number)          # while the processes speak it
        spoken, speaking_peaks = speakers.receive(round_, args.samples)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        earned = np.array([r["reward"] for r in spoken.rows])
        advantages, _, keep = trained_scenes(spoken, args.samples)
        write_sentences(directory / "sentences.npz", round_, spoken, advantages)
        described = {**round_summary(round_, spoken, args.samples), "trained_on": len(keep),
                     "augmented_left_out": left_out, "drawn": drawn}
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
        tuner.restart(np.random.default_rng([args.seed, round_number, PASS_STREAM]))
        passed = {**tuner.one_pass(sentences, advantages[keep], data, allowed, slots=slots, passes=args.passes),
                  "distance_at_start": start_distance, "distance_sentences": measured_on}
        passed["speaking_gpu_peak_gb"] = speaking_peaks
        if device.type == "cuda":
            passed["pass_gpu_peak_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        sweeps = ", ".join(f"{p['kl_mean']:.4f} / {p['clipped_share']:.4f}" for p in passed["sweeps"])
        log(f"round {round_number}: {args.passes} pass(es) over {len(keep)} sentences ({passed['batches']} updates; "
            f"per pass KL / outside the clip {sweeps}), reward term "
            f"{passed['reward_mean']:.4f}, KL to the base at the start {start_distance}, {passed['kl_mean']:.4f} in the "
            f"pass (max {passed['kl_max']:.4f}), data NLL {passed['data_mean']:.4f}, words outside the clip "
            f"{passed['clipped_share']:.4f}, {passed['seconds']:.0f}s; GPU peak: speaking processes "
            + " ".join(f"{p:.2f}" for p in speaking_peaks) + " GB reserved each"
            + (f", the pass {passed['pass_gpu_peak_gb']:.2f} GB allocated" if "pass_gpu_peak_gb" in passed else ""))
        write_traffic_prior(directory, model, prior_dir, start, spec.sha256, git=git, smoke=args.smoke,
                            fine_tuning={"schema": TRAFFIC_REWARD_SCHEMA, "from": str(prior_dir), "base": str(base_dir),
                                         "round": round_number, "optimiser": asdict(config),
                                         "traffic_learning_rate": args.traffic_learning_rate,
                                         "samples": args.samples, "passes": args.passes})
        torch.save(tuner.state(), directory / OPTIMISER_FILE)
        readout = read_select(round_number)
        history.append(history_row(round_number, readout, train_pass=passed,
                                   sentences={k: v for k, v in described.items() if k != "drawn"}))
        write_json_atomic(out / "history.json", {"rounds": history})
        write_json_atomic(directory / "readout.json", readout)                       # last: the round finished
    write_choice(out, history, labelled, prior_dir, log)
    speakers.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
