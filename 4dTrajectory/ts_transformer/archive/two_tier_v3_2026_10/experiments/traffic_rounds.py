"""The round protocol of the multi-aircraft post-training (design §6.2, §6.6 step 6 items 8, 11, 13, 14; step 9's
"9.4 的代码") — M4 in windows (R37, `traffic_window_reward`) trains by it; not a runner. Written for M4's first runner
(R32, now archived) and lifted here unchanged.

**The round kept** (`guarded_choice`, design §6.6 step 6 item 8): among the rounds within round 0's guards — on the real
side, landed (the executor's: separation aside) at most `GUARD_LANDED_DROP` below, landed on the observed runway at most
`GUARD_RUNWAY_DROP` below, lost separation at most `GUARD_LOSS_RISE` above, the median time to land and the median gap
to the landing before on the runway both at most `GUARD_ORDER_GROWTH` × the same flights' recorded ones
(`ordering_failures`); on the real and the augmented side, the words a sentence says after its first step no farther
from the labelled ones than round 0 plus ln `GUARD_WORD_GROWTH` (stage 2's) — the augmented side's reward, its ties read
against paired standard errors (design §6.2, §9 item 27): the select readouts speak the same draws every round, so two
rounds compare sentence by sentence and their difference's standard error is √(sentences whose reward flipped) /
sentences (`paired_difference`, over the sentences counted in both). A candidate beats round 0 by at least
`TIE_STANDARD_ERRORS` of them; of the candidates, the earliest the highest does not beat by as much; none: round 0.

**Round by round** (design §6.6 step 6 item 11): every stream a round draws from is its own ([``seed``, round,
`AUGMENT_STREAM` / `SAMPLING_STREAM` / `PASS_STREAM`], `round_seed`), and each round leaves beside its weights the
optimiser's state (`OPTIMISER_FILE`), so ``--resume`` continues a run from its last finished round (`completed_rounds`)
as one invocation would have — refused when the run was made with another configuration, code or device
(`run_differences`, apart from `RESUMABLE`).

**In several processes** (design §6.6 step 6 item 13): a round's sentences are spoken by `Speakers`, forked once the data
are built and before the parent starts the GPU; each process rebuilds a training round from its number and speaks the
loop batches dealt to it, each batch from its own stream (`batch_seed`), so the sentences are the same whatever the
number of processes. How a round is spoken is the runner's (`Speaking`).

A round's model is written as a traffic prior `prior_train.load_prior` reads (`write_traffic_prior`), and the select
readout reads its traffic attention's output (`traffic_readout`).
"""

from __future__ import annotations

import copy
import ctypes
import json
import math
import multiprocessing
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch

from ts_transformer.experiments.prior_augmented_reward import (
    GUARD_LANDED_DROP, GUARD_WORD_COLUMNS, GUARD_WORD_GROWTH, word_distance,
)
from ts_transformer.experiments.prior_landing_reward import GUARD_RUNWAY_DROP
from ts_transformer.experiments.prior_train import TRAFFIC_CHECKPOINT_SCHEMA, LoadedPrior
from ts_transformer.experiments.traffic_prior_train import scene_evaluate
from ts_transformer.experiments.traffic_scene_data import Built, edge_source_sha256
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.masks import write_masks
from ts_transformer.prior.model import Prior
from ts_transformer.prior.train import TrainConfig


#: The traffic attention's learning rate (design §8): pretraining's; the rest of the model trains at the second stage's.
TRAFFIC_LEARNING_RATE = 3e-4
#: The ordering guard (design §4.3): the median time to land and the median gap to the landing before on the runway, at
#: most this × the same flights' recorded ones (the words guard's margin).
GUARD_ORDER_GROWTH = 1.2
#: The separation guard (design §6.6 step 6 item 8): the real side's lost-separation share at most this above round 0's
#: (the landed guard's margin; about 1.4 binomial standard deviations over 2,000 select sentences at 11.5 %). The
#: recorded share × `GUARD_ORDER_GROWTH` is the target line, reported (round 0 is four times the record).
GUARD_LOSS_RISE = 0.01
#: A round must beat round 0 on the augmented side by this many paired standard errors to be chosen, and the highest
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
#: Beside a round's weights: what its pass leaves for the next (`traffic_window_tuner.WindowRewardTuner.state`).
OPTIMISER_FILE = "optimiser.pt"
#: What a resumed run's ``config.json`` may differ in: when it was written, how far it was asked to go and its resumptions.
RESUMABLE = ("written_utc", "rounds", "resumed", "speakers")
#: A speaking process's loop batch, its most aircraft-steps (`traffic_window_generation.window_batches`): chosen for four
#: processes' pasts and blocks beside the parent's models within the 8 GB GPU (a third of M3's one-process batch; each
#: process's peak is logged, design §6.6 step 6 item 13).
SPEAKER_AIRCRAFT_STEPS = 100_000


def batch_seed(seed: int, number: int) -> int:
    """A torch generator's seed for the loop batch ``number`` of a stream seeded ``seed``: each batch its own, so what it
    says depends neither on the batches before it nor on the process that speaks it (design §6.6 step 6 item 13)."""
    return int(np.random.SeedSequence([seed, number]).generate_state(1)[0])


class Speaking(Protocol):
    """How a runner's round is spoken (`traffic_window_reward.WindowSpeaking`): its plan of loop batches, one batch
    spoken from its own stream (`batch_seed`), the round's fingerprint (what a speaking process must have rebuilt the
    same as the parent) and the batches assembled in the round's order."""

    def plan(self, round_: Any, samples: int) -> list[list[int]]: ...

    def speak(self, model: Prior, round_: Any, chunk: Sequence[int], number: int, samples: int, *, seed: int,
              source: str) -> Any: ...

    def fingerprint(self, round_: Any) -> Any: ...

    def assemble(self, round_: Any, samples: int, plan: Sequence[Sequence[int]], parts: Mapping[int, Any]) -> Any: ...


#: `prctl` option: the signal a process gets when its parent dies (linux/prctl.h).
PR_SET_PDEATHSIG = 1


def _speaker(pipe: Any, parent_ends: Sequence[Any], parent_pid: int, rounds: Callable[[int], Any],
             select: Mapping[str, Any], model: Prior, device: str, speaking: Speaking) -> None:
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
        built: dict[int, Any] = {}
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
            del parts, round_                                   # the next task builds its own
    except BaseException:
        traceback.print_exc()
        pipe.send(("failed", traceback.format_exc()))


class Speakers:
    """``count`` processes speaking a round's loop batches in parallel (design §6.6 step 6 item 13), forked from the
    parent once its data are built and BEFORE it starts the GPU: each holds the parent's data (shared, not copied — the
    caller freezes the collector's view of it first), rebuilds a training round from its number with the parent's own
    ``rounds`` (a round holds its airports' whole data: rebuilt, not sent), reads the select rounds (``select``, by side)
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
        parts: dict[int, Any] = {}
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


def ordering_failures(row: Mapping[str, Any]) -> list[str]:
    """The ordering guards (design §4.3) a round's real-side readout fails: its median time to land and its median gap to
    the landing before on the runway at most `GUARD_ORDER_GROWTH` × the same flights' recorded ones — unreadable (no
    landing, or none with a landing before it) fails. Against the record, not round 0: the start is read against them
    before training (`main` refuses a start that fails them — no round could then be chosen)."""
    order = row["real"]["ordering"]
    return [name for name, ratio in (("time to land", order["time_ratio"]), ("landing gap", order["gap_ratio"]))
            if ratio is None or ratio > GUARD_ORDER_GROWTH]


def guarded_choice(history: Sequence[Mapping[str, Any]], labelled: Mapping[str, Mapping[str, float]],
                   rewards: Sequence[Mapping[Any, float]]) -> tuple[int, dict[int, list[str]]]:
    """``(the round kept, each excluded round's failed guards)`` (module docstring); ``rewards`` each round's augmented
    select sentences', keyed alike in every round (the runner's ``select_rewards``)."""
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
                        git: Mapping[str, Any], smoke: bool, fine_tuning: Mapping[str, Any], writer: str) -> None:
    """``model`` as a prior run `prior_train.load_prior` reads (`prior_train.TRAFFIC_CHECKPOINT_SCHEMA`): its checkpoint
    — the start's payload with this state, the edge and traffic features, the edge code's hash and ``start`` (the
    single-aircraft prior it grew from: its directory and checkpoint sha256) — its ``config.json`` (the start's, with
    these) and the procedure's masks it speaks under (the start's; ``start`` as `load_prior` opened ``start_dir``; ``writer``
    the runner's module)."""
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
    write_masks(directory, start.procedure_masks, writer=writer, git=dict(git))
