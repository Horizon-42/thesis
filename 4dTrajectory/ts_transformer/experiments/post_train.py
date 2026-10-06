"""C10: the post-training as one campaign (post-training §2, §8 C10; D29, D30, D36, D37, D76, D94, D100, D103, D107,
D113–D116) — rounds of branch training in windows of recorded traffic, from the base of stage B.

THE START (D29): the base (`prior.checkpoint.open_prior`) with a traffic attention at each layer whose output is zero
(`post.traffic_attention`, D116); the base itself, frozen, for the pull (§2 item 5).

EACH ROUND r:

1. **The draw** (`draw_round`): windows of the train days, the count of each kind given (D100: real, A and D in equal
   counts; B with a count of its own): the real windows in a permutation by a generator of the seed and r, each kind
   built from them (A, D: D103's shifts; B: the moved start, C9; the census's proposals, `post_windows`) and kept unless
   it opens inside a loss of separation (D113; window B on its moved record). A shortfall is recorded, never silent.
2. **The batches** (`batches`): each commands each flight once (a loop holds each flight once), so a real window and
   its A, B or D are in different batches.
3. **The speaking** (`speak_round`, `experiments.post_branches.branch_round`): every batch's two passes of D94 with
   the model at the start of the round; its informative groups written to ``round_<r>/groups_<k>.pt`` (a round's groups
   are not held in memory, the reviewer's estimate of C6) and its first pass's ends counted. With ``--speak-workers N``
   (N ≥ 2) the batches are spoken by N worker processes (`Speakers`), with the same groups and records (the pass after
   them agrees to float rounding): the speaking is simulation on the CPU (the executor, the masks, the scene), one batch
   at a time a process.
4. **One training pass** (`train_pass`, §2 item 5): the written groups, each file's in an order shuffled by the
   round's numbers (D130), `update_groups` at a time, each paired with `data_sentences` single-aircraft sentences of
   the train days in the base's selection (D36, D76; `prior.source.ArtefactSource`), through the loss of C7
   (`post.loss.one_pass`: the surrogate and the pull in eval mode, the data term with dropout, D107; every counted row
   alike, D115), the traffic modules at their own learning rate (`parameter_groups`).
5. **The selection readout** (`selection_readout`): a fixed set of real windows of the select days (drawn once with the
   seed, D113 applied; the same numbers every round), the first pass only, by airport: the rewards, the outcomes (a loss
   of separation among them), the rows the speed-word mask acted (D101), the steps reading a faulty point and the losses
   near one (D114). No criterion is applied (D7); the validation days are never read here (`post_validation` reads
   them once, for the chosen round, with the same readout).
6. **The round's record** ``round_<r>/round.json`` (the draw, its windows, the speaking, the bytes of its groups, the
   pass, the readout, the commit as information), then **its checkpoint** ``round_<r>/checkpoint.pt`` (§4 item 3): the
   model, the optimizer and the identity (`identity`); then its groups files are deleted (only its own pass reads
   them, and a round is done once its checkpoint is written).

THE CHECKS (§4 item 2), each in this process before its first use: the closed loop's (`require_conforming_closed_loop`,
D69, with the labeller's and the executor's) and the edge features' reference of the census (D104, `checked_edges`).

ONE CAMPAIGN: from a clean checkout (``--smoke``: a tree with changes too; no result); each round's commit recorded as
information and never compared (prior D108). A formal campaign needs its intent in `docs/experiments/intents.json`
under the campaign's name (its directory's) before it starts (L27). RESUMABLE: a round is done when its checkpoint is
there (the last file it writes); a rerun with the same inputs and settings continues from the last checkpoint, a round
left half done is moved aside as ``round_<r>.aborted-<UTC>`` (outline E8) and run again; other inputs are refused. Every
random number of a round comes from the seed and the round (the draw, the speaking's, the data term's sentences and
its dropout, `pass_seed`), so a resumed campaign is the campaign run through.

    python run_ts.py post_train --prior <the base's directory> --instructions <A34's artefact> --executor <its spec> \\
        --windows <the census> --out 4dTrajectory/outputs/POOLED/post/<campaign id> --rounds … (every count given)
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.autopilot.start import start_moved
from ts_transformer.experiments.post_branches import branch_round
from ts_transformer.experiments.post_window_loop import WindowLoop, checked_edges, moved_commanded, start_move_of
from ts_transformer.experiments.post_windows import (    # D103's shifts and C9's ranges: one definition, the census's
    A_APART_S, A_LANDING_SHIFT_S, B_HEIGHT_M, B_SPEED_SCALE, B_TURN_DEG, D_SHIFT_S,
)
from ts_transformer.instructions.artefact import closed_loop_sentences, load_spec, signals_flights
from ts_transformer.instructions.faults import faulty_flights
from ts_transformer.instructions.words import Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.post.branches import CONTINUATIONS, Group, samples
from ts_transformer.post.fault_census import fault_rows
from ts_transformer.post.loss import Samples, one_pass, stacked
from ts_transformer.post.runways import airport_separation
from ts_transformer.post.scene import (
    INSERTED, LEADER_MOVED, MOVED_START, REAL, Window, airport_scenes, inserted_window, leader_moved_window,
    moved_start_window, real_windows,
)
from ts_transformer.post.traffic import opens_inside_loss
from ts_transformer.post.traffic_attention import TrafficConfig, add_traffic_attention, parameter_groups
from ts_transformer.prior.batch import RowTensors, collate
from ts_transformer.prior.checkpoint import OpenedPrior, open_prior
from ts_transformer.prior.model import Prior
from ts_transformer.prior.procedure import PROCEDURE_MASKS, airport_finals
from ts_transformer.prior.source import ArtefactSource
from ts_transformer.prior.speaker import MOST_GO_AROUNDS
from ts_transformer.repo_layout import REPO_ROOT, git_state

CAMPAIGN_SCHEMA = "ts-post-train-v1"
POST_CHECKPOINT_SCHEMA = "ts-post-checkpoint-v1"
KINDS = (REAL, INSERTED, LEADER_MOVED, MOVED_START)
#: The intents of the published experiments (L27; post-training §8 C10: written before the launch).
INTENTS = REPO_ROOT / "4dTrajectory" / "ts_transformer" / "docs" / "experiments" / "intents.json"


@dataclass(frozen=True)
class Settings:
    """What a campaign is: everything a resume must find the same."""

    rounds: int
    per_kind: dict[str, int]
    batch_windows: int
    continuations: int
    seed: int
    prior_lr: float
    traffic_lr: float
    weight_decay: float
    update_groups: int
    data_sentences: int
    select_per_airport: int
    traffic_hidden: int
    traffic_heads: int

    def __post_init__(self) -> None:
        if set(self.per_kind) != set(KINDS) or min(self.per_kind.values()) < 0 or not any(self.per_kind.values()):
            raise ValueError(f"a count of each kind of window {KINDS}, none negative, not all zero")
        if min(self.rounds, self.batch_windows, self.continuations, self.update_groups, self.data_sentences,
               self.select_per_airport) <= 0:
            raise ValueError("rounds, batch windows, continuations, groups an update, data sentences and select windows "
                             "are positive")


@dataclass
class Context:
    """What every round reads: the artefact and the executor spec, the base, the airports' rules and each split's
    windows with what flying them reads (``splits``: "train" and "select", each a mapping of ``windows``, ``sentences``,
    ``flights``, ``signals`` by key and ``faults``), and the data term's sentences (``data``)."""

    instructions: Path
    executor: Path
    edges_reference: Path
    device: torch.device
    words: Words
    interval_s: float
    base: Prior
    base_identity: dict[str, Any]
    geometries: Mapping[str, Any]
    rosters: Mapping[str, Any]
    finals: Mapping[str, Any]
    splits: Mapping[str, Mapping[str, Any]]
    data: Sequence[Any]

    @property
    def variant(self) -> str:
        return self.base.config.variant

    @property
    def separations(self) -> dict[str, Any]:
        return {code: airport_separation(g) for code, g in self.geometries.items()}

    def start_loop(self, split: str, windows: Sequence[Window]):
        """The ``start_loop`` of `branch_round`: the flights of ``windows`` started with their windows' moves (C9)."""
        sentences = self.splits[split]["sentences"]
        moves = {w.signal_index: start_move_of(w) for w in windows}

        def started(flights: Sequence[int]):
            return start_moved(self.instructions, split, self.interval_s, {i: sentences[i] for i in flights},
                               self.executor, {i: moves[i] for i in flights}, most_go_arounds=MOST_GO_AROUNDS,
                               device=self.device)

        return started


def split_data(instructions: Path, split: str, words: Words, interval_s: float, geometries: Mapping[str, Any]
               ) -> dict[str, Any]:
    """A split's real windows (`post.scene`), closed-loop sentences, flights, signals by key and each airport's fault rows
    by flight key (vocabulary D111, D114)."""
    scenes, signals = airport_scenes(instructions, split, words.spec, interval_s, geometries)
    faults: dict[str, dict[str, frozenset[int]]] = {code: {} for code in geometries}
    for index, found in faulty_flights(instructions, split).items():
        faults[signals[index].airport][signals[index].dataset_id] = fault_rows(found)
    return {"windows": real_windows(instructions, split, words.spec, interval_s, scenes, signals),
            "sentences": closed_loop_sentences(instructions, split, interval_s, words.spec),
            "flights": dict(enumerate(signals_flights(instructions, split))),
            "signals": {s.dataset_id: s for s in signals}, "faults": faults}


def open_context(prior_dir: Path, instructions: Path, executor: Path, edges_reference: Path, device: torch.device,
                 procedure_root: Path, *, formal: bool, data: bool = True, splits: Sequence[str] = ("train", "select")
                 ) -> Context:
    """The context of a campaign: the base opened (`open_prior`: its artefact, Δ, selection and procedure masks) on
    ``device`` in eval mode (the pull of §2 item 5 reads it beside the model, `start_model`'s copy of it), the
    finals of its procedure data, the ``splits`` (the campaign's: train and select; none for `post_validation`, which
    adds val after its claim), and the data term's sentences (none for a caller that trains nothing, ``data`` False:
    the export of the Training view, the validation readout). ``formal`` (a formal campaign and its validation readout,
    D132): the base must be stage B's formal base (`require_formal_base`)."""
    prior = open_prior(prior_dir, instructions, procedure_root=procedure_root)
    if formal:
        require_formal_base(prior)
    words = Words(load_spec(instructions))
    source = ArtefactSource(instructions, prior.interval_s, prior.checkpoint.model.config.variant, prior.landings,
                            prior.selection)
    sentences = [s for code in sorted(prior.geometries) for s in source.sentences("train", code)] if data else []
    if data and not sentences:
        raise ValueError(f"no train sentence in the base's selection {prior.selection!r}: no data term (D36, D76)")
    return Context(instructions, executor, edges_reference, device, words, prior.interval_s,
                   prior.checkpoint.model.to(device).eval(), prior.checkpoint.identity, prior.geometries, prior.landings,
                   {code: airport_finals(g, root=procedure_root) for code, g in prior.geometries.items()},
                   {split: split_data(instructions, split, words, prior.interval_s, prior.geometries)
                    for split in splits},
                   sentences)


def require_formal_base(prior: OpenedPrior) -> None:
    """D132: the base of a formal campaign and of its validation readout is stage B's formal base — refused by name when
    its run held an airport out (a fold) or trained on a sample of the sentences (a smoke prior), the fields stage B's
    own validation readout checks (``config.json``'s run, the checkpoint's run)."""
    held_out, sample = prior.config["run"]["held_out"], prior.checkpoint.run["sample"]
    if held_out is not None:
        raise SystemExit(f"{prior.directory} is a fold (held out {held_out}), not stage B's formal base (D132)")
    if sample is not None:
        raise SystemExit(f"{prior.directory} is a smoke prior ({sample['per_airport_and_split']} sentences of each "
                         f"airport and split), not stage B's formal base (D132)")


# ---- the draw
def candidate(kind: str, window: Window, rng: np.random.Generator, separation: Any) -> Window | None:
    """A window of ``kind`` built from the real ``window`` (None where it admits none: A without a flight far enough
    from it, D without an aircraft ahead)."""
    if kind == REAL:
        return window
    if kind == INSERTED:
        return inserted_window(window, rng, landing_shift_s=A_LANDING_SHIFT_S, apart_s=A_APART_S)
    if kind == LEADER_MOVED:
        return leader_moved_window(window, separation, rng, shift_s=D_SHIFT_S)
    return moved_start_window(window, rng, turn_deg=B_TURN_DEG, height_m=B_HEIGHT_M, speed_scale=B_SPEED_SCALE)


def draw_round(context: Context, per_kind: Mapping[str, int], rng: np.random.Generator
               ) -> tuple[list[Window], dict[str, Any]]:
    """The round's windows (module docstring, step 1), kind after kind in `KINDS` order, and what the draw counted: the
    windows drawn of each kind, those left out for opening inside a loss (D113), and a shortfall."""
    train = context.splits["train"]
    separations, step_s = context.separations, context.words.spec.step_s
    out: dict[str, list[Window]] = {kind: [] for kind in KINDS if per_kind[kind] > 0}
    left_out: Counter = Counter()
    for i in rng.permutation(len(train["windows"])):
        real = train["windows"][int(i)]
        code = real.scene.geometry.code
        for kind, drawn in out.items():
            if len(drawn) == per_kind[kind]:
                continue
            window = candidate(kind, real, rng, separations[code])
            if window is None:
                continue
            judged = window if kind != MOVED_START else replace(
                window, commanded=moved_commanded(window, train["signals"][window.commanded.key], step_s))
            if opens_inside_loss(judged, separations[code], context.finals[code], step_s):
                left_out[kind] += 1
            else:
                drawn.append(window)
        if all(len(drawn) == per_kind[kind] for kind, drawn in out.items()):
            break
    return [w for drawn in out.values() for w in drawn], {
        "drawn": {kind: len(drawn) for kind, drawn in out.items()}, "left_out_inside_loss": dict(left_out),
        "shortfall": {kind: per_kind[kind] - len(drawn) for kind, drawn in out.items() if len(drawn) < per_kind[kind]}}


def batches(windows: Sequence[Window], size: int) -> list[list[int]]:
    """The windows' places in batches of at most ``size`` that command each flight once (the first batch with room and
    without the flight), each in the order of its flights' places in the signals (`branch_round`'s rule)."""
    out: list[list[int]] = []
    for place, window in enumerate(windows):
        home = next((b for b in out if len(b) < size and all(windows[p].signal_index != window.signal_index for p in b)),
                    None)
        if home is None:
            out.append([place])
        else:
            home.append(place)
    return [sorted(b, key=lambda p: windows[p].signal_index) for b in out]


# ---- the speaking
def speak_batch(model: Prior, context: Context, windows: Sequence[Window], places: Sequence[int], settings: Settings,
                round_: int, directory: Path, k: int) -> dict[str, Any]:
    """Batch ``k`` of a round (the windows at ``places``): its two passes (`branch_round`), its informative groups written
    to ``groups_<k>.pt``, and its record (the counts of its groups and the ends of its first pass)."""
    train = context.splits["train"]
    batch = [windows[p] for p in places]
    found = branch_round(model, context.start_loop("train", batch), batch, places, train["sentences"], train["flights"],
                         context.geometries, context.rosters, context.finals, context.words,
                         interval_s=context.interval_s, variant=context.variant,
                         edges_reference=context.edges_reference, faults=train["faults"], device=context.device,
                         seed=settings.seed, round_=round_, split="train", continuations=settings.continuations)
    informative = [g for g in found.groups if g.informative]
    torch.save(informative, directory / f"groups_{k}.pt")
    return {"groups": len(found.groups), "informative_groups": len(informative),
            "spoken_again": len(found.spoken_again), "differed": list(found.differed),
            "reward_sum": sum(r.reward for r in found.first), "faulty_steps": sum(r.faulty_steps for r in found.first),
            "losses_reading_fault": sum(r.loss_reads_fault for r in found.first),
            "outcomes": dict(Counter(r.outcome for r in found.first))}


def speak_round(model: Prior, context: Context, windows: Sequence[Window], settings: Settings, round_: int,
                directory: Path, speakers: Speakers | None = None) -> dict[str, Any]:
    """Step 3 of a round (module docstring): every batch's two passes (`speak_batch`), here or by ``speakers`` (the
    same groups and records: a batch reads only its windows' own random numbers), its informative groups written, its
    first pass's ends counted; the batches' records summed in batch order."""
    places = batches(windows, settings.batch_windows)
    parts = (speakers.speak(model, windows, places, round_, directory) if speakers is not None else
             [speak_batch(model, context, windows, p, settings, round_, directory, k) for k, p in enumerate(places)])
    record: dict[str, Any] = {"batches": 0, "groups": 0, "informative_groups": 0, "spoken_again": 0, "differed": [],
                              "reward_sum": 0.0, "faulty_steps": 0, "losses_reading_fault": 0}
    outcomes: Counter = Counter()
    for part in parts:
        record["batches"] += 1
        for key in ("groups", "informative_groups", "spoken_again", "differed", "reward_sum", "faulty_steps",
                    "losses_reading_fault"):
            record[key] += part[key]
        outcomes.update(part["outcomes"])
    return {**record, "windows": len(windows), "outcomes": dict(outcomes)}


#: What a speaker process holds (`Speakers`): set before the fork, so each worker inherits the campaign's context.
_SPEAKER: dict[str, Any] = {}


class Speakers:
    """Worker processes that speak a round's batches in parallel (step 3), each a whole batch (`speak_batch`) with the
    round's model, its groups file written by it, one task a batch. They are forked from the campaign's process once its
    context is open and BEFORE that process uses the GPU, so they share the context's memory (its splits are read
    once) and each starts the GPU itself; every worker is started here, at once. A worker runs torch on one thread
    (`_initialise_speaker`: a fork after the parent's CPU threads ran would hang otherwise), draws the round's windows
    itself (`draw_round`, the same random numbers) and refuses a batch whose windows are not the ones the campaign
    sent; the model of the round reaches it as a file in the round's directory, deleted after the speaking. A worker
    that dies fails the round (`BrokenProcessPool`), never hangs it. The speaking is the speaking here (a batch reads
    only its windows' own random numbers): the same groups and records; the pass after it agrees to float rounding (the
    GPU's kernels depend on what a process ran before, as on any resume). The number of workers is recorded as
    information. NOT CHECKED here: the memory of N workers at the formal size (watched; an open item)."""

    def __init__(self, context: Context, settings: Settings, workers: int, device: torch.device) -> None:
        if workers < 2:
            raise ValueError(f"{workers} worker(s): speakers are 2 or more processes (one speaks without them)")
        if context.device.type != "cpu" or torch.cuda.is_initialized():
            raise ValueError("speakers are forked before the campaign's process uses the GPU: open the context on the "
                             "CPU, fork them, then move it")
        _SPEAKER.clear()
        _SPEAKER.update(context=context, settings=settings, device=device)
        self.pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork"),
                                        initializer=_initialise_speaker)
        list(self.pool.map(_started, range(workers)))         # every worker forked now, before the GPU is used
        self.workers = workers

    def speak(self, model: Prior, windows: Sequence[Window], places: Sequence[Sequence[int]], round_: int,
              directory: Path) -> list[dict[str, Any]]:
        """Every batch's record, in batch order; on a failure the batches not started are cancelled."""
        state = directory / "speaking_model.pt"
        torch.save(model.state_dict(), state)
        futures = [self.pool.submit(_speak, round_, str(state), str(directory), k, list(p),
                                    [window_record(windows[i]) for i in p]) for k, p in enumerate(places)]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            wait(futures)                                  # the running batches end before their model file goes
            state.unlink()

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)


def _initialise_speaker() -> None:
    """A worker's torch on one thread, its compiler too (module docstring of `Speakers`)."""
    import torch._inductor.config as inductor

    torch.set_num_threads(1)
    inductor.compile_threads = 1


def _started(_: int) -> int:
    return os.getpid()


def _speak(round_: int, state: str, directory: str, k: int, places: list[int], records: list[dict[str, Any]]
           ) -> dict[str, Any]:
    """A worker's batch (`Speakers`): the round's model and windows made once a round, then `speak_batch`."""
    held = _SPEAKER
    if held["context"].device != held["device"]:                         # the worker's first batch: its own GPU
        context = held["context"]
        held["context"] = replace(context, device=held["device"], base=context.base.to(held["device"]).eval())
    context, settings = held["context"], held["settings"]
    if held.get("round") != round_:
        model, _ = start_model(context, settings)
        model.load_state_dict(torch.load(state, map_location=context.device, weights_only=True))
        windows, _ = draw_round(context, settings.per_kind, np.random.default_rng([settings.seed, round_]))
        held.update(round=round_, model=model.eval(), windows=windows)
    windows = held["windows"]
    if [window_record(windows[i]) for i in places] != records:
        raise ValueError(f"round {round_}, batch {k}: the worker drew other windows than the campaign")
    try:
        return speak_batch(held["model"], context, windows, places, settings, round_, Path(directory), k)
    finally:
        if context.device.type == "cuda":
            torch.cuda.empty_cache()


# ---- the training pass
def update_pairs(directory: Path, data: Sequence[Any], settings: Settings, rng: np.random.Generator,
                 device: torch.device) -> Iterator[tuple[Samples, RowTensors]]:
    """The updates of a round's pass (module docstring, step 4): its written groups, file by file in the order they were
    spoken, each file's groups in an order shuffled by ``rng`` (the round's numbers; D130: an update mixes branch points
    and windows), `Settings.update_groups` at a time (a file's last update may hold fewer), each paired with
    `Settings.data_sentences` sentences of ``data`` drawn by ``rng``. A file is read when its first update is drawn."""
    for path in sorted(directory.glob("groups_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
        loaded: list[Group] = torch.load(path, weights_only=False)
        groups = [loaded[int(i)] for i in rng.permutation(len(loaded))]
        for k in range(0, len(groups), settings.update_groups):
            chosen = rng.choice(len(data), size=min(settings.data_sentences, len(data)), replace=False)
            yield samples(groups[k:k + settings.update_groups], device), collate([data[int(i)] for i in chosen], device)


def pass_seed(seed: int, round_: int) -> int:
    """The torch seed of round ``round_``'s pass (the data term's dropout): a round is the same from its last
    checkpoint, so a resumed campaign is the campaign run through."""
    return int(np.random.default_rng([seed, round_, 2]).integers(1 << 62))


def train_pass(model: Prior, context: Context, optimizer: torch.optim.Optimizer, directory: Path, settings: Settings,
               rng: np.random.Generator) -> dict[str, Any]:
    """Step 4 of a round: one pass (`post.loss.one_pass`: the model at its start is the one that spoke the groups) over
    `update_pairs`; its means, or no update where the round has no informative group."""
    parts = one_pass(model, context.base, optimizer, update_pairs(directory, context.data, settings, rng, context.device))
    return {"updates": len(parts), **(stacked(parts) if parts else {})}


# ---- the selection readout
def readout_numbers(seed: int, place: int) -> np.random.Generator:
    """The random numbers of the selection readout's window ``place``: the same every round (the readout compares the
    rounds' models on the same windows and numbers)."""
    return np.random.default_rng([seed, 1 << 30, place])


def selection_windows(context: Context, settings: Settings, split: str = "select") -> list[Window]:
    """The selection readout's windows (module docstring, step 5): at most ``select_per_airport`` real windows of each
    airport of the select days (``split``; the val days for `post_validation`, drawn alike), drawn once with the seed
    from those that do not open inside a loss (D113)."""
    rng = np.random.default_rng([settings.seed, 1 << 31])
    out = []
    for mine in readout_pool(context, split).values():
        out += [mine[int(i)] for i in sorted(rng.choice(len(mine), min(settings.select_per_airport, len(mine)),
                                                        replace=False))]
    return out


def readout_pool(context: Context, split: str) -> dict[str, list[Window]]:
    """Each airport's real windows of ``split`` that do not open inside a loss (D113), in airport order: what the
    readout's windows are drawn from."""
    windows, separations = context.splits[split]["windows"], context.separations
    return {code: [w for w in windows if w.scene.geometry.code == code
                   and not opens_inside_loss(w, separations[code], context.finals[code], context.words.spec.step_s)]
            for code in sorted(context.geometries)}


def selection_readout(model: Prior, context: Context, windows: Sequence[Window], settings: Settings,
                      split: str = "select") -> dict[str, Any]:
    """Step 5 of a round: the select windows' first pass (no branch), its ends by airport (module docstring); the
    same readout of ``split``'s windows (`post_validation`: the val days)."""
    select = context.splits[split]
    counted: dict[str, dict[str, Any]] = {}
    for places in batches(windows, settings.batch_windows):
        batch = [windows[p] for p in places]
        loop, order, observed = context.start_loop(split, batch)([w.signal_index for w in batch])
        ends = WindowLoop(model, loop, order, batch, select["sentences"], select["flights"], context.geometries,
                          context.rosters, context.finals, context.words, interval_s=context.interval_s,
                          variant=context.variant, edges_reference=context.edges_reference, faults=select["faults"],
                          observed=observed, device=context.device).run(
            [readout_numbers(settings.seed, p) for p in places])
        for window, end in zip(batch, ends):
            c = counted.setdefault(window.scene.geometry.code, {"windows": 0, "reward_sum": 0.0, "outcomes": Counter(),
                                                                "speed_mask_rows": 0, "faulty_steps": 0,
                                                                "losses_reading_fault": 0})
            c["windows"] += 1
            c["reward_sum"] += end.reward
            c["outcomes"][end.outcome] += 1
            c["speed_mask_rows"] += end.speed_mask_rows
            c["faulty_steps"] += end.faulty_steps
            c["losses_reading_fault"] += end.loss_reads_fault
    return {code: {**c, "outcomes": dict(c["outcomes"]), "reward_mean": c["reward_sum"] / c["windows"]}
            for code, c in sorted(counted.items())}


# ---- the campaign
def window_record(window: Window) -> dict[str, Any]:
    """A window by its commanded flight, its kind, its start time, the flights it moved and its start's move (§4 item 3)."""
    return {"flight": window.commanded.key, "kind": window.kind, "row0_s": window.row0_s,
            "moved": [list(m) for m in window.moved], "start_move": asdict(window.start_move)}


def identity(context: Context, settings: Settings, out: Path, rounds: int) -> dict[str, Any]:
    """A post-trained checkpoint's identity after ``rounds`` rounds (§4 item 3): its format, the base's identity, the
    procedure masks it speaks under, the traffic module's shape, the seed, and each round's windows (its
    ``round.json``)."""
    return {"schema": POST_CHECKPOINT_SCHEMA, "base": context.base_identity, "procedure_masks": PROCEDURE_MASKS,
            "traffic": TrafficConfig(settings.traffic_hidden, settings.traffic_heads).to_dict(), "seed": settings.seed,
            "rounds": [json.loads((out / f"round_{r}" / "round.json").read_text(encoding="utf-8"))["windows"]
                       for r in range(rounds)]}


def done_rounds(out: Path) -> int:
    """The rounds done: round 0, 1, … each with its checkpoint (the last file a round writes)."""
    r = 0
    while (out / f"round_{r}" / "checkpoint.pt").exists():
        r += 1
    return r


def open_campaign(out: Path, inputs: dict[str, Any], git: dict[str, Any], checks: Any) -> dict[str, Any]:
    """``campaign.json``: a new campaign's, or a resume's (refused unless its inputs and settings are the same); a round
    left half done (its directory without its checkpoint) is moved aside as ``round_<r>.aborted-<UTC>``."""
    path = out / "campaign.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["schema"] != CAMPAIGN_SCHEMA or record["inputs"] != inputs:
            raise SystemExit(f"{out}: a campaign of other inputs or settings; a resume takes the same")
        left = out / f"round_{done_rounds(out)}"
        if left.exists():
            aborted = left.with_name(f"{left.name}.aborted-{utc_now().replace(':', '')}")
            left.rename(aborted)
            record["aborted"].append(aborted.name)
        record["resumed"].append({"utc": utc_now(), "git": git, "checks": checks})
        write_json_atomic(path, record)
        return record
    if out.exists():
        raise SystemExit(f"{out} exists and holds no campaign")
    out.mkdir(parents=True)
    record = {"schema": CAMPAIGN_SCHEMA, "started_utc": utc_now(), "inputs": inputs, "git": git, "checks": checks,
              "aborted": [], "resumed": []}
    write_json_atomic(path, record)
    return record


def start_model(context: Context, settings: Settings) -> tuple[Prior, torch.optim.Optimizer]:
    """The model at the start (D29: the base with zero-output traffic modules, their other weights drawn from the
    campaign's seed, so a campaign is the same from its start) and its optimizer."""
    model = copy.deepcopy(context.base)
    torch.manual_seed(settings.seed)
    add_traffic_attention(model, TrafficConfig(settings.traffic_hidden, settings.traffic_heads))
    model.to(context.device).eval()
    return model, torch.optim.AdamW(parameter_groups(model, settings.prior_lr, settings.traffic_lr),
                                    weight_decay=settings.weight_decay)


def settings_of(record: Mapping[str, Any]) -> Settings:
    """A campaign's settings as its ``campaign.json`` records them."""
    return Settings(**record["inputs"]["settings"])


def round_model(context: Context, settings: Settings, out: Path, round_: int | None) -> Prior:
    """The model of campaign ``out`` after round ``round_`` (its checkpoint), or at its start (None: the base with
    zero-output traffic modules, D29), in eval mode; a checkpoint is refused unless its identity is this campaign's on
    this base, under today's procedure masks (§4 item 3)."""
    model, _ = start_model(context, settings)
    if round_ is None:
        return model
    state = torch.load(out / f"round_{round_}" / "checkpoint.pt", weights_only=False, map_location=context.device)
    held = state["identity"]
    expected = {"schema": POST_CHECKPOINT_SCHEMA, "base": context.base_identity, "procedure_masks": PROCEDURE_MASKS,
                "traffic": TrafficConfig(settings.traffic_hidden, settings.traffic_heads).to_dict(),
                "seed": settings.seed}
    if {k: held[k] for k in expected} != expected or len(held["rounds"]) != round_ + 1:
        raise ValueError(f"{out}/round_{round_}: a checkpoint of another base, masks, traffic shape, seed or round")
    model.load_state_dict(state["model"])
    return model.eval()


def run_campaign(out: Path, settings: Settings, context: Context, speakers: Speakers | None = None) -> None:
    """The rounds not yet done (module docstring), each closed by its checkpoint; the speaking by ``speakers`` when
    given (`Speakers`)."""
    model, optimizer = start_model(context, settings)
    first = done_rounds(out)
    if first:
        state = torch.load(out / f"round_{first - 1}" / "checkpoint.pt", weights_only=False, map_location=context.device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
    select = selection_windows(context, settings)
    for round_ in range(first, settings.rounds):
        directory = out / f"round_{round_}"
        directory.mkdir()
        windows, drawn = draw_round(context, settings.per_kind, np.random.default_rng([settings.seed, round_]))
        if speakers is not None and context.device.type == "cuda":
            torch.cuda.empty_cache()     # the speakers share the GPU: the memory this process cached in the last pass
        spoken = speak_round(model, context, windows, settings, round_, directory, speakers)
        torch.manual_seed(pass_seed(settings.seed, round_))                    # the data term's dropout
        passed = train_pass(model, context, optimizer, directory, settings,
                            np.random.default_rng([settings.seed, round_, 1]))
        readout = selection_readout(model, context, select, settings)
        written = sorted(directory.glob("groups_*.pt"))
        write_json_atomic(directory / "round.json", {
            "round": round_, "git": git_state(), "finished_utc": utc_now(), "draw": drawn,
            "speak_workers": speakers.workers if speakers is not None else 1,         # information (`Speakers`)
            "windows": [window_record(w) for w in windows], "speaking": spoken,
            "groups_bytes": sum(p.stat().st_size for p in written), "pass": passed, "selection_readout": readout})
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "identity": identity(context, settings, out, round_ + 1)}, directory / "checkpoint.pt")
        for path in written:                                # read by this round's pass only; their size is recorded
            path.unlink()
        print(json.dumps({"round": round_, "pass": passed,
                          "readout": {code: r["reward_mean"] for code, r in readout.items()}}), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the base: a prior_train run's directory")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the directory of the artefact's executor spec")
    parser.add_argument("--windows", type=Path, required=True,
                        help="the census (post_windows) whose conformance/edges.npz is the edge reference (D104)")
    parser.add_argument("--procedure-root", type=Path, default=DEFAULT_PROCEDURE_ROOT,
                        help="the CIFP procedures of the finals (prior §7 item 6)")
    parser.add_argument("--out", type=Path, required=True, help="the campaign's directory (new, or a resume)")
    parser.add_argument("--rounds", type=int, required=True)
    for kind in KINDS:
        parser.add_argument(f"--windows-{kind.lower()}", type=int, required=True,
                            help=f"the windows of kind {kind} in a round (D100)")
    parser.add_argument("--batch-windows", type=int, required=True, help="the windows flown in one batch")
    parser.add_argument("--continuations", type=int, default=CONTINUATIONS, help="D94: K")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prior-lr", type=float, required=True)
    parser.add_argument("--traffic-lr", type=float, required=True, help="§2 item 5: the traffic modules' own")
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--update-groups", type=int, required=True, help="the branch groups of an update")
    parser.add_argument("--data-sentences", type=int, required=True, help="the data term's sentences of an update")
    parser.add_argument("--select-per-airport", type=int, required=True)
    parser.add_argument("--traffic-hidden", type=int, required=True)
    parser.add_argument("--traffic-heads", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speak-workers", type=int, default=1,
                        help="processes that speak a round's batches in parallel (Speakers; the same results)")
    parser.add_argument("--smoke", action="store_true", help="SMOKE: a tree with changes too; no result")
    args = parser.parse_args(argv)
    if args.speak_workers < 1:
        parser.error("--speak-workers is at least 1")
    prior_dir, instructions, executor, census, procedure_root, out = (p if p.is_absolute() else REPO_ROOT / p for p in (
        args.prior, args.instructions, args.executor, args.windows, args.procedure_root, args.out))
    settings = Settings(args.rounds, {kind: getattr(args, f"windows_{kind.lower()}") for kind in KINDS},
                        args.batch_windows, args.continuations, args.seed, args.prior_lr, args.traffic_lr,
                        args.weight_decay, args.update_groups, args.data_sentences, args.select_per_airport,
                        args.traffic_hidden, args.traffic_heads)
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("a campaign that is not a smoke runs on a clean checkout")
    if not args.smoke and out.name not in json.loads(INTENTS.read_text(encoding="utf-8"))["campaigns"]:
        parser.error(f"campaign {out.name}: no intent in {INTENTS} (written before the launch, L27)")
    _, opened, _ = require_conforming_closed_loop(instructions, executor)          # D69: the checks run here (D73)
    edges_reference = census / "conformance" / "edges.npz"
    checked_edges(edges_reference)                                                 # D104
    device = torch.device(args.device)
    # with speakers, the context is opened on the CPU and they are forked before this process uses the GPU
    context = open_context(prior_dir, instructions, executor, edges_reference,
                           torch.device("cpu") if args.speak_workers > 1 else device, procedure_root,
                           formal=not args.smoke)                                  # D132
    speakers = Speakers(context, settings, args.speak_workers, device) if args.speak_workers > 1 else None
    context = replace(context, device=device, base=context.base.to(device).eval())
    inputs = {"prior": str(prior_dir), "instructions": str(instructions), "executor": str(executor),
              "windows": str(census), "procedure_root": str(procedure_root), "settings": asdict(settings),
              "smoke": args.smoke}
    try:
        open_campaign(out, inputs, git, opened["checks"])
        run_campaign(out, settings, context, speakers)
    finally:
        if speakers is not None:
            speakers.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
