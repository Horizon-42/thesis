"""C10: the post-training as one campaign (post-training §2, §8 C10; D29, D30, D36, D37, D76, D94, D100, D103, D107,
D113–D116) — rounds of branch training in windows of recorded traffic, from the base of stage B.

THE START (D29; or a round of another campaign, below): the base (`prior.checkpoint.open_prior`) with a traffic attention at each layer whose output is zero
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
   at a time a process. Before the first round, one worker speaks a batch at the formal size and N workers are refused
   by name where its memory times N does not fit (`require_workers_fit`, O15). Every window is started through its
   split's `Start`, opened once before the workers fork (C13, vocabulary A44).
4. **One training pass** (`train_pass`, §2 item 5): the written groups, each file's in an order shuffled by the
   round's numbers (D130), `update_groups` at a time, each paired with `data_sentences` single-aircraft sentences of
   the train days in the base's selection (D36, D76; `prior.source.ArtefactSource`), through the loss of C7
   (`post.loss.one_pass`: the surrogate and the pull in eval mode, the data term with dropout, D107; every counted row
   alike, D115), the traffic modules at their own learning rate (`parameter_groups`).
5. **The selection readout** (`selection_readout`): a fixed set of real windows of the select days (drawn once with the
   seed, D113 applied; the same numbers every round), the first pass only, by airport: the rewards, the outcomes (a loss
   of separation among them), the rows the speed-word mask acted (D101), the steps reading a faulty point and the losses
   near one (D114); read by the speaking workers when there are (C13; one process is the reference mode, the same
   ends). No criterion is applied (D7); the validation days are never read here (`post_validation` reads
   them once, for the chosen round, with the same readout).
6. **The round's record** ``round_<r>/round.json`` (the draw, its windows, the speaking, the bytes of its groups, the
   pass, the readout, the commit as information), then **its checkpoint** ``round_<r>/checkpoint.pt`` (§4 item 3): the
   model, the optimizer and the identity (`identity`); then its groups files are deleted (only its own pass reads
   them, and a round is done once its checkpoint is written).

THE CHECKS (§4 item 2), each in this process before its first use: the closed loop's (`require_conforming_closed_loop`,
D69, with the labeller's and the executor's) and the edge features' reference of the census (D104, `checked_edges`).

ONE CAMPAIGN: from a clean checkout (``--smoke``: a tree with changes too; no result); each round's commit recorded as
information and never compared (prior D108). Its intent is written by hand (L27, read by the publisher), not checked
here (the user, 2026-10-07). RESUMABLE: a round is done when its checkpoint is
there (the last file it writes); a rerun with the same inputs and settings continues from the last checkpoint, a round
left half done is moved aside as ``round_<r>.aborted-<UTC>`` (outline E8) and run again; other inputs are refused. A
rerun may raise the rounds and change nothing else (D157): the campaign goes on to the new count, unless its val read
is claimed (P47: the round would then be chosen after the val days were read). The recorded paths
are compared as this checkout reads them (`inputs_here`: a campaign recorded in a worktree resumes from another
checkout). Every random number of a round comes from the seed and the round (the draw, the speaking's, the data term's
sentences and its dropout, `pass_seed`), so a resumed campaign is the campaign run through, a raised one the campaign
of the larger count from its start. A START FROM A ROUND (``--start-campaign``, ``--start-round``;
`campaign_start`, D162): instead of the base with zero-output traffic modules (D29), the weights of a round of another
campaign on the same base, its checkpoint's bytes recorded in the settings, with a seed other than its campaign's; a
new optimizer, and the pull term toward the base either way. A formal campaign does not start from a smoke one.

    python run_ts.py post_train --prior <the base's directory> --instructions <A34's artefact> --executor <its spec> \\
        --windows <the census> --out 4dTrajectory/outputs/POOLED/post/<campaign id> --rounds … (every count given)
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import os
import pickle
import subprocess
import tempfile
import threading
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.autopilot.start import NO_MOVE, Start
from ts_transformer.experiments.post_branches import branch_round
from ts_transformer.experiments.post_window_loop import (
    WindowLoop, WindowResult, checked_edges, moved_commanded, start_move_of,
)
from ts_transformer.experiments.post_windows import (    # D103's shifts and C9's ranges: one definition, the census's
    A_APART_S, A_LANDING_SHIFT_S, B_HEIGHT_M, B_SPEED_SCALE, B_TURN_DEG, D_SHIFT_S,
)
from ts_transformer.instructions.artefact import closed_loop_sentences, load_spec, signals_flights
from ts_transformer.instructions.faults import faulty_flights
from ts_transformer.instructions.words import Words
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.post.branches import CONTINUATIONS, Group, samples
from ts_transformer.post.fault_census import fault_rows
from ts_transformer.post.loss import Samples, one_pass, stacked
from ts_transformer.post.runways import airport_separation
from ts_transformer.post.scene import (
    INSERTED, LEADER_MOVED, MOVED_START, REAL, Window, inserted_window, leader_moved_window, moved_start_window,
    real_windows, scenes_of,
)
from ts_transformer.post.traffic import opens_inside_loss
from ts_transformer.post.traffic_attention import TrafficConfig, add_traffic_attention, parameter_groups
from ts_transformer.prior.batch import RowTensors, collate
from ts_transformer.prior.checkpoint import OpenedPrior, open_prior, validation_claim
from ts_transformer.prior.model import Prior
from ts_transformer.prior.procedure import PROCEDURE_MASKS, airport_finals
from ts_transformer.prior.source import ArtefactSource
from ts_transformer.prior.speaker import MOST_GO_AROUNDS
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

CAMPAIGN_SCHEMA = "ts-post-train-v1"
POST_CHECKPOINT_SCHEMA = "ts-post-checkpoint-v1"
KINDS = (REAL, INSERTED, LEADER_MOVED, MOVED_START)
#: The paths among a campaign's inputs (``campaign.json``), read as this checkout reads them (`inputs_here`, D157).
INPUT_PATHS = ("prior", "instructions", "executor", "windows", "procedure_root")
#: The reader's name in the claim of a campaign's val read (prior D119; `post_validation`'s, named here so that a raise
#: of the rounds is refused once the claim is made, P47).
CLAIM_READER = "post_validation"


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
    #: The model the campaign starts from (`campaign_start`, the user, 2026-10-07): None, the base with zero-output traffic
    #: modules (D29); or a round of another campaign on the same base — ``campaign`` (its directory, repo-relative),
    #: ``round`` and ``checkpoint_sha256`` (the bytes it started from).
    start: dict[str, Any] | None

    def __post_init__(self) -> None:
        if set(self.per_kind) != set(KINDS) or min(self.per_kind.values()) < 0 or not any(self.per_kind.values()):
            raise ValueError(f"a count of each kind of window {KINDS}, none negative, not all zero")
        if min(self.rounds, self.batch_windows, self.continuations, self.update_groups, self.data_sentences,
               self.select_per_airport) <= 0:
            raise ValueError("rounds, batch windows, continuations, groups an update, data sentences and select windows "
                             "are positive")
        if self.start is not None and set(self.start) != {"campaign", "round", "checkpoint_sha256"}:
            raise ValueError(f"a start is its campaign, round and checkpoint_sha256, not {sorted(self.start)}")


@dataclass
class Context:
    """What every round reads: the artefact and the executor spec, the base, the airports' rules and each split's
    windows with what flying them reads (``splits``: "train" and "select", each a mapping of ``windows``, ``sentences``,
    ``flights``, ``signals`` by key, ``faults`` and ``start``, its opened `Start`), and the data term's sentences
    (``data``)."""

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
    #: a formal campaign's context (`open_context`'s ``formal``, D132): a start from a smoke campaign is refused (D162)
    formal: bool = False

    @property
    def variant(self) -> str:
        return self.base.config.variant

    @property
    def separations(self) -> dict[str, Any]:
        return {code: airport_separation(g) for code, g in self.geometries.items()}

    def start_loop(self, split: str, windows: Sequence[Window]):
        """The ``start_loop`` of `branch_round`: the flights of ``windows`` started with their windows' moves (C9), through
        the split's opened `Start` (C13, vocabulary A44: what `start_moved` gives, read once); a window's other commanded
        aircraft (multi-aircraft control D150) start unmoved at their join steps."""
        data = self.splits[split]
        sentences = data["sentences"]
        moves = {w.signal_index: start_move_of(w) for w in windows}
        moves.update({i: NO_MOVE for w in windows for i in w.signal_indices[1:]})
        joins = {i: int(step) for w in windows for i, step in zip(w.signal_indices, w.join_steps())}
        several = any(w.joined for w in windows)

        def started(flights: Sequence[int]):
            ticks = {"join_ticks": {i: joins[i] for i in flights}} if several else {}
            return data["start"].moved({i: sentences[i] for i in flights}, {i: moves[i] for i in flights},
                                       most_go_arounds=MOST_GO_AROUNDS, device=self.device, **ticks)

        return started


def split_data(instructions: Path, split: str, words: Words, interval_s: float, geometries: Mapping[str, Any],
               executor: Path) -> dict[str, Any]:
    """A split's real windows (`post.scene`), closed-loop sentences, flights, signals by key, each airport's fault rows
    by flight key (vocabulary D111, D114) and its start opened by the executor spec in ``executor`` (`Start`, C13)."""
    start = Start(instructions, split, interval_s, executor)
    signals = start.signals                                 # read once, by the start (C13)
    scenes = scenes_of(signals, instructions, split, words.spec, interval_s, geometries)
    faults: dict[str, dict[str, frozenset[int]]] = {code: {} for code in geometries}
    for index, found in faulty_flights(instructions, split).items():
        faults[signals[index].airport][signals[index].dataset_id] = fault_rows(found)
    return {"windows": real_windows(instructions, split, words.spec, interval_s, scenes, signals),
            "sentences": closed_loop_sentences(instructions, split, interval_s, words.spec),
            "flights": dict(enumerate(signals_flights(instructions, split))),
            "signals": {s.dataset_id: s for s in signals}, "faults": faults,
            "start": start}


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
                   {split: split_data(instructions, split, words, prior.interval_s, prior.geometries, executor)
                    for split in splits},
                   sentences, formal)


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


# ---- the stage
@dataclass(frozen=True)
class Stage:
    """A stage's parts of a campaign's round (post-training §9 item 11; multi-aircraft control D149): the model at the
    start and its optimizer, the identity of a round's checkpoint, the draw of a round's windows (and what it counted),
    the speaking of a round (its two passes, its groups written; its record), the selection readout's windows and its
    readout, the record of a window in the round's record (which the identity holds: a later stage's names its
    commanded aircraft and their shifts, multi-aircraft control §7 row 3), what a speaking worker runs of a batch, and
    the width of its token part. `run_campaign` is the round's skeleton; `STAGE_C` is stage C's campaign."""

    #: the model of the stage's shape and a new optimizer (what a resume and a speaking worker load a state into), and
    #: the model a new campaign starts from with a new optimizer (stage C's: `campaign_start`, the base with zero-output
    #: traffic modules or a round of another campaign, D162)
    start_model: Callable[[Context, Settings], tuple[Prior, torch.optim.Optimizer]]
    start: Callable[[Context, Settings], tuple[Prior, torch.optim.Optimizer]]
    identity: Callable[[Context, Settings, Path, int], dict[str, Any]]
    draw: Callable[[Context, Settings, int], tuple[list[Window], dict[str, Any]]]
    #: the speaking and the readout of a round, each given the stage itself (``stage=``: its batch speaker and reader
    #: drive the one-process mode, and its workers are refused unless they run it)
    speak: Callable[..., dict[str, Any]]
    selection: Callable[[Context, Settings, str], list[Window]]
    readout: Callable[..., dict[str, Any]]
    record: Callable[[Window], dict[str, Any]]
    #: a batch of a round spoken (its two passes, its groups written: ``(model, context, windows, places, settings,
    #: round, directory, k)``, `speak_batch`'s) and a batch of a readout read (``(model, context, windows, places,
    #: settings, split, draw)``, `read_batch`'s, draw 0 by default; C15): what a speaking worker runs (`Speakers`)
    speak_batch: Callable[[Prior, Context, Sequence[Window], Sequence[int], Settings, int, Path, int], dict[str, Any]]
    read_batch: Callable[[Prior, Context, Sequence[Window], Sequence[int], Settings, str, int], list[WindowResult]]
    #: the width of the stage's token part beside the edge features (post-training §9 item 7; stage C's: none, 0), which
    #: the pass's samples read
    part_width: int
    #: a round's windows in batches (`batches` of the stage's batch size: stage C's ``batch_windows``, stage D's rows,
    #: multi-aircraft control D146)
    batches: Callable[[Sequence[Window], Settings], list[list[int]]]


def _stage_c_draw(context: Context, settings: Settings, round_: int) -> tuple[list[Window], dict[str, Any]]:
    return draw_round(context, settings.per_kind, np.random.default_rng([settings.seed, round_]))


def _stage_c_readout(model: Prior, context: Context, windows: Sequence[Window], settings: Settings,
                     speakers: Speakers | None, *, stage: Stage) -> dict[str, Any]:
    return selection_readout(model, context, windows, settings, speakers=speakers, stage=stage)


#: Stage C's campaign (module docstring): its parts of a round, each read from this module when it is called (as the
#: campaign read them before the stage existed: a caller or a test that replaces one replaces it here too).
STAGE_C = Stage(start_model=lambda context, settings: start_model(context, settings),
                start=lambda context, settings: campaign_start(context, settings),
                identity=lambda context, settings, out, rounds: identity(context, settings, out, rounds),
                draw=lambda context, settings, round_: _stage_c_draw(context, settings, round_),
                speak=lambda *args, stage: speak_round(*args, stage=stage),
                selection=lambda context, settings, split: selection_windows(context, settings, split),
                readout=lambda *args, stage: _stage_c_readout(*args, stage=stage),
                record=lambda window: window_record(window),
                speak_batch=lambda *args: speak_batch(*args), read_batch=lambda *args: read_batch(*args), part_width=0,
                batches=lambda windows, settings: batches(windows, settings.batch_windows))



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
    """The windows' places in batches of at most ``size`` rows (a window's rows are its commanded aircraft: stage C's
    window one, so ``size`` windows), each of one span L, that command each flight once (the first batch of the
    window's span with room for its rows and without any of its commanded flights; multi-aircraft control D146: a
    batch's rows and ticks alike), each in the order of its anchors' places in the signals (`branch_round`'s rule). A
    window of more rows than ``size`` is a batch alone."""
    out: list[list[int]] = []
    rows: list[int] = []
    for place, window in enumerate(windows):
        mine = set(window.signal_indices)
        home = next((k for k, b in enumerate(out)
                     if windows[b[0]].span_s == window.span_s and rows[k] + len(mine) <= size
                     and all(not mine & set(windows[p].signal_indices) for p in b)), None)
        if home is None:
            out.append([place])
            rows.append(len(mine))
        else:
            out[home].append(place)
            rows[home] += len(mine)
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
                directory: Path, speakers: Speakers | None = None, *, stage: Stage = STAGE_C) -> dict[str, Any]:
    """Step 3 of a round (module docstring): every batch's two passes (``stage``'s `Stage.speak_batch`, stage C's
    `speak_batch`), here or by ``speakers`` (the same groups and records: a batch reads only its windows' own random
    numbers; refused unless they run ``stage``), its informative groups written, its first pass's ends counted; the
    batches' records summed in batch order."""
    _require_stage(speakers, stage)
    places = stage.batches(windows, settings)
    parts = (speakers.speak(model, windows, places, round_, directory) if speakers is not None else
             [stage.speak_batch(model, context, windows, p, settings, round_, directory, k)
              for k, p in enumerate(places)])
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


def _require_stage(speakers: Speakers | None, stage: Stage) -> None:
    """Speakers run one stage's parts (`Speakers`): refused by name for another stage's speaking or reading."""
    if speakers is not None and speakers.stage is not stage:
        raise ValueError("the speaking workers run another stage's parts than the one given (Speakers(..., stage=))")


#: How often a worker's own host memory is sampled while it speaks the measured batch (O15, `_PeakSampler`), s.
PEAK_SAMPLE_S = 0.2
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
    information. They read the selection readout's batches too (`read`, C13; one process is the reference mode). Before
    a campaign, one worker speaks a batch at the formal size and its memory is measured (`measure`, O15)."""

    def __init__(self, context: Context, settings: Settings, workers: int, device: torch.device,
                 stage: Stage = STAGE_C) -> None:
        if workers < 2:
            raise ValueError(f"{workers} worker(s): speakers are 2 or more processes (one speaks without them)")
        if context.device.type != "cpu" or torch.cuda.is_initialized():
            raise ValueError("speakers are forked before the campaign's process uses the GPU: open the context on the "
                             "CPU, fork them, then move it")
        _SPEAKER.clear()
        _SPEAKER.update(context=context, settings=settings, device=device, stage=stage)
        self.stage = stage
        self.pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork"),
                                        initializer=_initialise_speaker)
        list(self.pool.map(_started, range(workers)))         # every worker forked now, before the GPU is used
        self.workers = workers
        self.readings = 0

    def speak(self, model: Prior, windows: Sequence[Window], places: Sequence[Sequence[int]], round_: int,
              directory: Path) -> list[dict[str, Any]]:
        """Every batch's record, in batch order; on a failure the batches not started are cancelled."""
        state = directory / "speaking_model.pt"
        torch.save(model.state_dict(), state)
        return self._results(state, [self.pool.submit(_speak, round_, str(state), str(directory), k, list(p),
                                                      [self.stage.record(windows[i]) for i in p])
                                     for k, p in enumerate(places)])

    def read(self, model: Prior, windows: Sequence[Window], places: Sequence[Sequence[int]], split: str,
             draw: int = 0) -> list[list[WindowResult]]:
        """The selection readout's batches of ``split`` (``windows``, its `selection_windows`, at ``places``): each
        batch's ends (`read_batch` of ``draw``), in batch order (C13)."""
        self.readings += 1
        with tempfile.TemporaryDirectory(prefix="post_reading_") as scratch:
            state = Path(scratch) / "reading_model.pt"
            torch.save(model.state_dict(), state)
            return self._results(state, [self.pool.submit(_read, self.readings, str(state), split, list(p),
                                                          [self.stage.record(windows[i]) for i in p], draw)
                                         for p in places])

    def measure(self, round_: int, directory: Path, model: Prior) -> dict[str, Any]:
        """O15: one worker's memory at the formal size — the measured batches of round ``round_`` (`measured_batches`: stage
        C's first batch) spoken by one worker with ``model``, the model the round starts from (`campaign_model`; it
        reaches the worker as a file in ``directory``, deleted after), its groups written to ``directory`` — its own peak
        host memory (sampled, not shared with the campaign's process), and on the GPU its peak (the CUDA context and the
        most the allocator held), each with what it still holds after the batch, and what a worker holds besides in a
        round (`_measure`)."""
        state = directory / "measuring_model.pt"
        torch.save(model.state_dict(), state)
        try:
            return self.pool.submit(_measure, round_, str(state), str(directory)).result()
        finally:
            state.unlink()

    @staticmethod
    def _results(state: Path, futures: list[Any]) -> list[Any]:
        """Every future's result, in order; on a failure the tasks not started are cancelled; the model file ``state``
        goes once every running task has ended."""
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            wait(futures)
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


def _worker_context() -> tuple[Context, Settings, Stage]:
    """A worker's context, on its own device from its first task on (`Speakers`), the campaign's settings and its
    stage."""
    held = _SPEAKER
    if held["context"].device != held["device"]:                         # the worker's first task: its own GPU
        context = held["context"]
        held["context"] = replace(context, device=held["device"], base=context.base.to(held["device"]).eval())
    return held["context"], held["settings"], held["stage"]


def _speak(round_: int, state: str, directory: str, k: int, places: list[int], records: list[dict[str, Any]]
           ) -> dict[str, Any]:
    """A worker's batch (`Speakers`): the round's model and windows made once a round (the stage's), then the stage's
    `Stage.speak_batch`."""
    held = _SPEAKER
    context, settings, stage = _worker_context()
    if held.get("round") != round_:
        context.splits["train"]["start"].release()        # a round's flights kept, not the campaign's (C13)
        model, _ = stage.start_model(context, settings)
        model.load_state_dict(torch.load(state, map_location=context.device, weights_only=True))
        windows, _ = stage.draw(context, settings, round_)
        held.update(round=round_, model=model.eval(), windows=windows)
    windows = held["windows"]
    if [stage.record(windows[i]) for i in places] != records:
        raise ValueError(f"round {round_}, batch {k}: the worker drew other windows than the campaign")
    try:
        return stage.speak_batch(held["model"], context, windows, places, settings, round_, Path(directory), k)
    finally:
        if context.device.type == "cuda":
            torch.cuda.empty_cache()


def _read(reading: int, state: str, split: str, places: list[int], records: list[dict[str, Any]], draw: int
          ) -> list[WindowResult]:
    """A worker's batch of the selection readout (`Speakers.read`): the reading's model made once a reading, the split's
    readout windows once (the stage's `Stage.selection`, the same draw), then the stage's `Stage.read_batch`."""
    held = _SPEAKER
    context, settings, stage = _worker_context()
    selection = held.setdefault("selection", {})
    if split not in selection:
        selection[split] = stage.selection(context, settings, split)
    windows = selection[split]
    if [stage.record(windows[i]) for i in places] != records:
        raise ValueError(f"reading {reading}: the worker drew other readout windows than the campaign")
    if held.get("reading") != reading:
        model, _ = stage.start_model(context, settings)
        model.load_state_dict(torch.load(state, map_location=context.device, weights_only=True))
        held.update(reading=reading, reader=model.eval())
    try:
        return stage.read_batch(held["reader"], context, windows, places, settings, split, draw)
    finally:
        if context.device.type == "cuda":
            torch.cuda.empty_cache()


def _measure(round_: int, state: str, directory: str) -> dict[str, Any]:
    """A worker's measure (`Speakers.measure`): the measured batches of round ``round_`` spoken one after another
    (`measured_batches`), its own host memory sampled meanwhile (`_PeakSampler`: the most of any), then its memory after
    them (`_memory_after_batch`: the GPU's peak the most of any) and what a worker holds besides
    in a round (``held``): the selection readout's model (as large as the round's: `_read`) and the round's kept series
    (`Start.release` at each round) — at most the round's flights, at the mean pickled size of the series this batch
    kept (an estimate: a series in memory is not its pickle). Not counted: the selection readout's windows and the
    series its start keeps (the select split's, never released: bounded by `Settings.select_per_airport` × the
    airports). The swap share is read against its value here, once every worker is forked (`SwapPss` divides a shared
    page among the processes holding it, so it falls with each fork); a later move of the shared swap moves it too (an
    estimate)."""
    swapped_here = _own_memory()["swapped"]
    context, settings, stage = _worker_context()
    start = context.splits["train"]["start"]
    model, _ = stage.start_model(context, settings)
    model.load_state_dict(torch.load(state, map_location=context.device, weights_only=True))
    model.eval()
    windows, _ = stage.draw(context, settings, round_)
    found = stage.batches(windows, settings)
    measured = measured_batches(windows, found)
    with _PeakSampler(swapped_here) as sampled:
        for k in measured:
            stage.speak_batch(model, context, windows, found[k], settings, round_, Path(directory), k)
    kept = [len(pickle.dumps(series)) for series in start.series.values()]
    flights = len({i for w in windows for i in w.signal_indices})      # every commanded flight's series
    held = {"reader_model": sum(t.numel() * t.element_size() for t in (*model.parameters(), *model.buffers())),
            "series": flights * sum(kept) // len(kept), "round_flights": flights}
    return {"round": round_, "windows": sum(len(found[k]) for k in measured),
            "batches": [{"batch": k, "span_s": windows[found[k][0]].span_s, "windows": len(found[k]),
                         "rows": sum(len(windows[p].signal_indices) for p in found[k])} for k in measured],
            **_memory_after_batch(context.device, sampled.peak), "held": held}


def measured_batches(windows: Sequence[Window], found: Sequence[Sequence[int]]) -> list[int]:
    """The batches (their places in ``found``) that O15's measure speaks: the first batch of each span L and the batch
    of the most rows (multi-aircraft control D146: a batch's ticks grow with its span, and a window of more rows than the
    batch size is a batch alone); stage C's windows, of one span, give its first batch (the batches are full in order,
    so no later one has more rows)."""
    rows = [sum(len(windows[p].signal_indices) for p in batch) for batch in found]
    firsts = {}
    for k, batch in enumerate(found):
        firsts.setdefault(windows[batch[0]].span_s, k)
    most = max(range(len(found)), key=lambda k: (rows[k], -k))
    return sorted({*firsts.values(), most})


def _own_memory() -> dict[str, int]:
    """This process's own host memory now, bytes (``/proc/self/smaps_rollup``): its private resident pages, and its
    share of what is swapped out (``SwapPss``: a page it shares with the process it was forked from counts in part)."""
    fields = {}
    for line in Path("/proc/self/smaps_rollup").read_text(encoding="utf-8").splitlines()[1:]:
        name, value = line.split(":")
        fields[name] = int(value.split()[0]) * 1024
    return {"resident": fields["Private_Clean"] + fields["Private_Dirty"], "swapped": fields["SwapPss"]}


class _PeakSampler:
    """The most of this process's own host memory while a block runs (`_own_memory`: its private resident pages and its
    share of the swap less ``swapped_at_start``), sampled every `PEAK_SAMPLE_S` by a thread and once at the end; a
    failure of the thread is raised at the end. The kernel's peak (``ru_maxrss``) is not used: a forked process
    inherits its parent's, and it counts the pages it shares."""

    def __init__(self, swapped_at_start: int) -> None:
        self.swapped_at_start = swapped_at_start
        self.peak = 0
        self._failed: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        own = _own_memory()
        self.peak = max(self.peak, own["resident"] + own["swapped"] - self.swapped_at_start)

    def _run(self) -> None:
        try:
            while not self._stop.wait(PEAK_SAMPLE_S):
                self._sample()
        except BaseException as error:                  # raised in the measuring thread at the end
            self._failed = error

    def __enter__(self) -> _PeakSampler:
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join()
        if self._failed is not None:
            raise RuntimeError("the memory sampler failed") from self._failed
        self._sample()


def _memory_after_batch(device: torch.device, host_peak: int) -> dict[str, Any]:
    """This process's memory, bytes (`Speakers.measure`). Host: its own peak (``host_peak``, `_PeakSampler`) and its own
    resident memory now (`_own_memory`; what is swapped out gives back no RAM). GPU (a CUDA device): its use now by
    ``nvidia-smi`` (the context and what the allocator holds), less what the allocator holds now, plus the most it
    held — the peak — and its use after the cache is released."""
    now = _own_memory()["resident"]
    out: dict[str, Any] = {"host": {"peak": max(host_peak, now), "now": now}, "gpu": None}   # held after: held at the peak
    if device.type == "cuda":
        used = _gpu_used()
        peak_gpu = used - torch.cuda.memory_reserved(device) + torch.cuda.max_memory_reserved(device)
        torch.cuda.empty_cache()
        out["gpu"] = {"peak": peak_gpu, "now": _gpu_used()}
    return out


def _gpu_used() -> int:
    """This process's GPU memory by ``nvidia-smi``, bytes."""
    listed = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True).stdout
    mine = [int(used) for pid, used in (line.split(",") for line in listed.splitlines()) if int(pid) == os.getpid()]
    if len(mine) != 1:
        raise RuntimeError(f"nvidia-smi lists process {os.getpid()} {len(mine)} times")
    return mine[0] * (1 << 20)


def pass_memory_of(context: Context, settings: Settings, directory: Path, stage: Stage = STAGE_C
                   ) -> dict[str, Any] | None:
    """O15: this process's GPU memory in one update of the pass (`post_profile.pass_memory`: forward and backward on
    the `Settings.update_groups` longest groups written in ``directory`` with the data term's sentences, and on one
    group — the longest, the widest and the densest alone: an update runs in pieces of a group each, post-training §2
    item 5, so its peak is its largest piece's): its peak (the CUDA context and the most the allocator held in any
    update measured) and what it holds after; None on the CPU.
    Refused by name when the update did not form (fewer groups written than an update takes) or ran out of memory.
    APPROXIMATION, stated: the groups of one batch (a round's longest may need more: `post_profile` measures that
    bound); the optimizer's state (twice the parameters) is not in it. ``stage``: its model at the start and its token
    part (the stage's speakers measured the groups)."""
    if context.device.type != "cuda":
        return None
    from ts_transformer.experiments.post_profile import pass_memory          # post_profile imports this module

    device = context.device
    model, optimizer = stage.start_model(context, settings)
    measured = pass_memory(model, context, directory, settings, sorted({1, settings.update_groups}),
                           part_width=stage.part_width)
    update = measured["updates"][str(settings.update_groups)]
    if update == "fewer groups held":
        raise SystemExit(f"the pass's memory is not measured (O15): the measured batch wrote "
                         f"{measured['groups_written']} groups, an update takes {settings.update_groups}")
    if update == "out of memory":
        raise SystemExit(f"an update of the pass ({settings.update_groups} groups) ran out of the GPU's memory beside "
                         f"one speaking worker (O15)")
    peaks = [u["gpu_peak_reserved_gib"] for u in measured["updates"].values() if isinstance(u, dict)]
    peak = _gpu_used() - torch.cuda.memory_reserved(device) + int(max(peaks) * (1 << 30))
    del model, optimizer
    torch.cuda.empty_cache()
    return {"peak": peak, "now": _gpu_used(), "groups": measured["groups_written"]}


def workers_fit(workers: int, measured: Mapping[str, Any], passed: Mapping[str, Any] | None,
                available: Mapping[str, int | None]) -> list[str]:
    """O15: where ``workers`` workers do not fit, read after the measures (``available``: the host's and the GPU's free
    memory, with the measured worker and this process holding what they hold after them). Each worker needs the
    measured worker's peak (``measured``, `Speakers.measure`; it holds part of it already) and what a worker holds
    besides in a round (``measured["held"]``: the readout's model on the campaign's device, the round's kept series on
    the host). The pass, on a GPU: this process grows to its pass's peak (``passed``, `pass_memory_of`) while each of
    the other workers holds what the measured one holds after its batch and the readout's model, and the measured one
    its readout's model besides. One line for each that does not fit; none: they fit."""
    held = measured["held"]
    reader = {"host": 0 if measured["gpu"] is not None else held["reader_model"],
              "gpu": held["reader_model"] if measured["gpu"] is not None else 0}
    besides = {"host": reader["host"] + held["series"], "gpu": reader["gpu"]}
    out = []
    for name in ("host", "gpu"):
        if measured[name] is None:
            continue
        one = measured[name]["peak"] + besides[name]
        need, have = workers * one, available[name] + measured[name]["now"]
        if need > have:
            out.append(f"{name}: {workers} workers speaking need {need / 2**30:.1f} GiB (one worker's peak "
                       f"{measured[name]['peak'] / 2**30:.2f} GiB and {besides[name] / 2**30:.2f} GiB held in a round), "
                       f"{have / 2**30:.1f} GiB available")
    if passed is not None:
        each = measured["gpu"]["now"] + reader["gpu"]
        need = passed["peak"] - passed["now"] + (workers - 1) * each + reader["gpu"]
        if need > available["gpu"]:
            out.append(f"gpu: the pass beside {workers} workers needs {need / 2**30:.1f} GiB more (its peak "
                       f"{passed['peak'] / 2**30:.2f} GiB, each worker holding {each / 2**30:.2f} GiB), "
                       f"{available['gpu'] / 2**30:.1f} GiB free")
    return out


def available_memory(device: torch.device) -> dict[str, int | None]:
    """The host's available memory (``MemAvailable``) and the GPU's free memory (a CUDA device; else None), bytes."""
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines())
    host = int(meminfo["MemAvailable"].split()[0]) * 1024
    if device.type != "cuda":
        return {"host": host, "gpu": None}
    free = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
                           f"--id={device.index or 0}"], check=True, capture_output=True, text=True).stdout
    return {"host": host, "gpu": int(free.strip()) * (1 << 20)}


def require_workers_fit(speakers: Speakers, context: Context, settings: Settings, round_: int, model: Prior
                        ) -> dict[str, Any]:
    """O15, before a campaign's rounds: one worker's memory measured on the measured batches of round ``round_``
    (`Speakers.measure`, `measured_batches`; spoken with ``model``, the round's start: `campaign_model`) and this
    process's in one update of the pass on their groups (`pass_memory_of`), and
    the workers refused by name where they do not fit (`workers_fit`). The measures are printed (the campaign's log)
    and returned."""
    with tempfile.TemporaryDirectory(prefix="post_measure_") as scratch:
        measured = speakers.measure(round_, Path(scratch), model)
        passed = pass_memory_of(context, settings, Path(scratch), speakers.stage)
    available = available_memory(context.device)
    out = {"speak_workers": speakers.workers, "measured": measured, "pass": passed, "available": available}
    print(json.dumps(out), flush=True)
    short = workers_fit(speakers.workers, measured, passed, available)
    if short:
        raise SystemExit("the speaking workers do not fit (O15): " + "; ".join(short) + " — start fewer "
                         "(--speak-workers)")
    return out


# ---- the training pass
def update_pairs(directory: Path, data: Sequence[Any], settings: Settings, rng: np.random.Generator,
                 device: torch.device, *, part_width: int) -> Iterator[tuple[list[Samples], RowTensors]]:
    """The updates of a round's pass (module docstring, step 4): its written groups, file by file in the order they were
    spoken, each file's groups in an order shuffled by ``rng`` (the round's numbers; D130: an update mixes branch points
    and windows), `Settings.update_groups` at a time (a file's last update may hold fewer; each group a piece of it,
    `post.loss.update_step`: its memory a group's), each paired with
    `Settings.data_sentences` sentences of ``data`` drawn by ``rng``. A file is read when its first update is drawn.
    ``part_width``: the stage's token part (`Stage.part_width`)."""
    for path in sorted(directory.glob("groups_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
        loaded: list[Group] = torch.load(path, weights_only=False)
        groups = [loaded[int(i)] for i in rng.permutation(len(loaded))]
        for k in range(0, len(groups), settings.update_groups):
            chosen = rng.choice(len(data), size=min(settings.data_sentences, len(data)), replace=False)
            yield ([samples([group], device, part_width) for group in groups[k:k + settings.update_groups]],  # a piece a group
                   collate([data[int(i)] for i in chosen], device))


def pass_seed(seed: int, round_: int) -> int:
    """The torch seed of round ``round_``'s pass (the data term's dropout): a round is the same from its last
    checkpoint, so a resumed campaign is the campaign run through."""
    return int(np.random.default_rng([seed, round_, 2]).integers(1 << 62))


def train_pass(model: Prior, context: Context, optimizer: torch.optim.Optimizer, directory: Path, settings: Settings,
               rng: np.random.Generator, *, part_width: int) -> dict[str, Any]:
    """Step 4 of a round: one pass (`post.loss.one_pass`: the model at its start is the one that spoke the groups) over
    `update_pairs` (``part_width``: the stage's token part); its means, or no update where the round has no informative
    group."""
    parts = one_pass(model, context.base, optimizer, update_pairs(directory, context.data, settings, rng, context.device,
                                                                  part_width=part_width))
    return {"updates": len(parts), **(stacked(parts) if parts else {})}


# ---- the selection readout
def readout_numbers(seed: int, place: int, draw: int = 0) -> np.random.Generator:
    """The random numbers of the selection readout's window ``place``: the same every round (the readout compares the
    rounds' models on the same windows and numbers). Draw 0 is the readout's; the ceiling readout (`post_ceiling`, P48)
    reads each window again with draws 1, 2, …."""
    return np.random.default_rng([seed, 1 << 30, place] if draw == 0 else [seed, 1 << 30, place, draw])


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


def read_batch(model: Prior, context: Context, windows: Sequence[Window], places: Sequence[int], settings: Settings,
               split: str, draw: int = 0) -> list[WindowResult]:
    """A batch of the selection readout (the windows at ``places``): their first pass, each with its own numbers
    (`readout_numbers` of ``draw``), its ends."""
    data = context.splits[split]
    batch = [windows[p] for p in places]
    loop, order, observed = context.start_loop(split, batch)([w.signal_index for w in batch])
    return WindowLoop(model, loop, order, batch, data["sentences"], data["flights"], context.geometries,
                      context.rosters, context.finals, context.words, interval_s=context.interval_s,
                      variant=context.variant, edges_reference=context.edges_reference, faults=data["faults"],
                      observed=observed, device=context.device).run([readout_numbers(settings.seed, p, draw) for p in places])


def selection_readout(model: Prior, context: Context, windows: Sequence[Window], settings: Settings,
                      split: str = "select", speakers: Speakers | None = None, *, stage: Stage = STAGE_C
                      ) -> dict[str, Any]:
    """Step 5 of a round: the select windows' first pass (no branch), its ends by airport (module docstring); the
    same readout of ``split``'s windows (`post_validation`: the val days). Its batches read here (``stage``'s
    `Stage.read_batch`, stage C's `read_batch`), or by ``speakers`` (C13: the same ends, summed here in batch order;
    one process is the reference mode; refused unless they run ``stage``)."""
    _require_stage(speakers, stage)
    places = stage.batches(windows, settings)
    read = (speakers.read(model, windows, places, split) if speakers is not None else
            [stage.read_batch(model, context, windows, p, settings, split) for p in places])
    return counted_ends(windows, places, read)


def counted_ends(windows: Sequence[Window], places: Sequence[Sequence[int]], read: Sequence[Sequence[WindowResult]]
                 ) -> dict[str, Any]:
    """The selection readout's record: the ends of the batches at ``places`` (``read``, in batch order) by airport."""
    counted: dict[str, dict[str, Any]] = {}
    for batch, ends in zip(places, read, strict=True):
        for window, end in zip((windows[p] for p in batch), ends, strict=True):
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


def inputs_here(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """A campaign's inputs with each recorded path (`INPUT_PATHS`) as this checkout reads it (`this_checkout`, the rule
    of the Training exports, outline §5 rule 1): a campaign recorded in a worktree is read from another checkout (D157)."""
    from ts_transformer.experiments.training_export import this_checkout

    return {**inputs, **{key: str(this_checkout(inputs[key])) for key in INPUT_PATHS}}


def _but_rounds(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """The inputs with the rounds left out (D157: the one setting a resume may change)."""
    return {**inputs, "settings": {k: v for k, v in inputs["settings"].items() if k != "rounds"}}


def open_campaign(out: Path, inputs: dict[str, Any], git: dict[str, Any], checks: Any, *,
                  schema: str = CAMPAIGN_SCHEMA, reader: str = CLAIM_READER) -> dict[str, Any]:
    """``campaign.json``: a new campaign's, or a resume's. A resume is refused unless its inputs are the record's — the
    recorded paths as this checkout reads them (`inputs_here`), the rounds left out — and its rounds at least the
    record's; more rounds raise the record's count (D157), its paths kept as recorded, and are refused once the
    campaign's val read is claimed (`post_validation`, P47), spent or not. Each resume adds its entry: the
    time, the commit, the checks, the paths it read and the rounds before and after it. A round left half done (its
    directory without its checkpoint) is moved aside as ``round_<r>.aborted-<UTC>``. A stage's own campaign gives its
    format (``schema``) and the reader of its val read's claim (``reader``); stage C's by default (post-training §9 item
    11)."""
    path = out / "campaign.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["schema"] != schema:
            raise SystemExit(f"{out}: a {record['schema']} campaign, not {schema}")
        recorded, asked = inputs_here(record["inputs"]), inputs_here(inputs)
        if _but_rounds(recorded) != _but_rounds(asked):
            raise SystemExit(f"{out}: a campaign of other inputs or settings; a resume takes the same, its rounds or more")
        before, after = recorded["settings"]["rounds"], asked["settings"]["rounds"]
        if after < before:
            raise SystemExit(f"{out}: a campaign of {before} rounds; a resume may raise the rounds, not lower them to "
                             f"{after}")
        if after > before and validation_claim(out, reader) is not None:
            raise SystemExit(f"{out}: its val read is claimed ({validation_claim(out, reader)}); its rounds are not "
                             f"raised after it (P47)")
        left = out / f"round_{done_rounds(out)}"
        if left.exists():
            aborted = left.with_name(f"{left.name}.aborted-{utc_now().replace(':', '')}")
            left.rename(aborted)
            record["aborted"].append(aborted.name)
        record["inputs"]["settings"]["rounds"] = after
        record["resumed"].append({"utc": utc_now(), "git": git, "checks": checks,
                                  "inputs": {key: asked[key] for key in INPUT_PATHS},
                                  "rounds": {"before": before, "after": after}})
        write_json_atomic(path, record)
        return record
    if out.exists():
        raise SystemExit(f"{out} exists and holds no campaign")
    out.mkdir(parents=True)
    record = {"schema": schema, "started_utc": utc_now(), "inputs": inputs, "git": git, "checks": checks,
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


def start_checkpoint(start: Mapping[str, Any]) -> Path:
    """The checkpoint a campaign's ``start`` names (`Settings.start`), in this checkout."""
    return REPO_ROOT / start["campaign"] / f"round_{start['round']}" / "checkpoint.pt"


def source_campaign(campaign: Path, round_: int, *, formal: bool) -> dict[str, Any]:
    """The record of the campaign a start comes from (post-training §9 item 12, D162, multi-aircraft control D164),
    refused by name unless it is a campaign of stage C (`CAMPAIGN_SCHEMA`) with round ``round_`` done and, for a formal
    campaign (``formal``), not a smoke one."""
    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    if record["schema"] != CAMPAIGN_SCHEMA:
        raise ValueError(f"{campaign} is a {record['schema']} campaign, not {CAMPAIGN_SCHEMA}")
    if formal and record["inputs"]["smoke"]:
        raise ValueError(f"{campaign} is a smoke campaign: a formal campaign does not start from it")
    if not 0 <= round_ < done_rounds(campaign):
        raise ValueError(f"{campaign} holds the checkpoints of rounds 0–{done_rounds(campaign) - 1}, not {round_}")
    return record


def start_of(campaign: Path, round_: int, *, formal: bool) -> dict[str, Any]:
    """The setting of a start from round ``round_`` of ``campaign`` (`Settings.start`, D162): the campaign as this
    checkout names it (repository-relative), the round and its checkpoint's sha256; the source's rules
    (`source_campaign`) refused by name first."""
    from ts_transformer.experiments.training_export import this_checkout

    source_campaign(campaign, round_, formal=formal)
    return {"campaign": repo_relative(this_checkout(campaign)), "round": round_,
            "checkpoint_sha256": file_sha256(campaign / f"round_{round_}" / "checkpoint.pt")}


def round_start(context: Context, start: Mapping[str, Any], seed: int) -> tuple[Prior, dict[str, Any]]:
    """Post-training §9 item 12 (D162, multi-aircraft control D164): round ``start["round"]`` of the campaign
    ``start["campaign"]`` opened as a start, in eval mode, with its identity — refused by name unless the checkpoint's
    bytes are the recorded ones, the source's rules hold (`source_campaign`: a formal campaign, ``context.formal``,
    from a formal one), its identity is of this base, under today's procedure masks, with its campaign's traffic shape,
    after that round, and the new campaign's ``seed`` is not its source's (so that its rounds draw new windows). Stage C's
    start of a campaign (`campaign_start`) and stage D's start call it."""
    path = start_checkpoint(start)
    if file_sha256(path) != start["checkpoint_sha256"]:
        raise ValueError(f"{path}: not the checkpoint the campaign started from (its sha256 is not the recorded one)")
    source = settings_of(source_campaign(REPO_ROOT / start["campaign"], start["round"], formal=context.formal))
    state = torch.load(path, weights_only=False, map_location=context.device)
    held = state["identity"]
    expected = {"schema": POST_CHECKPOINT_SCHEMA, "base": context.base_identity, "procedure_masks": PROCEDURE_MASKS,
                "traffic": TrafficConfig(source.traffic_hidden, source.traffic_heads).to_dict()}
    if {k: held[k] for k in expected} != expected or len(held["rounds"]) != start["round"] + 1:
        raise ValueError(f"{path}: a checkpoint of another base, masks, traffic shape or round than the start's")
    if held["seed"] == seed:
        raise ValueError(f"{path}: its campaign's seed is {seed}; a campaign from a round takes another seed, so that its "
                         f"rounds draw new windows (D162)")
    model, _ = start_model(context, source)
    model.load_state_dict(state["model"])
    return model.eval(), held


def campaign_start(context: Context, settings: Settings) -> tuple[Prior, torch.optim.Optimizer]:
    """The model a campaign starts from and a new optimizer (the user, 2026-10-07): the base with zero-output traffic
    modules (D29, ``settings.start`` None), or the weights of a round of another campaign (``settings.start``,
    `round_start`), refused by name besides unless its traffic shape is this campaign's (D162). The optimizer starts
    afresh, and the pull term pulls toward the base either way (the user, 2026-10-07)."""
    model, optimizer = start_model(context, settings)
    if settings.start is None:
        return model, optimizer
    source, held = round_start(context, settings.start, settings.seed)
    if held["traffic"] != TrafficConfig(settings.traffic_hidden, settings.traffic_heads).to_dict():
        raise ValueError(f"{start_checkpoint(settings.start)}: a checkpoint of another base, masks, traffic shape or round "
                         f"than the start's (its traffic shape is not this campaign's)")
    model.load_state_dict(source.state_dict())     # in place: the optimizer holds these parameters, with no state yet
    return model, optimizer


def settings_of(record: Mapping[str, Any]) -> Settings:
    """A campaign's settings as its ``campaign.json`` records them."""
    return Settings(**record["inputs"]["settings"])


def round_model(context: Context, settings: Settings, out: Path, round_: int | None) -> Prior:
    """The model of campaign ``out`` after round ``round_`` (its checkpoint), or at its start (None: `campaign_start`),
    in eval mode; a checkpoint is refused unless its identity is this campaign's on this base, under today's procedure
    masks (§4 item 3)."""
    if round_ is None:
        return campaign_start(context, settings)[0].eval()
    model, _ = start_model(context, settings)
    state = torch.load(out / f"round_{round_}" / "checkpoint.pt", weights_only=False, map_location=context.device)
    held = state["identity"]
    expected = {"schema": POST_CHECKPOINT_SCHEMA, "base": context.base_identity, "procedure_masks": PROCEDURE_MASKS,
                "traffic": TrafficConfig(settings.traffic_hidden, settings.traffic_heads).to_dict(),
                "seed": settings.seed}
    if {k: held[k] for k in expected} != expected or len(held["rounds"]) != round_ + 1:
        raise ValueError(f"{out}/round_{round_}: a checkpoint of another base, masks, traffic shape, seed or round")
    model.load_state_dict(state["model"])
    return model.eval()


def campaign_model(out: Path, context: Context, settings: Settings, stage: Stage = STAGE_C
                   ) -> tuple[Prior, torch.optim.Optimizer]:
    """The model and optimizer that campaign ``out``'s next round starts from: a new campaign's the stage's start
    (`Stage.start`), a resumed one's its last checkpoint loaded into the stage's model (`Stage.start_model`)."""
    first = done_rounds(out)
    model, optimizer = stage.start_model(context, settings) if first else stage.start(context, settings)
    if first:
        state = torch.load(out / f"round_{first - 1}" / "checkpoint.pt", weights_only=False, map_location=context.device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
    return model, optimizer


def run_campaign(out: Path, settings: Settings, context: Context, speakers: Speakers | None = None,
                 stage: Stage = STAGE_C) -> None:
    """The rounds not yet done (module docstring), each closed by its checkpoint; the speaking by ``speakers`` when
    given (`Speakers`, refused unless they run ``stage``); ``stage``'s parts of a round (stage C's, `STAGE_C`): a new
    campaign starts from the stage's start (`Stage.start`), a resumed one loads its last checkpoint into the stage's
    model (`Stage.start_model`)."""
    _require_stage(speakers, stage)
    first = done_rounds(out)
    model, optimizer = campaign_model(out, context, settings, stage)
    select = stage.selection(context, settings, "select")
    for round_ in range(first, settings.rounds):
        directory = out / f"round_{round_}"
        directory.mkdir()
        windows, drawn = stage.draw(context, settings, round_)
        context.splits["train"]["start"].release()        # a round's flights kept, not the campaign's (C13)
        if speakers is not None and context.device.type == "cuda":
            torch.cuda.empty_cache()     # the speakers share the GPU: the memory this process cached in the last pass
        spoken = stage.speak(model, context, windows, settings, round_, directory, speakers, stage=stage)
        torch.manual_seed(pass_seed(settings.seed, round_))                    # the data term's dropout
        passed = train_pass(model, context, optimizer, directory, settings,
                            np.random.default_rng([settings.seed, round_, 1]), part_width=stage.part_width)
        if speakers is not None and context.device.type == "cuda":
            torch.cuda.empty_cache()     # the memory of the pass, before the speakers read
        readout = stage.readout(model, context, select, settings, speakers, stage=stage)
        written = sorted(directory.glob("groups_*.pt"))
        write_json_atomic(directory / "round.json", {
            "round": round_, "git": git_state(), "finished_utc": utc_now(), "draw": drawn,
            "speak_workers": speakers.workers if speakers is not None else 1,         # information (`Speakers`)
            "windows": [stage.record(w) for w in windows], "speaking": spoken,
            "groups_bytes": sum(p.stat().st_size for p in written), "pass": passed, "selection_readout": readout})
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "identity": stage.identity(context, settings, out, round_ + 1)}, directory / "checkpoint.pt")
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
    parser.add_argument("--start-campaign", type=Path,
                        help="start from a round of this campaign (with --start-round); the base when left out")
    parser.add_argument("--start-round", type=int, help="the round of --start-campaign whose weights the campaign starts from")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speak-workers", type=int, default=1,
                        help="processes that speak a round's batches in parallel (Speakers; the same results)")
    parser.add_argument("--smoke", action="store_true", help="SMOKE: a tree with changes too; no result")
    args = parser.parse_args(argv)
    if args.speak_workers < 1:
        parser.error("--speak-workers is at least 1")
    if (args.start_campaign is None) != (args.start_round is None):
        parser.error("--start-campaign and --start-round go together")
    start = None
    if args.start_campaign is not None:
        source = args.start_campaign if args.start_campaign.is_absolute() else REPO_ROOT / args.start_campaign
        try:
            start = start_of(source, args.start_round, formal=not args.smoke)
        except ValueError as refused:
            parser.error(str(refused))
    prior_dir, instructions, executor, census, procedure_root, out = (p if p.is_absolute() else REPO_ROOT / p for p in (
        args.prior, args.instructions, args.executor, args.windows, args.procedure_root, args.out))
    settings = Settings(args.rounds, {kind: getattr(args, f"windows_{kind.lower()}") for kind in KINDS},
                        args.batch_windows, args.continuations, args.seed, args.prior_lr, args.traffic_lr,
                        args.weight_decay, args.update_groups, args.data_sentences, args.select_per_airport,
                        args.traffic_hidden, args.traffic_heads, start)
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("a campaign that is not a smoke runs on a clean checkout")
    _, opened, _ = require_conforming_closed_loop(instructions, executor)          # D69: the checks run here (D73)
    edges_reference = census / "conformance" / "edges.npz"
    checked_edges(edges_reference)                                                 # D104
    device = torch.device(args.device)
    # with speakers, the context is opened on the CPU and they are forked before this process uses the GPU
    context = open_context(prior_dir, instructions, executor, edges_reference,
                           torch.device("cpu") if args.speak_workers > 1 else device, procedure_root,
                           formal=not args.smoke)                                  # D132
    if settings.start is not None:      # refused by name before the workers and campaign.json (D162)
        campaign_start(context, settings)
    speakers = Speakers(context, settings, args.speak_workers, device) if args.speak_workers > 1 else None
    context = replace(context, device=device, base=context.base.to(device).eval())
    inputs = {"prior": str(prior_dir), "instructions": str(instructions), "executor": str(executor),
              "windows": str(census), "procedure_root": str(procedure_root), "settings": asdict(settings),
              "smoke": args.smoke}
    try:
        open_campaign(out, inputs, git, opened["checks"])           # a resume of other inputs refused before the measure
        if speakers is not None and done_rounds(out) < settings.rounds:          # O15, before any round
            require_workers_fit(speakers, context, settings, done_rounds(out), campaign_model(out, context, settings)[0])
        run_campaign(out, settings, context, speakers)
    finally:
        if speakers is not None:
            speakers.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
