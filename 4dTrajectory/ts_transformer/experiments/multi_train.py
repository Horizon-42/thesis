"""MC4: stage D's campaign, version 1 (multi-aircraft control §3–§5, §7, §11 MC4; D141–D148, D152) — rounds of branch
training in windows in which every arrival of a span L is commanded, from a chosen round of stage C. With
``--joint-heads`` (MC8; D153, D212), version 2: each commanded aircraft also reads the other commanded aircraft of its
window at the same tick (`multi.joint`).

It is stage C's campaign (`post_train.run_campaign`, the round's skeleton, post-training §9 item 11) with stage D's
parts (`stage_d`):

- **The start** (§5 item 1, D152, D164): the chosen round of stage C's campaign (``--start-campaign``,
  ``--start-round``), recorded in the settings as ``{campaign, round, checkpoint_sha256}`` (`post_train.start_of`) and
  opened by post-training §9 item 12's one function (`post_train.round_start`: the bytes, the identity of this base,
  masks, stage C's traffic shape and that round, a formal campaign from a formal one, a seed other than the source's,
  each refused by name; a formal start also from a campaign that has done every round, `require_finished`, D166
  item 19), with stage D's token part added to its traffic attention (`multi.tokens`,
  `multi-commanded-tokens-v1`, its projection at zero: the start says what the round says) and a new optimizer. Or
  (D200; no ``--start-campaign``) the base itself, with zero-output traffic modules of the campaign's seed and its
  traffic shape (``--traffic-hidden``, ``--traffic-heads``; `post_train.traffic_model`, D29) and the token part at zero.
  Or (D218; ``--start-campaign`` a campaign of stage D) a round of stage D's campaign, as it trained (its token part
  and its module; refused by name besides unless it is version 2 or version 1 as the new campaign is,
  `multi_source_campaign`, `stage_d`). The pull is to the base, as in every stage of the post-training. A resume and a
  speaking worker load a state into the model of that shape (stage C's start model of the source campaign's settings,
  or the base's, with the token part; a stage D source's own shape).
- **The draw of a round** (D146, `draw_windows`): each kind's count in equal parts of the spans L (``--spans-s``,
  `span_counts`); the train days' real windows in a permutation by the seed and the round, each in turn an anchor
  (`multi.windows.Anchors`) of each kind still short: its span drawn with the round's numbers among the spans of that
  kind still short, its window of that span, and for a compressed one its later aircraft moved toward the anchor by c
  in [c_min, 1) (`multi.windows.compressed`; a window without a later aircraft has no compressed form); a window left
  out when one of its commanded aircraft opens inside a loss it answers for (`multi.windows.left_out`), until each
  kind's count of each span is reached; a shortfall of a span is recorded, never silent. The batches hold windows of
  one span, at most that span's ``--batch-rows`` commanded aircraft (one number a span: a batch's memory grows with its
  ticks too, the user, 2026-10-07, after MC5's profile), and command each flight once (`span_batches`); with
  ``--batch-squares`` also at most that span's sum of commanded aircraft squared (the user, 2026-10-11).
- **The speaking** (`speak_batch`, `post_branches.branch_round`): stage D's rules (`multi.credit.rules`: W the sum,
  spoken again below the count, the varied aircraft and their branch points, each aircraft's own numbers, D141–D143)
  and its loop (`loop_options`: the rule of who answers, D145; the token part, D152). Its record is stage C's
  (`post_train.speak_round`'s sums), its rewards and outcomes those of each commanded aircraft. ``--branching backward``
  (`MultiSettings.branching`, D182, MC12; `grid` the default, D143's points): the branch points found back from each
  varied aircraft's event (`post_branches.backward_round`, `multi.backward`), the record holding besides each group's
  span and depth (``group_depths_s``), the searches and those that found a decision point, and the continuations flown
  only to the event + 60 s and those flown to their window's end. ``--branching majority`` (D192): the same search with
  the majority point in place of point B (`multi.backward`), its record holding besides each group's kind
  (``group_kinds``) and each majority point's span and depth (``majority_depths_s``). The readout's ``go_arounds`` and
  ``aircraft`` of each span give the go-arounds per aircraft that D182 asks to watch. ``--front-attention`` (`MultiSettings.front_attention`,
  D190 (2); `copied` the default): how a window's copies read the front of the speaker's cache they hold once.
  ``--data-weight`` (`MultiSettings.data_weight`; R2 of stage D's plan, the user, 2026-10-10): the data term's weight in
  each update (`post.loss.update_step`), `post.loss.DATA_WEIGHT` the default. ``--group-weight``
  (`MultiSettings.group_weight`, D195; the user, 2026-10-11): each group's weight in the pass
  (`post.branches.GROUP_WEIGHTS`), `EQUAL` the default; with `MINORITY_SIDE` each batch's record holds its groups'
  minority sides (``group_minorities``), from which the pass takes the round's mean. ``--branch-every-s`` and
  ``--segment-only`` (D200, with `GRID` only): the grid's interval (D143's 120 s the default), and a group's advantage
  only up to the varied aircraft's next branch point (stage C's P55; default: up to its event).
- **The selection readout** (§5 item 2, `selection_windows`, `read_batch`, `selection_readout`): at most
  ``select_per_airport`` real windows of each span of each airport of the select days, their anchors in an order drawn
  once with the seed (D146's rule), the first pass only, each aircraft with its own numbers (the same every round). By
  airport, span and kind (each with its ``all``): the
  windows, the commanded aircraft, W per aircraft (``reward_mean``) and per window, the outcomes, the go-arounds, the
  silent aircraft, the rows of the speed-word mask, the faulty points (post-training D114), and the losses of
  separation by pair (`multi.separation.PAIRS`; judged again on the states flown, `multi.census.flown_positions`, the
  census's rule). No criterion is applied (D7).
- **The identity** (§7 row 3): `MULTI_CHECKPOINT_SCHEMA`, the start's identity (`start_identity`), the procedure masks, the token part's
  format and features, the seed, the spans and their counts (`span_counts`), the kinds and c_min, and each round's
  windows by their anchors, spans, commanded flights and shifts (`window_record`).
- **Version 2** (D153, D212; ``--joint-heads``, `MultiSettings.joint_heads`): the module beside each layer's traffic
  attention (`multi.joint`, its output at zero at the start; its parameters at the traffic learning rate), the window
  loop's window input (`loop_options`' ``joint``), each group's sentences with their windows (``whole_windows``), its
  formats `MULTI_CAMPAIGN_SCHEMA_V2` and `MULTI_CHECKPOINT_SCHEMA_V2` and the module's shape in the identity. Its start
  is one of version 1's (a round of stage C, or the base) with the module at zero, or a round of version 2 (D218) with
  the module as trained.

THE CHECKS, as stage C's: the closed loop's (D69, with the labeller's and the executor's) and the edge features' (D104).
A formal campaign (not ``--smoke``) runs from a clean checkout (a run worktree at the merged commit, outline D163; its
intent is written by hand before the publication, not checked here), on stage B's formal base, from a formal campaign
of stage C or from the base. Its campaign record is `MULTI_CAMPAIGN_SCHEMA`, resumable as stage C's (`post_train.open_campaign` with stage
D's format and reader). Its speaking workers (``--speak-workers`` above 1) are sized from stage D's profile of these
inputs and settings (``--profile``, `profiled_fit`: the profile's measure with its margin and the memory free now, each
worker's GPU capped at its share), never measured before a run (the user, 2026-10-07); ``--speak-device`` puts them on
a device of their own, e.g. the CPU beside the pass on the GPU (D180; by default the campaign's ``--device``).
``--executor-device cpu`` (multi-aircraft control S1) starts the executor of each loop on the CPU. The speaker stays on
its device (`Context.executor_device`). The default, ``speak``, keeps the executor on the speaker's device. This is a
process option, not a campaign setting: a resume can give a different value. Each batch record holds the device of its
executor.

    python run_ts.py multi_train --prior <the base> --instructions <A34's artefact> --executor <its spec> \\
        --windows <stage C's census: its edge reference> --start-campaign <stage C's campaign> --start-round <r> \\
        --out 4dTrajectory/outputs/POOLED/multi/<campaign id> --rounds … --spans-s 300 600 1200 --c-min … \
        --batch-rows 64 48 24 (every count given)
    (from the base, D200: --traffic-hidden 64 --traffic-heads 4 in place of --start-campaign and --start-round)

The readout also gives the time the aircraft take (§5 item 4, O18; `multi.timing`): each landing's delay against its
record (p50, p90), the spacing at the threshold of successive landings on one runway (p1, p5, p50) beside the records' of
the same windows, and the pairs landed in the recorded order. The validation readout is `multi_validation`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import product
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.experiments.post_branches import Rules, backward_round, branch_round
from ts_transformer.experiments.post_train import (
    Context, Speakers, Stage, available_memory, batches, cpu_workers_refused, done_rounds, open_campaign, open_context,
    round_start, run_campaign, save_batch_file, settings_of, source_campaign, speak_device_refused, speak_round,
    start_checkpoint, start_model, start_of, traffic_model, train_pass, workers_fit,
)
from ts_transformer.experiments.post_window_loop import (
    LANDED, LOST_SEPARATION, WindowLoop, checked_edges, turns_readout,
)
from ts_transformer.multi import credit
from ts_transformer.multi.backward import AFTER_S, BRANCHINGS, GRID, MAJORITY, POINTS
from ts_transformer.multi.census import WindowCount, loop_positions, window_losses
from ts_transformer.multi.joint import JointConfig, add_joint_attention
from ts_transformer.multi.separation import PAIRS, Answering
from ts_transformer.multi.timing import (
    delays, landed, loop_spacings, order, quantiles, record_spacings, recorded_landings,
)
from ts_transformer.multi.tokens import PART_FEATURES, TOKENS_SCHEMA, TokenPart
from ts_transformer.multi.windows import COMPRESSED, KINDS, REAL_KIND, Anchors, Drawn, compressed, left_out
from ts_transformer.post.branches import BRANCH_EVERY_S, CONTINUATIONS, EQUAL, GROUP_WEIGHTS, MINORITY_SIDE, branch_rows
from ts_transformer.post.loss import DATA_WEIGHT
from ts_transformer.post.scene import Window
from ts_transformer.post.window_lists import identity_of
from ts_transformer.post.traffic_attention import TrafficConfig, add_token_part, parameter_groups
from ts_transformer.prior.model import COPIED, FRONT_ATTENTIONS, Prior
from ts_transformer.prior.procedure import PROCEDURE_MASKS
from ts_transformer.io_utils import file_sha256
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

#: The formats of stage D's campaign and checkpoints, version 1 (§10).
MULTI_CAMPAIGN_SCHEMA = "ts-multi-train-v1"
MULTI_CHECKPOINT_SCHEMA = "ts-multi-checkpoint-v1"
#: The formats of version 2 (D153, D212): a campaign with `MultiSettings.joint_heads`.
MULTI_CAMPAIGN_SCHEMA_V2 = "ts-multi-train-v2"
MULTI_CHECKPOINT_SCHEMA_V2 = "ts-multi-checkpoint-v2"
#: The reader's name in the claim of a campaign's val read (the validation readout of stage D).
MULTI_CLAIM_READER = "multi_validation"
#: The format of stage D's profile (`multi_profile`), which a campaign's speaking workers are sized from (`profiled_fit`).
MULTI_PROFILE_SCHEMA = "ts-multi-profile-v2"
#: The factor on each measured peak before the workers are counted (D179: a batch that was not measured needed about
#: 30 % more than its span's measured peak), recorded in the profile.
MEASURE_MARGIN = 1.3
#: The settings that set a round's memory, which a campaign shares with its profile (`profiled_fit`); the others (its
#: rounds, seed, learning rates, a start of the same traffic shape, and the select set, which the measure does not read,
#: D179, D181 (55)) change the words spoken, not the shapes — a worker past its share stops by name.
PROFILED_SETTINGS = ("per_kind", "spans_s", "c_min", "batch_rows", "continuations", "update_groups", "data_sentences",
                     "branching", "front_attention", "batch_squares", "joint_heads", "search_points", "judged_after_s",
                     "fine_point")
#: The devices of ``--executor-device`` (multi-aircraft control S1). ``speak``: the executor flies on the speaker's
#: device (the behaviour before S1). ``cpu``: the executor flies on the CPU (`Context.executor_device`).
EXECUTOR_DEVICES = ("speak", "cpu")
#: The inputs a campaign shares with its profile.
PROFILED_INPUTS = ("prior", "instructions", "executor", "windows", "procedure_root")


@dataclass(frozen=True)
class MultiSettings:
    """What a campaign of stage D is: everything a resume must find the same (the rounds may be raised, D157)."""

    rounds: int
    per_kind: dict[str, int]
    #: the spans L of the windows, s, increasing (D146; each kind's count in equal parts of them, `span_counts`)
    spans_s: list[float]
    c_min: float
    #: the most commanded aircraft (rows) of a batch of each span, in the order of `spans_s` (`span_batches`)
    batch_rows: list[int]
    continuations: int
    seed: int
    prior_lr: float
    traffic_lr: float
    weight_decay: float
    update_groups: int
    data_sentences: int
    select_per_airport: int
    #: the start (D164): round ``round`` of stage C's campaign ``campaign`` (repository-relative; D218: or of stage D's,
    #: `multi_start_of`) and its checkpoint's ``checkpoint_sha256`` (`post_train.start_of`); or None (D200): the base
    #: with zero-output traffic modules of the
    #: shape `traffic_hidden`, `traffic_heads` (`post_train.traffic_model`)
    start: dict[str, Any] | None
    #: where a round's branch points are (D182, MC12): `GRID`, D143's (the default: the behaviour before D182; a record
    #: without the setting reads as it), or `BACKWARD` / `MAJORITY`, found back from each varied aircraft's event with
    #: point B (D182) or the majority point (D192) (`multi.backward`)
    branching: str = GRID
    #: how a window's copies (a branch point's continuations) read the front of the speaker's cache they hold once
    #: (D190 (2), the user's choice; `FRONT_ATTENTIONS`): `COPIED`, the default (bit for bit the behaviour before it; a
    #: record without the setting reads as it), or `MERGED` (no copy; a sum's last bits may move)
    front_attention: str = COPIED
    #: D191 (the user, 2026-10-10): of every ``cpu_period`` batches of a round, by index, the first ``cpu_batches``
    #: spoken by the CPU speaking workers (`post_train.batch_device`), the others on the speak device, the selection
    #: readout on the speak device alone; a resume may change it (`post_train.RESUME_MAY_CHANGE`); the default every
    #: batch on the speak device (the behaviour before it; a record without it reads so)
    cpu_batches: int = 0
    cpu_period: int = 1
    #: the data term's weight in each update (R2 of stage D's plan, the user, 2026-10-10; `post.loss.update_step`); the
    #: default `DATA_WEIGHT` (the behaviour before it; a record without it reads so)
    data_weight: float = DATA_WEIGHT
    #: each group's weight in the pass (D195, the user, 2026-10-11; `post.branches.GROUP_WEIGHTS`); the default `EQUAL`
    #: (the behaviour before it; a record without it reads so)
    group_weight: str = EQUAL
    #: the user, 2026-10-11 (after batch 186 of the formal round 5): the most sum of commanded aircraft squared of a
    #: batch of each span, in the order of `spans_s`, besides `batch_rows` (`span_batches`, `post_train.batches`); empty,
    #: the default: no such bound (the behaviour before it; a record without it reads so)
    batch_squares: list[int] = field(default_factory=list)
    #: a batch whose sum of commanded aircraft squared reaches this is spoken on the speak device (`post_train.Settings
    #: .gpu_squares`, `post_train.batch_kinds`); a resume may change it; the default 0: off (a record without it reads so)
    gpu_squares: int = 0
    #: the traffic shape of a start from the base (D200; `TrafficConfig`: the token network's hidden width and the
    #: attention's heads), given with ``start`` None only; a start from a round of stage C takes its campaign's shape
    #: (D164; the default None: the behaviour before D200, a record without them reads so)
    traffic_hidden: int | None = None
    traffic_heads: int | None = None
    #: D200: the interval of the grid's branch points (`multi.credit.points`), s; D143's `BRANCH_EVERY_S`, the default
    #: (the behaviour before D200; a record without it reads so); with `GRID` only
    branch_every_s: float = BRANCH_EVERY_S
    #: D200: a group's advantage and pull only on the rows from its branch point to the next one (stage C's P55,
    #: `post.branches.samples`' ``segment_rows``); False, the default: up to the event (the behaviour before D200; a
    #: record without it reads so); with `GRID` only
    segment_only: bool = False
    #: D212: version 2, the heads of the module beside each layer's traffic attention (`multi.joint.JointConfig`); None,
    #: the default: version 1 (the behaviour before it; a record without it reads so)
    joint_heads: int | None = None
    #: D217 (the user, 2026-10-16), with `BACKWARD` / `MAJORITY` only, each default D182's (the behaviour before it; a
    #: record without it reads so): the search's candidate points at most (`multi.backward.points`), the span after the
    #: event at which a point's continuations are judged, s (`multi.backward.judged_until`), the fine point (D182 (5)),
    #: and ``common_cut``: every sentence of a group counted only up to that judged tick (False: each up to its own
    #: event, D142; `post_branches.backward_round`)
    search_points: int = POINTS
    judged_after_s: float = AFTER_S
    fine_point: bool = True
    common_cut: bool = False

    def __post_init__(self) -> None:
        if set(self.per_kind) != set(KINDS) or min(self.per_kind.values()) < 0 or not any(self.per_kind.values()):
            raise ValueError(f"a count of each kind of window {KINDS}, none negative, not all zero")
        if (not self.spans_s or list(self.spans_s) != sorted(set(self.spans_s)) or self.spans_s[0] <= 0.0
                or not 0.0 < self.c_min < 1.0):
            raise ValueError(f"spans L above 0 s, increasing, and c_min in (0, 1), not {self.spans_s} s and "
                             f"{self.c_min:g} (L = 0 is stage C's window, not one of stage D's spans)")
        if len(self.batch_rows) != len(self.spans_s):
            raise ValueError(f"a batch's rows for each span, not {self.batch_rows} for {self.spans_s}")
        if min(self.rounds, *self.batch_rows, self.continuations, self.update_groups, self.data_sentences,
               self.select_per_airport) <= 0:
            raise ValueError("rounds, batch rows, continuations, groups an update, data sentences and select windows "
                             "are positive")
        if self.start is not None and set(self.start) != {"campaign", "round", "checkpoint_sha256"}:
            raise ValueError(f"a start is its campaign, round and checkpoint_sha256, not {sorted(self.start)}")
        shape = (self.traffic_hidden, self.traffic_heads)
        if (self.start is None and (None in shape or min(shape) <= 0)) or (self.start is not None
                                                                          and shape != (None, None)):
            raise ValueError(f"a traffic shape (traffic_hidden, traffic_heads) {shape}: positive with a start from the "
                             f"base, none with a start from a round of stage C (its campaign's shape, D164)")
        if not self.branch_every_s > 0:
            raise ValueError(f"branch_every_s {self.branch_every_s}: positive")
        if self.branching != GRID and (self.segment_only or self.branch_every_s != BRANCH_EVERY_S):
            raise ValueError(f"segment_only and branch_every_s are the grid's (branching {GRID!r}), not "
                             f"{self.branching!r}'s")
        if self.branching not in BRANCHINGS:
            raise ValueError(f"branching {self.branching!r} is none of {BRANCHINGS}")
        searched = (self.search_points, self.judged_after_s, self.fine_point, self.common_cut)
        if self.branching == GRID and searched != (POINTS, AFTER_S, True, False):
            raise ValueError(f"search_points, judged_after_s, fine_point and common_cut are the backward search's "
                             f"(D217), not the grid's")
        if self.search_points < 1 or not self.judged_after_s > 0:
            raise ValueError(f"search_points {self.search_points} and judged_after_s {self.judged_after_s}: positive")
        if self.front_attention not in FRONT_ATTENTIONS:
            raise ValueError(f"front_attention {self.front_attention!r} is none of {FRONT_ATTENTIONS}")
        if not 0 <= self.cpu_batches <= self.cpu_period or self.cpu_period < 1:
            raise ValueError(f"cpu_batches {self.cpu_batches} of every cpu_period {self.cpu_period}: a period of 1 or "
                             f"more, at most its batches on the CPU")
        if not self.data_weight >= 0:
            raise ValueError(f"data_weight must be 0 or more, not {self.data_weight}")
        if self.group_weight not in GROUP_WEIGHTS:
            raise ValueError(f"group_weight {self.group_weight!r} is none of {GROUP_WEIGHTS}")
        if self.batch_squares and (len(self.batch_squares) != len(self.spans_s) or min(self.batch_squares) <= 0):
            raise ValueError(f"batch_squares: none, or a positive bound for each span, not {self.batch_squares} for "
                             f"{self.spans_s}")
        if self.gpu_squares < 0:
            raise ValueError(f"gpu_squares {self.gpu_squares}: 0 (off) or more")
        if self.joint_heads is not None and self.joint_heads <= 0:
            raise ValueError(f"joint_heads {self.joint_heads}: positive (version 2), or none (version 1)")

    @property
    def joint(self) -> bool:
        """Whether the campaign is version 2 (D212)."""
        return self.joint_heads is not None

    @property
    def campaign_schema(self) -> str:
        """The format of the campaign's record: version 1's or version 2's."""
        return MULTI_CAMPAIGN_SCHEMA_V2 if self.joint else MULTI_CAMPAIGN_SCHEMA

    @property
    def checkpoint_schema(self) -> str:
        """The format of the campaign's checkpoints: version 1's or version 2's."""
        return MULTI_CHECKPOINT_SCHEMA_V2 if self.joint else MULTI_CHECKPOINT_SCHEMA


def span_key(span_s: float) -> str:
    """A span's name in the records (its seconds, ``"600"``)."""
    return f"{span_s:g}"


def span_counts(settings: MultiSettings) -> dict[str, dict[str, int]]:
    """Each kind's windows of a round in equal parts of the spans (D146), by kind and span (`span_key`): its count
    divided by the spans, the remainder one each to the first spans."""
    n = len(settings.spans_s)
    return {kind: {span_key(span): count // n + (k < count % n) for k, span in enumerate(settings.spans_s)}
            for kind, count in settings.per_kind.items()}


def span_batches(windows: Sequence[Window], settings: MultiSettings) -> list[list[int]]:
    """The windows' places in batches (`Stage.batches`): the windows of each span in `post_train.batches` of that span's
    `MultiSettings.batch_rows` (and `MultiSettings.batch_squares`, where given), span after span; a window of another
    span is refused by name."""
    others = {w.span_s for w in windows} - set(settings.spans_s)
    if others:
        raise ValueError(f"windows of spans {sorted(others)} s, not of this campaign's {settings.spans_s}")
    out = []
    squares = settings.batch_squares or [None] * len(settings.spans_s)
    for span, size, bound in zip(settings.spans_s, settings.batch_rows, squares):
        places = [p for p, w in enumerate(windows) if w.span_s == span]
        out += [[places[i] for i in batch] for batch in batches([windows[p] for p in places], size, bound)]
    return out


def kind_of(window: Window) -> str:
    """The kind of a window of stage D as it flies: compressed where a commanded aircraft is moved (a compressed draw
    whose shifts all round to 0 flies the real window, and reads as one)."""
    return COMPRESSED if any(item.shift_s for item in window.joined) else REAL_KIND


def window_record(window: Window) -> dict[str, Any]:
    """A window of stage D (§7 row 3): its anchor, its row 0, its span, its commanded flights, each one's shift and its
    kind."""
    return {"anchor": window.commanded.key, "row0_s": window.row0_s, "span_s": window.span_s,
            "commanded": [record.key for record in window.commanded_all],
            "shifts_s": [0.0] + [item.shift_s for item in window.joined], "kind": kind_of(window)}


def loop_options(context: Context, joint: bool = False) -> dict[str, Any]:
    """Stage D's options of the window loop (post-training §9 item 5): the rule of who answers (D145), the token part
    (D152) and the device of the executor (multi-aircraft control S1). The executor device is the device that
    `Context.start_loop` starts the executor on. ``joint``: version 2's window input (D212; `MultiSettings.joint`)."""
    return {"answering": Answering(context.separations, context.finals, context.words.spec.step_s),
            "token_part": TokenPart(context.words), "part_width": TokenPart.width,
            "executor_device": context.executor_device, "joint": joint}


# ---- the draw
def draw_windows(context: Context, settings: MultiSettings, rng: np.random.Generator
                 ) -> tuple[list[Window], dict[str, Any]]:
    """A round's windows (module docstring), kind after kind in `KINDS` order and span after span, and what the draw
    counted, by kind and span: the windows drawn, those left out (D146) and those without a compressed form, and a
    shortfall."""
    train = context.splits["train"]
    anchors = Anchors(train["windows"])
    separations, step_s = context.separations, context.words.spec.step_s
    wanted = span_counts(settings)
    out = {kind: {span_key(span): [] for span in settings.spans_s} for kind in KINDS if settings.per_kind[kind] > 0}
    left = {kind: dict.fromkeys(spans, 0) for kind, spans in out.items()}
    alone = {kind: dict.fromkeys(spans, 0) for kind, spans in out.items()}
    for i in rng.permutation(len(train["windows"])):
        anchor = train["windows"][int(i)]
        code = anchor.scene.geometry.code
        for kind, by_span in out.items():
            short = [span for span in settings.spans_s if len(by_span[span_key(span)]) < wanted[kind][span_key(span)]]
            if not short:
                continue
            span = short[int(rng.integers(len(short)))]                  # the window's span: the round's numbers
            window = anchors.window_of(anchor, span)
            if kind == COMPRESSED and not window.joined:
                alone[kind][span_key(span)] += 1
                continue
            chosen = (Drawn(window, REAL_KIND) if kind == REAL_KIND else compressed(window, rng, settings.c_min)).window
            if left_out(chosen, separations[code], context.finals[code], step_s):
                left[kind][span_key(span)] += 1
            else:
                by_span[span_key(span)].append(chosen)
        if all(len(drawn) == wanted[kind][key] for kind, by_span in out.items() for key, drawn in by_span.items()):
            break
    return [w for by_span in out.values() for drawn in by_span.values() for w in drawn], {
        "drawn": {kind: {key: len(drawn) for key, drawn in by_span.items()} for kind, by_span in out.items()},
        "left_out": left, "no_later_aircraft": alone,
        "shortfall": {kind: short for kind, by_span in out.items()
                      if (short := {key: wanted[kind][key] - len(drawn) for key, drawn in by_span.items()
                                    if len(drawn) < wanted[kind][key]})}}


# ---- the speaking
def speak_batch(model: Prior, context: Context, windows: Sequence[Window], places: Sequence[int],
                settings: MultiSettings, round_: int, directory: Path, k: int) -> dict[str, Any]:
    """Batch ``k`` of a round (the windows at ``places``): its two passes under stage D's rules and loop (module
    docstring), its branch points D143's or found backward (`MultiSettings.branching`, D182: `post_branches.
    backward_round`), its informative groups written to ``groups_<k>.pt``, and its record (`post_train.speak_round`'s
    sums, of every commanded aircraft; backward, besides: each group's span and depth, the searches and those that found
    a decision point, the continuations flown to the judged tick only and those flown to their window's end, and the
    search's seconds, steps and rows; majority, besides: each group's kind and each majority point's span and depth;
    with `MINORITY_SIDE`, its informative groups' minority sides, in the order written).
    The majority point needs more than half of the ``continuations`` avoiding (D192: 5 of 8)."""
    train = context.splits["train"]
    batch = [windows[p] for p in places]
    common = dict(interval_s=context.interval_s, variant=context.variant, edges_reference=context.edges_reference,
                  faults=train["faults"], device=context.device, split="train", continuations=settings.continuations,
                  rules=Rules(**credit.rules(settings.seed, round_, context.interval_s, settings.branch_every_s)),
                  loop_options={**loop_options(context, settings.joint), "attention": settings.front_attention},
                  whole_windows=settings.joint)
    data = (model, context.start_loop("train", batch), batch, places, train["sentences"], train["flights"],
            context.geometries, context.finals, context.words)
    majority = settings.continuations // 2 + 1 if settings.branching == MAJORITY else None
    found = (branch_round(*data, seed=settings.seed, round_=round_, **common) if settings.branching == GRID else
             backward_round(*data, **common, majority=majority, search_points=settings.search_points,
                            judged_after_s=settings.judged_after_s, fine_point=settings.fine_point,
                            common_cut=settings.common_cut))
    informative = [g for g in found.groups if g.informative]
    save_batch_file(informative, directory / f"groups_{k}.pt")
    anchors = {w.signal_index for w in batch}              # a window's count is in each of its aircraft's ends
    searched = {} if settings.branching == GRID else {
        "group_depths_s": [[batch[0].span_s, depth] for depth in found.depths_s], "searches": found.searches,
        "searches_found": found.found, "continuations_to_event": found.to_event, "continuations_to_end": found.to_end,
        "search_s": found.search_s, "search_steps": found.search_steps, "search_rows": found.search_rows}
    if settings.branching == MAJORITY:
        searched |= {"group_kinds": found.kinds,
                     "majority_depths_s": [[batch[0].span_s, depth] for depth in found.majority_depths_s]}
    if settings.group_weight == MINORITY_SIDE:
        searched |= {"group_minorities": [g.minority for g in informative]}
    return {"groups": len(found.groups), "informative_groups": len(informative),
            "spoken_again": len(found.spoken_again), "differed": list(found.differed),
            "reward_sum": sum(r.reward for r in found.first),
            "faulty_steps": sum(r.faulty_steps for r in found.first if r.index in anchors),    # once a window
            "losses_reading_fault": sum(r.loss_reads_fault for r in found.first),
            "outcomes": dict(Counter(r.outcome for r in found.first)), **searched}


# ---- the selection readout
def readout_numbers(seed: int, place: int, member: int, draw: int = 0) -> np.random.Generator:
    """The numbers of commanded aircraft ``member`` of the readout's window ``place``: the same every round; draw 0 is
    the readout's, a later draw (a read of the windows again, as stage C's ceiling readout reads, C15) its own."""
    return np.random.default_rng([seed, 1 << 30, place, member] if draw == 0 else [seed, 1 << 30, place, member, draw])


def selection_windows(context: Context, settings: MultiSettings, split: str = "select") -> list[Window]:
    """The readout's windows (module docstring): of each airport, its anchors of ``split`` in an order drawn once with
    the seed; of each span, each anchor's real window of that span in that order kept unless it is left out (D146), up
    to ``select_per_airport``; by airport, then span, in the order of the anchors."""
    anchors = Anchors(context.splits[split]["windows"])
    rng = np.random.default_rng([settings.seed, 1 << 31])
    separations, step_s = context.separations, context.words.spec.step_s
    out = []
    for code in sorted(anchors.by_airport):
        items, order = anchors.by_airport[code], rng.permutation(len(anchors.by_airport[code]))
        for span in settings.spans_s:
            kept = []
            for i in order:
                if not left_out(anchors.window_of(items[int(i)], span), separations[code], context.finals[code],
                                step_s):
                    kept.append(int(i))
                if len(kept) == settings.select_per_airport:
                    break
            out += [anchors.window_of(items[i], span) for i in sorted(kept)]
    return out


def read_batch(model: Prior, context: Context, windows: Sequence[Window], places: Sequence[int],
               settings: MultiSettings, split: str, draw: int = 0
               ) -> list[tuple[list[Any], dict[str, int], int, dict[str, Any]]]:
    """A batch of the readout (the windows at ``places``): their first pass under stage D's loop, each aircraft with its
    own numbers (`readout_numbers` of ``draw``); for each window, its aircraft's ends (`WindowResult`, in the window's
    order), its steps with a loss of each pair and its steps judged (the states flown judged again, module docstring),
    and the time its aircraft took (`window_timing`)."""
    data = context.splits[split]
    batch = [windows[p] for p in places]
    loop, order, observed = context.start_loop(split, batch)(sorted(i for w in batch for i in w.signal_indices))
    window_loop = WindowLoop(model, loop, order, batch, data["sentences"], data["flights"], context.geometries,
                             context.finals, context.words, interval_s=context.interval_s,
                             variant=context.variant, edges_reference=context.edges_reference, faults=data["faults"],
                             observed=observed, device=context.device, **loop_options(context, settings.joint))
    numbers = [readout_numbers(settings.seed, places[int(window_loop.window_of[b])], int(window_loop.member_of[b]),
                               draw) for b in range(len(window_loop.order))]
    results = window_loop.run(numbers)
    out = []
    for window, rows in zip(batch, window_loop.members):
        ends = [results[b] for b in rows]
        losses, steps = window_losses_of(window, ends, window_loop.speaking.start, context)
        out.append((ends, losses, steps, window_timing(window, ends, window_loop.speaking.loop.params.cycle_s)))
    return out


def window_timing(window: Window, ends: Sequence[Any], cycle_s: float) -> dict[str, Any]:
    """The time a window's commanded aircraft took (`multi.timing`, §5 item 4, O18): each landing's delay against its
    record, the spacing at the threshold of successive landings in the loop and on the records, and the pairs landed in
    the recorded order (with the pairs)."""
    items = landed(window.commanded_all, [e.crossing if e.outcome == LANDED else None for e in ends], cycle_s)
    last_s = max([record.landing_s for record in window.commanded_all] + [item.loop_s for item in items])
    recorded = recorded_landings(window, last_s)
    return {"delays_s": delays(items), "spacing_s": loop_spacings(items, recorded),
            "record_spacing_s": record_spacings(window.commanded_all, recorded), "order": list(order(items))}


def window_losses_of(window: Window, ends: Sequence[Any], start: int, context: Any) -> tuple[dict[str, int], int]:
    """A window's steps with a loss of each pair (`multi.separation.PAIRS`) and its steps judged, its commanded
    aircraft at the states they flew (``ends``: their `WindowResult`s, in the window's order, ``start`` the Δ rows to a
    first predicted step), with the census's rule (`multi.census.window_losses`). As the loop judges it
    (`post_window_loop.WindowLoop._landed`, `_judged`): in a window of several commanded aircraft, one that landed is
    judged once more over its threshold, as a recorded one (it answers for nothing)."""
    code = window.scene.geometry.code
    count = WindowCount(airport=code, kind=kind_of(window), commanded=len(ends), recorded_at_first_steps=[],
                        left_out_by=[])
    window_losses(window, loop_positions(window, ends, start, context.words), context.separations[code],
                  context.finals[code], context.words.spec.step_s, count)
    return dict(count.loss_steps), count.steps


def window_line(place: int, window: Window, ends: Sequence[Any], losses: Mapping[str, int]) -> dict[str, Any]:
    """The readout's line of one window (post-training D176 (2): what a window list reads of stage D): its place among
    the readout's windows, its airport, its identity (`post.window_lists.identity_of`), span and kind, and each
    commanded aircraft's key, outcome, reward and go-arounds in the window's order, with its steps of loss by pair."""
    return {"place": place, "airport": window.scene.geometry.code, "identity": identity_of("D", window),
            "span": span_key(window.span_s), "kind": kind_of(window),
            "aircraft": [record.key for record in window.commanded_all], "outcomes": [e.outcome for e in ends],
            "rewards": [e.reward for e in ends], "go_arounds": [e.go_arounds for e in ends], "loss_steps": dict(losses)}


def selection_readout(model: Prior, context: Context, windows: Sequence[Window], settings: MultiSettings,
                      speakers: Speakers | None, *, stage: Stage, split: str = "select",
                      lines: list[dict[str, Any]] | None = None, turns: dict[str, Any] | None = None
                      ) -> dict[str, Any]:
    """The readout of ``windows`` of ``split`` (module docstring; `multi_validation`: the val days): its batches read
    here (``stage``'s `Stage.read_batch`) or by ``speakers`` (refused unless they run ``stage``), summed by airport,
    span (`span_key`) and kind, each with its ``all``; an airport's ``reward_mean`` is its ``all`` of both. ``lines``,
    when given, gets each window's line (`window_line`) in place order; ``turns``, when given, the turns of every
    commanded aircraft (`turns_readout`, post-training D214), apart from the readout."""
    places, read = selection_read(model, context, windows, settings, speakers, stage=stage, split=split)
    if turns is not None:
        turns.update(turns_readout((windows[place].scene.geometry.code, end)
                                   for batch, parts in zip(places, read, strict=True)
                                   for place, (ends, *_) in zip(batch, parts, strict=True) for end in ends))
    return selection_summed(windows, places, read, lines)


def selection_read(model: Prior, context: Context, windows: Sequence[Window], settings: MultiSettings,
                   speakers: Speakers | None, *, stage: Stage, split: str) -> tuple[list[list[int]], list[Any]]:
    """The batches of the readout of ``windows`` (`selection_readout`): their places and, for each, what
    ``stage``'s `Stage.read_batch` gives, read here or by ``speakers`` (refused unless they run ``stage``)."""
    if speakers is not None and speakers.stage is not stage:
        raise ValueError("the speaking workers run another stage's parts than the one given (Speakers(..., stage=))")
    places = stage.batches(windows, settings)
    read = (speakers.read(model, windows, places, split) if speakers is not None else
            [stage.read_batch(model, context, windows, p, settings, split) for p in places])
    return places, read


def selection_summed(windows: Sequence[Window], places: Sequence[Sequence[int]], read: Sequence[Any],
                     lines: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The readout (`selection_readout`) of the batches ``read`` at ``places`` of ``windows``; ``lines``, when given,
    gets each window's line (`window_line`) in place order."""
    counted: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    read_lines = []
    for batch, parts in zip(places, read, strict=True):
        for place, (ends, losses, steps, timing) in zip(batch, parts, strict=True):
            window = windows[place]
            if lines is not None:
                read_lines.append(window_line(place, window, ends, losses))
            for span, kind in product((span_key(window.span_s), "all"), (kind_of(window), "all")):
                c = counted.setdefault(window.scene.geometry.code, {}).setdefault(span, {}).setdefault(kind, _empty())
                c["windows"] += 1
                c["aircraft"] += len(ends)
                c["reward_sum"] += sum(e.reward for e in ends)
                c["outcomes"].update(e.outcome for e in ends)
                c["go_arounds"] += sum(e.go_arounds for e in ends)
                c["silent"] += sum(e.outcome == LOST_SEPARATION for e in ends)
                c["speed_mask_rows"] += sum(e.speed_mask_rows for e in ends)
                c["faulty_steps"] += ends[0].faulty_steps
                c["losses_reading_fault"] += sum(e.loss_reads_fault for e in ends)
                c["steps_judged"] += steps
                for pair in PAIRS:
                    c["loss_windows"][pair] += losses[pair] > 0
                    c["loss_steps"][pair] += losses[pair]
                for key in ("delays_s", "spacing_s", "record_spacing_s"):
                    c[key] += timing[key]
                c["order"] = [c["order"][0] + timing["order"][0], c["order"][1] + timing["order"][1]]
    if lines is not None:
        lines += sorted(read_lines, key=lambda line: line["place"])
    out: dict[str, Any] = {}
    for code, spans in sorted(counted.items()):
        out[code] = {span: {kind: _summed(c) for kind, c in kinds.items()} for span, kinds in spans.items()}
        out[code]["reward_mean"] = out[code]["all"]["all"]["reward_mean"]    # what the campaign's log prints
    return out


def _summed(c: dict[str, Any]) -> dict[str, Any]:
    """A cell of the readout (`selection_readout`) with its means and quantiles."""
    return {**c, "outcomes": dict(c["outcomes"]), "reward_mean": c["reward_sum"] / c["aircraft"],
            "window_reward_mean": c["reward_sum"] / c["windows"], "delays_s": quantiles(c["delays_s"], (50, 90)),
            "spacing_s": quantiles(c["spacing_s"], (1, 5, 50)),
            "record_spacing_s": quantiles(c["record_spacing_s"], (1, 5, 50)),
            "order": {"same": c["order"][0], "pairs": c["order"][1]}}


def _empty() -> dict[str, Any]:
    return {"windows": 0, "aircraft": 0, "reward_sum": 0.0, "outcomes": Counter(), "go_arounds": 0, "silent": 0,
            "speed_mask_rows": 0, "faulty_steps": 0, "losses_reading_fault": 0, "steps_judged": 0,
            "loss_windows": dict.fromkeys(PAIRS, 0), "loss_steps": dict.fromkeys(PAIRS, 0), "delays_s": [],
            "spacing_s": [], "record_spacing_s": [], "order": [0, 0]}


def multi_settings_of(record: dict[str, Any]) -> MultiSettings:
    """A campaign's settings as its ``campaign.json`` records them, refused unless it is a campaign of stage D whose
    format is its settings' version's (D212)."""
    if record["schema"] not in (MULTI_CAMPAIGN_SCHEMA, MULTI_CAMPAIGN_SCHEMA_V2):
        raise ValueError(f"a {record['schema']} campaign, not {MULTI_CAMPAIGN_SCHEMA} or {MULTI_CAMPAIGN_SCHEMA_V2}")
    settings = MultiSettings(**record["inputs"]["settings"])
    if record["schema"] != settings.campaign_schema:
        raise ValueError(f"a {record['schema']} campaign whose settings are {settings.campaign_schema}'s")
    return settings


def round_model(context: Context, settings: MultiSettings, out: Path, round_: int) -> Prior:
    """The model of stage D's campaign ``out`` after round ``round_`` (its checkpoint), in eval mode, refused by name
    unless its identity is this campaign's (§7 row 3): stage D's format, the start's identity on this base, today's
    procedure masks, the token part, the seed, the spans and their counts, the kinds, c_min and that round."""
    model, _ = stage_d().start_model(context, settings)
    state = torch.load(out / f"round_{round_}" / "checkpoint.pt", weights_only=False, map_location=context.device)
    held = state["identity"]
    expected = {"schema": settings.checkpoint_schema, "procedure_masks": PROCEDURE_MASKS,
                "token_part": {"schema": TOKENS_SCHEMA, "features": list(PART_FEATURES)}, "seed": settings.seed,
                "spans_s": list(settings.spans_s), "per_kind": settings.per_kind, "per_span": span_counts(settings),
                "c_min": settings.c_min, **joint_identity(settings)}
    start = start_identity(context, settings)
    if ({k: held[k] for k in expected} != expected or held["start"] != start
            or start_base(held["start"]) != context.base_identity or len(held["rounds"]) != round_ + 1):
        raise ValueError(f"{out}/round_{round_}: a checkpoint of another start, base, masks, token part, settings or "
                         f"round than this campaign's")
    model.load_state_dict(state["model"])
    return model.eval()


def with_margin(measure: Mapping[str, Any] | None, margin: float) -> dict[str, Any] | None:
    """A measure of `Speakers.measure` or `post_train.pass_memory_of` with each peak (its own, or its host's and its
    GPU's) multiplied by ``margin`` (D179) — what it holds now unchanged; None stays None."""
    if measure is None:
        return None
    if "peak" in measure:
        return {**measure, "peak": int(measure["peak"] * margin)}
    return {**measure, **{name: None if measure[name] is None else {**measure[name],
                                                                    "peak": int(measure[name]["peak"] * margin)}
                          for name in ("host", "gpu")}}


def _profiled(profile: Path, settings: MultiSettings, inputs: dict[str, Any], device: torch.device,
              speak_device: torch.device) -> dict[str, Any]:
    """Stage D's profile ``profile``'s measure of the speaking workers (`multi_profile`), refused by name unless it is of
    these inputs (`PROFILED_INPUTS`, as this checkout reads them) and these settings (`PROFILED_SETTINGS`), its pass on
    this kind of device and its worker measured on ``speak_device``'s kind."""
    from ts_transformer.experiments.training_export import this_checkout

    record = json.loads((profile / "profile.json").read_text(encoding="utf-8"))
    if record["schema"] != MULTI_PROFILE_SCHEMA:
        raise SystemExit(f"{profile} is a {record['schema']} record, not {MULTI_PROFILE_SCHEMA}")
    if "workers" not in record:
        raise SystemExit(f"{profile}: no measure of the speaking workers (a profile with --speak-workers 2 or more)")
    held = record["inputs"]
    other = [key for key in PROFILED_INPUTS if this_checkout(held[key]) != this_checkout(inputs[key])]
    profiled = asdict(MultiSettings(**held["settings"]))   # a setting added after the profile: its default (D182)
    other += [key for key in PROFILED_SETTINGS if profiled[key] != asdict(settings)[key]]
    measure = record["workers"]
    if (measure["pass"] is None) != (device.type != "cuda"):
        other.append("device")
    if torch.device(measure["speak_device"]).type != speak_device.type:
        other.append("speaking device")
    if other:
        raise SystemExit(f"{profile} is a profile of other {', '.join(other)} than this campaign's: profile these "
                         f"(multi_profile)")
    return measure


def profiled_fit(profile: Path, settings: MultiSettings, inputs: dict[str, Any], workers: int,
                 device: torch.device, speak_device: torch.device, cpu_profile: Path | None = None,
                 cpu_workers: int = 0) -> dict[str, Any]:
    """The speaking workers' memory read from stage D's profile (the user, 2026-10-07: measured once, never before each
    run): the profile's measure of one worker and of the pass (`multi_profile`, `_profiled`), each peak times its
    recorded margin (D179), and the memory free now (`post_train.available_memory`, before any worker or this process
    uses the GPU), ``workers`` workers refused by name where they do not fit (`post_train.workers_fit` with nothing
    held yet: a worker on the CPU counts the host's memory only, D180), and each worker's GPU budget (the GPU free now
    less what this process holds beside its pass, in equal shares: `Speakers`' ``gpu_budget``; None for workers off the
    GPU). ``cpu_workers`` CPU workers besides (D191): their measure from ``cpu_profile`` (a profile of these inputs and
    settings with ``--speak-device cpu``), counted in the host's budget with the others'. A GPU batch past its worker's
    share is spoken again after the others at fewer at once (D191, the user, 2026-10-10: the batches past their share
    go last, and more workers; `Speakers`' ``deferred``): at half the workers, a quarter, … one, each batch's budget
    the GPU less this process's hold and what each idle GPU worker holds, shared by the batches at once. So the GPU
    refuses only workers that could not hold what they hold between batches; D179's margin is the pass's and the GPU
    workers' host part (a pass short of memory has no second try). The CPU workers in the same way (D193 (4)): each
    has the host's part left by the GPU workers' (at their peak with its margin) in equal shares (`Speakers`'
    ``host_budget``), a CPU batch past its share goes to the CPU's levels (``cpu_deferred``), and the fit counts a CPU
    worker at what it holds between batches, refused only where the last level cannot hold the CPU profile's host peak
    with its margin."""
    measure = _profiled(profile, settings, inputs, device, speak_device)
    cpu = _profiled(cpu_profile, settings, inputs, device, torch.device("cpu")) if cpu_workers else None
    measured, passed = (with_margin(measure[k], measure["margin"]) for k in ("measured", "pass"))
    gpu = measure["measured"]["gpu"]
    if gpu is not None:                     # a batch past its share goes to the levels (D191): a worker's floor its hold
        measured = {**measured, "gpu": {**gpu, "peak": gpu["now"]}}
    cpu_measured = None
    if cpu is not None:                     # a CPU batch past its share goes to the levels (D193 (4)): its floor its hold
        host = cpu["measured"]["host"]
        cpu_measured = {**cpu["measured"], "host": {**host, "peak": host["now"]}}
    available = available_memory(device)
    short = workers_fit(workers, measured, passed, available, held_now=False, cpu_workers=cpu_workers,
                        cpu_measured=cpu_measured)
    if short:
        raise SystemExit("the speaking workers do not fit (O15, from the profile): " + "; ".join(short)
                         + " — start fewer (--speak-workers, --cpu-workers)")
    budget, deferred = None, []
    if speak_device.type == "cuda":
        reader = measure["measured"]["held"]["reader_model"]
        budget, deferred = _levels(available["gpu"] - passed["now"], workers, gpu["now"] + reader,
                                   int(gpu["peak"] * measure["margin"]) + reader, "GPU", "--speak-workers")
    host_budget, cpu_deferred = None, []
    if cpu is not None:
        held = cpu["measured"]["held"]
        besides = held["reader_model"] + held["series"]                         # a CPU worker's, beside its batch
        gpu_host = workers * (measured["host"]["peak"] + measure["measured"]["held"]["series"])
        host_budget, cpu_deferred = _levels(available["host"] - gpu_host, cpu_workers, host["now"] + besides,
                                            int(host["peak"] * cpu["margin"]) + besides, "host", "--cpu-workers")
    return {"profile": str(profile), "speak_workers": workers, "speak_device": str(speak_device),
            "cpu_profile": None if cpu_profile is None else str(cpu_profile), "cpu_workers": cpu_workers,
            "margin": measure["margin"], "available": available, "gpu_budget": budget, "deferred": deferred,
            "host_budget": host_budget, "cpu_deferred": cpu_deferred}


def _levels(shared: int, workers: int, hold: int, peak: int, memory: str, option: str) -> tuple[int, list[list[int]]]:
    """``workers`` workers' share of ``shared`` bytes, and the levels a batch past its share is spoken again at (D191):
    half the workers at once, a quarter, … one, each batch's budget ``shared`` less what each idle worker holds
    (``hold``), in equal parts. Refused by name where the last level cannot hold a batch at ``peak`` (the profile's
    peak with its margin and what a worker holds beside its batch; ``memory`` names it, ``option`` the workers)."""
    levels = []
    at_once = workers // 2
    while at_once >= 1:
        levels.append([at_once, (shared - (workers - at_once) * hold) // at_once])
        at_once = at_once // 2 if at_once > 1 else 0
    alone = levels[-1][1] if levels else shared // workers
    if alone < peak:
        raise SystemExit(f"the speaking workers do not fit (O15, from the profile): a batch at the profile's {memory} "
                         f"peak ({peak / 2**30:.2f} GiB with its margin) has {alone / 2**30:.2f} GiB at the last "
                         f"level — start fewer ({option})")
    return shared // workers, levels


def of_stage_d(campaign: Path) -> bool:
    """Whether ``campaign`` is a campaign of stage D (D218: a start from one of its rounds), by its record's format."""
    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    return record["schema"] in (MULTI_CAMPAIGN_SCHEMA, MULTI_CAMPAIGN_SCHEMA_V2)


def multi_source_campaign(campaign: Path, round_: int, *, formal: bool) -> MultiSettings:
    """D218: the settings of stage D's campaign a start comes from, refused by name (as `post_train.source_campaign`'s
    rules) unless its round ``round_`` is done and, for a formal campaign (``formal``), it is not a smoke one."""
    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    settings = multi_settings_of(record)
    if formal and record["inputs"]["smoke"]:
        raise ValueError(f"{campaign} is a smoke campaign: a formal campaign does not start from it")
    if not 0 <= round_ < done_rounds(campaign):
        raise ValueError(f"{campaign} holds the checkpoints of rounds 0–{done_rounds(campaign) - 1}, not {round_}")
    return settings


def multi_start_of(campaign: Path, round_: int, *, formal: bool) -> dict[str, Any]:
    """D218: the setting of a start from round ``round_`` of stage D's campaign ``campaign``, as `post_train.start_of`'s
    (the campaign repository-relative, the round, its checkpoint's sha256); the source's rules
    (`multi_source_campaign`) refused by name first."""
    from ts_transformer.experiments.training_export import this_checkout

    multi_source_campaign(campaign, round_, formal=formal)
    return {"campaign": repo_relative(this_checkout(campaign)), "round": round_,
            "checkpoint_sha256": file_sha256(campaign / f"round_{round_}" / "checkpoint.pt")}


def require_finished(campaign: Path) -> None:
    """D166 item 19: a formal start of stage D comes from a campaign of stage C (or D, D218) that has done every round
    (as `post_validation`'s formal read: its round is chosen on every round's selection readout, D7), refused by name
    otherwise."""
    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    rounds = (multi_settings_of(record) if of_stage_d(campaign) else settings_of(record)).rounds
    if done_rounds(campaign) < rounds:
        raise ValueError(f"{campaign} has done {done_rounds(campaign)} of its {rounds} rounds: a formal campaign of "
                         f"stage D starts from a campaign that has done every round (D166 item 19)")


def start_identity(context: Context, settings: MultiSettings) -> dict[str, Any]:
    """The identity of a campaign's start in its checkpoints (§7 row 3): from the base (D200), the base's identity and
    the traffic shape; from a round of stage C (D164) or of stage D (D218), that round's checkpoint identity."""
    if settings.start is None:
        return {"base": context.base_identity,
                "traffic": TrafficConfig(settings.traffic_hidden, settings.traffic_heads).to_dict()}
    return torch.load(start_checkpoint(settings.start), weights_only=False, map_location="cpu")["identity"]


def start_base(identity: Mapping[str, Any]) -> Any:
    """The base's identity in a start's identity (`start_identity`): its own (the base's, D200; a round of stage C's,
    D164), or for a round of stage D (D218, whose identity holds its own start's) its start's."""
    return identity["base"] if "base" in identity else start_base(identity["start"])


def joint_identity(settings: MultiSettings) -> dict[str, Any]:
    """Version 2's part of a checkpoint's identity (D212): the module's shape; none for version 1."""
    return {"joint": JointConfig(settings.joint_heads).to_dict()} if settings.joint else {}


def segment_rows_of(settings: MultiSettings, interval_s: float) -> int | None:
    """D200: the rows from a branch point to the next (`MultiSettings.branch_every_s` in Δ rows) where the campaign
    counts only them (`MultiSettings.segment_only`); None: up to the event."""
    return branch_rows(interval_s, settings.branch_every_s) if settings.segment_only else None


# ---- the stage
def stage_d() -> Stage:
    """Stage D's parts of a round (module docstring)."""
    #: the start's identity, read once (the bytes were checked when the campaign started, `round_start`)
    started: dict[str, Any] = {}

    def shaped(context: Context, settings: MultiSettings) -> Prior:
        """The model of stage D's shape, with the token part (its weights at zero): from the base (D200), the base
        with zero-output traffic modules of the campaign's seed and traffic shape (`post_train.traffic_model`); from a
        round of stage C, stage C's start model of the source campaign's settings."""
        if settings.start is None:
            model = traffic_model(context, settings.seed, TrafficConfig(settings.traffic_hidden, settings.traffic_heads))
        elif of_stage_d(REPO_ROOT / settings.start["campaign"]):         # D218: the source's own shape
            source_d = multi_source_campaign(REPO_ROOT / settings.start["campaign"], settings.start["round"],
                                             formal=context.formal)
            if source_d.joint_heads != settings.joint_heads:
                raise ValueError(f"{settings.start['campaign']}: its joint_heads {source_d.joint_heads}, not this "
                                 f"campaign's {settings.joint_heads} (a start from stage D keeps its version, D218)")
            return shaped(context, source_d)
        else:
            source = settings_of(source_campaign(REPO_ROOT / settings.start["campaign"], settings.start["round"],
                                                 formal=context.formal))
            model, _ = start_model(context, source)
        add_token_part(model, TokenPart.width, TOKENS_SCHEMA)
        if settings.joint:                                            # D212: version 2's module, its output at 0
            add_joint_attention(model, JointConfig(settings.joint_heads))
        return model.to(context.device).eval()

    def optimizer_of(model: Prior, settings: MultiSettings) -> torch.optim.Optimizer:
        return torch.optim.AdamW(parameter_groups(model, settings.prior_lr, settings.traffic_lr),
                                 weight_decay=settings.weight_decay)

    def shape(context: Context, settings: MultiSettings) -> tuple[Prior, torch.optim.Optimizer]:
        model = shaped(context, settings)
        return model, optimizer_of(model, settings)

    def start(context: Context, settings: MultiSettings) -> tuple[Prior, torch.optim.Optimizer]:
        if settings.start is None:                                     # D200: the start is the shaped base itself
            return shape(context, settings)
        campaign = REPO_ROOT / settings.start["campaign"]
        if of_stage_d(campaign):                                       # D218: as `post_train.round_start`'s rules
            if file_sha256(start_checkpoint(settings.start)) != settings.start["checkpoint_sha256"]:
                raise ValueError(f"{start_checkpoint(settings.start)}: not the checkpoint the campaign started from "
                                 f"(its sha256 is not the recorded one)")
            source_d = multi_source_campaign(campaign, settings.start["round"], formal=context.formal)
            if source_d.seed == settings.seed:
                raise ValueError(f"{campaign}: its seed is {settings.seed}; a campaign from a round takes another seed, "
                                 f"so that its rounds draw new windows (D162)")
            source = round_model(context, source_d, campaign, settings.start["round"])
            identity = start_identity(context, settings)
        else:
            source, identity = round_start(context, settings.start, settings.seed)    # D164: refused by name
        if context.formal:
            require_finished(REPO_ROOT / settings.start["campaign"])
        model = shaped(context, settings)
        model.load_state_dict({**model.state_dict(), **source.state_dict()})       # the round's, the token part at 0
        started["identity"] = identity
        return model, optimizer_of(model, settings)

    def identity(context: Context, settings: MultiSettings, out: Path, rounds: int) -> dict[str, Any]:
        if "identity" not in started:                                 # a resumed campaign, or a start from the base
            started["identity"] = start_identity(context, settings)
        return {"schema": settings.checkpoint_schema, "start": started["identity"], "procedure_masks": PROCEDURE_MASKS,
                "token_part": {"schema": TOKENS_SCHEMA, "features": list(PART_FEATURES)}, "seed": settings.seed,
                "spans_s": list(settings.spans_s), "per_kind": settings.per_kind, "per_span": span_counts(settings),
                "c_min": settings.c_min, **joint_identity(settings),
                "rounds": [json.loads((out / f"round_{r}" / "round.json").read_text(encoding="utf-8"))["windows"]
                           for r in range(rounds)]}

    return Stage(start_model=shape, start=start, identity=identity,
                 draw=lambda context, settings, round_: draw_windows(context, settings,
                                                                     np.random.default_rng([settings.seed, round_])),
                 speak=lambda *args, stage, launch: speak_round(*args, stage=stage, launch=launch),
                 selection=lambda context, settings, split: selection_windows(context, settings, split),
                 readout=selection_readout, record=window_record, speak_batch=speak_batch, read_batch=read_batch,
                 part_width=TokenPart.width, batches=span_batches,
                 train=lambda model, context, optimizer, directory, settings, rng, *, part_width, round_, companion:
                 train_pass(model, context, optimizer, directory, settings, rng, part_width=part_width,
                            segment_rows=segment_rows_of(settings, context.interval_s),
                            data_weight=settings.data_weight, group_weight=settings.group_weight))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The inputs and settings of a campaign of stage D (`settings_from`), which its profile (`multi_profile`) takes
    too."""
    parser.add_argument("--prior", type=Path, required=True, help="the base: a prior_train run's directory")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the directory of the artefact's executor spec")
    parser.add_argument("--windows", type=Path, required=True,
                        help="stage C's census (post_windows), whose conformance/edges.npz is the edge reference (D104)")
    parser.add_argument("--procedure-root", type=Path, default=DEFAULT_PROCEDURE_ROOT)
    parser.add_argument("--start-campaign", type=Path,
                        help="stage C's campaign (post_train, D164) or stage D's (multi_train, D218), with "
                             "--start-round; not given: the base (D200)")
    parser.add_argument("--start-round", type=int, help="its chosen round (§5 item 1, D164)")
    parser.add_argument("--traffic-hidden", type=int, help="D200: a start from the base: the traffic shape's hidden width")
    parser.add_argument("--traffic-heads", type=int, help="D200: a start from the base: the traffic shape's heads")
    parser.add_argument("--rounds", type=int, required=True)
    for kind in KINDS:
        parser.add_argument(f"--windows-{kind}", type=int, required=True, help=f"the {kind} windows of a round (O16)")
    parser.add_argument("--spans-s", type=float, nargs="+", required=True, help="the spans L, s, increasing (D146)")
    parser.add_argument("--c-min", type=float, required=True, help="the least c of a compressed window (O16)")
    parser.add_argument("--batch-rows", type=int, nargs="+", required=True,
                        help="the most commanded aircraft flown in one batch of each span, in the order of --spans-s")
    parser.add_argument("--continuations", type=int, default=CONTINUATIONS, help="D94, D143: K")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prior-lr", type=float, required=True)
    parser.add_argument("--traffic-lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--update-groups", type=int, required=True, help="the branch groups of an update")
    parser.add_argument("--data-sentences", type=int, required=True, help="the data term's sentences of an update")
    parser.add_argument("--select-per-airport", type=int, required=True)
    parser.add_argument("--branching", choices=BRANCHINGS, default=GRID,
                        help="D182, D192: the branch points, D143's grid (default) or found back from each event "
                             "with point B (backward) or the majority point (majority)")
    parser.add_argument("--branch-every-s", type=float, default=MultiSettings.branch_every_s,
                        help="D200: the interval of the grid's branch points, s (default: D143's)")
    parser.add_argument("--segment-only", action="store_true",
                        help="D200: a group's advantage only up to the next branch point (default: up to the event)")
    parser.add_argument("--search-points", type=int, default=MultiSettings.search_points,
                        help="D217: the backward search's candidate points at most, 32 s apart (default: D182's)")
    parser.add_argument("--judged-after-s", type=float, default=MultiSettings.judged_after_s,
                        help="D217: a point's continuations judged this long after the event, s (default: D182's)")
    parser.add_argument("--fine-point", action=argparse.BooleanOptionalAction, default=MultiSettings.fine_point,
                        help="D217: the point 16 s after the decision point (D182 (5); default: on)")
    parser.add_argument("--common-cut", action="store_true",
                        help="D217: every sentence of a group counted only up to the judged tick (default: each up to "
                             "its own event)")
    parser.add_argument("--front-attention", choices=FRONT_ATTENTIONS, default=COPIED,
                        help="D190 (2): how a window's copies read the front they hold once (copied: bit for bit; "
                             "merged: no copy)")
    parser.add_argument("--cpu-batches", type=int, default=MultiSettings.cpu_batches,
                        help="D191: of every --cpu-period batches by index, the first this many on the CPU workers")
    parser.add_argument("--cpu-period", type=int, default=MultiSettings.cpu_period, help="D191: the rule's period")
    parser.add_argument("--batch-squares", type=int, nargs="+", default=[],     # mirror of the field's default factory
                        help="the most sum of commanded aircraft squared of a batch of each span (default: no "
                             "bound)")
    parser.add_argument("--gpu-squares", type=int, default=MultiSettings.gpu_squares,
                        help="a batch whose sum of commanded aircraft squared reaches this is spoken on the speak device "
                             "(default 0: off)")
    parser.add_argument("--data-weight", type=float, default=MultiSettings.data_weight,
                        help="R2: the data term's weight in each update (default: post.loss.DATA_WEIGHT)")
    parser.add_argument("--group-weight", choices=GROUP_WEIGHTS, default=MultiSettings.group_weight,
                        help="D195: each group's weight in the pass (default: every group's advantages as they are)")
    parser.add_argument("--joint-heads", type=int, default=MultiSettings.joint_heads,
                        help="D212: version 2, the heads of the module that reads the other commanded aircraft of a "
                             "window at the same tick (default: none, version 1)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speak-workers", type=int, default=1)
    parser.add_argument("--speak-device",
                        help="the speaking workers' device (default: --device); e.g. cpu beside a pass on cuda (D180)")


def start_from(args: argparse.Namespace, *, formal: bool) -> dict[str, Any] | None:
    """The start of `add_arguments`' ``args``: None (the base, D200) without ``--start-campaign``; else the record of
    that round (`post_train.start_of`, or `multi_start_of` for a campaign of stage D, D218; the source's rules refused
    by name, ``formal`` as there). ``--start-campaign`` and
    ``--start-round`` go together, the traffic shape with the base only (refused by name)."""
    if (args.start_campaign is None) != (args.start_round is None):
        raise ValueError("--start-campaign and --start-round go together (a round of stage C or D; neither: the base)")
    given = (args.traffic_hidden, args.traffic_heads)
    if (args.start_campaign is None and None in given) or (args.start_campaign is not None and given != (None, None)):
        raise ValueError("--traffic-hidden and --traffic-heads go with a start from the base (D200); a start from a "
                         "round of stage C takes its campaign's traffic shape (D164)")
    if args.start_campaign is None:
        return None
    source = args.start_campaign if args.start_campaign.is_absolute() else REPO_ROOT / args.start_campaign
    if of_stage_d(source):                                                    # D218
        return multi_start_of(source, args.start_round, formal=formal)
    return start_of(source, args.start_round, formal=formal)


def settings_from(args: argparse.Namespace, start: dict[str, Any] | None) -> MultiSettings:
    """The settings of `add_arguments`' ``args`` with the start ``start`` (`start_from`)."""
    return MultiSettings(args.rounds, {kind: getattr(args, f"windows_{kind}") for kind in KINDS}, args.spans_s,
                         args.c_min, args.batch_rows, args.continuations, args.seed, args.prior_lr, args.traffic_lr,
                         args.weight_decay, args.update_groups, args.data_sentences, args.select_per_airport, start,
                         args.branching, args.front_attention, args.cpu_batches, args.cpu_period, args.data_weight,
                         args.group_weight, args.batch_squares, args.gpu_squares, args.traffic_hidden,
                         args.traffic_heads, args.branch_every_s, args.segment_only, args.joint_heads,
                         search_points=args.search_points, judged_after_s=args.judged_after_s,
                         fine_point=args.fine_point, common_cut=args.common_cut)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    add_arguments(parser)
    parser.add_argument("--out", type=Path, required=True, help="the campaign's directory (new, or a resume)")
    parser.add_argument("--profile", type=Path,
                        help="stage D's profile (multi_profile) of these inputs and settings, which the speaking workers "
                             "are sized from: with speaking workers (--speak-workers above 1, or --cpu-workers)")
    parser.add_argument("--cpu-workers", type=int, default=0,
                        help="D191: speaking workers on the CPU besides, for the batches --cpu-batches gives the CPU")
    parser.add_argument("--cpu-profile", type=Path,
                        help="D191: the profile of a CPU worker (multi_profile --speak-device cpu) of these inputs and "
                             "settings: with --cpu-workers")
    parser.add_argument("--executor-device", choices=EXECUTOR_DEVICES, default="speak",
                        help="S1: where the executor of each loop flies: speak (on the speaker's device, the default) "
                             "or cpu (the speaker stays on its device)")
    parser.add_argument("--restart-round", action="store_true",
                        help="D185: a resume moves the round left half done aside and speaks it whole (default: its "
                             "done batches kept; on changed code, keep them only after a behaviour check)")
    parser.add_argument("--smoke", action="store_true", help="SMOKE: a tree with changes too; no result")
    args = parser.parse_args(argv)
    if args.speak_workers < 1:
        parser.error("--speak-workers is at least 1")
    refused = speak_device_refused(args.speak_device, args.speak_workers + args.cpu_workers, args.device)
    if refused is not None:
        parser.error(refused)
    speaking = args.speak_workers > 1 or args.cpu_workers > 0
    if speaking != (args.profile is not None):
        parser.error("--profile goes with speaking workers (--speak-workers above 1, or --cpu-workers; they are sized "
                     "from it)")
    if (args.cpu_workers > 0) != (args.cpu_profile is not None):
        parser.error("--cpu-profile goes with --cpu-workers (the CPU workers are sized from it)")
    try:
        start = start_from(args, formal=not args.smoke)                               # D164, D200: the start's setting
        if start is not None and not args.smoke:
            require_finished(REPO_ROOT / start["campaign"])                           # D166 item 19
        settings = settings_from(args, start)
    except ValueError as refused:
        parser.error(str(refused))
    prior_dir, instructions, executor, census, procedure_root, out = (
        p if p.is_absolute() else REPO_ROOT / p for p in (args.prior, args.instructions, args.executor, args.windows,
                                                          args.procedure_root, args.out))
    refused = cpu_workers_refused(args.cpu_workers, settings.cpu_batches,
                                  torch.device(args.speak_device or args.device))           # D191
    if refused is not None:
        parser.error(refused)
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("a campaign that is not a smoke runs on a clean checkout")
    _, opened, _ = require_conforming_closed_loop(instructions, executor)          # D69: the checks run here (D73)
    edges_reference = census / "conformance" / "edges.npz"
    checked_edges(edges_reference)                                                 # D104
    device = torch.device(args.device)
    context = open_context(prior_dir, instructions, executor, edges_reference,
                           torch.device("cpu") if speaking else device, procedure_root,
                           formal=not args.smoke)
    if args.executor_device == "cpu":                # S1: set before the fork, so each worker gets it
        context = replace(context, executor_device=torch.device("cpu"))
    try:                                            # D200, as stage C's D170: refused before the workers
        branch_rows(context.interval_s, settings.branch_every_s)
    except ValueError as refused:
        parser.error(f"--branch-every-s: {refused}")
    stage = stage_d()
    stage.start(context, settings)                  # refused by name before the workers and campaign.json (D164)
    inputs = {"prior": str(prior_dir), "instructions": str(instructions), "executor": str(executor),
              "windows": str(census), "procedure_root": str(procedure_root), "settings": asdict(settings),
              "smoke": args.smoke}
    speakers = None
    if speaking:                                    # O15 from the profiles, before any process uses the GPU
        profile, cpu_profile = (None if p is None else p if p.is_absolute() else REPO_ROOT / p
                                for p in (args.profile, args.cpu_profile))
        speak_device = torch.device(args.speak_device or args.device)
        fit = profiled_fit(profile, settings, inputs, args.speak_workers, device, speak_device, cpu_profile,
                           args.cpu_workers)
        print(json.dumps({"profiled_fit": fit}), flush=True)
        speakers = Speakers(context, settings, args.speak_workers, speak_device, stage=stage,
                            gpu_budget=fit["gpu_budget"], cpu_workers=args.cpu_workers, deferred=fit["deferred"],
                            host_budget=fit["host_budget"], cpu_deferred=fit["cpu_deferred"])
    context = replace(context, device=device, base=context.base.to(device).eval())
    try:
        open_campaign(out, inputs, git, opened["checks"], schema=settings.campaign_schema, reader=MULTI_CLAIM_READER,
                      settings_type=MultiSettings, restart_round=args.restart_round)
        run_campaign(out, settings, context, speakers, stage=stage)
    finally:
        if speakers is not None:
            speakers.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
