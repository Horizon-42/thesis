"""MC4: stage D's campaign, version 1 (multi-aircraft control §3–§5, §7, §11 MC4; D141–D148, D152) — rounds of branch
training in windows in which every arrival of a span L is commanded, from a chosen round of stage C.

It is stage C's campaign (`post_train.run_campaign`, the round's skeleton, post-training §9 item 11) with stage D's
parts (`stage_d`):

- **The start** (§5 item 1, D152, D164): the chosen round of stage C's campaign (``--start-campaign``,
  ``--start-round``), recorded in the settings as ``{campaign, round, checkpoint_sha256}`` (`post_train.start_of`) and
  opened by post-training §9 item 12's one function (`post_train.round_start`: the bytes, the identity of this base,
  masks, stage C's traffic shape and that round, a formal campaign from a formal one, a seed other than the source's,
  each refused by name), with stage D's token part added to its traffic attention (`multi.tokens`,
  `multi-commanded-tokens-v1`, its projection at zero: the start says what the round says) and a new optimizer. The pull
  is to the base, as in every stage of the post-training. A resume and a speaking worker load a state into the model of
  that shape (stage C's start model of the source campaign's settings, with the token part).
- **The draw of a round** (D146, `draw_windows`): the train days' real windows in a permutation by the seed and the
  round, each in turn an anchor (`multi.windows.Anchors`): its window of span L (`--span-s`), and for a compressed one
  its later aircraft moved toward the anchor by c in [c_min, 1) (`multi.windows.compressed`; a window without a later
  aircraft has no compressed form); a window left out when one of its commanded aircraft opens inside a loss it answers
  for (`multi.windows.left_out`), until each kind's count is reached; a shortfall is recorded, never silent. The
  batches command each flight once (`post_train.batches`).
- **The speaking** (`speak_batch`, `post_branches.branch_round`): stage D's rules (`multi.credit.rules`: W the sum,
  spoken again below the count, the varied aircraft and their branch points, each aircraft's own numbers, D141–D143)
  and its loop (`loop_options`: the rule of who answers, D145; the token part, D152). Its record is stage C's
  (`post_train.speak_round`'s sums), its rewards and outcomes those of each commanded aircraft.
- **The selection readout** (§5 item 2, `selection_windows`, `read_batch`, `selection_readout`): at most
  ``select_per_airport`` real windows of span L of each airport of the select days, drawn once with the seed (D146's
  rule), the first pass only, each aircraft with its own numbers (the same every round). By airport and kind: the
  windows, the commanded aircraft, W per aircraft (``reward_mean``) and per window, the outcomes, the go-arounds, the
  silent aircraft, the rows of the speed-word mask, the faulty points (post-training D114), and the losses of
  separation by pair (`multi.separation.PAIRS`; judged again on the states flown, `multi.census.flown_positions`, the
  census's rule). No criterion is applied (D7).
- **The identity** (§7 row 3): `MULTI_CHECKPOINT_SCHEMA`, the start's identity, the procedure masks, the token part's
  format and features, the seed, L, the kinds and c_min, and each round's windows by their anchors, commanded flights
  and shifts (`window_record`).

THE CHECKS, as stage C's: the closed loop's (D69, with the labeller's and the executor's) and the edge features' (D104).
A formal campaign (not ``--smoke``) runs from a clean checkout (a run worktree at the merged commit, outline D163; its
intent is written by hand before the publication, not checked here), on stage B's formal base, from a formal campaign
of stage C. Its campaign record is `MULTI_CAMPAIGN_SCHEMA`, resumable as stage C's (`post_train.open_campaign` with stage
D's format and reader).

    python run_ts.py multi_train --prior <the base> --instructions <A34's artefact> --executor <its spec> \\
        --windows <stage C's census: its edge reference> --start-campaign <stage C's campaign> --start-round <r> \\
        --out 4dTrajectory/outputs/POOLED/multi/<campaign id> --rounds … --span-s … --c-min … (every count given)

The readout also gives the time the aircraft take (§5 item 4, O18; `multi.timing`): each landing's delay against its
record (p50, p90), the spacing at the threshold of successive landings on one runway (p1, p5, p50) beside the records' of
the same windows, and the pairs landed in the recorded order. Not yet here (stated): the validation readout
(`multi_validation`).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.experiments.post_branches import Rules, branch_round
from ts_transformer.experiments.post_train import (
    Context, Speakers, Stage, batches, done_rounds, open_campaign, open_context, require_workers_fit, round_start,
    run_campaign, settings_of, source_campaign, speak_round, start_checkpoint, start_model, start_of,
)
from ts_transformer.experiments.post_window_loop import LANDED, LOST_SEPARATION, WindowLoop, checked_edges
from ts_transformer.multi import credit
from ts_transformer.multi.census import WindowCount, flown_positions, window_losses
from ts_transformer.multi.separation import PAIRS, Answering
from ts_transformer.multi.timing import (
    delays, landed, loop_spacings, order, quantiles, record_spacings, recorded_landings,
)
from ts_transformer.multi.tokens import PART_FEATURES, TOKENS_SCHEMA, TokenPart
from ts_transformer.multi.windows import COMPRESSED, KINDS, REAL_KIND, Anchors, Drawn, compressed, left_out
from ts_transformer.post.branches import CONTINUATIONS
from ts_transformer.post.scene import Window
from ts_transformer.post.traffic_attention import add_token_part, parameter_groups
from ts_transformer.prior.model import Prior
from ts_transformer.prior.procedure import PROCEDURE_MASKS
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: The formats of stage D's campaign and checkpoints, version 1 (§10).
MULTI_CAMPAIGN_SCHEMA = "ts-multi-train-v1"
MULTI_CHECKPOINT_SCHEMA = "ts-multi-checkpoint-v1"
#: The reader's name in the claim of a campaign's val read (the validation readout of stage D).
MULTI_CLAIM_READER = "multi_validation"


@dataclass(frozen=True)
class MultiSettings:
    """What a campaign of stage D is: everything a resume must find the same (the rounds may be raised, D157)."""

    rounds: int
    per_kind: dict[str, int]
    span_s: float
    c_min: float
    batch_windows: int
    continuations: int
    seed: int
    prior_lr: float
    traffic_lr: float
    weight_decay: float
    update_groups: int
    data_sentences: int
    select_per_airport: int
    #: the start (D164): round ``round`` of stage C's campaign ``campaign`` (repository-relative) and its checkpoint's
    #: ``checkpoint_sha256`` (`post_train.start_of`)
    start: dict[str, Any]

    def __post_init__(self) -> None:
        if set(self.per_kind) != set(KINDS) or min(self.per_kind.values()) < 0 or not any(self.per_kind.values()):
            raise ValueError(f"a count of each kind of window {KINDS}, none negative, not all zero")
        if self.span_s < 0.0 or not 0.0 < self.c_min < 1.0:
            raise ValueError(f"a span L of 0 s or more and c_min in (0, 1), not {self.span_s:g} s and {self.c_min:g}")
        if min(self.rounds, self.batch_windows, self.continuations, self.update_groups, self.data_sentences,
               self.select_per_airport) <= 0:
            raise ValueError("rounds, batch windows, continuations, groups an update, data sentences and select windows "
                             "are positive")
        if set(self.start) != {"campaign", "round", "checkpoint_sha256"}:
            raise ValueError(f"a start is its campaign, round and checkpoint_sha256, not {sorted(self.start)}")


def kind_of(window: Window) -> str:
    """The kind of a window of stage D as it flies: compressed where a commanded aircraft is moved (a compressed draw
    whose shifts all round to 0 flies the real window, and reads as one)."""
    return COMPRESSED if any(item.shift_s for item in window.joined) else REAL_KIND


def window_record(window: Window) -> dict[str, Any]:
    """A window of stage D (§7 row 3): its anchor, its row 0, its commanded flights, each one's shift and its kind."""
    return {"anchor": window.commanded.key, "row0_s": window.row0_s,
            "commanded": [record.key for record in window.commanded_all],
            "shifts_s": [0.0] + [item.shift_s for item in window.joined], "kind": kind_of(window)}


def loop_options(context: Context) -> dict[str, Any]:
    """Stage D's options of the window loop (post-training §9 item 5): its rule of who answers (D145) and its token
    part (D152)."""
    return {"answering": Answering(context.separations, context.finals, context.words.spec.step_s),
            "token_part": TokenPart(context.words), "part_width": TokenPart.width}


# ---- the draw
def draw_windows(context: Context, settings: MultiSettings, rng: np.random.Generator
                 ) -> tuple[list[Window], dict[str, Any]]:
    """A round's windows (module docstring), kind after kind in `KINDS` order, and what the draw counted: the windows
    drawn of each kind, those left out (D146) and those without a compressed form, and a shortfall."""
    train = context.splits["train"]
    anchors = Anchors(train["windows"])
    separations, step_s = context.separations, context.words.spec.step_s
    out: dict[str, list[Window]] = {kind: [] for kind in KINDS if settings.per_kind[kind] > 0}
    left: Counter = Counter()
    alone: Counter = Counter()
    for i in rng.permutation(len(train["windows"])):
        anchor = train["windows"][int(i)]
        code = anchor.scene.geometry.code
        window = anchors.window_of(anchor, settings.span_s)
        for kind, drawn in out.items():
            if len(drawn) == settings.per_kind[kind]:
                continue
            if kind == COMPRESSED and not window.joined:
                alone[kind] += 1
                continue
            chosen = (Drawn(window, REAL_KIND) if kind == REAL_KIND else compressed(window, rng, settings.c_min)).window
            if left_out(chosen, separations[code], context.finals[code], step_s):
                left[kind] += 1
            else:
                drawn.append(chosen)
        if all(len(drawn) == settings.per_kind[kind] for kind, drawn in out.items()):
            break
    return [w for drawn in out.values() for w in drawn], {
        "drawn": {kind: len(drawn) for kind, drawn in out.items()}, "left_out": dict(left),
        "no_later_aircraft": dict(alone),
        "shortfall": {kind: settings.per_kind[kind] - len(drawn) for kind, drawn in out.items()
                      if len(drawn) < settings.per_kind[kind]}}


# ---- the speaking
def speak_batch(model: Prior, context: Context, windows: Sequence[Window], places: Sequence[int],
                settings: MultiSettings, round_: int, directory: Path, k: int) -> dict[str, Any]:
    """Batch ``k`` of a round (the windows at ``places``): its two passes under stage D's rules and loop (module
    docstring), its informative groups written to ``groups_<k>.pt``, and its record (`post_train.speak_round`'s sums,
    of every commanded aircraft)."""
    train = context.splits["train"]
    batch = [windows[p] for p in places]
    found = branch_round(model, context.start_loop("train", batch), batch, places, train["sentences"], train["flights"],
                         context.geometries, context.rosters, context.finals, context.words,
                         interval_s=context.interval_s, variant=context.variant,
                         edges_reference=context.edges_reference, faults=train["faults"], device=context.device,
                         seed=settings.seed, round_=round_, split="train", continuations=settings.continuations,
                         rules=Rules(**credit.rules(settings.seed, round_, context.interval_s)),
                         loop_options=loop_options(context))
    informative = [g for g in found.groups if g.informative]
    torch.save(informative, directory / f"groups_{k}.pt")
    anchors = {w.signal_index for w in batch}              # a window's count is in each of its aircraft's ends
    return {"groups": len(found.groups), "informative_groups": len(informative),
            "spoken_again": len(found.spoken_again), "differed": list(found.differed),
            "reward_sum": sum(r.reward for r in found.first),
            "faulty_steps": sum(r.faulty_steps for r in found.first if r.index in anchors),    # once a window
            "losses_reading_fault": sum(r.loss_reads_fault for r in found.first),
            "outcomes": dict(Counter(r.outcome for r in found.first))}


# ---- the selection readout
def readout_numbers(seed: int, place: int, member: int, draw: int = 0) -> np.random.Generator:
    """The numbers of commanded aircraft ``member`` of the readout's window ``place``: the same every round; draw 0 is
    the readout's, a later draw (a read of the windows again, as stage C's ceiling readout reads, C15) its own."""
    return np.random.default_rng([seed, 1 << 30, place, member] if draw == 0 else [seed, 1 << 30, place, member, draw])


def selection_windows(context: Context, settings: MultiSettings, split: str = "select") -> list[Window]:
    """The readout's windows (module docstring): of each airport, its anchors of ``split`` in an order drawn once with
    the seed, each one's real window of span L kept unless it is left out (D146), up to ``select_per_airport``; in
    the order of the anchors."""
    anchors = Anchors(context.splits[split]["windows"])
    rng = np.random.default_rng([settings.seed, 1 << 31])
    separations, step_s = context.separations, context.words.spec.step_s
    out = []
    for code in sorted(anchors.by_airport):
        items, kept = anchors.by_airport[code], []
        for i in rng.permutation(len(items)):
            window = anchors.window_of(items[int(i)], settings.span_s)
            if not left_out(window, separations[code], context.finals[code], step_s):
                kept.append(int(i))
            if len(kept) == settings.select_per_airport:
                break
        out += [anchors.window_of(items[i], settings.span_s) for i in sorted(kept)]
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
                             context.rosters, context.finals, context.words, interval_s=context.interval_s,
                             variant=context.variant, edges_reference=context.edges_reference, faults=data["faults"],
                             observed=observed, device=context.device, **loop_options(context))
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
    several = len(ends) > 1
    positions = flown_positions(window, [(e.states, e.words, start,
                                          int(e.crossing["runway_index"]) if several and e.crossing is not None else None)
                                         for e in ends], context.words)
    window_losses(window, positions, context.separations[code], context.finals[code], context.words.spec.step_s,
                  count)
    return dict(count.loss_steps), count.steps


def selection_readout(model: Prior, context: Context, windows: Sequence[Window], settings: MultiSettings,
                      speakers: Speakers | None, *, stage: Stage, split: str = "select") -> dict[str, Any]:
    """The readout of ``windows`` of ``split`` (module docstring; `multi_validation`: the val days): its batches read
    here (``stage``'s `Stage.read_batch`) or by ``speakers`` (refused unless they run ``stage``), summed by airport and
    kind."""
    if speakers is not None and speakers.stage is not stage:
        raise ValueError("the speaking workers run another stage's parts than the one given (Speakers(..., stage=))")
    places = batches(windows, settings.batch_windows)
    read = (speakers.read(model, windows, places, split) if speakers is not None else
            [stage.read_batch(model, context, windows, p, settings, split) for p in places])
    counted: dict[str, dict[str, dict[str, Any]]] = {}
    for batch, parts in zip(places, read, strict=True):
        for window, (ends, losses, steps, timing) in zip((windows[p] for p in batch), parts, strict=True):
            for kind in (kind_of(window), "all"):
                c = counted.setdefault(window.scene.geometry.code, {}).setdefault(kind, _empty())
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
    out = {}
    for code, kinds in sorted(counted.items()):
        out[code] = {kind: {**c, "outcomes": dict(c["outcomes"]), "reward_mean": c["reward_sum"] / c["aircraft"],
                            "window_reward_mean": c["reward_sum"] / c["windows"],
                            "delays_s": quantiles(c["delays_s"], (50, 90)),
                            "spacing_s": quantiles(c["spacing_s"], (1, 5, 50)),
                            "record_spacing_s": quantiles(c["record_spacing_s"], (1, 5, 50)),
                            "order": {"same": c["order"][0], "pairs": c["order"][1]}} for kind, c in kinds.items()}
        out[code]["reward_mean"] = out[code]["all"]["reward_mean"]          # what the campaign's log prints
    return out


def _empty() -> dict[str, Any]:
    return {"windows": 0, "aircraft": 0, "reward_sum": 0.0, "outcomes": Counter(), "go_arounds": 0, "silent": 0,
            "speed_mask_rows": 0, "faulty_steps": 0, "losses_reading_fault": 0, "steps_judged": 0,
            "loss_windows": dict.fromkeys(PAIRS, 0), "loss_steps": dict.fromkeys(PAIRS, 0), "delays_s": [],
            "spacing_s": [], "record_spacing_s": [], "order": [0, 0]}


def multi_settings_of(record: dict[str, Any]) -> MultiSettings:
    """A campaign's settings as its ``campaign.json`` records them, refused unless it is a campaign of stage D."""
    if record["schema"] != MULTI_CAMPAIGN_SCHEMA:
        raise ValueError(f"a {record['schema']} campaign, not {MULTI_CAMPAIGN_SCHEMA}")
    return MultiSettings(**record["inputs"]["settings"])


def round_model(context: Context, settings: MultiSettings, out: Path, round_: int) -> Prior:
    """The model of stage D's campaign ``out`` after round ``round_`` (its checkpoint), in eval mode, refused by name
    unless its identity is this campaign's (§7 row 3): stage D's format, the start's identity on this base, today's
    procedure masks, the token part, the seed, L, the kinds, c_min and that round."""
    model, _ = stage_d().start_model(context, settings)
    state = torch.load(out / f"round_{round_}" / "checkpoint.pt", weights_only=False, map_location=context.device)
    held = state["identity"]
    expected = {"schema": MULTI_CHECKPOINT_SCHEMA, "procedure_masks": PROCEDURE_MASKS,
                "token_part": {"schema": TOKENS_SCHEMA, "features": list(PART_FEATURES)}, "seed": settings.seed,
                "span_s": settings.span_s, "per_kind": settings.per_kind, "c_min": settings.c_min}
    start = torch.load(start_checkpoint(settings.start), weights_only=False, map_location="cpu")["identity"]
    if ({k: held[k] for k in expected} != expected or held["start"] != start
            or held["start"]["base"] != context.base_identity or len(held["rounds"]) != round_ + 1):
        raise ValueError(f"{out}/round_{round_}: a checkpoint of another start, base, masks, token part, settings or "
                         f"round than this campaign's")
    model.load_state_dict(state["model"])
    return model.eval()


# ---- the stage
def stage_d() -> Stage:
    """Stage D's parts of a round (module docstring)."""
    #: the start's identity, read once (the bytes were checked when the campaign started, `round_start`)
    started: dict[str, Any] = {}

    def shaped(context: Context, settings: MultiSettings) -> Prior:
        """The model of stage D's shape: stage C's start model of the source campaign's settings, with the token
        part (its weights at zero)."""
        source = settings_of(source_campaign(REPO_ROOT / settings.start["campaign"], settings.start["round"],
                                             formal=context.formal))
        model, _ = start_model(context, source)
        add_token_part(model, TokenPart.width, TOKENS_SCHEMA)
        return model.to(context.device).eval()

    def optimizer_of(model: Prior, settings: MultiSettings) -> torch.optim.Optimizer:
        return torch.optim.AdamW(parameter_groups(model, settings.prior_lr, settings.traffic_lr),
                                 weight_decay=settings.weight_decay)

    def shape(context: Context, settings: MultiSettings) -> tuple[Prior, torch.optim.Optimizer]:
        model = shaped(context, settings)
        return model, optimizer_of(model, settings)

    def start(context: Context, settings: MultiSettings) -> tuple[Prior, torch.optim.Optimizer]:
        source, identity = round_start(context, settings.start, settings.seed)        # D164: refused by name
        model = shaped(context, settings)
        model.load_state_dict({**model.state_dict(), **source.state_dict()})       # the round's, the token part at 0
        started["identity"] = identity
        return model, optimizer_of(model, settings)

    def identity(context: Context, settings: MultiSettings, out: Path, rounds: int) -> dict[str, Any]:
        if "identity" not in started:                                             # a resumed campaign
            started["identity"] = torch.load(start_checkpoint(settings.start), weights_only=False,
                                             map_location="cpu")["identity"]
        return {"schema": MULTI_CHECKPOINT_SCHEMA, "start": started["identity"], "procedure_masks": PROCEDURE_MASKS,
                "token_part": {"schema": TOKENS_SCHEMA, "features": list(PART_FEATURES)}, "seed": settings.seed,
                "span_s": settings.span_s, "per_kind": settings.per_kind, "c_min": settings.c_min,
                "rounds": [json.loads((out / f"round_{r}" / "round.json").read_text(encoding="utf-8"))["windows"]
                           for r in range(rounds)]}

    return Stage(start_model=shape, start=start, identity=identity,
                 draw=lambda context, settings, round_: draw_windows(context, settings,
                                                                     np.random.default_rng([settings.seed, round_])),
                 speak=lambda *args, stage: speak_round(*args, stage=stage),
                 selection=lambda context, settings, split: selection_windows(context, settings, split),
                 readout=selection_readout, record=window_record, speak_batch=speak_batch, read_batch=read_batch,
                 part_width=TokenPart.width)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the base: a prior_train run's directory")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the directory of the artefact's executor spec")
    parser.add_argument("--windows", type=Path, required=True,
                        help="stage C's census (post_windows), whose conformance/edges.npz is the edge reference (D104)")
    parser.add_argument("--procedure-root", type=Path, default=DEFAULT_PROCEDURE_ROOT)
    parser.add_argument("--start-campaign", type=Path, required=True, help="stage C's campaign (post_train)")
    parser.add_argument("--start-round", type=int, required=True, help="its chosen round (§5 item 1, D164)")
    parser.add_argument("--out", type=Path, required=True, help="the campaign's directory (new, or a resume)")
    parser.add_argument("--rounds", type=int, required=True)
    for kind in KINDS:
        parser.add_argument(f"--windows-{kind}", type=int, required=True, help=f"the {kind} windows of a round (O16)")
    parser.add_argument("--span-s", type=float, required=True, help="L, s (O16)")
    parser.add_argument("--c-min", type=float, required=True, help="the least c of a compressed window (O16)")
    parser.add_argument("--batch-windows", type=int, required=True, help="the windows flown in one batch")
    parser.add_argument("--continuations", type=int, default=CONTINUATIONS, help="D94, D143: K")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prior-lr", type=float, required=True)
    parser.add_argument("--traffic-lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--update-groups", type=int, required=True, help="the branch groups of an update")
    parser.add_argument("--data-sentences", type=int, required=True, help="the data term's sentences of an update")
    parser.add_argument("--select-per-airport", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speak-workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="SMOKE: a tree with changes too; no result")
    args = parser.parse_args(argv)
    if args.speak_workers < 1:
        parser.error("--speak-workers is at least 1")
    source = args.start_campaign if args.start_campaign.is_absolute() else REPO_ROOT / args.start_campaign
    try:
        start = start_of(source, args.start_round, formal=not args.smoke)            # D164: the start's setting
    except ValueError as refused:
        parser.error(str(refused))
    prior_dir, instructions, executor, census, procedure_root, out = (
        p if p.is_absolute() else REPO_ROOT / p for p in (args.prior, args.instructions, args.executor, args.windows,
                                                          args.procedure_root, args.out))
    settings = MultiSettings(args.rounds, {kind: getattr(args, f"windows_{kind}") for kind in KINDS}, args.span_s,
                             args.c_min, args.batch_windows, args.continuations, args.seed, args.prior_lr,
                             args.traffic_lr, args.weight_decay, args.update_groups, args.data_sentences,
                             args.select_per_airport, start)
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("a campaign that is not a smoke runs on a clean checkout")
    _, opened, _ = require_conforming_closed_loop(instructions, executor)          # D69: the checks run here (D73)
    edges_reference = census / "conformance" / "edges.npz"
    checked_edges(edges_reference)                                                 # D104
    device = torch.device(args.device)
    context = open_context(prior_dir, instructions, executor, edges_reference,
                           torch.device("cpu") if args.speak_workers > 1 else device, procedure_root,
                           formal=not args.smoke)
    stage = stage_d()
    stage.start(context, settings)                  # refused by name before the workers and campaign.json (D164)
    speakers = Speakers(context, settings, args.speak_workers, device, stage=stage) if args.speak_workers > 1 else None
    context = replace(context, device=device, base=context.base.to(device).eval())
    inputs = {"prior": str(prior_dir), "instructions": str(instructions), "executor": str(executor),
              "windows": str(census), "procedure_root": str(procedure_root), "settings": asdict(settings),
              "smoke": args.smoke}
    try:
        open_campaign(out, inputs, git, opened["checks"], schema=MULTI_CAMPAIGN_SCHEMA, reader=MULTI_CLAIM_READER,
                      settings_type=MultiSettings)
        if speakers is not None and done_rounds(out) < settings.rounds:
            require_workers_fit(speakers, context, settings, done_rounds(out))       # O15, before any round
        run_campaign(out, settings, context, speakers, stage=stage)
    finally:
        if speakers is not None:
            speakers.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
