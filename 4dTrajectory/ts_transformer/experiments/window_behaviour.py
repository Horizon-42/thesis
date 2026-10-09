"""The window loop's behaviour on fixed inputs (root `CLAUDE.md`: results of different code are compared once the code is
shown to behave the same, by a check on fixed inputs; multi-aircraft control D181, D183, D186; post-training D185: the
check before a resume on changed code keeps a round's done batches). No criterion but equality (D7): two records are the
same behaviour when every digest is equal.

WHAT IT READS: a campaign of stage C (``ts-post-train-v1``, its branch method) or of stage D (``ts-multi-train-v1``),
``--campaign``: its
``campaign.json``, settings and inputs (the checks of the closed loop and of the edges run first, D73, D104). In this
process, the CPU, torch on one thread (a GPU's rounding is not this check's):

- the model of round ``--model-round`` (its checkpoint; not given: the campaign's start, `Stage.start`);
- round ``--draw-round``'s draw (default 0): of each span in the order of the batches, its first batch (stage C's
  windows have one span: batch 0), at most ``--windows`` of its windows (default all), spoken by the stage's own batch
  speaker (`Stage.speak_batch`: its two passes and its groups, written to a scratch directory and not kept);
- the readout: the first ``--readout-per-span`` selection windows of each span (`Stage.selection`, select days), read
  by the stage's reader (`Stage.read_batch`, draw 0) in the stage's batches of them.

Digested (`Digest`, sha256 of the arrays and values in order): each first-pass end of a batch (`WindowResult` and its
crossing), each branch group (its window, point, varied aircraft; each sentence's rows, records, tokens, reward and
bound), the windows spoken again and those that differed; each readout window's read (stage D's: its ends, losses by
pair, steps judged and timing). The record is ``behaviour.json``
(`WINDOW_BEHAVIOUR_SCHEMA`) under ``--out`` (a new directory), with the commit as information (outline D21: no code is
compared). ``--against`` another record: the parts that differ named, exit status 1 when any does; refused unless both
read the same campaign, rounds, windows and readout.

    python run_ts.py window_behaviour --campaign 4dTrajectory/outputs/POOLED/multi/multi_train_20261009 \\
        --out <a new directory> [--model-round N] [--windows W] [--against <another behaviour.json>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from contextlib import contextmanager
from dataclasses import astuple
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.experiments import multi_train, post_train
from ts_transformer.experiments.post_branches import BackwardRound, BranchRound
from ts_transformer.experiments.post_window_loop import WindowResult, checked_edges
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state

WINDOW_BEHAVIOUR_SCHEMA = "ts-window-behaviour-v1"
#: The readout's windows of each span, by default.
READOUT_PER_SPAN = 4


class Digest:
    """A sha256 of output data (outline D139 item 5), fed arrays and values in order."""

    def __init__(self) -> None:
        self.hash = hashlib.sha256()

    def add(self, *values: Any) -> None:
        for value in values:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                self.hash.update(str(value.dtype).encode() + str(value.shape).encode())
                self.hash.update(np.ascontiguousarray(value).tobytes())
            else:
                self.hash.update(repr(value).encode())

    def rows(self, rows: Any) -> None:
        for name in ("time_s", "own", "candidates", "runway_in_force", "go_around", "heading_in_force",
                     "words_in_force", "since", "targets"):
            self.add(getattr(rows, name))

    def permitted(self, permitted: Any) -> None:
        self.add(*permitted.masks, permitted.time_s, permitted.own, permitted.temperature)

    def sentence(self, rows: Any, permitted: Any, tokens: Any) -> None:
        self.rows(rows)
        self.permitted(permitted)
        self.add(len(tokens), *tokens)

    def result(self, result: WindowResult) -> None:
        self.add(result.index, result.kind, result.outcome, None if result.loss is None else astuple(result.loss),
                 result.loss_step, result.other, result.reward, result.go_arounds, result.words, result.states,
                 result.speed_mask_rows, result.faulty_steps, result.loss_reads_fault)

    def samples(self, batch: Any) -> None:
        self.add(*batch.rows, *batch.permitted.masks, batch.permitted.time_s, batch.permitted.own,
                 batch.traffic.tokens, batch.traffic.present, batch.advantage, batch.counted)


def end_digest(result: WindowResult) -> str:
    """A first-pass or readout end: the result and its crossing."""
    digest = Digest()
    digest.result(result)
    digest.add(None if result.crossing is None else sorted((k, repr(v)) for k, v in result.crossing.items()))
    return digest.hash.hexdigest()


def readout_digest(read: WindowResult | tuple[Any, ...]) -> str:
    """A readout window's read: stage C's end, or stage D's ends, losses by pair, steps judged and timing."""
    if isinstance(read, WindowResult):
        return end_digest(read)
    ends, losses, steps, timing = read
    digest = Digest()
    digest.add([end_digest(r) for r in ends], sorted(losses.items()), steps,
               json.dumps(timing, sort_keys=True, default=repr))
    return digest.hash.hexdigest()


def round_digests(found: BranchRound) -> dict[str, Any]:
    """A batch's two passes (`branch_round`'s): each first-pass end, each group, the windows spoken again and differed."""
    groups = []
    for group in found.groups:
        digest = Digest()
        digest.add(group.window, group.branch, group.varied)
        for sentence in group.sentences:
            digest.sentence(sentence.rows, sentence.permitted, sentence.tokens)
            digest.add(sentence.reward, sentence.until)
        groups.append(digest.hash.hexdigest())
    out = {"first": [end_digest(r) for r in found.first], "groups": groups,
           "spoken_again": list(found.spoken_again), "differed": list(found.differed)}
    if isinstance(found, BackwardRound):          # its points found backward (D182): their depths and counts
        out["searched"] = [found.depths_s, found.searches, found.found, found.to_event, found.to_end]
    return out


@contextmanager
def captured_rounds() -> Iterator[list[BranchRound]]:
    """The `branch_round` and `backward_round` results of the stages' batch speakers while open (`post_train.speak_batch`,
    `multi_train.speak_batch` call them by their module's names), in the order spoken; taken off after."""
    found: list[BranchRound] = []
    real, real_backward = post_train.branch_round, multi_train.backward_round
    if multi_train.branch_round is not real:
        raise RuntimeError("the stages' batch speakers call different branch rounds")

    def capturing(round_: Callable[..., BranchRound]) -> Callable[..., BranchRound]:
        def call(*args: Any, **kwargs: Any) -> BranchRound:
            out = round_(*args, **kwargs)
            found.append(out)
            return out
        return call

    post_train.branch_round = multi_train.branch_round = capturing(real)
    multi_train.backward_round = capturing(real_backward)
    try:
        yield found
    finally:
        post_train.branch_round = multi_train.branch_round = real
        multi_train.backward_round = real_backward


def opened_campaign(record: dict[str, Any]) -> tuple[Any, post_train.Stage]:
    """A campaign's settings and stage, by its schema; of stage C, its branch method only (the landed and value methods
    speak no branch round), refused by name otherwise."""
    if record["schema"] == multi_train.MULTI_CAMPAIGN_SCHEMA:
        return multi_train.multi_settings_of(record), multi_train.stage_d()
    if record["schema"] == post_train.CAMPAIGN_SCHEMA:
        settings = post_train.settings_of(record)
        if settings.method != post_train.BRANCH:
            raise ValueError(f"window_behaviour digests branch rounds; stage C's method {settings.method!r} speaks none")
        return settings, post_train.STAGE_C
    raise ValueError(f"a {record['schema']} campaign: neither stage C's nor stage D's")


def model_of(context: post_train.Context, settings: Any, stage: post_train.Stage, campaign: Path,
             model_round: int | None) -> Any:
    """The model of round ``model_round`` (its checkpoint) or the campaign's start (None), in eval mode."""
    if model_round is None:
        return stage.start(context, settings)[0].eval()
    if isinstance(settings, multi_train.MultiSettings):
        return multi_train.round_model(context, settings, campaign, model_round)
    return post_train.round_model(context, settings, campaign, model_round)


def chosen_batches(windows: list[Any], found: list[list[int]], most: int | None) -> list[tuple[int, list[int]]]:
    """Of each span in the order of the batches, its first batch (its index) and at most ``most`` of its windows."""
    out, spans = [], set()
    for k, places in enumerate(found):
        span = windows[places[0]].span_s
        if span not in spans:
            spans.add(span)
            out.append((k, places if most is None else places[:most]))
    return out


def readout_places(selection: list[Any], per_span: int) -> list[int]:
    """The first ``per_span`` selection windows of each span, spans in the order they first appear."""
    out: list[int] = []
    for span in dict.fromkeys(w.span_s for w in selection):
        out += [p for p, w in enumerate(selection) if w.span_s == span][:per_span]
    return out


def behaviour(context: post_train.Context, settings: Any, stage: post_train.Stage, model: Any, draw_round: int,
              most: int | None, per_span: int) -> dict[str, Any]:
    """The digests of the module docstring: ``batches`` (by batch index: its windows' places and `round_digests`) and
    ``readout`` (by selection place: its read, `readout_digest`)."""
    windows, _ = stage.draw(context, settings, draw_round)
    batches: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="window_behaviour_") as scratch, captured_rounds() as found:
        for k, places in chosen_batches(windows, stage.batches(windows, settings), most):
            stage.speak_batch(model, context, windows, places, settings, draw_round, Path(scratch), k)
            (spoken,) = found
            found.clear()
            batches[str(k)] = {"places": list(places), "span_s": windows[places[0]].span_s, **round_digests(spoken)}
            print(f"batch {k}: {len(places)} windows, {len(batches[str(k)]['groups'])} groups", flush=True)
    selection = stage.selection(context, settings, "select")
    places = readout_places(selection, per_span)
    readout = {}
    for batch in stage.batches([selection[p] for p in places], settings):     # read as the readout reads: in batches
        chosen = [places[i] for i in batch]
        for p, read in zip(chosen, stage.read_batch(model, context, selection, chosen, settings, "select", 0)):
            readout[str(p)] = readout_digest(read)
    return {"batches": batches, "readout": readout}


def differences(ours: dict[str, Any], theirs: dict[str, Any]) -> list[str]:
    """The parts of two records' digests that differ, by name (refused unless they read the same inputs)."""
    if ours["inputs"] != theirs["inputs"]:
        raise ValueError(f"records of other inputs: {ours['inputs']} and {theirs['inputs']}")
    out = []
    a, b = ours["digests"], theirs["digests"]
    for k in sorted(set(a["batches"]) | set(b["batches"]), key=int):
        mine, other = a["batches"].get(k), b["batches"].get(k)
        if mine is None or other is None:
            out.append(f"batch {k}: in one record only")
            continue
        for key in sorted((set(mine) | set(other)) - {"first", "groups"}):   # the searched counts of D182's rounds too
            if key not in mine or key not in other or mine[key] != other[key]:
                out.append(f"batch {k}: {key}")
        for key in ("first", "groups"):
            if len(mine[key]) != len(other[key]):
                out.append(f"batch {k}: {len(mine[key])} and {len(other[key])} {key}")
            else:
                out += [f"batch {k}: {key} {i}" for i, (x, y) in enumerate(zip(mine[key], other[key])) if x != y]
    if a["readout"] != b["readout"]:
        out += [f"readout window {p}" for p in sorted(set(a["readout"]) | set(b["readout"]), key=int)
                if a["readout"].get(p) != b["readout"].get(p)]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True, help="a campaign of stage C or D (its campaign.json)")
    parser.add_argument("--model-round", type=int, help="the round whose checkpoint speaks (default: the start)")
    parser.add_argument("--draw-round", type=int, default=0, help="the round whose draw is spoken (default 0)")
    parser.add_argument("--windows", type=int, help="at most this many windows of each batch (default all)")
    parser.add_argument("--readout-per-span", type=int, default=READOUT_PER_SPAN,
                        help=f"selection windows read of each span (default {READOUT_PER_SPAN})")
    parser.add_argument("--against", type=Path, help="another record (behaviour.json) to compare with")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)
    campaign, out = (p if p.is_absolute() else REPO_ROOT / p for p in (args.campaign, args.out))
    if out.exists():
        parser.error(f"{out} exists: the check writes a new directory")
    if args.windows is not None and args.windows < 1 or args.readout_per_span < 0:
        parser.error("--windows is at least 1 and --readout-per-span not negative")
    against = None if args.against is None else json.loads(args.against.read_text(encoding="utf-8"))
    if against is not None and against["schema"] != WINDOW_BEHAVIOUR_SCHEMA:
        parser.error(f"{args.against}: a {against['schema']} record, not {WINDOW_BEHAVIOUR_SCHEMA}")
    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    try:
        settings, stage = opened_campaign(record)
    except ValueError as refused:
        parser.error(str(refused))
    inputs = post_train.inputs_here(record["inputs"])
    _, opened, _ = require_conforming_closed_loop(Path(inputs["instructions"]), Path(inputs["executor"]))   # D73
    edges_reference = Path(inputs["windows"]) / "conformance" / "edges.npz"
    checked_edges(edges_reference)                                                                   # D104
    torch.set_num_threads(1)
    context = post_train.open_context(Path(inputs["prior"]), Path(inputs["instructions"]), Path(inputs["executor"]),
                                      edges_reference, torch.device("cpu"), Path(inputs["procedure_root"]),
                                      formal=True, data=False, splits=("train", "select"))
    model = model_of(context, settings, stage, campaign, args.model_round)
    out.mkdir(parents=True)
    head = {"schema": WINDOW_BEHAVIOUR_SCHEMA, "started_utc": utc_now(), "git": git_state(), "checks": opened["checks"],
            "inputs": {"campaign": str(campaign.relative_to(REPO_ROOT) if campaign.is_relative_to(REPO_ROOT)
                                       else campaign),
                       "model_round": args.model_round, "draw_round": args.draw_round, "windows": args.windows,
                       "readout_per_span": args.readout_per_span}}
    begin = time.perf_counter()
    found = {**head, "digests": behaviour(context, settings, stage, model, args.draw_round, args.windows,
                                          args.readout_per_span), "s": time.perf_counter() - begin}
    differ = None
    if against is not None:
        differ = differences(found, against)
        found["against"] = {"record": str(args.against), "git": against["git"], "differ": differ}
    write_json_atomic(out / "behaviour.json", {**found, "finished_utc": utc_now()})
    print(f"written {out / 'behaviour.json'} ({found['s']:.0f} s)", flush=True)
    if differ is not None:
        print("the same behaviour" if not differ else "DIFFERENT: " + "; ".join(differ), flush=True)
    return 1 if differ else 0


if __name__ == "__main__":
    raise SystemExit(main())
