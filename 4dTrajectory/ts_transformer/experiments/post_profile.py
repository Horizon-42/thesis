"""C8: the profile of the post-training at the formal size (post-training §8 C8; outline §5 rule 13, the user's rule of
2026-09-30: a smoke at a small size proves nothing about the formal size) — one round of `post_train` with the formal
counts, measured part by part. No criterion is applied (D7): the report goes to the user.

WHAT IT MEASURES, from the base of stage B with zero-output traffic modules (the model of round 0, D29):

1. **One speaking batch** (the round's first batch of `post_train.batches`, its two passes of D94) under cProfile: its
   wall time and the cumulative time of each part (`PARTS`: the prior's forward step, the executor's step, the
   speaker's masks, the speed-word mask, the scene of the separation judge, the edge features, the copies of a loop at
   the branch points). APPROXIMATION, stated in the output: cProfile's cumulative time, on a GPU a part's kernels run
   asynchronously and their time shows where the host waits for them (the speaker's draw), not in the part.
2. **The round** (`post_train.speak_round` on every batch, then `post_train.train_pass`), each timed: the speaking (the
   second pass included), the training pass, the selection readout; the groups written (their count and bytes on
   disk).
3. **One update's memory** (`pass_memory`, before the pass): the peak GPU memory and the time of one update's forward
   and backward (no optimizer step: the model is unchanged) for k of the round's branch groups that bound any update
   of k ≥ 2 groups in rows and traffic (the longest, the widest, then the next longest), k in
   ``--pass-memory-groups``, with
   ``data_sentences`` sentences of the data term — what `Settings.update_groups` costs (O13). An update that runs out
   of the GPU's memory is recorded as such (it is the measurement), and no larger k is tried.
4. **Memory** at the formal size, after each part: the GPU's peak allocated and reserved memory DURING that part (the
   peak is reset before it) and its free memory; the host's free memory and this process's peak resident memory SINCE
   ITS START (the host keeps no peak that can be reset: a part's own figure is the rise over the part before).

The windows are drawn as `post_train` draws round 0 (the same seed gives the same windows); its groups are written
under ``--out`` (a scratch directory: no campaign, nothing published) and the record is ``profile.json``, written again
after each part, so a part that fails (the GPU's memory at a size tried) leaves the parts before it.

    python run_ts.py post_profile --prior <the base> --instructions <A34's artefact> --executor <its spec> \\
        --windows <the census> --out <scratch>/post_profile --batch-windows … (post_train's counts)
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import resource
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from flight_scenarios.procedure_final import DEFAULT_PROCEDURE_ROOT

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.experiments.post_branches import branch_round
from ts_transformer.experiments.post_train import (
    BRANCH, KINDS, STAGE_C, Context, Settings, batches, draw_round, open_context, selection_readout,
    selection_windows, speak_round, start_model, train_pass,
)
from ts_transformer.experiments.post_window_loop import checked_edges
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.post.branches import CONTINUATIONS, samples
from ts_transformer.post.loss import PassStart, update_step
from ts_transformer.prior.batch import collate
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: v2 (C8, 2026-10-06): one update's memory by its groups (``pass_memory``); the record written after each part. v3
#: (2026-10-06): an update measured in pieces of a group each (`post.loss.update_step`), ``1_densest`` beside.
PROFILE_SCHEMA = "ts-post-profile-v3"
#: The parts of a speaking batch (module docstring, item 1): each the functions (a file of the package and a name) whose
#: cumulative time it is.
PARTS = {
    "prior_step": (("prior/model.py", "extend"),),
    "executor_step": (("autopilot/start.py", "step"),),
    "speaker_masks": (("prior/speaker.py", "_allowed"),),
    "speed_mask": (("post/speed_mask.py", "speed_check"),),
    "separation_scene": (("post/traffic.py", "traffic"),),
    "edge_features": (("post/edges.py", "tokens"),),
    "loop_copies": (("experiments/post_window_loop.py", "copy"),),
}
#: The branch groups of one update whose memory `pass_memory` measures, by default.
PASS_MEMORY_GROUPS = (1, 2, 4, 8, 16)
APPROXIMATION = ("cProfile's cumulative time of each part's functions; on a GPU a part's kernels run asynchronously and "
                 "their time shows where the host waits for them (the speaker's draw), not in the part")


def part_times(stats: pstats.Stats) -> dict[str, float]:
    """The cumulative time (s) of each part of `PARTS` in ``stats``: the largest of its functions' (a method calling
    itself, or one function calling another of its part, is not counted twice)."""
    out = {}
    for part, functions in PARTS.items():
        found = [entry[3] for (path, _, name), entry in stats.stats.items()
                 if any(path.endswith(f"ts_transformer/{file}") and name == wanted for file, wanted in functions)]
        out[part] = max(found, default=0.0)
    return out


def memory(device: torch.device) -> dict[str, Any]:
    """This process's peak resident memory and the host's free memory, and on a GPU its peak allocated and reserved
    memory and its free memory, in GiB."""
    gib = float(1 << 30)
    with open("/proc/meminfo", encoding="utf-8") as f:
        info = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f}
    out: dict[str, Any] = {"host_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / gib,
                           "host_available_gib": info["MemAvailable"] / gib}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        out.update(gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / gib,
                   gpu_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / gib, gpu_free_gib=free / gib,
                   gpu_total_gib=total / gib)
    return out


def timed(device: torch.device, run: Callable[[], Any]) -> tuple[Any, float]:
    """``run()`` and its wall time (s), the GPU's queue drained before the clock stops; the GPU's peak memory is reset
    before it, so `memory` after it gives the part's own peak."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    out = run()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return out, time.perf_counter() - start


def group_size(group: Any) -> tuple[int, int]:
    """What an update holding ``group`` is padded to: its longest sentence's rows and the most traffic tokens at a row
    of any of its sentences (the update's tensors grow with both, D116)."""
    return (max(len(s.tokens) for s in group.sentences),
            max([1] + [len(row) for s in group.sentences for row in s.tokens]))


def pass_memory(model: Any, context: Context, directory: Path, settings: Settings, counts: Sequence[int],
                saved: Callable[[dict[str, Any]], None] = lambda part: None, *, part_width: int) -> dict[str, Any]:
    """Part 3 of the module docstring: for each k of ``counts`` (ascending), one update's forward and backward on k
    branch groups written in ``directory``, in pieces of a group each as the campaign runs it (`update_step`): its
    peak is its largest piece's (a piece padded to its own longest sentence and widest traffic, `group_size`, D116) and
    the data term's, so the size of k matters little. The k are the longest group, the widest group (by traffic, then
    rows) and the next longest; besides k = 1 (the longest), the widest alone (``1_widest``) and the group of the
    largest rows × traffic (``1_densest``) when they are other groups. A PROXY, not a bound: a group long and wide but
    none of these three is not measured. With ``data_sentences`` sentences of the data term; its peak GPU memory, its
    time and its shape (the pieces' sentences, the longest piece's rows, the widest piece's traffic); the gradients
    dropped after each, no optimizer step. The groups are read file by file, the longest and the widest kept
    (the round's are not held at once). A k past the groups held is recorded as such; an update out of the GPU's memory
    is recorded and stops the rest. ``saved`` gets the record after each one measured. APPROXIMATIONS, stated: the data
    term's sentences are one draw, not a bound (the pass draws them again for each update; they carry no traffic); an
    update of the pass holds the groups of one file (`update_pairs`), so a k past a file's groups is measured but never
    formed; the first update's time includes the process's first backward. ``part_width``: the stage's token part
    (`post_train.Stage.part_width`), which the groups' samples read."""
    device = context.device
    wanted = max(counts)
    longest: list[Any] = []
    widest: Any = None
    densest: Any = None
    sizes: list[tuple[int, int]] = []
    for path in sorted(directory.glob("groups_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
        loaded = torch.load(path, weights_only=False)
        sizes += [group_size(g) for g in loaded]
        longest = sorted(longest + loaded, key=group_size, reverse=True)[:wanted]
        widest = max(([widest] if widest is not None else []) + loaded, key=lambda g: group_size(g)[::-1],
                     default=None)
        densest = max(([densest] if densest is not None else []) + loaded,
                      key=lambda g: (group_size(g)[0] * group_size(g)[1], group_size(g)[0]), default=None)
    order = longest if widest is None or widest is longest[0] else (
        [longest[0], widest] + [g for g in longest[1:] if g is not widest])
    runs = []
    for k in sorted(counts):
        runs.append((str(k), order[:k]))
        if k == 1 and order and order[0] is not widest:
            runs.append(("1_widest", [widest]))
        if k == 1 and order and densest is not order[0] and densest is not widest:
            runs.append(("1_densest", [densest]))
    chosen = np.random.default_rng([settings.seed, 0, 3]).choice(
        len(context.data), size=min(settings.data_sentences, len(context.data)), replace=False)
    data = [context.data[int(i)] for i in chosen]
    start = PassStart(model)
    out: dict[str, Any] = {"data_sentences": len(data), "groups_written": len(sizes),
                           "largest_rows": max([0] + [r for r, _ in sizes]),
                           "largest_traffic": max([0] + [n for _, n in sizes]), "updates": {}}
    for label, groups in runs:
        if label not in ("1_widest", "1_densest") and int(label) > len(order):
            out["updates"][label] = "fewer groups held"
            continue
        try:
            batch = [samples([group], device, part_width) for group in groups]   # the campaign's pieces (`update_step`)
            rows = collate(data, device)
            shape = {"sentences": sum(int(p.rows.asked.shape[0]) for p in batch),
                     "rows": max(int(p.rows.asked.shape[1]) for p in batch),
                     "traffic": max(int(p.traffic.tokens.shape[2]) for p in batch),
                     "data_rows": int(rows.asked.shape[1])}
            _, seconds = timed(device, lambda: update_step(model, start, context.base, batch, rows))
            out["updates"][label] = {"s": seconds, **shape, **memory(device)}
        except torch.OutOfMemoryError:
            out["updates"][label] = "out of memory"
            break
        finally:
            batch = rows = None
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            saved(out)
    return out


def profile(context: Context, settings: Settings, out: Path, counts: Sequence[int] = PASS_MEMORY_GROUPS,
            save: Callable[[dict[str, Any]], None] = lambda record: None) -> dict[str, Any]:
    """The record of the profile (module docstring), handed to ``save`` after each part."""
    device = context.device
    record: dict[str, Any] = {"memory_before": memory(device)}
    model, optimizer = start_model(context, settings)
    windows, drawn = draw_round(context, settings.per_kind, np.random.default_rng([settings.seed, 0]))
    record["draw"] = drawn
    all_batches = batches(windows, settings.batch_windows)
    record["batches"] = [len(b) for b in all_batches]
    # 1. one speaking batch under cProfile
    places = all_batches[0]
    batch = [windows[p] for p in places]
    train = context.splits["train"]
    profiler = cProfile.Profile()
    found, wall = timed(device, lambda: profiler.runcall(
        branch_round, model, context.start_loop("train", batch), batch, places, train["sentences"], train["flights"],
        context.geometries, context.rosters, context.finals, context.words, interval_s=context.interval_s,
        variant=context.variant, edges_reference=context.edges_reference, faults=train["faults"], device=device,
        seed=settings.seed, round_=0, split="train", continuations=settings.continuations))
    stats = pstats.Stats(profiler)
    record["one_batch"] = {"windows": len(batch), "wall_s": wall, "parts_s": part_times(stats),
                           "approximation": APPROXIMATION, "spoken_again": len(found.spoken_again),
                           "groups": len(found.groups), "memory": memory(device)}
    save(record)
    # 2. the round
    directory = out / "round_0"
    directory.mkdir(parents=True)
    spoken, speaking_s = timed(device, lambda: speak_round(model, context, windows, settings, 0, directory))
    record["round"] = {"speaking_s": speaking_s, "speaking": spoken, "memory_after_speaking": memory(device),
                       "groups_bytes": sum(p.stat().st_size for p in directory.glob("groups_*.pt"))}
    save(record)
    record["pass_memory"] = pass_memory(model, context, directory, settings, counts,
                                        lambda part: save({**record, "pass_memory": part}),
                                        part_width=STAGE_C.part_width)
    save(record)
    passed, pass_s = timed(device, lambda: train_pass(model, context, optimizer, directory, settings,
                                                      np.random.default_rng([settings.seed, 0, 1]),
                                                      part_width=STAGE_C.part_width))
    record["round"].update(pass_s=pass_s, passed=passed, memory_after_pass=memory(device))
    save(record)
    select = selection_windows(context, settings)
    _, readout_s = timed(device, lambda: selection_readout(model, context, select, settings))
    record["round"].update(selection_windows=len(select), selection_readout_s=readout_s,
                           memory_after_readout=memory(device))
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the base: a prior_train run's directory")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True, help="the census whose conformance/edges.npz is read")
    parser.add_argument("--procedure-root", type=Path, default=DEFAULT_PROCEDURE_ROOT)
    parser.add_argument("--out", type=Path, required=True, help="a new scratch directory")
    for kind in KINDS:
        parser.add_argument(f"--windows-{kind.lower()}", type=int, required=True)
    parser.add_argument("--batch-windows", type=int, required=True)
    parser.add_argument("--continuations", type=int, default=CONTINUATIONS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prior-lr", type=float, required=True)
    parser.add_argument("--traffic-lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--update-groups", type=int, required=True)
    parser.add_argument("--data-sentences", type=int, required=True)
    parser.add_argument("--select-per-airport", type=int, required=True)
    parser.add_argument("--traffic-hidden", type=int, required=True)
    parser.add_argument("--traffic-heads", type=int, required=True)
    parser.add_argument("--pass-memory-groups", type=int, nargs="+", default=list(PASS_MEMORY_GROUPS),
                        help="the branch groups of one update whose memory is measured (part 3)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if min(args.pass_memory_groups) < 1:
        parser.error("--pass-memory-groups are at least 1")
    prior_dir, instructions, executor, census, procedure_root, out = (p if p.is_absolute() else REPO_ROOT / p for p in (
        args.prior, args.instructions, args.executor, args.windows, args.procedure_root, args.out))
    if out.exists():
        parser.error(f"{out} exists: a profile writes a new directory")
    settings = Settings(1, {kind: getattr(args, f"windows_{kind.lower()}") for kind in KINDS}, args.batch_windows,
                        args.continuations, args.seed, args.prior_lr, args.traffic_lr, args.weight_decay,
                        args.update_groups, args.data_sentences, args.select_per_airport, args.traffic_hidden,
                        args.traffic_heads, None, BRANCH, args.seed)  # the base's start (D29), D94's method, the readout's seed
    _, opened, _ = require_conforming_closed_loop(instructions, executor)
    edges_reference = census / "conformance" / "edges.npz"
    checked_edges(edges_reference)
    device = torch.device(args.device)
    context = open_context(prior_dir, instructions, executor, edges_reference, device, procedure_root, formal=False)
    out.mkdir(parents=True)
    head = {"schema": PROFILE_SCHEMA, "started_utc": utc_now(), "git": git_state(), "checks": opened["checks"],
            "inputs": {"prior": str(prior_dir), "instructions": str(instructions), "executor": str(executor),
                       "windows": str(census), "settings": {k: v for k, v in vars(args).items()
                                                            if k not in ("prior", "instructions", "executor",
                                                                         "windows", "out", "procedure_root")}}}

    def save(parts: dict[str, Any], **more: Any) -> None:
        write_json_atomic(out / "profile.json", json.loads(json.dumps({**head, **parts, **more}, default=str)))

    record = profile(context, settings, out, args.pass_memory_groups, save)
    save(record, finished_utc=utc_now())
    print(json.dumps({"one_batch": record["one_batch"], "round": {k: v for k, v in record["round"].items()
                                                                  if k.endswith("_s")}}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
