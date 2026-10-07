"""MC5: the profile of stage D at the formal size (multi-aircraft control §11 MC5; as post-training C8; outline §5
rule 13: a smoke at a small size proves nothing about the formal size) — round 0 of `multi_train` with the formal
counts, measured span by span. No criterion is applied (D7): the report goes to the user, who sets the campaign.

WHAT IT MEASURES, from stage D's start (`multi_train.stage_d`: the round of stage C given, its token part at zero):

1. **One batch of each span** (the round's first batch of each span L, `Stage.batches`: at most ``--batch-rows``
   commanded aircraft of one span), spoken in this process with its two passes (`multi_train.speak_batch`): its wall
   time, its windows, rows and groups, the bytes of its groups, the GPU's peak during it and the host's memory after.
2. **The speaking workers' memory** (post-training O15, with ``--speak-workers`` of 2 or more): one worker's peak on the
   first batch of each span and the batch of the most rows (`post_train.measured_batches`), spoken with round 0's model,
   this process's in one update of the pass on their groups (`post_train.pass_memory_of`), and for each N up to
   ``--speak-workers`` what does not fit (`post_train.workers_fit`; empty: N fits). Where ``--speak-workers`` do not fit,
   the profile stops there by name (the record up to part 2 written).
3. **The round**: its speaking (`post_train.speak_round` by the workers), timed, with the batches of each span and
   the bytes of all its groups; the selection readout of the select windows (`multi_train.selection_windows`, at most
   ``--select-per-airport`` an airport and span) read twice, draws 0 and 1 (`multi_train.read_batch`'s draw), each
   timed (draw 0's time includes each worker's first build of the select windows and its split, draw 1's is a round's);
   then the training pass (`post_train.train_pass`), timed.
4. **The spread of round 0's W per aircraft** on the select windows, by span and over all (`spread`): the mean W per
   aircraft (a window's W over its commanded aircraft, summed over the windows), its standard error by units (the
   ratio estimator: a window of a span is the unit, its aircraft are not independent; over all, an anchor, whose
   windows of the spans are nested and share their aircraft), and the paired standard error of the difference between
   the two draws (the same windows and model, independent numbers), each with the units that would make it 0.01 and
   0.02 (`windows_for`: the standard error falls as one over the root of the units), and those of each airport.
   APPROXIMATION, stated: two rounds' readouts use the same numbers, so the two draws' paired standard error is an upper
   bound of the noise of comparing two rounds that differ little, not the noise of a model's change.

The groups are written under ``--out`` (a scratch directory: no campaign, nothing published); the record is
``profile.json``, written again after each part, so a part that fails leaves the parts before it.

    python run_ts.py multi_profile --prior <the base> --instructions <A34's artefact> --executor <its spec> \\
        --windows <stage C's census> --start-campaign <stage C's campaign> --start-round <r> --out <scratch> \\
        (multi_train's settings) --speak-workers N
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.experiments.multi_train import add_arguments, settings_from, span_key, stage_d
from ts_transformer.experiments.post_profile import memory, timed
from ts_transformer.experiments.post_train import (
    Context, Speakers, Stage, available_memory, measured_batches, open_context, pass_memory_of, start_of, train_pass,
    workers_fit,
)
from ts_transformer.experiments.post_window_loop import checked_edges
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.post.scene import Window
from ts_transformer.prior.model import Prior
from ts_transformer.repo_layout import REPO_ROOT, git_state

PROFILE_SCHEMA = "ts-multi-profile-v1"
#: The standard errors of W per aircraft for which `spread` gives the windows needed.
TARGET_SE = (0.01, 0.02)


def ratio_se(sums: Sequence[float], counts: Sequence[int]) -> tuple[float, float]:
    """The mean of ``sums`` over ``counts`` (Σ sums / Σ counts: W per aircraft) and its standard error by units (each a
    window: the ratio estimator's, n / (n − 1) Σ (sum − mean × count)² over (Σ counts)²); nan for fewer than 2 units."""
    sums, counts = np.asarray(sums, dtype=float), np.asarray(counts, dtype=float)
    mean = float(sums.sum() / counts.sum())
    n = len(sums)
    if n < 2:
        return mean, math.nan
    return mean, float(math.sqrt(n / (n - 1) * np.sum((sums - mean * counts) ** 2)) / counts.sum())


def windows_for(se: float, windows: int, target: float) -> int:
    """The windows that make a standard error ``se`` measured on ``windows`` into ``target`` (one over the root)."""
    return math.ceil(windows * (se / target) ** 2)


#: A window read (`read_values`): its airport, span, anchor, W (the sum of its aircraft's rewards) and commanded aircraft.
Value = tuple[str, str, str, float, int]


def spread(first: Sequence[Value], second: Sequence[Value]) -> dict[str, Any]:
    """Part 4 of the module docstring: for each span (a window the unit) and over all (an anchor the unit: its windows'
    W and aircraft summed), of the windows read twice (``first`` and ``second``, in one order): the units, the windows,
    the aircraft, the airports, the mean W per aircraft of each draw and its standard error, the paired standard error
    of their difference, and the units that make each standard error a target of `TARGET_SE`, in all and an airport."""
    if [(a, s, k, n) for a, s, k, _, n in first] != [(a, s, k, n) for a, s, k, _, n in second]:
        raise ValueError("the two draws read other windows")
    groups: dict[str, dict[tuple[str, ...], list[int]]] = defaultdict(lambda: defaultdict(list))
    for k, (code, span, anchor, _, _) in enumerate(first):
        groups[span][(code, span, anchor)].append(k)
        groups["all"][(code, anchor)].append(k)
    out = {}
    for name, units in groups.items():
        places = list(units.values())
        counts = [sum(first[k][4] for k in unit) for unit in places]
        mean_0, se_0 = ratio_se([sum(first[k][3] for k in unit) for unit in places], counts)
        mean_1, se_1 = ratio_se([sum(second[k][3] for k in unit) for unit in places], counts)
        difference, se_d = ratio_se([sum(first[k][3] - second[k][3] for k in unit) for unit in places], counts)
        n, airports = len(places), len({key[0] for key in units})
        out[name] = {"units": n, "windows": sum(len(unit) for unit in places), "aircraft": int(sum(counts)),
                     "airports": airports, "mean_0": mean_0, "mean_1": mean_1, "se_0": se_0, "se_1": se_1,
                     "difference": difference, "paired_se": se_d,
                     "units_for": {f"{target:g}": {
                         key: {"all": windows_for(se, n, target), "an_airport": math.ceil(windows_for(se, n, target)
                                                                                           / airports)}
                         for key, se in (("se_0", se_0), ("paired_se", se_d))} for target in TARGET_SE}
                     if n > 1 else None}
    return out


def read_values(model: Prior, context: Context, windows: Sequence[Window], settings: Any, speakers: Speakers | None,
                stage: Stage, draw: int) -> list[Value]:
    """The select windows read with ``draw`` (by ``speakers`` or here): each window's `Value`, in the windows'
    order."""
    places = stage.batches(windows, settings)
    read = (speakers.read(model, windows, places, "select", draw) if speakers is not None else
            [stage.read_batch(model, context, windows, p, settings, "select", draw) for p in places])
    out: dict[int, Value] = {}
    for batch, parts in zip(places, read, strict=True):
        for p, (ends, _, _, _) in zip(batch, parts, strict=True):
            out[p] = (windows[p].scene.geometry.code, span_key(windows[p].span_s), windows[p].commanded.key,
                      sum(e.reward for e in ends), len(ends))
    return [out[p] for p in range(len(windows))]


def rows_of(windows: Sequence[Window], batch: Sequence[int]) -> int:
    return sum(len(windows[p].signal_indices) for p in batch)


def profile(context: Context, settings: Any, out: Path, speakers: Speakers | None, stage: Stage,
            save: Callable[[dict[str, Any]], None] = lambda record: None) -> dict[str, Any]:
    """The record of the profile (module docstring), handed to ``save`` after each part."""
    device = context.device
    record: dict[str, Any] = {"memory_before": memory(device)}
    model, optimizer = stage.start(context, settings)
    windows, drawn = stage.draw(context, settings, 0)
    found = stage.batches(windows, settings)
    record["draw"] = drawn
    by_span: dict[str, list[int]] = defaultdict(list)
    for k, batch in enumerate(found):
        by_span[span_key(windows[batch[0]].span_s)].append(k)
    record["batches"] = {span: {"batches": len(ks), "windows": sum(len(found[k]) for k in ks),
                                "rows": [rows_of(windows, found[k]) for k in ks]} for span, ks in by_span.items()}
    save(record)
    # 1. one batch of each span, here
    record["one_batch"] = {}
    with tempfile.TemporaryDirectory(prefix="multi_profile_batch_", dir=out) as scratch:
        for span, ks in by_span.items():
            k = ks[0]
            spoken, seconds = timed(device, lambda: stage.speak_batch(model, context, windows, found[k], settings, 0,
                                                                      Path(scratch), k))
            record["one_batch"][span] = {"batch": k, "windows": len(found[k]), "rows": rows_of(windows, found[k]),
                                         "wall_s": seconds, "speaking": spoken,
                                         "groups_bytes": (Path(scratch) / f"groups_{k}.pt").stat().st_size,
                                         "memory": memory(device)}
            if device.type == "cuda":
                torch.cuda.empty_cache()
            save(record)
    # 2. the speaking workers' memory
    if speakers is not None:
        with tempfile.TemporaryDirectory(prefix="multi_profile_measure_", dir=out) as scratch:
            measured = speakers.measure(0, Path(scratch), model)                # round 0's model
            passed = pass_memory_of(context, settings, Path(scratch), stage)
        available = available_memory(device)
        record["workers"] = {"measured": measured, "pass": passed, "available": available,
                             "measured_batches": measured_batches(windows, found),
                             "short": {str(n): workers_fit(n, measured, passed, available)
                                       for n in range(1, speakers.workers + 1)}}
        save(record)
        if record["workers"]["short"][str(speakers.workers)]:
            raise SystemExit(f"{speakers.workers} speaking workers do not fit (O15): "
                             + "; ".join(record["workers"]["short"][str(speakers.workers)]) + " — profile fewer")
    # 3. the round: its speaking, the selection readout twice, the pass
    directory = out / "round_0"
    directory.mkdir()
    spoken, speaking_s = timed(device, lambda: stage.speak(model, context, windows, settings, 0, directory, speakers,
                                                           stage=stage))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    record["round"] = {"speaking_s": speaking_s, "speaking": spoken, "memory_after_speaking": memory(device),
                       "groups_bytes": sum(p.stat().st_size for p in directory.glob("groups_*.pt")),
                       "speak_workers": speakers.workers if speakers is not None else 1}
    save(record)
    select = stage.selection(context, settings, "select")
    read = []
    for draw in (0, 1):
        values, seconds = timed(device, lambda: read_values(model, context, select, settings, speakers, stage, draw))
        read.append(values)
        record["round"][f"selection_readout_{draw}_s"] = seconds
        save(record)
    record["round"]["selection_windows"] = len(select)
    record["spread"] = spread(*read)
    save(record)
    passed, pass_s = timed(device, lambda: train_pass(model, context, optimizer, directory, settings,
                                                      np.random.default_rng([settings.seed, 0, 1]),
                                                      part_width=stage.part_width))
    record["round"].update(pass_s=pass_s, passed=passed, memory_after_pass=memory(device))
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    add_arguments(parser)
    parser.add_argument("--out", type=Path, required=True, help="a new scratch directory")
    args = parser.parse_args(argv)
    if args.speak_workers < 1:
        parser.error("--speak-workers is at least 1")
    source = args.start_campaign if args.start_campaign.is_absolute() else REPO_ROOT / args.start_campaign
    try:
        start = start_of(source, args.start_round, formal=False)        # a profile is no campaign: no formal start
    except ValueError as refused:
        parser.error(str(refused))
    prior_dir, instructions, executor, census, procedure_root, out = (
        p if p.is_absolute() else REPO_ROOT / p for p in (args.prior, args.instructions, args.executor, args.windows,
                                                          args.procedure_root, args.out))
    if out.exists():
        parser.error(f"{out} exists: a profile writes a new directory")
    settings = settings_from(args, start)
    _, opened, _ = require_conforming_closed_loop(instructions, executor)          # D69: the checks run here (D73)
    edges_reference = census / "conformance" / "edges.npz"
    checked_edges(edges_reference)                                                 # D104
    device = torch.device(args.device)
    context = open_context(prior_dir, instructions, executor, edges_reference,
                           torch.device("cpu") if args.speak_workers > 1 else device, procedure_root, formal=False)
    stage = stage_d()
    stage.start(context, settings)                        # refused by name before the workers and the output (D164)
    speakers = Speakers(context, settings, args.speak_workers, device, stage=stage) if args.speak_workers > 1 else None
    context = replace(context, device=device, base=context.base.to(device).eval())
    out.mkdir(parents=True)
    head = {"schema": PROFILE_SCHEMA, "started_utc": utc_now(), "git": git_state(), "checks": opened["checks"],
            "inputs": {"prior": str(prior_dir), "instructions": str(instructions), "executor": str(executor),
                       "windows": str(census), "procedure_root": str(procedure_root), "settings": asdict(settings),
                       "speak_workers": args.speak_workers, "approximation": (
                           "the two draws' paired standard error uses independent numbers: an upper bound of the noise "
                           "of comparing two rounds that share their numbers and differ little")}}

    def save(parts: dict[str, Any], **more: Any) -> None:
        write_json_atomic(out / "profile.json", json.loads(json.dumps({**head, **parts, **more}, default=str)))

    try:
        record = profile(context, settings, out, speakers, stage, save)
    finally:
        if speakers is not None:
            speakers.close()
    save(record, finished_utc=utc_now())
    print(json.dumps({"one_batch": {span: {k: b[k] for k in ("windows", "rows", "wall_s")}
                                    for span, b in record["one_batch"].items()},
                      "round": {k: v for k, v in record["round"].items() if k.endswith("_s")},
                      "spread": {span: {k: s[k] for k in ("units", "mean_0", "se_0", "paired_se")}
                                 for span, s in record["spread"].items()}}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
