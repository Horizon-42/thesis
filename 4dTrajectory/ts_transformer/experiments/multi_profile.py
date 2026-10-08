"""MC5: the memory measure of stage D at the formal size (multi-aircraft control §11 MC5, D179, D180; as post-training
O15) — what must be known before the campaign's first round: how many speaking workers fit. No criterion is applied
(D7): the report goes to the user, who sets the campaign.

WHAT IT MEASURES, from stage D's start (`multi_train.stage_d`: the round of stage C given, its token part at zero), on
round 0's draw at the formal counts (its batches by span recorded):

- **One speaking worker** (``--speak-workers`` of 2 or more, on ``--speak-device``, by default ``--device``): its peaks
  of the host and of the GPU (none on the CPU) on the first batch of each span and the batch of the most rows
  (`post_train.measured_batches`; each batch's time and GPU peak beside), spoken with round 0's model;
- **the pass** (`post_train.pass_memory_of`): this process's peak in one update on the worker's groups;
- for each N up to ``--speak-workers``, what does not fit (`post_train.workers_fit`; empty: N fits), each measured peak
  times `multi_train.MEASURE_MARGIN` (D179), the factor and the workers' device recorded. Where ``--speak-workers`` do
  not fit, the measure stops there by name, its record written.

It speaks no round, reads no readout and measures no spread (D179): a round's times and the readout's noise come from
the campaign's own round 0, and the select set's size is a setting. The record is ``profile.json`` (`MULTI_PROFILE_SCHEMA`),
written after each part, under ``--out`` (a new directory); a campaign reads it with ``--profile`` (`profiled_fit`).

    python run_ts.py multi_profile --prior <the base> --instructions <A34's artefact> --executor <its spec> \\
        --windows <stage C's census> --start-campaign <stage C's campaign> --start-round <r> --out <a new directory> \\
        (multi_train's settings) --speak-workers N [--speak-device cpu]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.experiments.multi_train import (
    MEASURE_MARGIN, MULTI_PROFILE_SCHEMA, add_arguments, settings_from, span_key, stage_d, with_margin,
)
from ts_transformer.experiments.post_profile import memory
from ts_transformer.experiments.post_train import (
    Context, Speakers, Stage, available_memory, measured_batches, open_context, pass_memory_of, speak_device_refused,
    start_of, workers_fit,
)
from ts_transformer.experiments.post_window_loop import checked_edges
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.post.scene import Window
from ts_transformer.repo_layout import REPO_ROOT, git_state


def rows_of(windows: Sequence[Window], batch: Sequence[int]) -> int:
    return sum(len(windows[p].signal_indices) for p in batch)


def profile(context: Context, settings: Any, out: Path, speakers: Speakers, stage: Stage,
            save: Callable[[dict[str, Any]], None] = lambda record: None) -> dict[str, Any]:
    """The record of the measure (module docstring), handed to ``save`` after each part."""
    device = context.device
    record: dict[str, Any] = {"memory_before": memory(device)}
    model, _ = stage.start(context, settings)
    windows, drawn = stage.draw(context, settings, 0)
    found = stage.batches(windows, settings)
    record["draw"] = drawn
    by_span: dict[str, list[int]] = defaultdict(list)
    for k, batch in enumerate(found):
        by_span[span_key(windows[batch[0]].span_s)].append(k)
    record["batches"] = {span: {"batches": len(ks), "windows": sum(len(found[k]) for k in ks),
                                "rows": [rows_of(windows, found[k]) for k in ks]} for span, ks in by_span.items()}
    save(record)
    with tempfile.TemporaryDirectory(prefix="multi_profile_measure_", dir=out) as scratch:
        measured = speakers.measure(0, Path(scratch), model)                    # round 0's model
        passed = pass_memory_of(context, settings, Path(scratch), stage)
    available = available_memory(device)
    margined = [with_margin(m, MEASURE_MARGIN) for m in (measured, passed)]
    record["workers"] = {"measured": measured, "pass": passed, "speak_device": str(speakers.device),
                         "margin": MEASURE_MARGIN, "available": available,
                         "measured_batches": measured_batches(windows, found),
                         "short": {str(n): workers_fit(n, *margined, available, held_now=True)
                                   for n in range(1, speakers.workers + 1)}}
    save(record)
    if record["workers"]["short"][str(speakers.workers)]:
        raise SystemExit(f"{speakers.workers} speaking workers do not fit (O15, measured peaks × {MEASURE_MARGIN}): "
                         + "; ".join(record["workers"]["short"][str(speakers.workers)]) + " — measure fewer")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    add_arguments(parser)
    parser.add_argument("--out", type=Path, required=True, help="a new scratch directory")
    args = parser.parse_args(argv)
    if args.speak_workers < 2:
        parser.error("--speak-workers is 2 or more: the measure is a speaking worker's")
    refused = speak_device_refused(args.speak_device, args.speak_workers, args.device)
    if refused is not None:
        parser.error(refused)
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
    context = open_context(prior_dir, instructions, executor, edges_reference, torch.device("cpu"), procedure_root,
                           formal=False)
    stage = stage_d()
    stage.start(context, settings)                        # refused by name before the workers and the output (D164)
    speakers = Speakers(context, settings, args.speak_workers, torch.device(args.speak_device or args.device),
                        stage=stage)
    context = replace(context, device=device, base=context.base.to(device).eval())
    out.mkdir(parents=True)
    head = {"schema": MULTI_PROFILE_SCHEMA, "started_utc": utc_now(), "git": git_state(), "checks": opened["checks"],
            "inputs": {"prior": str(prior_dir), "instructions": str(instructions), "executor": str(executor),
                       "windows": str(census), "procedure_root": str(procedure_root), "settings": asdict(settings),
                       "speak_workers": args.speak_workers}}

    def save(parts: dict[str, Any], **more: Any) -> None:
        write_json_atomic(out / "profile.json", json.loads(json.dumps({**head, **parts, **more}, default=str)))

    try:
        record = profile(context, settings, out, speakers, stage, save)
    finally:
        speakers.close()
    save(record, finished_utc=utc_now())
    workers = record["workers"]
    print(json.dumps({"speak_device": workers["speak_device"], "margin": workers["margin"],
                      "batches": workers["measured"]["batches"], "pass_peak": None if workers["pass"] is None else
                      workers["pass"]["peak"], "short": workers["short"]}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
