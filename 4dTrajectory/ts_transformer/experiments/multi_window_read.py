"""Stage D: one round's model read on its campaign's select windows, a line a window (post-training D176 (2): the
per-window readout of stage D that a window list reads; the user, 2026-10-12: a set of the windows not landed).

THE READ: round ``--round``'s model (`multi_train.round_model`) read on the campaign's select windows by its own
selection readout (`multi_train.selection_readout`: the windows and random numbers of every round's readout), here or
by ``--speak-workers`` workers on ``--device`` (the rule of the CPU's batches left off: every round's readout is read
on the speak device alone, D191). The summed readout must be the round's own (``round.json``'s
``selection_readout``), or the run stops and writes nothing: the behaviour check on fixed inputs of the root's rule,
on the device and code of this read. From a clean checkout unless ``--smoke``; the closed loop's (D69) and the edge
features' (D104) checks first.

WRITES, into a new directory, ``config.json`` (`WINDOW_READ_SCHEMA`: the campaign, the round, its inputs, the commit,
the checks, the device and workers, the selection the lines index — the select seed, the windows an airport and the
spans, as the Training export's `StageExport.selection_fields` gives them — and the count of windows),
``per_window.jsonl`` (`multi_train.window_line`, one a window in place order) and ``SHA256SUMS``; the directory
read-only. ``window_list --multi DIR`` reads it.

    python run_ts.py multi_window_read --campaign 4dTrajectory/outputs/POOLED/multi/<campaign> --round <r> \\
        --out 4dTrajectory/outputs/POOLED/multi/<read id> [--speak-workers N] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.experiments.multi_train import multi_settings_of, round_model, selection_readout, stage_d
from ts_transformer.experiments.multi_training_export import STAGE_D
from ts_transformer.experiments.post_diagnose import seal
from ts_transformer.experiments.post_train import Speakers, done_rounds, inputs_here, open_context
from ts_transformer.experiments.post_window_loop import checked_edges
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: The format of a per-window read (``config.json``).
WINDOW_READ_SCHEMA = "ts-multi-window-read-v1"
#: The files of a read in its directory.
CONFIG, LINES = "config.json", "per_window.jsonl"


def window_read(campaign: Path, record: Mapping[str, Any], round_: int, out: Path, device: torch.device, workers: int,
                smoke: bool) -> dict[str, Any]:
    """The read (module docstring) of round ``round_``, written to ``out``; stopped by name, nothing written, unless its
    summed readout is the round's own."""
    settings = multi_settings_of(dict(record))
    stage = stage_d()
    git = git_state()
    inputs = inputs_here(record["inputs"])         # the paths as this checkout reads them (D157)
    instructions, executor = Path(inputs["instructions"]), Path(inputs["executor"])
    _, opened, _ = require_conforming_closed_loop(instructions, executor)          # D69: the checks run here (D73)
    edges_reference = Path(inputs["windows"]) / "conformance" / "edges.npz"
    checked_edges(edges_reference)                                                 # D104
    # with workers, the context is opened on the CPU and they are forked before this process uses the GPU
    context = open_context(Path(inputs["prior"]), instructions, executor, edges_reference,
                           torch.device("cpu") if workers > 1 else device, Path(inputs["procedure_root"]),
                           formal=not smoke, data=False, splits=("select",))
    speakers = (Speakers(context, replace(settings, cpu_batches=0, cpu_period=1, gpu_squares=0), workers, device,
                         stage=stage)
                if workers > 1 else None)
    try:
        context = replace(context, device=device, base=context.base.to(device).eval())
        windows = stage.selection(context, settings, "select")
        lines: list[dict[str, Any]] = []
        readout = selection_readout(round_model(context, settings, campaign, round_), context, windows, settings,
                                    speakers, stage=stage, lines=lines)
    finally:
        if speakers is not None:
            speakers.close()
    held = json.loads((campaign / f"round_{round_}" / "round.json").read_text(encoding="utf-8"))
    if json.loads(json.dumps(readout)) != held["selection_readout"]:
        raise SystemExit(f"round {round_}: this code's readout of its model on the {device.type} is not the round's "
                         f"(round.json): its lines would not be the round's — stopped")
    config = {"schema": WINDOW_READ_SCHEMA, "written_utc": utc_now(), "campaign": str(campaign), "round": round_,
              "inputs": record["inputs"], "git": git, "checks": opened["checks"], "smoke": smoke,
              "device": str(device), "speak_workers": workers, "selection": STAGE_D.selection_fields(settings),
              "windows": len(windows)}
    out.mkdir(parents=True)
    write_json_atomic(out / CONFIG, config)
    (out / LINES).write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    seal(out)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True, help="a multi_train campaign's directory")
    parser.add_argument("--round", type=int, required=True, help="a done round of the campaign")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--device", default="cuda", help="the read's device (a round's readout: its speak device)")
    parser.add_argument("--speak-workers", type=int, default=1, help="the read's processes (Speakers)")
    parser.add_argument("--smoke", action="store_true", help="SMOKE: allowed from a tree with changes; recorded")
    args = parser.parse_args(argv)

    def here(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    campaign, out = here(args.campaign), here(args.out)
    if out.exists():
        parser.error(f"{out} exists: a read is written into a new directory")
    if not 0 <= args.round < done_rounds(campaign):
        parser.error(f"{campaign}: round {args.round} is not a done round")
    if git_state()["dirty"] and not args.smoke:
        parser.error("the tree has uncommitted changes; a formal read runs from a commit")
    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    if record["inputs"]["smoke"] and not args.smoke:
        parser.error(f"{campaign} is a smoke campaign: only a --smoke read reads it")
    config = window_read(campaign, record, args.round, out, torch.device(args.device), args.speak_workers, args.smoke)
    print(json.dumps({"windows": config["windows"], "out": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
