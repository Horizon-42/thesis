"""Stage D: a campaign's start and its rounds side by side on the select windows (the user, 2026-10-10) — the aircraft
landed, the reward and the pull to the base, each number named by its file and field.

THE START READ (``--read-start DIR``): the campaign's start model (stage D's `Stage.start`: the start round's weights,
the token part at zero) read on the campaign's select windows by its own selection readout
(`multi_train.selection_readout`: the windows and random numbers of every round's readout), here or by
``--speak-workers`` workers on ``--device`` (the rule of the CPU's batches left off: every round's readout is read on
the speak device alone, D191); written as ``start_read.json`` into a new directory, with the campaign's inputs. The
start read is made by other code than the rounds' readouts (it comes after them): the same run first reads the last
done round's model on the same windows and stops unless that is the round's own readout (``round.json``), the
behaviour check on fixed inputs of the root's rule (as `post_ceiling`'s draw 0, R64) — on the device and code of this
read. From a clean checkout unless ``--smoke``; the closed loop's (D69) and the edge features' (D104) checks first.

THE COMPARISON (``--out FILE``): each done round's ``round.json`` and, with ``--start DIR``, a start read of this
campaign (refused for other inputs or settings than the campaign's, the rule of the CPU's batches apart, or read on
another kind of device than a round's readout, ``round.json``'s ``speak_device``), each row:
- landed: ``outcomes.landed`` of ``aircraft`` of the readout;
- reward: ``reward_sum`` over ``aircraft`` (post-training D30: 1 a landing on a runway of the present direction, 0.9ⁿ
  after n go-arounds, 0 every other outcome and every loss of separation the aircraft answers for — at most the landed
  share; the campaign's log prints it as each airport's ``reward_mean``);
- lost separation: ``outcomes.lost_separation`` of ``aircraft``;
- the pull to the base: the round's pass ``kl`` (the KL to the base on the round's sampled words, the mean over its
  updates, before its weight `post.loss.KL_WEIGHT`), beside its ``data`` (the single-aircraft data term), ``surrogate``
  and ``updates``; the start has no pass, a pass of no update no means.
Each airport's cells of span × kind are summed and checked against its ``all`` cell, and every row reads the same
aircraft of each airport, both refused by name otherwise; the sums over the airports are the row's ``all``. Written as
one JSON (``ts-multi-compare-v1``) and printed as a table.

    python run_ts.py multi_compare --campaign 4dTrajectory/outputs/POOLED/multi/<campaign> --read-start <a new dir> \\
        [--speak-workers N] [--device cuda]
    python run_ts.py multi_compare --campaign 4dTrajectory/outputs/POOLED/multi/<campaign> [--start <its dir>] \\
        --out <a new file>.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.experiments.multi_train import (
    multi_settings_of, round_model, selection_readout, stage_d, window_record,
)
from ts_transformer.experiments.post_train import Speakers, but_resumable, done_rounds, inputs_here, open_context
from ts_transformer.experiments.post_window_loop import LANDED, LOST_SEPARATION, checked_edges
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: The format of a start read (``start_read.json``).
START_READ_SCHEMA = "ts-multi-start-read-v1"
#: The format of a comparison.
COMPARE_SCHEMA = "ts-multi-compare-v1"
#: The file of a start read in its directory.
START_READ = "start_read.json"
#: A readout's cell, apart from its spans: every span and kind summed, and the number the campaign's log prints.
ALL, LOGGED = "all", "reward_mean"
#: A round's pass: the means compared beside its updates (`post.loss.stacked`; a pass with no update has none).
PASS_MEANS = ("kl", "data", "surrogate")


def counted(readout: Mapping[str, Any], source: str) -> dict[str, dict[str, float]]:
    """Each airport of a selection readout (`multi_train.selection_readout`): its aircraft, landed, losses of separation
    and reward sum, summed over its cells of span × kind and refused by name unless they are its ``all`` cell's."""
    out = {}
    for code, airport in readout.items():
        cells = [cell for span, kinds in airport.items() if span not in (ALL, LOGGED)
                 for kind, cell in kinds.items() if kind != ALL]
        summed = {"aircraft": sum(c["aircraft"] for c in cells),
                  "landed": sum(c["outcomes"].get(LANDED, 0) for c in cells),
                  "lost_separation": sum(c["outcomes"].get(LOST_SEPARATION, 0) for c in cells),
                  "reward_sum": math.fsum(c["reward_sum"] for c in cells)}
        whole = airport[ALL][ALL]
        held = {"aircraft": whole["aircraft"], "landed": whole["outcomes"].get(LANDED, 0),
                "lost_separation": whole["outcomes"].get(LOST_SEPARATION, 0), "reward_sum": whole["reward_sum"]}
        if ({k: summed[k] for k in ("aircraft", "landed", "lost_separation")}
                != {k: held[k] for k in ("aircraft", "landed", "lost_separation")}
                or not math.isclose(summed["reward_sum"], held["reward_sum"], rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError(f"{source}: {code}'s cells of span and kind are not its all cell ({summed} against {held})")
        out[code] = summed
    return out


def pass_numbers(passed: Mapping[str, Any]) -> dict[str, Any]:
    """A round's pass (``round.json``'s ``pass``): its updates and, when it made any, the means of its parts
    (`PASS_MEANS`; `post_train` records none for a pass of no update)."""
    return {"updates": passed["updates"], **({key: passed[key] for key in PASS_MEANS} if passed["updates"] else {})}


def _row(model: str, source: str, readout: Mapping[str, Any], passed: Mapping[str, Any] | None) -> dict[str, Any]:
    airports = counted(readout, source)
    whole = {key: sum(a[key] for a in airports.values()) for key in ("aircraft", "landed", "lost_separation")}
    whole["reward_sum"] = math.fsum(a["reward_sum"] for a in airports.values())
    return {"model": model, "source": source, "airports": airports, ALL: whole,
            "pass": None if passed is None else pass_numbers(passed)}


def compared(campaign: Path, record: Mapping[str, Any], start: Path | None) -> dict[str, Any]:
    """The comparison (module docstring) of ``campaign``'s done rounds and, with ``start``, its start read."""
    rows, kind = [], None
    if start is not None:
        held = json.loads((start / START_READ).read_text(encoding="utf-8"))
        if held["schema"] != START_READ_SCHEMA:
            raise ValueError(f"{start}: a {held['schema']} file, not {START_READ_SCHEMA}")
        if held["smoke"] and not record["inputs"]["smoke"]:
            raise ValueError(f"{start}: a smoke start read, the campaign a formal one")
        if but_resumable(inputs_here(held["inputs"])) != but_resumable(inputs_here(record["inputs"])):
            raise ValueError(f"{start}: a start read of other inputs or settings than {campaign}'s")
        rows.append(_row("start", str(start / START_READ), held["readout"], None))
        kind = torch.device(held["device"]).type
    for round_ in range(done_rounds(campaign)):
        path = campaign / f"round_{round_}" / "round.json"
        held = json.loads(path.read_text(encoding="utf-8"))
        if kind is not None and torch.device(held["speak_device"]).type != kind:
            raise ValueError(f"{start}: read on the {kind}, round {round_}'s readout on {held['speak_device']}: a read "
                             f"on another kind of device")
        rows.append(_row(f"round {round_}", str(path), held["selection_readout"], held["pass"]))
    if not rows:
        raise ValueError(f"{campaign}: no done round and no start read: nothing to compare")
    for row in rows[1:]:
        if {c: a["aircraft"] for c, a in row["airports"].items()} != {c: a["aircraft"] for c, a in
                                                                      rows[0]["airports"].items()}:
            raise ValueError(f"{row['source']}: other aircraft than {rows[0]['source']}'s: a read of other windows")
    return {"schema": COMPARE_SCHEMA, "written_utc": utc_now(), "campaign": str(campaign),
            "start": None if start is None else str(start), "rows": rows}


def table(comparison: Mapping[str, Any]) -> str:
    """The comparison as a table: each row's landed share by airport and over all, its reward and losses of separation
    per aircraft, and its pass's pull to the base."""
    codes = list(comparison["rows"][0]["airports"])
    lines = ["model | " + " | ".join(f"{c} landed" for c in codes) + " | all landed | reward | lost sep. | kl | data"]
    for row in comparison["rows"]:
        whole, passed = row[ALL], row["pass"]
        cells = [f"{a['landed'] / a['aircraft']:.1%}" for a in row["airports"].values()]
        cells.append(f"{whole['landed']}/{whole['aircraft']} = {whole['landed'] / whole['aircraft']:.1%}")
        cells.append(f"{whole['reward_sum'] / whole['aircraft']:.4f}")
        cells.append(f"{whole['lost_separation'] / whole['aircraft']:.1%}")
        cells += ([f"{passed['kl']:.4f}", f"{passed['data']:.4f}"] if passed is not None and passed["updates"] else
                  ["–", "–"])
        lines.append(f"{row['model']} | " + " | ".join(cells))
    return "\n".join(lines)


def read_start(campaign: Path, record: Mapping[str, Any], out: Path, device: torch.device, workers: int,
               smoke: bool) -> dict[str, Any]:
    """The start read (module docstring), written to ``out``, after the last done round's model is read on the same
    windows as its round's readout (refused by name otherwise, nothing written)."""
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
    speakers = (Speakers(context, replace(settings, cpu_batches=0, cpu_period=1,
                                         gpu_squares=0), workers, device, stage=stage)
                if workers > 1 else None)
    checked = done_rounds(campaign) - 1
    try:
        context = replace(context, device=device, base=context.base.to(device).eval())
        windows = stage.selection(context, settings, "select")
        again = selection_readout(round_model(context, settings, campaign, checked), context, windows, settings,
                                  speakers, stage=stage)
        held = json.loads((campaign / f"round_{checked}" / "round.json").read_text(encoding="utf-8"))
        if json.loads(json.dumps(again)) != held["selection_readout"]:
            raise SystemExit(f"round {checked}: this code's readout of its model on the {device.type} is not the "
                             f"round's (round.json): a start read here would not be comparable — stopped")
        model, _ = stage.start(context, settings)
        readout = selection_readout(model, context, windows, settings, speakers, stage=stage)
    finally:
        if speakers is not None:
            speakers.close()
    out.mkdir(parents=True)
    held = {"schema": START_READ_SCHEMA, "written_utc": utc_now(), "campaign": str(campaign),
            "inputs": record["inputs"], "git": git, "checks": opened["checks"], "reread_round": checked,
            "smoke": smoke, "device": str(device),
            "speak_workers": workers, "windows": [window_record(w) for w in windows], "readout": readout}
    write_json_atomic(out / START_READ, held)
    return held


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True, help="a multi_train campaign's directory")
    parser.add_argument("--read-start", type=Path, help="read the start: a new directory")
    parser.add_argument("--start", type=Path, help="a start read of this campaign (its directory)")
    parser.add_argument("--out", type=Path, help="the comparison: a new JSON file")
    parser.add_argument("--device", help="the start read's device (default cuda)")
    parser.add_argument("--speak-workers", type=int, help="the start read's processes (Speakers; default 1)")
    parser.add_argument("--smoke", action="store_true", help="SMOKE: a start read from a tree with changes too")
    args = parser.parse_args(argv)
    if (args.read_start is None) == (args.out is None):
        parser.error("one of --read-start (the start read) and --out (the comparison)")
    if args.read_start is not None and args.start is not None:
        parser.error("--start is the comparison's; a start read writes one")
    if args.out is not None and (args.device is not None or args.speak_workers is not None or args.smoke):
        parser.error("--device, --speak-workers and --smoke are the start read's")
    workers = 1 if args.speak_workers is None else args.speak_workers
    if workers < 1:
        parser.error("--speak-workers is at least 1")
    campaign = args.campaign if args.campaign.is_absolute() else REPO_ROOT / args.campaign
    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    try:
        multi_settings_of(record)
    except ValueError as refused:
        parser.error(f"{campaign}: {refused}")
    if args.read_start is not None:
        out = args.read_start if args.read_start.is_absolute() else REPO_ROOT / args.read_start
        if out.exists():
            parser.error(f"{out} exists: a start read writes into a new directory")
        if record["inputs"]["smoke"] and not args.smoke:
            parser.error(f"{campaign} is a smoke campaign: only a --smoke start read reads it")
        if git_state()["dirty"] and not args.smoke:
            parser.error("a start read that is not a smoke needs a clean tree")
        if done_rounds(campaign) < 1:
            parser.error(f"{campaign} has no done round: a start read is checked against a round's readout")
        held = read_start(campaign, record, out, torch.device(args.device or "cuda"), workers, args.smoke)
        print(json.dumps({"out": str(out), "landed": {code: a["landed"] for code, a in
                                                      counted(held["readout"], str(out)).items()}}), flush=True)
        return 0
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists: the comparison writes a new file")
    start = None if args.start is None else (args.start if args.start.is_absolute() else REPO_ROOT / args.start)
    try:
        comparison = compared(campaign, record, start)
    except ValueError as refused:
        parser.error(str(refused))
    write_json_atomic(out, comparison)
    print(table(comparison), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
