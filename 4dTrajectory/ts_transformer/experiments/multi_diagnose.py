"""D211: the diagnostic readout of stage D — whether a model of stage D uses the traffic it reads, what its rounds
changed against its start, and which losses of separation are left (multi-aircraft control D211; the user,
2026-10-15, after D200: the losses of separation fell in none of its arms). Stage C's readout (D175,
`post_diagnose`) on stage D's windows. Select days only, never val (outline D85); no training.

THE READS: one round of a campaign of stage D (``--round``, its checkpoint by `multi_train.round_model`) reads the
campaign's selection readout's windows (`multi_train.selection_windows`) with the readout's numbers (each commanded
aircraft its own, `multi_train.readout_numbers`) three times in one process: ``on``, the round's model; ``off``, the
same model with the output layer of every traffic module set to zero (`post_diagnose.traffic_off`), the rest
unchanged; ``start``, the campaign's start (`Stage.start`: from the base, the base with zero-output traffic modules,
D200; from a round of stage C, that round). The read ``on`` must give the round's ``round.json`` readout again, or the
run stops by name. A campaign of version 2 (D212; D216) is read four times: ``on``; ``joint_off``, the same model with
the output layer of every joint module set to zero (`multi.joint.joint_off`: no commanded aircraft reads the others'
rows); ``off``, besides that the traffic modules' (the model reads no other aircraft); ``start`` (its joint modules at
zero, so the start reads as version 1 of the same start).

WHAT A READ WRITES: one line per window and read in ``per_window.jsonl`` (`multi_train.window_line` and its read) and,
for each commanded aircraft that lost separation, its place in the window, its key, the other aircraft, the window's
step of the loss and the fields of D175 (`post_diagnose.loss_fields`): (1) the other aircraft, `COMMANDED` when it is
commanded too; (2) the distance to its runway's threshold at the loss; (3) the time from its own first predicted step;
(4) against a recorded aircraft only, in conflict at its first predicted step. Field 5 is not read (stage D has no
readout of draws).

THE OUTPUTS, into a new directory: ``intent.json`` (``--intent``'s decision rules, written before the reads),
``config.json``, ``per_window.jsonl``, ``summary.json`` (each read's commanded aircraft: landed, lost separation, mean
reward and outcomes, in all, by airport and by span; the paired differences on − off, on − start and off − start, and
of version 2 on − joint_off, with a 95 % interval from 2,000 bootstrap resamples of the windows, an aircraft's outcome
counted in its window),
``failures.json`` (each field's counts and shares over the aircraft lost, of each read, in all, by airport and by
span), ``log.jsonl`` and ``SHA256SUMS``; the directory read-only at the end. No criterion is applied here: the
intent's rules are read by whoever reports.

THE CHECKS run first: the closed loop's (with the labeller's and the executor's, D69) and the edge features' reference
of the census (D104). A clean tree unless ``--smoke``. ``--speak-workers`` reads the batches in worker processes
(`Speakers.read`, every batch on ``--device``: the readout's rule, D191).

    python run_ts.py multi_diagnose --campaign 4dTrajectory/outputs/POOLED/multi/<campaign id> --round 3 \\
        --intent <intent.json> --out 4dTrajectory/outputs/POOLED/multi/diagnose_<campaign>_r<round>_<date> \\
        [--speak-workers 3]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
import torch

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.experiments.multi_train import (
    MULTI_CAMPAIGN_SCHEMA, MULTI_CAMPAIGN_SCHEMA_V2, multi_settings_of, round_model, selection_read, selection_summed,
    span_key, stage_d, window_line, window_record,
)
from ts_transformer.experiments.post_diagnose import (
    BOOTSTRAP_RESAMPLES, COMMANDED, INTERVAL, OTHER_CLASSES, STRAIGHT_LINE_S, failures, loss_fields, seal, traffic_off,
)
from ts_transformer.experiments.post_train import Speakers, done_rounds, inputs_here, open_context
from ts_transformer.experiments.post_window_loop import LANDED, LOST_SEPARATION, checked_edges
from ts_transformer.instructions.artefact import load_day_split
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.multi.joint import joint_off
from ts_transformer.post.landings import window_landings
from ts_transformer.prior.source import airport_landings
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: The format of the readout's files.
MULTI_DIAGNOSE_SCHEMA = "ts-multi-diagnose-v1"
#: The split read: the select days (D85: never val).
SPLIT = "select"
#: The three reads (module docstring), in the order read; version 2's four (D216).
ON, OFF, START, JOINT_OFF = "on", "off", "start", "joint_off"
READS = (ON, OFF, START)
READS_V2 = (ON, JOINT_OFF, OFF, START)
#: The paired differences of `summary`; version 2's (D216).
PAIRS = ((ON, OFF), (ON, START), (OFF, START))
PAIRS_V2 = (*PAIRS, (ON, JOINT_OFF))
#: Field 1's classes in stage D (D175's and `COMMANDED`).
D_OTHER_CLASSES = (*OTHER_CLASSES, COMMANDED)


class End(NamedTuple):
    """What the summary reads of a commanded aircraft's end (its `WindowResult` without its words and states)."""

    outcome: str
    reward: float


def lost_lines(window: Any, ends: Sequence[Any], context: Any, rosters: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The commanded aircraft of ``window`` that lost separation (``ends``: their `WindowResult`s in the window's order),
    each with its place in the window, key, other aircraft, the window's step of the loss and its fields
    (`post_diagnose.loss_fields`; no field 5)."""
    code = window.scene.geometry.code
    landings = window_landings(window, rosters[code])
    return [{"member": m, "aircraft": record.key, "other": end.other, "loss_step": end.loss_step,
             "fields": loss_fields(window, m, end, context.separations[code], context.finals[code], landings,
                                   context.words.spec.step_s, None)}
            for m, (record, end) in enumerate(zip(window.commanded_all, ends, strict=True))
            if end.outcome == LOST_SEPARATION]


def reading(windows: Sequence[Any], ends: Sequence[Sequence[Any]]) -> dict[str, Any]:
    """Landed, lost separation, mean reward and outcomes of the commanded aircraft of ``windows`` (``ends``: each
    window's `End`s), in all, by airport and by span."""

    def part(places: Sequence[int]) -> dict[str, Any]:
        mine = [e for w in places for e in ends[w]]
        outcomes = Counter(e.outcome for e in mine)
        return {"windows": len(places), "aircraft": len(mine), "landed": outcomes[LANDED] / len(mine),
                "lost_separation": outcomes[LOST_SEPARATION] / len(mine),
                "reward_mean": float(np.mean([e.reward for e in mine])), "outcomes": dict(outcomes)}

    return {"all": part(range(len(windows))), **_parts(windows, part)}


def paired(windows: Sequence[Any], a: Sequence[Sequence[Any]], b: Sequence[Sequence[Any]], rng: np.random.Generator
           ) -> dict[str, Any]:
    """The paired differences a − b of the landed share, the lost-separation share and the mean reward of the
    commanded aircraft of ``windows`` (each window's `End`s in ``a`` and ``b``), each with its `INTERVAL`
    interval from `BOOTSTRAP_RESAMPLES` resamples of the windows (an aircraft is counted in its window), in all, by
    airport and by span."""
    count = np.array([len(e) for e in a], dtype=np.float64)
    values = {"landed": [np.array([sum(x.outcome == LANDED for x in e) for e in s], dtype=np.float64) for s in (a, b)],
              "lost_separation": [np.array([sum(x.outcome == LOST_SEPARATION for x in e) for e in s], dtype=np.float64)
                                  for s in (a, b)],
              "reward_mean": [np.array([sum(x.reward for x in e) for e in s], dtype=np.float64) for s in (a, b)]}
    tail = (1.0 - INTERVAL) / 2.0

    def part(places: Sequence[int]) -> dict[str, Any]:
        places = np.asarray(places)
        draws = places[rng.integers(0, len(places), size=(BOOTSTRAP_RESAMPLES, len(places)))]
        out = {}
        for name, (x, y) in values.items():
            d = x - y
            resampled = d[draws].sum(axis=1) / count[draws].sum(axis=1)
            out[name] = {"difference": float(d[places].sum() / count[places].sum()),
                         "interval": [float(np.quantile(resampled, tail)), float(np.quantile(resampled, 1.0 - tail))]}
        return out

    return {"all": part(range(len(windows))), **_parts(windows, part)}


def _parts(windows: Sequence[Any], part: Any) -> dict[str, Any]:
    """``part`` of the places of each airport and of each span of ``windows``."""
    codes = [w.scene.geometry.code for w in windows]
    spans = [span_key(w.span_s) for w in windows]
    return {"airports": {code: part([p for p, c in enumerate(codes) if c == code]) for code in sorted(set(codes))},
            "spans": {span: part([p for p, s in enumerate(spans) if s == span])
                      for span in sorted(set(spans), key=float)}}


def counted_failures(lines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Each field's counts and shares (`post_diagnose.failures`) over the aircraft lost in the window lines ``lines``,
    in all, by airport and by span (every span read; the airports with a loss). Field 4's shares are over the losses
    against a recorded aircraft only, field 1's over every loss."""
    lost = [{"airport": line["airport"], "span": line["span"], "outcome": LOST_SEPARATION, "fields": item["fields"]}
            for line in lines for item in line["lost"]]
    out = failures(lost, D_OTHER_CLASSES)
    out["spans"] = {span: failures([x for x in lost if x["span"] == span], D_OTHER_CLASSES)["all"]
                    for span in sorted({line["span"] for line in lines}, key=float)}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True, help="a multi_train campaign's directory")
    parser.add_argument("--round", type=int, required=True, help="the round read (its checkpoint)")
    parser.add_argument("--intent", type=Path, required=True, help="the decision rules, a JSON object (intent.json)")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--device", default="cuda", help="the reads' device (a round's readout: its speak device)")
    parser.add_argument("--speak-workers", type=int, default=1, help="processes that read the batches (Speakers)")
    parser.add_argument("--smoke", action="store_true", help="SMOKE: a tree with changes too; no result")
    args = parser.parse_args(argv)
    campaign, out, intent_path = (p if p.is_absolute() else REPO_ROOT / p for p in (args.campaign, args.out, args.intent))
    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    if record["schema"] not in (MULTI_CAMPAIGN_SCHEMA, MULTI_CAMPAIGN_SCHEMA_V2):
        parser.error(f"{campaign} is a {record['schema']} campaign, not {MULTI_CAMPAIGN_SCHEMA} or "
                     f"{MULTI_CAMPAIGN_SCHEMA_V2}")
    if record["inputs"]["smoke"] and not args.smoke:
        parser.error(f"{campaign} is a smoke campaign: only a --smoke readout reads it")
    done, settings = done_rounds(campaign), multi_settings_of(record)
    reads, pairs = (READS_V2, PAIRS_V2) if settings.joint else (READS, PAIRS)
    if not 0 <= args.round < done:
        parser.error(f"{campaign} holds the checkpoints of rounds 0–{done - 1}: --round {args.round}")
    if args.speak_workers < 1:
        parser.error("--speak-workers is at least 1")
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    if not isinstance(intent, dict) or not intent:
        parser.error(f"{intent_path}: the decision rules are a JSON object")
    if out.exists():
        parser.error(f"{out} exists: the readout writes into a new directory")
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("a readout that is not a smoke needs a clean tree")
    inputs = inputs_here(record["inputs"])         # the paths as this checkout reads them (D157)
    instructions, executor = Path(inputs["instructions"]), Path(inputs["executor"])
    _, opened, _ = require_conforming_closed_loop(instructions, executor)          # D69: the checks run here (D73)
    edges_reference = Path(inputs["windows"]) / "conformance" / "edges.npz"
    checked_edges(edges_reference)                                                 # D104
    device = torch.device(args.device)
    stage = stage_d()
    # with workers, the context is opened on the CPU and they are forked before this process uses the GPU
    context = open_context(Path(inputs["prior"]), instructions, executor, edges_reference,
                           torch.device("cpu") if args.speak_workers > 1 else device, Path(inputs["procedure_root"]),
                           formal=not args.smoke, data=False, splits=(SPLIT,))   # D132
    # every batch on the speak device: the readout's rule (D191)
    speakers = (Speakers(context, replace(settings, cpu_batches=0, cpu_period=1, gpu_squares=0), args.speak_workers,
                         device, stage=stage)
                if args.speak_workers > 1 else None)
    try:
        context = replace(context, device=device, base=context.base.to(device).eval())
        # each airport's roster landings, for field 1's window landings (D105, D175)
        rosters = airport_landings(context.geometries, load_day_split(instructions))
        windows = stage.selection(context, settings, SPLIT)
        out.mkdir(parents=True)
        write_json_atomic(out / "intent.json", intent)                             # the rules, before the reads
        write_json_atomic(out / "config.json", {
            "schema": MULTI_DIAGNOSE_SCHEMA, "written_utc": utc_now(), "campaign": str(campaign), "round": args.round,
            "reads": list(reads), "settings": inputs["settings"], "start": record["inputs"]["settings"]["start"],
            "instructions": str(instructions), "executor": str(executor), "checks": opened["checks"], "git": git,
            "smoke": args.smoke, "device": str(device), "speak_workers": args.speak_workers,
            "straight_line_s": STRAIGHT_LINE_S, "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "interval": INTERVAL,
                                                              "seed": [settings.seed, 1 << 29]},
            "windows": [window_record(w) for w in windows]})

        def log(event: Mapping[str, Any]) -> None:
            line = json.dumps({"utc": utc_now(), **event})
            with open(out / "log.jsonl", "a", encoding="utf-8") as f:
                f.write(line + "\n")
            print(line, flush=True)

        model = round_model(context, settings, campaign, args.round)
        ends: dict[str, list[list[Any]]] = {}
        lines: dict[str, list[dict[str, Any]]] = {}
        for read in reads:
            if read == JOINT_OFF:
                joint_off(model)
            elif read == OFF:                          # version 2: its joint modules are off from the read before
                traffic_off(model)
            elif read == START:
                model = stage.start(context, settings)[0].eval()
            places, got = selection_read(model, context, windows, settings, speakers, stage=stage, split=SPLIT)
            if read == ON:
                held = json.loads((campaign / f"round_{args.round}" / "round.json").read_text(encoding="utf-8"))
                if json.loads(json.dumps(selection_summed(windows, places, got))) != held["selection_readout"]:
                    raise SystemExit(f"round {args.round}: the read with traffic on is not the round's selection "
                                     f"readout (round.json); the readout is not the campaign's — stopped")
                log({"read": ON, "checked": "the round's selection readout"})
            by_place: dict[int, list[Any]] = {}
            lines[read] = []
            for batch, parts in zip(places, got, strict=True):
                for place, (window_ends, losses, _, _) in zip(batch, parts, strict=True):
                    window = windows[place]
                    by_place[place] = window_ends
                    lines[read].append({"read": read, **window_line(place, window, window_ends, losses),
                                        "lost": lost_lines(window, window_ends, context, rosters)})
            ends[read] = [[End(e.outcome, e.reward) for e in by_place[p]] for p in range(len(windows))]
            lines[read].sort(key=lambda line: line["place"])
            with open(out / "per_window.jsonl", "a", encoding="utf-8") as f:
                f.writelines(json.dumps(line) + "\n" for line in lines[read])
            summary = reading(windows, ends[read])["all"]
            log({"read": read, "landed": summary["landed"], "lost_separation": summary["lost_separation"]})
    finally:
        if speakers is not None:
            speakers.close()
    rng = np.random.default_rng([settings.seed, 1 << 29])
    write_json_atomic(out / "summary.json", {
        "schema": MULTI_DIAGNOSE_SCHEMA, "written_utc": utc_now(), "round": args.round,
        "reads": {read: reading(windows, ends[read]) for read in reads},
        "differences": {f"{a}-{b}": paired(windows, ends[a], ends[b], rng) for a, b in pairs}})
    write_json_atomic(out / "failures.json", {"schema": MULTI_DIAGNOSE_SCHEMA,
                                              **{read: counted_failures(lines[read]) for read in reads}})
    log({"written": "summary.json, failures.json"})
    seal(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
