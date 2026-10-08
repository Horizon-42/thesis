"""D175: the diagnostic readout — whether a model of stage C uses the traffic it reads (G2), and which losses of
separation are left (G1) (post-training C24; stage C's request, the user's choice of 2026-10-08). Select days only,
never val (outline D85); no training.

THE READS: one round of a campaign (``--round``, its checkpoint by `post_train.round_model`) reads the selection
readout's windows (`post_train.selection_windows`) with the readout's numbers (`readout_numbers`, draw 0) twice in one
process: first with its traffic attention, then with the output layer of every traffic module set to zero
(`traffic_off`: the start's initialisation, post-training §2 item 1, prior §7 item 5), the rest of the model unchanged.
The read with traffic on must give the round's ``round.json`` readout again, refused by name otherwise (as D161's draw
0); the read with traffic off follows. A model whose traffic output is zero (a campaign's start) is not read: with
``--ceiling`` (a D161 readout of the same campaign that read its start), its results per window are that readout's draw
0 of the start, which read the same windows with the same numbers; the windows are compared by their flights and start
times, refused by name where they differ.

WHAT A READ WRITES: one line per window in ``per_window.jsonl`` (its read, place, flight, airport, kind, outcome, reward,
go-arounds, the loss's step, its other aircraft and ``loss_reads_fault``, D114); for each window that lost separation,
five fields (`loss_fields`): (1) the other aircraft (`other_class`), (2) the commanded aircraft's horizontal distance to
its runway's threshold at the loss (`DISTANCE_BINS`), (3) the time from its first predicted step to the loss
(`TIME_BINS`), (4) in conflict at the start (`straight_line_loss`), (5) with a ``--ceiling`` that read this round,
whether the window lost separation in every one of its draws (absent otherwise). A value on a bin's edge falls in the
upper bin.

THE OUTPUTS, into a new directory: ``intent.json`` (``--intent``'s decision rules, written before the read),
``config.json``, ``per_window.jsonl``, ``summary.json`` (landed, lost separation and mean reward, in all and by airport,
of each read and of the start; the paired differences between the reads with a 95 % interval from 2,000 bootstrap
resamples of the windows), ``failures.json`` (each field's counts and shares, in all and by airport, of each read),
``log.jsonl`` and ``SHA256SUMS``; the directory read-only at the end (`seal`). No criterion is applied here: the intent's
rules are read by whoever reports.

THE CHECKS run first: the closed loop's (with the labeller's and the executor's, D69) and the edge features' reference
of the census (D104); the campaign's recorded paths as this checkout reads them (`inputs_here`). A clean tree unless
``--smoke``. ``--speak-workers`` reads the batches in worker processes (`Speakers.read`: the same ends as one process).

    python run_ts.py post_diagnose --campaign 4dTrajectory/outputs/POOLED/post/<campaign id> --round 5 \\
        --intent <intent.json> --out 4dTrajectory/outputs/POOLED/post/diagnose_<campaign>_r<round>_<date> \\
        [--ceiling 4dTrajectory/outputs/POOLED/post/<ceiling id>] [--speak-workers 4]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.experiments.post_ceiling import CEILING_SCHEMA, START, in_window_order
from ts_transformer.experiments.post_train import (
    CAMPAIGN_SCHEMA, Speakers, batches, counted_ends, done_rounds, inputs_here, open_context, read_batch, round_model,
    selection_windows, settings_of, window_record,
)
from ts_transformer.experiments.post_window_loop import LOST_SEPARATION, WindowResult, checked_edges
from ts_transformer.inference.runway_schedule import Separation
from ts_transformer.instructions.words import RUNWAY
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.post.branches import BRANCH_EVERY_S, branch_rows
from ts_transformer.post.landings import roster_key, window_landings
from ts_transformer.post.reward import LANDED
from ts_transformer.post.runways import approach_clock_m
from ts_transformer.post.scene import AircraftAt, Window
from ts_transformer.post.traffic import commanded_loss, joined, traffic
from ts_transformer.post.traffic_attention import traffic_modules
from ts_transformer.prior.landings import LandingIndex
from ts_transformer.prior.model import Prior
from ts_transformer.prior.procedure import Final
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: The format of the readout's files.
DIAGNOSE_SCHEMA = "ts-post-diagnose-v1"
#: The split read: the select days (D85: never val).
SPLIT = "select"
#: The two reads of a model (module docstring).
ON, OFF = "on", "off"
#: Field 1 (D175): the other aircraft of a loss.
NO_LANDING, LEADER, FOLLOWER, OTHER_RUNWAY = "no_landing", "leader", "follower", "other_runway"
OTHER_CLASSES = (NO_LANDING, LEADER, FOLLOWER, OTHER_RUNWAY)
#: Field 2: the commanded aircraft's distance to its threshold at the loss, km (a value on an edge: the upper bin).
DISTANCE_EDGES_KM = (10.0, 20.0, 40.0)
DISTANCE_BINS = ("0-10", "10-20", "20-40", "over 40")
#: Field 3: the time from the first predicted step to the loss, s (a value on an edge: the upper bin).
TIME_EDGES_S = (60.0, 120.0, 300.0)
TIME_BINS = ("below 60", "60-120", "120-300", "over 300")
#: Field 4: how long the commanded aircraft's straight line is judged after its first predicted step (D175: D37's
#: interval, over which a straight line stands for a turning arrival).
STRAIGHT_LINE_S = BRANCH_EVERY_S
#: The paired differences' interval (D175): the bootstrap resamples of the windows and the share covered.
BOOTSTRAP_RESAMPLES = 2000
INTERVAL = 0.95


# ---- the model with its traffic off
def traffic_off(model: Prior) -> Prior:
    """``model`` with the output layer of every traffic module (`traffic_modules`) set to zero, in place: the traffic
    attention adds nothing to any layer, as at a campaign's start (module docstring); every other weight unchanged."""
    with torch.no_grad():
        for module in traffic_modules(model):
            module.out.weight.zero_()
            module.out.bias.zero_()
    return model


# ---- the five fields of a loss of separation
def bin_of(value: float, edges: Sequence[float], names: Sequence[str]) -> str:
    """The bin of ``value`` between ``edges`` (a value on an edge falls in the upper bin)."""
    return names[int(np.searchsorted(np.asarray(edges), value, side="right"))]


def runway_at_loss(result: WindowResult) -> int:
    """The commanded aircraft's runway in force at its loss: the last candidate its runway words said (an "unchanged"
    or a go-around keeps the runway in force, `prior.speaker.runway_after`). Its window ends at the loss, so every word
    it was said came before it; a loss is judged after a row said and the first predicted step always says a candidate,
    so D175's "else its recorded runway" never binds here — a loss with none said is refused."""
    said = [int(word) for word in result.words[:, RUNWAY] if word >= 0]
    if not said:
        raise ValueError(f"flight {result.index}: a loss of separation before any runway was said")
    return said[-1]


def state_at(result: WindowResult, step: int, every: int) -> np.ndarray:
    """``[3]`` the commanded aircraft's e, n and MSL height at the window's step ``step`` (its 2 s rows from row 0,
    ``every`` rows a step)."""
    return np.asarray(result.states[step * every, :3], dtype=np.float64)


def other_class(window: Window, other: str, runway_index: int, at: np.ndarray, time_s: float, separation: Separation,
                landings: LandingIndex) -> str:
    """Field 1 (D175): `NO_LANDING` when the window's landings (D105) hold none of the other aircraft ``other``; else
    `LEADER` or `FOLLOWER` when it lands on the commanded aircraft's runway ``runway_index`` or on one separated as one
    (`Separation.one_runway`), ahead of it or behind it on the approach clock at ``time_s`` (the commanded aircraft at
    ``at``); else `OTHER_RUNWAY`."""
    geometry = window.scene.geometry
    if roster_key(other, geometry.code) not in {landing.flight_key for landing in landings.landings}:
        return NO_LANDING
    record = window.scene.flight(other)
    if not separation.one_runway(geometry.candidates[runway_index].ident, geometry.candidates[record.runway_index].ident):
        return OTHER_RUNWAY
    row = record.row_at(time_s)
    mine = float(approach_clock_m(separation, geometry, runway_index, at[0], at[1]))
    theirs = float(approach_clock_m(separation, geometry, record.runway_index, record.e_m[row], record.n_m[row]))
    return LEADER if theirs > mine else FOLLOWER


def straight_line_loss(window: Window, other: str, runway_index: int, at: np.ndarray, velocity: np.ndarray, steps: int,
                       separation: Separation, finals: Sequence[Final], step_s: float) -> bool:
    """Field 4 (D175): whether the commanded aircraft, moved from ``at`` (its e, n and height at its first predicted
    step) along a straight line at ``velocity`` (m/s, the same three) with its runway ``runway_index`` in force, loses
    separation from the other aircraft ``other`` flying its record (as the loop flies it) at one of the window's next
    ``steps`` Δ rows: the one judge (`post.traffic`, VISUAL) at each, a loss the commanded aircraft answers for, as the
    loop's event. A row after the other aircraft's last is not judged (it no longer counts)."""
    interval = window.scene.interval_s
    record, own = window.scene.flight(other), window.commanded
    geometry = window.scene.geometry
    for k in range(1, steps + 1):
        time_s = window.first_step_s + k * interval
        if not record.start_s <= time_s <= record.end_s:
            continue
        here = at + velocity * (k * interval)
        aircraft = joined(AircraftAt.of([(own.key, tuple(here), tuple(here - velocity * step_s), True, runway_index,
                                          own.category, False, False)]),
                          AircraftAt.of([record.at_step(time_s, interval)]))
        if commanded_loss(traffic(aircraft, geometry, separation, finals, step_s), aircraft.last_step,
                          separation) is not None:
            return True
    return False


def loss_fields(window: Window, result: WindowResult, separation: Separation, finals: Sequence[Final],
                landings: LandingIndex, step_s: float, lost_every_draw: bool | None) -> dict[str, Any]:
    """The five fields of a window that lost separation (module docstring); field 5 only where it is known — the
    window's ceiling draws, which read the round with its traffic on, attached to the window's loss in either read."""
    interval = window.scene.interval_s
    every = int(round(interval / step_s))
    first = int(round((window.first_step_s - window.row0_s) / interval))
    runway = runway_at_loss(result)
    at = state_at(result, result.loss_step, every)
    time_s = window.step_s(result.loss_step)
    threshold = window.scene.geometry.candidates[runway]
    distance_km = math.hypot(at[0] - threshold.threshold_e_m, at[1] - threshold.threshold_n_m) / 1000.0
    after_s = time_s - window.first_step_s
    start = state_at(result, first, every)
    velocity = (start - np.asarray(result.states[first * every - 1, :3], dtype=np.float64)) / step_s
    steps = min(result.loss_step - first, branch_rows(interval, STRAIGHT_LINE_S))
    out = {"other_class": other_class(window, result.other, runway, at, time_s, separation, landings),
           "runway": window.scene.geometry.candidates[runway].ident,
           "distance_km": distance_km, "distance_bin": bin_of(distance_km, DISTANCE_EDGES_KM, DISTANCE_BINS),
           "after_first_step_s": after_s, "time_bin": bin_of(after_s, TIME_EDGES_S, TIME_BINS),
           "start_conflict": straight_line_loss(window, result.other, runway, start, velocity, steps, separation,
                                                finals, step_s)}
    if lost_every_draw is not None:
        out["lost_every_draw"] = lost_every_draw
    return out


# ---- what a read is
def window_line(read: str, place: int, window: Window, result: WindowResult) -> dict[str, Any]:
    """A window's line of ``per_window.jsonl`` before its loss's fields."""
    return {"read": read, "place": place, "flight": window.commanded.key, "airport": window.scene.geometry.code,
            "kind": result.kind, "outcome": result.outcome, "reward": result.reward, "go_arounds": result.go_arounds,
            "loss_step": result.loss_step, "other": result.other, "loss_reads_fault": result.loss_reads_fault}


def reading(codes: Sequence[str], outcomes: Sequence[str], rewards: Sequence[float]) -> dict[str, Any]:
    """Landed, lost separation and mean reward of the windows (their airports ``codes``), in all and by airport."""
    codes_ = np.asarray(codes)

    def part(mine: np.ndarray) -> dict[str, Any]:
        o = [outcomes[w] for w in np.flatnonzero(mine)]
        return {"windows": int(mine.sum()), "landed": sum(x == LANDED for x in o) / len(o),
                "lost_separation": sum(x == LOST_SEPARATION for x in o) / len(o),
                "reward_mean": float(np.mean([rewards[w] for w in np.flatnonzero(mine)])),
                "outcomes": dict(Counter(o))}

    return {"all": part(np.ones(len(codes_), dtype=bool)),
            "airports": {code: part(codes_ == code) for code in sorted(set(codes))}}


def paired(codes: Sequence[str], a: tuple[Sequence[str], Sequence[float]], b: tuple[Sequence[str], Sequence[float]],
           rng: np.random.Generator) -> dict[str, Any]:
    """The paired differences a − b of landed, lost separation and mean reward over the same windows (their airports
    ``codes``), each with its `INTERVAL` interval from `BOOTSTRAP_RESAMPLES` resamples of the windows, in all and by
    airport."""
    values = {"landed": [np.array([o == LANDED for o in x[0]], dtype=np.float64) for x in (a, b)],
              "lost_separation": [np.array([o == LOST_SEPARATION for o in x[0]], dtype=np.float64) for x in (a, b)],
              "reward_mean": [np.asarray(x[1], dtype=np.float64) for x in (a, b)]}
    codes_ = np.asarray(codes)
    tail = (1.0 - INTERVAL) / 2.0

    def part(mine: np.ndarray) -> dict[str, Any]:
        places = np.flatnonzero(mine)
        draws = rng.integers(0, len(places), size=(BOOTSTRAP_RESAMPLES, len(places)))
        out = {}
        for name, (x, y) in values.items():
            d = x[places] - y[places]
            resampled = d[draws].mean(axis=1)
            out[name] = {"difference": float(d.mean()),
                         "interval": [float(np.quantile(resampled, tail)), float(np.quantile(resampled, 1.0 - tail))]}
        return out

    return {"all": part(np.ones(len(codes_), dtype=bool)),
            "airports": {code: part(codes_ == code) for code in sorted(set(codes))}}


def failures(lines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Each field's counts and shares over the lines of windows lost to separation, in all and by airport."""
    fields = {"other_class": OTHER_CLASSES, "distance_bin": DISTANCE_BINS, "time_bin": TIME_BINS,
              "start_conflict": (True, False), "lost_every_draw": (True, False)}

    def part(mine: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"lost": len(mine)}
        for name, values in fields.items():
            known = [line["fields"][name] for line in mine if name in line["fields"]]
            if not known:
                continue
            counts = Counter(known)
            out[name] = {str(v).lower() if isinstance(v, bool) else v:
                         {"count": counts[v], "share": counts[v] / len(known)} for v in values}
        return out

    lost = [line for line in lines if line["outcome"] == LOST_SEPARATION]
    return {"all": part(lost),
            "airports": {code: part([line for line in lost if line["airport"] == code])
                         for code in sorted({line["airport"] for line in lines})}}


def ceiling_draw0(ceiling: Path, campaign: Path, windows: Sequence[Window], model: str
                  ) -> tuple[list[str], list[float], list[bool]] | None:
    """``model``'s draw 0 in the D161 readout ``ceiling`` of this campaign (each window's outcome and reward) and
    whether each window lost separation in every draw; None when it did not read that model. Refused by name unless it
    read this campaign's windows (by their flights and start times)."""
    config = json.loads((ceiling / "config.json").read_text(encoding="utf-8"))
    if config["schema"] != CEILING_SCHEMA:
        raise SystemExit(f"{ceiling} is a {config['schema']} readout, not {CEILING_SCHEMA}")
    if Path(config["campaign"]).name != campaign.name:
        raise SystemExit(f"{ceiling} read {Path(config['campaign']).name}, not {campaign.name}")
    if [(w["flight"], w["row0_s"]) for w in config["windows"]] != [(w.commanded.key, w.row0_s) for w in windows]:
        raise SystemExit(f"{ceiling} read other windows than this readout's (by their flights and start times)")
    if model not in config["models"]:
        return None
    held = json.loads((ceiling / f"model_{model}.json").read_text(encoding="utf-8"))
    return ([row[0] for row in held["outcomes"]], [row[0] for row in held["rewards"]],
            [all(o == LOST_SEPARATION for o in row) for row in held["outcomes"]])


def seal(out: Path) -> None:
    """``SHA256SUMS`` of every file under ``out``, then every file and directory read-only."""
    files = sorted(p for p in out.rglob("*") if p.is_file())
    (out / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}\n"
                                            for p in files), encoding="utf-8")
    for p in sorted(out.rglob("*"), reverse=True):
        os.chmod(p, 0o555 if p.is_dir() else 0o444)
    os.chmod(out, 0o555)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True, help="a post_train campaign's directory")
    parser.add_argument("--round", type=int, required=True, help="the round read (its checkpoint)")
    parser.add_argument("--intent", type=Path, required=True, help="the decision rules, a JSON object (intent.json)")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--ceiling", type=Path, help="a D161 readout of the campaign: its start, its draws of the round")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speak-workers", type=int, default=1, help="processes that read the batches (Speakers)")
    parser.add_argument("--smoke", action="store_true", help="SMOKE: a tree with changes too; no result")
    args = parser.parse_args(argv)
    campaign, out, intent_path = (p if p.is_absolute() else REPO_ROOT / p for p in (args.campaign, args.out, args.intent))
    ceiling = None if args.ceiling is None else (args.ceiling if args.ceiling.is_absolute() else REPO_ROOT / args.ceiling)
    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    if record["schema"] != CAMPAIGN_SCHEMA:
        parser.error(f"{campaign} is a {record['schema']} campaign, not {CAMPAIGN_SCHEMA}")
    if record["inputs"]["smoke"] and not args.smoke:
        parser.error(f"{campaign} is a smoke campaign: only a --smoke readout reads it")
    done, settings = done_rounds(campaign), settings_of(record)
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
    # with workers, the context is opened on the CPU and they are forked before this process uses the GPU
    context = open_context(Path(inputs["prior"]), instructions, executor, edges_reference,
                           torch.device("cpu") if args.speak_workers > 1 else device, Path(inputs["procedure_root"]),
                           formal=not args.smoke, data=False, splits=(SPLIT,))   # D132
    speakers = Speakers(context, settings, args.speak_workers, device) if args.speak_workers > 1 else None
    context = replace(context, device=device, base=context.base.to(device).eval())
    windows = selection_windows(context, settings, SPLIT)
    places = batches(windows, settings.batch_windows)
    start = round_draws = None
    if ceiling is not None:
        if settings.start is not None:
            raise SystemExit(f"{campaign} starts from a round of another campaign: its start's traffic output is not "
                             f"zero, so a ceiling's start is not the model read with traffic off at a start")
        start = ceiling_draw0(ceiling, campaign, windows, START)
        if start is None:
            raise SystemExit(f"{ceiling} did not read the campaign's start, which --ceiling is given for")
        round_draws = ceiling_draw0(ceiling, campaign, windows, str(args.round))
    codes = [w.scene.geometry.code for w in windows]
    out.mkdir(parents=True)
    write_json_atomic(out / "intent.json", intent)                                 # the rules, before the read
    write_json_atomic(out / "config.json", {
        "schema": DIAGNOSE_SCHEMA, "written_utc": utc_now(), "campaign": str(campaign), "round": args.round,
        "ceiling": None if ceiling is None else str(ceiling), "start_from_ceiling": start is not None,
        "lost_every_draw_from_ceiling": round_draws is not None, "settings": inputs["settings"],
        "instructions": str(instructions), "executor": str(executor), "checks": opened["checks"], "git": git,
        "smoke": args.smoke, "device": str(device), "speak_workers": args.speak_workers,
        "straight_line_s": STRAIGHT_LINE_S, "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "interval": INTERVAL,
                                                          "seed": [settings.select_seed, 1 << 29]},
        "windows": [window_record(w) for w in windows]})

    def log(event: Mapping[str, Any]) -> None:
        line = json.dumps({"utc": utc_now(), **event})
        with open(out / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    model = round_model(context, settings, campaign, args.round)
    separations = context.separations
    step_s = context.words.spec.step_s
    ends: dict[str, list[WindowResult]] = {}
    lines: dict[str, list[dict[str, Any]]] = {}
    try:
        for read in (ON, OFF):
            if read == OFF:
                traffic_off(model)
            got = (speakers.read(model, windows, places, SPLIT) if speakers is not None else
                   [read_batch(model, context, windows, p, settings, SPLIT) for p in places])
            if read == ON:
                held = json.loads((campaign / f"round_{args.round}" / "round.json").read_text(encoding="utf-8"))
                if json.loads(json.dumps(counted_ends(windows, places, got))) != held["selection_readout"]:
                    raise SystemExit(f"round {args.round}: the read with traffic on is not the round's selection "
                                     f"readout (round.json); the readout is not the campaign's — stopped")
                log({"read": ON, "checked": "the round's selection readout"})
            ends[read] = in_window_order(places, got, len(windows))
            lines[read] = []
            for place, (window, result) in enumerate(zip(windows, ends[read], strict=True)):
                line = window_line(read, place, window, result)
                if result.outcome == LOST_SEPARATION:
                    code = window.scene.geometry.code
                    line["fields"] = loss_fields(
                        window, result, separations[code], context.finals[code],
                        window_landings(window, context.rosters[code]), step_s,
                        None if round_draws is None else round_draws[2][place])
                lines[read].append(line)
            with open(out / "per_window.jsonl", "a", encoding="utf-8") as f:
                f.writelines(json.dumps(line) + "\n" for line in lines[read])
            log({"read": read, "landed": reading(codes, [e.outcome for e in ends[read]],
                                                 [e.reward for e in ends[read]])["all"]["landed"]})
    finally:
        if speakers is not None:
            speakers.close()
    reads = {read: ([e.outcome for e in ends[read]], [e.reward for e in ends[read]]) for read in (ON, OFF)}
    if start is not None:
        reads[START] = (start[0], start[1])
    rng = np.random.default_rng([settings.select_seed, 1 << 29])
    pairs = [(ON, OFF)] + ([(OFF, START), (ON, START)] if start is not None else [])
    write_json_atomic(out / "summary.json", {
        "schema": DIAGNOSE_SCHEMA, "written_utc": utc_now(), "round": args.round,
        "reads": {name: reading(codes, *values) for name, values in reads.items()},
        "differences": {f"{a}-{b}": paired(codes, reads[a], reads[b], rng) for a, b in pairs}})
    write_json_atomic(out / "failures.json", {"schema": DIAGNOSE_SCHEMA, **{read: failures(lines[read])
                                                                           for read in (ON, OFF)}})
    log({"written": "summary.json, failures.json"})
    seal(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
