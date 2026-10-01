"""Can a loss of separation be undone by rewinding one aircraft? (multi-aircraft design §6.6 step 7.7.) A readout only:
nothing is trained.

**The original pass**: every drawn window (`traffic_window.draw_windows`, ``--windows-per-airport`` of the split) spoken
to once by the model, every commanded aircraft together (R34's "scene" reading at ``--samples 1``, from its streams:
`traffic_window_generation.batch_seed`), its words and the judge's books kept.

**Events**: each loss of separation there that ended a commanded aircraft (a wake shortfall at a landing included), not
one it started in; a pair ended together at one step is one event. Its time ``t_L`` is the step the judge ended it at.

**Who speaks again** (a role each): the aircraft the judge ended (``answered``) and the other of the pair when it is
commanded (``partner``); when both were ended for each other at that step, each is done once (``both``).

**From where** (an offset each): ``t_L`` less each of ``--offsets-s`` (in steps), and ``start`` — the aircraft's first
predicted step (design: the one-aircraft credit, candidate B). An offset before its first predicted step, or at an own
step it said nothing at (ended or stopped before it), is skipped and counted.

**A branch**: the window flown again from its start with every word given (`traffic_window.Given`) — the speaking
aircraft's to its own step, from where the prior speaks for it; every other commanded aircraft's in full, past their end
the prior speaking for one still flying (design item 7) — ``--branches`` times an offset, each from a stream of its own
(`rewind_seed`). Given words are neither sampled nor masked. **The control**: one more, every word given; it must replay
the original pass to the last field, or the run stops (design item 9).

**Rescued** (design item 10): (i) the pair loses no separation from the branch's step on; (ii) the speaking aircraft
lands in the airport's landing direction, not ended (M4's reward 1); (iii) no aircraft is ended that the original pass
did not end. Each is kept beside.

**The readout**, per role and offset, pooled and per airport, stratum of the speaking aircraft (`instructions.readout.
stratum`), kind and relation of the loss, window size and — augmented — kind: the cells (an event's role and offset),
the share with at least one branch rescued, the mean share of branches rescued, how many of a cell's branches were
(`TALLIES`), the three conditions' shares, and the words the speaking aircraft changed between its step and ``t_L``
(steps per column differing from the original), rescued and not.

Writes into a NEW directory ``original.jsonl`` (the original pass's rows, R34's), ``events.jsonl`` (an event a row with
its branches) and ``window_rewind.json`` (the header and the readout). Both passes run in ``--workers`` forked processes
(`traffic_window_generation.in_processes`); what is read does not depend on their number.

    python run_ts.py traffic_window_rewind \\
        --prior 4dTrajectory/outputs/POOLED/prior/m4_passes_20260929/traffic_s1337/round_05 \\
        --executor 4dTrajectory/outputs/POOLED/executor/v11_20260927 \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --split select --out 4dTrajectory/outputs/POOLED/traffic/window_rewind_<date>
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.experiments.prior_free_generation import start_altitude_windows
from ts_transformer.experiments.prior_train import rosters
from ts_transformer.experiments.traffic_speaking import HISTORY_S, scene_airports
from ts_transformer.experiments.traffic_window import Given, draw_windows
from ts_transformer.experiments.traffic_window_augment import busiest
from ts_transformer.experiments.traffic_window_generation import (
    AIRCRAFT_STEPS, WINDOWS_PER_AIRPORT, WORKERS, Drawn, augmented_windows, batch_seed, drawn_windows, flown_sentences,
    fly_windows, in_processes, packed, size_of, summaries, window_batches, window_prior, window_size,
)
from ts_transformer.inference.separation import VISUAL
from ts_transformer.instructions.artefact import SPLITS
from ts_transformer.instructions.readout import stratum
from ts_transformer.instructions.words import COLUMNS, UNCHANGED, Words
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.data import VARIANTS, airport_landings
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

SCHEMA = "ts-traffic-window-rewind-v1"
#: Design §6.6 step 7.7 item 6: how long before the loss an aircraft speaks again, beside its first predicted step.
OFFSETS_S = (10.0, 30.0, 60.0, 120.0)
START = "start"
#: Item 8: the branches an offset.
BRANCHES = 8
ANSWERED, PARTNER, BOTH = "answered", "partner", "both"
ROLES = (ANSWERED, PARTNER, BOTH)
BEFORE_FIRST_STEP, SILENT_THEN = "before_its_first_step", "silent_then"
#: How many of a cell's branches were rescued, binned.
TALLIES = ("0", "1-2", "3-5", "6+")
#: The branches' streams' key beside the seed (R34's sources use 0–3).
REWIND_STREAM = 100
#: The original pass's row fields a control replays (the masks' counts differ: given words are not masked).
REPLAYED_FIELDS = ("outcome", "own", "end", "counted", "landing_s", "runway", "reward", "said_steps", "episodes",
                   "relations", "answered", "flown_s", "ifr_outcome")


def rewind_seed(seed: int, number: int) -> int:
    """A torch generator's seed for branch batch ``number``: each its own stream."""
    return int(np.random.SeedSequence([seed, REWIND_STREAM, number]).generate_state(1)[0])


@dataclasses.dataclass(frozen=True)
class Original:
    """One window as the original pass flew it: its commanded aircraft (``keys``, the window's order), what each said
    (`Commanded.said`), its first state's time and its row (R34's), and the judge's books."""

    window: int
    keys: list[str]
    said: list[np.ndarray]
    first_s: list[float]
    rows: list[dict[str, Any]]
    ended: dict[str, dict[str, Any]]
    episodes: list[dict[str, Any]]
    at_threshold: list[dict[str, Any]]


@dataclasses.dataclass(frozen=True)
class Event:
    """A loss of separation of the original pass that ended a commanded aircraft (module docstring)."""

    number: int
    window: int
    t_s: float
    pair: tuple[str, str]
    kind: str
    relation: str
    speakers: tuple[tuple[str, str], ...]   # (key, role)


@dataclasses.dataclass(frozen=True)
class Branch:
    """One flight of an event's window: ``speaker`` speaks again from its own ``step`` (``offset``: an `OFFSETS_S`
    value in seconds as text, or `START`), ``copy`` of `BRANCHES`; the control (``speaker`` None) gives every word."""

    event: int
    speaker: str | None
    role: str | None
    offset: str
    step: int
    copy: int


def flown_originals(flown: Any, sentences: Any, chunk: Sequence[int]) -> list[Original]:
    """The windows of a loop that has run (`fly_windows` at one sample, `flown_sentences`) as `Original` each."""
    out = []
    for b, w in enumerate(chunk):
        here = flown.members(b)
        run = flown.loop.runs[b]
        out.append(Original(w, [flown.results[i].key for i in here], [flown.results[i].said.copy() for i in here],
                            [flown.results[i].first_s for i in here], [sentences.rows[i] for i in here],
                            dict(run.ended), list(run.episodes), list(run.at_threshold)))
    return out


def events_of(original: Original, first: int) -> list[Event]:
    """The original window's events (module docstring), numbered from ``first``."""
    out: list[Event] = []
    seen = set()
    for key, row in zip(original.keys, original.rows):
        end = original.ended.get(key)
        if end is None or row["starts_in_a_loss"]:
            continue
        other = end["with"]
        pair = tuple(sorted((key, other)))
        if (pair, end["t_s"]) in seen:
            continue
        seen.add((pair, end["t_s"]))
        theirs = original.ended.get(other)
        if other in original.keys and theirs is not None and theirs["with"] == key and theirs["t_s"] == end["t_s"]:
            speakers = ((key, BOTH), (other, BOTH))
        elif other in original.keys:
            speakers = ((key, ANSWERED), (other, PARTNER))
        else:
            speakers = ((key, ANSWERED),)
        out.append(Event(first + len(out), original.window, float(end["t_s"]), pair, end["kind"], end["relation"],
                         speakers))
    return out


def branches_of(event: Event, original: Original, offsets_s: Sequence[float], branches: int, step_s: float
                ) -> tuple[list[Branch], Counter]:
    """The event's branches (module docstring), its control last, and the offsets skipped by ``(role, offset,
    why)``."""
    out: list[Branch] = []
    skipped: Counter = Counter()
    for key, role in event.speakers:
        i = original.keys.index(key)
        for offset in (*offsets_s, None):
            name = START if offset is None else f"{offset:g}"
            step = 0 if offset is None else int(round((event.t_s - offset - original.first_s[i]) / step_s))
            if step < 0:
                skipped[(role, name, BEFORE_FIRST_STEP)] += 1
            elif step >= len(original.said[i]):
                skipped[(role, name, SILENT_THEN)] += 1
            else:
                out += [Branch(event.number, key, role, name, step, copy) for copy in range(branches)]
    out.append(Branch(event.number, None, None, "control", -1, 0))
    return out, skipped


def given_of(branch: Branch, original: Original) -> list[Given]:
    """Every commanded aircraft's words in a branch, in the window's order (module docstring)."""
    return [Given(said, branch.step if key == branch.speaker else len(said))
            for key, said in zip(original.keys, original.said)]


def _pair_again(run: Any, pair: tuple[str, str], from_s: float) -> bool:
    """Whether ``pair`` lost separation at or after ``from_s`` in a judged window (an episode, or a wake shortfall at a
    landing)."""
    return (any(tuple(sorted(e["pair"])) == pair and e["last_s"] >= from_s for e in run.episodes)
            or any(tuple(sorted((a["leader"], a["follower"]))) == pair and a["t_s"] >= from_s
                   for a in run.at_threshold))


def _changed(original: np.ndarray, again: np.ndarray, first: int, last: int) -> dict[str, int]:
    """Steps ``first … last`` (own steps, inclusive) on which a column's word differs (past a sentence's end: unchanged)."""
    steps = last + 1 - first
    def grid(said: np.ndarray) -> np.ndarray:
        out = np.full((max(steps, 0), 6), UNCHANGED, dtype=np.int64)
        part = said[first: last + 1]
        out[: len(part)] = part
        return out
    differs = grid(original) != grid(again)
    return {COLUMNS[c]: int(differs[:, c].sum()) for c in range(6)}


def branch_result(branch: Branch, event: Event, original: Original, keys: Sequence[str], said: Sequence[np.ndarray],
                  rows: Sequence[dict[str, Any]], run: Any, step_s: float) -> dict[str, Any]:
    """A branch read against its event (module docstring: the three conditions, rescued, the words changed)."""
    if list(keys) != original.keys:
        raise ValueError(f"event {event.number}: the branch flew other aircraft than its window")
    i = original.keys.index(branch.speaker)
    from_s = original.first_s[i] + branch.step * step_s
    cleared = not _pair_again(run, event.pair, from_s)
    landed = rows[i]["reward"] == 1.0
    new = sorted(set(run.ended) - set(original.ended))
    loss_step = int(round((event.t_s - original.first_s[i]) / step_s))
    return {"speaker": branch.speaker, "role": branch.role, "offset": branch.offset, "step": branch.step,
            "copy": branch.copy, "pair_cleared": cleared, "landed": landed, "new_losses": new,
            "rescued": cleared and landed and not new, "outcome": rows[i]["outcome"],
            "changed": _changed(original.said[i], said[i], branch.step, loss_step)}


def check_control(event: Event, original: Original, keys: Sequence[str], said: Sequence[np.ndarray],
                  rows: Sequence[dict[str, Any]], run: Any) -> None:
    """The control replays the original pass to the last field (design item 9), or the run stops."""
    def fail(what: str) -> None:
        raise ValueError(f"event {event.number} (window {event.window}): the control did not replay the original "
                         f"pass — {what}")
    if list(keys) != original.keys:
        fail("other aircraft")
    for key, a, b, x, y in zip(keys, original.said, said, original.rows, rows):
        if a.shape != b.shape or not (a == b).all():
            fail(f"{key}'s words")
        for field in REPLAYED_FIELDS:
            if x[field] != y[field]:
                fail(f"{key}'s {field}: {x[field]!r} then, {y[field]!r} now")
    if dict(run.ended) != original.ended or list(run.episodes) != original.episodes \
            or list(run.at_threshold) != original.at_threshold:
        fail("the judge's books")


def tally(rescued: int) -> str:
    """`TALLIES`' bin of a cell's rescued branches."""
    return TALLIES[0] if rescued == 0 else TALLIES[1] if rescued <= 2 else TALLIES[2] if rescued <= 5 else TALLIES[3]


def cell_summary(cells: Sequence[list[dict[str, Any]]]) -> dict[str, Any]:
    """Cells (an event's role and offset: its branches) summed up (module docstring)."""
    rescued = [sum(b["rescued"] for b in cell) for cell in cells]
    branches = [b for cell in cells for b in cell]
    def changed(part: list[dict[str, Any]]) -> dict[str, float] | None:
        return {c: float(np.mean([b["changed"][c] for b in part])) for c in COLUMNS} if part else None
    return {"cells": len(cells), "branches": len(branches),
            "rescued_any": float(np.mean([r > 0 for r in rescued])),
            "rescued_mean": float(np.mean([r / len(cell) for r, cell in zip(rescued, cells)])),
            "tallies": {name: sum(tally(r) == name for r in rescued) for name in TALLIES},
            "pair_cleared": float(np.mean([b["pair_cleared"] for b in branches])),
            "landed": float(np.mean([b["landed"] for b in branches])),
            "no_new_loss": float(np.mean([not b["new_losses"] for b in branches])),
            "changed_rescued": changed([b for b in branches if b["rescued"]]),
            "changed_not_rescued": changed([b for b in branches if not b["rescued"]])}


#: The groups the readout splits by: each a cell's value (`cells_of`).
GROUPS = ("airport", "stratum", "kind", "relation", "window_size", "augmented_kind")


def cells_of(records: Sequence[dict[str, Any]], offsets: Sequence[str]) -> list[dict[str, Any]]:
    """Every cell of ``records`` (an event's speaker and offset: its branches), with its role, offset and its value in
    each of `GROUPS` — the stratum the speaking aircraft's."""
    out = []
    for record in records:
        for key, role in record["speakers"]:
            for offset in offsets:
                cell = [b for b in record["branches"] if b["speaker"] == key and b["offset"] == offset]
                if cell:
                    out.append({"role": role, "offset": offset, "branches": cell,
                                "airport": record["airport"], "stratum": record["strata"][key],
                                "kind": record["kind"], "relation": record["relation"],
                                "window_size": size_of(record["commanded"]),
                                "augmented_kind": None if record["augmented"] is None else record["augmented"]["kind"]})
    return out


def readout(records: Sequence[dict[str, Any]], skipped: Counter, offsets: Sequence[str]) -> dict[str, Any]:
    """Per role and offset (module docstring), pooled and per group (`GROUPS`); ``records``: an event a record, its
    branches beside (`events.jsonl`)."""
    cells = cells_of(records, offsets)

    def table(part: Sequence[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for role in ROLES:
            for offset in offsets:
                here = [c["branches"] for c in part if c["role"] == role and c["offset"] == offset]
                if here:
                    out.setdefault(role, {})[offset] = cell_summary(here)
        return out

    groups: dict[str, dict[str, Any]] = {}
    for name in GROUPS:
        values = sorted({c[name] for c in cells if c[name] is not None})
        if values:
            groups[name] = {value: table([c for c in cells if c[name] == value]) for value in values}
    return {"events": len(records),
            "skipped": [{"role": role, "offset": offset, "why": why, "count": count}
                        for (role, offset, why), count in sorted(skipped.items())],
            "pooled": table(cells), "groups": groups}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the traffic prior the windows are spoken by")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--windows-per-airport", type=int, default=WINDOWS_PER_AIRPORT)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--augment-seed", type=int, default=None, help="every window augmented with this seed "
                        "(`traffic_window_augment`)")
    parser.add_argument("--offsets-s", type=float, nargs="+", default=list(OFFSETS_S),
                        help="how long before the loss an aircraft speaks again (beside its first predicted step)")
    parser.add_argument("--branches", type=int, default=BRANCHES, help="the branches an offset")
    parser.add_argument("--aircraft-steps", type=int, default=AIRCRAFT_STEPS, help="a loop batch's most")
    parser.add_argument("--workers", type=int, default=WORKERS, help="reading processes (what is read does not depend "
                        "on it)")
    parser.add_argument("--device", default="cuda", help="the prior's; the executors fly on CPU")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)
    if args.windows_per_airport < 1 or args.branches < 1 or args.workers < 1:
        parser.error("at least one window an airport, one branch and one reading process")
    if any(o <= 0 for o in args.offsets_s) or len(set(args.offsets_s)) != len(args.offsets_s):
        parser.error("the offsets are distinct and positive")

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    prior_dir, instructions, executor_dir, out = map(resolved, (args.prior, args.instructions, args.executor, args.out))
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    git = git_state()
    started = time.perf_counter()
    params, spec_record, words = replay.open_executor(executor_dir, instructions)
    try:                                               # on the CPU until the reading processes are forked
        model, attention, _, own_masks = window_prior(prior_dir, instructions, args.seed)
    except ValueError as refusal:
        parser.error(str(refusal))
    every_landing = airport_landings(instructions, rosters(instructions))
    landings = every_landing if VARIANTS[model.config.variant].landing_context else None
    step_s = words.spec.step_s
    airports, built = scene_airports(instructions, args.split, words.spec, model.config.airports, landings,
                                     model.config.max_rows)
    draw = draw_windows(instructions, args.split, words.spec, words, airports, per_airport=args.windows_per_airport,
                        seed=args.seed, step_s=step_s)
    drawn = drawn_windows(draw, airports, params, step_s)
    augmenting = None
    if args.augment_seed is not None:
        most = busiest(instructions, words.spec, model.config.airports, step_s)
        drawn, augmenting = augmented_windows(drawn, params, most, model.config.max_rows, args.augment_seed,
                                              start_altitude_windows(instructions), words.spec)
        if not drawn.windows:
            parser.error("every window was left out")
    print(f"{len(drawn.windows)} {args.split} windows, {len(drawn.batch.readings)} commanded aircraft "
          f"({draw.counts}), scenes built ({built}), {time.perf_counter() - started:.0f}s", flush=True)
    out.mkdir(parents=True)
    device = torch.device(args.device)
    common = dict(temperature=args.temperature, procedure_masks=own_masks)

    # -- the original pass: R34's scene reading at one sample
    batches = window_batches(drawn, 1, args.aircraft_steps, step_s)

    def read_original(number: int) -> list[Original]:
        generator = torch.Generator(device=device).manual_seed(batch_seed(args.seed, "scene", number))
        flown = fly_windows(model.to(device), drawn, batches[number], "scene", words, params, landings, 1,
                            generator=generator, **common)
        sentences = flown_sentences(flown, drawn, "scene", words, every_landing, 1)
        got = flown_originals(flown, sentences, batches[number])
        for o in got:
            for row in o.rows:
                row["batch"] = number
        flown.loop.close()
        return got

    gc.collect()
    gc.freeze()
    originals: dict[int, Original] = {}
    peaks = [0.0]
    for number, got, peak in in_processes(args.workers, list(range(len(batches))), read_original):
        originals.update((o.window, o) for o in got)
        peaks.append(peak)
    rows = [row for w in sorted(originals) for row in originals[w].rows]
    with (out / "original.jsonl").open("w", encoding="utf-8") as stream:
        for row in sorted(rows, key=lambda row: row["batch"]):
            stream.write(json.dumps(row) + "\n")
    print(f"original pass: {len(batches)} batches, {time.perf_counter() - started:.0f}s", flush=True)

    # -- events and their branches
    events: list[Event] = []
    for w in sorted(originals):
        events += events_of(originals[w], len(events))
    branches: list[Branch] = []
    skipped: Counter = Counter()
    for event in events:
        got, why = branches_of(event, originals[event.window], args.offsets_s, args.branches, step_s)
        branches += got
        skipped.update(why)
    sizes = {w: window_size(drawn.windows[w], [drawn.limits[j] for j in drawn.members[w]], step_s)
             for w in {e.window for e in events}}
    branch_batches = packed([sizes[events[b.event].window] for b in branches], 1, args.aircraft_steps)
    print(f"{len(events)} events, {len(branches)} window flights in {len(branch_batches)} batches "
          f"(skipped {sum(skipped.values())}), {time.perf_counter() - started:.0f}s", flush=True)

    def read_branches(number: int) -> list[tuple[int, dict[str, Any] | None]]:
        chunk = branch_batches[number]
        windows = [events[branches[b].event].window for b in chunk]
        given = [g for b in chunk for g in given_of(branches[b], originals[events[branches[b].event].window])]
        generator = torch.Generator(device=device).manual_seed(rewind_seed(args.seed, number))
        flown = fly_windows(model.to(device), drawn, windows, "scene", words, params, landings, 1,
                            generator=generator, given=given, **common)
        sentences = flown_sentences(flown, drawn, "scene", words, every_landing, 1)
        got = []
        for k, (again, b) in enumerate(zip(flown_originals(flown, sentences, windows), chunk)):
            branch, event = branches[b], events[branches[b].event]
            original, run = originals[event.window], flown.loop.runs[k]
            if branch.speaker is None:
                check_control(event, original, again.keys, again.said, again.rows, run)
                got.append((b, None))
            else:
                got.append((b, branch_result(branch, event, original, again.keys, again.said, again.rows, run,
                                             step_s)))
        flown.loop.close()
        return got

    results: dict[int, dict[str, Any]] = {}
    done = 0
    for number, got, peak in in_processes(args.workers, list(range(len(branch_batches))), read_branches):
        results.update((b, r) for b, r in got if r is not None)
        done += len(branch_batches[number])
        peaks.append(peak)
        print(f"  branch batch {number}: {done}/{len(branches)} flights ({len(branch_batches)} batches), "
              f"{time.perf_counter() - started:.0f}s, GPU {max(peaks):.2f} GB a process at most", flush=True)

    # -- the readout
    by_event: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for b in sorted(results):
        by_event[branches[b].event].append(results[b])
    records = []
    for event in events:
        original = originals[event.window]
        members = drawn.members[event.window]
        strata = {key: stratum(drawn.batch.readings[members[i]]) for i, key in enumerate(original.keys)}
        records.append({"event": event.number, "window": event.window,
                        "airport": original.rows[0]["airport"], "commanded": len(original.keys),
                        "augmented": drawn.augmented[event.window], "t_s": event.t_s, "pair": list(event.pair),
                        "kind": event.kind, "relation": event.relation,
                        "speakers": [list(s) for s in event.speakers], "strata": strata,
                        "branches": by_event[event.number]})
    with (out / "events.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    offsets = [*(f"{o:g}" for o in args.offsets_s), START]
    rewound = readout(records, skipped, offsets)
    write_json_atomic(out / "window_rewind.json", {
        "schema": SCHEMA, "written_utc": utc_now(), "git": git, "split": args.split, "drawn": draw.counts,
        "windows_per_airport": args.windows_per_airport, "temperature": args.temperature, "seed": args.seed,
        "augment_seed": args.augment_seed, "augmenting": augmenting,
        "prior": {"directory": repo_relative(prior_dir), "procedure_masks": list(own_masks.names),
                  "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt")},
        "traffic_attention": attention,
        "executor": {"directory": repo_relative(executor_dir), "sha256": spec_record["sha256"]},
        "instructions": repo_relative(instructions), "scenes": built, "history_s": HISTORY_S,
        "reading": VISUAL, "offsets_s": list(args.offsets_s), "branches": args.branches,
        "aircraft_steps": args.aircraft_steps, "batches": {"original": len(batches), "branches": len(branch_batches)},
        "workers": args.workers, "gpu_peak_gb_a_process": max(peaks),
        "original": summaries(rows, args.augment_seed is not None)["pooled"],
        "readout": rewound, "files": {"original": "original.jsonl", "events": "events.jsonl"},
        "elapsed_s": time.perf_counter() - started})
    for role, by_offset in rewound["pooled"].items():
        for offset, cell in by_offset.items():
            print(f"{role:8s} {offset:>5s}: cells {cell['cells']:4d}  rescued any {cell['rescued_any']:.3f}  "
                  f"mean {cell['rescued_mean']:.3f}  pair cleared {cell['pair_cleared']:.3f}  landed "
                  f"{cell['landed']:.3f}  no new loss {cell['no_new_loss']:.3f}  {cell['tallies']}")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
