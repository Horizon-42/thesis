"""Multi-aircraft M3's second pass (design §6.6 step 7 item 4): the start of the post-training — augmented with a traffic
attention at zero (`prior.model.with_traffic`: it answers as the prior does, to rounding) — commanding every aircraft of
a window at once (`traffic_window`), read beside the same model alone, the labelled words and the records, every one
judged in its window. A readout only.

``--windows-per-airport`` windows of the split are drawn (`traffic_window.draw_windows`: seeded, a window without a flight
that flies on its own dynamics passed over and counted), and every commanded aircraft is read four ways:

- **scene** — the model speaking to every commanded aircraft of the window, ``--samples`` times (a sample is the whole
  window: its aircraft's words said together, the separation masks on);
- **alone** — the same model, each aircraft seeing no other and under no separation mask, ``--samples`` times: flown
  and judged together in the window all the same;
- **labelled** — every commanded aircraft flying its labelled words from its first predicted step;
- **recorded** — every commanded aircraft along its own record on the loop's steps.

The model's sources are judged as they fly under VISUAL (`traffic_window.WindowLoop`: an aircraft ended there flies on
silent, still in the scene); the labelled and the recorded paths do not react to anything, so they are judged afterwards
the same way (`traffic_loop.Loop` keeping an aircraft it ends in the scene on its own path). IFR beside it: every
source's paths as they were flown judged again under IFR afterwards (a readout — the loop's reading is VISUAL). An
aircraft already in a loss it answers for at its first predicted step is counted and left out of the summaries, as in
M3's first pass.

Per source, pooled, per airport and per window size (the commanded aircraft of the window): the outcomes, lost
separation (and with whom: a commanded aircraft or a replayed one), the losses per aircraft and per hour flown to the
judged end, the reward M4 gives (landed in the airport's landing direction, not ended), the landing time against the
record and how often two commanded aircraft of a window land in the other order than recorded; for the model's sources
the separation masks, and — design §6.6 step 7 item 5 — how the rewards of one window's aircraft go together over its
samples (the pooled correlation of each aircraft's reward less its mean over the samples, over the pairs of a window).

Writes into a NEW directory ``aircraft.jsonl`` (a row per commanded aircraft, source and sample, appended as each batch
ends) and ``window_generation.json`` (the readout). A batch holds at most ``--aircraft-steps`` window samples × their
aircraft × their steps (the model's past, 6 KB an aircraft-step).

    python run_ts.py traffic_window_generation \\
        --prior 4dTrajectory/outputs/POOLED/prior/v3_stage2_clip_20260926/aug_s1337/round_07 \\
        --executor 4dTrajectory/outputs/POOLED/executor/<spec> \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --split select --out 4dTrajectory/outputs/POOLED/traffic/window_generation_<date>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.flights import flight_inputs
from ts_transformer.autopilot.judge import outcome_of
from ts_transformer.experiments.prior_free_generation import (
    _physics, fly_reference, glidepath_stops, limits_s, reference_grid,
)
from ts_transformer.experiments.prior_train import PRIOR_CHECKPOINT_SCHEMA, load_prior, rosters
from ts_transformer.experiments.traffic_loop import LOST_SEPARATION, Controlled, Loop, Run, recorded
from ts_transformer.experiments.traffic_labelled import own_end
from ts_transformer.experiments.traffic_speaking import (
    HISTORY_S, MASK_COLUMNS, scene_airports, scene_landings, speaking_aircraft,
)
from ts_transformer.experiments.traffic_window import (
    Window, WindowLoop, _with_landing, draw_windows, window_landings, window_of,
)
from ts_transformer.inference.scene_edges import EDGE_FEATURES
from ts_transformer.inference.separation import IFR, VISUAL
from ts_transformer.instructions.artefact import SPLITS
from ts_transformer.instructions.words import COLUMNS, Words
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.data import VARIANTS, airport_landings
from ts_transformer.prior.generate import rows_for
from ts_transformer.prior.landing_reward import landing_direction
from ts_transformer.prior.model import Prior, with_traffic
from ts_transformer.prior.scene import N_LOOK, Landings, hang, presence
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

SCHEMA = "ts-traffic-window-generation-v1"
SOURCES = ("scene", "alone", "labelled", "recorded")
#: A batch's most aircraft-steps (module docstring): about 1.8 GB of the model's past on its d and layers.
AIRCRAFT_STEPS = 300_000
#: Window sizes the readout is split by: the commanded aircraft of a window.
SIZES = ("1", "2", "3+")


@dataclasses.dataclass(frozen=True)
class Drawn:
    """The windows read: each window (its commanded aircraft over their time limits) and its commanded aircraft's
    places in ``batch`` (`traffic_window.WindowDraw.members`), with each one's time limit."""

    windows: list[Window]
    members: list[range]
    batch: replay.Batch
    limits: list[float]


def drawn_windows(draw: Any, airports: Mapping[str, Any], params: Any, step_s: float) -> Drawn:
    """`traffic_window.draw_windows`' windows over their commanded aircraft's time limits (the executor spec's)."""
    limits = limits_s(draw.batch, params, step_s, augmented=False)
    members = draw.members()
    windows = [window_of(airports[code], opens, commanded, [limits[j] for j in part], step_s)
               for (code, opens, commanded), part in zip(draw.openings, members)]
    return Drawn(windows, members, draw.batch, limits)


def window_size(window: Window, limits: Sequence[float], step_s: float) -> tuple[int, int, int, int]:
    """A window's aircraft, pre-roll (before its first commanded aircraft's row 0, as the speaker caps it), latest
    entry after it and rows (its longest time limit's), in steps."""
    first = window.first_step_s(window.commanded[0], step_s)
    pre = max([0] + [int(round((first - float(hang(window.rows(k).presence.start_s, step_s))) / step_s))
                     for k in window.others])
    late = max(int(round((window.first_step_s(k, step_s) - first) / step_s)) for k in window.commanded)
    return (len(window.commanded) + len(window.others), min(pre, int(HISTORY_S // step_s)), late,
            rows_for(max(limits) + step_s, step_s))


def batch_cost(sizes: Sequence[tuple[int, int, int, int]], samples: int) -> int:
    """The aircraft-steps a batch of windows (`window_size` each) holds in the speaker's past: every window sample on
    the most aircraft, over the batch's steps — its longest pre-roll, latest entry and rows together, as the loop lays
    them out (`WindowLoop`: one pre-roll for the batch)."""
    return (len(sizes) * samples * max(s[0] for s in sizes)
            * (max(s[1] for s in sizes) + max(s[2] for s in sizes) + max(s[3] for s in sizes)))


def window_batches(drawn: Drawn, samples: int, budget: int, step_s: float) -> list[list[int]]:
    """The windows in batches of the loop, by their size (`window_size`), each at most ``budget`` aircraft-steps
    (`batch_cost`; a larger window is a batch of its own)."""
    sizes = [window_size(w, [drawn.limits[j] for j in part], step_s) for w, part in zip(drawn.windows, drawn.members)]
    batches: list[list[int]] = [[]]
    for w in sorted(range(len(sizes)), key=lambda w: (sizes[w][1] + sizes[w][2] + sizes[w][3], sizes[w][0], w)):
        grown = batches[-1] + [w]
        if batches[-1] and batch_cost([sizes[k] for k in grown], samples) > budget:
            batches.append([w])
        else:
            batches[-1] = grown
    return batches


def batch_seed(seed: int, source: str, number: int) -> int:
    """A torch generator's seed for loop batch ``number`` of ``source``: each its own stream."""
    return int(np.random.SeedSequence([seed, SOURCES.index(source), number]).generate_state(1)[0])


def path_fields(run: Run, again: Run, key: str, path: Controlled, commanded: Sequence[str], own: str,
                until_s: float) -> dict[str, Any]:
    """How a commanded aircraft fared in its window: its outcome under VISUAL (the loop's ``run``) and under IFR
    (``again``, its paths as flown judged afterwards), the losses it was in to its judged end — the judge's, else its own
    end (``until_s``: a glidepath stop comes before its path's end) — by relation, how many it answered for, with a
    commanded aircraft or a replayed one — and the time flown to that end."""
    end = run.ended.get(key)
    last = end["t_s"] if end is not None else until_s
    mine = [e for e in run.episodes if key in e["pair"] and e["first_s"] <= last]
    others = [next(k for k in e["pair"] if k != key) for e in mine]
    return {"outcome": LOST_SEPARATION if end is not None else own, "own": own, "end": end,
            "starts_in_a_loss": end is not None and end["t_s"] <= path.first_step_s,
            "ifr_outcome": LOST_SEPARATION if key in again.ended else own,
            "episodes": len(mine), "relations": dict(Counter(e["relation"] for e in mine)),
            "answered": sum(any(r["key"] == key for r in e["responsible"]) for e in mine),
            "with_commanded": sum(k in commanded for k in others), "with_replayed": sum(k not in commanded for k in others),
            "ended_with": None if end is None else ("commanded" if end["with"] in commanded else "replayed"),
            "flown_s": float(last - path.first_step_s)}


def _judged_again(window: Window, paths: Sequence[Controlled], until: Mapping[str, float], reading: str,
                  step_s: float) -> Run:
    """A window's commanded aircraft along their paths judged afterwards (`Loop` keeping the ended ones on their paths,
    and each one past ``until`` — its own end, a glidepath stop — passive, as the window loop keeps them)."""
    return Loop(window.airport.flights.separation, reading, step_s, keep_ended=True).run(
        list(paths), [window.track(k) for k in window.others], until)


def _reward(row: dict[str, Any], direction: np.ndarray) -> float:
    """M4's reward: landed (the executor's judge) on a runway in the airport's landing direction, not ended."""
    return float(row["outcome"] == "landed" and bool(direction[row["runway"]]))


def model_rows(model: Prior, drawn: Drawn, chunk: Sequence[int], source: str, words: Words, params: Any,
               landings: Any, every_landing: Mapping[str, Landings], samples: int, *, generator: torch.Generator,
               temperature: float, procedure_masks: Any) -> list[dict[str, Any]]:
    """The windows at ``chunk`` spoken to ``samples`` times each (`WindowLoop`; ``source`` "alone": each aircraft
    hearing no other), a row per commanded aircraft and sample; ``landings``: the model's landing context (None for a
    variant without one), ``every_landing``: the airports' landings the reward's landing direction reads."""
    cpu, step_s = torch.device("cpu"), words.spec.step_s
    instances = [w for w in chunk for _ in range(samples)]
    index = [j for w in instances for j in drawn.members[w]]
    part = replay.subset(drawn.batch, index)
    runways, charts, approach = _physics(part, cpu)
    loop = WindowLoop(model, [drawn.windows[w] for w in instances], part.signals, part.geometries,
                      flight_inputs(part.series, device=cpu, anchor=N_LOOK), runways, charts, approach,
                      [drawn.limits[j] for j in index], words, params, landings, generator=generator,
                      temperature=temperature, procedure_masks=procedure_masks, alone=source == "alone")
    while loop.running:
        loop.step()
    results = loop.results()
    rows = []
    for b, w in enumerate(instances):
        window = drawn.windows[w]
        here = [i for i, r in enumerate(results) if r.window == b]
        until = {results[i].key: loop.judged_until_s(i) for i in here}
        again = _judged_again(window, [loop.path(i) for i in here], until, IFR, step_s)
        loop_landings = [(results[i].landing_s, part.geometries[i].candidates[results[i].runway].ident)
                         for i in here if results[i].landing_s is not None]
        for i in here:
            got = results[i]
            row = _aircraft_row(part, i, w, window, source, b % samples)
            row.update(path_fields(loop.runs[b], again, got.key, loop.path(i), window.commanded, got.own,
                                   until[got.key]),
                       said_steps=len(got.said), counted=got.counted, landing_s=got.landing_s, runway=got.runway,
                       mask_steps={COLUMNS[c]: int(loop.separation_masked[c][i, : got.counted].sum())
                                   for c in MASK_COLUMNS},
                       masked_mass={COLUMNS[c]: float(loop.speaker.forbidden[c][i, : got.counted].mean())
                                    if got.counted and c in loop.speaker.forbidden else 0.0 for c in MASK_COLUMNS})
            direction = landing_direction(part.signals[i], part.geometries[i],
                                          _loop_context(window, got.key, every_landing, loop_landings, step_s))
            row["reward"] = _reward(row, direction)
            rows.append(row)
    loop.close()
    return rows


def _loop_context(window: Window, key: str, landings: Mapping[str, Landings],
                  loop_landings: Sequence[tuple[float, str]], step_s: float) -> Landings:
    """``key``'s landings as its window had them in the loop — `window_landings` and the others' landings there — for
    the landing direction (its own recorded one still in, for `data.own_context`)."""
    out = window_landings(window, key, landings, step_s)
    for landed_s, runway in loop_landings:
        out = _with_landing(out, landed_s, runway)
    return out


def _aircraft_row(batch: replay.Batch, j: int, w: int, window: Window, source: str, sample: int | None
                  ) -> dict[str, Any]:
    signals = batch.signals[j]
    return {"dataset_id": signals.dataset_id, "airport": signals.airport, "window": w,
            "commanded": len(window.commanded), "source": source, "sample": sample,
            "observed_runway": batch.readings[j].runway_index}


def fixed_rows(drawn: Drawn, chunk: Sequence[int], source: str, words: Words, params: Any, landings: Any,
               procedure_masks: Any) -> list[dict[str, Any]]:
    """The windows at ``chunk`` with every commanded aircraft on a path that reacts to nothing — its labelled words
    flown from its first predicted step (``labelled``) or its own record on the loop's steps (``recorded``) — judged
    afterwards (`_judged_again`) under VISUAL and IFR, a row per commanded aircraft. A labelled path runs to its
    executor's end and is passive past a glidepath stop, as a model aircraft flies on past one (design §9 item 29). The
    landing direction reads the path's own world: the recorded landings for the record, the labelled paths' landings in
    place of the commanded aircraft's recorded ones for the labelled words."""
    step_s = words.spec.step_s
    index = [j for w in chunk for j in drawn.members[w]]
    part = replay.subset(drawn.batch, index)
    if source == "labelled":
        flown = fly_reference(part, words, params, device=torch.device("cpu"))
        grids = [reference_grid(r.words) for r in part.readings]
        stops = (glidepath_stops(flown, grids, part.geometries,
                                 [procedure_masks.finals[g.code] for g in part.geometries], words)
                 if procedure_masks.altitudes else None)
        step_rows = int(round(step_s / flown.cycle_s))
    rows, at = [], 0
    for w in chunk:
        window = drawn.windows[w]
        paths, owns, runways, until = [], [], [], {}
        for m, key in enumerate(window.commanded):
            j = at + m
            geometry = part.geometries[j]
            runways.append(part.readings[j].runway_index)
            if source == "labelled":
                ended = outcome_of(flown, j, geometry, part.readings[j].runway_index, words.spec)
                path = speaking_aircraft(window.scene(key, step_s), flown, j, grids[j], ended.outcome, ended.end_row,
                                         ended.crossing, -1, step_s)
                stop = -1 if stops is None else int(stops.step[j])
                own, last_row = own_end(ended.outcome, ended.end_row, stop, step_rows)
                until[key] = float(path.times_s[min(last_row // step_rows, len(path.times_s) - 1)])
            else:
                path = _recorded_path(part, j, window, key, step_s)
                own, until[key] = path.outcome, path.last_step_s
            paths.append(path)
            owns.append(own)
        run, again = (_judged_again(window, paths, until, reading, step_s) for reading in (VISUAL, IFR))
        for m, (key, path) in enumerate(zip(window.commanded, paths)):
            j = at + m
            row = _aircraft_row(part, j, w, window, source, None)
            row.update(path_fields(run, again, key, path, window.commanded, owns[m], until[key]),
                       landing_s=path.landing_s if owns[m] == "landed" else None, runway=runways[m])
            if source == "labelled":
                context = _loop_context(window, key, landings,
                                        [(other.landing_s, part.geometries[at + n].candidates[runways[n]].ident)
                                         for n, other in enumerate(paths)
                                         if n != m and owns[n] == "landed"], step_s)
            else:
                context = scene_landings(landings[window.airport.flights.code], window.scene(key, step_s))
            row["reward"] = _reward(row, landing_direction(part.signals[j], part.geometries[j], context))
            rows.append(row)
        at += len(window.commanded)
    return rows


def _recorded_path(batch: replay.Batch, j: int, window: Window, key: str, step_s: float) -> Controlled:
    """A commanded aircraft along its own record on the loop's steps from its first predicted step
    (`traffic_free_generation.recorded_rows`' path)."""
    flight, reading, geometry = batch.signals[j], batch.readings[j], batch.geometries[j]
    own = window.track(key)
    seen = presence(flight, len(reading.words), geometry)
    capture = int(np.searchsorted(seen.times_s, own.captured_s))
    whole = recorded(seen, flight, capture, replay.observed_landing_s(flight, reading, geometry), geometry,
                     window.airport.flights.separation, window.rows(key).category, step_s)
    cut = slice(N_LOOK, None)
    return dataclasses.replace(whole, times_s=whole.times_s[cut], e_m=whole.e_m[cut], n_m=whole.n_m[cut],
                               height_m=whole.height_m[cut], runway=whole.runway[cut], along_m=whole.along_m[cut],
                               track_minus_course_deg=whole.track_minus_course_deg[cut],
                               right_of_course_m=whole.right_of_course_m[cut],
                               along_speed_mps=whole.along_speed_mps[cut], established=whole.established[cut])


def summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per source, the aircraft not starting in a loss (those counted): outcomes, lost separation (VISUAL, IFR
    afterwards; ended with a commanded aircraft or a replayed one), losses per aircraft and per hour flown, their
    relations, the reward, the landing time against the record, the order of a window's landings against the record's,
    and — the model's sources — the separation masks and how one window's rewards go together."""
    recorded_landing = {(r["window"], r["dataset_id"]): r["landing_s"] for r in rows
                        if r["source"] == "recorded" and r["outcome"] == "landed"}
    out: dict[str, Any] = {}
    for source in SOURCES:
        every = [r for r in rows if r["source"] == source]
        part = [r for r in every if not r["starts_in_a_loss"]]
        if not part:
            continue
        hours = sum(r["flown_s"] for r in part) / 3600.0
        lost = [r for r in part if r["outcome"] == LOST_SEPARATION]
        deltas = [r["landing_s"] - recorded_landing[(r["window"], r["dataset_id"])] for r in part
                  if r["outcome"] == "landed" and recorded_landing.get((r["window"], r["dataset_id"])) is not None]
        entry: dict[str, Any] = {
            "aircraft": len(part), "left_out_starting_in_a_loss": len(every) - len(part),
            "outcomes": {k: v / len(part) for k, v in Counter(r["outcome"] for r in part).most_common()},
            "lost_separation": len(lost) / len(part),
            "lost_separation_with": {k: v / len(part) for k, v in Counter(r["ended_with"] for r in lost).items()},
            "lost_separation_ifr": sum(r["ifr_outcome"] == LOST_SEPARATION for r in part) / len(part),
            "episodes_per_aircraft": sum(r["episodes"] for r in part) / len(part),
            "episodes_per_hour": sum(r["episodes"] for r in part) / hours if hours else None,
            "episodes_with": {"commanded": sum(r["with_commanded"] for r in part),
                              "replayed": sum(r["with_replayed"] for r in part)},
            "relations": dict(sum((Counter(r["relations"]) for r in part), Counter())),
            "reward": sum(r["reward"] for r in part) / len(part),
            "landing_vs_recorded_s": ({"n": len(deltas), **{f"p{q}": float(np.percentile(deltas, q))
                                                            for q in (10, 50, 90)}} if deltas else {"n": 0}),
            "order": _order(every, recorded_landing)}
        if source in ("scene", "alone"):
            steps = sum(r["counted"] for r in part)
            entry["mask_steps_share"] = {name: sum(r["mask_steps"][name] for r in part) / steps if steps else None
                                         for name in (COLUMNS[c] for c in MASK_COLUMNS)}
            entry["masked_mass_per_step"] = {
                name: sum(r["masked_mass"][name] * r["counted"] for r in part) / steps if steps else None
                for name in (COLUMNS[c] for c in MASK_COLUMNS)}
            entry["reward_together"] = together(every)
        out[source] = entry
    return out


def _order(rows: Sequence[dict[str, Any]], recorded_landing: Mapping[tuple[int, str], float | None]) -> dict[str, int]:
    """Pairs of one window's commanded aircraft that both landed (not ended by the judge), in a sample and in the
    record: how many, and how many landed the other way round."""
    pairs = swapped = 0
    by_sample: dict[tuple[int, Any], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_sample[(r["window"], r["sample"])].append(r)
    for (w, _), members in by_sample.items():
        landed = [(r["landing_s"], recorded_landing.get((w, r["dataset_id"]))) for r in members
                  if r["outcome"] == "landed" and recorded_landing.get((w, r["dataset_id"])) is not None]
        for a in range(len(landed)):
            for b in range(a + 1, len(landed)):
                pairs += 1
                swapped += (landed[a][0] - landed[b][0]) * (landed[a][1] - landed[b][1]) < 0
    return {"pairs": pairs, "swapped": swapped}


def together(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How one window's rewards go together over its samples (design §6.6 step 7 item 5): each aircraft's reward less
    its mean over the samples, the pooled correlation over every pair of aircraft of a window and sample — the
    advantages M4 trains on; windows of one aircraft have no pair."""
    by_window: dict[int, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by_window[r["window"]][r["dataset_id"]][r["sample"]] = r["reward"]
    products = squares_a = squares_b = 0.0
    pairs = 0
    for aircraft in by_window.values():
        centred = {}
        for key, samples in aircraft.items():
            values = np.array([samples[s] for s in sorted(samples)])
            centred[key] = values - values.mean()
        keys = sorted(centred)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                x, y = centred[keys[a]], centred[keys[b]]
                if len(x) != len(y):
                    raise ValueError("a window's aircraft are read over the same samples")
                pairs += 1
                products += float(x @ y)
                squares_a += float(x @ x)
                squares_b += float(y @ y)
    denominator = (squares_a * squares_b) ** 0.5
    return {"pairs": pairs, "correlation": products / denominator if denominator else None}


def summaries(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """`summary` pooled, per airport and per window size."""
    by_airport: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_size: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_airport[row["airport"]].append(row)
        by_size[SIZES[min(row["commanded"], 3) - 1]].append(row)
    return {"pooled": summary(rows), "airports": {code: summary(part) for code, part in sorted(by_airport.items())},
            "window_sizes": {size: summary(by_size[size]) for size in SIZES if by_size[size]}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the single-aircraft prior it starts from (augmented)")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--windows-per-airport", type=int, default=200)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--aircraft-steps", type=int, default=AIRCRAFT_STEPS, help="a batch's most (module docstring)")
    parser.add_argument("--device", default="cuda", help="the prior's; the executors fly on CPU")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)
    if args.windows_per_airport < 1 or args.samples < 1:
        parser.error("at least one window an airport and one sample")

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    prior_dir, instructions, executor_dir, out = map(resolved, (args.prior, args.instructions, args.executor, args.out))
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    git = git_state()
    started = time.perf_counter()
    params, record, words = replay.open_executor(executor_dir, instructions)
    single, payload, prior_config, own_masks = load_prior(prior_dir, instructions)
    if prior_config["smoke"] or payload["schema"] != PRIOR_CHECKPOINT_SCHEMA:
        parser.error(f"{prior_dir} is not a single-aircraft prior's formal run")
    torch.manual_seed(args.seed)                       # the traffic attention's weights: at zero it reads nothing
    model = with_traffic(single, EDGE_FEATURES).to(torch.device(args.device)).eval()
    every_landing = airport_landings(instructions, rosters(instructions))
    landings = every_landing if VARIANTS[model.config.variant].landing_context else None
    step_s = words.spec.step_s
    airports, built = scene_airports(instructions, args.split, words.spec, model.config.airports, landings,
                                     model.config.max_rows)
    draw = draw_windows(instructions, args.split, words.spec, words, airports, per_airport=args.windows_per_airport,
                        seed=args.seed, step_s=step_s)
    drawn = drawn_windows(draw, airports, params, step_s)
    print(f"{len(drawn.windows)} {args.split} windows, {len(drawn.batch.readings)} commanded aircraft "
          f"({draw.counts}), scenes built ({built}), {time.perf_counter() - started:.0f}s", flush=True)

    batches = window_batches(drawn, args.samples, args.aircraft_steps, step_s)
    out.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    done = 0
    for number, chunk in enumerate(batches):
        new: list[dict[str, Any]] = []
        for source in ("scene", "alone"):
            generator = torch.Generator(device=torch.device(args.device)).manual_seed(
                batch_seed(args.seed, source, number))
            new += model_rows(model, drawn, chunk, source, words, params, landings, every_landing, args.samples,
                              generator=generator, temperature=args.temperature, procedure_masks=own_masks)
        for source in ("labelled", "recorded"):
            new += fixed_rows(drawn, chunk, source, words, params, every_landing, own_masks)
        with (out / "aircraft.jsonl").open("a", encoding="utf-8") as stream:
            for row in new:
                stream.write(json.dumps(row) + "\n")
        rows += new
        done += len(chunk)
        held = (f", GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak" if torch.device(args.device).type == "cuda"
                else "")
        print(f"  {done}/{len(drawn.windows)} windows ({len(batches)} batches), {time.perf_counter() - started:.0f}s"
              f"{held}", flush=True)

    readout = summaries(rows)
    write_json_atomic(out / "window_generation.json", {
        "schema": SCHEMA, "written_utc": utc_now(), "git": git, "split": args.split, "drawn": draw.counts,
        "windows_per_airport": args.windows_per_airport, "samples": args.samples, "temperature": args.temperature,
        "seed": args.seed, "prior": {"directory": repo_relative(prior_dir), "procedure_masks": list(own_masks.names),
                                     "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt")},
        "traffic_attention": "zero (with_traffic): the prior's answers to rounding",
        "executor": {"directory": repo_relative(executor_dir), "sha256": record["sha256"]},
        "instructions": repo_relative(instructions), "scenes": built, "history_s": HISTORY_S,
        "readings": {"ends": VISUAL, "beside": IFR}, "aircraft_steps": args.aircraft_steps, "batches": len(batches),
        "readout": readout, "aircraft_file": "aircraft.jsonl", "elapsed_s": time.perf_counter() - started})
    for source, entry in readout["pooled"].items():
        shares = "  ".join(f"{name} {share:.3f}" for name, share in entry["outcomes"].items())
        print(f"{source:9s} n={entry['aircraft']:5d}  reward {entry['reward']:.3f}  lost separation "
              f"{entry['lost_separation']:.3f} (IFR {entry['lost_separation_ifr']:.3f}, with {entry['lost_separation_with']})"
              f"  order {entry['order']}  {shares}")
        if "reward_together" in entry:
            print(f"          masks: steps {entry['mask_steps_share']}  rewards together {entry['reward_together']}")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
