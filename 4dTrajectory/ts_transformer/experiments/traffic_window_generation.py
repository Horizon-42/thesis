"""Multi-aircraft M3's second pass (design §6.6 step 7 item 4): the start of the post-training — augmented with a traffic
attention at zero (`prior.model.with_traffic`: it answers as the prior does, to rounding) — commanding every aircraft of
a window at once (`traffic_window`), read beside the same model alone, the labelled words and the records, every one
judged in its window. A readout only. ``--prior`` may instead be a traffic prior (an M4 round,
`prior_train.TRAFFIC_CHECKPOINT_SCHEMA`), read as it was trained.

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
judged end, the reward M4 gives (landed in the airport's landing direction, not ended; a sentence with a go-around
scored on it, `traffic_go_around`), the landing time against the record and how often two commanded aircraft of a window
land in the other order than recorded; for the model's sources the masks on the masked columns (the probability every
mask took away there — the separation masks, and on the approach column the vocabulary's transitions too: multi-aircraft
design §6.6 step 8 item 6), the go-arounds said, and — design §6.6 step 7 item 5 — how the rewards of one window's
aircraft go together over its samples (the pooled correlation of each aircraft's reward less its mean over the samples,
over the pairs of a window).

With ``--augment-seed`` every window is augmented instead (`traffic_window_augment`: the flow compressed, a start moved,
a flight inserted and commanded, a third each; qualified, and never more aircraft at once than the airport's busiest step
on the training days), read by the model's sources only — as M3's first pass on augmented scenes: a moved start has no
record, and its labelled words would fly from the recorded start. Each row carries its window's augmentation and its
aircraft's part in it; the readout adds each kind and each part, and counts the windows left out and the draws refused,
by why.

Writes into a NEW directory ``aircraft.jsonl`` (a row per commanded aircraft, source and sample, appended as each batch
ends) and ``window_generation.json`` (the readout). A batch holds at most ``--aircraft-steps`` window samples × their
aircraft × their steps (the model's past, 6 KB an aircraft-step). **In several processes** (``--workers``, forked once the
data are built and before the GPU is started, as M4's speaking processes are): each reads the batches its index deals it,
each batch — every source of it — from its own streams (`batch_seed`), so what is read does not depend on the number of
processes; the parent writes each batch's rows as they arrive and the readout at the end.

    python run_ts.py traffic_window_generation \\
        --prior 4dTrajectory/outputs/POOLED/prior/v3_stage2_clip_20260926/aug_s1337/round_07 \\
        --executor 4dTrajectory/outputs/POOLED/executor/<spec> \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --split select --out 4dTrajectory/outputs/POOLED/traffic/window_generation_<date>
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import gc
import json
import multiprocessing
import multiprocessing.connection
import os
import signal
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.flights import flight_inputs
from ts_transformer.autopilot.judge import outcome_of
from ts_transformer.experiments.prior_free_generation import (
    _physics, augmented_inputs, fly_reference, glidepath_stops, limits_s, reference_grid, start_altitude_windows,
)
from ts_transformer.experiments.prior_train import (
    PRIOR_CHECKPOINT_SCHEMA, TRAFFIC_CHECKPOINT_SCHEMA, load_prior, rosters,
)
from ts_transformer.experiments.traffic_go_around import (
    GO_AROUND_EXTRA_S, PROBE_MARGIN, AfterGoAround, after_go_around, approach_altitude_m, runway_at,
)
from ts_transformer.experiments.traffic_loop import LOST_SEPARATION, Controlled, Loop, Run, recorded
from ts_transformer.experiments.traffic_labelled import own_end
from ts_transformer.experiments.traffic_speaking import (
    HISTORY_S, MASK_COLUMNS, scene_airports, scene_landings, speaking_aircraft,
)
from ts_transformer.experiments.traffic_window import (
    Commanded, Given, Window, WindowLoop, WindowRecord, _with_landing, draw_windows, window_landings, window_of,
)
from ts_transformer.experiments.traffic_window_augment import KINDS, REFUSALS, ROLES, augment_window, busiest
from ts_transformer.inference.scene_edges import EDGE_FEATURES
from ts_transformer.inference.separation import IFR, VISUAL
from ts_transformer.instructions.artefact import SPLITS
from ts_transformer.instructions.words import COLUMNS, Words
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.augment import Augmentation
from ts_transformer.prior.data import VARIANTS, airport_landings
from ts_transformer.prior.generate import rows_for
from ts_transformer.prior.landing_reward import landing_direction
from ts_transformer.prior.model import Prior, with_traffic
from ts_transformer.prior.scene import N_LOOK, Landings, hang, presence
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

#: v3 (2026-10-01): the model's sources read (`model_sources`) in the header.
SCHEMA = "ts-traffic-window-generation-v4"
SOURCES = ("scene", "alone", "labelled", "recorded")
#: The sources the model speaks in (`--model-sources`: a read that needs only one — a pair of priors on the same windows
#: reads "scene" — skips the other, half the model's time; each source from its own streams, so the rows of the one read
#: do not change).
MODEL_SOURCES = ("scene", "alone")
#: A batch's most aircraft-steps (module docstring): about 0.6 GB of the model's past on its d and layers — a process's,
#: four beside each other in the 8 GB GPU (M4's speaking processes hold as much).
AIRCRAFT_STEPS = 100_000
#: Processes reading the batches (module docstring): the loop is bound by the CPU (the executors, the judge, the masks);
#: the 2026-09-30 smoke held 0.88 GB of the GPU a process, and read the same with 4 and 6.
WORKERS = 6
#: Window sizes the readout is split by: the commanded aircraft of a window.
SIZES = ("1", "2", "3+")
#: Windows drawn an airport by default (`draw_windows`): the formal window readouts' draw, which the Training module's
#: window sets are chosen from (`window_training_export`).
WINDOWS_PER_AIRPORT = 200


def size_of(commanded: int) -> str:
    """A window's size (`SIZES`) by its commanded aircraft."""
    return SIZES[min(commanded, 3) - 1]


def window_prior(prior_dir: Path, instructions: Path, seed: int) -> tuple[Prior, str, dict[str, Any], Any]:
    """The prior that commands a window's aircraft, on the CPU, in eval mode: a single-aircraft prior (augmented) with a
    traffic attention at zero (`with_traffic`, its weights drawn under ``seed``: at zero it reads nothing, so it answers
    as the prior does, to rounding), or a traffic prior (an M4 round) as it was trained — with how its traffic attention
    reads, its config file and its own procedure's masks; refused unless it is either kind's formal run, a traffic prior
    reading today's edge features."""
    loaded, payload, config, own_masks = load_prior(prior_dir, instructions)
    if config["smoke"] or payload["schema"] not in (PRIOR_CHECKPOINT_SCHEMA, TRAFFIC_CHECKPOINT_SCHEMA):
        raise ValueError(f"{prior_dir} is not a single-aircraft or a traffic prior's formal run")
    if payload["schema"] == PRIOR_CHECKPOINT_SCHEMA:
        torch.manual_seed(seed)
        return (with_traffic(loaded, EDGE_FEATURES).eval(), "zero (with_traffic): the prior's answers to rounding",
                config, own_masks)
    if tuple(loaded.traffic_features) != EDGE_FEATURES:
        raise ValueError(f"{prior_dir}'s traffic attention reads other edge features than today's")
    return loaded, "the traffic prior's own", config, own_masks


@dataclasses.dataclass(frozen=True)
class Drawn:
    """The windows read: each window (its commanded aircraft over their time limits), what augmented it
    (`traffic_window_augment`: its kind, draws and what was drawn; None: as drawn) and its commanded aircraft's places
    in ``batch`` (`traffic_window.WindowDraw.members`), with each one's time limit, moved start (None: its own) and part
    in its window's augmentation (`traffic_window_augment.ROLES`; None: as drawn)."""

    windows: list[Window]
    augmented: list[dict[str, Any] | None]
    members: list[range]
    batch: replay.Batch
    limits: list[float]
    moves: list[Augmentation | None]
    roles: list[str | None]


def drawn_windows(draw: Any, airports: Mapping[str, Any], params: Any, step_s: float) -> Drawn:
    """`traffic_window.draw_windows`' windows over their commanded aircraft's time limits (the executor spec's)."""
    limits = limits_s(draw.batch, params, step_s, augmented=False)
    members = draw.members()
    windows = [window_of(airports[code], opens, commanded, [limits[j] for j in part], step_s)
               for (code, opens, commanded), part in zip(draw.openings, members)]
    count = len(draw.batch.signals)
    return Drawn(windows, [None] * len(windows), members, draw.batch, limits, [None] * count, [None] * count)


def augmented_windows(drawn: Drawn, params: Any, most: Mapping[str, int], max_rows: int, seed: int,
                      windows: dict[str, tuple[float, float]], spec: Any, kinds: Sequence[str] = KINDS
                      ) -> tuple[Drawn, dict[str, Any]]:
    """Every window of ``drawn`` augmented (`traffic_window_augment.augment_window` over ``kinds``, one generator from
    ``seed`` in the windows' order; an insertion drawn from the draw's flights at the window's airport, ``most`` each
    airport's busiest step on the training days, ``max_rows`` the model's positions): the windows kept, and the count of
    the kinds kept, of the windows left out (by airport and kind) and of the draws refused, by why."""
    rng = np.random.default_rng(seed)
    moved_limits = limits_s(drawn.batch, params, spec.step_s, augmented=True)
    pools: dict[str, dict[str, int]] = defaultdict(dict)
    for j, signals in enumerate(drawn.batch.signals):
        pools[signals.airport].setdefault(signals.dataset_id, j)
    kept, left_out, refused = [], defaultdict(Counter), Counter()
    for window, part in zip(drawn.windows, drawn.members):
        code = window.airport.flights.code
        got, kind, why = augment_window(window, drawn.batch, list(part), pools[code], drawn.limits, moved_limits,
                                        most[code], max_rows, rng, windows, spec, kinds)
        refused.update(why)
        if got is None:
            left_out[code][kind] += 1
        else:
            kept.append(got)
    batch = dataclasses.replace(replay.subset(drawn.batch, [j for a in kept for j in a.places]),
                                signals=[s for a in kept for s in a.signals])
    members, at = [], 0
    for a in kept:
        members.append(range(at, at + len(a.places)))
        at += len(a.places)
    counts = {"kinds": {kind: sum(a.kind == kind for a in kept) for kind in KINDS},
              "left_out": {code: dict(by_kind) for code, by_kind in sorted(left_out.items())},
              "refused": {why: refused[why] for why in REFUSALS},
              "busiest_training_step": dict(most),
              "cap_raised_by_the_recorded_window": dict(Counter(a.window.airport.flights.code for a in kept
                                                                if a.drawn["most_at_once"]["cap"]
                                                                > most[a.window.airport.flights.code]))}
    return Drawn([a.window for a in kept], [{"kind": a.kind, "draws": a.draws, **a.drawn} for a in kept], members,
                 batch, [x for a in kept for x in a.limits], [m for a in kept for m in a.moves],
                 [r for a in kept for r in a.roles]), counts


def window_size(window: Window, limits: Sequence[float], step_s: float) -> tuple[int, int, int, int]:
    """A window's aircraft, pre-roll (before its first commanded aircraft's row 0, as the speaker caps it), latest
    entry after it and rows (its longest time limit's with a go-around's `GO_AROUND_EXTRA_S`, as the loop lays it out),
    in steps."""
    first = window.first_step_s(window.commanded[0], step_s)
    pre = max([0] + [int(round((first - float(hang(window.rows(k).presence.start_s, step_s))) / step_s))
                     for k in window.others])
    late = max(int(round((window.first_step_s(k, step_s) - first) / step_s)) for k in window.commanded)
    return (len(window.commanded) + len(window.others), min(pre, int(HISTORY_S // step_s)), late,
            rows_for(max(limits) + GO_AROUND_EXTRA_S + step_s, step_s))


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
    return packed(sizes, samples, budget)


def packed(sizes: Sequence[tuple[int, int, int, int]], samples: int, budget: int) -> list[list[int]]:
    """Windows of ``sizes`` (`window_size` each; a window may be there more than once) in batches of the loop, smallest
    first, each at most ``budget`` aircraft-steps (`batch_cost`; a larger one is a batch of its own): their indices."""
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


def go_around_fields(loop: WindowLoop, i: int, got: Commanded, row: Mapping[str, Any],
                     direction: np.ndarray) -> AfterGoAround:
    """Commanded aircraft ``i``'s first go-around scored (`traffic_go_around.after_go_around`) from what the loop kept of
    it: its margins, its states a step, its last judged step and the approach altitude of the runway in force then
    (`traffic_go_around.approach_altitude_m`)."""
    states = loop.states[i]
    entry = approach_altitude_m(loop.geometries[i], runway_at(got.said, got.go_around))
    judged_to = int(min(loop.judged_to[i], len(states) - 1))
    return after_go_around(got.said, got.go_around, row["outcome"], bool(direction[row["runway"]]), loop.margin[i],
                           [s.height_m for s in states], [s.ground_speed_mps for s in states],
                           [s.captured for s in states], judged_to, entry, loop.step_s, float(loop.extra_s[i]),
                           loop.go_around_extra_s)


@dataclasses.dataclass
class Flown:
    """Windows spoken to and flown (`fly_windows`): the loop once it has run, its results, the flights it flew (``part``,
    each at ``index`` of the draw's batch) and each loop window's drawn window (``instances``: a window spoken to ``K``
    times is there ``K`` times, its samples in order)."""

    loop: WindowLoop
    results: list[Commanded]
    part: replay.Batch
    index: list[int]
    instances: list[int]

    def members(self, b: int) -> list[int]:
        """Loop window ``b``'s commanded aircraft: their places in the loop."""
        return [i for i, r in enumerate(self.results) if r.window == b]

    def judged_again(self, b: int, reading: str, step_s: float, window: Window) -> Run:
        """Loop window ``b``'s commanded aircraft along the paths they flew, judged afterwards under ``reading``
        (`_judged_again`: the IFR reading beside the loop's VISUAL one)."""
        here = self.members(b)
        until = {self.results[i].key: self.loop.judged_until_s(i) for i in here}
        return _judged_again(window, [self.loop.path(i) for i in here], until, reading, step_s)


def fly_windows(model: Prior, drawn: Drawn, chunk: Sequence[int], source: str, words: Words, params: Any,
                landings: Any, samples: int, *, generator: torch.Generator, temperature: float,
                procedure_masks: Any, given: Sequence[Given | None] | None = None, probe_samples: int = 0,
                probe_margin: float = PROBE_MARGIN) -> Flown:
    """The windows at ``chunk`` spoken to ``samples`` times each (`WindowLoop`; ``source`` "alone": each aircraft
    hearing no other) and flown to their end; ``landings``: the model's landing context (None for a variant without
    one); ``given``: each loop aircraft's given words (`traffic_window.Given`, None: spoken), in the loop's order;
    ``probe_samples``: the last that many samples of each window probed (`WindowLoop`'s ``probing``, multi-aircraft design
    §6.6 step 8 item 10). The caller closes the loop (`WindowLoop.close`)."""
    if not 0 <= probe_samples <= samples:
        raise ValueError(f"{probe_samples} probed samples of {samples}")
    cpu = torch.device("cpu")
    instances = [w for w in chunk for _ in range(samples)]
    index = [j for w in instances for j in drawn.members[w]]
    part = replay.subset(drawn.batch, index)
    runways, charts, approach = _physics(part, cpu)
    inputs = augmented_inputs(flight_inputs(part.series, device=cpu, anchor=N_LOOK), part.geometries,
                              [drawn.moves[j] for j in index])
    probing = [b % samples >= samples - probe_samples for b, w in enumerate(instances) for _ in drawn.members[w]]
    loop = WindowLoop(model, [drawn.windows[w] for w in instances], part.signals, part.geometries, inputs, runways,
                      charts, approach, [drawn.limits[j] for j in index], words, params, landings, generator=generator,
                      temperature=temperature, procedure_masks=procedure_masks, alone=source == "alone", given=given,
                      probing=probing, probe_margin=probe_margin)
    while loop.running:
        loop.step()
    return Flown(loop, loop.results(), part, index, instances)


@dataclasses.dataclass(frozen=True)
class WindowSentences:
    """What windows said (`window_sentences`): a row per commanded aircraft and sample (`model_rows`'), window sample
    after window sample, and beside each what the speaker read of it (`traffic_window.WindowRecord`) and what the masks
    let it say over its counted steps (per masked column, its bit-packed classes a step: `WindowSpeaker.allowed`)."""

    rows: list[dict[str, Any]]
    records: list[WindowRecord]
    allowed: list[dict[int, np.ndarray]]


def model_rows(model: Prior, drawn: Drawn, chunk: Sequence[int], source: str, words: Words, params: Any,
               landings: Any, every_landing: Mapping[str, Landings], samples: int, *, generator: torch.Generator,
               temperature: float, procedure_masks: Any, probe_samples: int = 0,
               probe_margin: float = PROBE_MARGIN) -> list[dict[str, Any]]:
    """The windows at ``chunk`` spoken to ``samples`` times each (`fly_windows`), a row per commanded aircraft and
    sample; ``every_landing``: the airports' landings the reward's landing direction reads; ``probe_samples``,
    ``probe_margin``: `fly_windows`'."""
    return window_sentences(model, drawn, chunk, source, words, params, landings, every_landing, samples,
                            generator=generator, temperature=temperature, procedure_masks=procedure_masks,
                            probe_samples=probe_samples, probe_margin=probe_margin).rows


def window_sentences(model: Prior, drawn: Drawn, chunk: Sequence[int], source: str, words: Words, params: Any,
                     landings: Any, every_landing: Mapping[str, Landings], samples: int, *, generator: torch.Generator,
                     temperature: float, procedure_masks: Any, probe_samples: int = 0,
                     probe_margin: float = PROBE_MARGIN) -> WindowSentences:
    """`model_rows` with what the speaker read of each aircraft and what its masks allowed (`WindowSentences`: what a
    trainer scores the words with); ``probe_samples``, ``probe_margin``: `fly_windows`'."""
    flown = fly_windows(model, drawn, chunk, source, words, params, landings, samples, generator=generator,
                        temperature=temperature, procedure_masks=procedure_masks, probe_samples=probe_samples,
                        probe_margin=probe_margin)
    out = flown_sentences(flown, drawn, source, words, every_landing, samples)
    flown.loop.close()
    return out


def flown_sentences(flown: Flown, drawn: Drawn, source: str, words: Words, every_landing: Mapping[str, Landings],
                    samples: int) -> WindowSentences:
    """`window_sentences` of windows already flown (`fly_windows`; the loop is left open)."""
    step_s = words.spec.step_s
    loop, results, part, index, instances = flown.loop, flown.results, flown.part, flown.index, flown.instances
    records = loop.records()
    rows, read, allowed = [], [], []
    for b, w in enumerate(instances):
        window = drawn.windows[w]
        here = flown.members(b)
        until = {results[i].key: loop.judged_until_s(i) for i in here}
        again = flown.judged_again(b, IFR, step_s, window)
        loop_landings = [(results[i].landing_s, part.geometries[i].candidates[results[i].runway].ident)
                         for i in here if results[i].landing_s is not None]
        for i in here:
            got = results[i]
            row = _aircraft_row(drawn, part, i, index[i], w, source, b % samples)
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
            row["landed_here"] = bool(row["outcome"] == "landed" and direction[row["runway"]])
            row["probed"], row["forced"] = bool(loop.probing[i]), got.forced
            row["go_around"] = None
            if got.go_around is not None:                   # multi-aircraft design §6.6 step 8 item 9
                scored = go_around_fields(loop, i, got, row, direction)
                row["go_around"], row["reward"] = scored.fields(), scored.reward
            rows.append(row)
            read.append(records[i])
            allowed.append({c: masks[i, : got.counted].copy() for c, masks in loop.speaker.allowed.items()})
    return WindowSentences(rows, read, allowed)


def _loop_context(window: Window, key: str, landings: Mapping[str, Landings],
                  loop_landings: Sequence[tuple[float, str]], step_s: float) -> Landings:
    """``key``'s landings as its window had them in the loop — `window_landings` and the others' landings there — for
    the landing direction (its own recorded one still in, for `data.own_context`)."""
    out = window_landings(window, key, landings, step_s)
    for landed_s, runway in loop_landings:
        out = _with_landing(out, landed_s, runway)
    return out


def _aircraft_row(drawn: Drawn, part: replay.Batch, i: int, j: int, w: int, source: str, sample: int | None
                  ) -> dict[str, Any]:
    """Commanded aircraft ``i`` of ``part`` (``j`` of ``drawn``) in window ``w``: who it is and how it was read."""
    signals = part.signals[i]
    return {"dataset_id": signals.dataset_id, "airport": signals.airport, "window": w,
            "commanded": len(drawn.windows[w].commanded), "source": source, "sample": sample,
            "observed_runway": part.readings[i].runway_index, "augmented": drawn.augmented[w], "role": drawn.roles[j]}


@dataclasses.dataclass
class FixedWindow:
    """A window's commanded aircraft on paths that react to nothing (`fixed_paths`): each one's path, own end, runway
    (a candidate's index) and last judged instant, in the window's order, judged under VISUAL (``run``) and IFR
    (``again``)."""

    w: int
    paths: list[Controlled]
    owns: list[str]
    runways: list[int]
    until: dict[str, float]
    run: Run
    again: Run


def fixed_paths(drawn: Drawn, chunk: Sequence[int], source: str, words: Words, params: Any, procedure_masks: Any
                ) -> tuple[replay.Batch, list[int], list[FixedWindow]]:
    """The windows at ``chunk`` with every commanded aircraft on a path that reacts to nothing — its labelled words
    flown from its first predicted step (``labelled``) or its own record on the loop's steps (``recorded``) — judged
    afterwards (`_judged_again`) under VISUAL and IFR: the flights (``part``, each at ``index`` of the draw's batch) and
    each window's `FixedWindow`. A labelled path runs to its executor's end and is passive past a glidepath stop, as a
    model aircraft flies on past one (design §9 item 29)."""
    if any(drawn.augmented[w] is not None for w in chunk):
        raise ValueError("an augmented window is read by the model's sources only: a moved start has no record, and "
                         "the scene of an inserted aircraft is its source's")
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
    out, at = [], 0
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
        out.append(FixedWindow(w, paths, owns, runways, until, run, again))
        at += len(window.commanded)
    return part, index, out


def fixed_rows(drawn: Drawn, chunk: Sequence[int], source: str, words: Words, params: Any, landings: Any,
               procedure_masks: Any) -> list[dict[str, Any]]:
    """`fixed_paths`' windows, a row per commanded aircraft. The landing direction reads the path's own world: the
    recorded landings for the record, the labelled paths' landings in place of the commanded aircraft's recorded ones
    for the labelled words."""
    step_s = words.spec.step_s
    part, index, fixed = fixed_paths(drawn, chunk, source, words, params, procedure_masks)
    rows, at = [], 0
    for item in fixed:
        window, paths, runways = drawn.windows[item.w], item.paths, item.runways
        for m, (key, path) in enumerate(zip(window.commanded, paths)):
            j = at + m
            row = _aircraft_row(drawn, part, j, index[j], item.w, source, None)
            row.update(path_fields(item.run, item.again, key, path, window.commanded, item.owns[m], item.until[key]),
                       landing_s=path.landing_s if item.owns[m] == "landed" else None, runway=runways[m])
            if source == "labelled":
                context = _loop_context(window, key, landings,
                                        [(other.landing_s, part.geometries[at + n].candidates[runways[n]].ident)
                                         for n, other in enumerate(paths)
                                         if n != m and other.landing_s is not None], step_s)
            else:
                context = scene_landings(landings[window.airport.flights.code], window.scene(key, step_s))
            direction = landing_direction(part.signals[j], part.geometries[j], context)
            row["reward"] = _reward(row, direction)
            row["landed_here"] = bool(row["outcome"] == "landed" and direction[row["runway"]])
            row["probed"], row["forced"] = False, None
            row["go_around"] = None                         # the labelled words and the record say none
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
    and — the model's sources — the separation masks, how one window's rewards go together and the go-arounds said
    (`go_arounds`)."""
    # the record's landing, whatever the judge made of the record's path: a fact to read the others against
    recorded_landing = {(r["window"], r["dataset_id"]): r["landing_s"] for r in rows if r["source"] == "recorded"}
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
            entry["go_arounds"] = go_arounds(part)
        out[source] = entry
    return out


def go_arounds(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The sentences of ``rows`` that said a go-around (multi-aircraft design §6.6 step 8): how many, their outcomes,
    the mean of each of their reward's parts and of the reward, and how many got less than the time a go-around gives
    (the model's positions ended first)."""
    said = [r for r in rows if r["go_around"] is not None]
    if not said:
        return {"sentences": 0, "share": 0.0}
    parts = ("separation", "climb", "back", "landed", "reward")
    return {"sentences": len(said), "share": len(said) / len(rows),
            "outcomes": dict(Counter(r["outcome"] for r in said).most_common()),
            "said_by_a_probe": sum(r["forced"] is not None and r["forced"] == r["go_around"]["step"] for r in said),
            "mean": {name: sum(r["go_around"][name] for r in said) / len(said) for name in parts},
            "short_of_the_extra_time": sum(r["go_around"]["extra_s"] < r["go_around"]["extra_wanted_s"] for r in said)}


def _order(rows: Sequence[dict[str, Any]], recorded_landing: Mapping[tuple[int, str], float | None]) -> dict[str, int]:
    """Pairs of one window's commanded aircraft that both landed (not ended by the judge) in a sample and both have a
    recorded landing (the record's, whatever the judge made of its paths): how many, and how many landed the other way
    round."""
    pairs = swapped = 0
    by_sample: dict[tuple[int, Any], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_sample[(r["window"], r["sample"])].append(r)
    for (w, _), members in by_sample.items():
        landed = [(r["landing_s"], recorded_landing.get((w, r["dataset_id"]))) for r in members
                  if (r["outcome"] == "landed" or r["source"] == "recorded")
                  and recorded_landing.get((w, r["dataset_id"])) is not None]
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


def summaries(rows: Sequence[dict[str, Any]], augmented: bool) -> dict[str, Any]:
    """`summary` pooled, per airport and per window size — and, on ``augmented`` windows, per kind and per commanded
    aircraft's part in its window's augmentation (`traffic_window_augment.ROLES`; "as drawn": the others of a B or an
    A window)."""
    by_airport: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_size: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_airport[row["airport"]].append(row)
        by_size[size_of(row["commanded"])].append(row)
    out = {"pooled": summary(rows), "airports": {code: summary(part) for code, part in sorted(by_airport.items())},
           "window_sizes": {size: summary(by_size[size]) for size in SIZES if by_size[size]}}
    if augmented:
        kinds = {kind: [r for r in rows if r["augmented"]["kind"] == kind] for kind in KINDS}
        roles = {role or "as drawn": [r for r in rows if r["role"] == role] for role in (*ROLES, None)}
        out["kinds"] = {kind: summary(part) for kind, part in kinds.items() if part}
        out["roles"] = {role: summary(part) for role, part in roles.items() if part}
    return out


def batch_rows(model: Prior, drawn: Drawn, number: int, chunk: Sequence[int], words: Words, params: Any,
               landings: Any, every_landing: Mapping[str, Landings], samples: int, *, seed: int, temperature: float,
               procedure_masks: Any, device: torch.device, model_sources: Sequence[str] = MODEL_SOURCES,
               probe_samples: int = 0, probe_margin: float = PROBE_MARGIN) -> list[dict[str, Any]]:
    """Loop batch ``number`` (the windows at ``chunk``) read by the model's sources (``model_sources``), each from its
    own stream (`batch_seed`), and — windows as drawn — the labelled words and the records: its rows, each marked with
    the batch."""
    rows: list[dict[str, Any]] = []
    for source in model_sources:
        generator = torch.Generator(device=device).manual_seed(batch_seed(seed, source, number))
        rows += model_rows(model, drawn, chunk, source, words, params, landings, every_landing, samples,
                           generator=generator, temperature=temperature, procedure_masks=procedure_masks,
                           probe_samples=probe_samples, probe_margin=probe_margin)
    for source in ("labelled", "recorded") if drawn.augmented[chunk[0]] is None else ():
        rows += fixed_rows(drawn, chunk, source, words, params, every_landing, procedure_masks)
    for row in rows:
        row["batch"] = number
    return rows


#: `prctl` option: the signal a process gets when its parent dies (linux/prctl.h).
PR_SET_PDEATHSIG = 1


def _worker(pipe: Any, parent_ends: Sequence[Any], parent_pid: int, index: int, count: int, numbers: Sequence[int],
            read: Callable[[int], Any]) -> None:
    """A reading process (`in_processes`): it dies with the parent, keeps none of the others' pipes and runs one thread;
    it answers with ``read(number)`` of each of ``numbers`` its index deals it, its GPU peak beside, then "done" — or
    the traceback that ended it (also printed)."""
    if sys.platform == "linux":                                 # the death signal is Linux's; elsewhere none is asked for
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG)")
        if os.getppid() != parent_pid:
            os._exit(1)
    for end in parent_ends:
        end.close()
    torch.set_num_threads(1)
    try:
        for number in numbers[index::count]:
            got = read(number)
            peak = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_initialized() else 0.0
            pipe.send(("ok", number, got, peak))
        pipe.send(("done",))
    except BaseException:
        traceback.print_exc()
        pipe.send(("failed", traceback.format_exc()))


def in_processes(count: int, numbers: Sequence[int], read: Callable[[int], Any]) -> Iterator[tuple[int, Any, float]]:
    """``read(number)`` of every one of ``numbers`` in ``count`` forked processes (`_worker`), as each is read:
    ``(number, what it read, its process's GPU peak GB)``; a process that fails, or is gone, ends the run with what it
    said. Fork before the parent starts the GPU."""
    if count < 1:
        raise ValueError("at least one reading process")
    context = multiprocessing.get_context("fork")
    pipes, processes = [], []
    for index in range(count):
        parent, child = context.Pipe()
        process = context.Process(target=_worker, args=(child, list(pipes), os.getpid(), index, count, list(numbers),
                                                        read), daemon=True)
        process.start()
        child.close()
        pipes.append(parent)
        processes.append(process)
    live = dict(enumerate(pipes))
    try:
        while live:
            for pipe in multiprocessing.connection.wait(list(live.values())):
                w = next(k for k, v in live.items() if v is pipe)
                try:
                    message = pipe.recv()
                except (EOFError, OSError):
                    processes[w].join(timeout=5)
                    raise SystemExit(f"reading process {w} is gone (exit code {processes[w].exitcode})") from None
                if message[0] == "failed":
                    raise SystemExit(f"reading process {w} failed:\n{message[1]}")
                if message[0] == "done":
                    del live[w]
                    continue
                _, number, got, peak = message
                yield number, got, peak
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the single-aircraft prior it starts from (augmented), "
                        "or a traffic prior (an M4 round)")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--windows-per-airport", type=int, default=WINDOWS_PER_AIRPORT)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--aircraft-steps", type=int, default=AIRCRAFT_STEPS, help="a batch's most (module docstring)")
    parser.add_argument("--workers", type=int, default=WORKERS, help="reading processes (what is read does not depend "
                        "on it)")
    parser.add_argument("--augment-seed", type=int, default=None, help="every window augmented with this seed "
                        "(`traffic_window_augment`; the model's sources only)")
    parser.add_argument("--device", default="cuda", help="the prior's; the executors fly on CPU")
    parser.add_argument("--model-sources", nargs="+", choices=MODEL_SOURCES, default=list(MODEL_SOURCES),
                        help="the model's sources to read (`MODEL_SOURCES`; the rows show which were read)")
    parser.add_argument("--probe-samples", type=int, default=0, help="the last that many samples of each window probed "
                        "for a go-around (multi-aircraft design §6.6 step 8 item 10: what a training round's probes do; "
                        "0: none)")
    parser.add_argument("--probe-margin", type=float, default=PROBE_MARGIN, help="a probe's trigger: the tightest margin")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)
    if args.windows_per_airport < 1 or args.samples < 1 or args.workers < 1:
        parser.error("at least one window an airport, one sample and one reading process")
    if not 0 <= args.probe_samples <= args.samples:
        parser.error(f"--probe-samples {args.probe_samples} of {args.samples} samples")

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    prior_dir, instructions, executor_dir, out = map(resolved, (args.prior, args.instructions, args.executor, args.out))
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    git = git_state()
    started = time.perf_counter()
    params, record, words = replay.open_executor(executor_dir, instructions)
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
    augmented = args.augment_seed is not None
    augmenting = None
    if augmented:
        most = busiest(instructions, words.spec, model.config.airports, step_s)
        drawn, augmenting = augmented_windows(drawn, params, most, model.config.max_rows, args.augment_seed,
                                              start_altitude_windows(instructions), words.spec)
        print(f"augmented windows: {augmenting}", flush=True)
        if not drawn.windows:
            parser.error("every window was left out")
    print(f"{len(drawn.windows)} {args.split} windows, {len(drawn.batch.readings)} commanded aircraft "
          f"({draw.counts}), scenes built ({built}), {time.perf_counter() - started:.0f}s", flush=True)

    batches = window_batches(drawn, args.samples, args.aircraft_steps, step_s)
    out.mkdir(parents=True)
    device = torch.device(args.device)

    def read(number: int) -> list[dict[str, Any]]:
        speaking = model.to(device)
        return batch_rows(speaking, drawn, number, batches[number], words, params, landings, every_landing,
                          args.samples, seed=args.seed, temperature=args.temperature, procedure_masks=own_masks,
                          device=device, model_sources=tuple(s for s in MODEL_SOURCES if s in args.model_sources),
                          probe_samples=args.probe_samples, probe_margin=args.probe_margin)

    gc.collect()
    gc.freeze()                                         # the reading processes share the parent's data, not copy it
    rows: list[dict[str, Any]] = []
    done, peaks = 0, [0.0]
    for number, new, peak in in_processes(args.workers, list(range(len(batches))), read):
        with (out / "aircraft.jsonl").open("a", encoding="utf-8") as stream:
            for row in new:
                stream.write(json.dumps(row) + "\n")
        rows += new
        done += len(batches[number])
        peaks.append(peak)
        print(f"  batch {number}: {done}/{len(drawn.windows)} windows ({len(batches)} batches), "
              f"{time.perf_counter() - started:.0f}s, GPU {max(peaks):.2f} GB a process at most", flush=True)

    # the batches in their order (they arrive in the processes' order): the readout, and the rows written again so, the
    # same whatever the number of processes
    rows = sorted(rows, key=lambda row: row["batch"])      # (stable: a batch's rows keep their order)
    with (out / "aircraft.jsonl.sorted").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    (out / "aircraft.jsonl.sorted").replace(out / "aircraft.jsonl")
    readout = summaries(rows, augmented)
    write_json_atomic(out / "window_generation.json", {
        "schema": SCHEMA, "written_utc": utc_now(), "git": git, "split": args.split, "drawn": draw.counts,
        "windows_per_airport": args.windows_per_airport, "samples": args.samples, "temperature": args.temperature,
        "probes": {"samples": args.probe_samples, "margin": args.probe_margin},
        "seed": args.seed, "augment_seed": args.augment_seed, "augmenting": augmenting,
        "prior": {"directory": repo_relative(prior_dir), "procedure_masks": list(own_masks.names),
                  "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt")},
        "traffic_attention": attention,
        "executor": {"directory": repo_relative(executor_dir), "sha256": record["sha256"]},
        "instructions": repo_relative(instructions), "scenes": built, "history_s": HISTORY_S,
        "readings": {"ends": VISUAL, "beside": IFR}, "aircraft_steps": args.aircraft_steps, "batches": len(batches),
        "model_sources": [s for s in MODEL_SOURCES if s in args.model_sources],
        "workers": args.workers, "gpu_peak_gb_a_process": max(peaks),
        "readout": readout, "aircraft_file": "aircraft.jsonl", "elapsed_s": time.perf_counter() - started})
    for source, entry in readout["pooled"].items():
        shares = "  ".join(f"{name} {share:.3f}" for name, share in entry["outcomes"].items())
        print(f"{source:9s} n={entry['aircraft']:5d}  reward {entry['reward']:.3f}  lost separation "
              f"{entry['lost_separation']:.3f} (IFR {entry['lost_separation_ifr']:.3f}, with {entry['lost_separation_with']})"
              f"  order {entry['order']}  {shares}")
        if "reward_together" in entry:
            print(f"          masks: steps {entry['mask_steps_share']}  rewards together {entry['reward_together']}")
    for group in ("kinds", "roles") if augmented else ():
        for name, part in readout[group].items():
            print(f"{name:9s} " + "  ".join(f"{source} n={entry['aircraft']} reward {entry['reward']:.3f} lost "
                                            f"{entry['lost_separation']:.3f}" for source, entry in part.items()))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
