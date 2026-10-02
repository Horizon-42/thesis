"""Multi-aircraft M4's second pass (design §6.6 step 7 item 7, the 7.6 plan): the traffic post-training in windows — the
model commanding every aircraft of a window at once (`traffic_window.WindowLoop`), rewarded per aircraft for landing
without losing separation, pulled back to base; the round protocol M4's (`traffic_rounds`, written for M4's first
runner), the unit a window sample instead of a scene's one aircraft.

Each round:

1. **windows** — the training days' windows drawn with the round's seed (`traffic_window.draw_windows`, ``seed`` + the
   round): of each airport's, the first ``--real-per-airport`` as they are and, of the next `POOL_FACTOR` ×
   ``--augmented-per-airport``, the first ``--augmented-per-airport`` whose augmentation qualifies
   (`traffic_window_augment`: the flow compressed, a start moved, a flight inserted and commanded, a third each; the
   round's augmentation stream) — refused when an airport runs short. ``--commanded one`` (design §6.6 step 9, "9.4 的代码":
   the setting "一架由模型指挥"): as many flights an airport drawn as the single-aircraft runners draw them, each the one
   commanded aircraft of its own window, the others replayed; augmented by its leader moved, its start moved, a flight
   inserted and replayed (`traffic_window_augment.KINDS_OF`); with ``--events``, ``--events-per-airport`` hard
   events an airport (all where it has fewer; the round's pick stream) from training-day window rewind runs (R43,
   `traffic_window_events`: the answered aircraft could not undo the loss speaking again from its start) — the window
   flown again with every other commanded aircraft given its words, the answered one spoken (multi-aircraft design §6.6
   step 8 item 11);
2. **sentences** — each window spoken to ``--samples`` times (`traffic_window_generation.window_sentences`: every
   commanded aircraft together, the vocabulary's rules, the start's procedure's masks, the two separation masks, judged
   as it flies under VISUAL), by ``--speakers`` processes (`traffic_rounds.Speakers`, each loop batch its own stream);
   the last ``--probe-samples`` of them probes (multi-aircraft design §6.6 step 8 item 10: an established aircraft
   whose margin fell is made to go around once);
3. **rewards** — per commanded aircraft: 1 for landing on a runway in the airport's landing direction (against its
   window's landings as the loop had them) with no loss of separation ending it first, else 0; a sentence that says a
   go-around is scored on what it did, at most 0.9 (`traffic_go_around`, multi-aircraft design §6.6 step 8 item 9); its
   advantage its reward less its mean over its window's samples spoken the same way (the unprobed, the probes that said
   no go-around, the probes that said one), a probe's go-around word learned where it beat the unprobed samples' mean;
   an aircraft is trained on only where a group's rewards differ, or for a go-around word that did better, and none of
   its samples starts in a loss it answers for (`traffic_window_tuner.window_advantages`);
4. **``--passes`` passes** (`traffic_window_tuner.WindowRewardTuner`): M4's loss, each window sample scored whole as the
   speaker read it (the other commanded aircraft at the rows they flew with the words they were said), the loss on its
   trained aircraft's own words, the pull to base reading each alone, the data term the training days' scene samples;
   the distance to base on the fresh sentences before the pass, the real windows', the augmented ones' and the hard
   events' apart;
5. **the select readouts** (every round, round 0 the start): the select days' ``--select-per-airport`` windows ×
   ``--select-samples``, as drawn and augmented once (``seed`` + `SELECT_AUGMENT_OFFSET`), the same windows, streams and
   batches every round — per side the executor's landed share, landed on the observed runway, the words a sentence says
   after its first step, lost separation, the reward (and per kind), on the real windows the ordering (`ordering`); the
   teacher-forced NLL on the select days' scene samples and the traffic attention's output (`traffic_rounds.
   traffic_readout`); with ``--select-events``, every select-day hard event of those rewind runs too, as its window is
   flown for training (`event_readout`: the answered aircraft's reward, landing, loss and go-arounds) — read, never in the
   round's choice.

The round kept (``choice.json``): `traffic_rounds.guarded_choice` — within round 0's guards on these readouts, the
augmented-window reward beating round 0 by `TIE_STANDARD_ERRORS` paired standard errors (the select readouts speak the
same draws every round: each aircraft sentence of a round is paired with its own in round 0), the earliest the highest
does not beat by as much; none: round 0.

**Round by round** as M4: every stream of a round its own ([``seed``, round, stream]), the optimiser's state beside each
round's weights, ``--resume`` refused on another configuration, code or device, or a round cut short. ``--prior`` is the
start: a single-aircraft prior (augmented, given a traffic attention at zero) or a traffic prior (an M4 round, read as it
was trained); ``--base`` the pull's reference, a single-aircraft prior.

Writes into ``--out`` (a new directory, from a clean tree unless ``--smoke``; the same one with ``--resume``):
``config.json``, ``preflight.json``, ``round_00/readout.json``, ``round_<k>/{sentences.json, sentences.npz,
checkpoint.pt, optimiser.pt, config.json, procedure_masks.json, readout.json}``, ``history.json``, ``choice.json``.

**Checked at the formal size before anything is spoken** (the first chain failed twice where a smoke could not reach):
the host's free memory against ``--speakers`` × `SPEAKER_HOST_GB` + `PARENT_GROWTH_GB`, and — on a fresh run — round
1's costliest window sample spoken and scored with gradients on the GPU beside the speaking processes (`preflight`).

    python run_ts.py traffic_window_reward \\
        --prior 4dTrajectory/outputs/POOLED/prior/m4_passes_20260929/traffic_s1337/round_05 \\
        --base 4dTrajectory/outputs/POOLED/prior/v3_step1_20260924/full_s1337 \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --executor 4dTrajectory/outputs/POOLED/executor/v11_20260927 \\
        --out 4dTrajectory/outputs/POOLED/prior/<campaign>/window_s1337 --rounds 1
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import psutil
import torch

from ts_transformer.autopilot import replay
from ts_transformer.experiments.prior_augmented_reward import (
    GUARD_LANDED_DROP, GUARD_WORD_COLUMNS, GUARD_WORD_GROWTH, POOL_FACTOR, SELECT_AUGMENT_OFFSET, labelled_words,
)
from ts_transformer.experiments.prior_free_generation import start_altitude_windows
from ts_transformer.experiments.prior_landing_reward import GUARD_RUNWAY_DROP
from ts_transformer.experiments.prior_train import (
    PRIOR_CHECKPOINT_SCHEMA, TRAFFIC_CHECKPOINT_SCHEMA, load_prior, rosters,
)
from ts_transformer.experiments.traffic_loop import LOST_SEPARATION
from ts_transformer.experiments.traffic_rounds import (
    AUGMENT_STREAM, GUARD_LOSS_RISE, GUARD_ORDER_GROWTH, OPTIMISER_FILE, PASS_STREAM, SAMPLING_STREAM, SELECT_STREAMS,
    SPEAKER_AIRCRAFT_STEPS, TIE_STANDARD_ERRORS, TRAFFIC_LEARNING_RATE, Speakers, batch_seed, completed_rounds,
    guarded_choice, ordering_failures, paired_difference, round_seed, run_differences, traffic_readout,
    write_traffic_prior,
)
from ts_transformer.experiments.traffic_scene_data import airport_flights, edge_source_sha256, split_samples
from ts_transformer.experiments.traffic_speaking import with_tracks
from ts_transformer.experiments.traffic_tuner import part_cost
from ts_transformer.experiments.traffic_window import COMMANDED, Given, draw_windows, window_places
from ts_transformer.experiments.traffic_window_augment import KINDS_OF, busiest
from ts_transformer.experiments.traffic_window_events import EventPool, event_pool, pick
from ts_transformer.experiments.traffic_go_around import GO_AROUND_EXTRA_S, IMITATION_WEIGHT, PROBE_MARGIN
from ts_transformer.experiments.traffic_window_generation import (
    Drawn, WindowSentences, augmented_windows, drawn_join, drawn_subset, drawn_windows, fixed_rows, window_batches,
    window_sentences,
)
from ts_transformer.experiments.traffic_window_tuner import (
    PAIRS_PER_BLOCK, WindowRewardTuner, WindowSplit, scoring_cost, window_advantages, window_flight,
)
from ts_transformer.inference.scene_edges import EDGE_FEATURES
from ts_transformer.instructions.artefact import load_candidates, load_spec
from ts_transformer.instructions.words import COLUMNS, UNCHANGED, Words
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.data import (
    VARIANTS, Split, airport_landings, candidate_table, column_classes, runway_names,
)
from ts_transformer.prior.generate import rows_for
from ts_transformer.prior.landing_reward import LANDED
from ts_transformer.prior.masks import ProcedureMasks
from ts_transformer.prior.model import Prior, with_traffic
from ts_transformer.prior.scene import N_LOOK, Landings
from ts_transformer.prior.train import RewardConfig
from ts_transformer.repo_layout import REPO_ROOT, git_state

#: v5 (multi-aircraft step 9): how a window's aircraft are commanded (``commanded``, `traffic_window.COMMANDED`).
SCHEMA = "ts-traffic-window-reward-v5"
RUNNER = "ts_transformer.experiments.traffic_window_reward"
#: Host memory a speaking process holds beyond what it shares with the parent, and what the parent grows by in a round
#: (a round's windows built, the pass), GB — measured on the formal run (2026-09-30: a speaking process's own memory for a
#: round's 875 windows 1.2 GB, the run with 3 at about 13 GB all told): the run refuses to start without them free (the
#: first formal chain lost a speaking process to the kernel's out-of-memory killer beside other jobs).
SPEAKER_HOST_GB = 2.5
PARENT_GROWTH_GB = 2.0
#: The preflight's windows: the costliest of round 1's by their time limits, each spoken twice.
PREFLIGHT_WINDOWS = 3
#: Hard events (multi-aircraft design §6.6 step 8 item 11): a training round's pick, and the select side's speaking —
#: streams beside `traffic_rounds`' 0–4.
EVENT_PICK_STREAM = 5
SELECT_EVENT_STREAM = 6
EVENT = "event"


@dataclasses.dataclass(frozen=True)
class WindowRound:
    """A round's windows (`traffic_window_generation.Drawn`: each window, its commanded aircraft's flights, limits and
    moved starts), each window's kind — ``real``, its augmentation's (`traffic_window_augment.KINDS_OF`) or `EVENT`
    (a hard event, `traffic_window_events`) — and each window's given lines: None (every aircraft spoken), or one per
    commanded aircraft in the window's order (`traffic_window.Given`; None: spoken)."""

    drawn: Drawn
    kinds: list[str]
    given: list[tuple[Given | None, ...] | None]

    def __post_init__(self) -> None:
        if not len(self.drawn.windows) == len(self.kinds) == len(self.given):
            raise ValueError("a round: a kind and a given line set (or None) per window")
        for w, lines in enumerate(self.given):
            if lines is not None and len(lines) != len(self.drawn.members[w]):
                raise ValueError(f"window {w}: {len(lines)} given lines for {len(self.drawn.members[w])} aircraft")


def spoken_round(drawn: Drawn, kinds: Sequence[str]) -> WindowRound:
    """A round whose every aircraft is spoken."""
    return WindowRound(drawn, list(kinds), [None] * len(drawn.windows))


def event_round(pools: Sequence[EventPool], places: Sequence[tuple[int, int]]) -> WindowRound:
    """The hard events at ``places`` (`traffic_window_events.pick`) as a round: each its pool's window, kind `EVENT`,
    the other aircraft given their words."""
    parts = [drawn_subset(pools[p].drawn, [pools[p].scenes[s].window]) for p, s in places]
    return WindowRound(drawn_join(parts), [EVENT] * len(places), [pools[p].scenes[s].given for p, s in places])


def loop_given(round_: WindowRound, chunk: Sequence[int], samples: int) -> list[Given | None] | None:
    """The given lines of a loop over ``chunk``'s windows spoken to ``samples`` times each (`fly_windows`' order: each
    window ``samples`` times, its commanded aircraft in order); None when none is given."""
    if all(round_.given[w] is None for w in chunk):
        return None
    return [line for w in chunk for _ in range(samples)
            for line in (round_.given[w] or (None,) * len(round_.drawn.members[w]))]


def round_join(parts: Sequence[WindowRound]) -> WindowRound:
    """Several rounds' windows as one, in order."""
    return WindowRound(drawn_join([part.drawn for part in parts]), [k for part in parts for k in part.kinds],
                       [g for part in parts for g in part.given])


def first_windows_per_airport(drawn: Drawn, per_airport: int, airports: Sequence[str]) -> list[int]:
    """The first ``per_airport`` windows of each of ``airports``, in the draw's order; refused when one has fewer."""
    taken: Counter = Counter()
    keep = []
    for w, window in enumerate(drawn.windows):
        code = window.airport.flights.code
        if taken[code] < per_airport:
            taken[code] += 1
            keep.append(w)
    short = {code: per_airport - taken[code] for code in airports if taken[code] < per_airport}
    if short:
        raise ValueError(f"{per_airport} windows an airport wanted: {short} short")
    return keep


@dataclasses.dataclass(frozen=True)
class WindowSpeaking:
    """How a window round is spoken (`traffic_rounds.Speaking`), with the vocabulary, the executor's parameters, the
    landing context, the procedure's masks and the loop batch's most aircraft-steps: a round's loop batches are its
    windows' (`traffic_window_generation.window_batches`), each spoken by `window_sentences` from its own stream, each
    row marked with its window's kind;
    ``every_landing`` the airports' landings the reward's landing direction reads; a training round's windows probed in
    their last ``probe_samples`` samples under ``probe_margin`` (multi-aircraft design §6.6 step 8 item 10; the select
    readouts never); a window's given lines (`WindowRound.given`) given in every one of its samples (`loop_given`)."""

    words: Words
    params: Any
    landings: Any
    procedures: ProcedureMasks
    budget: int
    every_landing: Any
    probe_samples: int = 0
    probe_margin: float = PROBE_MARGIN

    def plan(self, round_: WindowRound, samples: int) -> list[list[int]]:
        return window_batches(round_.drawn, samples, self.budget, self.words.spec.step_s)

    def speak(self, model: Prior, round_: WindowRound, chunk: Sequence[int], number: int, samples: int, *, seed: int,
              source: str) -> WindowSentences:
        generator = torch.Generator(device=next(model.parameters()).device).manual_seed(batch_seed(seed, number))
        got = window_sentences(model, round_.drawn, chunk, "scene", self.words, self.params, self.landings,
                               self.every_landing, samples, generator=generator, temperature=1.0,
                               procedure_masks=self.procedures,
                               probe_samples=self.probe_samples if source == "train" else 0,
                               probe_margin=self.probe_margin, given=loop_given(round_, chunk, samples))
        for row in got.rows:
            row["kind"] = round_.kinds[row["window"]]
        return got

    def fingerprint(self, round_: WindowRound) -> list[tuple[Any, ...]]:
        """Each window: its airport, opening, commanded and replayed aircraft, the flights moved in it (their keys and
        first row times), its kind, each commanded aircraft's moved start and time limit, and its given lines."""
        drawn = round_.drawn
        def lines(given: tuple[Given | None, ...] | None) -> Any:
            return None if given is None else tuple(None if g is None else (g.until, g.said.tobytes()) for g in given)
        return [(window.airport.flights.code, window.opens_s, window.commanded, window.others,
                 tuple((rows.presence.dataset_id, float(rows.presence.times_s[0])) for rows, _ in window.moved), kind,
                 tuple((None if drawn.moves[j] is None else asdict(drawn.moves[j]), drawn.limits[j]) for j in members),
                 lines(given))
                for window, kind, members, given in zip(drawn.windows, round_.kinds, drawn.members, round_.given)]

    def assemble(self, round_: WindowRound, samples: int, plan: Sequence[Sequence[int]],
                 parts: Mapping[int, WindowSentences]) -> WindowSentences:
        """The loop batches' sentences in their numbers' order."""
        return WindowSentences([r for n in range(len(plan)) for r in parts[n].rows],
                               [r for n in range(len(plan)) for r in parts[n].records],
                               [a for n in range(len(plan)) for a in parts[n].allowed])


def window_split(round_: WindowRound, spoken: WindowSentences, advantages: np.ndarray, gains: np.ndarray,
                 trained: np.ndarray, table: Split, landings: Mapping[str, Landings] | None, step_s: float
                 ) -> tuple[WindowSplit, list[str]]:
    """The window samples with an aircraft trained on (`window_advantages`), as the tuner reads them — each one's
    window, every commanded aircraft's rows as the speaker read them (`window_flight`: the trained ones' words asked over
    their counted steps, the others' over none), records, the trained ones' places, advantages, masks and probe gains —
    and each sample's window's kind."""
    drawn = round_.drawn
    samples: dict[tuple[int, int], list[int]] = defaultdict(list)
    for k, row in enumerate(spoken.rows):
        samples[(row["window"], row["sample"])].append(k)
    chosen = set(trained.tolist())
    windows, flights, records, places, weights, masks, probes, learned, kinds = [], [], [], [], [], [], [], [], []
    for (w, _), rows in samples.items():
        if not chosen & set(rows):
            continue
        window = drawn.windows[w]
        if [spoken.rows[k]["dataset_id"] for k in rows] != list(window.commanded):
            raise ValueError(f"window {w}: its rows are not its commanded aircraft in order")
        members = list(drawn.members[w])
        built = []
        for k, j in zip(rows, members):
            signals = drawn.batch.signals[j]
            counted = spoken.rows[k]["counted"] if k in chosen else 0
            built.append(window_flight(spoken.records[k], window, signals, drawn.batch.geometries[j], landings,
                                       table.airports.index(signals.airport), drawn.batch.readings[j].capture_row,
                                       counted, step_s, probed_at(spoken.rows[k], counted)))
        own = [m for m, k in enumerate(rows) if k in chosen]
        windows.append(window)
        flights.append(built)
        records.append([spoken.records[k] for k in rows])
        places.append(own)
        weights.append(advantages[[rows[m] for m in own]])
        learned.append(gains[[rows[m] for m in own]])
        masks.append([spoken.allowed[rows[m]] for m in own])
        at = [probed_at(spoken.rows[rows[m]], spoken.rows[rows[m]]["counted"]) for m in own]
        probes.append([-1 if step is None else step for step in at])
        kinds.append(round_.kinds[w])
    return WindowSplit(table, windows, flights, records, places, weights, masks, probes, learned), kinds


def probed_at(row: Mapping[str, Any], counted: int) -> int | None:
    """The own step a probe said a row's go-around at, when inside its counted steps (None: none there)."""
    return row["forced"] if row["forced"] is not None and row["forced"] < counted else None


def preflight(model: Prior, base: Prior, round_: WindowRound, speaking: WindowSpeaking, table: Split,
              config: RewardConfig, device: torch.device, step_s: float, log: Any) -> dict[str, Any]:
    """Before anything is spoken (the formal run's size, not a smoke's): round 1's `PREFLIGHT_WINDOWS` costliest windows
    (by their time limits with a go-around's time) spoken twice in this process — with probes, every cleared aircraft
    made to go around at once (a go-around's sentence is the longest to score) — and the costliest sample scored as the pass scores it — every
    aircraft trained, in blocks of steps, the layers recomputed in the backward — beside the speaking processes on the
    GPU: its peak. Refused (the run does not start) when it does not fit. "Costliest" is `part_cost`'s, the whole
    encoding's (pairs weigh most); in blocks the time attention's share grows, so the pick is the likely largest, and
    its peak (1.5 GB for round 1's largest) far under the 8 GB."""
    drawn = round_.drawn

    def bound(w: int) -> float:
        rows = [rows_for(drawn.limits[j] + GO_AROUND_EXTRA_S + step_s, step_s) for j in drawn.members[w]]
        places = window_places([drawn.windows[w]], step_s)
        aircraft = len(drawn.windows[w].commanded) + len(drawn.windows[w].others)
        return part_cost(1, aircraft, int((places.start + np.array(rows)).max()))

    costliest = sorted(range(len(drawn.windows)), key=bound)[-PREFLIGHT_WINDOWS:]
    part = WindowRound(drawn_subset(drawn, costliest), [round_.kinds[w] for w in costliest],
                       [round_.given[w] for w in costliest])
    # with probes, every cleared aircraft goes around at once: a go-around's sentences are the longest to score
    probed = 2 if speaking.probe_samples else 0
    spoken = window_sentences(model, part.drawn, list(range(len(costliest))), "scene", speaking.words, speaking.params,
                              speaking.landings, speaking.every_landing, 2,
                              generator=torch.Generator(device=device).manual_seed(0), temperature=1.0,
                              procedure_masks=speaking.procedures, probe_samples=probed, probe_margin=math.inf,
                              given=loop_given(part, range(len(costliest)), 2))
    # every aircraft trained (a given one too: the most) and every probe's go-around word learned: the memory is
    # measured, not the values
    rows = spoken.rows
    split, _ = window_split(part, spoken, np.ones(len(rows)), np.array([float(r["forced"] is not None) for r in rows]),
                            np.arange(len(rows)), table, speaking.landings, step_s)
    costs = [scoring_cost(w, f, step_s) for w, f in zip(split.windows, split.flights)]
    s = int(np.argmax(costs))
    one = WindowSplit(split.table, [split.windows[s]], [split.flights[s]], [split.records[s]], [split.trained[s]],
                      [split.advantages[s]], [split.allowed[s]], [split.forced[s]], [split.gains[s]])
    tuner = WindowRewardTuner(copy.deepcopy(model), base, config, device, seed=0, traffic_learning_rate=0.0,
                              step_s=step_s)
    frozen = copy.deepcopy(model).eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    try:
        _, logits, _, _, _, _, _ = tuner._window_scored(one, [0], frozen)
        sum(x.float().logsumexp(-1).sum() for x in logits).backward()
    except torch.OutOfMemoryError as error:
        raise SystemExit(f"preflight: the costliest window sample (cost {costs[s]:.0f}) does not fit the GPU beside "
                         f"the speaking processes — smaller blocks (traffic_window_tuner.PAIRS_PER_BLOCK) or fewer "
                         f"--speakers: {error}") from None
    peak = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
    del tuner, frozen, logits
    if device.type == "cuda":
        torch.cuda.empty_cache()
    out = {"windows": costliest, "scored_cost": costs[s], "scored_aircraft": len(split.windows[s].commanded),
           "pairs_per_block": PAIRS_PER_BLOCK, "gpu_peak_gb": peak}
    log(f"preflight: round 1's costliest window sample (cost {costs[s]:.0f}, {len(split.windows[s].commanded)} "
        f"commanded) scored with gradients, GPU peak {peak} GB")
    return out


def side_of(kind: str) -> str:
    """A window kind's side: ``real``, ``events`` (`EVENT`) or ``augmented`` (an augmentation's kind)."""
    return "real" if kind == "real" else "events" if kind == EVENT else "augmented"


def split_of_kinds(split: WindowSplit, kinds: Sequence[str], side: str) -> WindowSplit:
    """``split``'s window samples whose kind (``kinds``, per sample) is of ``side`` (`side_of`)."""
    keep = [s for s, kind in enumerate(kinds) if side_of(kind) == side]
    return WindowSplit(split.table, [split.windows[s] for s in keep], [split.flights[s] for s in keep],
                       [split.records[s] for s in keep], [split.trained[s] for s in keep],
                       [split.advantages[s] for s in keep], [split.allowed[s] for s in keep],
                       [split.forced[s] for s in keep], [split.gains[s] for s in keep])


def landing_gap_s(window: Any, runway: str, landing_s: float, commanded: Sequence[tuple[float, str]]) -> float | None:
    """The time since the last landing before ``landing_s`` on ``runway`` or a runway separated as one with it: the
    window's replayed aircraft at their records, and ``commanded`` — the other commanded aircraft's landings (time,
    runway) as they were in the same sample, or recorded."""
    separation = window.airport.flights.separation
    before = [window.track(k).presence.landing_s for k in window.others
              if separation.one_runway(window.track(k).presence.runway, runway)
              and window.track(k).presence.landing_s < landing_s]
    before += [t for t, r in commanded if separation.one_runway(r, runway) and t < landing_s]
    return float(landing_s - max(before)) if before else None


def ordering(rows: Sequence[Mapping[str, Any]], round_: WindowRound, step_s: float) -> dict[str, Any]:
    """On the real windows (design §4.3 per commanded aircraft): the sentences that landed with no loss ending them first
    and not starting in one — their median time from the first predicted step to the landing and median gap to the
    landing before on the runway (`landing_gap_s`) — each against the same aircraft's record. The landings before are
    the replayed aircraft's and those of the other commanded aircraft that landed in the same sample (``rows``: every
    row of it), at their times there and — the recorded side — the same aircraft's recorded ones: the two sides hold the
    same leaders, so another aircraft's failure to land moves neither."""
    drawn = round_.drawn
    by_sample: dict[tuple[int, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[(row["window"], row["sample"])].append(row)
    times, recorded_times, gaps, recorded_gaps = [], [], [], []
    for (w, _), members in by_sample.items():
        window = drawn.windows[w]
        geometry = window.airport.flights.geometry
        landed = {r["dataset_id"]: (r["landing_s"], geometry.candidates[r["runway"]].ident) for r in members
                  if r["landing_s"] is not None}
        recorded = {k: (window.rows(k).presence.landing_s, window.rows(k).presence.runway) for k in landed}
        for row in members:
            if row["outcome"] != LANDED or row["starts_in_a_loss"]:
                continue
            key = row["dataset_id"]
            first_s = window.first_step_s(key) + N_LOOK * step_s
            times.append(row["landing_s"] - first_s)
            recorded_times.append(recorded[key][0] - first_s)
            gap = landing_gap_s(window, landed[key][1], row["landing_s"],
                                [v for k, v in landed.items() if k != key])
            then = landing_gap_s(window, recorded[key][1], recorded[key][0],
                                 [v for k, v in recorded.items() if k != key])
            if gap is not None and then is not None:
                gaps.append(gap)
                recorded_gaps.append(then)

    def median(values: Sequence[float]) -> float | None:
        return float(np.median(values)) if values else None

    out = {"time_to_land_s": median(times), "recorded_time_to_land_s": median(recorded_times),
           "gap_s": median(gaps), "recorded_gap_s": median(recorded_gaps), "landed_with_a_gap": len(gaps)}
    out["time_ratio"] = out["time_to_land_s"] / out["recorded_time_to_land_s"] if times else None
    out["gap_ratio"] = out["gap_s"] / out["recorded_gap_s"] if gaps else None
    return out


def side_readout(spoken: WindowSentences, round_: WindowRound, step_s: float, *, real: bool) -> dict[str, Any]:
    """One side of the select readout (module docstring), in the shape `traffic_rounds.guarded_choice` reads, R32's
    counts: over every aircraft sentence (`prior_free_generation.grouped`'s) the executor's landed share (separation
    aside), landed on the observed runway and the words said after the first step per sentence (to its own end — a
    window aircraft ended by the judge says nothing after: its count stops there); over those not starting in a loss they
    answer for (`traffic_free_generation.summary`'s) lost separation and the reward (and per kind); every sentence's
    row."""
    rows = [row for row in spoken.rows if not row["starts_in_a_loss"]]
    landed = [row for row in spoken.rows if row["own"] == LANDED]
    said = np.array([(record.grid[1: row["said_steps"]] != UNCHANGED).sum(axis=0)
                     for row, record in zip(spoken.rows, spoken.records)])
    out: dict[str, Any] = {
        "free_generation": {"all": {
            "sentences": len(spoken.rows),
            "outcomes": {"landed": len(landed) / len(spoken.rows)},
            "landed_on_observed_runway": (sum(r["runway"] == r["observed_runway"] for r in landed) / len(landed)
                                          if landed else None),
            "words_after_first_per_flight": {name: float(said[:, c].mean()) for c, name in enumerate(COLUMNS)}}},
        "separation": {"lost_separation": sum(r["outcome"] == LOST_SEPARATION for r in rows) / len(rows),
                       "left_out_starting_in_a_loss": len(spoken.rows) - len(rows)},
        "reward": float(np.mean([r["reward"] for r in rows])),
        "reward_by_kind": {kind: float(np.mean([r["reward"] for r in rows if r["kind"] == kind]))
                           for kind in sorted({r["kind"] for r in rows})}}
    if real:
        out["ordering"] = ordering(spoken.rows, round_, step_s)
    out["aircraft"] = spoken.rows
    return out


def event_readout(spoken: WindowSentences) -> dict[str, Any]:
    """The select side of hard events (multi-aircraft design §6.6 step 8 item 11): over the answered aircraft's sentences
    (the others' words given: ``given``) — the reward, landed in the landing direction, lost separation, said a
    go-around, and the outcomes; every one of their rows."""
    rows = [row for row in spoken.rows if not row["given"]]
    return {"sentences": len(rows), "reward": float(np.mean([r["reward"] for r in rows])),
            "landed_here": float(np.mean([r["landed_here"] for r in rows])),
            "lost_separation": float(np.mean([r["outcome"] == LOST_SEPARATION for r in rows])),
            "said_a_go_around": float(np.mean([r["go_around"] is not None for r in rows])),
            "outcomes": {k: v / len(rows) for k, v in sorted(Counter(r["outcome"] for r in rows).items())},
            "aircraft": rows}


def select_counted(rows: Sequence[Mapping[str, Any]], value: Callable[[Mapping[str, Any]], float]
                   ) -> dict[tuple[int, str, int], float]:
    """``value`` of each select sentence the reward counts (not starting in a loss it answers for), keyed by window,
    aircraft and sample: every round speaks the select windows with the same streams, so a key is the same draw in
    every round — the round choice's pairs (`select_rewards`) and its readout's (R39)."""
    return {(r["window"], r["dataset_id"], r["sample"]): value(r) for r in rows if not r["starts_in_a_loss"]}


def select_rewards(out: Path, rounds: int) -> list[dict[tuple[int, str, int], float]]:
    """Each round's augmented select sentences' rewards (0 … ``rounds``), those the reward counts (`select_counted`)."""
    rewards = []
    for number in range(rounds + 1):
        readout = json.loads((out / f"round_{number:02d}" / "readout.json").read_text(encoding="utf-8"))
        rewards.append(select_counted(readout["augmented"]["aircraft"], lambda r: r["reward"]))
    return rewards


def write_choice(out: Path, history: Sequence[Mapping[str, Any]], labelled: Mapping[str, Mapping[str, float]],
                 prior_dir: Path, log: Any) -> None:
    """``choice.json`` over the rounds finished (`traffic_rounds.guarded_choice`; round 0 kept: the start)."""
    rewards = select_rewards(out, len(history) - 1)
    kept_round, excluded = guarded_choice(history, labelled, rewards)
    write_json_atomic(out / "choice.json", {
        "rule": f"within round 0's guards (real windows: landed ≥ − {GUARD_LANDED_DROP}, on the observed runway ≥ − "
                f"{GUARD_RUNWAY_DROP}, lost separation ≤ + {GUARD_LOSS_RISE}, median time to land and landing gap ≤ "
                f"{GUARD_ORDER_GROWTH} × recorded; real and augmented: each of {', '.join(GUARD_WORD_COLUMNS)} words per "
                f"sentence off the labelled ≤ round 0's + ln {GUARD_WORD_GROWTH}), the augmented-window reward: beating "
                f"round 0 by ≥ {TIE_STANDARD_ERRORS} paired standard errors, of those the highest and the earliest it "
                f"does not beat by as much; none: round 0",
        "against_round_0": [dict(zip(("difference", "standard_error"), paired_difference(rewards[0], r)))
                            for r in rewards],
        "augmented_reward": [row["augmented"]["reward"] for row in history],
        "real_reward": [row["real"]["reward"] for row in history], "excluded_by_the_guards": excluded,
        "round": kept_round,
        "directory": str(out / f"round_{kept_round:02d}") if kept_round else str(prior_dir)})
    log(f"kept round {kept_round} (the guards excluded {excluded}) → {out}")


def write_sentences(path: Path, spoken: WindowSentences, advantages: np.ndarray, gains: np.ndarray) -> None:
    """A round's words kept (R32's ``sentences.npz``): per aircraft sentence its window, key, sample, the words it said to
    its own end (``said`` over ``step_offsets``), its outcome, runway, reward and advantage, whether a probe watched it
    and whether its words were given (a hard event's other aircraft),
    the own step a probe said a go-around for it at (``forced``, −1: none) and its probe gain (`window_advantages`)."""
    said = [record.grid[: row["said_steps"]] for row, record in zip(spoken.rows, spoken.records)]
    np.savez_compressed(path, window=np.array([r["window"] for r in spoken.rows]),
                        dataset_id=np.array([r["dataset_id"] for r in spoken.rows]),
                        sample=np.array([r["sample"] for r in spoken.rows]),
                        step_offsets=np.concatenate(([0], np.cumsum([len(g) for g in said]))),
                        said=np.concatenate(said).astype(np.int16),
                        outcome=np.array([r["outcome"] for r in spoken.rows]),
                        runway=np.array([r["runway"] for r in spoken.rows]),
                        reward=np.array([r["reward"] for r in spoken.rows]), advantage=advantages,
                        probed=np.array([r["probed"] for r in spoken.rows]),
                        given=np.array([r["given"] for r in spoken.rows]),
                        forced=np.array([-1 if r["forced"] is None else r["forced"] for r in spoken.rows]),
                        probe_gain=gains)


def history_row(round_number: int, readout: Mapping[str, Any], **more: Any) -> dict[str, Any]:
    """A round's ``history.json`` row: its select readout without the aircraft sentences (those stay in the round's
    ``readout.json``), and ``more`` (the training round's sentences and pass)."""
    return {"round": round_number,
            **{side: {k: v for k, v in readout[side].items() if k != "aircraft"}
               for side in ("real", "augmented", "events") if side in readout},
            "traffic": readout["traffic"], **more}


def round_summary(round_: WindowRound, spoken: WindowSentences, trained: np.ndarray,
                  gains: np.ndarray) -> dict[str, Any]:
    """How a round's sentences did: the reward, lost separation and the executor's landed share, in all, per kind and
    for the unprobed and the probed samples apart; the aircraft sentences trained on, those starting in a loss; and the
    go-arounds (multi-aircraft design §6.6 step 8): said by the model, said by a probe, and a probe's learned (its
    sentence trained on, its probe gain above 0, `window_advantages`). Over the model's sentences: a hard event's other
    aircraft, given their words, are counted apart (``given``)."""
    taught = set(trained.tolist())
    rows = [r for r in spoken.rows if not r["given"]]

    def shares(members: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        return {"sentences": len(members), "reward": float(np.mean([r["reward"] for r in members])),
                "lost_separation": float(np.mean([r["outcome"] == LOST_SEPARATION for r in members])),
                "landed": float(np.mean([r["own"] == LANDED for r in members]))}

    return {"all": shares(rows),
            "by_kind": {kind: shares([r for r in rows if r["kind"] == kind]) for kind in sorted({r["kind"] for r in rows})},
            "windows": len(round_.drawn.windows), "kinds": dict(Counter(round_.kinds)),
            "aircraft_sentences": len(rows), "given": len(spoken.rows) - len(rows), "trained_on": int(len(trained)),
            "starting_in_a_loss": sum(r["starts_in_a_loss"] for r in rows),
            "unprobed": shares([r for r in rows if not r["probed"]]),
            "probed": shares([r for r in rows if r["probed"]]) if any(r["probed"] for r in rows) else None,
            "go_arounds": {"said_by_the_model": sum(r["go_around"] is not None and r["forced"] is None for r in rows),
                           "said_by_a_probe": sum(r["forced"] is not None for r in rows),
                           "probes_learned": sum(bool(r["forced"] is not None and k in taught and gains[k] > 0.0)
                                                 for k, r in enumerate(spoken.rows))}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--prior", type=Path, required=True, help="the start: a single-aircraft prior (augmented) or a "
                        "traffic prior (an M4 round)")
    parser.add_argument("--base", type=Path, required=True, help="the pull's reference: base")
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--rounds", type=int, default=8, help="the last round to run (0: round 0 alone)")
    parser.add_argument("--resume", action="store_true", help="continue the run at --out from its last finished round")
    parser.add_argument("--commanded", choices=COMMANDED, default="every", help="every aircraft of a window commanded, "
                        "or one a window (`traffic_window.draw_windows`; module docstring, item 1)")
    parser.add_argument("--real-per-airport", type=int, default=70, help="real windows an airport a round")
    parser.add_argument("--augmented-per-airport", type=int, default=70, help="augmented windows an airport a round")
    parser.add_argument("--samples", type=int, default=8, help="sentences a window")
    parser.add_argument("--select-per-airport", type=int, default=70)
    parser.add_argument("--select-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--speakers", type=int, default=4, help="speaking processes (not part of the run: the "
                        "sentences do not depend on it)")
    parser.add_argument("--aircraft-steps", type=int, default=SPEAKER_AIRCRAFT_STEPS, help="a loop batch's most")
    parser.add_argument("--traffic-learning-rate", type=float, default=TRAFFIC_LEARNING_RATE)
    parser.add_argument("--passes", type=int, default=1, help="sweeps over a round's sentences")
    parser.add_argument("--probe-samples", type=int, default=0, help="a training round's samples a window probed for "
                        "a go-around (multi-aircraft design §6.6 step 8 item 10; 0: none)")
    parser.add_argument("--probe-margin", type=float, default=PROBE_MARGIN, help="a probe's go-around: the first step "
                        "a cleared aircraft's tightest margin was under this")
    parser.add_argument("--imitation-weight", type=float, default=IMITATION_WEIGHT, help="a probe's word learned: "
                        "this × its probe gain (its reward less its aircraft's unprobed samples' mean) × its "
                        "cross-entropy, where the gain is above 0")
    parser.add_argument("--events", type=Path, nargs="*", default=[], help="window rewind runs (R43) on the training "
                        "days whose hard events a round adds (multi-aircraft design §6.6 step 8 item 11)")
    parser.add_argument("--events-per-airport", type=int, default=0, help="hard events an airport a round (all an "
                        "airport has where fewer)")
    parser.add_argument("--select-events", type=Path, nargs="*", default=[], help="window rewind runs (R43) on the "
                        "select days whose hard events every select readout reads")
    parser.add_argument("--device", default="cuda", help="the prior's; the executor flies on CPU")
    parser.add_argument("--smoke", action="store_true", help="a dirty tree allowed; the runs are marked smoke")
    for field, default in asdict(RewardConfig()).items():
        parser.add_argument(f"--{field.replace('_', '-')}", type=type(default), default=default)
    args = parser.parse_args(argv)

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    prior_dir, base_dir, instructions, executor_dir, out = map(
        resolved, (args.prior, args.base, args.instructions, args.executor, args.out))
    event_runs, select_event_runs = [resolved(r) for r in args.events], [resolved(r) for r in args.select_events]
    if args.resume and not (out / "config.json").exists():
        parser.error(f"--resume: {out} holds no run")
    if not args.resume and out.exists():
        parser.error(f"{out} exists; a fine-tuning run is never overwritten (--resume continues it)")
    if args.rounds < 0:
        parser.error("--rounds is the last round to run, 0 or more")
    if args.samples < 2:
        parser.error("an aircraft's sentences are compared with each other: --samples ≥ 2")
    if args.probe_samples and not 2 <= args.probe_samples <= args.samples - 2:
        parser.error(f"--probe-samples {args.probe_samples}: probes and the unprobed samples are each compared among "
                     f"themselves (`window_advantages`) — at least 2 of each of a window's {args.samples} samples")
    if not args.probe_margin > 0.0 or not args.imitation_weight >= 0.0:
        parser.error("--probe-margin above 0, --imitation-weight at least 0")
    if args.passes < 1:
        parser.error("--passes is the sweeps over a round's sentences, 1 or more")
    if args.data_weight <= 0.0:
        parser.error("the post-training trains beside the data (design §6.2): --data-weight > 0")
    if bool(args.events) != (args.events_per_airport > 0):
        parser.error("--events and --events-per-airport > 0 go together")
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("the tree has uncommitted changes; a fine-tuning run is made at a commit")
    config = RewardConfig(**{f: getattr(args, f) for f in asdict(RewardConfig())})
    started = time.perf_counter()
    device = torch.device(args.device)
    params, record, words = replay.open_executor(executor_dir, instructions)
    spec = load_spec(instructions)
    step_s = spec.step_s
    start = load_prior(prior_dir, instructions)
    loaded, start_payload, start_config, start_masks = start
    base, base_payload, base_config, _ = load_prior(base_dir, instructions)
    if start_config["smoke"] or start_payload["schema"] not in (PRIOR_CHECKPOINT_SCHEMA, TRAFFIC_CHECKPOINT_SCHEMA):
        parser.error(f"{prior_dir} is not a single-aircraft or a traffic prior's formal run")
    if base_config["smoke"] or base_payload["schema"] != PRIOR_CHECKPOINT_SCHEMA:
        parser.error(f"{base_dir} is not a single-aircraft prior's formal run")
    if base.config.to_dict() != loaded.config.to_dict():
        parser.error("the base model and the start are not one architecture")
    if start_payload["schema"] == PRIOR_CHECKPOINT_SCHEMA:
        torch.manual_seed(args.seed)                    # the traffic attention's weights: at zero it reads nothing
        model = with_traffic(loaded, EDGE_FEATURES)     # on the CPU until the speaking processes are forked
    elif tuple(loaded.traffic_features) != EDGE_FEATURES:
        parser.error(f"{prior_dir}'s traffic attention reads other edge features than today's")
    else:
        model = loaded
    variant, airports, slots = model.config.variant, model.config.airports, model.config.candidate_slots
    every_landing = airport_landings(instructions, rosters(instructions))
    landings = every_landing if VARIANTS[variant].landing_context else None
    geometries = load_candidates(instructions)
    table = Split([], airports, candidate_table(geometries, airports, slots), *runway_names(geometries, airports),
                  column_classes(words, slots), variant)
    windows_alt = start_altitude_windows(instructions)
    most = busiest(instructions, spec, airports, step_s)
    max_rows = model.config.max_rows
    # the training days built once: the loop's windows and the data term's samples read each flight's same rows
    train_flights, train_counts = airport_flights(instructions, "train", spec, airports, landings, max_rows)
    train_airports = with_tracks(instructions, "train", spec, train_flights)
    data, data_counts = split_samples(train_flights, train_counts, step_s)
    data_counts["samples_asking_nothing"] = sum(not b.sample.asks for b in data)
    data = [b for b in data if b.sample.asks]
    select_flights, select_counts = airport_flights(instructions, "select", spec, airports, landings, max_rows)
    select_airports = with_tracks(instructions, "select", spec, select_flights)
    select_built = [b for b in split_samples(select_flights, select_counts, step_s)[0] if b.sample.asks]
    select_draw = draw_windows(instructions, "select", spec, words, select_airports,
                               per_airport=args.select_per_airport, seed=args.seed, step_s=step_s,
                               commanded=args.commanded)
    select_as_drawn = drawn_windows(select_draw, select_airports, params, step_s)
    select_real = spoken_round(select_as_drawn, ["real"] * len(select_as_drawn.windows))
    select_augmented_drawn, select_augmenting = augmented_windows(select_as_drawn, params, most, max_rows,
                                                                  args.seed + SELECT_AUGMENT_OFFSET, windows_alt, spec,
                                                                  KINDS_OF[args.commanded], args.commanded)
    select_augmented = spoken_round(select_augmented_drawn, [a["kind"] for a in select_augmented_drawn.augmented])
    # hard events (multi-aircraft design §6.6 step 8 item 11): the training days' to train on, the select days' read
    pooled = dict(instructions=instructions, spec=spec, words=words, params=params, executor_sha256=record["sha256"],
                  most=most, max_rows=max_rows, windows_alt=windows_alt)
    try:
        event_pools = [event_pool(run, "train", airports=train_airports, **pooled) for run in event_runs]
        select_pools = [event_pool(run, "select", airports=select_airports, **pooled) for run in select_event_runs]
    except ValueError as refusal:
        parser.error(str(refusal))
    select_events = (event_round(select_pools, [(p, s) for p, pool in enumerate(select_pools)
                                                for s in range(len(pool.scenes))]) if select_pools else None)
    labelled = {"real": labelled_words(select_real.drawn.batch),
                "augmented": labelled_words(select_augmented.drawn.batch)}
    recorded_rows = fixed_rows(select_real.drawn, list(range(len(select_real.drawn.windows))), "recorded", words,
                               params, every_landing, start_masks)
    recorded_counted = [r for r in recorded_rows if not r["starts_in_a_loss"]]         # (the model's shares' basis)
    recorded_lost = sum(r["outcome"] == LOST_SEPARATION for r in recorded_counted) / len(recorded_counted)
    record = {
        "schema": SCHEMA, "written_utc": utc_now(), "git": git, "smoke": args.smoke, "commanded": args.commanded,
        "prior": {"directory": str(prior_dir), "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt"),
                  "schema": start_payload["schema"], "procedure_masks": list(start_masks.names)},
        "base": {"directory": str(base_dir), "checkpoint_sha256": file_sha256(base_dir / "checkpoint.pt")},
        "executor": {"directory": str(executor_dir), "sha256": record["sha256"]}, "instructions": str(instructions),
        "edge_features": list(EDGE_FEATURES), "edge_source_sha256": edge_source_sha256(),
        "optimiser": asdict(config), "traffic_learning_rate": args.traffic_learning_rate, "rounds": args.rounds,
        "real_per_airport": args.real_per_airport, "augmented_per_airport": args.augmented_per_airport,
        "samples": args.samples, "pool_factor": POOL_FACTOR, "aircraft_steps": args.aircraft_steps,
        "probes": {"samples": args.probe_samples, "margin": args.probe_margin, "imitation_weight": args.imitation_weight},
        "events": {"runs": [pool.record for pool in event_pools], "per_airport": args.events_per_airport,
                   "select_runs": [pool.record for pool in select_pools]},
        "speakers": args.speakers, "busiest_training_step": most,
        "data": {"split": "train", "scene_samples": len(data), "built": data_counts},
        "select": {"per_airport": args.select_per_airport, "samples": args.select_samples, "drawn": select_draw.counts,
                   "augment_seed": args.seed + SELECT_AUGMENT_OFFSET, "augmenting": select_augmenting,
                   "scene_samples": len(select_built), "labelled_words_after_first_per_flight": labelled,
                   "recorded_lost_separation": recorded_lost},
        "seed": args.seed, "tie_standard_errors": TIE_STANDARD_ERRORS, "passes": args.passes,
        "guards": {"landed_drop": GUARD_LANDED_DROP, "runway_drop": GUARD_RUNWAY_DROP, "loss_rise": GUARD_LOSS_RISE,
                   "order_growth": GUARD_ORDER_GROWTH, "word_columns": list(GUARD_WORD_COLUMNS),
                   "word_margin_ln": math.log(GUARD_WORD_GROWTH)},
        "streams": {"windows": "seed + round", "augmentations": [AUGMENT_STREAM], "sampling": [SAMPLING_STREAM],
                    "pass": [PASS_STREAM], "select": {**SELECT_STREAMS, "events": SELECT_EVENT_STREAM},
                    "event_pick": [EVENT_PICK_STREAM], "loop_batches": "[stream seed, batch number]"},
        "device": str(device), "n_look": N_LOOK, "resumed": []}
    if args.resume:
        stored = json.loads((out / "config.json").read_text(encoding="utf-8"))
        differences = run_differences(stored, record)
        if differences:
            raise SystemExit(f"{out} was run with another configuration or code: {differences} differ")
        last = completed_rounds(out)
        if args.rounds < last:
            parser.error(f"{out} has finished round {last}; --rounds {args.rounds} is behind it")
        history = [row for row in json.loads((out / "history.json").read_text(encoding="utf-8"))["rounds"]
                   if row["round"] <= last]
        if [row["round"] for row in history] != list(range(last + 1)):
            raise SystemExit(f"{out}/history.json does not hold rounds 0 … {last}")
        failing = ordering_failures(history[0])
        if failing:
            raise SystemExit(f"the start failed the ordering guards {failing} (round 0): no round could be chosen")
        record = {**stored, "rounds": args.rounds,
                  "resumed": [*stored["resumed"], {"utc": utc_now(), "from_round": last, "to": args.rounds,
                                                   "speakers": args.speakers}]}
    else:
        out.mkdir(parents=True)
        last, history = -1, []
        write_json_atomic(out / "config.json", record)

    def log(line: str) -> None:
        print(f"{line}  [{time.perf_counter() - started:.0f}s]", flush=True)

    log(f"{f'resumed after round {last}; ' if args.resume else ''}train days: {len(data)} scene samples; select: "
        f"{len(select_real.drawn.windows)} windows ({len(select_real.drawn.batch.signals)} aircraft), "
        f"{len(select_augmented.drawn.windows)} augmented ({dict(Counter(select_augmented.kinds))}); recorded lost "
        f"separation {recorded_lost:.3f}; hard events {[pool.record['hard'] for pool in event_pools]} to train on, "
        f"{[pool.record['hard'] for pool in select_pools]} read")

    def free_gpu() -> None:
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def read_select(round_number: int) -> dict[str, Any]:
        model.eval()
        readout: dict[str, Any] = {}
        for side, round_ in (("real", select_real), ("augmented", select_augmented)):
            free_gpu()
            spoken, peaks = speakers.speak("select", side, round_, model, args.select_samples,
                                           seed=round_seed(args.seed, 0, SELECT_STREAMS[side]), source="scene")
            readout[side] = side_readout(spoken, round_, step_s, real=side == "real")
            readout[side]["speaking_gpu_peak_gb"] = peaks
        if select_events is not None:
            free_gpu()
            spoken, peaks = speakers.speak("select", "events", select_events, model, args.select_samples,
                                           seed=round_seed(args.seed, 0, SELECT_EVENT_STREAM), source="scene")
            readout["events"] = {**event_readout(spoken), "speaking_gpu_peak_gb": peaks}
        readout["traffic"] = traffic_readout(model, select_built, slots, device)
        real, augmented = readout["real"], readout["augmented"]
        log(f"round {round_number}: select reward real {real['reward']:.3f} augmented {augmented['reward']:.3f} ("
            + " ".join(f"{k} {v:.3f}" for k, v in augmented["reward_by_kind"].items())
            + f"); lost separation real {real['separation']['lost_separation']:.3f} augmented "
              f"{augmented['separation']['lost_separation']:.3f}; "
            + (f"hard events reward {readout['events']['reward']:.3f} landed {readout['events']['landed_here']:.3f} "
               f"go-around {readout['events']['said_a_go_around']:.3f}; " if "events" in readout else "")
            + f"landed real "
              f"{real['free_generation']['all']['outcomes']['landed']:.3f}; ordering time "
              f"{real['ordering']['time_ratio']} gap {real['ordering']['gap_ratio']}; TF NLL "
              f"{readout['traffic']['teacher_forced']['nll_per_step']:.4f}")
        return readout

    if last > 0:
        resumed = load_prior(out / f"round_{last:02d}", instructions)
        grown_from = {"directory": str(prior_dir), "checkpoint_sha256": file_sha256(prior_dir / "checkpoint.pt")}
        if resumed.payload["start"] != grown_from:
            raise SystemExit(f"{out}/round_{last:02d} grew from {resumed.payload['start']}, not {grown_from}")
        model = resumed.model

    def train_round(round_number: int) -> tuple[WindowRound, dict[str, Any]]:
        """Round ``round_number``'s windows (module docstring, item 1), the same in every process: the round (with its
        hard events, picked by the round's stream) and what its draw and augmentation counted and which events it
        picked."""
        per_airport = args.real_per_airport + math.ceil(args.augmented_per_airport * POOL_FACTOR)
        drawn = draw_windows(instructions, "train", spec, words, train_airports, per_airport=per_airport,
                             seed=args.seed + round_number, step_s=step_s, commanded=args.commanded)
        pool = drawn_windows(drawn, train_airports, params, step_s)
        real = first_windows_per_airport(pool, args.real_per_airport, airports)
        rest = [w for w in range(len(pool.windows)) if w not in set(real)]
        candidates, augmenting = augmented_windows(drawn_subset(pool, rest), params, most, max_rows,
                                                   [args.seed, round_number, AUGMENT_STREAM], windows_alt, spec,
                                                   KINDS_OF[args.commanded], args.commanded)
        augmented = drawn_subset(candidates, first_windows_per_airport(candidates, args.augmented_per_airport,
                                                                       airports))
        parts = [spoken_round(drawn_join([drawn_subset(pool, real), augmented]),
                              ["real"] * len(real) + [a["kind"] for a in augmented.augmented])]
        picked = []
        if event_pools:
            picked = pick(event_pools, args.events_per_airport,
                          np.random.default_rng([args.seed, round_number, EVENT_PICK_STREAM]))
            parts.append(event_round(event_pools, picked))
        return round_join(parts), {"drawn": drawn.counts, "augmenting": augmenting,
                                   "events": [[p, event_pools[p].scenes[s].event] for p, s in picked]}

    if args.resume and args.rounds == last:                 # the choice alone: nothing to speak
        write_json_atomic(out / "config.json", record)
        write_choice(out, history, labelled, prior_dir, log)
        return 0
    free_gb = psutil.virtual_memory().available / 1e9
    needed_gb = args.speakers * SPEAKER_HOST_GB + PARENT_GROWTH_GB
    if free_gb < needed_gb:
        raise SystemExit(f"{free_gb:.1f} GB of the host's memory free, {needed_gb:.1f} GB needed for {args.speakers} "
                         f"speaking processes and a round (SPEAKER_HOST_GB, PARENT_GROWTH_GB): fewer --speakers, or "
                         f"wait for the other jobs")
    gc.collect()
    gc.freeze()
    speaking = WindowSpeaking(words, params, landings, start_masks, args.aircraft_steps, every_landing,
                              args.probe_samples, args.probe_margin)
    speakers = Speakers(args.speakers, lambda n: train_round(n)[0],
                        {"real": select_real, "augmented": select_augmented,
                         **({"events": select_events} if select_events is not None else {})},
                        model, args.device, speaking)
    model.to(device)
    base.to(device)
    if last < 0:                                        # the first invocation: the formal size checked before round 0
        checked = preflight(model, base, train_round(1)[0], speaking, table, config, device, step_s, log)
        write_json_atomic(out / "preflight.json", {**checked, "host_free_gb": free_gb, "written_utc": utc_now()})
    if last < 0:
        directory = out / "round_00"
        directory.mkdir()
        readout = read_select(0)
        history = [history_row(0, readout)]
        write_json_atomic(out / "history.json", {"rounds": history})
        write_json_atomic(directory / "readout.json", readout)                       # last: the round finished
        failing = ordering_failures(history[0])
        if failing:
            raise SystemExit(f"the start fails the ordering guards {failing} against the record "
                             f"({history[0]['real']['ordering']}): no round could be chosen — not training")
        last = 0
    tuner = WindowRewardTuner(model, base, config, device, seed=args.seed,
                              traffic_learning_rate=args.traffic_learning_rate, step_s=step_s,
                              imitation_weight=args.imitation_weight)
    if last > 0:
        tuner.load_state(torch.load(out / f"round_{last:02d}" / OPTIMISER_FILE, map_location="cpu", weights_only=True))
    if args.resume:
        write_json_atomic(out / "config.json", record)
    for round_number in range(last + 1, args.rounds + 1):
        directory = out / f"round_{round_number:02d}"
        directory.mkdir()
        model.eval()
        free_gpu()
        speakers.send("train", round_number, model, args.samples,
                      seed=round_seed(args.seed, round_number, SAMPLING_STREAM), source="train")
        round_, drawn_counts = train_round(round_number)          # while the processes speak it
        spoken, speaking_peaks = speakers.receive(round_, args.samples)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        advantages, gains, trained = window_advantages(spoken.rows, args.samples)
        described = {**round_summary(round_, spoken, trained, gains), **drawn_counts}
        write_sentences(directory / "sentences.npz", spoken, advantages, gains)
        write_json_atomic(directory / "sentences.json", {
            **described, "aircraft": [{k: r[k] for k in ("window", "dataset_id", "sample", "kind", "role", "reward",
                                                         "outcome", "own", "counted", "starts_in_a_loss", "probed",
                                                         "forced", "given", "go_around")}
                                      | {"advantage": float(a), "probe_gain": float(g)}
                                      for r, a, g in zip(spoken.rows, advantages, gains)]})
        log(f"round {round_number}: {len(spoken.rows)} aircraft sentences in {len(round_.drawn.windows)} windows, "
            f"reward {described['all']['reward']:.3f} ("
            + " ".join(f"{k} {v['reward']:.3f}" for k, v in described["by_kind"].items())
            + f"), lost separation {described['all']['lost_separation']:.3f}, {len(trained)} trained on, "
              f"{described['starting_in_a_loss']} starting in a loss; go-arounds {described['go_arounds']}")
        split, sample_kinds = window_split(round_, spoken, advantages, gains, trained, table, landings, step_s)
        start_distance, measured_on = {}, {}
        for side in ("real", "augmented", *(("events",) if event_pools else ())):
            part = split_of_kinds(split, sample_kinds, side)
            measured_on[side] = part.sentences
            start_distance[side] = tuner.window_distance(part) if part.windows else None
        tuner.restart(np.random.default_rng([args.seed, round_number, PASS_STREAM]))
        passed = {**tuner.window_pass(split, data, slots=slots, passes=args.passes),
                  "distance_at_start": start_distance, "distance_sentences": measured_on,
                  "speaking_gpu_peak_gb": speaking_peaks}
        if device.type == "cuda":
            passed["pass_gpu_peak_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        log(f"round {round_number}: {args.passes} pass(es) over {split.sentences} aircraft sentences in "
            f"{len(split.windows)} window samples ({passed['batches']} updates), reward term {passed['reward_mean']:.4f}, "
            f"KL to the base at the start {start_distance}, {passed['kl_mean']:.4f} in the pass (max "
            f"{passed['kl_max']:.4f}), data NLL {passed['data_mean']:.4f}, words outside the clip "
            f"{passed['clipped_share']:.4f}, {passed['seconds']:.0f}s; GPU peak: speaking processes "
            + " ".join(f"{p:.2f}" for p in speaking_peaks) + " GB reserved each"
            + (f", the pass {passed['pass_gpu_peak_gb']:.2f} GB allocated" if "pass_gpu_peak_gb" in passed else ""))
        del split, part, spoken                         # the round's sentences go before the select readout's come
        write_traffic_prior(directory, model, prior_dir, start, spec.sha256, git=git, smoke=args.smoke, writer=RUNNER,
                            fine_tuning={"schema": SCHEMA, "commanded": args.commanded, "from": str(prior_dir),
                                         "base": str(base_dir),
                                         "round": round_number, "optimiser": asdict(config),
                                         "traffic_learning_rate": args.traffic_learning_rate,
                                         "samples": args.samples, "passes": args.passes})
        torch.save(tuner.state(), directory / OPTIMISER_FILE)
        readout = read_select(round_number)
        history.append(history_row(round_number, readout, train_pass=passed, sentences=described))
        write_json_atomic(out / "history.json", {"rounds": history})
        write_json_atomic(directory / "readout.json", readout)                       # last: the round finished
    write_choice(out, history, labelled, prior_dir, log)
    speakers.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
