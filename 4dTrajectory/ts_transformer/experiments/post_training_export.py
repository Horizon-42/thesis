"""C11: the Training export of stage C (post-training §8 C11, outline §6) — for each airport, a set of windows of recorded
traffic that the frontend's Training view reads: the commanded flight's head (stage A's: its observed track, open-loop
sentence and closed-loop sentence at the campaign's Δ, `training_export.split_flights`), the other aircraft on their
records over the window, and for each chosen round of a post-training campaign the sentence its model said for the
commanded aircraft, flown — its words, flown track and attitude, outcome and crossing, its reward and the rows the
speed-word mask acted — and the window's end (its losses of separation, each with its two aircraft, the one that answers
and the minimum it broke; the steps reading a faulty point, D114). The window format is stages C's and D's
(`post.training_files`, frontend D156): its commanded aircraft are a list — stage C's windows of one; a window of
several (post-training §9 item 1: the synthetic fixture of frontend §8 F3) is written with each aircraft's sentence, the
row from which it is silent (`silent_row`, D144) and the window's losses once a pair (`round_end_payload`), each aircraft
flown with its own numbers (`member_numbers`).

WHICH WINDOWS. ``--per-airport`` real windows of each airport of ``--split`` (train or select; the val days are read
only by a planned readout, outline §6 item 4), drawn once with ``--seed`` from those that do not open inside a loss
(D113); for each, the windows of ``--kinds`` built from it (A, D, B: `post_train.candidate`, the census's ranges; a
kind the window admits none of, or one that opens inside a loss, is left out and counted). Every round flies the same
windows with the same numbers (`post_train.readout_numbers`), so the rounds stand side by side. Or ``--windows`` (a
window list, `post.window_lists`, post-training D176 (3)): the list's windows in place of the draw — refused with
``--per-airport``, ``--seed`` or ``--kinds`` —, each found at its place among the campaign's selection windows by its
identity (a list of another stage, split or selection, or a window that differs, refused by name) and flown with its
selection readout's numbers (`StageExport.readout_numbers`), so that each round says the sentence its readout judged
(within post-training §6.4: a batch of other windows can change a word drawn near a boundary). The set's cohort is
``drawn`` or ``listed`` (the list's path, sha256, sentence and count, and its select seed).

WHICH ROUNDS. ``--rounds``: ``start`` (the model at the start, `post_train.campaign_start`: the base with zero-output
traffic modules, D29, or the round of another campaign it starts from, D162; the set names it by
``model.settings.start``) and the numbers of rounds whose checkpoint the campaign holds (`post_train.round_model`,
refused for another base, masks or traffic shape).

THE SENTENCE IS THE EXPORT'S OWN READOUT: the windows are flown here (`WindowLoop`), and what is written comes from that
flight — the words, the executor's record on the 2 s rows (the judge's outcome and crossing for a flight the executor
was done with; a window ended at a loss of separation is drawn to the end of the row it ended in). The live executor
flies a word's segment again from the same start and must give the same states (the backend's test). The speaker's
per-row records that stage B's sets carry (the probability of "go-around", the blocked words) are not written: a
window ended by its caller has none in stage B's loop.

CHECKS: the closed loop's (`require_conforming_closed_loop`, D69, with the labeller's and the executor's) and the edge
features' reference of the campaign's census (D104), here first.

WRITES a set ``<root>/<airport>/training/<set-id>/sample.json`` and its entry in
``<root>/<airport>/training/index_post_v4.json`` (`post.training_files`); refused when the set exists. Every airport is
built before any is written. From a clean tree (the set records the commit) unless ``--smoke``; a smoke campaign gives
only a smoke set.

    python run_ts.py post_training_export --campaign <a post_train directory> --rounds start 0 4 --split select \\
        --per-airport 10 --set-id <id> --speed <a model_speed directory of stage C>
    python run_ts.py post_training_export --campaign <a post_train directory> --rounds start 8 --split select \
        --windows <a window_list directory>/list.json --set-id <id> --speed <a model_speed directory of stage C>
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.autopilot.judge import TIMEOUT
from ts_transformer.experiments.training_flights import crossing_payload, last_state_cycle
from ts_transformer.experiments.post_train import (
    CAMPAIGN_SCHEMA, KINDS, POST_CHECKPOINT_SCHEMA, Context, batches, candidate, open_context, readout_numbers,
    round_model, selection_windows, settings_of,
)
from ts_transformer.experiments.model_speed import speed_source
from ts_transformer.experiments.post_window_loop import WindowLoop, WindowResult, checked_edges, moved_commanded
from ts_transformer.experiments.prior_training_export import procedure_block
from ts_transformer.experiments.training_export import (
    FORMATS, candidate_hae_minus_msl_m, candidates_block, events, flown_sentence, split_flights, this_checkout,
    vocabulary_block,
)
from ts_transformer.instructions import training_files as stage_a_files
from ts_transformer.instructions.spec import READING_RULE
from ts_transformer.instructions.words import Words
from ts_transformer.io_utils import file_sha256
from ts_transformer.post import training_files as files
from ts_transformer.post.window_lists import WindowList, read_window_list
from ts_transformer.post.scene import INSERTED_SUFFIX, MOVED_START, NO_START_MOVE, REAL, StartMove, Window
from ts_transformer.post.traffic import opens_inside_loss
from ts_transformer.prior.checkpoint import CHECKPOINT_SCHEMA
from ts_transformer.prior.model import Prior
from ts_transformer.prior.procedure import PROCEDURE_MASKS
from ts_transformer.prior.speaker import MOST_GO_AROUNDS
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

#: The splits a set is exported from (outline §6 item 4: the val days only from a planned readout).
SPLITS = ("train", "select")
#: The model at the start of a campaign, as ``--rounds`` names it.
START = "start"


# ---- the windows
def chosen_windows(context: Context, split: str, airport: str, per_airport: int, kinds: Sequence[str], seed: int
                   ) -> tuple[list[Window], dict[str, Any]]:
    """The airport's windows (module docstring): ``per_airport`` real windows drawn with ``seed`` from those of
    ``split`` that do not open inside a loss (D113), and for each the windows of ``kinds`` built from it, in that order;
    and the count of what the draw left out."""
    data, separation, finals = context.splits[split], context.separations[airport], context.finals[airport]
    step_s = context.words.spec.step_s
    pool = [w for w in data["windows"] if w.scene.geometry.code == airport
            and not opens_inside_loss(w, separation, finals, step_s)]
    picked = sorted(np.random.default_rng([seed, 1 << 29]).permutation(len(pool))[:per_airport])
    out: list[Window] = []
    left_out: Counter = Counter()
    for place in picked:
        real = pool[int(place)]
        rng = np.random.default_rng([seed, 1 << 28, int(place)])
        for kind in kinds:
            window = candidate(kind, real, rng, separation)
            judged = window if window is None or kind != MOVED_START else replace(
                window, commanded=moved_commanded(window, data["signals"][window.commanded.key], step_s))
            if window is None or opens_inside_loss(judged, separation, finals, step_s):
                left_out[kind] += 1
            else:
                out.append(window)
    return out, {"pool": len(pool), "real": len(picked), "leftOut": dict(left_out)}


# ---- flying a round's model
def sentence_payload(loop: WindowLoop, b: int, result: WindowResult, first_row: int, executed: Any, aero: np.ndarray,
                     reference: Any, words: Words) -> dict[str, Any]:
    """Row ``b``'s commanded aircraft's sentence as the Training view draws it (module docstring): its words and
    their events, the block of a flown sentence (`training_export.flown_sentence`: the executor's record on the 2 s rows
    from the first predicted step, its attitude, outcome, crossing and the envelopes of its words), its reward, the row
    from which it is silent (`silent_row`) and the rows the speed-word mask acted. ``reference``: the commanded
    flight's observed signals (its identity)."""
    closed = loop.speaking.loop
    geometry = closed.geometries[b]
    first = loop.speaking.start
    if not loop.speaking.ended[b]:
        # the executor was done with it — a silent aircraft too (D144: it answered a loss and flew on to an end of its
        # own): the judge's outcome, end and crossing (its reward stays the loop's: 0 for the loss it answered)
        judged = closed.outcome(b)
        outcome = judged.outcome
        last = last_state_cycle(judged.outcome, judged.end_row)
        crossing = crossing_payload(judged, executed, b, geometry)
        end_cycle = int(judged.end_row)
    else:                       # its caller ended it at its loss (stage C's window), or halted it silent: its row's end
        outcome = result.outcome
        last = (len(result.states) - first * loop.every - 1) * closed.row_cycles
        crossing, end_cycle = None, last
    # unrounded (prior D127, followed for windows): the live segment is checked against it within the executor's bound
    block = flown_sentence(executed, b, geometry, reference, result.words, closed.interval_s, aero[b], words,
                           outcome=outcome, end_cycle=end_cycle, last_cycle=last, crossing=crossing)
    return {
        "rows": len(result.words), "words": result.words.astype(int).tolist(),
        "events": events(result.words, None, geometry, words), "firstRow": first_row, "startRow": first,
        "flownFromRow": first * loop.every, **block, "timedOut": outcome == TIMEOUT,
        "goArounds": int(result.go_arounds), "reward": float(result.reward),
        "silentFromRow": silent_row(result, int(loop.speaking.loop.join_ticks[b]), first),
        "speedMaskRows": int(result.speed_mask_rows)}


def silent_row(result: WindowResult, join: int, start: int) -> int | None:
    """The row of an aircraft's sentence (from its first predicted step) from which it is silent (D144: it answered for
    a loss and its window flew on, the model saying it "unchanged" alone): the row of its loss's tick, ``join`` its join
    tick and ``start`` the Δ rows to its first predicted step; None when it was not (no loss, or its loss ended the
    window there: stage C's window of one commanded aircraft, D93)."""
    if result.loss_step is None:
        return None
    row = result.loss_step - join - start
    return row if row < len(result.words) else None


def round_end_payload(results: Sequence[WindowResult], keys: Sequence[str], window: Window) -> dict[str, Any]:
    """Window ``window``'s end in a round, from its commanded aircraft's ends (``results``, their ``keys``, in the
    window's order): its losses of separation — those its commanded aircraft answer for (stage C's window of one, at
    most one), one entry a pair and step with every commanded aircraft that answers it, each costing W (an answering
    aircraft's reward is 0, D30, multi-aircraft control D141) — each with its step and time from the window's row 0,
    its two aircraft, the minimum it broke and whether the other aircraft read a faulty point near it (D114); and the
    steps at which a recorded aircraft read a faulty point."""
    losses: dict[tuple[int, frozenset[str]], dict[str, Any]] = {}
    for key, result in zip(keys, results, strict=True):
        if result.loss is None:
            continue
        pair = (int(result.loss_step), frozenset((key, result.other)))
        if pair in losses:
            losses[pair]["answering"].append(key)
            losses[pair]["readsFault"] = losses[pair]["readsFault"] or bool(result.loss_reads_fault)
            continue
        losses[pair] = {
            "step": int(result.loss_step), "timeS": round(window.step_s(result.loss_step) - window.row0_s, 3),
            "aircraft": [key, result.other], "answering": [key],
            "kind": result.loss.kind, "relation": result.loss.relation, "requiredM": round(result.loss.required_m, 1),
            "distanceM": round(result.loss.distance_m, 1), "verticalM": round(result.loss.vertical_m, 1),
            "wakeKnown": bool(result.loss.wake_known), "readsFault": bool(result.loss_reads_fault), "costsW": True}
    return {"losses": list(losses.values()), "faultySteps": int(results[0].faulty_steps)}


def member_numbers(seed: int, place: int, member: int) -> np.random.Generator:
    """The numbers of commanded aircraft ``member`` of window ``place``: the anchor's are the selection readout's
    (`post_train.readout_numbers`, stage C's window of one as before); a later member's stage D's readout's
    (`multi_train.readout_numbers`, MIRROR: the runner is not imported here)."""
    return readout_numbers(seed, place) if member == 0 else np.random.default_rng([seed, 1 << 30, place, member])


@dataclass(frozen=True)
class Flying:
    """How a stage's set flies its windows (`fly_round`): its batches of the windows' places, each commanded aircraft's
    numbers (window place, member), the window loop's options (post-training §9 item 5) and a window's end in a round
    (the loop, its window w of the batch, the window, its commanded aircraft's ends in its order → the payload)."""

    batches: Callable[[Sequence[Window]], list[list[int]]]
    numbers: Callable[[int, int], np.random.Generator]
    options: Mapping[str, Any]
    end: Callable[[WindowLoop, int, Window, Sequence[WindowResult]], dict[str, Any]]


#: Each commanded aircraft's numbers in a set: (the window's place in the set, the aircraft's in the window) → numbers.
Numbers = Callable[[int, int], np.random.Generator]


def stage_c_flying(batch_windows: int, numbers: Numbers) -> Flying:
    """Stage C's flying (module docstring): `post_train.batches` of ``batch_windows`` rows, the set's ``numbers``, stage
    C's loop (no options) and `round_end_payload` (the losses its commanded aircraft answer for)."""
    return Flying(batches=lambda windows: batches(windows, batch_windows), numbers=numbers, options={},
                  end=lambda flown, w, window, ends: round_end_payload(
                      ends, [flown.records[b].key for b in flown.members[w]], window))


def fly_round(model: Prior, context: Context, split: str, windows: Sequence[Window], flying: Flying
              ) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]], list[np.ndarray]]:
    """Every window flown by ``model`` (module docstring) as ``flying`` gives (`Flying`: its batches, which command each
    flight once, each aircraft's fixed numbers, the loop's options, a window's end): each window's commanded aircraft's
    sentence payloads (in the window's order), its end and the observed rows before the first predicted step the start
    gave its anchor (window B's moved, `start_moved`), in the order of ``windows``."""
    data = context.splits[split]
    out: dict[int, list[dict[str, Any]]] = {}
    ends: dict[int, dict[str, Any]] = {}
    starts: dict[int, np.ndarray] = {}
    for places in flying.batches(windows):
        batch = [windows[p] for p in places]
        loop, order, observed = context.start_loop(split, batch)(sorted(i for w in batch for i in w.signal_indices))
        flown = WindowLoop(model, loop, order, batch, data["sentences"], data["flights"], context.geometries,
                           context.rosters, context.finals, context.words, interval_s=context.interval_s,
                           variant=context.variant, edges_reference=context.edges_reference, faults=data["faults"],
                           observed=observed, device=context.device, **flying.options)
        results = flown.run([flying.numbers(places[int(flown.window_of[b])], int(flown.member_of[b]))
                             for b in range(len(flown.order))])
        executed = loop.executor.flown()
        aero = loop.executor.inputs.aero_params.cpu().numpy()
        for w, (place, window, rows) in enumerate(zip(places, batch, flown.members, strict=True)):
            out[place] = [sentence_payload(flown, b, results[b], data["sentences"][flown.order[b]].rows.first_row,
                                           executed, aero, data["signals"][flown.records[b].key], context.words)
                          for b in rows]
            ends[place] = flying.end(flown, w, window, [results[b] for b in rows])
            starts[place] = observed[window.signal_index]
    return ([out[p] for p in range(len(windows))], [ends[p] for p in range(len(windows))],
            [starts[p] for p in range(len(windows))])


# ---- the window's traffic
def traffic_payload(window: Window, end_s: float, hae_minus_msl_m: dict[str, float]) -> list[dict[str, Any]]:
    """The other aircraft of ``window`` from its row 0 to ``end_s`` (UTC epoch s), each on its record's 2 s rows in that
    span: its key, its role (``recorded``; ``inserted``, window A's; ``moved``, window D's, with its shift), its runway
    and category, and its positions with the times from the window's row 0."""
    geometry = window.scene.geometry
    shifts = dict(window.moved)
    out = []
    commanded = {record.key for record in window.commanded_all}
    for flight in window.scene.between(window.row0_s, end_s):
        if flight.key in commanded:
            continue
        times = flight.start_s + flight.step_s * np.arange(len(flight.e_m))
        kept = (times >= window.row0_s) & (times <= end_s)
        lat, lon = geometry.frame.latlon_from_horizontal(flight.e_m[kept], flight.n_m[kept])
        runway = geometry.candidates[flight.runway_index].ident
        role = ("inserted" if flight.key.endswith(INSERTED_SUFFIX) else "moved" if flight.key in shifts
                else "recorded")
        out.append({"key": flight.key, "role": role, "shiftS": shifts.get(flight.key), "runway": runway,
                    "category": flight.category, "haeMinusMslM": hae_minus_msl_m[runway],
                    "landingS": round(flight.landing_s - window.row0_s, 3),
                    "tS": stage_a_files.rounded(times[kept] - window.row0_s, 3),
                    "eM": stage_a_files.rounded(flight.e_m[kept], 1), "nM": stage_a_files.rounded(flight.n_m[kept], 1),
                    "latDeg": stage_a_files.rounded(lat, 7), "lonDeg": stage_a_files.rounded(lon, 7),
                    "heightMslM": stage_a_files.rounded(flight.height_m[kept], 1)})
    return out


def start_move_payload(move: StartMove) -> dict[str, float]:
    """A commanded aircraft's start move as the set writes it (unrounded: the live executor moves the start with it
    again): window B's anchor's its window's, every other one's none."""
    return {"turnDeg": move.turn_deg, "heightM": move.height_m, "speedScale": move.speed_scale}


def start_payload(window: Window, observed: np.ndarray) -> dict[str, Any] | None:
    """Window B's observed rows before its first predicted step as its start moved them (``observed``: `start_moved`'s,
    `STATE_COLUMNS` on the 2 s rows from the sentence's first row) — what the view draws in place of the head's observed
    track there; None for every other window (its start is the head's)."""
    if window.kind != MOVED_START:
        return None
    lat, lon = window.scene.geometry.frame.latlon_from_horizontal(observed[:, 0], observed[:, 1])
    return {"rows": len(observed), "eM": stage_a_files.rounded(observed[:, 0], 1),
            "nM": stage_a_files.rounded(observed[:, 1], 1), "latDeg": stage_a_files.rounded(lat, 7),
            "lonDeg": stage_a_files.rounded(lon, 7), "heightMslM": stage_a_files.rounded(observed[:, 2], 1)}


def window_payload(window: Window, rounds: Sequence[str | int], said: Sequence[Sequence[dict[str, Any]]],
                   ends: Sequence[dict[str, Any]], observed: np.ndarray, hae_minus_msl_m: dict[str, float],
                   step_s: float) -> dict[str, Any]:
    """A window as the set holds it (frontend §5.7): its kind and c (none: stage C's windows are not compressed), its
    commanded aircraft in the order they join (``said``: each round's sentences of them) — each one's ``datasetId``
    (its head is the set's flight of that id), its join offset from the window's row 0 (the anchor's 0; none moved in
    time), its first predicted step, its start move (window B's anchor: its moved observed rows, `start_payload`) and
    each round's sentence —, the recorded aircraft moved in time (window D), the traffic to the last row any of them
    flew in any round, and each round's end (`round_end_payload`)."""
    members = window.commanded_all
    end_s = max(record.first_step_s + (len(s[m]["track"]["eM"]) - 1) * step_s
                for s in said for m, record in enumerate(members))
    joins = window.join_steps()
    aircraft = [{"datasetId": record.key, "joinS": round(float(joins[m]) * window.scene.interval_s, 3), "shiftS": None,
                 "firstStepS": round(record.first_step_s - window.row0_s, 3),
                 "startMove": start_move_payload(window.start_move if m == 0 else NO_START_MOVE),
                 "movedStart": start_payload(window, observed) if m == 0 else None,
                 "rounds": [{"round": r, **s[m]} for r, s in zip(rounds, said)]}
                for m, record in enumerate(members)]
    return {"kind": window.kind, "c": None, "commanded": aircraft,
            "moved": [[key, shift] for key, shift in window.moved],
            "traffic": traffic_payload(window, end_s, hae_minus_msl_m),
            "rounds": [{"round": r, **e} for r, e in zip(rounds, ends)]}


# ---- one airport, the set
def window_set(context: Context, split: str, windows: Sequence[Window], rounds: Sequence[str | int],
               model_of: Callable[[str | int], Prior], flying: Flying, params: Any, hae_minus_msl_m: dict[str, float]
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``windows`` flown by each round's model (``model_of``) as ``flying`` gives: the heads of their commanded flights
    (stage A's, `split_flights`) and the windows as the set holds them (`window_payload`) — stage C's sets and stage D's
    (`multi_training_export`) alike."""
    flown = [fly_round(model_of(r), context, split, windows, flying) for r in rounds]
    said = [payloads for payloads, _, _ in flown]
    ends = [round_ends for _, round_ends, _ in flown]
    observed = flown[0][2]                      # the start's rows: the same for every round (the move is the window's)
    ids = sorted({record.key for w in windows for record in w.commanded_all})
    heads, _ = split_flights(context.instructions, split, ids, (context.interval_s,), params, context.words,
                             device=context.device)
    for head in heads:
        head["haeMinusMslM"] = hae_minus_msl_m[head["runway"]]
    step_s = context.words.spec.step_s
    return heads, [window_payload(w, rounds, [s[k] for s in said], [e[k] for e in ends], observed[k], hae_minus_msl_m,
                                  step_s) for k, w in enumerate(windows)]


#: Stage C's campaign and checkpoint formats, as a set names them (stage D's export gives its own).
CAMPAIGN_FORMATS = {"postCheckpoint": POST_CHECKPOINT_SCHEMA, "campaign": CAMPAIGN_SCHEMA}


def sample_of(set_id: str, context: Context, airport: str, hae_minus_msl_m: dict[str, float], source: dict[str, Any],
              model: dict[str, Any], cohort: dict[str, Any], cycle_s: float, flights: list[dict[str, Any]],
              windows: list[dict[str, Any]], campaign_formats: Mapping[str, str] = CAMPAIGN_FORMATS) -> dict[str, Any]:
    """A set's sample: its head (formats — the stage's campaign and checkpoint, ``campaign_formats`` —, source, the
    model, cohort, vocabulary, the airport frame, the candidates with their HAE − MSL, the procedure's limits), its
    flights (stage A's heads) and its windows."""
    geometry = context.geometries[airport]
    return {"schema": files.SAMPLE_SCHEMA, "setId": set_id, "airport": airport, "readingRule": READING_RULE,
            "formats": {**FORMATS, "checkpoint": CHECKPOINT_SCHEMA, **campaign_formats},
            "source": source, "model": model, "cohort": cohort,
            "vocabulary": vocabulary_block(context.words.spec, context.words), "executor": {"cycleS": cycle_s},
            "airportFrame": {"code": airport, "lat": geometry.frame.lat0, "lon": geometry.frame.lon0,
                             "elevationM": geometry.elevation_m},
            "candidatesSha256": stage_a_files.candidates_sha256(geometry),
            "candidates": candidates_block(geometry, hae_minus_msl_m),
            "procedure": procedure_block(context.finals[airport]), "flights": flights, "windows": windows}


#: Each stage's Training files (frontend §5.7: one format, the stage the index file's).
STAGE_FILES = {"C": files.FILES, "D": files.MULTI_FILES}


def index_entry(stage: str, set_id: str, sample: dict[str, Any]) -> dict[str, Any]:
    """A set of ``stage`` as its airport's index lists it (a listed set's title says so)."""
    model, cohort = sample["model"], sample["cohort"]
    windows = "listed windows" if cohort["form"] == "listed" else "windows"
    return {"id": set_id, "kind": files.SET_KIND, "readingRule": READING_RULE,
            "title": f"Stage {stage} · {windows} · rounds {', '.join(str(r) for r in model['rounds'])} · Δ "
                     f"{model['rowIntervalS']:g} s · {cohort['split']}",
            "file": f"{set_id}/{files.SAMPLE_FILE}", "windows": len(sample["windows"]),
            "flights": len(sample["flights"]), "formats": sample["formats"], "model": model, "cohort": cohort,
            "source": sample["source"]}


@dataclass(frozen=True)
class StageExport:
    """What a stage gives the shared export of window sets (`export_sets`): its stage (C or D: its index,
    `STAGE_FILES`), its campaign's schema and formats, its settings from a campaign record, an airport's drawn windows
    and the counts of their draw (context, settings, split, airport, ``--per-airport``, ``--kinds``, ``--seed``), a
    round's model (context, settings, campaign directory, round), its flying with given numbers (context, settings,
    `Numbers`), the numbers of a drawn set (settings, ``--seed``) and of the readout (settings: by a window's place
    among the selection windows, post-training D176 (3)), the campaign's selection windows (context, settings), a
    window's identity and the selection's fields as a window list names them (`post.window_lists`), the cohort's
    sentence on the draw, and whether a set names a speed readout of the stage (D136; ``--speed``)."""

    stage: str
    campaign_schema: str
    campaign_formats: Mapping[str, str]
    settings_of: Callable[[Mapping[str, Any]], Any]
    windows: Callable[..., tuple[list[Window], dict[str, Any]]]
    model_of: Callable[[Context, Any, Path, str | int], Prior]
    flying: Callable[[Context, Any, Numbers], Flying]
    drawn_numbers: Callable[[Any, int], Numbers]
    readout_numbers: Callable[[Any], Numbers]
    selection: Callable[[Context, Any], list[Window]]
    identity: Callable[[Window], dict[str, Any]]
    selection_fields: Callable[[Any], dict[str, Any]]
    drawn_from: str
    speed: bool


#: A drawn set's windows an airport and seed when the arguments do not say (``--per-airport``, ``--seed``; stage C's
#: ``--kinds`` real): they are refused with ``--windows``, so they have no argparse default.
DRAWN_PER_AIRPORT, DRAWN_SEED = 10, 1337


def add_arguments(parser: argparse.ArgumentParser, stage: StageExport) -> None:
    """The arguments of a stage's export (``--kinds`` stage C's only: stage D's sets are of real windows, D166 (33);
    ``--speed`` where the stage has a speed readout; ``--windows`` a window list, post-training D176 (3))."""
    parser.add_argument("--campaign", type=Path, required=True, help="the campaign's directory")
    parser.add_argument("--rounds", nargs="+", required=True, help=f"{START!r} and/or round numbers")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports")
    parser.add_argument("--airports", nargs="+", default=None,
                        help="a drawn set: default every airport of the artefact (a listed set: the list's)")
    parser.add_argument("--windows", type=Path, default=None,
                        help="a window list (`window_list`'s list.json, ts-window-list-v1), in place of the draw (D176)")
    parser.add_argument("--per-airport", type=int, default=None, help=f"a drawn set: default {DRAWN_PER_AIRPORT}")
    if stage.stage == "C":
        parser.add_argument("--kinds", nargs="+", choices=KINDS, default=None, help=f"a drawn set: default {REAL}")
    parser.add_argument("--seed", type=int, default=None, help=f"a drawn set: default {DRAWN_SEED}")
    parser.add_argument("--device", default="cpu")
    if stage.speed:
        parser.add_argument("--speed", type=Path, required=True,
                            help=f"the model's speed readout (a model_speed directory of stage {stage.stage}, D136)")
    parser.add_argument("--smoke", action="store_true", help="SMOKE: allowed from a tree with changes; recorded")


def listed_windows(parser: argparse.ArgumentParser, stage: StageExport, listed: WindowList, split: str,
                   selection: Sequence[Window], selection_fields: Mapping[str, Any]) -> list[tuple[int, Window]]:
    """A window list's windows (post-training D176 (3)), each with its place among the campaign's selection windows,
    in the list's order: refused by name for a list of another stage, split or selection, or a window that is not at
    its place (no window there, another airport or another identity)."""
    if listed.stage != stage.stage:
        parser.error(f"the window list is of stage {listed.stage}, the campaign of stage {stage.stage}")
    if listed.split != split:
        parser.error(f"the window list names {listed.split} windows, the set is of {split} windows")
    if dict(listed.selection) != dict(selection_fields):
        parser.error(f"the window list indexes the selection {dict(listed.selection)}, the campaign's is "
                     f"{dict(selection_fields)}")
    out = []
    for item in listed.windows:
        if item.place >= len(selection):
            parser.error(f"listed window {dict(item.identity)}: place {item.place}, the campaign has "
                         f"{len(selection)} selection windows")
        window = selection[item.place]
        found = stage.identity(window)
        if window.scene.geometry.code != item.airport or found != dict(item.identity):
            parser.error(f"listed window at place {item.place} ({item.airport}, {dict(item.identity)}) is not the "
                         f"campaign's ({window.scene.geometry.code}, {found})")
        out.append((item.place, window))
    return out


def export_sets(parser: argparse.ArgumentParser, args: argparse.Namespace, stage: StageExport) -> int:
    """A stage's window sets (module docstring), every airport built before any is written, into ``stage``'s index: of
    drawn windows, or of a window list's (``--windows``, D176 (3)), each flown with its selection readout's numbers."""
    stage_files = STAGE_FILES[stage.stage]
    # a listed set's windows and airports are the list's: the draw's arguments and --airports go without it
    drawn_args = {"--airports": args.airports, "--per-airport": args.per_airport, "--seed": args.seed,
                  **({"--kinds": args.kinds} if stage.stage == "C" else {})}
    if args.windows is not None and any(value is not None for value in drawn_args.values()):
        parser.error(f"--windows names the set's windows: refused with "
                     f"{', '.join(name for name, value in drawn_args.items() if value is not None)}")
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("the tree has uncommitted changes; a Training set is exported from a commit")
    campaign = args.campaign if args.campaign.is_absolute() else REPO_ROOT / args.campaign
    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    if record["schema"] != stage.campaign_schema:
        parser.error(f"{campaign} is a {record['schema']} campaign, not {stage.campaign_schema}")
    if record["inputs"]["smoke"] and not args.smoke:
        parser.error(f"{campaign} is a smoke campaign: only a --smoke set is made from it")
    speed = None
    if stage.speed:
        speed = speed_source(args.speed if args.speed.is_absolute() else REPO_ROOT / args.speed, stage.stage)
        if speed["smoke"] and not args.smoke:
            parser.error(f"{args.speed} is a smoke speed readout: only a --smoke set names it")
    rounds = [r if r == START else int(r) for r in args.rounds]
    # the campaign's paths as this checkout reads them (a campaign run in the main checkout, exported from a worktree)
    inputs = {**record["inputs"], **{key: str(this_checkout(record["inputs"][key]))
                                     for key in ("instructions", "executor", "prior", "windows", "procedure_root")}}
    instructions, executor = Path(inputs["instructions"]), Path(inputs["executor"])
    params, opened, words = require_conforming_closed_loop(instructions, executor)   # D69: the checks run here (D73)
    edges_reference = Path(inputs["windows"]) / "conformance" / "edges.npz"
    checked_edges(edges_reference)                                                  # D104
    settings = stage.settings_of(record)
    context = open_context(Path(inputs["prior"]), instructions, executor, edges_reference, torch.device(args.device),
                           Path(inputs["procedure_root"]), formal=False, data=False)
    listed_by_airport: dict[str, list[tuple[int, Window]]] = {}
    if args.windows is not None:
        list_path = args.windows if args.windows.is_absolute() else REPO_ROOT / args.windows
        listed = read_window_list(list_path)
        fields = stage.selection_fields(settings)
        for place, window in listed_windows(parser, stage, listed, args.split, stage.selection(context, settings),
                                            fields):
            listed_by_airport.setdefault(window.scene.geometry.code, []).append((place, window))
        airports = sorted(listed_by_airport)
        readout = stage.readout_numbers(settings)
        listed_cohort = {"form": "listed", "split": args.split, "list": repo_relative(list_path),
                         "sha256": file_sha256(list_path), "chose": listed.chose,
                         "listCount": len(listed.windows), "selectSeed": fields["select_seed"]}
    else:
        airports = args.airports or sorted(context.geometries)
        per_airport = DRAWN_PER_AIRPORT if args.per_airport is None else args.per_airport
        seed = DRAWN_SEED if args.seed is None else args.seed
        kinds = (args.kinds or [REAL]) if stage.stage == "C" else [REAL]
    signals_record = json.loads((instructions / "signals.json").read_text(encoding="utf-8"))
    existing = {airport: stage_files.read_index(args.root / airport / "training", airport, args.set_id)
                for airport in airports}
    model = {"campaign": repo_relative(campaign), "rounds": rounds, "rowIntervalS": context.interval_s,
             "mostGoArounds": MOST_GO_AROUNDS, "procedureMasks": PROCEDURE_MASKS, "settings": inputs["settings"]}
    source = {"campaign": repo_relative(campaign), "campaignGit": record["git"], "campaignSmoke": inputs["smoke"],
              "instructions": repo_relative(instructions), "executor": repo_relative(executor),
              "specSha256": words.spec.sha256, "executorSpecSha256": opened["sha256"], "checks": opened["checks"],
              "git": git, "smoke": args.smoke, "device": args.device, "speed": speed}
    started = time.perf_counter()
    built = {}
    for airport in airports:
        hae = candidate_hae_minus_msl_m(signals_record["runway_ends_from"], context.geometries[airport])
        if args.windows is not None:
            places = [place for place, _ in listed_by_airport[airport]]
            chosen = [window for _, window in listed_by_airport[airport]]
            numbers: Numbers = lambda k, member, places=places: readout(places[k], member)
            cohort = {**listed_cohort, "windows": len(chosen)}
        else:
            chosen, drawn = stage.windows(context, settings, args.split, airport, per_airport, kinds, seed)
            numbers = stage.drawn_numbers(settings, seed)
            cohort = {"form": "drawn", "split": args.split, "perAirport": per_airport, "kinds": kinds, "seed": seed,
                      "windows": len(chosen), **drawn, "drawnFrom": stage.drawn_from}
        flights, windows = window_set(context, args.split, chosen, rounds,
                                      lambda r: stage.model_of(context, settings, campaign, r),
                                      stage.flying(context, settings, numbers), params, hae)
        sample = sample_of(args.set_id, context, airport, hae, source, model, cohort, params.cycle_s, flights, windows,
                           stage.campaign_formats)
        entry = index_entry(stage.stage, args.set_id, sample)
        built[airport] = (entry, stage_a_files.serialise(sample))
        print(f"{airport}: {len(windows)} windows, {len(flights)} flights, {len(built[airport][1]) / 1e6:.1f} MB, "
              f"{time.perf_counter() - started:.0f}s", flush=True)
    for airport, (entry, _) in built.items():          # every airport writable before any is written
        stage_files.require_writable(args.root / airport / "training", airport, entry, existing[airport])
    for airport, (entry, text) in built.items():
        print(f"→ {stage_files.write_set(args.root / airport / 'training', airport, entry, text, existing[airport])}")
    return 0


#: Stage C's export (module docstring).
STAGE_C = StageExport(
    stage="C", campaign_schema=CAMPAIGN_SCHEMA, campaign_formats=CAMPAIGN_FORMATS, settings_of=settings_of,
    windows=lambda context, settings, split, airport, per_airport, kinds, seed: chosen_windows(
        context, split, airport, per_airport, kinds, seed),
    model_of=lambda context, settings, campaign, r: round_model(context, settings, campaign, None if r == START else int(r)),
    flying=lambda context, settings, numbers: stage_c_flying(settings.batch_windows, numbers),
    drawn_numbers=lambda settings, seed: lambda place, member: member_numbers(seed, place, member),
    # the selection readout's numbers (`post_train.read_batch`): its select seed and the window's place
    readout_numbers=lambda settings: lambda place, member: member_numbers(settings.select_seed, place, member),
    selection=lambda context, settings: selection_windows(context, settings, "select"),
    identity=lambda window: {"flight": window.commanded.key, "row0_s": window.row0_s, "kind": window.kind},
    selection_fields=lambda settings: {"select_seed": settings.select_seed, "per_airport": settings.select_per_airport},
    drawn_from="a seeded draw of the airport's real windows that do not open inside a loss (D113)", speed=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    add_arguments(parser, STAGE_C)
    return export_sets(parser, parser.parse_args(argv), STAGE_C)


if __name__ == "__main__":
    raise SystemExit(main())
