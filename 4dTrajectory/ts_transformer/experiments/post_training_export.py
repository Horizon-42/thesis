"""C11: the Training export of stage C (post-training §8 C11, outline §6) — for each airport, a set of windows of recorded
traffic that the frontend's Training view reads: the commanded flight's head (stage A's: its observed track, open-loop
sentence and closed-loop sentence at the campaign's Δ, `training_export.split_flights`), the other aircraft on their
records over the window, and for each chosen round of a post-training campaign the sentence its model said for the
commanded aircraft, flown — its words, flown track and attitude, outcome and crossing — with the window's end (a loss
of separation, its other aircraft and the minimum it broke; the reward; the rows the speed-word mask acted; the steps
reading a faulty point, D114).

WHICH WINDOWS. ``--per-airport`` real windows of each airport of ``--split`` (train or select; the val days are read
only by a planned readout, outline §6 item 4), drawn once with ``--seed`` from those that do not open inside a loss
(D113); for each, the windows of ``--kinds`` built from it (A, D, B: `post_train.candidate`, the census's ranges; a
kind the window admits none of, or one that opens inside a loss, is left out and counted). Every round flies the same
windows with the same numbers (`post_train.readout_numbers`), so the rounds stand side by side.

WHICH ROUNDS. ``--rounds``: ``start`` (the model at the start: the base with zero-output traffic modules, D29) and the
numbers of rounds whose checkpoint the campaign holds (`post_train.round_model`, refused for another base, masks or
traffic shape).

THE SENTENCE IS THE EXPORT'S OWN READOUT: the windows are flown here (`WindowLoop`), and what is written comes from that
flight — the words, the executor's record on the 2 s rows (the judge's outcome and crossing for a flight the executor
was done with; a window ended at a loss of separation is drawn to the end of the row it ended in). The live executor
flies a word's segment again from the same start and must give the same states (the backend's test). The speaker's
per-row records that stage B's sets carry (the probability of "go-around", the blocked words) are not written: a
window ended by its caller has none in stage B's loop.

CHECKS: the closed loop's (`require_conforming_closed_loop`, D69, with the labeller's and the executor's) and the edge
features' reference of the campaign's census (D104), here first.

WRITES a set ``<root>/<airport>/training/<set-id>/sample.json`` and its entry in
``<root>/<airport>/training/index_post_v2.json`` (`post.training_files`); refused when the set exists. Every airport is
built before any is written. From a clean tree (the set records the commit) unless ``--smoke``; a smoke campaign gives
only a smoke set.

    python run_ts.py post_training_export --campaign <a post_train directory> --rounds start 0 4 --split select \\
        --per-airport 10 --set-id <id> --speed <a model_speed directory of stage C>
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.autopilot.judge import TIMEOUT
from ts_transformer.experiments.training_flights import crossing_payload, last_state_cycle
from ts_transformer.experiments.post_train import (
    CAMPAIGN_SCHEMA, KINDS, POST_CHECKPOINT_SCHEMA, Context, batches, candidate, open_context, readout_numbers,
    round_model, settings_of,
)
from ts_transformer.experiments.model_speed import speed_source
from ts_transformer.experiments.post_window_loop import WindowLoop, WindowResult, checked_edges, moved_commanded
from ts_transformer.experiments.prior_training_export import procedure_block
from ts_transformer.experiments.training_export import (
    FORMATS, candidate_hae_minus_msl_m, candidates_block, events, flown_sentence, split_flights, vocabulary_block,
)
from ts_transformer.instructions import training_files as stage_a_files
from ts_transformer.instructions.spec import READING_RULE
from ts_transformer.instructions.words import Words
from ts_transformer.post import training_files as files
from ts_transformer.post.scene import INSERTED_SUFFIX, MOVED_START, REAL, Window
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
def sentence_payload(loop: WindowLoop, b: int, result: WindowResult, window: Window, first_row: int,
                     executed: Any, aero: np.ndarray, reference: Any, words: Words) -> dict[str, Any]:
    """Window ``b``'s sentence as the Training view draws it (module docstring): its words and their events, the block
    of a flown sentence (`training_export.flown_sentence`: the executor's record on the 2 s rows from the first predicted
    step, its attitude, outcome, crossing and the envelopes of its words) and the window's end. ``reference``: the
    commanded flight's observed signals (its identity)."""
    closed = loop.speaking.loop
    geometry = closed.geometries[b]
    first = loop.speaking.start
    if result.loss is None:                                     # the executor was done: the judge's outcome
        judged = closed.outcome(b)
        last = last_state_cycle(judged.outcome, judged.end_row)
        crossing = crossing_payload(judged, executed, b, geometry)
        end_cycle = int(judged.end_row)
    else:                                                       # ended at the loss: to the end of its row
        last = (len(result.states) - first * loop.every - 1) * closed.row_cycles
        crossing, end_cycle = None, last
    # unrounded (prior D127, followed for windows): the live segment is checked against it within the executor's bound
    block = flown_sentence(executed, b, geometry, reference, result.words, closed.interval_s, aero[b], words,
                           outcome=result.outcome, end_cycle=end_cycle, last_cycle=last, crossing=crossing)
    loss = None if result.loss is None else {
        "step": int(result.loss_step), "timeS": round(window.step_s(result.loss_step) - window.row0_s, 3),
        "other": result.other, "kind": result.loss.kind, "relation": result.loss.relation,
        "requiredM": round(result.loss.required_m, 1), "distanceM": round(result.loss.distance_m, 1),
        "verticalM": round(result.loss.vertical_m, 1), "wakeKnown": bool(result.loss.wake_known)}
    return {
        "rows": len(result.words), "words": result.words.astype(int).tolist(),
        "events": events(result.words, None, geometry, words), "firstRow": first_row, "startRow": first,
        "flownFromRow": first * loop.every, **block, "timedOut": result.outcome == TIMEOUT,
        "goArounds": int(result.go_arounds),
        "end": {"reward": float(result.reward), "loss": loss, "speedMaskRows": int(result.speed_mask_rows),
                "faultySteps": int(result.faulty_steps), "lossReadsFault": bool(result.loss_reads_fault)}}


def fly_round(model: Prior, context: Context, split: str, windows: Sequence[Window], batch_windows: int, seed: int
              ) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Every window flown by ``model`` (module docstring), in batches that command each flight once, each with its
    fixed numbers: their sentence payloads and the observed rows before the first predicted step the start gave each
    (window B's moved, `start_moved`), in the order of ``windows``."""
    data = context.splits[split]
    out: dict[int, dict[str, Any]] = {}
    starts: dict[int, np.ndarray] = {}
    for places in batches(windows, batch_windows):
        batch = [windows[p] for p in places]
        loop, order, observed = context.start_loop(split, batch)([w.signal_index for w in batch])
        flown = WindowLoop(model, loop, order, batch, data["sentences"], data["flights"], context.geometries,
                           context.rosters, context.finals, context.words, interval_s=context.interval_s,
                           variant=context.variant, edges_reference=context.edges_reference, faults=data["faults"],
                           observed=observed, device=context.device)
        results = flown.run([readout_numbers(seed, p) for p in places])
        executed = loop.executor.flown()
        aero = loop.executor.inputs.aero_params.cpu().numpy()
        for b, (place, window, result) in enumerate(zip(places, batch, results)):
            out[place] = sentence_payload(flown, b, result, window, data["sentences"][window.signal_index].rows.first_row,
                                          executed, aero, data["signals"][window.commanded.key], context.words)
            starts[place] = observed[window.signal_index]
    return [out[p] for p in range(len(windows))], [starts[p] for p in range(len(windows))]


# ---- the window's traffic
def traffic_payload(window: Window, end_s: float, hae_minus_msl_m: dict[str, float]) -> list[dict[str, Any]]:
    """The other aircraft of ``window`` from its row 0 to ``end_s`` (UTC epoch s), each on its record's 2 s rows in that
    span: its key, its role (``recorded``; ``inserted``, window A's; ``moved``, window D's, with its shift), its runway
    and category, and its positions with the times from the window's row 0."""
    geometry = window.scene.geometry
    shifts = dict(window.moved)
    out = []
    for flight in window.scene.between(window.row0_s, end_s):
        if flight.key == window.commanded.key:
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


def start_move_payload(window: Window) -> dict[str, float]:
    """A window's start move as the set writes it (unrounded: the live executor moves the start with it again)."""
    move = window.start_move
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


def window_payload(window: Window, rounds: Sequence[str | int], said: Sequence[dict[str, Any]], observed: np.ndarray,
                   hae_minus_msl_m: dict[str, float], step_s: float) -> dict[str, Any]:
    """A window as the set holds it: its commanded flight (``datasetId``: its head is the set's flight of that id), kind,
    row 0, first predicted step, moves (window B's moved observed rows, `start_payload`), its traffic over the longest
    of its rounds, and each round's sentence."""
    longest = max(len(s["track"]["eM"]) for s in said)
    end_s = window.first_step_s + (longest - 1) * step_s
    return {"datasetId": window.commanded.key, "kind": window.kind, "row0S": window.row0_s,
            "firstStepS": round(window.first_step_s - window.row0_s, 3), "startMove": start_move_payload(window),
            "movedStart": start_payload(window, observed),
            "moved": [[key, shift] for key, shift in window.moved],
            "traffic": traffic_payload(window, end_s, hae_minus_msl_m),
            "rounds": [{"round": r, **s} for r, s in zip(rounds, said)]}


# ---- one airport, the set
def build_airport(context: Context, campaign: Path, settings: Any, airport: str, rounds: Sequence[str | int], *,
                  split: str, per_airport: int, kinds: Sequence[str], seed: int, params: Any,
                  hae_minus_msl_m: dict[str, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict]:
    """The airport's flights (stage A's heads of the commanded flights), windows and the draw's counts."""
    windows, drawn = chosen_windows(context, split, airport, per_airport, kinds, seed)
    flown = [fly_round(round_model(context, settings, campaign, None if r == START else int(r)), context, split,
                       windows, settings.batch_windows, seed) for r in rounds]
    said = [payloads for payloads, _ in flown]
    observed = flown[0][1]                      # the start's rows: the same for every round (the move is the window's)
    ids = sorted({w.commanded.key for w in windows})
    heads, _ = split_flights(context.instructions, split, ids, (context.interval_s,), params, context.words,
                             device=context.device)
    for head in heads:
        head["haeMinusMslM"] = hae_minus_msl_m[head["runway"]]
    step_s = context.words.spec.step_s
    return heads, [window_payload(w, rounds, [s[k] for s in said], observed[k], hae_minus_msl_m, step_s)
                   for k, w in enumerate(windows)], drawn


def sample_of(set_id: str, context: Context, airport: str, hae_minus_msl_m: dict[str, float], source: dict[str, Any],
              model: dict[str, Any], cohort: dict[str, Any], cycle_s: float, flights: list[dict[str, Any]],
              windows: list[dict[str, Any]]) -> dict[str, Any]:
    """A set's sample: its head (formats, source, the model, cohort, vocabulary, the airport frame, the candidates with
    their HAE − MSL, the procedure's limits), its flights (stage A's heads) and its windows."""
    geometry = context.geometries[airport]
    return {"schema": files.SAMPLE_SCHEMA, "setId": set_id, "airport": airport, "readingRule": READING_RULE,
            "formats": {**FORMATS, "checkpoint": CHECKPOINT_SCHEMA, "postCheckpoint": POST_CHECKPOINT_SCHEMA,
                        "campaign": CAMPAIGN_SCHEMA},
            "source": source, "model": model, "cohort": cohort,
            "vocabulary": vocabulary_block(context.words.spec, context.words), "executor": {"cycleS": cycle_s},
            "airportFrame": {"code": airport, "lat": geometry.frame.lat0, "lon": geometry.frame.lon0,
                             "elevationM": geometry.elevation_m},
            "candidatesSha256": stage_a_files.candidates_sha256(geometry),
            "candidates": candidates_block(geometry, hae_minus_msl_m),
            "procedure": procedure_block(context.finals[airport]), "flights": flights, "windows": windows}


def index_entry(set_id: str, sample: dict[str, Any]) -> dict[str, Any]:
    """A set as its airport's index lists it."""
    model, cohort = sample["model"], sample["cohort"]
    return {"id": set_id, "kind": files.SET_KIND, "readingRule": READING_RULE,
            "title": f"Stage C · windows · rounds {', '.join(str(r) for r in model['rounds'])} · Δ "
                     f"{model['rowIntervalS']:g} s · {cohort['split']}",
            "file": f"{set_id}/{files.SAMPLE_FILE}", "windows": len(sample["windows"]),
            "flights": len(sample["flights"]), "formats": sample["formats"], "model": model, "cohort": cohort,
            "source": sample["source"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True, help="a post_train directory")
    parser.add_argument("--rounds", nargs="+", required=True, help=f"{START!r} and/or round numbers")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports")
    parser.add_argument("--airports", nargs="+", default=None, help="default: every airport of the artefact")
    parser.add_argument("--per-airport", type=int, default=10)
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=[REAL])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--speed", type=Path, required=True,
                        help="the model's speed readout (a model_speed directory of stage C, D136)")
    parser.add_argument("--smoke", action="store_true", help="SMOKE: allowed from a tree with changes; recorded")
    args = parser.parse_args(argv)
    git = git_state()
    if git["dirty"] and not args.smoke:
        parser.error("the tree has uncommitted changes; a Training set is exported from a commit")
    campaign = args.campaign if args.campaign.is_absolute() else REPO_ROOT / args.campaign
    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    if record["schema"] != CAMPAIGN_SCHEMA:
        parser.error(f"{campaign} is a {record['schema']} campaign, not {CAMPAIGN_SCHEMA}")
    if record["inputs"]["smoke"] and not args.smoke:
        parser.error(f"{campaign} is a smoke campaign: only a --smoke set is made from it")
    speed = speed_source(args.speed if args.speed.is_absolute() else REPO_ROOT / args.speed, "C")
    if speed["smoke"] and not args.smoke:
        parser.error(f"{args.speed} is a smoke speed readout: only a --smoke set names it")
    rounds = [r if r == START else int(r) for r in args.rounds]
    inputs = record["inputs"]
    instructions, executor = Path(inputs["instructions"]), Path(inputs["executor"])
    params, opened, words = require_conforming_closed_loop(instructions, executor)   # D69: the checks run here (D73)
    edges_reference = Path(inputs["windows"]) / "conformance" / "edges.npz"
    checked_edges(edges_reference)                                                  # D104
    settings = settings_of(record)
    context = open_context(Path(inputs["prior"]), instructions, executor, edges_reference, torch.device(args.device),
                           Path(inputs["procedure_root"]), formal=False, data=False)
    airports = args.airports or sorted(context.geometries)
    signals_record = json.loads((instructions / "signals.json").read_text(encoding="utf-8"))
    existing = {airport: files.FILES.read_index(args.root / airport / "training", airport, args.set_id)
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
        flights, windows, drawn = build_airport(context, campaign, settings, airport, rounds, split=args.split,
                                                per_airport=args.per_airport, kinds=args.kinds, seed=args.seed,
                                                params=params, hae_minus_msl_m=hae)
        cohort = {"split": args.split, "perAirport": args.per_airport, "kinds": args.kinds, "seed": args.seed,
                  "windows": len(windows), **drawn,
                  "drawnFrom": "a seeded draw of the airport's real windows that do not open inside a loss (D113)"}
        sample = sample_of(args.set_id, context, airport, hae, source, model, cohort, params.cycle_s, flights, windows)
        entry = index_entry(args.set_id, sample)
        built[airport] = (entry, stage_a_files.serialise(sample))
        print(f"{airport}: {len(windows)} windows, {len(flights)} flights, {len(built[airport][1]) / 1e6:.1f} MB, "
              f"{time.perf_counter() - started:.0f}s", flush=True)
    for airport, (entry, _) in built.items():          # every airport writable before any is written
        files.FILES.require_writable(args.root / airport / "training", airport, entry, existing[airport])
    for airport, (entry, text) in built.items():
        print(f"→ {files.FILES.write_set(args.root / airport / 'training', airport, entry, text, existing[airport])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
