"""The single-flight executor against the batched torch executor on the Training sets' own flights (the results contract
of `single`, the user's choice of 2026-09-27): every segment the live executor can fly, answered twice through the
backend's own path (`fly.fly_segment`) — once as it now flies (`fly.fly_one_until`, the single-flight executor), once
with `torch_fly_one_until` (the backend's drive before, `executor.Executor` on a batch of one) — and compared: the same
refusal or none, the same outcome, end row, crossing runway and verdict for every word, the same stop; how far apart
the flown states are, and how long each took.

    python -m aeroviz_backend.autopilot_segment.check_single --out <directory>
        [--airport KRDU …] [--models base landing_r01 augmented_r07] [--limit N]

Truth: every word said in each flight's labelled sentence (the segments `segment.segment_of` accepts). Models: each
named overlay's samples, one segment per sample — its latest word the live executor accepts, which flies the sentence
from its first step to the outcome (or the stop) — so the whole flight is compared. Writes ``check.json`` (every
differing segment named) and prints the summary. Read-only: nothing is written outside ``--out``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.executor import Executor, Flown
from ts_transformer.autopilot.frame import AirportCharts
from ts_transformer.autopilot.lateral import Runways
from ts_transformer.autopilot.sentence import Sentences, TimeClock, row_at
from ts_transformer.data.dataset import series_from_row
from ts_transformer.instructions import training_files
from ts_transformer.instructions.words import COLUMNS, UNCHANGED
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import COMPARISON_AIRPORTS_ROOT

from aeroviz_backend.autopilot_segment import fly as fly_module
from aeroviz_backend.autopilot_segment.backend import AutopilotSegmentBackend
from aeroviz_backend.autopilot_segment.errors import NotFlyable, RequestRefused, Superseded
from aeroviz_backend.autopilot_segment.segment import model_sentence

CPU = torch.device("cpu")
AIRPORTS = ("KMSY", "KRDU", "KSJC", "KSMF", "KSTL")
MODELS = ("base", "landing_r01", "augmented_r07")


def never() -> bool:
    return False


def torch_fly_one_until(context, segment, reading, signals, params, words, stop_steps, superseded, *,
                        model_limit_s=None) -> tuple[Flown, bool, float]:
    """The backend's drive before the single-flight executor: `executor.Executor` on a batch of one, set up as
    `replay.fly_sentences` sets it up, stepped as `executor.fly` steps it and stopped at ``stop_steps``."""
    spec = words.spec
    limit_s = len(reading.words) * spec.step_s * params.timeout_factor if model_limit_s is None else model_limit_s
    batch = replay.Batch(signals=[signals], series=[series_from_row(context.series, segment.start_row)],
                         readings=[reading], geometries=[context.geometry], vertical_paths=[context.vertical_paths],
                         approach_ias_mps=[context.approach_ias_mps], groups=[context.group], drawn={})
    f64 = torch.float64
    executor = Executor(batch.inputs(CPU), Runways.of(batch.geometries, batch.vertical_paths, dtype=f64, device=CPU),
                        AirportCharts.of(batch.geometries, dtype=f64, device=CPU),
                        torch.tensor(batch.approach_ias_mps, dtype=f64, device=CPU), params, words,
                        time_limit_s=torch.tensor([limit_s], dtype=f64, device=CPU))
    sentences = Sentences([reading.words], words, device=CPU)
    clock = replay.word_clock(batch, params, spec.step_s, CPU) if model_limit_s is None else TimeClock(params.cycle_s)
    started = time.perf_counter()
    reached = False
    for cycle in range(executor.cycles):
        sentence_s = clock.now(cycle, executor.now())
        if cycle % executor.step_rows == 0:
            if stop_steps is not None and int(row_at(sentence_s, spec.step_s)[0]) >= stop_steps:
                reached = True
                break
            step_start_s = sentence_s
        executor.cycle(sentences.at(step_start_s), sentence_s)
        if bool(executor.done.all()):
            break
    fly_s = time.perf_counter() - started
    flown = executor.flown()
    if reached:
        flown = replace(flown, done_cycle=torch.full_like(flown.done_cycle, executor.count - 1))
    return flown, reached, fly_s


def answer(context, params, words, column, row, model=None, masks=None, *, reference: bool) -> dict[str, Any]:
    """One segment through `fly.fly_segment`, with the single-flight drive or (``reference``) the torch one."""
    drive = torch_fly_one_until if reference else fly_module.fly_one_until
    saved = fly_module.fly_one_until
    fly_module.fly_one_until = drive
    try:
        result = fly_module.fly_segment(context, params, words, column, row, never, model, masks)
    except (RequestRefused, NotFlyable) as error:
        return {"refused": f"{type(error).__name__}: {error}"}
    finally:
        fly_module.fly_one_until = saved
    verdict = result.verdict
    return {"refused": None, "outcome": verdict.outcome, "end_row": verdict.end_row,
            "crossing": verdict.crossing, "words": replay.word_results(verdict), "reached": result.reached_end,
            "glidepath_step": result.glidepath_step, "states": result.flown.states[0].numpy(),
            "cycles": int(result.flown.commands.shape[1]), "fly_s": result.fly_s}


def compare(ours: dict[str, Any], theirs: dict[str, Any], chart) -> dict[str, Any]:
    """What differs between the two answers (empty: nothing but round-off), and how far apart the states are."""
    if ours["refused"] or theirs["refused"]:
        return {"differs": [] if ours["refused"] == theirs["refused"] else ["refusal"], "refused": True}
    differs = [key for key in ("outcome", "end_row", "words", "reached", "glidepath_step", "cycles") if ours[key] != theirs[key]]
    a, b = ours["crossing"], theirs["crossing"]
    if (a is None) != (b is None) or (a is not None and a["runway_index"] != b["runway_index"]):
        differs.append("crossing")
    rows = min(len(ours["states"]), len(theirs["states"]))
    s, t = ours["states"][:rows], theirs["states"][:rows]
    finite = np.isfinite(s).all(axis=1) & np.isfinite(t).all(axis=1)
    e1, n1 = chart(s[finite])
    e2, n2 = chart(t[finite])
    return {"differs": differs, "refused": False,
            "horizontal_m": float(np.max(np.hypot(e1 - e2, n1 - n2))) if finite.any() else 0.0,
            "vertical_m": float(np.max(np.abs(s[finite, 2] - t[finite, 2]))) if finite.any() else 0.0,
            "crossing_m": (None if a is None or b is None else
                           max(abs(a["cross_m"] - b["cross_m"]), abs(a["height_m"] - b["height_m"]))),
            "fly_s": (ours["fly_s"], theirs["fly_s"]), "cycles": ours["cycles"]}


def airport_chart(geometry):
    frame = geometry.frame

    def chart(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return frame.horizontal_from_latlon(states[:, 0], states[:, 1])
    return chart


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--airports-root", type=Path, default=COMPARISON_AIRPORTS_ROOT)
    parser.add_argument("--set", default="instruction_v3_day_split")
    parser.add_argument("--airport", action="append", default=None)
    parser.add_argument("--models", nargs="*", default=list(MODELS), help="overlays by name and round, e.g. landing_r01")
    parser.add_argument("--limit", type=int, default=0, help="flights an airport (0: every one; a smoke run)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    backend = AutopilotSegmentBackend(airports_root=args.airports_root)
    rows, started = [], time.perf_counter()
    for airport in args.airport or AIRPORTS:
        artefact, split, sample = backend.training_set(airport, args.set)
        directory, params, record, words = backend.executor_for(artefact)
        flights = sample["flights"][: args.limit or None]
        overlays = {entry["id"]: entry for entry in backend._json(args.airports_root / airport / "training" / "overlays.json")["overlays"]
                    if entry["base"] == args.set and entry["kind"] == "prior-generation"}
        generations = {name: backend._json(args.airports_root / airport / "training" / entry["file"])
                       for name in args.models for oid, entry in overlays.items() if oid.startswith(f"generation_{name}_")}
        for item in flights:
            try:
                context, _kept = backend.flight(artefact, split, sample, item["datasetId"], words)
            except NotFlyable:
                continue
            chart = airport_chart(context.geometry)
            grid = context.reading.words
            for row, column in zip(*np.nonzero(grid != UNCHANGED)):
                ours = answer(context, params, words, int(column), int(row), reference=False)
                theirs = answer(context, params, words, int(column), int(row), reference=True)
                rows.append({"airport": airport, "flight": item["flightKey"], "sentence": "truth", "column": COLUMNS[column],
                             "row": int(row), **compare(ours, theirs, chart)})
            for name, generation in generations.items():
                entry = [f for f in generation["flights"] if f["flightKey"] == item["flightKey"]]
                if not entry or not entry[0]["flown"]:
                    continue
                for flown_sample in entry[0]["samples"]:
                    record_ = {"overlayId": generation["overlayId"], "sample": flown_sample["sample"],
                               "firstRow": generation["generation"]["firstPredictedRow"], "rows": flown_sample["rows"],
                               "events": flown_sample["events"],
                               "procedureMasks": generation["generation"]["procedureMasks"]}
                    model = model_sentence(record_, words, len(context.geometry.candidates), len(context.reading.words),
                                           params.timeout_factor)
                    masks = backend.procedure_masks(artefact, model.procedure_masks)
                    for event in sorted(flown_sample["events"], key=lambda e: (e["row"], e["column"]), reverse=True):
                        ours = answer(context, params, words, event["column"], event["row"], model, masks, reference=False)
                        theirs = answer(context, params, words, event["column"], event["row"], model, masks, reference=True)
                        row_ = {"airport": airport, "flight": item["flightKey"], "sentence": f"{name}#{flown_sample['sample']}",
                                "column": COLUMNS[event["column"]], "row": event["row"], **compare(ours, theirs, chart)}
                        rows.append(row_)
                        if not row_["refused"]:
                            break
        print(f"{airport}: {len(rows)} segments so far, {time.perf_counter() - started:.0f} s", flush=True)
    flown = [r for r in rows if not r["refused"]]
    differing = [r for r in rows if r["differs"]]
    ours_s = [r["fly_s"][0] for r in flown]
    theirs_s = [r["fly_s"][1] for r in flown]
    summary = {
        "segments": len(rows), "flown": len(flown), "refused_by_both": len(rows) - len(flown) - sum(r["differs"] == ["refusal"] for r in rows),
        "differing": len(differing),
        "horizontal_m_max": max((r["horizontal_m"] for r in flown), default=0.0),
        "vertical_m_max": max((r["vertical_m"] for r in flown), default=0.0),
        "crossing_m_max": max((r["crossing_m"] for r in flown if r["crossing_m"] is not None), default=0.0),
        "horizontal_m_p99": float(np.percentile([r["horizontal_m"] for r in flown], 99)) if flown else 0.0,
        "fly_s_single": {"total": sum(ours_s), "median": float(np.median(ours_s)) if ours_s else 0.0},
        "fly_s_torch": {"total": sum(theirs_s), "median": float(np.median(theirs_s)) if theirs_s else 0.0},
        "cycles_total": sum(r["cycles"] for r in flown),
    }
    write_json_atomic(args.out / "check.json", {"written_utc": utc_now(), "set": args.set, "models": args.models,
                                                "limit": args.limit, "summary": summary,
                                                "differing": differing,
                                                "segments": [{k: v for k, v in r.items() if k != "fly_s"} for r in rows]})
    print(json.dumps(summary, indent=1))
    for r in differing[:40]:
        print("DIFFERS", r["airport"], r["flight"], r["sentence"], r["column"], r["row"], r["differs"],
              {k: r.get(k) for k in ("horizontal_m", "vertical_m")})


if __name__ == "__main__":
    main()
