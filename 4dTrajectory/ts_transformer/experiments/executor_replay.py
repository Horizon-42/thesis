"""Executor E8: the replay readout (vocabulary §12.1 A6; executor design §11) — the labelled sentences flown and judged.

Who (§11, `replay.draw`): every labelled flight of ``--split`` (or ``--per-airport`` of each airport, a seeded
sample) whose identified type publishes an approach speed; flights on their own type's dynamics and flights on a
stand-in's are flown and reported apart; the rest are counted. Each sentence is put on the row interval
``--row-interval-s`` (2 s, or 4 or 8 s: vocabulary §4.8, D11, D25) and flown from its first row on that grid. Each airport is
flown in chunks of ``--chunk`` flights, judged (`autopilot.judge`), and written as control-path prediction records
(`build_prediction_record` / `write_batch`: the dense flown states from the first row, the thrust-fraction schedule
resolved to newtons by the contract's own law) that the evaluation package grades; its verdict is joined by
``flight_key`` to the observed flight's own verdict, graded HERE by this evaluation code over the drawn flights' own
observed records (`observed_verdicts`: the harvest's record files linked read-only under ``--out/observed/<ICAO>``).

The readout (`readout_table`), per group, airport (and pooled), stratum (`instructions.readout`: straight-in / vectored)
and kind of sentence (with a go-around or not): each outcome, the words inside their envelopes per column with the
envelopes' widths, the decision-altitude checks, and of the flights whose observed track passes evaluation the share
whose replay passes too. NO CRITERION IS READ (design D7): the user sets them after the design is settled.

With ``--closed-loop`` it flies the artefact's CLOSED-LOOP sentences instead (vocabulary §4.9, D32; written by
`instruction_closed_loop`, refused unless the code on disk reads their reference as it was read): each from its first
predicted step, on its own rows, from the observed state there. Every flight must fly its stored states again (within the executor conformance's bound);
each row adds the largest |e_y| and |e_h| against the observed path and the correction words for each column, and each
cell of the readout the flights that left the observed path by more than `LEFT_THE_PATH_M` and the correction words for
each sentence. The labelled flights without a closed-loop sentence are counted.

Open or closed loop, each row also counts its sentence's speed words (other than "unspecified") and the largest distance
along the observed path between the flown aircraft and the observed aircraft of the same time before "unspecified"
(`along_columns`), and each cell their mean, percentiles and the flights farther than `FAR_ALONG_M` (vocabulary §9.8, D43).

The VAL replay waits for the user's go-ahead and runs from a clean tree. Development runs use train.

    python run_ts.py executor_replay --split train --per-airport 20 --row-interval-s 2 \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/<artefact> \\
        --executor 4dTrajectory/outputs/POOLED/executor/<name> --out <new directory>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from aerodynamic_model.common import GeodeticState
from ts_transformer.autopilot import closed_loop, replay
from ts_transformer.autopilot.conformance import STATE_BOUND_M
from ts_transformer.autopilot.judge import flown_track
from ts_transformer.autopilot.executor import Flown
from ts_transformer.autopilot.judge import CROSSINGS, Outcome, Verdict
from ts_transformer.autopilot.plant import EXECUTOR_DYNAMICS
from ts_transformer.autopilot.spec import params_sha256
from ts_transformer.data.channels import channels_from_states
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.data.lateral_eligibility import default_evaluation_report_path
from ts_transformer.inference.export import build_prediction_record, observed_series_metrics, write_batch
from ts_transformer.inference.forecast import Forecast
from ts_transformer.instructions.artefact import (
    SPLITS, ClosedLoopSentence, closed_loop_path, closed_loop_sentences,
)
from ts_transformer.instructions.labeller.interval import in_force
from ts_transformer.instructions.words import COLUMNS, SPEED, UNCHANGED, Words
from ts_transformer.instructions.readout import KINDS, STRATA, stratum
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.outputs.envelope import control_contract
from ts_transformer.repo_layout import REPO_ROOT, arrival_manifest_path, git_state

#: The name the records carry as their predictor, and the horizon they were flown over.
PREDICTOR = "executor"
HORIZON = "sentence"
#: v6 (two-tier v4): the outcomes of vocabulary §5.8 (`unstable_at_minimums`, the decision-altitude check on a crossing), the
#: row interval, the sentences with a go-around apart, no gate.
#: v8 (A29, D73): the checks the run ran before its work (``checks``: the labeller's, the executor's and, for closed-loop
#: sentences, the closed loop's largest differences), as information.
REPLAY_SCHEMA = "ts-executor-replay-v8"
#: A closed-loop flight that goes farther than this from its observed path, laterally, LEFT the path: a reading of the
#: A9 smoke build (vocabulary §9.7) that A10 compares against (§12.1 A10), not a criterion (D7).
LEFT_THE_PATH_M = 300.0
#: A flight farther than this along the path from the observed aircraft of the same time, before "unspecified": the
#: reading of vocabulary §9.8 that A11 compares against (§12.1 A11), not a criterion (D7).
FAR_ALONG_M = 1000.0


def executor_forecast(flown: Flown, index: int, verdict: Outcome | Verdict, inputs: Any,
                      series: FlightSeries, *, anchor: int = 0) -> Forecast:
    """Flight ``index``'s flown track as a control-path forecast from the series row ``anchor`` it was flown from (a
    replay: row 0; free generation: the first predicted step): the states at every cycle to the one its outcome is
    read at (before it, for a dynamics failure: no state of the failure enters the record), the schedule in the
    plant's contract, and its newtons resolved by that contract's own law."""
    end = verdict.end_row - 1 if verdict.outcome == "dynamics_failure" else verdict.end_row
    dt = flown.cycle_s
    states = flown.states[index, : end + 1]
    commands = flown.commands[index, :end]
    contract = EXECUTOR_DYNAMICS.control_thrust_parameterization
    newtons = control_contract(contract).law.geodetic_controls(
        states[:-1].unsqueeze(0), commands.unsqueeze(0), max_thrust_n=inputs.max_thrust_n[index: index + 1],
        initial_geodetic_states=states[:1], aero_params=inputs.aero_params[index: index + 1])[0]
    offsets = np.arange(1, end + 1) * dt
    geodetic = states[1:].cpu().numpy().astype(np.float64)
    _, values = channels_from_states([(float(t), GeodeticState(*map(float, row))) for t, row in zip(offsets, geodetic)],
                                     series.frame)
    return Forecast(
        times=float(series.times[anchor]) + offsets, values=values, normalized_progress=offsets / offsets[-1],
        anchor=anchor,
        final_time_s=float(offsets[-1]), predicted_final_time_s=float(offsets[-1]), horizon_mode=HORIZON, passes=1,
        truncated_at_threshold=verdict.outcome in CROSSINGS, horizon_capped=verdict.outcome == "timeout",
        sample_durations_s=np.full(end, dt), segment_durations_s=np.full(end, dt),
        controls=newtons.cpu().numpy().astype(np.float64), commands=commands.cpu().numpy().astype(np.float64),
        control_parameterization=contract, geodetic_values=geodetic,
        prediction_output=EXECUTOR_DYNAMICS.prediction_output)


def _share(passed: int, total: int) -> float | None:
    return passed / total if total else None


def readout_table(rows: list[dict[str, Any]], *, closed_loop_rows: bool) -> dict[str, Any]:
    """Per group, airport (and "all"), stratum (and "all") and kind of sentence (and "all"): the outcomes, the words inside
    their envelopes per column, the decision-altitude checks and the evaluation pairing (module docstring); of
    closed-loop sentences (``closed_loop_rows``: the rows carry `closed_loop_columns`) also the flights that left the
    observed path (`LEFT_THE_PATH_M`) and the correction words for each sentence, per column."""
    table: dict[str, Any] = {}
    cells: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for airport in (row["airport"], "all"):
            for part in (row["stratum"], "all"):
                for kind in (row["kind"], "all"):
                    cells[(row["group"], airport, part, kind)].append(row)
    for (group, airport, part, kind), members in sorted(cells.items()):
        judged = [w for r in members if r["words"] is not None for w in r["words"]]
        paired = [r for r in members if r["observed_verdict"] == "pass"]
        decisions = [r["crossing"]["decision"] for r in members if r["crossing"] is not None and "decision" in r["crossing"]]
        table.setdefault(group, {}).setdefault(airport, {}).setdefault(part, {})[kind] = {
            "flights": len(members), "outcomes": dict(Counter(r["outcome"] for r in members).most_common()),
            "landed": _share(sum(r["outcome"] == "landed" for r in members), len(members)),
            "words_judged": len(judged), "words_inside": _share(sum(ok for _, ok in judged), len(judged)),
            "words_inside_by_column": {column: _share(sum(ok for c, ok in judged if c == column),
                                                      sum(c == column for c, _ in judged))
                                       for column in replay.JUDGED},
            "heading_words_not_judged": sum(r["heading_words_not_judged"] for r in members),
            "words_not_reached": sum(r["words_not_reached"] for r in members),
            "flights_with_unjudged_words": sum(r["words"] is None for r in members),
            "decision_checks": {"n": len(decisions), "no_da_point": sum(d is None for d in decisions),
                                "passed": sum(d is not None and d["passed"] for d in decisions)},
            "observed_passes": len(paired),
            "replay_passes_where_observed_passes": _share(sum(r["replay_verdict"] == "pass" for r in paired), len(paired)),
        }
        along = [r["largest_along_m"] for r in members if r["largest_along_m"] is not None]
        table[group][airport][part][kind]["along_the_path"] = {
            "speed_words_per_sentence": sum(r["speed_words"] for r in members) / len(members),
            "flights": len(along), "largest_along_m": _percentiles(along),
            "farther_than_1_km": sum(1 for a in along if a > FAR_ALONG_M)}
        if closed_loop_rows:
            table[group][airport][part][kind]["closed_loop"] = {
                "left_the_path": sum(1 for r in members if r["largest_lateral_m"] > LEFT_THE_PATH_M),
                "correction_words_per_sentence": {column: sum(r["correction_words"][column] for r in members)
                                                  / len(members) for column in COLUMNS}}
    return table


def envelope_widths(words: Any) -> dict[str, Any]:
    """The envelopes' widths the words are judged in (design principle 6: a containment rate is given with them)."""
    spec = words.spec
    return {"heading_tolerance_deg": spec.heading_tolerance_deg, "heading_lead_s": spec.heading_lead_s,
            "level_band_m": {f"{level:g}": float(band) for level, band in zip(words.altitude_levels, words.altitude_tolerances)},
            "speed_tolerance_mps": spec.speed_tolerance_mps}


def evaluate_records(records: Path) -> dict[str, Any]:
    report = records / "evaluation_report.json"
    subprocess.run([sys.executable, "-m", "evaluation", "--input", str(records), "--output", str(report)],
                   cwd=REPO_ROOT, check=True)
    return json.loads(report.read_text(encoding="utf-8"))


def observed_verdicts(airport: str, flight_keys: set[str], directory: Path) -> dict[str, Any]:
    """The observed records of ``flight_keys`` graded by this evaluation code: the harvest's own record files linked
    (read-only) into ``directory/records``, rostered by a ``summary.json`` cut from the harvest's, and evaluated
    there. Returns the evaluation report."""
    approach = default_evaluation_report_path(arrival_manifest_path(airport)).parent
    summary = json.loads((approach / "summary.json").read_text(encoding="utf-8"))
    if summary["subject"] != "observed":
        raise ValueError(f"{approach / 'summary.json'} is not an observed batch")
    rows = [row for row in summary["results"] if row["flight_key"] in flight_keys]
    missing = flight_keys - {row["flight_key"] for row in rows}
    if missing:
        raise ValueError(f"{airport}: {len(missing)} flight(s) have no observed record, e.g. {sorted(missing)[:3]}")
    (directory / "records").mkdir(parents=True)
    for row in rows:
        (directory / row["eval_file"]).symlink_to(approach / row["eval_file"])
    write_json_atomic(directory / "summary.json", {**summary, "total": len(rows), "results": rows})
    return evaluate_records(directory)


def require_same_grading(replayed: dict[str, Any], observed: dict[str, Any], airport: str) -> None:
    """Gate 3 pairs two reports: the same evaluation schema and methodology, or they are not one grading."""
    for key in ("schema_version", "methodology"):
        if replayed[key] != observed[key]:
            raise ValueError(f"{airport}: the replay's evaluation {key} differs from the observed report's — grade "
                             "the observed records again with this evaluation code before pairing")


def fly_airport(batch: replay.Batch, members: list[int], params: Any, words: Any, *, chunk: int, device: torch.device,
                records: Path, split: str, checkpoint: str, extra_summary: dict[str, Any],
                per_flight: Callable[[replay.Batch, Flown, list[Verdict]], list[dict[str, Any]]],
                ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fly and judge one airport's flights, write their records (``checkpoint`` and ``extra_summary`` name what flew
    them), grade them; one row each, and the report. ``per_flight`` adds columns to the rows, read off each chunk's
    flown tracks (one dict per flight, in the chunk's order)."""
    rows, predictions, metrics = [], [], []
    for start in range(0, len(members), chunk):
        part = replay.subset(batch, members[start: start + chunk])
        flown, verdicts = replay.fly_batch(part, params, words, device=device)
        inputs = part.inputs(params.start_rule, device)
        aligned = replay.flight_alignment(part, flown, verdicts)
        more = per_flight(part, flown, verdicts)
        for j, verdict in enumerate(verdicts):
            counted = replay.word_results(verdict)
            series = part.series[j]
            # a dynamics failure in the first cycle leaves no state to record (a stand-in's airframe below its
            # observed speed): no record, and the row says so
            recorded = not (verdict.outcome == "dynamics_failure" and verdict.end_row <= 1)
            if recorded:
                forecast = executor_forecast(flown, j, verdict, inputs, series, anchor=part.sentences[j].first_row)
                predictions.append(build_prediction_record(series, forecast, index=start + j, model_name=PREDICTOR,
                                                           horizon_mode=HORIZON, split=split))
                metrics.append(observed_series_metrics(series, forecast))
            reading = part.readings[j]
            rows.append({
                "dataset_id": reading.dataset_id, "flight_key": series.scenario.source["flight_key"],
                "airport": reading.airport, "group": part.groups[j], "stratum": stratum(reading),
                "kind": KINDS[part.sentences[j].go_arounds > 0], "first_row": part.sentences[j].first_row,
                "outcome": verdict.outcome, "flew_the_sentence": verdict.flew_the_sentence,
                "crossing": verdict.crossing, "words": None if counted is None else counted[0],
                "heading_words_not_judged": 0 if counted is None else counted[1],
                "words_not_reached": 0 if verdict.words is None else verdict.words["not_reached"],
                "limits": verdict.limits, "recorded": recorded, **aligned[j], **more[j]})
        del flown, verdicts
    write_batch(predictions, output_dir=records, config_dict={"model": PREDICTOR, "horizon_mode": HORIZON,
                                                              "prediction_output": EXECUTOR_DYNAMICS.prediction_output,
                                                              "executor_params": asdict(params)},
                flight_metrics=metrics, checkpoint=checkpoint, split=split, extra_summary=extra_summary)
    graded = evaluate_records(records)
    by_key = {row["flight_key"]: row for row in graded["trajectories"]}
    for row in rows:
        row["replay_verdict"] = by_key[row["flight_key"]]["verdict"] if row["recorded"] else "no record: failed at once"
    return rows, graded


def _percentiles(values: list[float]) -> dict[str, float] | None:
    return {f"p{q}": float(np.percentile(values, q)) for q in (50, 90)} if values else None


def along_columns(words: Words) -> Callable[[replay.Batch, Flown, list[Verdict]], list[dict[str, Any]]]:
    """The rows' speed columns (vocabulary §9.8, §12.1 A11), on the Δ rows of the flown sentence before "unspecified" is
    first in force (a closed-loop sentence's from its first predicted step): its speed words, and the largest distance
    along the observed path between the flown aircraft (its matched point, the closed loop's forward search,
    `closed_loop.ObservedPath`) and the observed aircraft of the same time while both fly — the failed state of a
    dynamics failure left out, as the records leave it (None: no such row)."""
    def columns(part: replay.Batch, flown: Flown, verdicts: list[Verdict]) -> list[dict[str, Any]]:
        out = []
        every = int(round(part.row_interval_s / words.spec.step_s))
        cycles = int(round(part.row_interval_s / flown.cycle_s))
        for j, verdict in enumerate(verdicts):
            grid, signals = part.sentences[j].grid, part.signals[j]
            held = in_force(grid)[:, SPEED] == words.speed_unspecified
            before = int(np.argmax(held)) if held.any() else len(grid)
            said = int((grid[:before, SPEED] != UNCHANGED).sum())
            span = len(part.readings[j].words) - part.sentences[j].first_row
            end = verdict.end_row - 1 if verdict.outcome == "dynamics_failure" else verdict.end_row
            rows = [k for k in range(before) if k * every < span and k * cycles <= end]
            largest = None
            if rows:
                track = flown_track(flown.states[j, : rows[-1] * cycles + 1].cpu().numpy(), part.geometries[j])
                path = closed_loop.ObservedPath(signals.e_m[:span], signals.n_m[:span], np.zeros(span), 0)
                largest = max(abs(path.match(float(track["e"][k * cycles]), float(track["n"][k * cycles]), 0.0).along_m
                                  - float(path.along_rows[k * every])) for k in rows)
            out.append({"speed_words": said, "largest_along_m": largest})
        return out

    return columns


def closed_loop_columns(stored: dict[int, ClosedLoopSentence], step_s: float
                        ) -> Callable[[replay.Batch, Flown, list[Verdict]], list[dict[str, Any]]]:
    """The rows' closed-loop columns: each flight's largest |e_y| and |e_h| and its correction words per column — after
    checking that it flew its stored states again (each 2 s row's position and height within `STATE_BOUND_M`, D51) to
    its stored outcome (D74)."""
    def columns(part: replay.Batch, flown: Flown, verdicts: list[Verdict]) -> list[dict[str, Any]]:
        out = []
        for j, index in enumerate(part.indices):
            sentence, withheld = stored[index].rows, stored[index].withheld
            flown_states = sentence.flown_states
            rows = np.arange(len(flown_states)) * int(round(step_s / flown.cycle_s))
            track = flown_track(flown.states[j, : rows[-1] + 1].cpu().numpy(), part.geometries[j])
            again = np.column_stack([track["e"][rows], track["n"][rows], track["height"][rows]])
            apart = float(np.abs(again - flown_states[:, :3]).max())
            if apart > STATE_BOUND_M:
                raise ValueError(f"{part.signals[j].dataset_id}: flown {apart:.3g} m from its closed-loop states")
            if verdicts[j].outcome != withheld.outcome:
                raise ValueError(f"{part.signals[j].dataset_id}: flown again to {verdicts[j].outcome}, stored "
                                 f"{withheld.outcome}")
            out.append({"largest_lateral_m": closed_loop.largest_m(withheld.lateral_m),
                        "largest_vertical_m": closed_loop.largest_m(withheld.vertical_m),
                        "uncorrected_lateral_m": closed_loop.uncorrected_m(withheld.lateral_m,
                                                                           withheld.uncorrectable[:, 0]),
                        "uncorrected_vertical_m": closed_loop.uncorrected_m(withheld.vertical_m,
                                                                            withheld.uncorrectable[:, 1]),
                        "correction_words": dict(zip(COLUMNS, sentence.correction.sum(axis=0).tolist()))})
        return out

    return columns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--per-airport", type=int, default=0, help="0: every labelled flight of the split")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--row-interval-s", type=float, default=2.0,
                        help="the sentences' row interval: 2, 4 or 8 s (vocabulary §4.8, D25)")
    parser.add_argument("--closed-loop", action="store_true",
                        help="fly the artefact's closed-loop sentences from the first predicted step (module docstring)")
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    instructions = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    if args.split != "train" and git_state()["dirty"]:
        parser.error(f"the {args.split} replay (stage 4) runs from a clean tree")
    # the checks every reader runs (D73): the labeller's and the executor's, and for closed-loop sentences the closed loop's
    params, record, words = (closed_loop.require_conforming_closed_loop(instructions, executor) if args.closed_loop
                             else replay.open_executor(executor, instructions))
    spec = words.spec
    started = time.perf_counter()
    batch = replay.draw(instructions, args.split, spec, words, per_airport=args.per_airport, seed=args.seed,
                        groups=(replay.OWN, replay.STAND_IN), row_interval_s=args.row_interval_s)
    print(f"{batch.drawn['flights']} {args.split} flights ({batch.drawn['by_group']}; not flown "
          f"{batch.drawn['excluded']}), {time.perf_counter() - started:.0f}s", flush=True)
    along = along_columns(words)
    per_flight = along
    if args.closed_loop:
        path = closed_loop_path(instructions, args.split, args.row_interval_s)
        if not path.exists():
            parser.error(f"{path} does not exist: the closed-loop reading wrote no sentence there "
                         f"({path.parent / 'summary.json'} says why)")
        with np.load(path) as data:                           # the parameters only: the reader loads the file
            flown_by = str(data["executor_params_sha256"])
        if flown_by != params_sha256(params):
            parser.error(f"the closed-loop sentences were flown by executor parameters {flown_by[:12]}, {executor} holds "
                         f"{params_sha256(params)[:12]}")
        stored = closed_loop_sentences(instructions, args.split, args.row_interval_s, spec)
        batch, missing = closed_loop.replay_batch(batch, stored, words)
        batch.drawn["without_a_closed_loop_sentence"] = missing
        print(f"{len(batch.sentences)} closed-loop sentences flown from the first predicted step, {missing} flights "
              f"without one", flush=True)
        closed = closed_loop_columns(stored, words.spec.step_s)

        def per_flight(part: replay.Batch, flown: Flown, verdicts: list[Verdict]) -> list[dict[str, Any]]:
            return [{**a, **c} for a, c in zip(along(part, flown, verdicts), closed(part, flown, verdicts))]

    out.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    by_airport: dict[str, list[int]] = defaultdict(list)
    for j, flight in enumerate(batch.signals):
        by_airport[flight.airport].append(j)
    observed_reports = {airport: observed_verdicts(airport, {batch.series[j].scenario.source["flight_key"] for j in members},
                                                   out / "observed" / airport)
                        for airport, members in sorted(by_airport.items())}
    for airport, members in sorted(by_airport.items()):
        observed = {row["flight_key"]: row["verdict"] for row in observed_reports[airport]["trajectories"]}
        flown_rows, graded = fly_airport(batch, members, params, words, chunk=args.chunk,
                                         device=torch.device(args.device), records=out / "records" / airport,
                                         split=args.split, checkpoint=str(executor),
                                         extra_summary={"executor_spec_sha256": record["sha256"],
                                                        "row_interval_s": args.row_interval_s,
                                                        "closed_loop": args.closed_loop},
                                         per_flight=per_flight)
        require_same_grading(graded, observed_reports[airport], airport)
        for row in flown_rows:
            row["observed_verdict"] = observed[row["flight_key"]]
        rows += flown_rows
        landed = sum(r["outcome"] == "landed" for r in flown_rows)
        print(f"  {airport}: {len(flown_rows)} flown, {landed} landed, {time.perf_counter() - started:.0f}s", flush=True)

    table = readout_table(rows, closed_loop_rows=args.closed_loop)
    write_json_atomic(out / "replay.json", {
        "schema": REPLAY_SCHEMA, "written_utc": utc_now(), "split": args.split, "row_interval_s": args.row_interval_s,
        "closed_loop": args.closed_loop, "checks": record["checks"],
        "executor_spec_sha256": record["sha256"], "vocabulary_spec_sha256": spec.sha256, "params": asdict(params),
        "envelope_widths": envelope_widths(words), "drawn": batch.drawn, "strata": list(STRATA), "kinds": list(KINDS),
        "readout": table, "flights": rows, "git": git_state(), "elapsed_s": time.perf_counter() - started})
    for group, airports in table.items():
        for airport, strata in airports.items():
            for part, kinds in strata.items():
                for kind, cell in kinds.items():
                    shares = "  ".join(f"{name} {cell[key] if cell[key] is None else round(cell[key], 3)}"
                                       for name, key in (("landed", "landed"), ("words", "words_inside"),
                                                         ("eval", "replay_passes_where_observed_passes")))
                    loop_text = (f"  left the path {cell['closed_loop']['left_the_path']}, heading corrections a "
                                 f"sentence {cell['closed_loop']['correction_words_per_sentence']['heading']:.1f}"
                                 if args.closed_loop else "")
                    speeds = cell["along_the_path"]
                    along_text = (f"  speed words a sentence {speeds['speed_words_per_sentence']:.1f}, along the path "
                                  f"{speeds['largest_along_m']}, farther than 1 km {speeds['farther_than_1_km']}")
                    print(f"  {group:18s} {airport:5s} {part:12s} {kind:17s} n={cell['flights']:5d}  {shares}  "
                          f"{cell['outcomes']}{loop_text}{along_text}")
    print(f"→ {out / 'replay.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
