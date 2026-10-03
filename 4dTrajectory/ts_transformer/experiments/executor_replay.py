"""Executor E8: the replay readout (design §14.2 A6; executor design §11) — the labelled sentences flown and judged.

Who (§11, `replay.draw`): every labelled flight of ``--split`` (or ``--per-airport`` of each airport, a seeded
sample) whose identified type publishes an approach speed; flights on their own type's dynamics and flights on a
stand-in's are flown and reported apart; the rest are counted. Each sentence is put on the row interval
``--row-interval-s`` (2 s, or 4 or 8 s: design §4.8, D11, D25) and flown from its first row on that grid. Each airport is
flown in chunks of ``--chunk`` flights, judged (`autopilot.judge`), and written as control-path prediction records
(`build_prediction_record` / `write_batch`: the dense flown states from the first row, the thrust-fraction schedule
resolved to newtons by the contract's own law) that the evaluation package grades; its verdict is joined by
``flight_key`` to the observed flight's own verdict, graded HERE by this evaluation code over the drawn flights' own
observed records (`observed_verdicts`: the harvest's record files linked read-only under ``--out/observed/<ICAO>``).

The readout (`readout_table`), per group, airport (and pooled), stratum (`instructions.readout`: straight-in / vectored)
and kind of sentence (with a go-around or not): each outcome, the words inside their envelopes per column with the
envelopes' widths, the decision-altitude checks, and of the flights whose observed track passes evaluation the share
whose replay passes too. NO CRITERION IS READ (design D7): the user sets them after the design is settled.

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
from ts_transformer.autopilot import replay
from ts_transformer.autopilot.executor import Flown
from ts_transformer.autopilot.judge import CROSSINGS, Outcome, Verdict
from ts_transformer.autopilot.plant import EXECUTOR_DYNAMICS
from ts_transformer.data.channels import channels_from_states
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.data.lateral_eligibility import default_evaluation_report_path
from ts_transformer.inference.export import build_prediction_record, observed_series_metrics, write_batch
from ts_transformer.inference.forecast import Forecast
from ts_transformer.instructions.artefact import SPLITS
from ts_transformer.instructions.readout import STRATA, flight_record
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.outputs.envelope import control_contract
from ts_transformer.repo_layout import REPO_ROOT, arrival_manifest_path, git_state

#: The name the records carry as their predictor, and the horizon they were flown over.
PREDICTOR = "executor"
HORIZON = "sentence"
#: v6 (two-tier v4): the outcomes of design §5.8 (`unstable_at_minimums`, the decision-altitude check on a crossing), the
#: row interval, the sentences with a go-around apart, no gate.
REPLAY_SCHEMA = "ts-executor-replay-v6"
#: The kinds of sentence the readout reads apart.
KINDS = ("without go-around", "with go-around")


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


def readout_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per group, airport (and "all"), stratum (and "all") and kind of sentence (and "all"): the outcomes, the words inside
    their envelopes per column, the decision-altitude checks and the evaluation pairing (module docstring)."""
    table: dict[str, Any] = {}
    cells: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for airport in (row["airport"], "all"):
            for stratum in (row["stratum"], "all"):
                for kind in (row["kind"], "all"):
                    cells[(row["group"], airport, stratum, kind)].append(row)
    for (group, airport, stratum, kind), members in sorted(cells.items()):
        judged = [w for r in members if r["words"] is not None for w in r["words"]]
        paired = [r for r in members if r["observed_verdict"] == "pass"]
        pairs = [r["heading_words_told_with_a_skipped_word"] for r in members]
        decisions = [r["crossing"]["decision"] for r in members if r["crossing"] is not None and "decision" in r["crossing"]]
        table.setdefault(group, {}).setdefault(airport, {}).setdefault(stratum, {})[kind] = {
            "flights": len(members), "outcomes": dict(Counter(r["outcome"] for r in members).most_common()),
            "landed": _share(sum(r["outcome"] == "landed" for r in members), len(members)),
            "words_judged": len(judged), "words_inside": _share(sum(ok for _, ok in judged), len(judged)),
            "words_inside_by_column": {column: _share(sum(ok for c, ok in judged if c == column),
                                                      sum(c == column for c, _ in judged))
                                       for column in replay.JUDGED},
            "heading_words_told_with_a_skipped_word": {"judged": sum(p["judged"] for p in pairs),
                                                       "inside": sum(p["inside"] for p in pairs)},
            "heading_words_not_judged": sum(r["heading_words_not_judged"] for r in members),
            "words_not_reached": sum(r["words_not_reached"] for r in members),
            "flights_with_unjudged_words": sum(r["words"] is None for r in members),
            "decision_checks": {"n": len(decisions), "no_da_point": sum(d is None for d in decisions),
                                "passed": sum(d is not None and d["passed"] for d in decisions)},
            "observed_passes": len(paired),
            "replay_passes_where_observed_passes": _share(sum(r["replay_verdict"] == "pass" for r in paired), len(paired)),
        }
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
                per_flight: Callable[[replay.Batch, Flown, list[Verdict]], list[dict[str, Any]]] | None = None,
                ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fly and judge one airport's flights, write their records (``checkpoint`` and ``extra_summary`` name what flew
    them), grade them; one row each, and the report. ``per_flight`` adds columns to the rows, read off each chunk's
    flown tracks (one dict per flight, in the chunk's order)."""
    rows, predictions, metrics = [], [], []
    for start in range(0, len(members), chunk):
        part = replay.subset(batch, members[start: start + chunk])
        flown, verdicts = replay.fly_batch(part, params, words, device=device)
        inputs = part.inputs(device)
        aligned = replay.flight_alignment(part, flown, verdicts)
        more = [{} for _ in verdicts] if per_flight is None else per_flight(part, flown, verdicts)
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
                "airport": reading.airport, "group": part.groups[j], "stratum": flight_record(reading)["stratum"],
                "kind": KINDS[part.sentences[j].go_arounds > 0], "first_row": part.sentences[j].first_row,
                "outcome": verdict.outcome, "flew_the_sentence": verdict.flew_the_sentence,
                "crossing": verdict.crossing, "words": None if counted is None else counted[0],
                "heading_words_not_judged": 0 if counted is None else counted[1],
                "heading_words_told_with_a_skipped_word": replay.clock_pairs(verdict),
                "words_not_reached": 0 if verdict.words is None else verdict.words["not_reached"],
                "words_superseded_before_flown": 0 if verdict.words is None else verdict.words["superseded_before_flown"],
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--per-airport", type=int, default=0, help="0: every labelled flight of the split")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--row-interval-s", type=float, default=2.0,
                        help="the sentences' row interval: 2, 4 or 8 s (design §4.8, D25)")
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
    params, record, words = replay.open_executor(executor, instructions)
    spec = words.spec
    started = time.perf_counter()
    batch = replay.draw(instructions, args.split, spec, words, per_airport=args.per_airport, seed=args.seed,
                        groups=(replay.OWN, replay.STAND_IN), row_interval_s=args.row_interval_s)
    print(f"{batch.drawn['flights']} {args.split} flights ({batch.drawn['by_group']}; not flown "
          f"{batch.drawn['excluded']}), {time.perf_counter() - started:.0f}s", flush=True)

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
                                                        "row_interval_s": args.row_interval_s})
        require_same_grading(graded, observed_reports[airport], airport)
        for row in flown_rows:
            row["observed_verdict"] = observed[row["flight_key"]]
        rows += flown_rows
        landed = sum(r["outcome"] == "landed" for r in flown_rows)
        print(f"  {airport}: {len(flown_rows)} flown, {landed} landed, {time.perf_counter() - started:.0f}s", flush=True)

    table = readout_table(rows)
    write_json_atomic(out / "replay.json", {
        "schema": REPLAY_SCHEMA, "written_utc": utc_now(), "split": args.split, "row_interval_s": args.row_interval_s,
        "executor_spec_sha256": record["sha256"], "vocabulary_spec_sha256": spec.sha256, "params": asdict(params),
        "envelope_widths": envelope_widths(words), "drawn": batch.drawn, "strata": list(STRATA), "kinds": list(KINDS),
        "readout": table, "flights": rows, "git": git_state(), "elapsed_s": time.perf_counter() - started})
    for group, airports in table.items():
        for airport, strata in airports.items():
            for stratum, kinds in strata.items():
                for kind, cell in kinds.items():
                    shares = "  ".join(f"{name} {cell[key] if cell[key] is None else round(cell[key], 3)}"
                                       for name, key in (("landed", "landed"), ("words", "words_inside"),
                                                         ("eval", "replay_passes_where_observed_passes")))
                    print(f"  {group:18s} {airport:5s} {stratum:12s} {kind:17s} n={cell['flights']:5d}  {shares}  "
                          f"{cell['outcomes']}")
    print(f"→ {out / 'replay.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
