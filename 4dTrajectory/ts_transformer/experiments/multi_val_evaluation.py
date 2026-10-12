"""Stage D: a round's val windows flown again and graded by the evaluation package (the user, 2026-10-12: "用 evaluation
跑一下 r7 的 val 轨迹"; outline: quality belongs to the evaluation module, vocabulary §5.8).

THE FLIGHTS: round ``--round``'s model (`multi_train.round_model`) on the campaign's val windows, flown as
`multi_validation` flew them (`multi_train.selection_windows` of the val days, `multi_train.selection_readout`: the same
windows and random numbers, on ``--device`` in one process). The summed readout must be the val read's own
(``--val-read``'s ``readout.json``), or the run stops and writes nothing: the behaviour check on fixed inputs; its
windows are checked against the read's before they are flown. The val
claim is neither checked nor written: the read flies the windows of the val read that the claim spent, by the user's
order, and chooses nothing.

THE RECORDS (`aircraft_record`): each commanded aircraft of each window, its states (`WindowResult.states`: the 2 s rows
from its row 0 — observed before its first predicted step, flown from it — to its end) as an evaluation record of the
subject ``predicted`` without controls (the speed gate reads the declared 1 g): latitude and longitude from the airport
frame, the MSL height, ψ from the compass track, V and γ from the ground speed and the vertical rate (no wind,
`autopilot.frame`), the mass the executor flew it at (its rebuilt series' scenario, `autopilot.flights.flight_inputs`;
not the observed record's, which is the harvest's nominal 60 t for a type it does not resolve). Its runway: the one its crossing landed on (a landed aircraft's, a
silent one's too), else its observed runway; the target at that runway's threshold as the evaluation context gives it
(LTP + TCH; where no path is published, the two-tier candidate's TCH: the vertical gate is then indeterminate). Its
source: the observed record's, without the observed-only fields and its ``aircraft_type``, with the type the executor
flew (the scenario's, a stand-in's where one flies it) as ``dynamics_typecode``: the speed gate reads one type, the one
whose mass the states carry (`evaluation.speed_gate.TYPECODE_KEYS`). One record a
window and aircraft: a flight in several windows has a record in each.

THE OBSERVED: the same flights' observed records, graded by this evaluation code once a flight
(`executor_replay.observed_verdicts`).

WRITES, into a new directory: ``config.json``; ``records/<ICAO>/`` (the records, ``summary.json`` and the evaluation
report); ``observed/<ICAO>/`` (the observed records linked, their report); ``aircraft.jsonl`` (one line an aircraft:
the window, the outcome, the evaluation's verdict and gates, the observed verdict); ``readout.json`` (`readout_table`:
by airport, pooled, and by span); ``SHA256SUMS``; the directory read-only. From a clean checkout; the closed loop's
(D69) and the edge features' (D104) checks first. NO CRITERION IS READ (D7).

    python run_ts.py multi_val_evaluation --campaign 4dTrajectory/outputs/POOLED/multi/<campaign> --round <r> \\
        --val-read 4dTrajectory/outputs/POOLED/multi/<val read> --out <a new directory> [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from evaluation.cli import DEFAULT_CIFP, DEFAULT_CONFIG
from geokit import compass_bearing_to_math_enu_rad
from evaluation.context import contexts_for_airport
from evaluation.thresholds import AssessmentContext
from trajectory_data_process.harvest.airports import load_airport
from ts_transformer.autopilot.closed_loop import require_conforming_closed_loop
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.data.lateral_eligibility import default_evaluation_report_path
from ts_transformer.experiments.executor_replay import evaluate_records, observed_verdicts, require_same_grading
from ts_transformer.experiments.multi_train import (
    multi_settings_of, round_model, selection_readout, selection_windows, span_key, stage_d, window_record,
)
from ts_transformer.experiments.post_diagnose import seal
from ts_transformer.experiments.post_train import done_rounds, inputs_here, open_context, split_data
from ts_transformer.experiments.post_window_loop import WindowResult, checked_edges
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.prior.source import require_selection_of
from ts_transformer.repo_layout import REPO_ROOT, arrival_manifest_path, git_state

#: The format of the run's files (``config.json``, ``readout.json``).
VAL_EVALUATION_SCHEMA = "ts-multi-val-evaluation-v1"
#: The name the records carry as their predictor.
PREDICTOR = "two_tier_stage_d"
#: The observed record's source fields that a computed record may not carry (`evaluation.records`: a computed
#: subject's crossing is derived from its states) or that describe the observed track only.
OBSERVED_ONLY = ("crossing_span", "observed_threshold_event", "source_integrity", "mass_source", "baro_vnav_minima",
                 "tch_source", "vertical_source", "aircraft_type")
#: The evaluation's verdicts, in the readout's order.
VERDICTS = ("pass", "fail", "indeterminate")
GATES = ("lateral_result", "vertical_result", "speed_result")


def math_heading_rad(compass_deg: float) -> float:
    """A compass track (true north 0, clockwise, degrees) as the records' ψ (`geokit.compass_bearing_to_math_enu_rad`)."""
    return compass_bearing_to_math_enu_rad(math.radians(float(compass_deg)))


def record_states(rows: np.ndarray, geometry: AirportGeometry, step_s: float, mass_kg: float) -> list[dict]:
    """The records' timed states of ``rows`` (`STATE_COLUMNS` on the vocabulary's ``step_s`` rows from row 0, t = 0
    at row 0)."""
    lat, lon = geometry.frame.latlon_from_horizontal(rows[:, 0], rows[:, 1])
    ground, vertical = rows[:, 4], rows[:, 5]
    speed, gamma = np.hypot(ground, vertical), np.arctan2(vertical, ground)
    psi = [math_heading_rad(track) for track in rows[:, 3]]
    return [{"t": i * step_s, "lat": float(lat[i]), "lon": float(lon[i]), "alt": float(rows[i, 2]),
             "V": float(speed[i]), "psi": float(psi[i]), "gamma": float(gamma[i]), "m": mass_kg}
            for i in range(len(rows))]


def target_state(context: AssessmentContext, geometry: AirportGeometry, speed_mps: float, mass_kg: float
                 ) -> dict[str, float]:
    """The target at ``context``'s threshold: its position and course, its LTP + TCH (the candidate's TCH where the
    context publishes none), the candidate's glidepath, ``speed_mps`` (the observed target's) and ``mass_kg`` (the
    flown)."""
    candidate = geometry.candidates[geometry.candidate_index(context.runway)]
    altitude = context.desired_threshold_altitude_msl_m
    if altitude is None:
        altitude = candidate.elevation_m + candidate.vertical_path.crossing_height_m
    return {"lat": context.threshold_lat, "lon": context.threshold_lon, "alt": altitude, "V": speed_mps,
            "psi": math_heading_rad(context.runway_course_deg),
            "gamma": -math.radians(candidate.vertical_path.glidepath_deg), "m": mass_kg}


def aircraft_record(end: WindowResult, flight: Mapping[str, Any], series: FlightSeries, observed: Mapping[str, Any],
                    geometry: AirportGeometry, contexts: Mapping[tuple[str, str], AssessmentContext],
                    step_s: float, extra: Mapping[str, Any]) -> dict[str, Any]:
    """The evaluation record of one commanded aircraft's end (module docstring, THE RECORDS): ``flight`` its flight
    record (`signals_flights`), ``series`` its rebuilt series (the airframe and mass flown), ``observed`` its observed
    evaluation record, ``step_s`` the rows' interval (the
    vocabulary's), ``extra`` the window's fields its source carries. The source's path facts (TCH, glidepath) are the
    landing runway's candidate's; the observed runway's other path fields are left out (`OBSERVED_ONLY`)."""
    runway = (geometry.candidates[int(end.crossing["runway_index"])].ident if end.crossing is not None
              else flight["runway"])
    context = contexts[(flight["airport"], runway)]
    path = geometry.candidates[geometry.candidate_index(runway)].vertical_path
    states = record_states(end.states, geometry, step_s, float(series.scenario.initial.m))
    source = {key: value for key, value in observed["source"].items() if key not in OBSERVED_ONLY}
    source.update({"subject": "predicted", "predictor": PREDICTOR, "runway": runway,
                   "runway_course_deg": context.runway_course_deg, "target_source": "runway_threshold",
                   "threshold_crossing_height_m": path.crossing_height_m, "published_glidepath_deg": path.glidepath_deg,
                   "dynamics_typecode": series.scenario.source["dynamics_typecode"], "two_tier_outcome": end.outcome, **extra})
    initial = {key: value for key, value in states[0].items() if key != "t"}
    return {"source": source, "initial_state": initial,
            "target_state": target_state(context, geometry, observed["target_state"]["V"], states[-1]["m"]),
            "final_time_s": states[-1]["t"], "states": states, "controls": []}


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One cell of the readout: the aircraft, the two-tier outcomes, the evaluation's verdicts and gates, the outcome
    by verdict, and the pairing with the observed track (of the aircraft whose observed track passes, the share whose
    flown track passes too)."""
    outcomes = Counter(row["outcome"] for row in rows)
    verdicts = Counter(row["verdict"] for row in rows)
    observed_pass = [row for row in rows if row["observed_verdict"] == "pass"]
    return {
        "aircraft": len(rows), "outcomes": dict(outcomes.most_common()),
        "verdicts": {v: verdicts[v] for v in VERDICTS}, "pass_share": _share(verdicts["pass"], len(rows)),
        "gates": {gate: dict(Counter(row[gate] for row in rows).most_common()) for gate in GATES},
        "verdict_by_outcome": {o: {v: sum(r["outcome"] == o and r["verdict"] == v for r in rows) for v in VERDICTS}
                               for o in outcomes},
        "landed_gates": {gate: dict(Counter(row[gate] for row in rows if row["outcome"] == "landed").most_common())
                         for gate in GATES},
        "observed_verdicts": dict(Counter(row["observed_verdict"] for row in rows).most_common()),
        "observed_pass_share": _share(len(observed_pass), len(rows)),
        "flown_pass_of_observed_pass": _share(sum(row["verdict"] == "pass" for row in observed_pass),
                                              len(observed_pass)),
    }


def readout_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The readout (`_cell`): pooled and by airport, each by span, each with its ``all``."""
    out: dict[str, Any] = {}
    for airport in ("all", *sorted({row["airport"] for row in rows})):
        mine = [row for row in rows if airport == "all" or row["airport"] == airport]
        out[airport] = {span: _cell([row for row in mine if span == "all" or row["span"] == span])
                        for span in ("all", *sorted({row["span"] for row in mine}, key=float))}
    return out


def observed_records(airport: str) -> tuple[Path, dict[str, str]]:
    """The harvest's observed evaluation records of ``airport``: their directory, and each flight's record file by
    flight key."""
    approach = default_evaluation_report_path(arrival_manifest_path(airport)).parent
    summary = json.loads((approach / "summary.json").read_text(encoding="utf-8"))
    return approach, {row["flight_key"]: row["eval_file"] for row in summary["results"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True, help="a multi_train campaign's directory")
    parser.add_argument("--round", type=int, required=True, help="the round of the val read")
    parser.add_argument("--val-read", type=Path, required=True, help="the round's val read (multi_validation)")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--device", default="cuda", help="the val read's device")
    args = parser.parse_args(argv)
    campaign, val_read, out = (p if p.is_absolute() else REPO_ROOT / p for p in (args.campaign, args.val_read, args.out))
    if out.exists():
        parser.error(f"{out} exists: a run is written into a new directory")
    held = json.loads((val_read / "readout.json").read_text(encoding="utf-8"))
    if (held["split"], held["round"]) != ("val", args.round):
        parser.error(f"{val_read} is not round {args.round}'s val read")
    if not 0 <= args.round < done_rounds(campaign):
        parser.error(f"{campaign}: round {args.round} is not a done round")
    git = git_state()
    if git["dirty"]:
        parser.error("the tree has uncommitted changes; the run reads from a commit")
    record = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
    settings = multi_settings_of(record)
    inputs = inputs_here(record["inputs"])         # the paths as this checkout reads them (D157)
    instructions, executor = Path(inputs["instructions"]), Path(inputs["executor"])
    _, opened, _ = require_conforming_closed_loop(instructions, executor)          # D69: the checks run here (D73)
    edges_reference = Path(inputs["windows"]) / "conformance" / "edges.npz"
    checked_edges(edges_reference)                                                 # D104
    device = torch.device(args.device)
    context = open_context(Path(inputs["prior"]), instructions, executor, edges_reference, device,
                           Path(inputs["procedure_root"]), formal=True, data=False, splits=())
    model = round_model(context, settings, campaign, args.round)
    require_selection_of(instructions, context.interval_s, context.base_identity, "val")
    context = replace(context, splits={"val": split_data(instructions, "val", context.words, context.interval_s,
                                                         context.geometries, executor)})
    windows = selection_windows(context, settings, "val")
    if json.loads(json.dumps([window_record(w) for w in windows])) != held["windows"]:
        raise SystemExit(f"the val windows drawn here are not the val read's ({val_read / 'readout.json'}) — stopped")
    window_ends: list[tuple[int, Sequence[WindowResult]]] = []
    readout = selection_readout(model, context, windows, settings, None, stage=stage_d(), split="val",
                                window_ends=window_ends)
    if json.loads(json.dumps(readout)) != held["readout"]:
        raise SystemExit(f"round {args.round}: this code's readout of its model on the {device.type} is not the val "
                         f"read's ({val_read / 'readout.json'}): its flights would not be the read's — stopped")

    flights, start = context.splits["val"]["flights"], context.splits["val"]["start"]
    airports = sorted({w.scene.geometry.code for w in windows})
    contexts = {key: c for code in airports
                for key, c in contexts_for_airport(load_airport(code, config_file=DEFAULT_CONFIG,
                                                                cifp_file=DEFAULT_CIFP)).items()}
    observed = {code: observed_records(code) for code in airports}
    out.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for code in airports:
        records = out / "records" / code
        records.mkdir(parents=True)
        approach, files = observed[code]
        roster, mine = [], []
        for place, ends in window_ends:
            window = windows[place]
            if window.scene.geometry.code != code:
                continue
            for member, end in enumerate(ends):
                flight = flights[end.index]
                key = flight["dataset_id"].split(":", 1)[1]
                seen = json.loads((approach / files[key]).read_text(encoding="utf-8"))
                name = f"w{place:04d}_a{member:02d}_{key}_eval.json"
                extra = {"window_place": place, "window_member": member, "window_span_s": window.span_s}
                made = aircraft_record(end, flight, start.series[end.index], seen, window.scene.geometry, contexts, context.words.spec.step_s,
                                       extra)
                write_json_atomic(records / name, made)
                runway = made["source"]["runway"]
                roster.append({"flight_key": key, "eval_file": name, "arr_airport": code, "runway": runway,
                               "status": "solved"})
                mine.append({"airport": code, "place": place, "member": member, "span": span_key(window.span_s),
                             "flight_key": key, "file": name, "outcome": end.outcome, "runway": runway,
                             "observed_runway": flight["runway"]})
        write_json_atomic(records / "summary.json", {"subject": "predicted", "predictor": PREDICTOR,
                                                     "total": len(roster), "results": roster})
        graded = evaluate_records(records)
        seen_report = observed_verdicts(code, {row["flight_key"] for row in mine}, out / "observed" / code)
        require_same_grading(graded, seen_report, code)
        by_file = {Path(row["file"]).name: row for row in graded["trajectories"]}
        by_key = {row["flight_key"]: row for row in seen_report["trajectories"]}
        for row in mine:
            graded_row = by_file[row["file"]]
            row.update({"verdict": graded_row["verdict"], "event_status": graded_row["event_status"],
                        **{gate: graded_row[gate] for gate in GATES},
                        "observed_verdict": by_key[row["flight_key"]]["verdict"]})
        rows += mine
    write_json_atomic(out / "config.json", {
        "schema": VAL_EVALUATION_SCHEMA, "written_utc": utc_now(), "campaign": str(campaign), "round": args.round,
        "val_read": str(val_read), "inputs": record["inputs"], "git": git, "checks": opened["checks"],
        "device": str(device), "windows": len(windows), "aircraft": len(rows),
        "evaluation_schema": graded["schema_version"]})
    (out / "aircraft.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    write_json_atomic(out / "readout.json", {"schema": VAL_EVALUATION_SCHEMA, "readout": readout_table(rows)})
    seal(out)
    pooled = readout_table(rows)["all"]["all"]
    print(json.dumps({"out": str(out), "aircraft": pooled["aircraft"], "pass_share": pooled["pass_share"],
                      "observed_pass_share": pooled["observed_pass_share"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
