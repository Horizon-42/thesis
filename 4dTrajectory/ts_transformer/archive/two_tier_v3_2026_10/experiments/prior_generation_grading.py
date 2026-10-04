"""A generation-records directory graded beyond its pass rate (`prior_generation_records`, R35): the pass rate on the
runway each sentence landed on, and FDE split into its time and its place.

1. **The landed runway.** The evaluation judges a record against its ``source.runway`` — the observed flight's — so a
   sentence the executor flew to another runway (a parallel, mostly) fails there however it flew. Every recorded sentence
   whose last pointed runway is not the observed one and that crossed the runway it pointed at (`POINTED_CROSSINGS`: the
   judge's crossings but ``crossed_other_runway``, which crossed ANOTHER runway than the pointed one and keeps its verdict;
   a sentence that crossed nothing fails on any runway and keeps it too) is graded again against the runway it pointed at: its evaluation
   record copied with ``source.runway`` that runway and its target the evaluation's own threshold point for it
   (`landed_target`: the assessment context's threshold, at its published LTP + TCH height, ψ its course; speed, path
   angle and mass kept), its flown states read from the original states file (never copied), and no reference (a path
   deviation to the observed flight on another runway means nothing). The pass rate on the landed runway is the
   sentences ending on the observed runway passing there plus the others passing on theirs, over EVERY sentence (one not
   recorded does not pass).
2. **FDE's time and place** (real starts only: an augmented start has no truth; over the LANDED sentences — one that never
   arrived has no arrival to be early or late with). FDE is the distance at the observed
   landing's TIME (`geometry.metrics.common_physical_time_flight_metrics`: the prediction sampled at the truth's final
   time, an early arrival held at its end), so a sentence that lands later than the observed aircraft is still short of
   the threshold then, by its lateness × its speed. Beside FDE: the arrival endpoint error (the prediction's own end
   against the observed end — where it landed, not when), the final time error (predicted − observed landing time), and
   FDE over the late and the early apart.

Per kind (``labelled``, ``sample_<k>``) and pooled over the samples (``prior``), in all / per airport / per approach
kind / per airport × kind. Reads a whole run's ``records.json`` (`prior_generation_records.RECORDS_SCHEMA`) and its record
directories; `prior_generation_records` runs this at its end.

    python run_ts.py prior_generation_grading \\
        --records 4dTrajectory/outputs/POOLED/prior/v3_reread_v11_20260927/records_val_augmented_400x4 \\
        --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926

Writes ``<records>/grading/`` (new): ``landed_runway/<kind>/<ICAO>/`` (the graded copies, their ``summary.json`` and
``evaluation_report.json``) and ``grading.json`` — built in ``grading.partial`` and renamed when whole, so a failed grading
leaves no ``grading/``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.cli import DEFAULT_CIFP, DEFAULT_CONFIG
from evaluation.context import contexts_for_airport
from evaluation.thresholds import AssessmentContext
from geokit import compass_bearing_to_math_enu_rad
from trajectory_data_process.harvest.airports import load_airport
from ts_transformer.autopilot.judge import CROSSINGS
from ts_transformer.experiments.executor_replay import evaluate_records
from ts_transformer.experiments.prior_generation_records import RECORDS_SCHEMA
from ts_transformer.experiments.prior_generation_training_export import outputs_path
from ts_transformer.inference.export import EVAL_SUFFIX
from ts_transformer.instructions.artefact import load_candidates
from ts_transformer.instructions.readout import STRATA
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state

GRADING_SCHEMA = "ts-prior-generation-grading-v1"
#: The outcomes that crossed the runway the sentence pointed at — the ones graded again on it.
POINTED_CROSSINGS = tuple(outcome for outcome in CROSSINGS if outcome != "crossed_other_runway")


def regraded(sentence: dict[str, Any]) -> bool:
    """Whether a sentence is graded again on its last pointed runway (module docstring, 1)."""
    return (sentence["recorded"] and sentence["last_runway"] != sentence["observed_runway"]
            and sentence["outcome"] in POINTED_CROSSINGS)


def landed_target(target: dict[str, float], context: AssessmentContext) -> dict[str, float]:
    """The record's target moved to ``context``'s threshold: its position, its published LTP + TCH height (the height
    the evaluation's own frame stands on) and its course as ψ; speed, path angle and mass kept. A runway publishing no
    threshold height keeps the target's: the evaluation then reads the plane from the target and grades the vertical
    indeterminate (`evaluation.arrival._authoritative_frame`)."""
    height = context.desired_threshold_altitude_msl_m
    return {**target, "lat": context.threshold_lat, "lon": context.threshold_lon,
            "alt": target["alt"] if height is None else height,
            "psi": compass_bearing_to_math_enu_rad(float(np.radians(context.runway_course_deg)))}


def graded_copy(record: dict[str, Any], runway: str, context: AssessmentContext, states: Path,
                directory: Path) -> dict[str, Any]:
    """An evaluation record graded against ``runway`` instead of its own, written into ``directory``: its states read
    from ``states`` (the original file), no reference."""
    source = {**record["source"], "runway": runway,
              "landedRunwayGrading": {"observedRunway": record["source"]["runway"], "landedRunway": runway}}
    copy = {key: value for key, value in record.items() if key != "reference_file"}
    return {**copy, "source": source, "target_state": landed_target(record["target_state"], context),
            "states_ref": {**record["states_ref"], "file": os.path.relpath(states, directory)}}


def _stats(values: list[float]) -> dict[str, float] | None:
    """As `prior_generation_records._stats`, with the p95 (its records.json payload is settled and keeps its fields)."""
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(array)), "mean": float(array.mean()), "p90": float(np.percentile(array, 90)),
            "p95": float(np.percentile(array, 95)), "count": int(len(array))}


def summarise(rows: list[dict[str, Any]], *, timing: bool) -> dict[str, Any]:
    """Over ``rows`` (sentences): the pass rate on the observed runway and on the landed one, the regraded sentences'
    verdicts; with ``timing``, over the landed sentences, FDE, the arrival endpoint error and the final time error, and
    FDE over the late and the early."""
    count = len(rows)
    again = [r for r in rows if r["regraded"]]
    verdicts: dict[str, int] = defaultdict(int)
    for r in again:
        verdicts[r["verdict_landed_runway"]] += 1
    out = {"sentences": count,
           "pass_rate_observed_runway": sum(r["verdict"] == "pass" for r in rows) / count,
           "pass_rate_landed_runway": sum(r["verdict_landed_runway"] == "pass" for r in rows) / count,
           "ending_on_another_runway": sum(r["last_runway"] != r["observed_runway"] for r in rows),
           "graded_again": {"sentences": len(again), "verdicts": dict(sorted(verdicts.items()))}}
    if timing:
        landed = [r for r in rows if r["outcome"] == "landed"]
        late = [r for r in landed if r["final_time_error_s"] > 0.0]
        early = [r for r in landed if r["final_time_error_s"] <= 0.0]
        out.update({
            "landed": len(landed),
            "fde_m": _stats([r["fde_m"] for r in landed]),
            "arrival_endpoint_error_m": _stats([r["arrival_endpoint_error_m"] for r in landed]),
            "final_time_error_s": _stats([r["final_time_error_s"] for r in landed]),
            "late_share": len(late) / len(landed) if landed else None,
            "fde_late_m": _stats([r["fde_m"] for r in late]),
            "fde_early_m": _stats([r["fde_m"] for r in early]),
        })
    return out


def grouped(rows: list[dict[str, Any]], *, timing: bool) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for key in ("all", r["airport"], r["stratum"], f"{r['airport']} {r['stratum']}"):
            groups[key].append(r)
    return {key: summarise(value, timing=timing) for key, value in sorted(groups.items())}


def grade(records: Path, instructions: Path) -> dict[str, Any]:
    """Grade ``records`` (a whole `prior_generation_records` run) into ``records/grading`` and return the readout."""
    started = time.perf_counter()
    out, partial = records / "grading", records / "grading.partial"
    for path in (out, partial):
        if path.exists():
            raise SystemExit(f"{path} exists; a grading is never overwritten (a partial one is a failed run's: inspect it)")
    document = json.loads((records / "records.json").read_text(encoding="utf-8"))
    if document["schema"] != RECORDS_SCHEMA:
        raise SystemExit(f"{records / 'records.json'} is a {document['schema']} file; this code reads {RECORDS_SCHEMA}")
    if document["partial"] is not None:
        raise SystemExit(f"{records} is a partial run: it is not graded")
    if outputs_path(document["instructions"]) != outputs_path(instructions):
        raise SystemExit(f"{records} was written with {document['instructions']}, not {instructions}")
    candidates = load_candidates(instructions)
    sentences = document["sentences"]
    contexts = {}
    for code in sorted({s["airport"] for s in sentences}):
        contexts.update(contexts_for_airport(load_airport(code, config_file=DEFAULT_CONFIG, cifp_file=DEFAULT_CIFP)))
    # every record directory's summary rows, by kind and flight key (the record's stem): one per recorded sentence
    summaries = {}
    for kind in sorted({s["kind"] for s in sentences}):
        for code in sorted({s["airport"] for s in sentences if s["kind"] == kind}):
            summary = json.loads((records / "records" / kind / code / "summary.json").read_text(encoding="utf-8"))
            recorded = sum(s["recorded"] for s in sentences if s["kind"] == kind and s["airport"] == code)
            if len(summary["results"]) != recorded:
                raise SystemExit(f"{records / 'records' / kind / code} holds {len(summary['results'])} records, "
                                 f"records.json {recorded} recorded sentences")
            for row in summary["results"]:
                summaries[(kind, row["eval_file"].removesuffix(EVAL_SUFFIX))] = row
    regrade: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for s in sentences:
        if regraded(s):
            regrade[(s["kind"], s["airport"])].append(s)
    runways = {(s["kind"], s["flight_key"]): candidates[s["airport"]].candidates[s["last_runway"]].ident
               for group in regrade.values() for s in group}
    missing = sorted({(s["airport"], runways[(s["kind"], s["flight_key"])]) for group in regrade.values() for s in group}
                     - set(contexts))
    if missing:
        raise SystemExit(f"no assessment context for {missing}: those runways cannot be graded on")
    verdicts: dict[tuple[str, str], str] = {}
    for (kind, code), group in sorted(regrade.items()):
        directory = partial / "landed_runway" / kind / code
        directory.mkdir(parents=True)
        original = records / "records" / kind / code
        rows = []
        for s in group:
            row = summaries[(kind, s["flight_key"])]
            runway = runways[(kind, s["flight_key"])]
            record = json.loads((original / row["eval_file"]).read_text(encoding="utf-8"))
            copy = graded_copy(record, runway, contexts[(code, runway)], original / record["states_ref"]["file"],
                               directory)
            (directory / row["eval_file"]).write_text(json.dumps(copy, separators=(",", ":"), allow_nan=False),
                                                     encoding="utf-8")
            rows.append({**row, "runway": runway})
        write_json_atomic(directory / "summary.json", {
            "results": rows, "split": document["readout"]["split"],
            "landed_runway_grading": {"records": str(original), "note": "each record graded against the runway its "
                                      "sentence last pointed at and crossed, not the observed flight's"}})
        graded = evaluate_records(directory)
        verdicts.update({(kind, row["flight_key"]): row["verdict"] for row in graded["trajectories"]})
        print(f"  {kind} {code}: {len(group)} graded on the landed runway, {time.perf_counter() - started:.0f}s",
              flush=True)

    timing = document["readout"]["augment_seed"] is None
    rows_out = []
    for s in sentences:
        row = summaries[(s["kind"], s["flight_key"])] if s["recorded"] else None
        rows_out.append({
            **{key: s[key] for key in ("dataset_id", "flight_key", "airport", "stratum", "kind", "source", "sample",
                                       "outcome", "recorded", "verdict", "observed_runway", "last_runway")},
            "regraded": regraded(s),
            "verdict_landed_runway": verdicts[(s["kind"], s["flight_key"])] if regraded(s) else s["verdict"],
            **({key: row[key] for key in ("fde_m", "arrival_endpoint_error_m", "final_time_error_s")}
               if timing and row is not None else {})})
    samples = document["readout"]["samples"]
    readouts = {"prior": grouped([r for r in rows_out if r["source"] == "prior"], timing=timing)}
    readouts.update({f"sample_{k}": grouped([r for r in rows_out if r["kind"] == f"sample_{k}"], timing=timing)
                     for k in range(samples)})
    if any(r["kind"] == "labelled" for r in rows_out):
        readouts["labelled"] = grouped([r for r in rows_out if r["kind"] == "labelled"], timing=timing)
    result = {"schema": GRADING_SCHEMA, "written_utc": utc_now(), "git": git_state(), "records": str(records),
              "instructions": str(instructions), "timing_read": timing,
              "landed_runway": ("each recorded sentence ending on another runway than the observed flight's and crossing "
                                "the runway it last pointed at, graded against that runway (its target that runway's "
                                "threshold point); every other sentence keeps its verdict"),
              "timing": ("over the landed sentences: FDE at the observed landing time; arrival endpoint error = the "
                         "prediction's own end against the observed end; final time error = predicted − observed "
                         "landing time" if timing else "none: an augmented start has no truth"),
              "readouts": readouts, "sentences": rows_out, "elapsed_s": time.perf_counter() - started}
    partial.mkdir(exist_ok=True)
    write_json_atomic(partial / "grading.json", result)
    partial.rename(out)
    return result


def print_readout(result: dict[str, Any]) -> None:
    for name, block in result["readouts"].items():
        print(f"{name}:")
        for key in ("all", *STRATA):
            if key not in block:
                continue
            part = block[key]
            line = (f"  {key:12s} n={part['sentences']:5d}  pass on the observed runway "
                    f"{part['pass_rate_observed_runway']:.3f}  on the landed one {part['pass_rate_landed_runway']:.3f}")
            if result["timing_read"] and part["fde_m"] is not None:
                line += (f"  FDE p50/p95 {part['fde_m']['median']:.0f}/{part['fde_m']['p95']:.0f} m  endpoint p95 "
                         f"{part['arrival_endpoint_error_m']['p95']:.0f} m  late {part['late_share']:.3f}")
            print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--records", type=Path, required=True, help="a whole prior_generation_records directory")
    parser.add_argument("--instructions", type=Path, required=True, help="its instruction artefact")
    args = parser.parse_args(argv)
    records, instructions = (path if path.is_absolute() else REPO_ROOT / path for path in (args.records, args.instructions))
    result = grade(records, instructions)
    print_readout(result)
    print(f"→ {records / 'grading'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
