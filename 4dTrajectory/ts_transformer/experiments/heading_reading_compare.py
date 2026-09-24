"""Vocabulary design §10.1: the heading readings compared by flying them with the real executor, on train.

Every variant reads the SAME train sample (`replay.draw_flights`: ``--per-airport`` flights of each airport flown on
their own type's dynamics, a seeded permutation) with the labeller under its own spec, and the executor flies each
sentence from row 0 (`replay.fly_sentences`: the track word clock, the capture, the full dynamics). The executor's
parameters are the formal spec's (``--executor``) and are HELD FIXED. The spec's executor source hash is not
required to match: this code moved it (the judge and replay were refactored, the laws did not change), and the
labeller changed under it. The vocabulary values are the base spec's (``--base-spec``, the previous vocabulary's spec
file, measured on the same flights' signals; its schema and sha are checked and recorded), except the heading
reading's and the heading tolerance, which follows the grid by the spec's own rule (half a step plus
`measure.HEADING_WANDER_ALLOWANCE_DEG`: 4.5° on 5°, 3.0° on 2°).

What differs between the variants besides the heading words: the clearance goes with the last heading word
(`join_row`), and the "unspecified" speed is placed from the clearance, so the speed column moves with the reading
too; every row records both.

The variants (`VARIANTS`): ``H1`` — the holds reading (§3.2, instruction-v2's); ``H3-<step>-L<lead>`` — the per-step
reading (§10.1) at a heading step of 5° or 2°, merged with a band of half a step, labelled ``lead`` seconds early.

Per flight and variant: the outcome (`judge.outcome_of`: the words are NOT judged — the per-step words' envelope is
still to be designed), the evaluation verdict of the flown record paired with the observed flight's, the time-aligned
horizontal distance to the observed
flight (mean and largest over the sentence rows both reach), and the heading words said. The observed verdicts are
graded HERE, by this evaluation code, over the sampled flights' own observed records (`observed_verdicts`: the harvest's
records linked read-only under ``--out``, rostered from its ``approach/summary.json``): the harvest's stored report may
predate a change of the evaluation's methodology, and the pairing needs one grading on both sides. The stratum (straight-in /
vectored) is the H1 reading's for every variant, so the strata hold the same flights. The headline table is over
the flights every variant labelled; each variant's refusals are listed beside it, and one line per variant counts
every drawn flight with a refusal as a failure. One more group is read on its own: the flights whose H1 reading
inserts an intercept of 90° or more (a continuous turn onto the final, §10.1's problem).

    python run_ts.py heading_reading_compare \\
        --signals 4dTrajectory/outputs/POOLED/instruction_language/v3_20260924 \\
        --base-spec 4dTrajectory/outputs/POOLED/instruction_language/v2_20260924/spec.json \\
        --executor 4dTrajectory/outputs/POOLED/executor/v2_20260924 --per-airport 400 --out <new directory>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.judge import Outcome, flown_track, outcome_of
from ts_transformer.autopilot.spec import load_spec as load_executor_spec
from ts_transformer.data.lateral_eligibility import default_evaluation_report_path
from ts_transformer.experiments.executor_replay import (
    HORIZON, PREDICTOR, evaluate_records, executor_forecast, require_same_grading,
)
from ts_transformer.inference.export import build_prediction_record, observed_series_metrics, write_batch
from ts_transformer.instructions.labeller.read import Reading, read_flight
from ts_transformer.instructions.measure import HEADING_WANDER_ALLOWANCE_DEG
from ts_transformer.instructions.labeller.records import Refused
from ts_transformer.instructions.readout import STRATA, flight_record
from ts_transformer.instructions.spec import READING_RULE, VocabularySpec
from ts_transformer.instructions.words import HEADING, Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.autopilot.plant import EXECUTOR_DYNAMICS
from ts_transformer.repo_layout import REPO_ROOT, arrival_manifest_path, git_state

COMPARE_SCHEMA = "ts-heading-reading-compare-v1"
SPLIT = "train"
#: The previous vocabulary's spec file (instruction-v2's), read for its measured values only.
BASE_SPEC_SCHEMA = "ts-instruction-spec-v3"
#: The group read on its own: an H1 intercept turn at least this large (§10.1).
ONTO_FINAL_TURN_DEG = 90.0


@dataclass(frozen=True)
class Variant:
    name: str
    reading: str
    step_deg: float
    lead_s: float


VARIANTS = (
    Variant("H1", "holds", 5.0, 0.0),
    Variant("H3-5-L0", "per-step", 5.0, 0.0),
    Variant("H3-5-L2", "per-step", 5.0, 2.0),
    Variant("H3-5-L4", "per-step", 5.0, 4.0),
    Variant("H3-5-L6", "per-step", 5.0, 6.0),
    Variant("H3-2-L4", "per-step", 2.0, 4.0),
    Variant("H3-2-L6", "per-step", 2.0, 6.0),
)


def base_spec(path: Path) -> tuple[VocabularySpec, str]:
    """The base spec's values under this code's reading rule, with the heading reading of instruction-v2 (holds),
    and the base file's sha: every value the labeller measured stays the base file's."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["schema"] != BASE_SPEC_SCHEMA:
        raise ValueError(f"{path} is a {record['schema']} file, not {BASE_SPEC_SCHEMA}")
    canonical = json.dumps(record["spec"], sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != record["sha256"]:
        raise ValueError(f"{path}: the stored sha does not match its spec")
    values = dict(record["spec"])
    values.pop("reading_rule")
    values.update(heading_reading="holds", heading_lead_s=0.0, heading_band_deg=values["heading_step_deg"] / 2,
                  reading_rule=READING_RULE)
    return VocabularySpec.from_dict(values), record["sha256"]


def variant_spec(base: VocabularySpec, variant: Variant) -> VocabularySpec:
    return replace(base, heading_reading=variant.reading, heading_step_deg=variant.step_deg,
                   heading_tolerance_deg=variant.step_deg / 2 + HEADING_WANDER_ALLOWANCE_DEG,
                   heading_lead_s=variant.lead_s, heading_band_deg=variant.step_deg / 2)


def heading_words(reading: Reading) -> int:
    """The heading words said after row 0."""
    return sum(1 for item in reading.instructions if item.column == HEADING and item.row > 0)


def onto_final_turn_deg(reading: Reading) -> float:
    """The largest intercept turn the H1 reading inserted (0 when none): the continuous turn onto the final."""
    return max((abs(t["turn_deg"]) for t in reading.checks["turns"] if t["kind"] == "intercept"), default=0.0)


def tag_rows(rows: list[dict[str, Any]], dataset_ids: list[str], h1: dict[int, Reading]) -> None:
    """Give each flown row its sample index — by its dataset id: `fly_variant` returns the rows grouped by airport,
    not in sample order — and the H1 reading's stratum and inserted intercept turn."""
    index = {dataset_id: k for k, dataset_id in enumerate(dataset_ids)}
    for row in rows:
        k = index[row["dataset_id"]]
        reading = h1.get(k)
        row["sample_index"] = k
        row["stratum"] = "refused by H1" if reading is None else flight_record(reading)["stratum"]
        row["h1_intercept_turn_deg"] = None if reading is None else onto_final_turn_deg(reading)


def distances(batch: replay.Batch, flown: Any, outcomes: list[Outcome]) -> list[tuple[float, float]]:
    """Per flight, the mean and the largest horizontal distance between the flown and the observed flight at the
    sentence rows both reach, time-aligned from row 0 (`replay.flight_alignment`'s rows); a dynamics failure is read
    to the row before it, as its record is (`executor_forecast`)."""
    out = []
    for j, ended in enumerate(outcomes):
        observed, reading = batch.signals[j], batch.readings[j]
        step_rows = int(round((observed.time_s[1] - observed.time_s[0]) / flown.cycle_s))
        end = ended.end_row - 1 if ended.outcome == "dynamics_failure" else ended.end_row
        track = flown_track(flown.states[j, : end + 1].cpu().numpy(), batch.geometries[j])
        rows = min(len(reading.words), end // step_rows + 1)
        at = np.arange(rows) * step_rows
        gap = np.hypot(track["e"][at] - observed.e_m[:rows], track["n"][at] - observed.n_m[:rows])
        out.append((float(gap.mean()), float(gap.max())))
    return out


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


def fly_variant(batch: replay.Batch, params: Any, words: Words, *, chunk: int, device: torch.device,
                records: Path, observed_reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Fly one variant's sentences airport by airport, write and grade their records; one row per flight."""
    rows: list[dict[str, Any]] = []
    by_airport: dict[str, list[int]] = defaultdict(list)
    for j, reading in enumerate(batch.readings):
        by_airport[reading.airport].append(j)
    for airport, members in sorted(by_airport.items()):
        airport_rows, predictions, metrics = [], [], []
        for start in range(0, len(members), chunk):
            part = replay.subset(batch, members[start: start + chunk])
            flown = replay.fly_sentences(part, params, words, device=device)
            outcomes = [outcome_of(flown, j, part.geometries[j], part.readings[j].runway_index, words.spec)
                        for j in range(len(part.readings))]
            inputs = part.inputs(device)
            aligned = replay.flight_alignment(part, flown, outcomes)
            gaps = distances(part, flown, outcomes)
            for j, ended in enumerate(outcomes):
                series, reading = part.series[j], part.readings[j]
                recorded = not (ended.outcome == "dynamics_failure" and ended.end_row <= 1)
                if recorded:
                    forecast = executor_forecast(flown, j, ended, inputs, series)
                    predictions.append(build_prediction_record(series, forecast, index=len(predictions),
                                                               model_name=PREDICTOR, horizon_mode=HORIZON, split=SPLIT))
                    metrics.append(observed_series_metrics(series, forecast))
                airport_rows.append({
                    "dataset_id": reading.dataset_id, "flight_key": series.scenario.source["flight_key"],
                    "airport": airport, "outcome": ended.outcome, "crossing": ended.crossing,
                    "heading_words": heading_words(reading), "capture_row": reading.capture_row,
                    "join_row": reading.join_row, "unspecified_row": reading.unspecified_row,
                    "rows": len(reading.words), "recorded": recorded,
                    "mean_horizontal_m": gaps[j][0], "max_horizontal_m": gaps[j][1],
                    "landing_time_minus_observed_s": aligned[j]["landing_time_minus_observed_s"]})
            del flown
        directory = records / airport
        write_batch(predictions, output_dir=directory,
                    config_dict={"model": PREDICTOR, "horizon_mode": HORIZON,
                                 "prediction_output": EXECUTOR_DYNAMICS.prediction_output,
                                 "executor_params": asdict(params), "vocabulary_spec": words.spec.to_dict()},
                    flight_metrics=metrics, checkpoint="heading_reading_compare", split=SPLIT)
        graded = evaluate_records(directory)
        require_same_grading(graded, observed_reports[airport], airport)
        replayed = {row["flight_key"]: row["verdict"] for row in graded["trajectories"]}
        seen = {row["flight_key"]: row["verdict"] for row in observed_reports[airport]["trajectories"]}
        for row in airport_rows:
            row["replay_verdict"] = replayed[row["flight_key"]] if row["recorded"] else "no record: failed at once"
            row["observed_verdict"] = seen[row["flight_key"]]
        rows += airport_rows
    return rows


def _percentiles(values: list[float]) -> dict[str, float] | None:
    return {f"p{q}": float(np.percentile(values, q)) for q in (10, 50, 90)} if values else None


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired = [r for r in rows if r["observed_verdict"] == "pass"]
    steps = [r["heading_words"] / r["capture_row"] for r in rows if r["capture_row"] > 0]
    return {
        "flights": len(rows),
        "landed": sum(r["outcome"] == "landed" for r in rows) / len(rows) if rows else None,
        "outcomes": dict(Counter(r["outcome"] for r in rows).most_common()),
        "observed_passes": len(paired),
        "replay_passes_where_observed_passes": (sum(r["replay_verdict"] == "pass" for r in paired) / len(paired)
                                                if paired else None),
        "mean_horizontal_m": _percentiles([r["mean_horizontal_m"] for r in rows]),
        "max_horizontal_m": _percentiles([r["max_horizontal_m"] for r in rows]),
        "heading_words_per_flight": _percentiles([r["heading_words"] for r in rows]),
        "share_of_pre_capture_steps_with_a_heading_word": float(np.mean(steps)) if steps else None,
    }


GROUPS = ("all", *STRATA, "onto final", "not onto final")


def in_group(row: dict[str, Any], group: str) -> bool:
    """`GROUPS`: every flight, an H1 stratum, or whether the H1 reading inserted an intercept of `ONTO_FINAL_TURN_DEG`
    or more (a continuous turn onto the final)."""
    if group == "all":
        return True
    if group in STRATA:
        return row["stratum"] == group
    onto = row["h1_intercept_turn_deg"] is not None and row["h1_intercept_turn_deg"] >= ONTO_FINAL_TURN_DEG
    return onto if group == "onto final" else not onto


def over_drawn(rows: list[dict[str, Any]], drawn: int, observed_passes: int) -> dict[str, Any]:
    """Every drawn flight, a flight the variant's labeller refused counted as a failure: landed, and the replays that
    pass where the observed flight passes."""
    return {"drawn": drawn, "landed": sum(r["outcome"] == "landed" for r in rows) / drawn,
            "replay_passes_where_observed_passes": sum(r["replay_verdict"] == "pass" == r["observed_verdict"]
                                                       for r in rows) / observed_passes}


def clearance_shift(rows: list[dict[str, Any]], h1_rows: list[dict[str, Any]], common: set[int]) -> dict[str, Any]:
    """Against H1 on the common flights: how many rows later the clearance and the "unspecified" speed are said."""
    h1 = {r["sample_index"]: r for r in h1_rows}
    pairs = [(r, h1[r["sample_index"]]) for r in rows if r["sample_index"] in common]
    return {"join_row_minus_h1": _percentiles([r["join_row"] - o["join_row"] for r, o in pairs]),
            "unspecified_row_minus_h1": _percentiles([r["unspecified_row"] - o["unspecified_row"] for r, o in pairs])}


def render(result: dict[str, Any]) -> str:
    lines = [f"heading readings on {result['sample']['flights']} {SPLIT} flights ({result['common_flights']} labelled "
             "by every variant); executor params held at the formal spec's", ""]
    head = (f"{'variant':10s} {'group':14s} {'n':>5s} {'landed':>7s} {'eval':>6s}  {'mean dist p50/p90 km':>21s}  "
            f"{'max dist p50/p90 km':>20s}  {'hdg words p50':>13s}  {'steps w/ word':>13s}")
    lines.append(head)
    for name, by_group in result["common"].items():
        for stratum, s in by_group.items():
            if not s["flights"]:
                continue
            mean, largest, words = s["mean_horizontal_m"], s["max_horizontal_m"], s["heading_words_per_flight"]
            evaluation = s["replay_passes_where_observed_passes"]
            lines.append(f"{name:10s} {stratum:14s} {s['flights']:5d} {100 * s['landed']:6.1f}% "
                         f"{'—' if evaluation is None else f'{100 * evaluation:5.1f}%'}  "
                         f"{mean['p50'] / 1000:9.2f} / {mean['p90'] / 1000:5.2f}     "
                         f"{largest['p50'] / 1000:8.2f} / {largest['p90'] / 1000:5.2f}      {words['p50']:9.0f}      "
                         + ("        —" if s["share_of_pre_capture_steps_with_a_heading_word"] is None else
                            f"{100 * s['share_of_pre_capture_steps_with_a_heading_word']:9.1f}%"))
    lines += ["", "every drawn flight, a refusal counted as a failure:"]
    for name, d in result["over_drawn"].items():
        lines.append(f"  {name:10s} landed {100 * d['landed']:5.1f}%  eval {100 * d['replay_passes_where_observed_passes']:5.1f}%")
    lines += ["", "clearance and unspecified speed, rows later than H1 (p10/p50/p90):"]
    for name, shift in result["clearance_shift"].items():
        j, u = shift["join_row_minus_h1"], shift["unspecified_row_minus_h1"]
        lines.append(f"  {name:10s} clearance {j['p10']:+.0f}/{j['p50']:+.0f}/{j['p90']:+.0f}  "
                     f"unspecified {u['p10']:+.0f}/{u['p50']:+.0f}/{u['p90']:+.0f}")
    lines += ["", "refused by the labeller, per variant: " + json.dumps(result["refused"])]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--signals", type=Path, required=True, help="an instruction artefact holding the signals")
    parser.add_argument("--base-spec", type=Path, required=True, help="the spec.json whose values the variants keep")
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec whose parameters are flown")
    parser.add_argument("--per-airport", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--variants", nargs="+", default=[v.name for v in VARIANTS])
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)
    resolve = lambda path: path if path.is_absolute() else REPO_ROOT / path  # noqa: E731
    signals_dir, out = resolve(args.signals), resolve(args.out)
    if out.exists():
        parser.error(f"{out} exists; a comparison is never overwritten")
    unknown = set(args.variants) - {v.name for v in VARIANTS}
    if unknown:
        parser.error(f"unknown variants {sorted(unknown)}")
    variants = [v for v in VARIANTS if v.name in args.variants]
    if variants[0].name != "H1":
        parser.error("H1 is required: it gives every flight its stratum")
    started = time.perf_counter()
    base, base_sha = base_spec(resolve(args.base_spec))
    params, executor_record = load_executor_spec(resolve(args.executor))
    device = torch.device(args.device)

    meta = json.loads((signals_dir / "signals.json").read_text(encoding="utf-8"))
    drawn = replay.draw_flights(signals_dir, SPLIT, list(range(len(meta["splits"][SPLIT]["flights"]))),
                                per_airport=args.per_airport, seed=args.seed)
    print(f"drawn {len(drawn.signals)} flights, {time.perf_counter() - started:.0f}s", flush=True)

    out.mkdir(parents=True)
    observed_reports = {
        airport: observed_verdicts(airport, {s.scenario.source["flight_key"] for f, s in zip(drawn.signals, drawn.series)
                                             if f.airport == airport}, out / "observed" / airport)
        for airport in sorted({f.airport for f in drawn.signals})}
    print(f"observed flights graded, {time.perf_counter() - started:.0f}s", flush=True)
    readings: dict[str, dict[int, Reading]] = {}
    refused: dict[str, dict[str, int]] = {}
    for variant in variants:
        spec = variant_spec(base, variant)
        params.check(spec)
        words = Words(spec)
        readings[variant.name], reasons = {}, Counter()
        for k, flight in enumerate(drawn.signals):
            try:
                readings[variant.name][k] = read_flight(flight, drawn.geometries[flight.airport], spec, words)
            except Refused as refusal:
                reasons[refusal.reason] += 1
        refused[variant.name] = dict(reasons.most_common())
        print(f"{variant.name}: {len(readings[variant.name])} labelled, {sum(reasons.values())} refused, "
              f"{time.perf_counter() - started:.0f}s", flush=True)
    dataset_ids = [flight.dataset_id for flight in drawn.signals]

    rows: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        spec = variant_spec(base, variant)
        keep = sorted(readings[variant.name])
        batch = replay.batch_of(drawn, keep, [readings[variant.name][k] for k in keep])
        flown_rows = fly_variant(batch, params, Words(spec), chunk=args.chunk, device=device,
                                 records=out / "records" / variant.name, observed_reports=observed_reports)
        tag_rows(flown_rows, dataset_ids, readings["H1"])
        rows[variant.name] = flown_rows
        print(f"{variant.name}: flown and graded, {time.perf_counter() - started:.0f}s", flush=True)

    common = set.intersection(*(set(readings[v.name]) for v in variants))
    observed_passes = sum(row["verdict"] == "pass" for report in observed_reports.values()
                          for row in report["trajectories"])
    result = {
        "schema": COMPARE_SCHEMA, "written_utc": utc_now(), "git": git_state(),
        "signals": str(signals_dir), "base_spec": {"path": str(resolve(args.base_spec)), "sha256": base_sha},
        "executor": {"path": str(resolve(args.executor)), "sha256": executor_record["sha256"],
                     "params": asdict(params), "source_hash_required": False},
        "sample": drawn.description,
        "variants": {v.name: {**asdict(v), "spec_sha256": variant_spec(base, v).sha256} for v in variants},
        "refused": refused, "common_flights": len(common),
        "common": {v.name: {group: summarise([r for r in rows[v.name] if r["sample_index"] in common and in_group(r, group)])
                            for group in GROUPS} for v in variants},
        "over_drawn": {v.name: over_drawn(rows[v.name], len(drawn.signals), observed_passes) for v in variants},
        "clearance_shift": {v.name: clearance_shift(rows[v.name], rows["H1"], common) for v in variants},
        "by_airport": {v.name: {airport: {stratum: summarise([r for r in rows[v.name] if r["sample_index"] in common
                                                             and r["airport"] == airport and r["stratum"] == stratum])
                                          for stratum in STRATA}
                                for airport in sorted(drawn.geometries)} for v in variants},
        "rows": rows, "elapsed_s": time.perf_counter() - started,
    }
    write_json_atomic(out / "compare.json", result)
    text = render(result)
    (out / "compare.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
