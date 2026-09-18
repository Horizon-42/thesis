"""Runway-intent R3.1: scheduling under ETA uncertainty, read as a landing-time predictor.

Plan `docs/2026-09-13_runway_intent_plan.zh.md` §18.1. R3's roster (R2b's flights: every day_a
validation-day flight the retrained expert flew) and R3's rules (`runway_intent_r3.airport_rules`). Each
flight's ETA error is drawn from an error model CALIBRATED on the same expert's own validation split —
flights on day_a's TRAINING days its weights never trained on — flown once under their landed runway
exactly as R2b flew (`forecast_approaches`), e = the head's predicted remaining time − the true one. The
model is stratified by what the anchor knows: the head's own predicted remaining time (under its top
runway), cut at the calibration set's terciles; a stratum under `MIN_CALIBRATION_FLIGHTS` falls back to
the pooled errors. One error per flight, the same on every runway; flights independent.

`runway_schedule.sample_schedules` then schedules the traffic ``--samples`` times, each on one draw of
every flight's free arrival (ETA − e), first come first served under the FAA minima. A flight's M draws
are its stratum's M equal-probability quantiles in a random order (`error_grid`): its own draws are the
same distribution in every sample and only the PAIRING across flights is random, so a flight nobody
interacts with gets exactly the calibrated prediction — with draws taken with replacement, the median of
a flight's own 200 draws misses its stratum's by 1.2-2.5 s and the noise read as an interaction effect
(R3.1 review). A flight's prediction is its MEDIAN landing time over the samples, its runway the most
frequent one. It is read against the CALIBRATED independent prediction — the same error model, no
interaction: the top runway's ETA less the median of the flight's own draws — so that correcting the
ETA's bias is not credited to the interaction; the raw independent ETA and R3's deterministic schedule
are read beside them, and the CAUSAL stochastic schedule (each flight placed when its anchor is reached;
the FCFS form reads ETAs later-anchored flights only have after it). Plan-only. The calibration split is
also the split the checkpoint was selected on (the plan head selects on the fixed-anchor objective,
which includes T), so the interval coverage is measured, never guaranteed.

    python run_ts.py runway_intent_r31 --airport KRDU \\
        --r2 4dTrajectory/outputs/POOLED/experiments/runway_intent_r2b_20260913/KRDU/runway_intent_r2.json \\
        --checkpoint 4dTrajectory/outputs/POOLED/experiments/runway_intent_r2b_20260913/KRDU_day_a_expert/checkpoint.pt \\
        --output-dir 4dTrajectory/outputs/POOLED/experiments/runway_intent_r31_20260914/KRDU
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ts_transformer.backbone.adapters import resolve_device
from ts_transformer.config import PREDICTION_PLAN, TSConfig
from ts_transformer.data.data_provenance import arrival_data_provenance, require_matching_data_provenance
from ts_transformer.data.dataset import build_series, load_flight_dicts
from ts_transformer.data.lateral_eligibility import default_lateral_pass_roster_path
from ts_transformer.experiments.runway_intent_r3 import (
    BUSY_PER_HOUR,
    STRATA,
    abs_quantiles,
    airport_rules,
    mark_busy_hours,
    roster,
)
from ts_transformer.inference.calibration import MIN_CALIBRATION_FLIGHTS
from ts_transformer.inference.export import observed_series_metrics
from ts_transformer.inference.forecast import forecast_approaches
from ts_transformer.inference.runway_schedule import fcfs_by_eta, sample_schedules, schedule
from ts_transformer.training.train import load_checkpoint

SCHEMA = "ts-runway-intent-r31-v1"
POOLED = "pooled"
MOVED_DELAY_S = 5.0               # plan §18.1 gate 3: a flight the stochastic schedule delays by more than this (median)
DELAYED_S = 10.0                  # P(delayed): a sample's delay over this
INTERVAL = (10.0, 90.0)           # the landing-time interval read for coverage (nominal 80 %)
PREDICTORS = ("stochastic", "stochastic_causal", "calibrated", "independent", "deterministic")


@dataclass(frozen=True)
class ErrorModel:
    """The ETA error (predicted − true remaining time, s) by the head's predicted remaining time,
    stratified at ``edges_s`` (the calibration set's terciles). ``residuals`` holds every stratum with
    at least `MIN_CALIBRATION_FLIGHTS` errors and always `POOLED`."""

    edges_s: tuple[float, ...]
    residuals: dict[str, np.ndarray]

    @classmethod
    def fit(cls, predicted_s: Sequence[float], errors_s: Sequence[float], *, bins: int = 3) -> "ErrorModel":
        predicted = np.asarray(predicted_s, dtype=float)
        errors = np.asarray(errors_s, dtype=float)
        edges = tuple(float(q) for q in np.percentile(predicted, [100.0 * k / bins for k in range(1, bins)]))
        residuals = {POOLED: errors}
        which = np.digitize(predicted, edges)
        for k in range(bins):
            members = errors[which == k]
            if len(members) >= MIN_CALIBRATION_FLIGHTS:
                residuals[cls.name(k)] = members
        return cls(edges, residuals)

    @staticmethod
    def name(k: int) -> str:
        return f"T tercile {k + 1}"

    def stratum(self, predicted_s: float) -> str:
        """The stratum a flight's errors are drawn from (its own, or `POOLED` where that is too thin)."""
        own = self.name(int(np.digitize([predicted_s], self.edges_s)[0]))
        return own if own in self.residuals else POOLED

    def centered(self) -> "ErrorModel":
        """The same spreads with every stratum's median taken out: the calibrated prediction becomes the
        raw ETA (to the quantile grid's interpolation about the median), and the stochastic schedule's
        difference from it is the interaction alone (the training days' median error need not carry to
        other days — R3.1 measured that it does not)."""
        return ErrorModel(self.edges_s, {name: e - np.median(e) for name, e in self.residuals.items()})

    def table(self) -> dict[str, Any]:
        return {"edges_s": list(self.edges_s), "strata": {
            name: {"n": len(e), "p10": float(np.percentile(e, 10)), "p50": float(np.median(e)),
                   "p90": float(np.percentile(e, 90))} for name, e in self.residuals.items()}}


def error_grid(residuals: np.ndarray, samples: int, rng: np.random.Generator) -> np.ndarray:
    """A flight's ``samples`` error draws: its stratum's equal-probability quantiles ((m + 0.5) / M) in a
    random order — the marginal is the stratum's, exactly, and only the pairing across flights is drawn."""
    return rng.permutation(np.quantile(residuals, (np.arange(samples) + 0.5) / samples))


def calibration_errors(model, config: TSConfig, normalizer, payload: dict[str, Any], airport: str,
                       manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The expert on its own validation split (day_a's training days), each flight under its landed
    runway, as R2b flew: the head's predicted remaining time at the anchor and its error; and what the
    series build kept of the split (a flight it drops is out of the calibration, counted)."""
    keys = {k for k in payload["split"]["val"] if k.startswith(f"{airport}:")}
    flights = load_flight_dicts(manifest_path, include_flight_keys=keys, verbose=False)
    series, report = build_series(flights, config, airport=airport, aircraft_type=config.aircraft_type)
    print(f"  calibration: {report.format().splitlines()[0]}", flush=True)
    build = {"split_flights": len(keys), "loaded": len(flights), "built": report.built, "skipped": dict(report.skipped)}
    forecasts = forecast_approaches(model, series, config, normalizer)
    points = config.validation_common_grid_points
    rows = []
    for x, f in zip(series, forecasts, strict=True):
        metrics = observed_series_metrics(x, f, points=points)
        rows.append({"dataset_id": x.dataset_id, "predicted_s": float(f.predicted_final_time_s),
                     "error_s": float(metrics["final_time_error_s"])})
    return rows, build


def stochastic_reading(s: Any, top: str) -> dict[str, Any]:
    """One flight's samples as a prediction: the median landing time, the most frequent runway (a tie to
    the head's top one), the delay's median, P(delayed), the q10–q90 interval."""
    counts = Counter(s.runways)
    mode = max(counts, key=lambda runway: (counts[runway], runway == top, runway))
    return {
        "runway": mode,
        "time_s": float(np.median(s.times_s)),
        "delay_s": float(np.median(s.delays_s)),
        "p_delayed": float(np.mean(s.delays_s > DELAYED_S)),
        "interval_s": [float(np.percentile(s.times_s, INTERVAL[0])), float(np.percentile(s.times_s, INTERVAL[1]))],
        "runway_share": counts[mode] / len(s.runways),
    }


def predictions(
    rows: list[dict[str, Any]], samples: dict[str, Any], causal: dict[str, Any], draws: dict[str, np.ndarray],
    model: ErrorModel, deterministic: dict[str, Any],
) -> None:
    """Fill every row's predictions: the stochastic schedule's and its causal form's
    (`stochastic_reading`), the calibrated independent one (the top runway's ETA less the median of the
    flight's own draws — what the stochastic schedule gives a flight nobody interacts with), the raw
    ETA and R3's deterministic slot."""
    for r in rows:
        key = r["flight_key"]
        top = r["independent"]["runway"]
        r["error_stratum"] = model.stratum(r["etas_s"][top] - r["anchor_s"])
        r["stochastic"] = stochastic_reading(samples[key], top)
        r["stochastic_causal"] = stochastic_reading(causal[key], top)
        r["calibrated"] = {"runway": top, "time_s": r["etas_s"][top] - float(np.median(draws[key]))}
        slot = deterministic[key]
        r["deterministic"] = {"runway": slot.runway, "time_s": slot.time_s, "delay_s": slot.delay_s}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stratum in STRATA:
        members = [r for r in rows if stratum == "all" or r["stratum"] == stratum]
        if not members:
            continue
        cell: dict[str, Any] = {"flights": len(members)}
        for name in PREDICTORS:
            signed = [r[name]["time_s"] - r["truth"]["time_s"] for r in members]
            cell[name] = {"runway_accuracy": float(np.mean([r[name]["runway"] == r["truth"]["runway"] for r in members])),
                          "time_error_s": abs_quantiles(signed), "signed_p50_s": float(np.median(signed))}
        moved = [r for r in members if r["stochastic"]["delay_s"] > MOVED_DELAY_S]
        cell["moved"] = {"flights": len(moved), **{
            name: abs_quantiles([r[name]["time_s"] - r["truth"]["time_s"] for r in moved]) for name in PREDICTORS}}
        cell["interval_coverage"] = float(np.mean(
            [r["stochastic"]["interval_s"][0] <= r["truth"]["time_s"] <= r["stochastic"]["interval_s"][1] for r in members]))
        cell["p_delayed_mean"] = float(np.mean([r["stochastic"]["p_delayed"] for r in members]))
        cell["stochastic_runway_changed"] = sum(r["stochastic"]["runway"] != r["independent"]["runway"] for r in members)
        out[stratum] = cell
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--airport", required=True)
    parser.add_argument("--r2", required=True, help="R2b's runway_intent_r2.json (--roster day-val) for this airport")
    parser.add_argument("--checkpoint", required=True, help="the plan expert R2b flew (trained on day_a's training days)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=200, help="M: schedules drawn (plan §18.1: 200)")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--delay-weight-per-s", type=float, default=1.0 / 60.0, help="lambda, as in R3")
    parser.add_argument("--min-probability", type=float, default=0.01, help="epsilon, as in R3")
    parser.add_argument("--center-errors", action="store_true",
                        help="sensitivity: take each stratum's median error out (the calibrated prediction is then "
                             "the raw ETA, and the stochastic schedule differs from it by the interaction alone)")
    args = parser.parse_args(argv)
    if args.samples < 2:
        parser.error("--samples must be at least 2")

    airport = args.airport.upper()
    r2 = json.loads(Path(args.r2).read_text(encoding="utf-8"))
    if r2["airport"] != airport or r2["evaluation"]["roster"] != "day-val":
        parser.error(f"--r2 must be {airport}'s R2b artifact (roster day-val); it is {r2['airport']} / {r2['evaluation']['roster']}")
    if str(Path(args.checkpoint).resolve()) != str(Path(r2["checkpoint"]).resolve()):
        parser.error(f"--checkpoint must be the expert R2b flew ({r2['checkpoint']})")
    config = TSConfig()
    manifest_path, targets, separation, speed_mps, speed_flights = airport_rules(airport, config)
    print(f"{airport}: approach speed {speed_mps:.1f} m/s; 3 NM = {separation.gap_s('x', '', 'x', ''):.1f} s", flush=True)

    model, ckpt_config, normalizer, payload = load_checkpoint(args.checkpoint)
    if ckpt_config.prediction_output != PREDICTION_PLAN:
        parser.error("R3.1 calibrates the PLAN expert")
    provenance = arrival_data_provenance(manifest_path, eligibility_rosters=[default_lateral_pass_roster_path(manifest_path)])
    require_matching_data_provenance(payload, provenance, allow_subset=True)
    model = model.to(resolve_device(args.device))
    calibration, calibration_build = calibration_errors(model, ckpt_config, normalizer, payload, airport, manifest_path)
    overlap = {row["dataset_id"].split(":", 1)[1] for row in calibration} & {f["flight_key"] for f in r2["flights"]}
    if overlap:
        raise RuntimeError(f"{len(overlap)} calibration flights are in the evaluation roster — the two must be disjoint")
    errors = ErrorModel.fit([row["predicted_s"] for row in calibration], [row["error_s"] for row in calibration])
    if args.center_errors:
        errors = errors.centered()
    print(f"  error model: {json.dumps(errors.table())}", flush=True)

    arrivals, rows, known_at = roster(r2, manifest_path)
    mark_busy_hours(rows)
    by_key = {r["flight_key"]: r for r in rows}
    rng = np.random.default_rng(args.seed)
    draws = {}
    for a in arrivals:
        r = by_key[a.key]
        draws[a.key] = error_grid(errors.residuals[errors.stratum(r["etas_s"][r["independent"]["runway"]] - r["anchor_s"])],
                                  args.samples, rng)
    started = time.perf_counter()
    samples = sample_schedules(arrivals, draws, separation, delay_weight_per_s=args.delay_weight_per_s,
                               min_probability=args.min_probability)
    causal = sample_schedules(arrivals, draws, separation, delay_weight_per_s=args.delay_weight_per_s,
                              min_probability=args.min_probability, known_at=known_at)
    print(f"  {args.samples} schedules x 2 in {time.perf_counter() - started:.0f} s", flush=True)
    deterministic = {s.key: s for s in schedule(fcfs_by_eta(arrivals, args.min_probability), separation,
                                                delay_weight_per_s=args.delay_weight_per_s, min_probability=args.min_probability)}
    predictions(rows, samples, causal, draws, errors, deterministic)
    summary = summarise(rows)
    print(json.dumps(summary, indent=1))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runway_intent_r31.json").write_text(json.dumps({
        "schema_version": SCHEMA,
        "airport": airport,
        "r2": str(args.r2),
        "checkpoint": str(args.checkpoint),
        "rules": {"separation": "faa_separation (runway_intent_r3.airport_rules)", "approach_speed_mps": speed_mps,
                  "approach_speed_flights": speed_flights, "delay_weight_per_s": args.delay_weight_per_s,
                  "min_probability": args.min_probability},
        "error_model": {
            "calibration": "the expert's own validation split (day_a training days), each flight under its landed "
                           "runway; e = the head's predicted remaining time - the true one",
            "centered": args.center_errors,
            "calibration_flights": len(calibration),
            "calibration_build": calibration_build,
            "caveat": "the calibration split is also the split the checkpoint was selected on: coverage is measured, "
                      "never guaranteed",
            "strata": "the head's predicted remaining time under its top runway, at the calibration set's terciles; "
                      f"under {MIN_CALIBRATION_FLIGHTS} errors -> pooled",
            "draws": "one error per flight per sample, the same on every runway; a flight's M draws are its stratum's "
                     "M equal-probability quantiles in a random order (error_grid), flights independent",
            **errors.table(),
        },
        "sampling": {"samples": args.samples, "seed": args.seed, "moved_delay_s": MOVED_DELAY_S, "delayed_s": DELAYED_S,
                     "interval_percentiles": list(INTERVAL),
                     "order": "stochastic: FCFS by the drawn arrivals (reads ETAs later-anchored flights have only after "
                              "a flight's own anchor); stochastic_causal: each flight placed at its anchor"},
        "strata": {"busy_per_hour": BUSY_PER_HOUR, "clock": "the truth landing's UTC hour, counted on this roster"},
        "summary": summary,
        "calibration_rows": calibration,
        "flights": rows,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out / 'runway_intent_r31.json'}: {len(rows)} flights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
