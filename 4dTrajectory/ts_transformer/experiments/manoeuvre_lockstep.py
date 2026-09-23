"""Fly the no-token closed loop (two-tier v3 §3.1) over the executor's val cohort and write the readout.

    python run_ts.py manoeuvre_lockstep --executor <arm>/checkpoint.pt --out <dir> \\
        [--batch-size 64] [--limit N] [--device auto] [--write-records]
        [--anchor-remaining-km 12]           # reading (b): each flight first seen at that remaining path
        [--first-prediction-row 59]          # reading (c): every cell starts at the same row of the flight
        [--execute-s 20]                     # A3-a: forecast the whole horizon, fly only its first 20 s, predict again
        [--cohort <development_cohort.json>] # v3 stage B0: only that cohort's flights of the split (a subset of the checkpoint's)

The intent-code protocols (``--protocol C | A | A-truth``, ``--codebook``, ``--prior``,
``--prior-landing-ends-flight``) are ARCHIVED 2026-09-20 (`archive/manoeuvre_codes_2026_09/`):
there is ONE protocol now, ``none`` — a `plan_conditioning = off` executor predicting again every
round on its own flown rows — and the payload still carries its name, because the gates refuse a
payload flown under anything else.

The cohort is the executor's val split (rebuilt, provenance verified); the first prediction is made
at the executor's fixed anchor L−1, or — ``--anchor-remaining-km X`` (one of
`anchor_strata.DEFAULT_ANCHOR_GRID_KM`) — the row where each flight has X km of path left to fly
(`lockstep.from_remaining_path`; a flight that never has an admissible row there is counted,
not flown; the row's ``first_prediction_row`` is that row in the whole flight, while a record written
under the bin reading carries the CUT flight's anchor, L−1, as its ``anchorIndex``).
``--first-prediction-row N`` (reading (c)) starts every flight at row N of the whole flight
instead, so cells of different lookback fly the SAME segment and differ only in what they saw
(`lockstep.from_row`; N is at or after the executor's own first row, and a flight without the
executor's horizon of truth after N is counted, not flown). Writes
``manoeuvre_lockstep.json`` (per-flight rows included) and ``manoeuvre_lockstep.txt`` under
``--out`` (refused if it exists); ``--write-records`` adds the flown paths as a predict-shaped
record directory under ``<out>/records/``. ``--cohort`` (two-tier v3 stage B0, D33) restricts
the split to a development cohort's roster of it, in the checkpoint's order — the baseline
re-read on the flights every stage B arm shares; a cohort flight the checkpoint's split does
not hold refuses (`cohort_keys`), so the reading is never a silent superset.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time
from typing import Any

import numpy as np

from ts_transformer.backbone.adapters import resolve_device
from ts_transformer.config import default_anchor as default_anchor_of
from ts_transformer.data.anchor_strata import DEFAULT_ANCHOR_GRID_KM
from ts_transformer.data.approach_difficulty import (
    STRATUM_ALL, STRATUM_ESTABLISHED, STRATUM_SHORT, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, strata_masks,
)
from ts_transformer.data.development_cohorts import DevelopmentCohort, development_cohort_audit, load_development_cohort
from ts_transformer.experiments.support import REPO_ROOT, rebuild_cohort
from ts_transformer.inference.export import build_prediction_record, write_batch
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.manoeuvre import lockstep as ls
from ts_transformer.run_naming import run_display_name
from ts_transformer.training.train import load_checkpoint

#: The strata every row is read in (the readout's order; `data/approach_difficulty` owns the names).
STRATA = (STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, STRATUM_ESTABLISHED)

#: v2 (2026-09-18, two-tier v3): `first_prediction` names where the closed loop starts, every row
#: carries `first_prediction_row`, the per-round record is `rounds` (v1: `asks_e`) and the round
#: count is `predictions` (v1: `asks`).
#: `executed_s` (2026-09-19, A3-a): the seconds of each forecast flown before the next prediction — the
#: horizon unless ``--execute-s``; `first_prediction.common_row` (2026-09-19, reading (c)).
#: v3 (2026-09-19, stage B) carried the intent-code token and prior columns; v4 (2026-09-20) is the
#: same payload with the intent-code layer ARCHIVED — no codebook / prior / token / landing keys, no
#: code columns per row, no `e_plan` per round. A v3 payload is not read by this code (the gates
#: refuse it by schema): there is no compatibility path.
LOCKSTEP_SCHEMA = "ts-manoeuvre-lockstep-v4"
#: The summary block a written record directory carries.
LOCKSTEP_RECORDS_BLOCK = "manoeuvre_lockstep"


def cohort_keys(payload: dict[str, Any], split: str, cohort: DevelopmentCohort) -> list[str]:
    """The checkpoint's ``split`` flights that ``cohort``'s roster of that split holds, in the
    checkpoint's order; a cohort flight the checkpoint does not hold refuses (the cohort is a
    subset of the executor's own split, never another population)."""
    roster = set(cohort.train_flight_ids if split == "train" else cohort.val_flight_ids)
    held = list(payload["split"][split])
    missing = roster - set(held)
    if missing:
        raise ValueError(f"{len(missing)} flight(s) of cohort {cohort.name!r} ({split}) are not in the executor's {split} split "
                         f"(first: {sorted(missing)[0]!r}): a lockstep cohort is a subset of the executor's own split")
    return [key for key in held if key in roster]


def _p50(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def stratum_table(rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keys = list(rows)
    masks = strata_masks(rows, keys)
    out: dict[str, dict[str, Any]] = {}
    for stratum in STRATA:
        chosen = [rows[key] for key, keep in zip(keys, masks[stratum]) if keep]
        if not chosen:
            out[stratum] = {"n": 0}
            continue
        out[stratum] = {
            "n": len(chosen),
            "ade_mean_m": float(np.mean([r["ade_m"] for r in chosen])),
            "ade_p50_m": _p50([r["ade_m"] for r in chosen]),
            "fde_p50_m": _p50([r["fde_m"] for r in chosen]),
            "chamfer_p50_m": _p50([r["chamfer_m"] for r in chosen]),
            "final_time_error_mae_s": float(np.mean([abs(r["final_time_error_s"]) for r in chosen])),
            "fully_flyable_share": float(np.mean([r["reference"]["fully_flyable"] for r in chosen])),
            "established_share": float(np.mean([r["reference"]["established"] for r in chosen])),
            "at_p50_m": {lead: _p50([r["at"][lead] for r in chosen]) for lead in chosen[0]["at"]},
            "at_n": {lead: sum(r["at"][lead] is not None for r in chosen) for lead in chosen[0]["at"]},
            "ended": dict(Counter(r["ended"] for r in chosen)),
            "predictions_p50": _p50([r["predictions"] for r in chosen]),
            "e_track_by_round_p50_m": _by_round(chosen, "e_track_m"),
        }
    return out


def _by_round(rows, key) -> dict[str, float | None]:
    by_round: dict[int, list[float]] = {}
    for row in rows:
        for record in row["rounds"]:
            if record.get(key) is not None:
                by_round.setdefault(record["round"], []).append(record[key])
    return {str(k): float(np.median(v)) for k, v in sorted(by_round.items())}


def _lead(value: float | None) -> str:
    return f"{value:>7.0f}" if value is not None else f"{'—':>7}"


def render(payload: dict[str, Any]) -> str:
    first = payload["first_prediction"]
    lines = [
        f"manoeuvre lockstep · {payload['protocol']} · {payload['executor_name']} · {payload['flights']} flights · "
        f"segment {payload['segment_s']:g} s"
        + (f" (flies the first {payload['executed_s']:g} s of each)" if payload["executed_s"] != payload["segment_s"] else "")
        + f" · first prediction {first['rule']}"
        + (f" ({first['flights_without_a_row']} flights not flown: no admissible row)" if first["flights_without_a_row"] else ""),
        "",
        f"{'stratum':<14}{'n':>6}{'ADE mean':>10}{'ADE p50':>9}{'FDE p50':>9}{'flyable':>9}{'estab':>8}"
        f"{'e60':>7}{'e120':>7}{'e180':>7}{'e300':>7}{'preds':>6}  ended · n at each lead",
        "  (the leads hold the forecast's last row past its end; a lead is absent only where the truth has ended)",
    ]
    for stratum in STRATA:
        cell = payload["strata"][stratum]
        if not cell["n"]:
            continue
        at, at_n = cell["at_p50_m"], cell["at_n"]
        lines.append(
            f"{STRATUM_SHORT[stratum]:<14}{cell['n']:>6}{cell['ade_mean_m']:>10.0f}{cell['ade_p50_m']:>9.0f}{cell['fde_p50_m']:>9.0f}"
            f"{cell['fully_flyable_share']:>9.3f}{cell['established_share']:>8.3f}"
            f"{_lead(at['60'])}{_lead(at['120'])}{_lead(at['180'])}{_lead(at['300'])}{cell['predictions_p50']:>6.0f}  "
            + ", ".join(f"{k} {v}" for k, v in sorted(cell["ended"].items()))
            + f" · n {at_n['60']}/{at_n['120']}/{at_n['180']}/{at_n['300']}"
        )
    pooled = payload["strata"][STRATUM_ALL]
    lines.append("")
    lines.append("e_track p50 by round: " + "  ".join(f"r{k} {v:.0f}" for k, v in pooled["e_track_by_round_p50_m"].items()))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--executor", type=Path, required=True)
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--anchor-remaining-km", type=int, default=None, choices=DEFAULT_ANCHOR_GRID_KM,
                       help="start the closed loop where each flight has this much path left to fly, instead of at L-1")
    start.add_argument("--first-prediction-row", type=int, default=None,
                       help="start the closed loop at this row of every flight (reading (c): the same segment for every lookback), instead of at L-1")
    parser.add_argument("--execute-s", type=float, default=None,
                        help="fly only the first N seconds of each forecast and predict again there (A3-a)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", default="val", choices=("train", "val"),
                        help="val (every readout); train as the closed-loop training's input")
    parser.add_argument("--limit", type=int, default=0, help="a PREFIX of the cohort (a smoke test)")
    parser.add_argument("--cohort", type=Path, default=None,
                        help="a development cohort file: fly only its roster of --split (a subset of the checkpoint's split)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--write-records", action="store_true")
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a lockstep readout is never overwritten")
    device = resolve_device(args.device)
    started = time.perf_counter()

    executor_path = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    model, config, normalizer, payload = load_checkpoint(executor_path)
    executor = ls.Executor(model=model.to(device).eval(), config=config, normalizer=normalizer)
    executor_sha = file_sha256(executor_path)
    executed_s = ls.executed_step_s(config, args.execute_s)      # a bad step is refused here, before the cohort is rebuilt
    cohort = None
    wanted = list(payload["split"][args.split])
    if args.cohort is not None:
        cohort_path = args.cohort if args.cohort.is_absolute() else REPO_ROOT / args.cohort
        cohort = load_development_cohort(cohort_path)
        try:
            wanted = cohort_keys(payload, args.split, cohort)
        except ValueError as exc:
            parser.error(str(exc))
    if args.limit:
        wanted = wanted[: args.limit]
    series = rebuild_cohort(payload, config, wanted)
    a0 = default_anchor_of(config)
    first_rows = {item.dataset_id: a0 for item in series}
    if args.anchor_remaining_km:
        series, first_rows = ls.from_remaining_path(series, config, args.anchor_remaining_km * 1000.0)
        if not series:
            parser.error(f"no flight of the cohort has an admissible row at {args.anchor_remaining_km} km of remaining path")
    elif args.first_prediction_row is not None:
        if args.first_prediction_row < a0:
            parser.error(f"--first-prediction-row {args.first_prediction_row} is before this executor's first possible prediction (row {a0})")
        series, first_rows = ls.from_row(series, config, args.first_prediction_row)
        if not series:
            parser.error(f"no flight of the cohort has the executor's horizon of truth after row {args.first_prediction_row}")
    first_prediction = {
        "rule": (f"remaining path {args.anchor_remaining_km} km" if args.anchor_remaining_km
                 else f"fixed row {args.first_prediction_row} (common start)" if args.first_prediction_row is not None
                 else f"fixed L-1 (row {a0})"),
        "remaining_km": args.anchor_remaining_km, "common_row": args.first_prediction_row,
        "flights_without_a_row": len(wanted) - len(series),
    }
    print(f"  {ls.PROTOCOL_NONE}: {len(series)} flights, first prediction {first_prediction['rule']}, {run_display_name(config.to_dict())}", flush=True)

    runs = ls.fly(executor, series, device=device, batch_size=args.batch_size,
                  log=lambda line: print(line, flush=True), execute_s=args.execute_s)
    rows: dict[str, dict[str, Any]] = {}
    pairs = []
    for index, run in enumerate(runs):
        if not run.legs:
            continue
        row, metrics = ls.flight_row(run, points=config.validation_common_grid_points)
        row["first_prediction_row"] = first_rows[run.series.dataset_id]
        rows[run.series.dataset_id] = row
        if args.write_records:
            forecast = ls.whole_forecast(run)
            pairs.append((build_prediction_record(run.series, forecast, index=index, model_name=config.model,
                                                  horizon_mode=config.horizon_mode, split=args.split), metrics))
    flown_none = len(runs) - len(rows)
    payload_out = {
        "schema": LOCKSTEP_SCHEMA, "written_utc": utc_now(), "protocol": ls.PROTOCOL_NONE,
        "executor": str(executor_path), "executor_sha256": executor_sha, "executor_name": run_display_name(config.to_dict()),
        "segment_s": config.control_horizon_s, "executed_s": executed_s,
        "anchor": a0, "first_prediction": first_prediction, "split": args.split, "limit": args.limit or None,
        "cohort": None if cohort is None else {**development_cohort_audit(cohort_path, cohort), "path": str(cohort_path), "flown": len(wanted)},
        "flights": len(rows), "flights_without_a_leg": flown_none,
        "budget_rule": "T0 + max(30 s, 0.1·T0), T0 = the truth's duration at the first prediction (a cap)",
        "strata": stratum_table(rows), "rows": rows, "elapsed_s": time.perf_counter() - started,
    }
    out.mkdir(parents=True)
    write_json_atomic(out / "manoeuvre_lockstep.json", payload_out)
    table = render(payload_out)
    (out / "manoeuvre_lockstep.txt").write_text(table, encoding="utf-8")
    print(table)
    if pairs:
        write_batch([r for r, _ in pairs], output_dir=out / "records", config_dict=config.to_dict(),
                    flight_metrics=[m for _, m in pairs], checkpoint=str(executor_path), split=args.split,
                    extra_summary={LOCKSTEP_RECORDS_BLOCK: {k: v for k, v in payload_out.items() if k not in ("rows", "strata")}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
