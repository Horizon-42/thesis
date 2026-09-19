"""Fly one lockstep protocol (plan §2.7) over the executor's val cohort and write the readout.

    python run_ts.py manoeuvre_lockstep --executor <arm>/checkpoint.pt --codebook <dir> --protocol C \\
        --out <dir> [--batch-size 64] [--limit N] [--device auto] [--write-records]
    python run_ts.py manoeuvre_lockstep --executor … --codebook … --protocol A --prior <dir>/prior.pt --out …
    python run_ts.py manoeuvre_lockstep --executor … --codebook … --protocol A-truth --prior … --out …
    python run_ts.py manoeuvre_lockstep --executor <no-token arm>/checkpoint.pt --protocol none --out …
        [--anchor-remaining-km 12]           # the baseline: no code, no codebook, same rounds and budget
        [--first-prediction-row 59]          # reading (c): every cell starts at the same row of the flight

The coded protocols' three artefacts must be ONE vocabulary: a jointly trained executor's
codebook is the one exported from it (the codebook's ``source.checkpoint_sha256`` is the
executor's), an executor trained against a codebook carries its sha (`codebook_sha256`), and
the prior carries the sha of the codebook it was trained on — a mismatch refuses. Protocol
``none`` (a `plan_conditioning = off` executor) takes no codebook. The cohort is the executor's
val split (rebuilt, provenance verified); the first prediction is made at the executor's fixed anchor L−1, or —
``--anchor-remaining-km X`` (two-tier v3 §3.1's second reading, one of
`anchor_strata.DEFAULT_ANCHOR_GRID_KM`) — the row where each flight has X km of path left to fly
(`lockstep.from_remaining_path`; a flight that never has an admissible row there is counted,
not flown; the row's ``first_prediction_row`` is that row in the whole flight, while a record written
under the bin reading carries the CUT flight's anchor, L−1, as its ``anchorIndex``).
``--first-prediction-row N`` (v3's reading (c)) starts every flight at row N of the whole flight
instead, so cells of different lookback fly the SAME segment and differ only in what they saw
(`lockstep.from_row`; N is at or after the executor's own first row, and a flight without the
executor's horizon of truth after N is counted, not flown). Writes
``manoeuvre_lockstep.json`` (per-flight rows included) and ``manoeuvre_lockstep.txt`` under
``--out`` (refused if it exists); ``--write-records`` adds the flown paths as a predict-shaped
record directory under ``<out>/records/``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from ts_transformer.backbone.adapters import resolve_device
from ts_transformer.config import default_anchor as default_anchor_of
from ts_transformer.data.anchor_strata import DEFAULT_ANCHOR_GRID_KM
from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_SHORT, strata_masks
from ts_transformer.experiments.support import REPO_ROOT, rebuild_cohort
from ts_transformer.inference.export import build_prediction_record, write_batch
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.manoeuvre import lockstep as ls
from ts_transformer.manoeuvre.context import TypeVocabulary
from ts_transformer.manoeuvre.prior import ManoeuvrePrior, PriorConfig
from ts_transformer.manoeuvre.readout import STRATA
from ts_transformer.manoeuvre.tokenizer import load_codebook
from ts_transformer.run_naming import run_display_name
from ts_transformer.training.train import load_checkpoint

#: v2 (2026-09-18, two-tier v3): `first_prediction` names where the closed loop starts, every row
#: carries `first_prediction_row`, the per-round record is `rounds` (v1: `asks_e`) and the round
#: counts `predictions` / `held_predictions` (v1: `asks` / `held_asks`), a protocol-none row has no
#: code columns and its codebook keys are null.
LOCKSTEP_SCHEMA = "ts-manoeuvre-lockstep-v2"
#: The summary block a written record directory carries.
LOCKSTEP_RECORDS_BLOCK = "manoeuvre_lockstep"


def load_prior(path: Path, device: torch.device) -> tuple[ls.Prior, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = ManoeuvrePrior(PriorConfig.from_dict(payload["prior_config"]))
    model.load_state_dict(payload["state_dict"])
    return ls.Prior(model=model.to(device).eval(), vocabulary=TypeVocabulary.from_dict(payload["vocabulary"])), payload


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
            "e_plan_by_round_p50_m": _by_round(chosen, "e_plan_m"),
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
        f"segment {payload['segment_s']:g} s · first prediction {first['rule']}"
        + (f" ({first['flights_without_a_row']} flights not flown: no admissible row)" if first["flights_without_a_row"] else "")
        + (f" · prior {payload['prior']}" if payload.get("prior") else ""),
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
    if pooled["e_plan_by_round_p50_m"]:
        lines.append("e_plan  p50 by round: " + "  ".join(f"r{k} {v:.0f}" for k, v in pooled["e_plan_by_round_p50_m"].items()))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, default=None, help="the coded protocols' vocabulary; none under protocol none")
    parser.add_argument("--protocol", required=True, choices=ls.PROTOCOLS)
    parser.add_argument("--prior", type=Path, default=None)
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--anchor-remaining-km", type=int, default=None, choices=DEFAULT_ANCHOR_GRID_KM,
                       help="start the closed loop where each flight has this much path left to fly, instead of at L-1")
    start.add_argument("--first-prediction-row", type=int, default=None,
                       help="start the closed loop at this row of every flight (reading (c): the same segment for every lookback), instead of at L-1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", default="val", choices=("train", "val"),
                        help="val (every readout); train ONLY as the closed-loop training's input (protocol C, plan §2.7)")
    parser.add_argument("--limit", type=int, default=0, help="a PREFIX of the cohort (a smoke test)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--write-records", action="store_true")
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a lockstep readout is never overwritten")
    if (args.protocol in (ls.PROTOCOL_A, ls.PROTOCOL_A_TRUTH)) != (args.prior is not None):
        parser.error("protocols A and A-truth take --prior; protocols C and none take none")
    if (args.protocol == ls.PROTOCOL_NONE) != (args.codebook is None):
        parser.error("protocol none takes no --codebook (a no-token executor has no vocabulary); every coded protocol takes one")
    if args.split == "train" and args.protocol != ls.PROTOCOL_C:
        parser.error("the train split is flown under protocol C only (the closed-loop training's input); a readout is val")
    device = resolve_device(args.device)
    started = time.perf_counter()

    executor_path = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    model, config, normalizer, payload = load_checkpoint(executor_path)
    executor = ls.Executor(model=model.to(device).eval(), config=config, normalizer=normalizer)
    executor_sha = file_sha256(executor_path)
    codebook = None
    if args.codebook is not None:
        codebook = load_codebook(args.codebook if args.codebook.is_absolute() else REPO_ROOT / args.codebook)
        bound = model.codebook_sha256 == codebook.sha256 if config.manoeuvre_codebook else codebook.source.get("checkpoint_sha256") == executor_sha
        if not bound:
            parser.error(f"{codebook.path} is not this executor's codebook (an executor trained against a codebook carries its sha; "
                         "a jointly trained one is the codebook's source)")
    prior = prior_payload = None
    if args.prior is not None:
        prior, prior_payload = load_prior(args.prior if args.prior.is_absolute() else REPO_ROOT / args.prior, device)
        if prior_payload["codebook_sha256"] != codebook.sha256:
            parser.error(f"{args.prior} was trained on codebook {prior_payload['codebook_sha256'][:12]}…, not {codebook.sha256[:12]}…")
        # the prior must never have seen this executor's val flights (§6.3) — several executors
        # can share one frozen codebook: its train set is disjoint from this val set, and its
        # own val (a prefix of it under --limit) lies inside it
        executor_val = set(payload["split"]["val"])
        seen = executor_val & set(prior_payload["split"]["train"])
        if seen or not set(prior_payload["split"]["val"]) <= executor_val:
            parser.error(f"{args.prior} was trained on another split than this executor's ({len(seen)} of its train "
                         "flights are in this val set, or its val is not inside it): the prior may have seen these val flights")
        if int(prior_payload["anchor"]) != default_anchor_of(config) or float(prior_payload["segment_s"]) != config.control_horizon_s:
            parser.error(f"{args.prior} was trained at anchor {prior_payload['anchor']} / segment {prior_payload['segment_s']:g} s, "
                         f"not this executor's {default_anchor_of(config)} / {config.control_horizon_s:g} s")
    wanted = payload["split"][args.split][: args.limit] if args.limit else payload["split"][args.split]
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
    print(f"  {args.protocol}: {len(series)} flights, first prediction {first_prediction['rule']}, {run_display_name(config.to_dict())}", flush=True)

    runs = ls.fly(executor, codebook, series, args.protocol, prior=prior, device=device, batch_size=args.batch_size,
                  log=lambda line: print(line, flush=True))
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
        "schema": LOCKSTEP_SCHEMA, "written_utc": utc_now(), "protocol": args.protocol,
        "executor": str(executor_path), "executor_sha256": executor_sha, "executor_name": run_display_name(config.to_dict()),
        "codebook": None if codebook is None else str(codebook.path), "codebook_sha256": None if codebook is None else codebook.sha256,
        "prior": None if args.prior is None else str(args.prior),
        "prior_sha256": None if args.prior is None else file_sha256(args.prior if args.prior.is_absolute() else REPO_ROOT / args.prior),
        "prior_continuous": None if prior is None else prior.model.config.continuous,
        "prior_executor_sha256": None if prior_payload is None else prior_payload["executor_sha256"],
        "prior_trained_on_this_executor": None if prior_payload is None else prior_payload["executor_sha256"] == executor_sha,
        "segment_s": config.control_horizon_s, "anchor": a0, "first_prediction": first_prediction, "split": args.split, "limit": args.limit or None,
        "flights": len(rows), "flights_without_a_leg": flown_none,
        "budget_rule": "T0 + max(30 s, 0.1·T0), T0 = the truth's duration at the first prediction (a cap; under A the prior's landed decides)",
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
