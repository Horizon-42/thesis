#!/usr/bin/env python
"""L0 — how many operating parameters does the decoder have to emit?

Latent-intent design (`4dTrajectory/ts_transformer/docs/2026-09-07_latent_intent_design.zh.md`)
§六 L0. The control head's 64 piecewise-constant segments (257 free numbers with its
duration logits) is an untested default; the collocation optimiser flies a whole arrival to
a pinned terminal state on 8 segments (`DEFAULT_N_SEGMENTS`, 25 numbers) and the closure
decoder draws one on 14. Those are "find A trajectory" numbers, not "reproduce THIS observed
track" numbers — this runner measures the second.

For every flight of a reference arm's cohort, at each segment count N and each duration
mode, it FITS a bounded piecewise-constant schedule to the observed track by direct
shooting through the package's own differentiable rollout, and reports the error that
remains. The result is the curve ADE(N): the decoder's width is the smallest N whose
representation error is negligible against the 962 m of intent the ego history cannot see.

The total duration is GIVEN (the truth's), so the fit measures the path alone. Two error
columns are reported for every arm, because they are two different questions and only one
of them is what the fit minimised:

* `fixed-dt ADE` — 3-D position error on the 2 s supervision grid. THIS is the fit's own
  objective and the value each flight's best step was chosen on.
* `ADE` / `FDE` — `common_physical_time_flight_metrics`, the readouts' own numbers, so an
  arm here sits beside a trained arm's `summary.json`.

An under-converged fit can only INFLATE the error, i.e. only produce the false answer "N is
not enough". So every arm also reports what the seed alone was worth (`seed ADE`: the
inverse dynamics of the truth track at the segment midpoints, before any update), the
`gain` over it, and the share of flights still improving when the step budget ran out.
Read the gate only when that share is small.

Truth = the post-anchor supervision rows (observed rows plus the fitted tail), as in the
closure oracles, so the numbers sit beside theirs. Cohort, anchor and strata come from the
reference arm's `summary.json`. Validation-split measurement, no training, nothing written
back to the data plane; the output directory is an immutable artifact and must not exist.

    python run_ts_control_basis_oracle.py --airport KRDU \\
        --reference 4dTrajectory/outputs/KRDU/experiments/control_procedure_20260905/A_control_v3_pred_val \\
        --out 4dTrajectory/outputs/KRDU/experiments/l0_control_basis_20260907 \\
        [--segments 4,8,16,32,64] [--duration-modes uniform,free] [--limit 400] [--steps 600]
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
for path in (TS_DIR, REPO_ROOT / "geokit" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

import geometric_metrics as gm  # noqa: E402
from approach_difficulty import STRATUM_ALL, STRATUM_VECTORED, strata_masks  # noqa: E402
from channels import POSITION_IDX  # noqa: E402
from config import (  # noqa: E402
    CONTROL_HOOK_OFF,
    CONTROL_RECIPE_CUSTOM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    INTENT_CONDITIONING_NONE,
    PREDICTION_CONTROL,
    TSConfig,
)
from control.loss.fixed_dt import fixed_dt_control_state_loss  # noqa: E402
from control.oracle.basis import (  # noqa: E402
    DEFAULT_LEARNING_RATE_FLOOR,
    DURATION_MODES,
    BasisSchedule,
    fit_basis_schedules,
    free_number_count,
    inverse_dynamics_seed,
    width_scaled_learning_rate,
)
from dataset import (  # noqa: E402
    FixedAnchorTrajectoryWindows,
    Normalizer,
    build_series,
    arrival_data_provenance,
    load_flight_dicts,
    provenance_manifest_digests,
)
from flight_scenarios.identity import flight_key  # noqa: E402
from metrics import common_physical_time_flight_metrics  # noqa: E402
from models import resolve_device  # noqa: E402
from physical_criteria import fixed_dt_position_ade_m  # noqa: E402
from prediction_outputs import ControlPrediction  # noqa: E402
import run_ts_pipeline as pipeline  # noqa: E402

RESULT_SCHEMA = "l0-control-basis-oracle-v1"
# §六 L0's gate: the representation error must be negligible against the 962 m of intent the
# ego history cannot see, at a width no larger than the optimiser's own control mesh.
GATE_ADE_M = 200.0
GATE_MAX_SEGMENTS = 16
# The outer-test split stays sealed (repo experiment rule); a reference arm scored on it
# would open those tracks here.
FORBIDDEN_SPLIT = "test"

# The reference arm's recipe describes a TRAINED model; this diagnostic has no network, no
# hook and no penalty, and scores on the fixed-dt grid. Only the DATA contract (frame,
# channels, dt, seq_len, split) is inherited.
_RECIPE_OVERRIDES = {
    "control_recipe_name": CONTROL_RECIPE_CUSTOM,  # a named recipe freezes width and grid
    "prediction_output": PREDICTION_CONTROL,
    "control_state_loss_grid": CONTROL_STATE_LOSS_GRID_FIXED_DT,
    # The fit's objective is the physical ADE below; this field only has to be one the
    # fixed-dt grid accepts, because the training objective is never called.
    "control_state_objective": CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    "control_state_supervision_clock": CONTROL_STATE_CLOCK_OBSERVED,
    "control_command_hook": CONTROL_HOOK_OFF,
    # No loss runs here, and a non-zero imitation weight would make the dataset solve the
    # per-flight inverse for a teacher target nothing reads, on every one of the arms.
    "control_imitation_loss_weight": 0.0,
    "procedure_loss_lateral_weight": 0.0,
    "procedure_loss_vertical_weight": 0.0,
    "intent_conditioning": INTENT_CONDITIONING_NONE,
    "closure_labels_path": None,
    "random_train_anchor": False,
    "dropout": 0.0,
}


def basis_config(config_dict: dict, n_segments: int, device: str) -> TSConfig:
    """The reference arm's data contract at width ``n_segments``."""
    known = {field.name for field in fields(TSConfig)}
    payload = {name: value for name, value in config_dict.items() if name in known}
    payload.update(_RECIPE_OVERRIDES)
    payload["n_segments"] = int(n_segments)
    payload["device"] = device
    return TSConfig(**payload)


def summary_row_key(row: dict) -> str:
    """The readout join key for one scored row (`flight_key` = id_runway_icao24_landing)."""
    return "_".join(
        str(row.get(name) or "") for name in ("id", "runway", "icao24", "landing_time_utc")
    )


def cohort(args: argparse.Namespace, out: Path):
    """The reference arm's flights, its config dict, and the strata aligned to them."""
    reference_dir = args.reference if args.reference.is_absolute() else REPO_ROOT / args.reference
    summary = json.loads((reference_dir / "summary.json").read_text())
    split = summary.get("split")
    if split == FORBIDDEN_SPLIT:
        raise SystemExit(f"{reference_dir} was scored on the sealed {FORBIDDEN_SPLIT} split")
    scored = summary["results"]
    reference = {
        summary_row_key(row): row
        for row in scored
        if row.get("ade_m") is not None and row.get("route_tortuosity") is not None
    }
    keys = sorted(reference)
    compact_of = {key: flight_key(reference[key], 0) for key in keys}
    if len(set(compact_of.values())) != len(keys):
        raise RuntimeError("compact flight keys collide across the reference cohort")
    coverage = {
        "reference_rows": len(scored),
        "scored_rows": len(keys),
        "dropped_unscored_rows": len(scored) - len(keys),
        "limit": int(args.limit),
    }
    if args.limit and args.limit < len(keys):
        rng = np.random.default_rng(args.seed)
        keys = sorted(rng.choice(np.array(keys, dtype=object), size=args.limit, replace=False))
    coverage["measured_flights"] = len(keys)
    masks = strata_masks(reference, keys)
    return reference_dir, summary, keys, compact_of, masks, coverage


def build_cohort_series(keys: list[str], compact_of: dict[str, str], airport: str, config: TSConfig):
    """The built series and their readout keys, in the SERIES' order (not the cohort's)."""
    wanted = {compact_of[key] for key in keys}
    manifest = pipeline.arrival_manifest_path(airport)
    flights = load_flight_dicts(
        [manifest],
        include_flight_keys={f"{airport}:{compact}" for compact in wanted},
        verbose=False,
    )
    series, report = build_series(flights, config, airport=airport)
    if report.built != len(wanted):
        raise RuntimeError(f"built {report.built} of {len(wanted)} flights:\n{report.format()}")
    key_of_compact = {compact: key for key, compact in compact_of.items()}
    return series, [key_of_compact[item.flight_id] for item in series], manifest


def score_flights(
    schedule: BasisSchedule, *, supervision, dynamics, config, normalizer,
    series_batch, anchor: int, keys: list[str], seed_clipped: np.ndarray, fit,
) -> list[dict]:
    """Score the fitted schedules the way the readouts do, plus the fit's own diagnostics."""
    with torch.no_grad():
        prediction = schedule()
        rollout = fixed_dt_control_state_loss(prediction, supervision, config, normalizer, dynamics)
        fixed_dt_ade = fixed_dt_position_ade_m(
            rollout.physical_query_states, supervision, normalizer
        )
    restored = torch.max(torch.abs(fixed_dt_ade - fit.best_value))
    if restored > 1e-6:
        raise RuntimeError(
            f"the restored best state does not reproduce its own objective (max {float(restored):.3g} m)"
        )
    states = rollout.physical_query_states.cpu().numpy().astype(np.float64)
    endpoints = normalizer.decode(
        rollout.normalized_segment_end_states[:, -1].cpu().numpy().astype(np.float64)
    )
    offsets = supervision.query_offsets_s.cpu().numpy().astype(np.float64)
    valid = supervision.valid.cpu().numpy()
    controls = prediction.controls.cpu().numpy().astype(np.float64)
    durations = prediction.segment_durations.cpu().numpy().astype(np.float64)
    lower = dynamics["control_lower"].cpu().numpy().astype(np.float64)
    upper = dynamics["control_upper"].cpu().numpy().astype(np.float64)
    seed_value = fit.seed_value.cpu().numpy().astype(np.float64)
    tail_gain = fit.tail_gain.cpu().numpy().astype(np.float64)

    rows = []
    for row, (series, key) in enumerate(zip(series_batch, keys)):
        active = valid[row]
        truth_values = series.supervision_values[anchor + 1:]
        truth_offsets = series.supervision_times[anchor + 1:] - series.times[anchor]
        final_time_s = float(durations[row].sum())
        # The fixed-dt queries stop at the last complete 2 s node, up to dt short of the
        # trajectory's own end; the last SEGMENT end is exactly at it, so the endpoint the
        # FDE is read at is the rollout's real terminal state, not a held sample. When the
        # duration IS a multiple of dt the last query sits at that same instant — the
        # segment end replaces it rather than repeating the clock.
        before_end = offsets[row][active] < final_time_s - 1e-9
        predicted = np.concatenate([states[row][active][before_end], endpoints[row][None, :]], axis=0)
        predicted_offsets = np.concatenate([offsets[row][active][before_end], [final_time_s]])
        metrics = common_physical_time_flight_metrics(
            anchor_values=series.values[anchor],
            predicted_values=predicted,
            predicted_offsets_s=predicted_offsets,
            predicted_final_time_s=final_time_s,
            truth_values=truth_values,
            truth_offsets_s=truth_offsets,
            true_final_time_s=float(truth_offsets[-1]),
        )
        anchor_xy = series.values[anchor][None, :2]
        truth_xy = np.concatenate([anchor_xy, truth_values[:, :2]], axis=0)
        predicted_xy = np.concatenate([anchor_xy, predicted[:, list(POSITION_IDX)][:, :2]], axis=0)
        unit = (controls[row] - lower[row]) / (upper[row] - lower[row])
        rows.append({
            "flight_key": key,
            "ade_m": metrics["ade_m"],
            "fde_m": metrics["fde_m"],
            "fixed_dt_ade_m": float(fixed_dt_ade[row]),
            "seed_fixed_dt_ade_m": float(seed_value[row]),
            "chamfer_m": gm.chamfer_m(predicted_xy, truth_xy),
            "frechet_m": gm.discrete_frechet_m(predicted_xy, truth_xy),
            "saturated_fraction": float(np.mean((unit <= 0.01) | (unit >= 0.99))),
            "seed_clipped_fraction": float(seed_clipped[row]),
            "segment_duration_min_s": float(durations[row].min()),
            "segment_duration_max_s": float(durations[row].max()),
            "best_step": int(fit.best_step[row]),
            "tail_gain": float(tail_gain[row]),
            "controls": controls[row].tolist(),
            "segment_durations_s": durations[row].tolist(),
        })
    return rows


def _p(values, quantile):
    return float(np.percentile(np.asarray(values, dtype=float), quantile))


def summarise(rows: list[dict], masks: dict[str, np.ndarray], cohort_keys: list[str]) -> dict:
    """Aggregate by stratum. ``cohort_keys`` is the order ``masks`` was built against —
    the rows may arrive in any order, which is why they are looked up by key."""
    by_key = {row["flight_key"]: row for row in rows}
    missing = set(by_key) - set(cohort_keys)
    if missing:
        raise RuntimeError(f"{len(missing)} scored flights are outside the cohort the masks cover")
    out = {}
    for stratum, mask in masks.items():
        selected = [by_key[key] for key, member in zip(cohort_keys, mask) if member and key in by_key]
        if not selected:
            continue
        out[stratum] = {
            "n": len(selected),
            "ade_mean_m": float(np.mean([row["ade_m"] for row in selected])),
            "ade_p50_m": _p([row["ade_m"] for row in selected], 50),
            "fde_p50_m": _p([row["fde_m"] for row in selected], 50),
            "fixed_dt_ade_mean_m": float(np.mean([row["fixed_dt_ade_m"] for row in selected])),
            "seed_fixed_dt_ade_mean_m": float(
                np.mean([row["seed_fixed_dt_ade_m"] for row in selected])
            ),
            "chamfer_p50_m": _p([row["chamfer_m"] for row in selected], 50),
            "frechet_p50_m": _p([row["frechet_m"] for row in selected], 50),
            "saturated_p50": _p([row["saturated_fraction"] for row in selected], 50),
            "tail_gain_p50": _p([row["tail_gain"] for row in selected], 50),
            "tail_gain_p90": _p([row["tail_gain"] for row in selected], 90),
        }
    return out


def fit_arm(
    *, series, series_keys, cohort_keys, masks, config, normalizer, anchor,
    n_segments, duration_mode, args, device,
) -> tuple[dict, list[dict]]:
    """One (N, duration mode) arm over the whole cohort, batch by batch."""
    started = time.time()
    windows = FixedAnchorTrajectoryWindows(series, config, normalizer)
    if len(windows) != len(series):
        raise RuntimeError(
            f"the fixed-anchor dataset holds {len(windows)} windows for {len(series)} flights"
        )
    rows: list[dict] = []
    clip_shares: list[float] = []
    for start in range(0, len(series), args.batch_size):
        indices = np.arange(start, min(start + args.batch_size, len(series)))
        _x, _target, _weights, final_time, _flight_weights, dynamics, supervision = windows.batch(indices)
        dynamics = {name: value.to(device) for name, value in dynamics.items()}
        supervision = supervision.to(device)
        series_batch = [series[int(index)] for index in indices]
        seed, seed_clipped = inverse_dynamics_seed(
            series_batch, anchor, dynamics,
            config=config, n_segments=n_segments,
            final_time_s=final_time.numpy().astype(np.float64),
        )
        schedule = BasisSchedule(
            torch.tensor(seed, dtype=torch.float64, device=device),
            dynamics["control_lower"].to(torch.float64),
            dynamics["control_upper"].to(torch.float64),
            final_time.to(dtype=torch.float64, device=device),
            duration_mode,
        ).to(device)

        def objective(prediction: ControlPrediction, *, _s=supervision, _d=dynamics) -> torch.Tensor:
            rollout = fixed_dt_control_state_loss(prediction, _s, config, normalizer, _d)
            return fixed_dt_position_ade_m(rollout.physical_query_states, _s, normalizer)

        fit = fit_basis_schedules(
            schedule, objective, steps=args.steps,
            control_learning_rate=width_scaled_learning_rate(args.control_learning_rate, n_segments),
            duration_learning_rate=width_scaled_learning_rate(args.duration_learning_rate, n_segments),
            gradient_clip_norm=args.gradient_clip_norm,
            learning_rate_floor=args.learning_rate_floor,
        )
        clip_shares.append(fit.clipped_share)
        rows += score_flights(
            schedule, supervision=supervision, dynamics=dynamics, config=config,
            normalizer=normalizer, series_batch=series_batch, anchor=anchor,
            keys=[series_keys[int(index)] for index in indices],
            seed_clipped=seed_clipped, fit=fit,
        )
    strata = summarise(rows, masks, cohort_keys)
    arm = {
        "n_segments": n_segments,
        "duration_mode": duration_mode,
        "free_numbers": free_number_count(n_segments, duration_mode),
        "seconds": time.time() - started,
        "gradient_clip_share": float(np.mean(clip_shares)),
        "strata": strata,
    }
    schedules = [
        {key: row[key] for key in ("flight_key", "controls", "segment_durations_s",
                                   "fixed_dt_ade_m", "seed_fixed_dt_ade_m", "best_step")}
        for row in rows
    ]
    for row in rows:
        del row["controls"], row["segment_durations_s"]
    arm["flights"] = rows
    return arm, schedules


def render(payload: dict) -> str:
    lines = [
        f"L0 control-basis oracle — {payload['airport']}, {payload['coverage']['measured_flights']} "
        f"flights of {payload['coverage']['scored_rows']} scored "
        f"({payload['coverage']['dropped_unscored_rows']} reference rows unscored), "
        f"anchor {payload['anchor_index']}, {payload['optimizer']['steps']} steps, "
        f"lr {payload['optimizer']['control_learning_rate']} annealed to "
        f"x{payload['optimizer']['learning_rate_floor']}",
        f"reference {payload['reference']} (split {payload['split']}), "
        f"dynamics {payload['config']['control_dynamics_model']} / "
        f"{payload['config']['control_dynamics_backend']}",
        "",
        "free# = the schedule's free numbers; the TOTAL duration is given, so a deployed head "
        "at the same width carries one more.",
        "",
        f"{'arm':>16s} {'free#':>6s} {'stratum':>46s} {'n':>5s} {'ADE mean':>9s} {'ADE p50':>8s} "
        f"{'FDE p50':>8s} {'fitADE':>8s} {'seedADE':>8s} {'chamfer':>8s} {'Frechet':>8s} "
        f"{'sat p50':>8s} {'tailp50':>8s} {'tailp90':>8s}",
    ]
    for label, arm in payload["arms"].items():
        for stratum, values in arm["strata"].items():
            lines.append(
                f"{label:>16s} {arm['free_numbers']:>6d} {stratum:>46s} {values['n']:>5d} "
                f"{values['ade_mean_m']:>9.1f} {values['ade_p50_m']:>8.1f} {values['fde_p50_m']:>8.1f} "
                f"{values['fixed_dt_ade_mean_m']:>8.1f} {values['seed_fixed_dt_ade_mean_m']:>8.1f} "
                f"{values['chamfer_p50_m']:>8.1f} {values['frechet_p50_m']:>8.1f} "
                f"{values['saturated_p50']:>8.3f} {values['tail_gain_p50']:>8.4f} "
                f"{values['tail_gain_p90']:>8.4f}"
            )
        lines.append("")
    verdict = payload["verdict"]
    lines.append("tail = relative gain the LAST 10 % of the step budget bought (p50 / p90); "
                 "read the gate only when both are small — under an annealed rate the "
                 "best-step share is ~1 by construction and says nothing.")
    lines.append(f"gate: {verdict['gate']} -> {verdict['status']} "
                 f"({', '.join(verdict['passing_arms']) or 'none'})")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--airport", default="KRDU")
    parser.add_argument("--reference", type=Path, required=True,
                        help="a scored prediction directory; its summary.json gives the cohort, "
                             "the data contract and the strata")
    parser.add_argument("--out", type=Path, required=True, help="must not exist (immutable artifact)")
    parser.add_argument("--segments", default="4,8,16,32,64")
    parser.add_argument("--duration-modes", default=",".join(DURATION_MODES))
    parser.add_argument("--limit", type=int, default=0, help="0 = the whole cohort")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    # Rates are per SEGMENT: the arm at width N starts from this over N (see
    # control.oracle.basis.width_scaled_learning_rate). 0.08 is 0.01 at the measured N=8 optimum.
    parser.add_argument("--control-learning-rate", type=float, default=0.08,
                        help="starting rate at N=1 segment; each arm uses it divided by its N")
    parser.add_argument("--duration-learning-rate", type=float, default=0.08,
                        help="starting rate at N=1 segment; each arm uses it divided by its N")
    parser.add_argument("--gradient-clip-norm", type=float, default=20.0)
    parser.add_argument("--learning-rate-floor", type=float, default=DEFAULT_LEARNING_RATE_FLOOR,
                        help="anneal the rates to this fraction of their starting value")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    segment_counts = [int(token) for token in args.segments.split(",") if token]
    duration_modes = [token.strip() for token in args.duration_modes.split(",") if token.strip()]
    if not segment_counts or any(count < 1 for count in segment_counts):
        parser.error("--segments must be positive integers")
    for mode in duration_modes:
        if mode not in DURATION_MODES:
            parser.error(f"unknown duration mode {mode!r}; expected one of {DURATION_MODES}")
    for name in ("steps", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("control_learning_rate", "duration_learning_rate", "gradient_clip_norm"):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.learning_rate_floor <= 1.0:
        parser.error("--learning-rate-floor must be in (0, 1]")

    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)   # experiment artifacts are immutable
    torch.manual_seed(args.seed)
    reference_dir, summary, cohort_keys, compact_of, masks, coverage = cohort(args, out)
    device = resolve_device(args.device)
    base_config = basis_config(summary["config"], max(segment_counts), args.device)
    series, series_keys, manifest = build_cohort_series(
        cohort_keys, compact_of, args.airport, base_config
    )
    anchor = base_config.seq_len - 1
    normalizer = Normalizer.fit(series, balance_airports_and_flights=True)
    print(f"{args.airport}: {len(series)} flights of {coverage['scored_rows']} scored, "
          f"anchor {anchor}, device {device}; N={segment_counts} x duration={duration_modes}",
          flush=True)

    arms: dict[str, dict] = {}
    schedules: dict[str, list[dict]] = {}
    for n_segments in segment_counts:
        config = basis_config(summary["config"], n_segments, args.device)
        for duration_mode in duration_modes:
            label = f"N={n_segments} {duration_mode}"
            arm, arm_schedules = fit_arm(
                series=series, series_keys=series_keys, cohort_keys=cohort_keys, masks=masks,
                config=config, normalizer=normalizer, anchor=anchor, n_segments=n_segments,
                duration_mode=duration_mode, args=args, device=device,
            )
            arms[label] = arm
            schedules[label] = arm_schedules
            everything = arm["strata"][STRATUM_ALL]
            vectored = arm["strata"].get(STRATUM_VECTORED, {})
            print(f"{label:>16s}: all ADE {everything['ade_mean_m']:8.1f} m "
                  f"(seed fit {everything['seed_fixed_dt_ade_mean_m']:7.1f} -> "
                  f"{everything['fixed_dt_ade_mean_m']:7.1f}) | "
                  f"vectored {vectored.get('ade_mean_m', float('nan')):8.1f} m | "
                  f"tail p50/p90 {everything['tail_gain_p50']:.4f}/{everything['tail_gain_p90']:.4f} | "
                  f"{arm['seconds']:.0f} s", flush=True)

    passing = [
        label for label, arm in arms.items()
        if arm["n_segments"] <= GATE_MAX_SEGMENTS
        and arm["strata"].get(STRATUM_VECTORED, {}).get("ade_mean_m", math.inf) <= GATE_ADE_M
    ]
    payload = {
        "schema": RESULT_SCHEMA,
        "airport": args.airport,
        "reference": str(reference_dir),
        "split": summary.get("split"),
        "anchor_index": anchor,
        "coverage": coverage,
        "config": asdict(base_config),
        "manifests": provenance_manifest_digests(arrival_data_provenance([manifest])),
        "optimizer": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "control_learning_rate": args.control_learning_rate,
            "duration_learning_rate": args.duration_learning_rate,
            "gradient_clip_norm": args.gradient_clip_norm,
            "learning_rate_floor": args.learning_rate_floor,
            "seed": args.seed,
            "device": str(device),
        },
        "arms": arms,
        "verdict": {
            "gate": f"vectored ADE <= {GATE_ADE_M:g} m at N <= {GATE_MAX_SEGMENTS}",
            "passing_arms": passing,
            "status": "pass" if passing else "fail",
        },
    }
    (out / "oracle_basis.json").write_text(json.dumps(payload, indent=2))
    (out / "basis_fit.json").write_text(json.dumps(
        {"schema": RESULT_SCHEMA, "airport": args.airport, "arms": schedules}, indent=2
    ))
    text = render(payload)
    (out / "oracle_basis.txt").write_text(text)
    print(text.splitlines()[-1])
    print(f"wrote {out / 'oracle_basis.txt'}, {out / 'oracle_basis.json'} and {out / 'basis_fit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
