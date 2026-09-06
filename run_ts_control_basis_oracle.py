#!/usr/bin/env python
"""Fit piecewise-constant control schedules to observed tracks — the width study, and the
teacher table it turned out to be worth building.

Latent-intent design (`4dTrajectory/ts_transformer/docs/2026-09-07_latent_intent_design.zh.md`)
§六 L0 and §六 L5.a. Both modes do the SAME thing to one flight — direct shooting through
the package's own differentiable rollout, total duration GIVEN (the truth's), the fit's
objective the 3-D position error on the 2 s supervision grid — and differ only in which
flights, at which width, from which starting schedule, and what is written out.

**`--reference` — the width study (L0).** For a scored arm's validation cohort, at each
segment count N and each duration mode, report the error that remains. The result is the
curve ADE(N): the decoder's width is the smallest N whose representation error is
negligible against the 962 m of intent the ego history cannot see::

    python run_ts_control_basis_oracle.py --airport KRDU \\
        --reference 4dTrajectory/outputs/KRDU/experiments/control_procedure_20260905/A_control_v3_pred_val \\
        --out 4dTrajectory/outputs/KRDU/experiments/l0_control_basis_20260907 \\
        --segments 4,8,16,32,64 --duration-modes uniform,free --limit 400 --steps 1200 --batch-size 256

Two error columns are reported for every arm, because they are two different questions and
only one of them is what the fit minimised: `fixed-dt ADE` is the fit's own objective and
the value each flight's best step was chosen on; `ADE` / `FDE` are
`common_physical_time_flight_metrics`, the readouts' own numbers, so an arm here sits
beside a trained arm's `summary.json`. An under-converged fit can only INFLATE the error,
i.e. only produce the false answer "N is not enough", so every arm also reports what the
seed alone was worth (`seed ADE`), the `gain` over it, and the share of flights still
improving when the budget ran out. Read the gate only when that share is small.

**`--checkpoint` — the fitted teacher table (L5.a).** The same fit, at the checkpoint's own
width, over the checkpoint's own train + validation splits, seeded by default from the
checkpoint's deterministic forward. L0 measured what the deployed imitation teacher is
worth: the inverse dynamics of the truth track, flown open-loop, lands 2.5–7.8 km from the
truth, while the fitted schedule at the same width lands 88–433 m from it. The table this
mode writes is that strictly better teacher, consumed by
`control_imitation_target="fitted"`::

    python run_ts_control_basis_oracle.py \\
        --checkpoint 4dTrajectory/outputs/KRDU/experiments/l1_lowdim_20260907/L1_native32/checkpoint.pt \\
        --out 4dTrajectory/outputs/KRDU/experiments/l5_fitted_teacher_20260907 \\
        --splits train,val --init network --steps 400 --batch-size 1024

The outer-test split is never fitted. The table is width-, anchor- and duration-specific,
which is why it carries the stamps the dataset checks it by.

Truth = the post-anchor supervision rows (observed rows plus the fitted tail), as in the
closure oracles, so the numbers sit beside theirs. Validation-split measurement, no
training, nothing written back to the data plane; the output directory is an immutable
artifact and must not exist.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
import hashlib
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
from batch_contract import model_forward  # noqa: E402
from channels import POSITION_IDX  # noqa: E402
from config import (  # noqa: E402
    CONTROL_HOOK_OFF,
    CONTROL_IMITATION_TARGET_INVERSE_DYNAMICS,
    CONTROL_RECIPE_CUSTOM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    INTENT_CONDITIONING_NONE,
    PREDICTION_CONTROL,
    TSConfig,
)
from control.loss.fixed_dt import fixed_dt_control_state_loss  # noqa: E402
from control.basis_fit import (  # noqa: E402
    DEFAULT_LEARNING_RATE_FLOOR,
    DURATION_MODES,
    DURATION_UNIFORM,
    FITTED_TEACHER_SCHEMA,
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
    checkpoint_data_provenance,
    load_flight_dicts,
    provenance_manifest_digests,
    require_matching_data_provenance,
    truth_duration_s,
)
from flight_scenarios.identity import flight_key, summary_row_key  # noqa: E402
from io_utils import file_sha256  # noqa: E402
from metrics import common_physical_time_flight_metrics  # noqa: E402
from models import resolve_device  # noqa: E402
from physical_criteria import fixed_dt_position_ade_m  # noqa: E402
from prediction_outputs import ControlPrediction  # noqa: E402
from train import load_checkpoint, usable_series  # noqa: E402
import run_ts_pipeline as pipeline  # noqa: E402

RESULT_SCHEMA = "l0-control-basis-oracle-v1"
# §六 L0's gate: the representation error must be negligible against the 962 m of intent the
# ego history cannot see, at a width no larger than the optimiser's own control mesh.
GATE_ADE_M = 200.0
GATE_MAX_SEGMENTS = 16
# The outer-test split stays sealed (repo experiment rule); a reference arm scored on it
# would open those tracks here, and a teacher fitted on it would train on it.
FORBIDDEN_SPLIT = "test"
TEACHER_SPLITS = ("train", "val")
# Where the teacher's starting schedule comes from. ``network`` is the checkpoint's own
# deterministic forward — already inside the box, and on straight-in approaches already
# within 445 m — so the fit starts from the answer the model would fly rather than from the
# inverse dynamics the fit exists to replace.
INIT_NETWORK = "network"
INIT_INVERSE_DYNAMICS = "inverse-dynamics"
TEACHER_INITS = (INIT_NETWORK, INIT_INVERSE_DYNAMICS)

# The width study's budget (the formal L0 run passed 1200 explicitly) and L5.a's, which is
# pre-registered in the design doc: L0's convergence evidence — the last 10 % of a 1200-step
# budget bought 0.4–1.2 % — puts 400 steps well past the knee, and the network seed starts
# far closer than the inverse dynamics did. The larger batch is what makes one arm 2–3 h
# rather than 15.
DEFAULT_WIDTH_STEPS, DEFAULT_WIDTH_BATCH_SIZE = 600, 256
DEFAULT_TEACHER_STEPS, DEFAULT_TEACHER_BATCH_SIZE = 400, 1024

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
    "control_imitation_target": CONTROL_IMITATION_TARGET_INVERSE_DYNAMICS,
    "control_fitted_teacher_path": "",
    "procedure_loss_lateral_weight": 0.0,
    "procedure_loss_vertical_weight": 0.0,
    "intent_conditioning": INTENT_CONDITIONING_NONE,
    "closure_labels_path": None,
    "random_train_anchor": False,
    "dropout": 0.0,
}

# The teacher mode inherits the checkpoint's contract WHOLE and overrides only what the fit
# itself needs. In particular the model INPUT contract (`intent_conditioning`,
# `target_conditioning`) is deliberately untouched: the seed is the checkpoint's own
# forward, and a history built under different conditioning is not the history it was
# trained on. The imitation fields are reset so the dataset never solves the per-flight
# inverse for a target this fit does not read — and so a checkpoint that was itself
# trained from a fitted table does not need that table on disk to seed the next one.
_TEACHER_OVERRIDES = {
    "control_recipe_name": CONTROL_RECIPE_CUSTOM,
    "control_state_loss_grid": CONTROL_STATE_LOSS_GRID_FIXED_DT,
    "control_state_objective": CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    "control_state_supervision_clock": CONTROL_STATE_CLOCK_OBSERVED,
    "control_command_hook": CONTROL_HOOK_OFF,
    "control_imitation_loss_weight": 0.0,
    "control_imitation_target": CONTROL_IMITATION_TARGET_INVERSE_DYNAMICS,
    "control_fitted_teacher_path": "",
    "procedure_loss_lateral_weight": 0.0,
    "procedure_loss_vertical_weight": 0.0,
    "random_train_anchor": False,
}


def basis_config(config_dict: dict, n_segments: int, device: str) -> TSConfig:
    """The reference arm's data contract at width ``n_segments``."""
    known = {field.name for field in fields(TSConfig)}
    payload = {name: value for name, value in config_dict.items() if name in known}
    payload.update(_RECIPE_OVERRIDES)
    payload["n_segments"] = int(n_segments)
    payload["device"] = device
    return TSConfig(**payload)


def teacher_config(config: TSConfig, device: str) -> TSConfig:
    """The checkpoint's own contract, with only the fit's own fields substituted."""
    return replace(config, **_TEACHER_OVERRIDES, device=device)


#: Excluded from :func:`config_sha256`: ``device`` is WHERE the fit ran, not what it fitted,
#: and the same table produced on CPU and on CUDA must carry the same contract digest.
_CONFIG_DIGEST_EXCLUDES = ("device",)


def config_sha256(config: TSConfig) -> str:
    """Digest of the contract a table was fitted under — width, anchor, dynamics, frame."""
    payload = {
        name: value for name, value in config.to_dict().items()
        if name not in _CONFIG_DIGEST_EXCLUDES
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


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


# ── one batch of the fit, shared by both modes ──────────────────────────────

def prepare_batch(windows: FixedAnchorTrajectoryWindows, indices: np.ndarray, device):
    """The batch's history, given durations, dynamics and dense supervision, on ``device``."""
    x, _target, _weights, final_time, _flight_weights, dynamics, supervision = windows.batch(indices)
    return (
        x,
        final_time,
        {name: value.to(device) for name, value in dynamics.items()},
        supervision.to(device),
    )


def fit_batch(
    seed: np.ndarray,
    final_time_s: np.ndarray,
    dynamics: dict,
    supervision,
    *,
    config: TSConfig,
    normalizer: Normalizer,
    device,
    duration_mode: str,
    steps: int,
    args: argparse.Namespace,
):
    """Shoot one batch of schedules at its truth tracks; return the schedule and the fit."""
    schedule = BasisSchedule(
        torch.tensor(seed, dtype=torch.float64, device=device),
        dynamics["control_lower"].to(torch.float64),
        dynamics["control_upper"].to(torch.float64),
        torch.tensor(final_time_s, dtype=torch.float64, device=device),
        duration_mode,
    ).to(device)

    def objective(prediction: ControlPrediction) -> torch.Tensor:
        rollout = fixed_dt_control_state_loss(prediction, supervision, config, normalizer, dynamics)
        return fixed_dt_position_ade_m(rollout.physical_query_states, supervision, normalizer)

    n_segments = int(config.n_segments)
    fit = fit_basis_schedules(
        schedule, objective, steps=steps,
        control_learning_rate=width_scaled_learning_rate(args.control_learning_rate, n_segments),
        duration_learning_rate=width_scaled_learning_rate(args.duration_learning_rate, n_segments),
        gradient_clip_norm=args.gradient_clip_norm,
        learning_rate_floor=args.learning_rate_floor,
    )
    return schedule, fit


def restored_fit(schedule: BasisSchedule, *, supervision, dynamics, config, normalizer, fit):
    """Replay the RESTORED best schedule and cross-check it against the fit's own answer.

    The fit selects a best step per flight and copies those parameters back; a schedule that
    no longer reproduces its recorded objective means the restore lost a flight, which would
    show up as a slightly worse number rather than as an error.
    """
    with torch.no_grad():
        prediction = schedule()
        rollout = fixed_dt_control_state_loss(prediction, supervision, config, normalizer, dynamics)
        fixed_dt_ade = fixed_dt_position_ade_m(
            rollout.physical_query_states, supervision, normalizer
        )
    drift = torch.max(torch.abs(fixed_dt_ade - fit.best_value))
    if drift > 1e-6:
        raise RuntimeError(
            f"the restored best state does not reproduce its own objective (max {float(drift):.3g} m)"
        )
    return prediction, rollout, fixed_dt_ade


# ── the width study ─────────────────────────────────────────────────────────

def score_flights(
    schedule: BasisSchedule, *, supervision, dynamics, config, normalizer,
    series_batch, anchor: int, keys: list[str], seed_clipped: np.ndarray, fit,
) -> list[dict]:
    """Score the fitted schedules the way the readouts do, plus the fit's own diagnostics."""
    prediction, rollout, fixed_dt_ade = restored_fit(
        schedule, supervision=supervision, dynamics=dynamics, config=config,
        normalizer=normalizer, fit=fit,
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
        _x, final_time, dynamics, supervision = prepare_batch(windows, indices, device)
        series_batch = [series[int(index)] for index in indices]
        given_duration_s = final_time.numpy().astype(np.float64)
        seed, seed_clipped = inverse_dynamics_seed(
            series_batch, anchor, dynamics,
            config=config, n_segments=n_segments,
            final_time_s=given_duration_s,
        )
        schedule, fit = fit_batch(
            seed, given_duration_s, dynamics, supervision,
            config=config, normalizer=normalizer, device=device,
            duration_mode=duration_mode, steps=args.steps, args=args,
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


def run_width_study(
    args: argparse.Namespace, parser: argparse.ArgumentParser, out: Path,
    segment_counts: list[int], duration_modes: list[str],
) -> int:
    """`--reference`: the ADE(N) curve over one scored arm's cohort (design §六 L0)."""
    airport = args.airport
    reference_dir, summary, cohort_keys, compact_of, masks, coverage = cohort(args, out)
    device = resolve_device(args.device)
    base_config = basis_config(summary["config"], max(segment_counts), args.device)
    # Every refusal this mode can make has been made; claim the immutable directory before
    # the expensive work rather than at the top, so a rejected invocation leaves nothing.
    out.mkdir(parents=True, exist_ok=False)
    series, series_keys, manifest = build_cohort_series(
        cohort_keys, compact_of, airport, base_config
    )
    anchor = base_config.seq_len - 1
    normalizer = Normalizer.fit(series, balance_airports_and_flights=True)
    print(f"{airport}: {len(series)} flights of {coverage['scored_rows']} scored, "
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
        "airport": airport,
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
        {"schema": RESULT_SCHEMA, "airport": airport, "arms": schedules}, indent=2
    ))
    text = render(payload)
    (out / "oracle_basis.txt").write_text(text)
    print(text.splitlines()[-1])
    print(f"wrote {out / 'oracle_basis.txt'}, {out / 'oracle_basis.json'} and {out / 'basis_fit.json'}")
    return 0


# ── the fitted teacher table ────────────────────────────────────────────────

def network_seed(model, history: torch.Tensor, dynamics: dict, device) -> np.ndarray:
    """``[B, N, 3]`` starting controls from the checkpoint's own deterministic forward.

    ``batch_contract.model_forward`` with no future is exactly the pass prediction makes —
    a latent control model decodes its prior's top-1 there — so the seed is the schedule
    this checkpoint would actually fly. Only the CONTROLS are taken: the total duration is
    given (the truth's) and its partition is uniform, as in the width study.
    """
    model.eval()
    with torch.no_grad():
        prediction = model_forward(model, history.to(device), dynamics)
    return prediction.controls.detach().cpu().numpy().astype(np.float64)


def fit_teacher_table(
    *, model, series, split_of: dict[str, str], config: TSConfig, normalizer: Normalizer,
    anchor: int, init: str, args: argparse.Namespace, device,
) -> list[dict]:
    """Fit one schedule per flight over the checkpoint's cohort; one row each.

    The cohort is sorted by its truth duration before batching, because the dense
    supervision of a batch is padded to its LONGEST flight: at KRDU the train split runs
    from a p50 of 183 s to 1454 s, so a batch of 1024 drawn in split order integrates about
    2.5x the flight-seconds it needs. The table is keyed by flight, so the order it was
    fitted in is not observable in the result — and sorting makes it deterministic rather
    than dependent on the order the splits happened to arrive in.
    """
    series = sorted(series, key=lambda item: (truth_duration_s(item, anchor), item.flight_id))
    windows = FixedAnchorTrajectoryWindows(series, config, normalizer)
    if len(windows) != len(series):
        raise RuntimeError(
            f"the fixed-anchor dataset holds {len(windows)} windows for {len(series)} flights"
        )
    n_segments = int(config.n_segments)
    rows: list[dict] = []
    for start in range(0, len(series), args.batch_size):
        indices = np.arange(start, min(start + args.batch_size, len(series)))
        history, _final_time, dynamics, supervision = prepare_batch(windows, indices, device)
        series_batch = [series[int(index)] for index in indices]
        # NOT the batch's own float32 final_time: the table's stamped duration is compared
        # against dataset.truth_duration_s to 1e-6 s, and float32 cannot carry a 300 s
        # duration to that precision. The fit is given the same float64 value it stamps.
        given_duration_s = np.array(
            [truth_duration_s(item, anchor) for item in series_batch], dtype=np.float64
        )
        if init == INIT_NETWORK:
            seed = network_seed(model, history, dynamics, device)
        else:
            seed, _clipped = inverse_dynamics_seed(
                series_batch, anchor, dynamics,
                config=config, n_segments=n_segments, final_time_s=given_duration_s,
            )
        schedule, fit = fit_batch(
            seed, given_duration_s, dynamics, supervision,
            config=config, normalizer=normalizer, device=device,
            duration_mode=DURATION_UNIFORM, steps=args.steps, args=args,
        )
        prediction, _rollout, fixed_dt_ade = restored_fit(
            schedule, supervision=supervision, dynamics=dynamics, config=config,
            normalizer=normalizer, fit=fit,
        )
        controls = prediction.controls.cpu().numpy().astype(np.float64)
        seed_value = fit.seed_value.cpu().numpy().astype(np.float64)
        for row, item in enumerate(series_batch):
            rows.append({
                "flight_key": item.flight_id,
                "split": split_of[item.dataset_id],
                "controls": controls[row].tolist(),
                "total_duration_s": float(given_duration_s[row]),
                "fit_ade_m": float(fixed_dt_ade[row]),
                "seed_ade_m": float(seed_value[row]),
                "best_step": int(fit.best_step[row]),
            })
        print(f"  {len(rows):>6d}/{len(series)} flights fitted "
              f"(batch fitADE mean {float(fixed_dt_ade.mean()):8.1f} m, "
              f"seed {float(np.mean(seed_value)):8.1f} m)", flush=True)
    return rows


def teacher_quantiles(rows: list[dict]) -> dict:
    """Per-split fitADE / seedADE quantiles — the table's own quality readout."""
    out: dict[str, dict] = {}
    for split in dict.fromkeys(row["split"] for row in rows):
        selected = [row for row in rows if row["split"] == split]
        out[split] = {
            "flights": len(selected),
            **{
                f"{name}_{label}": _p([row[f"{name}_m"] for row in selected], quantile)
                for name in ("fit_ade", "seed_ade")
                for label, quantile in (("p50", 50), ("p90", 90), ("p99", 99))
            },
            "fit_ade_mean_m": float(np.mean([row["fit_ade_m"] for row in selected])),
            "seed_ade_mean_m": float(np.mean([row["seed_ade_m"] for row in selected])),
            "best_step_p50": _p([row["best_step"] for row in selected], 50),
        }
    return out


def run_teacher_fit(
    args: argparse.Namespace, parser: argparse.ArgumentParser, out: Path, splits: list[str],
) -> int:
    """`--checkpoint`: the per-flight teacher table over the checkpoint's own splits (§六 L5.a)."""
    started = time.time()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    model, checkpoint_config, normalizer, payload = load_checkpoint(checkpoint)
    if checkpoint_config.prediction_output != PREDICTION_CONTROL:
        parser.error(
            "the fitted teacher IS a control schedule; "
            f"--checkpoint carries prediction_output={checkpoint_config.prediction_output!r}"
        )
    airports = tuple(entry["airport"] for entry in payload["data_provenance"]["manifests"])
    if args.airport is not None and args.airport not in airports:
        parser.error(
            f"--airport {args.airport} is not one of the checkpoint's own airports {airports}; "
            "the teacher cohort is the checkpoint's splits, not a chosen airport"
        )
    manifests = [pipeline.arrival_manifest_path(item) for item in airports]
    provenance = checkpoint_data_provenance(payload, manifests)
    require_matching_data_provenance(payload, provenance)

    device = resolve_device(args.device)
    # The RESOLVED device, so the stored config says where the fit ran rather than "auto"
    # (`config_sha256` excludes it either way — see _CONFIG_DIGEST_EXCLUDES).
    config = teacher_config(checkpoint_config, str(device))
    anchor = config.seq_len - 1
    model = model.to(device)
    split_of = {
        key: split for split in splits for key in payload["split"][split]
    }
    wanted = [key for split in splits for key in payload["split"][split]]
    if len(split_of) != len(wanted):
        raise RuntimeError("the checkpoint's splits overlap: one flight cannot be in two")
    print(f"{'/'.join(airports)}: {len(wanted)} flights over splits {splits}, N={config.n_segments}, "
          f"anchor {anchor}, init {args.init}, device {device}", flush=True)
    # As in the width study: the immutable directory is claimed only once every refusal
    # this mode can make has been made.
    out.mkdir(parents=True, exist_ok=False)

    built, report = build_series(
        load_flight_dicts(manifests, include_flight_keys=set(wanted), verbose=False),
        config,
        aircraft_type=config.aircraft_type,
    )
    print(report.format(), flush=True)
    by_id = {item.dataset_id: item for item in usable_series(built, config, verbose=False)}
    missing = [key for key in wanted if key not in by_id]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(wanted)} checkpoint flights could not be rebuilt; the "
            f"teacher must cover the whole split. First missing: {missing[0]!r}"
        )
    series = [by_id[key] for key in wanted]

    rows = fit_teacher_table(
        model=model, series=series, split_of=split_of, config=config, normalizer=normalizer,
        anchor=anchor, init=args.init, args=args, device=device,
    )
    quantiles = teacher_quantiles(rows)
    table = {
        "schema": FITTED_TEACHER_SCHEMA,
        # The cohort's airports: flight keys are unique WITHIN an airport only, so the
        # dataset checks its own airports against these before anything else.
        "airports": list(airports),
        # The three stamps a consumer is checked against (control.basis_fit.require_cover):
        # a schedule reproduces its truth only at the width, anchor and uniform partition it
        # was fitted under.
        "n_segments": int(config.n_segments),
        "duration_mode": DURATION_UNIFORM,
        "anchor_index": anchor,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        # The contract the table was FITTED under (frame, channels, dt, dynamics, width),
        # written out beside its digest so a reader never has to guess which fields it covers.
        "config": config.to_dict(),
        "config_sha256": config_sha256(config),
        "manifests": provenance_manifest_digests(provenance),
        "init": args.init,
        "splits": splits,
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
        "wall_time_s": time.time() - started,
        "coverage": {
            split: sum(1 for row in rows if row["split"] == split) for split in splits
        },
        "quantiles": quantiles,
        "flights": {
            row["flight_key"]: {
                "controls": row["controls"],
                "total_duration_s": row["total_duration_s"],
                "fit_ade_m": row["fit_ade_m"],
                "seed_ade_m": row["seed_ade_m"],
                "best_step": row["best_step"],
                "split": row["split"],
            }
            for row in rows
        },
    }
    if len(table["flights"]) != len(rows):
        raise RuntimeError("two flights share a flight_key; the table would drop one")
    (out / "basis_fit.json").write_text(json.dumps(table, indent=2))
    print(f"\nfitted teacher — {len(rows)} flights, {table['wall_time_s'] / 60.0:.1f} min, "
          f"init {args.init}, {args.steps} steps @ batch {args.batch_size}")
    print(f"{'split':>8s} {'n':>7s} {'fitADE p50':>11s} {'p90':>9s} {'p99':>9s} "
          f"{'seedADE p50':>12s} {'p90':>9s} {'p99':>9s} {'best step p50':>14s}")
    for split, values in quantiles.items():
        print(f"{split:>8s} {values['flights']:>7d} {values['fit_ade_p50']:>11.1f} "
              f"{values['fit_ade_p90']:>9.1f} {values['fit_ade_p99']:>9.1f} "
              f"{values['seed_ade_p50']:>12.1f} {values['seed_ade_p90']:>9.1f} "
              f"{values['seed_ade_p99']:>9.1f} {values['best_step_p50']:>14.0f}")
    print(f"wrote {out / 'basis_fit.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cohort_group = parser.add_argument_group("cohort — exactly one of these two")
    cohort_group.add_argument("--reference", type=Path, default=None,
                              help="width study: a scored prediction directory; its summary.json "
                                   "gives the cohort, the data contract and the strata")
    cohort_group.add_argument("--checkpoint", type=Path, default=None,
                              help="teacher table: a control checkpoint; its splits are the cohort "
                                   "and its config the width, anchor and dynamics")
    parser.add_argument("--out", type=Path, required=True, help="must not exist (immutable artifact)")
    parser.add_argument("--airport", default=None,
                        help="width study: the airport whose arrival manifest the cohort is "
                             "rebuilt from (default KRDU). Teacher table: the airports come from "
                             "the checkpoint, and this only cross-checks them")
    width = parser.add_argument_group("the width study (--reference only)")
    width.add_argument("--segments", default=None, help="default: 4,8,16,32,64")
    width.add_argument("--duration-modes", default=None,
                       help=f"default: {','.join(DURATION_MODES)}")
    width.add_argument("--limit", type=int, default=None, help="0 = the whole cohort (default)")
    teacher = parser.add_argument_group("the teacher table (--checkpoint only)")
    teacher.add_argument("--splits", default=None,
                         help=f"default: {','.join(TEACHER_SPLITS)}; the imitation term is defined "
                              f"on validation too, and {FORBIDDEN_SPLIT!r} is refused")
    teacher.add_argument("--init", choices=TEACHER_INITS, default=None,
                         help=f"starting schedule (default: {INIT_NETWORK})")
    parser.add_argument("--steps", type=int, default=None,
                        help=f"default: {DEFAULT_TEACHER_STEPS} with --checkpoint, "
                             f"{DEFAULT_WIDTH_STEPS} with --reference")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"default: {DEFAULT_TEACHER_BATCH_SIZE} with --checkpoint, "
                             f"{DEFAULT_WIDTH_BATCH_SIZE} with --reference")
    # Rates are per SEGMENT: the arm at width N starts from this over N (see
    # control.basis_fit.width_scaled_learning_rate). 0.08 is 0.01 at the measured N=8 optimum.
    parser.add_argument("--control-learning-rate", type=float, default=0.08,
                        help="starting rate at N=1 segment; each arm uses it divided by its N")
    parser.add_argument("--duration-learning-rate", type=float, default=0.08,
                        help="starting rate at N=1 segment; each arm uses it divided by its N")
    parser.add_argument("--gradient-clip-norm", type=float, default=20.0)
    parser.add_argument("--learning-rate-floor", type=float, default=DEFAULT_LEARNING_RATE_FLOOR,
                        help="anneal the rates to this fraction of their starting value")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.reference is None) == (args.checkpoint is None):
        parser.error(
            "give exactly one of --reference (the ADE(N) width study) or --checkpoint "
            "(the per-flight teacher table)"
        )
    teacher_mode = args.checkpoint is not None
    # Each mode's own options are refused in the other, and each mode's own defaults are
    # resolved here — one place, so no downstream reader has to know which mode it is in.
    for name, applies in (("segments", not teacher_mode), ("duration_modes", not teacher_mode),
                          ("limit", not teacher_mode), ("splits", teacher_mode),
                          ("init", teacher_mode)):
        if getattr(args, name) is not None and not applies:
            parser.error(
                f"--{name.replace('_', '-')} belongs to "
                f"{'--reference' if teacher_mode else '--checkpoint'} mode"
            )
    # ``or`` would turn an explicit 0 into the default and hide it from the guards below.
    args.limit = 0 if args.limit is None else args.limit
    args.init = INIT_NETWORK if args.init is None else args.init
    args.steps = (
        (DEFAULT_TEACHER_STEPS if teacher_mode else DEFAULT_WIDTH_STEPS)
        if args.steps is None else args.steps
    )
    args.batch_size = (
        (DEFAULT_TEACHER_BATCH_SIZE if teacher_mode else DEFAULT_WIDTH_BATCH_SIZE)
        if args.batch_size is None else args.batch_size
    )
    args.airport = None if args.airport is None else args.airport.upper()

    for name in ("steps", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("control_learning_rate", "duration_learning_rate", "gradient_clip_norm"):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.learning_rate_floor <= 1.0:
        parser.error("--learning-rate-floor must be in (0, 1]")

    # Each mode's list arguments are parsed HERE, beside the numeric guards, so that every
    # refusal that needs nothing but the command line happens before the immutable output
    # directory is claimed (each mode creates it once its own refusals are also past).
    splits = [token.strip() for token in (args.splits or ",".join(TEACHER_SPLITS)).split(",")
              if token.strip()]
    segment_counts = [int(token) for token in (args.segments or "4,8,16,32,64").split(",") if token]
    duration_modes = [token.strip() for token in
                      (args.duration_modes or ",".join(DURATION_MODES)).split(",") if token.strip()]
    if teacher_mode:
        if not splits or len(set(splits)) != len(splits):
            parser.error("--splits must be a non-repeating list of split names")
        for split in splits:
            if split == FORBIDDEN_SPLIT:
                parser.error(
                    f"the {FORBIDDEN_SPLIT} split is sealed: a teacher fitted on it would "
                    "train on it"
                )
            if split not in TEACHER_SPLITS:
                parser.error(f"unknown split {split!r}; expected one of {TEACHER_SPLITS}")
    else:
        if not segment_counts or any(count < 1 for count in segment_counts):
            parser.error("--segments must be positive integers")
        for mode in duration_modes:
            if mode not in DURATION_MODES:
                parser.error(f"unknown duration mode {mode!r}; expected one of {DURATION_MODES}")

    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    torch.manual_seed(args.seed)
    if teacher_mode:
        return run_teacher_fit(args, parser, out, splits)
    args.airport = args.airport or "KRDU"
    return run_width_study(args, parser, out, segment_counts, duration_modes)


if __name__ == "__main__":
    raise SystemExit(main())
