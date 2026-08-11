#!/usr/bin/env python
"""Fit a control schedule directly to one outer-train trajectory, without a predictor.

This is a representability diagnostic, not a deployable model.  It opens one outer-train
trajectory and gives the optimizer its complete future reference.  Validation and outer-test
trajectory values are never opened.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

import torch  # noqa: E402

import run_ts_pipeline as pipeline  # noqa: E402
from channels import POSITION_IDX, states_from_channels  # noqa: E402
from config import (  # noqa: E402
    CONTROL_DURATION_FACTORIZED,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    PREDICTION_CONTROL,
    TSConfig,
)
from control_oracle import (  # noqa: E402
    ORACLE_DURATION_LEARNED,
    ORACLE_DURATION_MODES,
    ORACLE_OBJECTIVE_ALL_STATE,
    ORACLE_OBJECTIVE_MODES,
    DirectControlOracle,
    OracleEvaluation,
    fit_control_oracle,
    evaluate_control_prediction,
)
from control_oracle_curriculum import (  # noqa: E402
    HorizonCurriculumStage,
    build_horizon_curriculum,
    build_horizon_stage_view,
)
from control_oracle_initialization import (  # noqa: E402
    inverse_dynamics_controls,
    refine_piecewise_constant_schedule,
)
from dataset import (  # noqa: E402
    FixedAnchorTrajectoryWindows,
    Normalizer,
    provenance_manifest_digests,
)
from models import resolve_device  # noqa: E402
from train_only_diagnostics import select_outer_train_series  # noqa: E402


RESULT_SCHEMA = "ts-direct-control-representability-oracle-v3-locked-split-output"
DEFAULT_ORACLE_SPLIT_SEED = 1337
CONTROL_NAMES = ("thrust_N", "bank_rad", "load_factor")
INITIALIZATION_NEUTRAL = "neutral"
INITIALIZATION_INVERSE_DYNAMICS = "inverse-dynamics"
INITIALIZATION_SCHEDULE = "schedule"
INITIALIZATIONS = (
    INITIALIZATION_NEUTRAL,
    INITIALIZATION_INVERSE_DYNAMICS,
    INITIALIZATION_SCHEDULE,
)


def build_oracle_config(
    *,
    n_segments: int,
    optimizer_seed: int,
    split_seed: int,
    device: str,
) -> TSConfig:
    """Keep optimizer randomness independent from the locked outer partition."""
    return TSConfig(
        model="itransformer",  # Dataset contract only; no model is constructed.
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=CONTROL_DURATION_FACTORIZED,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        n_segments=n_segments,
        batch_size=1,
        dropout=0.0,
        control_effort_loss_weight=0.0,
        control_smoothness_loss_weight=0.0,
        random_train_anchor=False,
        seed=optimizer_seed,
        split_seed=split_seed,
        device=device,
    )


def _slug(value: object, *, limit: int = 72) -> str:
    text = "".join(
        character if character.isalnum() else "-"
        for character in str(value)
    )
    return "-".join(part for part in text.split("-") if part)[:limit] or "unnamed"


def oracle_experiment_identity(dataset_id: str, recipe: dict[str, object]) -> str:
    """Stable, human-readable identity whose digest covers the complete run recipe."""
    payload = {"dataset_id": dataset_id, "recipe": recipe}
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()[:12]
    return "_".join(
        (
            _slug(dataset_id),
            f"N{recipe['n_segments']}",
            _slug(recipe["duration_mode"], limit=24),
            _slug(recipe["objective_mode"], limit=24),
            _slug(recipe["initialization"], limit=24),
            f"seed{recipe['optimizer_seed']}",
            digest,
        )
    )


def _move_dynamics(
    dynamics: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in dynamics.items()}


def _metrics(
    evaluation: OracleEvaluation,
    oracle: DirectControlOracle,
    *,
    supervision,
    terminal_target: torch.Tensor,
    normalizer: Normalizer,
) -> dict[str, object]:
    valid = supervision.valid[0].cpu().numpy()
    predicted = evaluation.rollout.physical_query_states[0].detach().cpu().numpy()[valid]
    truth = normalizer.decode(
        supervision.states[0, valid].cpu().numpy().astype(np.float64)
    )
    delta = predicted[:, list(POSITION_IDX)] - truth[:, list(POSITION_IDX)]
    distance = np.linalg.norm(delta, axis=1)

    endpoint = normalizer.decode(
        evaluation.rollout.normalized_segment_end_states[0, -1]
        .detach()
        .cpu()
        .numpy()
    )
    terminal = normalizer.decode(
        terminal_target[0].detach().cpu().numpy().astype(np.float64)
    )
    terminal_delta = endpoint[list(POSITION_IDX)] - terminal[list(POSITION_IDX)]

    prediction = evaluation.prediction
    controls = prediction.controls[0].detach().cpu().numpy()
    durations = prediction.segment_durations[0].detach().cpu().numpy()
    fractions = durations / durations.sum()
    lower = oracle.control_lower.detach().cpu().numpy()
    upper = oracle.control_upper.detach().cpu().numpy()
    unit_controls = (controls - lower) / (upper - lower)
    return {
        "objective": evaluation.objective.detached(),
        "fixed_dt": {
            "points": int(valid.sum()),
            "ade_m": float(distance.mean()),
            "fde_at_last_complete_dt_m": float(distance[-1]),
        },
        "terminal": {
            "distance_3d_m": float(np.linalg.norm(terminal_delta)),
            "horizontal_m": float(np.linalg.norm(terminal_delta[:2])),
            "vertical_abs_m": float(abs(terminal_delta[2])),
        },
        "duration_partition": {
            "total_s": float(durations.sum()),
            "minimum_s": float(durations.min()),
            "maximum_s": float(durations.max()),
            "entropy": float(
                -(fractions * np.log(np.clip(fractions, 1e-12, None))).sum()
            ),
        },
        "control_ranges": {
            name: {
                "minimum": float(controls[:, index].min()),
                "maximum": float(controls[:, index].max()),
            }
            for index, name in enumerate(CONTROL_NAMES)
        },
        "control_saturation": {
            "fraction_within_1pct_lower": float(np.mean(unit_controls <= 0.01)),
            "fraction_within_1pct_upper": float(np.mean(unit_controls >= 0.99)),
        },
    }


def _write_report(output_dir: Path, result: dict[str, object]) -> None:
    # Experiment artifacts are immutable. This also closes the race between main's early
    # collision check and the actual write.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "oracle_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    best = result["best"]
    np.savez_compressed(
        output_dir / "best_control_schedule.npz",
        controls=np.asarray(best["controls"], dtype=np.float64),
        segment_durations_s=np.asarray(best["segment_durations_s"], dtype=np.float64),
    )
    metrics = best["metrics"]
    verdict = result["verdict"]
    (output_dir / "README.md").write_text(
        "# Direct-control representability oracle\n\n"
        "This train-only diagnostic directly fits a control schedule to a known future; "
        "it is not a deployable predictor.\n\n"
        f"- Flight: `{result['flight']['dataset_id']}`\n"
        f"- Segments / duration mode: {result['optimizer']['n_segments']} / "
        f"{result['optimizer']['duration_mode']}\n"
        f"- Best restart / step: {best['restart']} / {best['best_step']}\n"
        f"- Fixed-{result['config']['dt_s']:g}s ADE: "
        f"{metrics['fixed_dt']['ade_m']:.3f} m\n"
        f"- Terminal 3D error: {metrics['terminal']['distance_3d_m']:.3f} m\n"
        f"- Diagnostic verdict: `{verdict['status']}`\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", default="KSJC")
    parser.add_argument("--flight-id", default=None)
    parser.add_argument("--n-segments", type=int, default=64)
    parser.add_argument(
        "--duration-mode", choices=ORACLE_DURATION_MODES, default=ORACLE_DURATION_LEARNED
    )
    parser.add_argument(
        "--initialization", choices=INITIALIZATIONS, default=INITIALIZATION_NEUTRAL
    )
    parser.add_argument(
        "--objective-mode",
        choices=ORACLE_OBJECTIVE_MODES,
        default=ORACLE_OBJECTIVE_ALL_STATE,
    )
    parser.add_argument(
        "--initial-schedule",
        type=Path,
        default=None,
        help="NPZ containing controls and segment_durations_s; requires --initialization schedule",
    )
    parser.add_argument(
        "--refine-initial-schedule",
        action="store_true",
        help="exactly split a coarser schedule to --n-segments before optimization",
    )
    parser.add_argument(
        "--freeze-duration",
        action="store_true",
        help="keep learned duration logits fixed even at full horizon",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--stage-horizons",
        nargs="+",
        default=None,
        metavar="SECONDS|full",
        help=(
            "opt into short-to-long single-shooting curriculum, for example "
            "60 120 240 full; requires --stage-steps"
        ),
    )
    parser.add_argument(
        "--stage-steps",
        nargs="+",
        type=int,
        default=None,
        metavar="STEPS",
        help="optimizer updates for each --stage-horizons entry; replaces --steps",
    )
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--control-learning-rate", type=float, default=1e-4)
    parser.add_argument("--duration-learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=20.0)
    parser.add_argument("--initial-noise-std", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--success-ade-m", type=float, default=100.0)
    parser.add_argument("--success-terminal-m", type=float, default=100.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_ORACLE_SPLIT_SEED,
        help="locked outer train/validation/test assignment; independent of --seed",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    for name in ("n_segments", "steps", "restarts", "log_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "control_learning_rate",
        "duration_learning_rate",
        "gradient_clip_norm",
        "success_ade_m",
        "success_terminal_m",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.initial_noise_std < 0.0:
        parser.error("--initial-noise-std must be non-negative")
    if (args.initialization == INITIALIZATION_SCHEDULE) != (
        args.initial_schedule is not None
    ):
        parser.error(
            "--initialization schedule and --initial-schedule must be supplied together"
        )
    if args.refine_initial_schedule and args.initialization != INITIALIZATION_SCHEDULE:
        parser.error("--refine-initial-schedule requires --initialization schedule")
    if args.freeze_duration and args.duration_mode != ORACLE_DURATION_LEARNED:
        parser.error("--freeze-duration requires --duration-mode learned")
    if (args.stage_horizons is None) != (args.stage_steps is None):
        parser.error("--stage-horizons and --stage-steps must be supplied together")

    airport = args.airport.strip().upper()
    manifest = pipeline.arrival_manifest_path(airport)
    if not manifest.is_file():
        parser.error(f"missing arrivals manifest {manifest}")
    config = build_oracle_config(
        n_segments=args.n_segments,
        optimizer_seed=args.seed,
        split_seed=args.split_seed,
        device=args.device,
    )
    try:
        selection = select_outer_train_series(
            manifest,
            config,
            airport=airport,
            requested_id=args.flight_id,
            ranking_namespace="fixed-dt-overfit",
        )
    except ValueError as exc:
        parser.error(str(exc))
    series = selection.series
    print(selection.report.format())
    print(
        "outer roster identities: "
        + ", ".join(
            f"{name}={len(keys)}" for name, keys in selection.split_keys.items()
        )
        + "; opened trajectory values: one outer-train flight only"
    )

    normalizer = Normalizer.fit([series], balance_airports_and_flights=True)
    dataset = FixedAnchorTrajectoryWindows([series], config, normalizer)
    x, target, _weights, final_time, _flight_weights, dynamics, supervision = (
        dataset.batch(np.array([0]))
    )
    device = resolve_device(args.device)
    dynamics_device = _move_dynamics(dynamics, device)
    supervision_device = supervision.to(device)
    terminal_target = target[:, -1].to(device)
    true_duration_s = float(final_time[0])
    if args.stage_horizons is None:
        curriculum = (
            HorizonCurriculumStage(
                horizon_s=true_duration_s,
                steps=args.steps,
                is_full_horizon=True,
                optimize_duration=not args.freeze_duration,
            ),
        )
    else:
        try:
            curriculum = build_horizon_curriculum(
                args.stage_horizons,
                args.stage_steps,
                total_duration_s=true_duration_s,
                supervision_dt_s=config.dt_s,
                optimize_full_duration=not args.freeze_duration,
            )
        except ValueError as exc:
            parser.error(str(exc))
    curriculum_enabled = args.stage_horizons is not None
    total_optimizer_steps = sum(stage.steps for stage in curriculum)
    print(
        f"fitting known future {series.dataset_id}; duration={true_duration_s:.3f}s, "
        f"fixed-dt points={int(supervision.valid.sum())}, device={device}"
    )
    if curriculum_enabled:
        print(
            "single-shooting horizon curriculum: "
            + " -> ".join(
                (
                    "full"
                    if stage.is_full_horizon
                    else f"{stage.horizon_s:g}s"
                )
                + f"/{stage.steps} steps"
                for stage in curriculum
            )
            + (
                "; duration frozen for every stage"
                if args.freeze_duration
                else "; duration frozen before full horizon"
            )
        )

    initial_controls = None
    initial_segment_durations_s = None
    initialization_diagnostics: dict[str, object] = {"strategy": args.initialization}
    if args.initialization == INITIALIZATION_INVERSE_DYNAMICS:
        valid = supervision.valid[0].numpy()
        reference_channels = np.concatenate(
            (
                normalizer.decode(x[0, -1].numpy().astype(np.float64))[None, :],
                normalizer.decode(
                    supervision.states[0, valid].numpy().astype(np.float64)
                ),
            ),
            axis=0,
        )
        reference_times = np.concatenate(
            (
                np.array([0.0], dtype=np.float64),
                supervision.query_offsets_s[0, valid].numpy(),
            )
        )
        mass_kg = float(dynamics["initial_state"][0, -1])
        states = states_from_channels(
            reference_times,
            reference_channels,
            series.frame,
            mass_kg=mass_kg,
        )
        reference_states = np.asarray(
            [
                [
                    state.latitude,
                    state.longitude,
                    state.altitude,
                    state.V,
                    state.psi,
                    state.gamma,
                    state.m,
                ]
                for _time, state in states
            ],
            dtype=np.float64,
        )
        inverse = inverse_dynamics_controls(
            reference_states,
            reference_times,
            aero_params=dynamics["aero_params"][0].numpy(),
            control_lower=dynamics["control_lower"][0].numpy(),
            control_upper=dynamics["control_upper"][0].numpy(),
            n_segments=args.n_segments,
            total_duration_s=true_duration_s,
        )
        initial_controls = torch.from_numpy(inverse.controls).to(
            device=device, dtype=torch.float64
        )
        initialization_diagnostics.update(inverse.to_dict())
        print(
            "inverse-dynamics warm start clipped fractions "
            + ", ".join(
                f"{name}={fraction:.3f}"
                for name, fraction in zip(CONTROL_NAMES, inverse.clipped_fraction)
            )
        )
    elif args.initialization == INITIALIZATION_SCHEDULE:
        if not args.initial_schedule.is_file():
            parser.error(f"missing initial schedule {args.initial_schedule}")
        with np.load(args.initial_schedule, allow_pickle=False) as schedule:
            if set(schedule.files) != {"controls", "segment_durations_s"}:
                parser.error(
                    "initial schedule must contain exactly controls and segment_durations_s"
                )
            schedule_controls = np.asarray(schedule["controls"], dtype=np.float64)
            schedule_durations = np.asarray(
                schedule["segment_durations_s"], dtype=np.float64
            )
        initial_controls = torch.from_numpy(schedule_controls).to(
            device=device, dtype=torch.float64
        )
        if args.refine_initial_schedule:
            source_segments = len(schedule_controls)
            try:
                schedule_controls, schedule_durations = (
                    refine_piecewise_constant_schedule(
                        schedule_controls,
                        schedule_durations,
                        target_segments=args.n_segments,
                    )
                )
            except ValueError as exc:
                parser.error(str(exc))
            initial_controls = torch.from_numpy(schedule_controls).to(
                device=device, dtype=torch.float64
            )
            initialization_diagnostics.update(
                {
                    "source_n_segments": source_segments,
                    "refinement_factor": args.n_segments // source_segments,
                }
            )
        initial_segment_durations_s = torch.from_numpy(schedule_durations).to(
            device=device, dtype=torch.float64
        )
        initialization_diagnostics.update(
            {
                "schedule_path": str(args.initial_schedule.resolve()),
                "schedule_sha256": hashlib.sha256(
                    args.initial_schedule.read_bytes()
                ).hexdigest(),
            }
        )

    manifest_digests = provenance_manifest_digests(selection.provenance)
    experiment_recipe = {
        "schema_version": RESULT_SCHEMA,
        "arrival_manifests": manifest_digests,
        "n_segments": args.n_segments,
        "duration_mode": args.duration_mode,
        "objective_mode": args.objective_mode,
        "initialization": args.initialization,
        "initial_schedule_sha256": initialization_diagnostics.get(
            "schedule_sha256"
        ),
        "refine_initial_schedule": args.refine_initial_schedule,
        "freeze_duration": args.freeze_duration,
        "optimizer_seed": args.seed,
        "split_seed": config.resolved_split_seed,
        "steps": total_optimizer_steps,
        "horizon_curriculum": [
            {
                "horizon_s": stage.horizon_s,
                "steps": stage.steps,
                "is_full_horizon": stage.is_full_horizon,
                "optimize_duration": stage.optimize_duration,
            }
            for stage in curriculum
        ],
        "restarts": args.restarts,
        "control_learning_rate": args.control_learning_rate,
        "duration_learning_rate": args.duration_learning_rate,
        "gradient_clip_norm": args.gradient_clip_norm,
        "initial_noise_std": args.initial_noise_std,
        "success_ade_m": args.success_ade_m,
        "success_terminal_m": args.success_terminal_m,
        "dt_s": config.dt_s,
        "control_rollout_integrator_dt_s": config.control_rollout_integrator_dt_s,
        "device": args.device,
    }
    experiment_id = oracle_experiment_identity(
        series.dataset_id, experiment_recipe
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else REPO_ROOT
        / "4dTrajectory"
        / "outputs"
        / airport
        / f"ts_control_oracle_{experiment_id}"
    )
    if output_dir.exists():
        parser.error(
            f"oracle output directory already exists: {output_dir}; "
            "choose a new --output-dir or change the experiment recipe"
        )

    restart_results: list[dict[str, object]] = []
    best_model: DirectControlOracle | None = None
    best_total = math.inf
    for restart in range(args.restarts):
        torch.manual_seed(args.seed + restart)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + restart)
        noise = 0.0 if restart == 0 else args.initial_noise_std
        oracle = DirectControlOracle(
            n_segments=args.n_segments,
            control_lower=dynamics_device["control_lower"][0].to(torch.float64),
            control_upper=dynamics_device["control_upper"][0].to(torch.float64),
            total_duration_s=true_duration_s,
            duration_mode=args.duration_mode,
            initial_controls=initial_controls,
            initial_segment_durations_s=initial_segment_durations_s,
            control_noise_std=noise,
            duration_noise_std=noise,
        ).to(device)

        stage_results: list[dict[str, object]] = []
        for stage_index, stage in enumerate(curriculum):
            stage_label = (
                "full"
                if stage.is_full_horizon
                else f"{stage.horizon_s:g}s"
            )

            def evaluate_stage() -> OracleEvaluation:
                view = build_horizon_stage_view(
                    oracle(),
                    supervision_device,
                    terminal_target,
                    stage,
                )
                return evaluate_control_prediction(
                    view.prediction,
                    supervision=view.supervision,
                    terminal_target=view.terminal_target,
                    config=config,
                    normalizer=normalizer,
                    dynamics=dynamics_device,
                    objective_mode=args.objective_mode,
                )

            def show(row) -> None:
                if not args.quiet:
                    gradient = (
                        "-"
                        if row.gradient_norm is None
                        else f"{row.gradient_norm:.3g}"
                    )
                    print(
                        f"restart {restart + 1}/{args.restarts} "
                        f"stage {stage_index + 1}/{len(curriculum)} "
                        f"({stage_label}) step {row.step:5d}: "
                        f"loss={row.total:.7g} state={row.state:.7g} "
                        f"terminal={row.terminal:.7g} grad={gradient}"
                    )

            fit = fit_control_oracle(
                oracle,
                lambda: evaluate_stage().objective,
                steps=stage.steps,
                control_learning_rate=args.control_learning_rate,
                duration_learning_rate=args.duration_learning_rate,
                optimize_duration=stage.optimize_duration,
                gradient_clip_norm=args.gradient_clip_norm,
                record_every=args.log_every,
                progress=show,
            )
            with torch.no_grad():
                stage_evaluation = evaluate_stage()
                full_view = build_horizon_stage_view(
                    oracle(),
                    supervision_device,
                    terminal_target,
                    curriculum[-1],
                )
                full_evaluation_after_stage = evaluate_control_prediction(
                    full_view.prediction,
                    supervision=full_view.supervision,
                    terminal_target=full_view.terminal_target,
                    config=config,
                    normalizer=normalizer,
                    dynamics=dynamics_device,
                    objective_mode=args.objective_mode,
                )
            stage_view_cpu = build_horizon_stage_view(
                oracle(),
                supervision,
                terminal_target.detach().cpu(),
                stage,
            )
            stage_results.append(
                {
                    "index": stage_index,
                    "label": stage_label,
                    "horizon_s": stage.horizon_s,
                    "steps": stage.steps,
                    "duration_optimized": stage.optimize_duration,
                    "best_step": fit.best_step,
                    "initial_objective": asdict(fit.history[0]),
                    "best_objective": fit.best_objective,
                    "metrics": _metrics(
                        stage_evaluation,
                        oracle,
                        supervision=stage_view_cpu.supervision,
                        terminal_target=stage_view_cpu.terminal_target,
                        normalizer=normalizer,
                    ),
                    "full_horizon_objective_after_stage": (
                        full_evaluation_after_stage.objective.detached()
                    ),
                    "history": [asdict(row) for row in fit.history],
                }
            )

        final_evaluation = full_evaluation_after_stage
        metrics = _metrics(
            final_evaluation,
            oracle,
            supervision=supervision,
            terminal_target=terminal_target,
            normalizer=normalizer,
        )
        prediction = final_evaluation.prediction
        restart_result = {
            "restart": restart,
            "seed": args.seed + restart,
            "initial_noise_std": noise,
            "best_step": fit.best_step,
            "initial_objective": stage_results[0]["initial_objective"],
            "best_objective": fit.best_objective,
            "metrics": metrics,
            "stages": stage_results,
        }
        restart_results.append(restart_result)
        final_total = float(final_evaluation.objective.total.detach().cpu())
        if final_total < best_total:
            best_total = final_total
            best_model = oracle
            best = {
                **restart_result,
                "controls": prediction.controls[0].detach().cpu().tolist(),
                "segment_durations_s": (
                    prediction.segment_durations[0].detach().cpu().tolist()
                ),
            }

    if best_model is None:
        raise RuntimeError("no finite oracle restart completed")
    ade_m = best["metrics"]["fixed_dt"]["ade_m"]
    terminal_m = best["metrics"]["terminal"]["distance_3d_m"]
    succeeded = ade_m <= args.success_ade_m and terminal_m <= args.success_terminal_m
    result = {
        "schema_version": RESULT_SCHEMA,
        "purpose": "control/dynamics representability with the complete future exposed",
        "test_policy": {
            "outer_test_tracks_opened": False,
            "validation_tracks_opened": False,
            "manifest_identities_used_only_for_locked_split_assignment": True,
            "locked_outer_split_seed": config.resolved_split_seed,
            "optimizer_seed_changes_outer_split": False,
        },
        "experiment": {
            "id": experiment_id,
            "recipe": experiment_recipe,
        },
        "flight": {
            "dataset_id": series.dataset_id,
            "airport": series.airport,
            "true_duration_s": true_duration_s,
        },
        "config": config.to_dict(),
        "normalizer": normalizer.to_dict(),
        "arrival_manifests": manifest_digests,
        "outer_split_identity_counts": {
            name: len(keys) for name, keys in selection.split_keys.items()
        },
        "optimizer": {
            "algorithm": "Adam direct shooting through differentiable RK4",
            "n_segments": args.n_segments,
            "duration_mode": args.duration_mode,
            "duration_frozen": args.freeze_duration,
            "objective_mode": args.objective_mode,
            "initialization": initialization_diagnostics,
            "steps": total_optimizer_steps,
            "horizon_curriculum_enabled": curriculum_enabled,
            "horizon_curriculum": experiment_recipe["horizon_curriculum"],
            "restarts": args.restarts,
            "control_learning_rate": args.control_learning_rate,
            "duration_learning_rate": args.duration_learning_rate,
            "gradient_clip_norm": args.gradient_clip_norm,
            "seed": args.seed,
            "split_seed": config.resolved_split_seed,
        },
        "verdict": {
            "status": (
                "representability-demonstrated"
                if succeeded
                else "oracle-fit-above-capacity-threshold"
            ),
            "success_criteria": {
                "ade_m_at_most": args.success_ade_m,
                "terminal_3d_m_at_most": args.success_terminal_m,
            },
            "casadi_attempted": False,
        },
        "best": best,
        "restarts": restart_results,
    }
    _write_report(output_dir, result)
    print(f"wrote {output_dir / 'oracle_result.json'}")
    print(json.dumps({"verdict": result["verdict"], "best": best["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
