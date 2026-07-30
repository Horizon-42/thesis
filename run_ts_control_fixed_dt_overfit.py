#!/usr/bin/env python
"""Single-flight capacity diagnostic for fixed-dt control state supervision.

Only one locked outer-train trajectory is opened.  The same trajectory is intentionally
used for gradient updates and no-dropout replay selection; this measures memorization, not
generalization, and never reads outer-test trajectory values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

import torch  # noqa: E402

import run_ts_pipeline as pipeline  # noqa: E402
from channels import POSITION_IDX  # noqa: E402
from config import (  # noqa: E402
    CONTROL_DURATION_FACTORIZED,
    CONTROL_DURATION_PARAMETERIZATIONS,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
    PREDICTION_CONTROL,
    TSConfig,
)
from dataset import (  # noqa: E402
    FixedAnchorTrajectoryWindows,
    provenance_manifest_digests,
)
from fixed_dt_control_loss import fixed_dt_control_state_loss  # noqa: E402
from models import parameter_count  # noqa: E402
from metrics import raw_kinematic_metrics  # noqa: E402
from train_only_diagnostics import select_outer_train_series  # noqa: E402
from train import (  # noqa: E402
    control_state_supervision_prediction,
    evaluate_fixed_anchor_common_grid,
    evaluate_split,
    fit_model,
    model_forward,
    move_dynamics,
    move_fixed_dt_supervision,
)

RESULT_SCHEMA = "ts-control-fixed-dt-single-flight-overfit-v1"


def _dense_replay_metrics(fit, series) -> dict[str, object]:
    dataset = FixedAnchorTrajectoryWindows([series], fit.config, fit.normalizer)
    x, target, weights, final_time, flight_weights, dynamics, dense = dataset.batch(
        np.array([0])
    )
    device = fit.device
    x_device = x.to(device)
    final_time_device = final_time.to(device)
    dynamics_device = move_dynamics(dynamics, device)
    dense_device = move_fixed_dt_supervision(dense, device)
    fit.model.eval()
    with torch.no_grad():
        deployable = model_forward(fit.model, x_device, dynamics_device)
        supervised = control_state_supervision_prediction(
            deployable, final_time_device, fit.config
        )
        dense_result = fixed_dt_control_state_loss(
            supervised,
            dense_device,
            fit.config,
            fit.normalizer,
            dynamics_device,
        )

    valid = dense.valid[0].numpy()
    predicted_dense = dense_result.physical_query_states[0].cpu().numpy()[valid]
    truth_dense = fit.normalizer.decode(
        dense.states[0, valid].numpy().astype(np.float64)
    )
    reference_fully_measured = np.all(
        dense.weights[0, valid].numpy() > 0.0, axis=1
    )
    anchor_physical = fit.normalizer.decode(x[0, -1].numpy().astype(np.float64))
    reference_kinematics = raw_kinematic_metrics(
        anchor_physical[None, :],
        truth_dense[None, ...],
        np.full((1, len(truth_dense)), fit.config.dt_s, dtype=np.float64),
        valid_segments=reference_fully_measured[None, :],
    )
    position_delta = (
        predicted_dense[:, list(POSITION_IDX)]
        - truth_dense[:, list(POSITION_IDX)]
    )
    distance = np.linalg.norm(position_delta, axis=1)
    predicted_endpoints = fit.normalizer.decode(
        dense_result.normalized_segment_end_states[0].cpu().numpy()
    )
    terminal_truth = fit.normalizer.decode(target[0, -1].numpy().astype(np.float64))
    terminal_delta = (
        predicted_endpoints[-1, list(POSITION_IDX)]
        - terminal_truth[list(POSITION_IDX)]
    )
    fractions = (
        deployable.segment_durations[0] / deployable.final_time_s[0]
    ).cpu().numpy()
    dense_loss = float(dense_result.per_flight_loss[0].cpu())
    common = evaluate_fixed_anchor_common_grid(
        fit.model, dataset, fit.normalizer, fit.config, fit.device
    )
    native = evaluate_split(
        fit.model, dataset, fit.normalizer, fit.config, fit.device
    )
    return {
        "fixed_dt_training_clock": {
            "dt_s": fit.config.dt_s,
            "points": int(valid.sum()),
            "state_loss": dense_loss,
            "ade_m": float(distance.mean()),
            "fde_at_last_complete_dt_m": float(distance[-1]),
            "terminal_distance_m": float(np.linalg.norm(terminal_delta)),
            "terminal_horizontal_m": float(np.linalg.norm(terminal_delta[:2])),
            "terminal_vertical_abs_m": float(abs(terminal_delta[2])),
        },
        "deployable_predicted_clock": {
            "predicted_final_time_s": float(deployable.final_time_s[0].cpu()),
            "true_final_time_s": float(final_time[0]),
            "final_time_abs_error_s": float(
                abs(float(deployable.final_time_s[0].cpu()) - float(final_time[0]))
            ),
            "common_grid_ade_m": common["ade_m"],
            "common_grid_fde_m": common["fde_m"],
            "native_endpoint_ade_m": native["ade_m"],
            "native_endpoint_fde_m": native["fde_m"],
        },
        "duration_partition": {
            "segments": len(fractions),
            "min_fraction": float(fractions.min()),
            "max_fraction": float(fractions.max()),
            "fraction_entropy": float(
                -(fractions * np.log(np.clip(fractions, 1e-12, None))).sum()
            ),
        },
        "reference_kinematics": reference_kinematics,
    }


def _write_report(output_dir: Path, result: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overfit_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    dense = result["metrics"]["fixed_dt_training_clock"]
    deployable = result["metrics"]["deployable_predicted_clock"]
    loss = result["loss"]
    (output_dir / "README.md").write_text(
        "# Fixed-dt control single-flight overfit\n\n"
        "This is a train-only memorization diagnostic; no outer-test trajectory was opened.\n\n"
        f"- Flight: `{result['flight']['dataset_id']}`\n"
        f"- Best epoch: {loss['best_epoch']} / {loss['epochs_run']}\n"
        f"- Replay loss: {loss['first_replay_loss']:.8g} → "
        f"{loss['best_replay_loss']:.8g}\n"
        f"- Fixed-{dense['dt_s']:g}s dense ADE: {dense['ade_m']:.3f} m\n"
        f"- Terminal 3D error: {dense['terminal_distance_m']:.3f} m\n"
        f"- Predicted-clock common-grid ADE: "
        f"{deployable['common_grid_ade_m']:.3f} m\n"
        f"- Final-time absolute error: {deployable['final_time_abs_error_s']:.3f} s\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", default="KSJC")
    parser.add_argument("--flight-id", default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--n-segments", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lr-plateau-patience", type=int, default=100)
    parser.add_argument(
        "--duration-parameterization",
        choices=CONTROL_DURATION_PARAMETERIZATIONS,
        default=CONTROL_DURATION_FACTORIZED,
    )
    parser.add_argument("--control-effort-weight", type=float, default=0.0)
    parser.add_argument("--control-smoothness-weight", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    for name in ("epochs", "patience", "n_segments", "lr_plateau_patience"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")

    airport = args.airport.strip().upper()
    manifest = pipeline.arrival_manifest_path(airport)
    if not manifest.is_file():
        parser.error(f"missing arrivals manifest {manifest}")
    config = TSConfig(
        model="itransformer",
        prediction_output=PREDICTION_CONTROL,
        control_duration_parameterization=args.duration_parameterization,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_loss_grid=CONTROL_STATE_LOSS_GRID_FIXED_DT,
        n_segments=args.n_segments,
        epochs=args.epochs,
        patience=min(args.patience, args.epochs),
        batch_size=1,
        learning_rate=args.learning_rate,
        lr_plateau_patience=args.lr_plateau_patience,
        dropout=0.0,
        control_effort_loss_weight=args.control_effort_weight,
        control_smoothness_loss_weight=args.control_smoothness_weight,
        random_train_anchor=False,
        seed=args.seed,
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
    report = selection.report
    provenance = selection.provenance
    split_keys = selection.split_keys
    print(report.format())
    print(
        f"outer roster identities: train={len(split_keys['train'])}, "
        f"val={len(split_keys['val'])}, test={len(split_keys['test'])}; "
        "opened trajectory values: one outer-train flight only"
    )
    print(f"memorizing {series.dataset_id}; remaining reference={series.supervision_times[-1] - series.times[config.seq_len - 1]:.1f}s")
    fit = fit_model([series], [series], config, verbose=not args.quiet)
    metrics = _dense_replay_metrics(fit, series)
    best = min(fit.history, key=lambda row: row.val_loss)
    first = fit.history[0].val_loss
    result = {
        "schema_version": RESULT_SCHEMA,
        "purpose": "single outer-train trajectory memorization, not generalization",
        "test_policy": {
            "outer_test_tracks_opened": False,
            "validation_tracks_opened": False,
            "manifest_identities_used_only_for_locked_split_assignment": True,
        },
        "flight": {
            "dataset_id": series.dataset_id,
            "airport": series.airport,
            "remaining_time_s": float(
                series.supervision_times[-1] - series.times[config.seq_len - 1]
            ),
        },
        "config": fit.config.to_dict(),
        "parameters": parameter_count(fit.model),
        "arrival_manifests": provenance_manifest_digests(provenance),
        "outer_split_identity_counts": {
            name: len(keys) for name, keys in split_keys.items()
        },
        "loss": {
            "epochs_run": len(fit.history),
            "best_epoch": best.epoch,
            "first_replay_loss": first,
            "best_replay_loss": best.val_loss,
            "reduction_pct": 100.0 * (1.0 - best.val_loss / first),
            "best_components": best.val_components,
        },
        "metrics": metrics,
        "history": [vars(row) for row in fit.history],
    }
    output_dir = args.output_dir or (
        REPO_ROOT
        / "4dTrajectory"
        / "outputs"
        / airport
        / "ts_control_fixed_dt_single_flight_overfit"
    )
    _write_report(output_dir, result)
    print(f"wrote {output_dir / 'overfit_result.json'}")
    print(json.dumps({"loss": result["loss"], "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
