"""CLI: train a trajectory predictor, or predict a batch of approaches with a trained one.

The default ``--prediction-output state`` remains the original purely kinematic baseline.
The opt-in ``--prediction-output control`` predicts bounded per-flight controls and rolls
them through a differentiable Torch twin of ``CasadiSimulator``; it currently uses the
normalized complete-remainder horizon. The output strategy is checkpointed and cannot be
changed at inference without retraining.

    TS=4dTrajectory/ts_transformer/__main__.py

    # train iTransformer on N normalized progress segments plus final_time_s
    python $TS train \
        --data trajectory_data_process/outputs/harvest/KRDU/arrivals/manifest.json \
        --airport KRDU --model itransformer --n-segments 128 \
        --output-dir 4dTrajectory/outputs/KRDU/ts_itransformer_normalized_time

    # the same normalized-time target with PatchTST
    python $TS train --data ... --model patchtst --n-segments 128 ...

    # development prediction (validation is the safe default)
    python $TS predict \
        --checkpoint 4dTrajectory/outputs/KRDU/ts_itransformer_normalized_time/checkpoint.pt \
        --data ... --airport KRDU --output-dir 4dTrajectory/outputs/KRDU/ts_pred

    # outer-test is a separate, irreversible release after every decision is frozen
    python $TS freeze-test --checkpoint ... --data ...
    python $TS predict --checkpoint ... --data ... --output-dir ... \
        --split test --test-release
    python -m evaluation --input 4dTrajectory/outputs/KRDU/ts_pred

Invoked by script path, like ``4dTrajectory/optimization/scenario_optimization.py`` — the
bootstrap below makes it work from any working directory. (``python -m ts_transformer``
also works, but only from inside ``4dTrajectory/``, so the path form is what the docs use.)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

# Same bootstrap as 4dTrajectory/optimization/*.py: the repo root for the shared packages
# (flight_scenarios, aerodynamic_model, geokit, evaluation), and this directory so sibling
# modules import flat.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TS_DIR = Path(__file__).resolve().parent
if str(_TS_DIR) not in sys.path:
    sys.path.insert(0, str(_TS_DIR))

from config import (  # noqa: E402
    AIRCRAFT_FILTER_OPENAP_DIRECT,
    AIRCRAFT_FILTERS,
    COORDINATE_FRAMES,
    STATE_POSITION_REFERENCES,
    TARGET_CONDITIONINGS,
    CHECKPOINT_SELECTION_METRICS,
    CONTROL_ARC_LOCAL_VELOCITY_PARAMETERIZATIONS,
    CONTROL_ARC_TERMINAL_PARAMETERIZATIONS,
    CONTROL_DYNAMICS_BACKENDS,
    CONTROL_DYNAMICS_MODELS,
    CONTROL_DURATION_PARAMETERIZATIONS,
    CONTROL_GRADIENT_CLIP_POLICIES,
    CONTROL_RECIPE_NAMES,
    CONTROL_RECIPE_CUSTOM,
    CONTROL_RECIPE_SIMPLE_V1,
    CONTROL_RECIPE_SIMPLE_V1_LAG,
    PROCEDURE_LOSS_FIELDS,
    TIME_CONSTANT_FIELDS,
    coerce_sequence_fields,
    CONTROL_STATE_LOSS_GRIDS,
    CONTROL_STATE_CLOCKS,
    CONTROL_STATE_OBJECTIVES,
    CONTROL_TERMINAL_CLOCKS,
    DEFAULT_AIRCRAFT_TYPE,
    HORIZON_MODES,
    CONTROL_HOOKS,
    CONTROL_HOOK_FIELDS,
    CONTROL_HOOK_OFF,
    CORRIDOR_GATES,
    HOOK_SATURATIONS,
    MODELS,
    PREDICTION_CONTROL,
    PREDICTION_OUTPUTS,
    TSConfig,
    control_recipe_overrides,
)
from approach_clustering.cli import (  # noqa: E402
    add_cli_arguments as add_approach_cohort_arguments,
    run_cli as run_approach_cohort_cli,
)
from batch_benchmark import (  # noqa: E402
    add_cli_arguments as add_batch_benchmark_arguments,
    run_cli as run_batch_benchmark_cli,
)
from cross_validation import (  # noqa: E402
    CV_PARAMETER_GRIDS,
    DEFAULT_CV_EPOCHS,
    DEFAULT_CV_PATIENCE,
    DEFAULT_CV_PARAMETERS,
    cross_validate,
    validate_cv_parameters,
)
from dataset import (  # noqa: E402
    arrival_data_provenance,
    build_series,
    data_selection_audit,
    dataset_flight_key,
    flight_keys_by_split,
    load_flight_dicts,
    require_matching_data_provenance,
)
from development_cohorts import (  # noqa: E402
    development_cohort_audit,
    load_development_cohort,
)
from export import (  # noqa: E402
    accuracy_block, build_prediction_record, observed_series_metrics, write_batch,
)
from evaluation_protocol import (  # noqa: E402
    TestReleaseError,
    begin_test_evaluation,
    complete_test_evaluation,
    create_test_release,
)
from experiment_index import begin_run, finish_run  # noqa: E402
from flyability import report_for_records  # noqa: E402
from forecast import forecast_approaches  # noqa: E402
from models import resolve_device  # noqa: E402
from reference_velocity import REFERENCE_VELOCITY_SOURCES  # noqa: E402
from train import (  # noqa: E402
    HISTORY_NAME, evaluate_fit_splits, load_checkpoint, train, write_fit_evaluation,
)


def _add_eligibility_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--eligibility-roster",
        action="append",
        default=None,
        help=(
            "pre-split flight eligibility JSON; repeat once per --data manifest. "
            "The top-level pipeline supplies evaluation-derived lateral-pass rosters."
        ),
    )


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data", required=True, action="append",
        help="airport harvest directory or arrivals/manifest.json; repeat for pooled training",
    )
    _add_eligibility_arg(parser)
    parser.add_argument("--airport", default=None,
                        help="ICAO code, when the flight dicts do not carry arr_airport")
    parser.add_argument("--aircraft-type", default=None,
                        help="fallback aircraft when the flight dict has no resolvable type "
                             f"(train default: {DEFAULT_AIRCRAFT_TYPE}; predict default: the "
                             "checkpoint's train-time value). This sets the "
                             "target Vref / threshold-crossing height the gates measure against")
    parser.add_argument(
        "--aircraft-filter",
        choices=AIRCRAFT_FILTERS,
        default=None,
        help=(
            "fleet selection stored in the checkpoint; 'openap-direct' keeps only ICAO "
            "Doc 8643 types with a native same-type OpenAP model (no synonym or fallback)"
        ),
    )
    parser.add_argument("--output-dir", required=True)


def _provenance_from_args(args: argparse.Namespace) -> dict[str, object]:
    rosters = getattr(args, "eligibility_roster", None)
    if rosters is None:
        return arrival_data_provenance(args.data)
    return arrival_data_provenance(args.data, eligibility_rosters=rosters)


def _batch_size(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch size must be a positive integer or 'auto'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("batch size must be positive")
    return parsed


def _cv_parameters(value: str) -> tuple[str, ...]:
    try:
        return validate_cv_parameters(
            name.strip() for name in value.split(",") if name.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _positive_float_csv(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(token.strip()) for token in value.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "horizon curriculum must be a comma-separated list of seconds"
        ) from exc
    if not parsed or any(item <= 0.0 for item in parsed):
        raise argparse.ArgumentTypeError(
            "horizon curriculum must contain positive seconds"
        )
    return parsed


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--control-recipe",
        choices=CONTROL_RECIPE_NAMES,
        default=None,
        help=(
            "named frozen control experiment recipe; simple-v1 fixes architecture, "
            "uniform durations, physical position/time loss and ADE selection. "
            "simple-v1-lag is the same recipe with the lagged flight model, leaving "
            "only the actuator time constants free"
        ),
    )
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument(
        "--prediction-output",
        choices=PREDICTION_OUTPUTS,
        default=None,
        help="predict state endpoints (default) or bounded controls with dynamics rollout",
    )
    parser.add_argument("--seq-len", type=int, default=None, help="lookback L, in steps")
    parser.add_argument("--n-segments", type=int, default=None,
                        help="N normalized state endpoints or non-uniform control segments")
    parser.add_argument(
        "--horizon-mode",
        choices=HORIZON_MODES,
        default=None,
        help="normalized complete trajectory, one-pass full horizon, or recursive window",
    )
    parser.add_argument(
        "--full-horizon-steps",
        type=int,
        default=None,
        help="H_full fixed-dt rows for full mode and window recursion cap (default: 300)",
    )
    parser.add_argument(
        "--window-horizon-steps",
        type=int,
        default=None,
        help="H_window fixed-dt rows emitted by each recursive window pass (default: 30)",
    )
    parser.add_argument("--dt", type=float, default=None, help="resample step, seconds")
    parser.add_argument(
        "--reference-velocity-source",
        choices=REFERENCE_VELOCITY_SOURCES,
        default=None,
        help="velocity-state source: upstream track fit or causal position differences",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=_batch_size, default=None,
                        help="positive integer, or 'auto' to probe the active CUDA GPU")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lr-plateau-factor", type=float, default=None)
    parser.add_argument("--lr-plateau-patience", type=int, default=None)
    parser.add_argument("--fitted-tail-weight", type=float, default=None,
                        help="position-only weight for fitted ADS-B tail rows (default: 0.25)")
    parser.add_argument("--fitted-terminal-weight", type=float, default=None,
                        help="additional position-only weight at the fitted crossing (default: 1.0)")
    parser.add_argument(
        "--state-endpoint-loss-weight",
        type=float,
        default=None,
        help=(
            "direct-state output-endpoint position task weight; shares the path-loss "
            "physical scale (default: 0.25)"
        ),
    )
    parser.add_argument("--kinematic-consistency-weight", type=float, default=None,
                        help="control/oracle compatibility weight; direct state ignores it")
    parser.add_argument("--terminal-loss-weight", type=float, default=None,
                        help="control/oracle compatibility weight; direct state ignores it")
    parser.add_argument("--control-effort-weight", type=float, default=None)
    parser.add_argument("--control-smoothness-weight", type=float, default=None)
    parser.add_argument("--control-dense-state-weight", type=float, default=None)
    parser.add_argument("--control-geometry-weight", type=float, default=None)
    parser.add_argument(
        "--control-arc-horizontal-velocity-weight", type=float, default=None
    )
    parser.add_argument(
        "--control-arc-vertical-velocity-weight", type=float, default=None
    )
    parser.add_argument(
        "--control-arc-horizontal-velocity-scale-mps", type=float, default=None
    )
    parser.add_argument(
        "--control-arc-vertical-velocity-scale-mps", type=float, default=None
    )
    parser.add_argument(
        "--control-arc-local-velocity",
        choices=CONTROL_ARC_LOCAL_VELOCITY_PARAMETERIZATIONS,
        default=None,
    )
    parser.add_argument("--control-arc-tangent-weight", type=float, default=None)
    parser.add_argument("--control-arc-position-end-weight", type=float, default=None)
    parser.add_argument(
        "--control-arc-terminal",
        choices=CONTROL_ARC_TERMINAL_PARAMETERIZATIONS,
        default=None,
    )
    parser.add_argument(
        "--control-arc-terminal-cross-track-emphasis", type=float, default=None
    )
    parser.add_argument(
        "--control-arc-terminal-vertical-emphasis", type=float, default=None
    )
    parser.add_argument("--control-terminal-position-weight", type=float, default=None)
    parser.add_argument("--control-terminal-velocity-weight", type=float, default=None)
    parser.add_argument("--control-terminal-position-scale-m", type=float, default=None)
    parser.add_argument("--control-terminal-velocity-scale-mps", type=float, default=None)
    parser.add_argument(
        "--control-terminal-clock",
        choices=CONTROL_TERMINAL_CLOCKS,
        default=None,
        help=(
            "terminal-state rollout clock: share dense supervision or use the deployable "
            "predicted clock"
        ),
    )
    parser.add_argument(
        "--control-duration-parameterization",
        choices=CONTROL_DURATION_PARAMETERIZATIONS,
        default=None,
        help=(
            "control duration head: total-time plus softmax fractions (factorized), "
            "or uniform, which fixes every segment to total-time/N"
        ),
    )
    parser.add_argument(
        "--control-duration-uniform-floor",
        type=float,
        default=None,
        help=(
            "fraction of total control duration reserved uniformly across segments "
            "to prevent a single-segment partition collapse (default: 0.8)"
        ),
    )
    parser.add_argument(
        "--control-dynamics-backend",
        choices=CONTROL_DYNAMICS_BACKENDS,
        default=None,
        help=(
            "control rollout state representation: re-anchored RK4 baseline or "
            "continuous full-transport chart/ENU-velocity dynamics"
        ),
    )
    parser.add_argument(
        "--control-dynamics-model",
        choices=CONTROL_DYNAMICS_MODELS,
        default=None,
        help=(
            "flight model: point-mass applies each control instantly; first-order-lag "
            "drives thrust, bank and load factor towards their command with a time "
            "constant, so they stay continuous across a segment boundary"
        ),
    )
    parser.add_argument("--control-thrust-tau-s", type=float, default=None)
    parser.add_argument(
        "--control-bank-tau-s",
        type=float,
        default=None,
        help="bank-angle time constant; the lagged model's swept parameter",
    )
    parser.add_argument("--control-load-tau-s", type=float, default=None)
    parser.add_argument(
        "--control-state-clock",
        choices=CONTROL_STATE_CLOCKS,
        default=None,
        help=(
            "total duration used for control state-loss rollout: model prediction "
            "(default) or observed train/validation duration"
        ),
    )
    parser.add_argument(
        "--control-state-loss-grid",
        choices=CONTROL_STATE_LOSS_GRIDS,
        default=None,
        help=(
            "control state-loss queries: learned segment endpoints (default) or every "
            "regular reference dt on the observed clock (fixed-dt)"
        ),
    )
    parser.add_argument(
        "--control-state-objective",
        choices=CONTROL_STATE_OBJECTIVES,
        default=None,
        help=(
            "control tracking objective: normalized-channel MSE (default), the smooth "
            "worst of physical fixed-dt ADE and terminal error, or separately weighted "
            "dense state / terminal position / terminal velocity; true-time-position "
            "is the minimal physical 3-D path and endpoint recipe"
        ),
    )
    parser.add_argument(
        "--control-state-duration-gradient",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "allow state-rollout loss to update duration fractions; use --no-control-"
            "state-duration-gradient to train the final-time clock independently"
        ),
    )
    parser.add_argument(
        "--control-horizon-curriculum",
        type=_positive_float_csv,
        default=None,
        metavar="SECONDS,...",
        help=(
            "physical-time control prefixes trained before the full horizon, e.g. "
            "60,120,240; train/validation only"
        ),
    )
    parser.add_argument(
        "--control-horizon-stage-epochs",
        type=int,
        default=None,
        help="epochs per numeric control-horizon curriculum stage (default: 10)",
    )
    parser.add_argument(
        "--control-gradient-clip-norm",
        type=float,
        default=None,
        help=(
            "global L2 gradient cap for deterministic control training; positive values "
            "also record gradient and control-saturation diagnostics"
        ),
    )
    parser.add_argument(
        "--control-gradient-clip-policy",
        choices=CONTROL_GRADIENT_CLIP_POLICIES,
        default=None,
        help=(
            "gradient clipping scope: one global cap (default) or leave only the "
            "final-time head outside the combined backbone/control cap"
        ),
    )
    parser.add_argument(
        "--control-rollout-dt",
        type=float,
        default=None,
        help="maximum differentiable dynamics RK4 step in seconds (default: 0.5)",
    )
    parser.add_argument("--patience", type=int, default=None, help="early-stopping patience")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--e-layers", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="lock outer train/validation/test identities independently of --seed",
    )
    parser.add_argument("--device", default=None, help='"auto" (default), "cpu", "cuda"')
    parser.add_argument("--coordinate-frame", choices=COORDINATE_FRAMES, default=None)
    parser.add_argument(
        "--state-position-reference",
        choices=STATE_POSITION_REFERENCES,
        default=None,
        help=(
            "state output only: 'anchor-relative' reads the position channels as "
            "displacements from the anchor; default: absolute chart coordinates"
        ),
    )
    parser.add_argument(
        "--target-conditioning",
        choices=TARGET_CONDITIONINGS,
        default=None,
        help=(
            "'channels' feeds the target's chart position + runway course to the model as "
            "input-only constant channels (iTransformer only); default: none"
        ),
    )
    parser.add_argument(
        "--random-train-anchor",
        action="store_true",
        default=None,
        help="train from random valid anchors instead of the fixed full-trajectory anchor L-1",
    )
    parser.add_argument(
        "--training-cohort-min-future-s",
        type=float,
        default=None,
        help=(
            "train-only fixed-L-1 future-duration floor used to lock a common "
            "comparison cohort (default: disabled)"
        ),
    )
    parser.add_argument(
        "--random-train-anchor-min-future-s",
        type=float,
        default=None,
        help="minimum future duration available after a random train anchor (default: 60 s)",
    )
    parser.add_argument(
        "--checkpoint-selection-metric",
        choices=CHECKPOINT_SELECTION_METRICS,
        default=None,
        help="fixed-anchor validation metric used for LR scheduling and early stopping",
    )
    parser.add_argument(
        "--validation-common-grid-points",
        type=int,
        default=None,
        help="physical-time query count for fixed-anchor common-grid selection (default: 64)",
    )
    parser.add_argument("--config-overrides", default=None,
                        help="JSON object of TSConfig overrides, e.g. CV best_config.json")
    parser.add_argument("--instance-norm", dest="instance_norm", action="store_true", default=None,
                        help="per-window de/normalisation; OFF by default because absolute "
                             "threshold-relative position is signal")
    parser.add_argument("--no-instance-norm", dest="instance_norm", action="store_false")
    parser.add_argument(
        "--campaign-id",
        default=None,
        help="formal experiment campaign identity; requires --experiment-id",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="immutable formal run identity; refuses an occupied output directory",
    )


def _config_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[TSConfig, bool]:
    overrides: dict[str, object] = {}
    if args.config_overrides:
        try:
            loaded = json.loads(Path(args.config_overrides).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read --config-overrides: {exc}")
        allowed = {field.name for field in fields(TSConfig)}
        if not isinstance(loaded, dict) or any(key not in allowed for key in loaded):
            parser.error("--config-overrides must be a JSON object containing TSConfig fields")
        # Tuples came back as lists; the frozen-recipe comparison below is on raw values.
        overrides.update(coerce_sequence_fields(loaded))

    batch_auto = args.batch_size == "auto"
    cli_values = (
        ("model", args.model), ("prediction_output", args.prediction_output),
        ("seq_len", args.seq_len),
        ("n_segments", args.n_segments),
        ("horizon_mode", args.horizon_mode),
        ("full_horizon_steps", args.full_horizon_steps),
        ("window_horizon_steps", args.window_horizon_steps),
        ("dt_s", args.dt), ("epochs", args.epochs),
        ("reference_velocity_source", args.reference_velocity_source),
        ("batch_size", args.batch_size if isinstance(args.batch_size, int) else None),
        ("learning_rate", args.learning_rate),
        ("lr_plateau_factor", args.lr_plateau_factor),
        ("lr_plateau_patience", args.lr_plateau_patience),
        ("patience", args.patience),
        ("fitted_tail_position_weight", args.fitted_tail_weight),
        ("fitted_terminal_position_weight", args.fitted_terminal_weight),
        ("state_endpoint_loss_weight", args.state_endpoint_loss_weight),
        ("kinematic_consistency_loss_weight", args.kinematic_consistency_weight),
        ("terminal_loss_weight", args.terminal_loss_weight),
        ("control_effort_loss_weight", args.control_effort_weight),
        ("control_smoothness_loss_weight", args.control_smoothness_weight),
        ("control_dense_state_loss_weight", args.control_dense_state_weight),
        ("control_geometry_loss_weight", args.control_geometry_weight),
        (
            "control_arc_horizontal_velocity_loss_weight",
            args.control_arc_horizontal_velocity_weight,
        ),
        (
            "control_arc_vertical_velocity_loss_weight",
            args.control_arc_vertical_velocity_weight,
        ),
        (
            "control_arc_horizontal_velocity_scale_mps",
            args.control_arc_horizontal_velocity_scale_mps,
        ),
        (
            "control_arc_vertical_velocity_scale_mps",
            args.control_arc_vertical_velocity_scale_mps,
        ),
        (
            "control_arc_local_velocity_parameterization",
            args.control_arc_local_velocity,
        ),
        ("control_arc_tangent_loss_weight", args.control_arc_tangent_weight),
        ("control_arc_position_end_weight", args.control_arc_position_end_weight),
        ("control_arc_terminal_parameterization", args.control_arc_terminal),
        (
            "control_arc_terminal_cross_track_emphasis",
            args.control_arc_terminal_cross_track_emphasis,
        ),
        (
            "control_arc_terminal_vertical_emphasis",
            args.control_arc_terminal_vertical_emphasis,
        ),
        (
            "control_terminal_position_loss_weight",
            args.control_terminal_position_weight,
        ),
        (
            "control_terminal_velocity_loss_weight",
            args.control_terminal_velocity_weight,
        ),
        (
            "control_terminal_position_scale_m",
            args.control_terminal_position_scale_m,
        ),
        (
            "control_terminal_velocity_scale_mps",
            args.control_terminal_velocity_scale_mps,
        ),
        ("control_terminal_supervision_clock", args.control_terminal_clock),
        ("control_duration_parameterization", args.control_duration_parameterization),
        ("control_duration_uniform_floor", args.control_duration_uniform_floor),
        ("control_dynamics_backend", args.control_dynamics_backend),
        ("control_dynamics_model", args.control_dynamics_model),
        ("control_thrust_time_constant_s", args.control_thrust_tau_s),
        ("control_bank_time_constant_s", args.control_bank_tau_s),
        ("control_load_time_constant_s", args.control_load_tau_s),
        ("control_state_supervision_clock", args.control_state_clock),
        ("control_state_loss_grid", args.control_state_loss_grid),
        ("control_state_objective", args.control_state_objective),
        ("control_state_duration_gradient", args.control_state_duration_gradient),
        ("control_horizon_curriculum_s", args.control_horizon_curriculum),
        (
            "control_horizon_curriculum_stage_epochs",
            args.control_horizon_stage_epochs,
        ),
        ("control_gradient_clip_norm", args.control_gradient_clip_norm),
        ("control_gradient_clip_policy", args.control_gradient_clip_policy),
        ("control_rollout_integrator_dt_s", args.control_rollout_dt),
        ("d_model", args.d_model), ("e_layers", args.e_layers), ("n_heads", args.n_heads),
        ("seed", args.seed), ("split_seed", args.split_seed), ("device", args.device),
        ("aircraft_type", args.aircraft_type), ("coordinate_frame", args.coordinate_frame),
        ("target_conditioning", args.target_conditioning),
        ("state_position_reference", args.state_position_reference),
        ("aircraft_filter", args.aircraft_filter),
        ("random_train_anchor", args.random_train_anchor),
        ("training_cohort_min_future_s", args.training_cohort_min_future_s),
        ("random_train_anchor_min_future_s", args.random_train_anchor_min_future_s),
        ("checkpoint_selection_metric", args.checkpoint_selection_metric),
        ("validation_common_grid_points", args.validation_common_grid_points),
        ("use_norm", args.instance_norm), ("revin", args.instance_norm),
    )
    overrides.update({key: value for key, value in cli_values if value is not None})

    override_recipe = overrides.get("control_recipe_name")
    if (
        args.control_recipe is not None
        and override_recipe is not None
        and args.control_recipe != override_recipe
    ):
        parser.error(
            "--control-recipe conflicts with control_recipe_name in --config-overrides"
        )
    requested_recipe = args.control_recipe or override_recipe or CONTROL_RECIPE_CUSTOM
    if requested_recipe not in CONTROL_RECIPE_NAMES:
        parser.error(
            f"unknown control recipe {requested_recipe!r}; choose from "
            f"{CONTROL_RECIPE_NAMES}"
        )
    frozen = control_recipe_overrides(requested_recipe)
    if frozen:
        if batch_auto:
            parser.error(f"{requested_recipe} fixes --batch-size 512; 'auto' is not allowed")
        # Run identity, plus the axes a named recipe deliberately leaves open. simple-v1-lag
        # exists to have its time constants swept, so pinning them would defeat it.
        runtime_fields = {"control_recipe_name", "seed", "split_seed", "device", "notes"}
        # …and the final-approach penalty, which every recipe leaves open.
        open_fields = set(PROCEDURE_LOSS_FIELDS) | set(CONTROL_HOOK_FIELDS) | (
            set(TIME_CONSTANT_FIELDS)
            if requested_recipe == CONTROL_RECIPE_SIMPLE_V1_LAG
            else set()
        )
        unsupported = sorted(
            set(overrides) - set(frozen) - runtime_fields - open_fields
        )
        if unsupported:
            parser.error(
                f"{requested_recipe} allows only seed, split_seed, device, run identity "
                f"and the final-approach penalty fields outside its frozen fields; "
                f"unsupported override {unsupported[0]!r}"
            )
        conflicts = {
            name: (overrides[name], expected)
            for name, expected in frozen.items()
            if name in overrides and overrides[name] != expected
        }
        if conflicts:
            details = ", ".join(
                f"{name}={actual!r} (expected {expected!r})"
                for name, (actual, expected) in sorted(conflicts.items())
            )
            parser.error(f"{requested_recipe} recipe fields are frozen: {details}")
        overrides.update(frozen)
    overrides["control_recipe_name"] = requested_recipe
    return TSConfig(**overrides), batch_auto


def _build_series_or_exit(args: argparse.Namespace, config: TSConfig,
                          parser: argparse.ArgumentParser, flights: list[dict]):
    # config.aircraft_type is the train-time value (checkpoint-carried on predict); an
    # explicit --aircraft-type wins, with the mismatch warned about at the predict site.
    aircraft_type = args.aircraft_type or config.aircraft_type
    series, report = build_series(flights, config, airport=args.airport,
                                  aircraft_type=aircraft_type)
    if config.aircraft_filter == AIRCRAFT_FILTER_OPENAP_DIRECT:
        print("  aircraft   ICAO Doc 8643 identity + native same-type OpenAP model only")
    else:
        print(f"  aircraft   {aircraft_type} (fallback for unresolvable types)")
    print(report.format())
    if not series:
        parser.error(f"no usable series built from {args.data} — see the skip reasons above")
    return series, report


def split_keys_for_current_data(
    checkpoint_split_keys: list[str],
    current_provenance: dict[str, object],
) -> list[str]:
    """Restrict a pooled checkpoint split to the supplied manifest airport subset."""
    manifests = current_provenance.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("current arrival-data provenance has no manifest entries")
    airports = {
        entry.get("airport")
        for entry in manifests
        if isinstance(entry, dict) and isinstance(entry.get("airport"), str)
    }
    if not airports:
        raise ValueError("current arrival-data provenance has no airport identities")
    prefixes = tuple(f"{airport}:" for airport in sorted(airports))
    return [key for key in checkpoint_split_keys if key.startswith(prefixes)]


def _teacher_pretrainer_from_args(
    args: argparse.Namespace,
    config: TSConfig,
    parser: argparse.ArgumentParser,
):
    """Validate optional teacher initialization before a formal run is created."""
    if not args.control_teacher_schedules:
        return None
    if config.prediction_output != PREDICTION_CONTROL:
        parser.error(
            "--control-teacher-schedules requires --prediction-output control"
        )
    from control.oracle.pretraining import CachedSchedulePretrainer

    try:
        return CachedSchedulePretrainer(
            schedule_path=Path(args.control_teacher_schedules),
            steps=args.control_teacher_steps,
            learning_rate=args.control_teacher_learning_rate,
            gradient_clip_norm=args.control_teacher_gradient_clip_norm,
            recipe_name=config.control_recipe_name,
        )
    except ValueError as exc:
        parser.error(str(exc))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ts_transformer",
        description="Learned 4D trajectory prediction (iTransformer / PatchTST)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train / cross validation ─────────────────────────────────────────────
    p_train = sub.add_parser("train", help="train a predictor and write a checkpoint")
    _add_data_args(p_train)
    _add_training_args(p_train)
    p_train.add_argument(
        "--development-cohort",
        default=None,
        help=(
            "explicit train/validation flight roster; the resulting checkpoint is "
            "development-only and cannot release outer-test"
        ),
    )
    p_train.add_argument(
        "--control-teacher-schedules",
        default=None,
        help=(
            "cached outer-train-only control teacher .npz used for initialization; "
            "omitting it runs the documented no-teacher ablation"
        ),
    )
    p_train.add_argument("--control-teacher-steps", type=int, default=1000)
    p_train.add_argument("--control-teacher-learning-rate", type=float, default=1e-4)
    p_train.add_argument(
        "--control-teacher-gradient-clip-norm", type=float, default=20.0
    )

    p_cv = sub.add_parser(
        "cross-validate", help="select hyperparameters using outer-train folds only"
    )
    _add_data_args(p_cv)
    _add_training_args(p_cv)
    p_cv.add_argument(
        "--development-cohort",
        default=None,
        help=(
            "explicit train/validation flight roster; the run is development-only and "
            "defines no outer-test population"
        ),
    )
    p_cv.add_argument("--folds", type=int, default=3)
    p_cv.add_argument(
        "--cv-parameters",
        type=_cv_parameters,
        default=DEFAULT_CV_PARAMETERS,
        metavar=",".join(DEFAULT_CV_PARAMETERS),
        help=(
            "comma-separated parameters to search exhaustively; default: "
            f"{','.join(DEFAULT_CV_PARAMETERS)}; available: {','.join(CV_PARAMETER_GRIDS)}"
        ),
    )
    p_cv.add_argument("--cv-epochs", type=int, default=DEFAULT_CV_EPOCHS)
    p_cv.add_argument("--cv-patience", type=int, default=DEFAULT_CV_PATIENCE)

    p_approach = sub.add_parser(
        "approach-cohorts",
        help="build approach clusters or compare checkpoints on a frozen cohort",
    )
    add_approach_cohort_arguments(p_approach)

    p_batch_benchmark = sub.add_parser(
        "benchmark-batch",
        help="benchmark outer-train CUDA batch throughput",
    )
    add_batch_benchmark_arguments(p_batch_benchmark)

    p_fit = sub.add_parser(
        "evaluate-fit",
        help="replay a best checkpoint deterministically on fixed-anchor train + validation",
    )
    p_fit.add_argument(
        "--data", required=True, action="append",
        help="the complete training arrival-manifest roster; repeat for pooled training",
    )
    _add_eligibility_arg(p_fit)
    p_fit.add_argument("--checkpoint", required=True)
    p_fit.add_argument(
        "--output-dir",
        default=None,
        help="destination for fit_evaluation.json (default: checkpoint directory)",
    )
    p_fit.add_argument("--device", default="auto", help='"auto", "cpu" or "cuda"')

    # ── irreversible outer-test release ─────────────────────────────────────
    p_freeze = sub.add_parser(
        "freeze-test",
        help="bind the one-shot outer-test ledger to a frozen checkpoint and data roster",
    )
    p_freeze.add_argument("--checkpoint", required=True)
    p_freeze.add_argument(
        "--data", required=True, action="append",
        help="the complete training arrival-manifest roster; repeat for pooled training",
    )
    _add_eligibility_arg(p_freeze)

    # ── predict ──────────────────────────────────────────────────────────────
    p_predict = sub.add_parser(
        "predict", help="forecast approaches with a checkpoint; write evaluation records")
    _add_data_args(p_predict)
    p_predict.add_argument("--checkpoint", required=True)
    p_predict.add_argument(
        "--no-truncate",
        action="store_true",
        help="keep full/window forecasts past closest threshold approach",
    )
    p_predict.add_argument(
        "--command-hook", choices=[hook for hook in CONTROL_HOOKS if hook != CONTROL_HOOK_OFF],
        default=None, metavar="HOOK",
        help="run the control rollout through this command hook at prediction time "
             "(the inference-only arms); the checkpoint's own hook applies otherwise",
    )
    p_predict.add_argument(
        "--hook-saturation", choices=HOOK_SATURATIONS, default=None,
        help="with --command-hook: soft (tanh / softplus) or hard (clamp) saturation",
    )
    p_predict.add_argument(
        "--project-final", choices=CORRIDOR_GATES, default=None, metavar="GATE",
        help="after truncation, clamp each state forecast's established tail (under this "
             "corridor gate) into the LPV corridor and glidepath window — the post-hoc "
             "projection arm; records carry source.projectedOntoFinal",
    )
    p_predict.add_argument("--split", choices=("test", "val", "train"), default="val",
                           help="which checkpoint split to predict (default: validation)")
    p_predict.add_argument(
        "--test-release",
        action="store_true",
        help="consume the checkpoint's frozen one-shot test ledger; required for --split test",
    )
    p_predict.add_argument("--device", default="auto",
                           help='"auto" (default: cuda when available), "cpu", "cuda". '
                                "The device is a runtime property, deliberately NOT taken "
                                "from the checkpoint — a cuda-trained checkpoint must stay "
                                "predictable on a CPU-only machine")

    args = parser.parse_args(argv)

    if args.command == "approach-cohorts":
        return run_approach_cohort_cli(args, parser)

    if args.command == "benchmark-batch":
        return run_batch_benchmark_cli(args, parser)

    if args.command == "freeze-test":
        _model, _config, _normalizer, payload = load_checkpoint(args.checkpoint)
        current_provenance = _provenance_from_args(args)
        try:
            release_path = create_test_release(
                args.checkpoint, payload, current_provenance
            )
        except TestReleaseError as exc:
            parser.error(str(exc))
        print(f"✓ outer-test frozen for one-shot evaluation: {release_path}")
        return 0

    if args.command == "evaluate-fit":
        checkpoint_path = Path(args.checkpoint).resolve()
        model, config, normalizer, payload = load_checkpoint(checkpoint_path)
        current_provenance = _provenance_from_args(args)
        require_matching_data_provenance(payload, current_provenance)

        wanted = payload["split"]["train"] + payload["split"]["val"]
        flights = load_flight_dicts(
            args.data,
            include_flight_keys=set(wanted),
        )
        indexed_flights = {
            dataset_flight_key(flight, index): flight
            for index, flight in enumerate(flights)
        }
        missing = [key for key in wanted if key not in indexed_flights]
        if missing:
            parser.error(
                f"{len(missing)} train/validation flight(s) from the checkpoint are absent; "
                f"first missing key: {missing[0]!r}"
            )
        selected_flights = [indexed_flights[key] for key in wanted]
        series, report = build_series(
            selected_flights,
            config,
            aircraft_type=config.aircraft_type,
        )
        print(f"  aircraft   {config.aircraft_type} (checkpoint training fallback)")
        print(report.format())
        indexed_series = {item.dataset_id: item for item in series}
        missing = [key for key in wanted if key not in indexed_series]
        if missing:
            parser.error(
                f"{len(missing)} checkpoint train/validation flight(s) could not be rebuilt; "
                f"first missing key: {missing[0]!r}"
            )
        train_series = [indexed_series[key] for key in payload["split"]["train"]]
        val_series = [indexed_series[key] for key in payload["split"]["val"]]

        history_path = checkpoint_path.parent / HISTORY_NAME
        history = []
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
        evaluation = evaluate_fit_splits(
            model,
            train_series,
            val_series,
            normalizer,
            config,
            resolve_device(args.device),
            history=history,
        )
        output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
        document = write_fit_evaluation(
            evaluation,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
        )
        for split in ("train", "val"):
            metrics = document["splits"][split]["metrics"]
            print(
                f"  {split:5s}  ADE {metrics['ade_m']:7.1f} m   "
                f"FDE {metrics['fde_m']:7.1f} m   "
                f"time MAE {metrics['final_time_s']['mae']:5.1f} s"
            )
        gap = document["diagnostics"]["generalization"]
        print(
            "  gap    "
            f"ADE {gap['ade_m']['absolute_gap']:+.1f} m "
            f"({gap['ade_m']['ratio']:.2f}×)   "
            f"FDE {gap['fde_m']['absolute_gap']:+.1f} m "
            f"({gap['fde_m']['ratio']:.2f}×)"
        )
        print(f"✓ wrote {output_dir / 'fit_evaluation.json'}")
        return 0

    if args.command in ("train", "cross-validate"):
        if len(args.data) > 1 and args.airport:
            parser.error("--airport cannot override flights when multiple --data manifests are used")
        config, batch_auto = _config_from_args(args, parser)
        if bool(args.campaign_id) != bool(args.experiment_id):
            parser.error("--campaign-id and --experiment-id must be supplied together")
        model_pretrainer = (
            _teacher_pretrainer_from_args(args, config, parser)
            if args.command == "train"
            else None
        )
        data_provenance = _provenance_from_args(args)
        outer_split_keys = flight_keys_by_split(data_provenance, config)
        development_cohort = None
        if args.development_cohort:
            development_cohort = load_development_cohort(args.development_cohort)
            development_keys = development_cohort.development_flight_ids(
                outer_split_keys
            )
        else:
            development_keys = set(
                outer_split_keys["train"] + outer_split_keys["val"]
            )
        print(
            f"loading {len(development_keys)} train/validation arrivals; "
            "outer-test source tracks stay closed"
        )
        series, build_report = _build_series_or_exit(
            args,
            config,
            parser,
            load_flight_dicts(args.data, include_flight_keys=development_keys),
        )
        if development_cohort is not None:
            rebuilt_keys = {item.dataset_id for item in series}
            if rebuilt_keys != development_keys:
                missing = sorted(development_keys - rebuilt_keys)
                unexpected = sorted(rebuilt_keys - development_keys)
                details = []
                if missing:
                    details.append(
                        f"missing {len(missing)} (first: {missing[0]!r})"
                    )
                if unexpected:
                    details.append(
                        f"unexpected {len(unexpected)} (first: {unexpected[0]!r})"
                    )
                parser.error(
                    "development cohort could not be rebuilt exactly: "
                    + "; ".join(details)
                )
        data_selection = data_selection_audit(
            series, build_report, config, outer_split_keys
        )
        data_selection["pre_split_eligibility"] = [
            {
                "airport": entry["airport"],
                "arrival_candidates": entry.get("arrival_candidate_count"),
                **(
                    entry["eligibility"]
                    if isinstance(entry.get("eligibility"), dict)
                    else {"policy": "none"}
                ),
            }
            for entry in data_provenance["manifests"]
        ]
        if development_cohort is not None:
            data_selection["development_cohort"] = development_cohort_audit(
                args.development_cohort, development_cohort
            )
        experiment_manifest = None
        if args.experiment_id:
            experiment_manifest = begin_run(
                args.output_dir,
                repo_root=_REPO_ROOT,
                campaign_id=args.campaign_id,
                run_id=args.experiment_id,
                config=config.to_dict(),
                data_selection=data_selection,
                command=sys.argv if argv is None else ["ts_transformer", *argv],
            )
        if args.command == "cross-validate":
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            selection_path = output_dir / "data_selection.json"
            selection_tmp = output_dir / "data_selection.json.tmp"
            selection_tmp.write_text(json.dumps(data_selection, indent=2), encoding="utf-8")
            selection_tmp.replace(selection_path)
            try:
                cross_validate(
                    series,
                    config,
                    output_dir=args.output_dir,
                    data_provenance=data_provenance,
                    n_splits=args.folds,
                    cv_parameters=args.cv_parameters,
                    cv_epochs=args.cv_epochs,
                    cv_patience=args.cv_patience,
                    auto_batch_size=batch_auto,
                )
            except Exception as exc:
                if experiment_manifest is not None:
                    finish_run(
                        experiment_manifest,
                        failure=f"{type(exc).__name__}: {exc}",
                    )
                raise
            if experiment_manifest is not None:
                finish_run(experiment_manifest)
            return 0
        try:
            train(
                series,
                config,
                output_dir=args.output_dir,
                data_provenance=data_provenance,
                reserved_test_keys=(
                    [] if development_cohort is not None else outer_split_keys["test"]
                ),
                data_selection=data_selection,
                auto_batch_size=batch_auto,
                model_pretrainer=model_pretrainer,
            )
        except Exception as exc:
            if experiment_manifest is not None:
                finish_run(
                    experiment_manifest,
                    failure=f"{type(exc).__name__}: {exc}",
                )
            raise
        if experiment_manifest is not None:
            finish_run(experiment_manifest)
        return 0

    # ── predict ──────────────────────────────────────────────────────────────
    if args.split == "test" and not args.test_release:
        parser.error(
            "--split test is sealed; first run freeze-test after all experiment decisions "
            "are final, then pass --test-release"
        )
    if args.split != "test" and args.test_release:
        parser.error("--test-release is valid only with --split test")
    model, config, normalizer, payload = load_checkpoint(args.checkpoint)
    if args.aircraft_filter and args.aircraft_filter != config.aircraft_filter:
        parser.error(
            f"--aircraft-filter {args.aircraft_filter!r} differs from checkpoint policy "
            f"{config.aircraft_filter!r}"
        )
    current_provenance = _provenance_from_args(args)
    require_matching_data_provenance(payload, current_provenance, allow_subset=True)
    device = resolve_device(args.device)
    model = model.to(device)

    if args.aircraft_type and args.aircraft_type != config.aircraft_type:
        print(f"  WARNING: predicting with --aircraft-type {args.aircraft_type}, but the "
              f"checkpoint was trained with {config.aircraft_type} — the ENU frames and "
              f"gate targets will differ from the ones the normalizer was fit under")
    # Resolve the exact identities before reading any trajectory rows. For test this claim
    # is deliberately written first: a crash or partial run still counts as exposure.
    split_keys = split_keys_for_current_data(
        payload["split"][args.split], current_provenance
    )
    test_claim: str | None = None
    if args.split == "test":
        try:
            test_claim = begin_test_evaluation(
                args.checkpoint,
                payload,
                current_provenance,
                split_keys,
                output_dir=args.output_dir,
            )
        except TestReleaseError as exc:
            parser.error(str(exc))
    flights = load_flight_dicts(
        args.data,
        include_flight_keys=set(split_keys),
    )
    indexed = {
        dataset_flight_key(flight, index): flight for index, flight in enumerate(flights)
    }
    missing = [key for key in split_keys if key not in indexed]
    if missing:
        parser.error(
            f"{len(missing)} flight(s) from the checkpoint's {args.split!r} split "
            f"are absent from {args.data}; first missing key: {missing[0]!r}"
        )
    flights = [indexed[key] for key in split_keys]
    series, _build_report = _build_series_or_exit(args, config, parser, flights)
    if args.command_hook is not None:
        if args.hook_saturation is None:
            parser.error("--command-hook needs --hook-saturation")
        config = replace(
            config,
            control_command_hook=args.command_hook,
            control_hook_saturation=args.hook_saturation,
        )
        print(f"  command hook at prediction time: {args.command_hook} ({args.hook_saturation})")
    elif args.hook_saturation is not None:
        parser.error("--hook-saturation needs --command-hook")
    print(f"predicting {len(series)} flight(s) from the {args.split!r} split")

    records, flight_metrics = [], []
    rollout_batch_size = max(1, min(config.batch_size, len(series)))
    print(f"  dense rollout batch size: {rollout_batch_size}")
    for start in range(0, len(series), rollout_batch_size):
        batch_series = series[start : start + rollout_batch_size]
        forecasts = forecast_approaches(
            model,
            batch_series,
            config,
            normalizer,
            device=device,
            truncate=not args.no_truncate,
            project_final=args.project_final,
        )
        for offset, (s, forecast) in enumerate(
            zip(batch_series, forecasts, strict=True)
        ):
            records.append(build_prediction_record(
                s,
                forecast,
                index=start + offset,
                model_name=config.model,
                horizon_mode=config.horizon_mode,
                split=args.split,
            ))
            flight_metrics.append(observed_series_metrics(
                s,
                forecast,
                points=config.validation_common_grid_points,
            ))

    paths = write_batch(
        records,
        output_dir=args.output_dir,
        config_dict=config.to_dict(),
        flight_metrics=flight_metrics,
        checkpoint=str(args.checkpoint),
        split=args.split,
    )

    if args.project_final is not None:
        print(f"  projected every state forecast onto the final ({args.project_final} gate)")
    capped = sum(record.source.get("horizonCapped", False) for record in records)
    if capped:
        print(
            f"  WARNING: {capped} forecast(s) reached the fixed H_full cap before a "
            "threshold closest-approach was detected; their final gate states are cap artifacts"
        )

    # Printed from the block that was just persisted, so the terminal and summary.json
    # cannot report different numbers for the same run.
    accuracy = accuracy_block(flight_metrics)
    if accuracy["flights"]:
        ade, fde = accuracy["ade_m"], accuracy["fde_m"]
        print(f"  vs observed track: ADE {ade['mean']:.1f} m (p95 {ade['p95']:.1f})   "
              f"FDE {fde['mean']:.1f} m (p95 {fde['p95']:.1f})   "
              f"over {accuracy['flights']} flight(s)")
    timing = accuracy["final_time_s"]
    print(f"  final time: MAE {timing['mae']:.1f} s   "
          f"p95 {timing['p95_abs']:.1f} s   bias {timing['mean_signed']:+.1f} s")
    raw = accuracy["raw_kinematics"]
    predicted_raw, observed_raw = raw["predicted"], raw["observed_baseline"]
    print(
        "  raw kinematics fleet p95 (prediction vs observed): "
        f"pos/vel RMSE {predicted_raw['position_velocity_rmse_mps']['p95']:.2f}/"
        f"{observed_raw['position_velocity_rmse_mps']['p95']:.2f} m/s   "
        f"turn {predicted_raw['turn_rate_p95_deg_s']['p95']:.2f}/"
        f"{observed_raw['turn_rate_p95_deg_s']['p95']:.2f} deg/s   "
        f"accel {predicted_raw['acceleration_p95_mps2']['p95']:.2f}/"
        f"{observed_raw['acceleration_p95_mps2']['p95']:.2f} m/s²   "
        f"jerk {predicted_raw['jerk_p95_mps3']['p95']:.2f}/"
        f"{observed_raw['jerk_p95_mps3']['p95']:.2f} m/s³"
    )

    # Flyability: what controls would these trajectories have REQUIRED, and does that sit
    # inside the airframe's envelope? Reported against the observed tracks measured the same
    # way — the check carries a known systematic bias (one clean-configuration drag polar
    # against approaches actually flown dirty), so the delta is the meaningful number and
    # the observed baseline is the floor, not 100%.
    # Each flight is judged against its OWN airframe's envelope: the KRDU harvest spans 14
    # types, so one shared envelope would grade an E170 or a CRJ9 by an A320's Cl_max and
    # max thrust.
    flyability = report_for_records(
        [record.eval_record["states"] for record in records],
        [record.reference_record["states"] for record in records],
        [s.scenario.aircraft for s in series],
    )
    (Path(args.output_dir) / "flyability_report.json").write_text(
        json.dumps(flyability, indent=2), encoding="utf-8")
    predicted, observed = flyability["predicted"], flyability["observed_baseline"]
    print(f"  flyability: {predicted['fully_flyable_rate'] * 100:.1f}% of predictions fully "
          f"flyable vs {observed['fully_flyable_rate'] * 100:.1f}% of the observed tracks "
          f"({flyability['delta']['fully_flyable_rate'] * 100:+.1f} pp)")

    if test_claim is not None:
        complete_test_evaluation(args.checkpoint, test_claim)

    print(f"✓ wrote {len(paths)} evaluation record(s) to {args.output_dir}")
    print(f"  grade them with:  python -m evaluation --input {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
