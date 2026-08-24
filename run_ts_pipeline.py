#!/usr/bin/env python
"""Run cross-validation -> train -> per-airport prediction/evaluation/CZML.

Two training scopes are explicit:

``per-airport``
    One CV search and checkpoint per airport (the historical organization).

``pooled``
    One CV search and checkpoint over all selected airport manifests, followed by separate
    predictions and publications for each airport's locked split.

The TS command locks outer train/validation/test before cross-validation. CV sees outer-train
only; final training uses outer-validation for early stopping. Routine runs publish train and
validation only. Outer-test requires a separate checkpoint-bound, one-shot release after every
experimental decision is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
HARVEST_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"
OPT_OUTPUTS_ROOT = REPO_ROOT / "4dTrajectory" / "outputs"
COMPARISON_AIRPORTS_ROOT = REPO_ROOT / "aeroviz-4d" / "public" / "data" / "airports"
TS_SCRIPT = REPO_ROOT / "4dTrajectory" / "ts_transformer" / "__main__.py"
CZML_SCRIPT = REPO_ROOT / "aeroviz-4d" / "python" / "build_scenario_comparison_czml.py"
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

from config import (  # noqa: E402
    AIRCRAFT_FILTER_ALL,
    AIRCRAFT_FILTERS,
    COORDINATE_FRAMES,
    CHECKPOINT_SELECTION_COMMON_GRID_ADE,
    CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
    CONTROL_ARC_LOCAL_VELOCITY_PARAMETERIZATIONS,
    CONTROL_ARC_LOCAL_VELOCITY_VECTOR,
    CONTROL_ARC_TERMINAL_PARAMETERIZATIONS,
    CONTROL_ARC_TERMINAL_RUNWAY_COMPONENTS,
    CHECKPOINT_SELECTION_METRICS,
    CHECKPOINT_SELECTION_OBJECTIVE,
    CONTROL_DYNAMICS_BACKENDS,
    CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    CONTROL_DYNAMICS_MODELS,
    CONTROL_DYNAMICS_POINT_MASS,
    CONTROL_DYNAMICS_REANCHORED_RK4,
    CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
    CONTROL_DURATION_FACTORIZED,
    CONTROL_DURATION_PARAMETERIZATIONS,
    CONTROL_GRADIENT_CLIP_GLOBAL,
    CONTROL_GRADIENT_CLIP_POLICIES,
    CONTROL_STATE_CLOCKS,
    CONTROL_STATE_CLOCK_PREDICTED,
    CONTROL_STATE_LOSS_GRIDS,
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    CONTROL_STATE_OBJECTIVES,
    CONTROL_TERMINAL_CLOCKS,
    CONTROL_TERMINAL_CLOCK_PREDICTED,
    CONTROL_TERMINAL_CLOCK_PREDICTED_DETACHED_TIME,
    CONTROL_TERMINAL_CLOCK_STATE_SUPERVISION,
    DEFAULT_CONTROL_HORIZON_CURRICULUM_STAGE_EPOCHS,
    DEFAULT_RANDOM_TRAIN_ANCHOR_MIN_FUTURE_S,
    DEFAULT_VALIDATION_COMMON_GRID_POINTS,
    HORIZON_MODES,
    HORIZON_NORMALIZED,
    HORIZON_WINDOW,
    MODELS,
    PREDICTION_CONTROL,
    PREDICTION_OUTPUTS,
    PREDICTION_STATE,
    TSConfig,
    control_recipe,
    uses_control_dynamics,
)
from cross_validation import (  # noqa: E402
    BEST_CONFIG_NAME,
    CV_PARAMETER_GRIDS,
    DEFAULT_CV_EPOCHS,
    DEFAULT_CV_PATIENCE,
    DEFAULT_CV_PARAMETERS,
    RESULTS_NAME as CV_RESULTS_NAME,
    RESULTS_SCHEMA as CV_RESULTS_SCHEMA,
    applicable_cv_parameters,
    parameter_grid,
)
from evaluation_protocol import TEST_RELEASE_NAME  # noqa: E402
from lateral_eligibility import (  # noqa: E402
    default_lateral_pass_roster_path,
    ensure_lateral_pass_roster,
)
from run_naming import (  # noqa: E402
    category_display_label,
    run_display_name,
)
from train import (  # noqa: E402
    CHECKPOINT_METADATA_NAME,
    CHECKPOINT_METADATA_SCHEMA,
    CHECKPOINT_NAME,
)

TRAINING_MODES = ("per-airport", "pooled")
MODEL_SHORT = {"itransformer": "itr", "patchtst": "ptst"}
OUTPUT_KINDS = ("czml", "eval")
PREDICTION_SPLITS = ("train", "val", "test")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arrival_manifest_path(airport: str) -> Path:
    return HARVEST_ROOT / airport.upper() / "arrivals" / "manifest.json"


def discover_k_airports() -> list[str]:
    if not HARVEST_ROOT.exists():
        return []
    return sorted(
        child.name.upper()
        for child in HARVEST_ROOT.iterdir()
        if child.is_dir()
        and child.name.upper().startswith("K")
        and arrival_manifest_path(child.name).exists()
    )


def _manifest_digests(airports: tuple[str, ...]) -> dict[str, str]:
    return {airport: _file_sha256(arrival_manifest_path(airport)) for airport in airports}


def _eligibility_digests(airports: tuple[str, ...]) -> dict[str, str]:
    return {
        airport: _file_sha256(default_lateral_pass_roster_path(arrival_manifest_path(airport)))
        for airport in airports
    }


def _frame_tag(coordinate_frame: str) -> str:
    return "" if coordinate_frame == "enu" else "_runway_aligned"


def _anchor_tag(random_train_anchor: bool) -> str:
    return "_random_anchor" if random_train_anchor else ""


def _training_cohort_tag(minimum_future_s: float) -> str:
    if minimum_future_s <= 0.0:
        return ""
    compact = f"{minimum_future_s:g}".replace(".", "p")
    return f"_cohort_min{compact}"


def _validation_selection_tag(metric: str) -> str:
    return {
        CHECKPOINT_SELECTION_COMMON_GRID_ADE: "",
        CHECKPOINT_SELECTION_OBJECTIVE: "_legacy_objective_selection",
        CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY: "_arc_length_selection",
    }.get(metric, "")


def _prediction_output_tag(prediction_output: str) -> str:
    return "" if prediction_output == PREDICTION_STATE else f"_{prediction_output}"


def _control_clock_tag(prediction_output: str, state_clock: str) -> str:
    if (
        not uses_control_dynamics(prediction_output)
        or state_clock == CONTROL_STATE_CLOCK_PREDICTED
    ):
        return ""
    return f"_{state_clock}_clock"


def _control_terminal_clock_tag(
    prediction_output: str, terminal_clock: str
) -> str:
    if (
        prediction_output != PREDICTION_CONTROL
        or terminal_clock == CONTROL_TERMINAL_CLOCK_STATE_SUPERVISION
    ):
        return ""
    return f"_terminal_{terminal_clock.replace('-', '_')}_clock"


_CONTROL_TERMINAL_CLOCK_FILESYSTEM_TAGS = {
    CONTROL_TERMINAL_CLOCK_STATE_SUPERVISION: "",
    CONTROL_TERMINAL_CLOCK_PREDICTED: "_tcp",
    CONTROL_TERMINAL_CLOCK_PREDICTED_DETACHED_TIME: "_tcpdt",
}


def _control_terminal_clock_filesystem_tag(
    prediction_output: str, terminal_clock: str
) -> str:
    if prediction_output != PREDICTION_CONTROL:
        return ""
    return _CONTROL_TERMINAL_CLOCK_FILESYSTEM_TAGS[terminal_clock]


def _control_state_loss_grid_tag(prediction_output: str, loss_grid: str) -> str:
    if not uses_control_dynamics(prediction_output) or loss_grid == CONTROL_STATE_LOSS_GRID_NATIVE:
        return ""
    return f"_{loss_grid.replace('-', '_')}_loss"


def _control_duration_tag(prediction_output: str, parameterization: str) -> str:
    if (
        not uses_control_dynamics(prediction_output)
        or parameterization == CONTROL_DURATION_FACTORIZED
    ):
        return ""
    return f"_{parameterization}_duration"


def _control_dynamics_tag(prediction_output: str, backend: str) -> str:
    if (
        not uses_control_dynamics(prediction_output)
        or backend == CONTROL_DYNAMICS_REANCHORED_RK4
    ):
        return ""
    return f"_{backend.replace('-', '_')}"


_CONTROL_DYNAMICS_FILESYSTEM_TAGS = {
    CONTROL_DYNAMICS_REANCHORED_RK4: "",
    CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY: "_tcv",
    CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY: "_stcv",
}


def _control_dynamics_filesystem_tag(prediction_output: str, backend: str) -> str:
    if not uses_control_dynamics(prediction_output):
        return ""
    return _CONTROL_DYNAMICS_FILESYSTEM_TAGS[backend]


def _control_objective_tag(prediction_output: str, objective: str) -> str:
    if (
        not uses_control_dynamics(prediction_output)
        or objective == CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE
    ):
        return ""
    return f"_{objective.replace('-', '_')}"


def _terminal_tracking_recipe_tag(
    prediction_output: str,
    objective: str,
    dense_weight: float,
    geometry_weight: float,
    arc_horizontal_velocity_weight: float,
    arc_vertical_velocity_weight: float,
    arc_horizontal_velocity_scale_mps: float,
    arc_vertical_velocity_scale_mps: float,
    arc_local_velocity: str,
    arc_tangent_weight: float,
    arc_position_end_weight: float,
    arc_terminal: str,
    arc_terminal_cross_track_emphasis: float,
    arc_terminal_vertical_emphasis: float,
    terminal_position_weight: float,
    terminal_velocity_weight: float,
    terminal_position_scale_m: float,
    terminal_velocity_scale_mps: float,
) -> str:
    if (
        prediction_output != PREDICTION_CONTROL
        or objective != CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY
    ):
        return ""

    def compact(value: float) -> str:
        return f"{value:g}".replace(".", "p")

    tracking = {
        CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY: (
            f"_g{compact(geometry_weight)}"
            f"_ahv{compact(arc_horizontal_velocity_weight)}"
            f"_avv{compact(arc_vertical_velocity_weight)}"
            f"_ahvs{compact(arc_horizontal_velocity_scale_mps)}mps"
            f"_avvs{compact(arc_vertical_velocity_scale_mps)}mps"
            f"_lv{arc_local_velocity.replace('-', '_')}"
            f"_at{compact(arc_tangent_weight)}"
            f"_pe{compact(arc_position_end_weight)}"
            f"_term{arc_terminal.replace('-', '_')}"
            f"_tc{compact(arc_terminal_cross_track_emphasis)}"
            f"_tu{compact(arc_terminal_vertical_emphasis)}"
        ),
    }[objective]
    return (
        f"{tracking}"
        f"_tp{compact(terminal_position_weight)}"
        f"_tv{compact(terminal_velocity_weight)}"
        f"_ps{compact(terminal_position_scale_m)}m"
        f"_vs{compact(terminal_velocity_scale_mps)}mps"
    )


def _control_duration_gradient_tag(
    prediction_output: str, state_duration_gradient: bool
) -> str:
    if not uses_control_dynamics(prediction_output) or state_duration_gradient:
        return ""
    return "_detached_duration_gradient"


def _control_horizon_curriculum_tag(
    horizons_s: tuple[float, ...], stage_epochs: int
) -> str:
    if not horizons_s:
        return ""
    horizons = "_".join(f"{value:g}".replace(".", "p") for value in horizons_s)
    return f"_horizon_curriculum_{horizons}s_x{stage_epochs}"


def _control_gradient_clip_tag(max_norm: float, policy: str) -> str:
    if max_norm <= 0.0:
        return ""
    compact = f"{max_norm:g}".replace(".", "p")
    policy_tag = (
        ""
        if policy == CONTROL_GRADIENT_CLIP_GLOBAL
        else f"_{policy.replace('-', '_')}"
    )
    return f"_gradient_clip{compact}{policy_tag}"


def _aircraft_filter_tag(aircraft_filter: str) -> str:
    return "" if aircraft_filter == AIRCRAFT_FILTER_ALL else "_openap_direct"


HORIZON_TAGS = {
    "normalized": "normalized_time",
    "full": "full",
    "window": "window",
}
HORIZON_LABELS = {
    "normalized": "normalized time",
    "full": "full horizon",
    "window": "recursive window",
}


# ext4/APFS cap one path component at 255 bytes, and the recipe suffix outgrew that when
# the arc-length-geometry weight block joined it (measured: 365 bytes). A name over the cap
# therefore keeps its readable head and ends in a digest of the WHOLE name, so two recipes
# whose heads happen to agree still land in different directories.
MAX_PATH_COMPONENT_BYTES = 255


def _bounded_component(name: str) -> str:
    """Return ``name`` unchanged, or a head + content digest that fits one component."""
    encoded = name.encode("utf-8")
    if len(encoded) <= MAX_PATH_COMPONENT_BYTES:
        return name
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    head = encoded[: MAX_PATH_COMPONENT_BYTES - len(digest) - 1]
    return f"{head.decode('utf-8', 'ignore')}_{digest}"


class TrainingPlan:
    """One CV/final-training cell, shared by one or more airport predictions."""

    def __init__(
        self,
        airports: tuple[str, ...],
        model: str,
        *,
        training_mode: str,
        prediction_output: str = PREDICTION_STATE,
        n_segments: int | None = None,
        horizon_mode: str = HORIZON_NORMALIZED,
        full_horizon_steps: int | None = None,
        window_horizon_steps: int | None = None,
        epochs: int | None = None,
        seed: int | None = None,
        split_seed: int | None = None,
        device: str | None = None,
        aircraft_type: str | None = None,
        aircraft_filter: str = AIRCRAFT_FILTER_ALL,
        coordinate_frame: str = "enu",
        batch_size: str = "2048",
        cv_folds: int = 3,
        cv_parameters: tuple[str, ...] = DEFAULT_CV_PARAMETERS,
        cv_epochs: int = DEFAULT_CV_EPOCHS,
        cv_patience: int = DEFAULT_CV_PATIENCE,
        random_train_anchor: bool = False,
        training_cohort_min_future_s: float = 0.0,
        random_train_anchor_min_future_s: float = DEFAULT_RANDOM_TRAIN_ANCHOR_MIN_FUTURE_S,
        checkpoint_selection_metric: str = CHECKPOINT_SELECTION_COMMON_GRID_ADE,
        validation_common_grid_points: int = DEFAULT_VALIDATION_COMMON_GRID_POINTS,
        control_effort_weight: float | None = None,
        control_smoothness_weight: float | None = None,
        control_dense_state_weight: float = 0.25,
        control_geometry_weight: float = 0.75,
        control_arc_horizontal_velocity_weight: float = 0.25,
        control_arc_vertical_velocity_weight: float = 0.25,
        control_arc_horizontal_velocity_scale_mps: float = 10.0,
        control_arc_vertical_velocity_scale_mps: float = 2.0,
        control_arc_local_velocity: str = CONTROL_ARC_LOCAL_VELOCITY_VECTOR,
        control_arc_tangent_weight: float = 0.25,
        control_arc_position_end_weight: float = 4.0,
        control_arc_terminal: str = CONTROL_ARC_TERMINAL_RUNWAY_COMPONENTS,
        control_arc_terminal_cross_track_emphasis: float = 3.0,
        control_arc_terminal_vertical_emphasis: float = 5.0,
        control_terminal_position_weight: float = 1.0,
        control_terminal_velocity_weight: float = 1.0,
        control_terminal_position_scale_m: float = 100.0,
        control_terminal_velocity_scale_mps: float = 10.0,
        control_terminal_clock: str = CONTROL_TERMINAL_CLOCK_STATE_SUPERVISION,
        control_duration_parameterization: str = CONTROL_DURATION_FACTORIZED,
        control_dynamics_backend: str = CONTROL_DYNAMICS_REANCHORED_RK4,
        control_dynamics_model: str = CONTROL_DYNAMICS_POINT_MASS,
        control_bank_time_constant_s: float | None = None,
        control_state_clock: str = CONTROL_STATE_CLOCK_PREDICTED,
        control_state_loss_grid: str = CONTROL_STATE_LOSS_GRID_NATIVE,
        control_state_objective: str = CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
        control_state_duration_gradient: bool = True,
        control_horizon_curriculum_s: tuple[float, ...] = (),
        control_horizon_curriculum_stage_epochs: int = (
            DEFAULT_CONTROL_HORIZON_CURRICULUM_STAGE_EPOCHS
        ),
        control_gradient_clip_norm: float = 0.0,
        control_gradient_clip_policy: str = CONTROL_GRADIENT_CLIP_GLOBAL,
        control_rollout_dt: float | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.airports = tuple(sorted(airport.strip().upper() for airport in airports))
        if not self.airports:
            raise ValueError("TrainingPlan requires at least one airport")
        self.model = model
        self.training_mode = training_mode
        self.prediction_output = prediction_output
        self.n_segments = n_segments
        self.horizon_mode = horizon_mode
        self.full_horizon_steps = full_horizon_steps
        self.window_horizon_steps = window_horizon_steps
        self.epochs = epochs
        self.seed = seed
        self.split_seed = split_seed
        self.device = device
        self.aircraft_type = aircraft_type
        self.aircraft_filter = aircraft_filter
        self.coordinate_frame = coordinate_frame
        self.batch_size = batch_size
        self.cv_folds = cv_folds
        self.cv_parameters = applicable_cv_parameters(
            cv_parameters, horizon_mode, control_dynamics_model
        )
        self.cv_epochs = cv_epochs
        self.cv_patience = cv_patience
        self.random_train_anchor = random_train_anchor
        self.training_cohort_min_future_s = training_cohort_min_future_s
        self.random_train_anchor_min_future_s = random_train_anchor_min_future_s
        self.checkpoint_selection_metric = checkpoint_selection_metric
        self.validation_common_grid_points = validation_common_grid_points
        self.control_effort_weight = control_effort_weight
        self.control_smoothness_weight = control_smoothness_weight
        self.control_dense_state_weight = control_dense_state_weight
        self.control_geometry_weight = control_geometry_weight
        self.control_arc_horizontal_velocity_weight = (
            control_arc_horizontal_velocity_weight
        )
        self.control_arc_vertical_velocity_weight = (
            control_arc_vertical_velocity_weight
        )
        self.control_arc_horizontal_velocity_scale_mps = (
            control_arc_horizontal_velocity_scale_mps
        )
        self.control_arc_vertical_velocity_scale_mps = (
            control_arc_vertical_velocity_scale_mps
        )
        self.control_arc_local_velocity = control_arc_local_velocity
        self.control_arc_tangent_weight = control_arc_tangent_weight
        self.control_arc_position_end_weight = control_arc_position_end_weight
        self.control_arc_terminal = control_arc_terminal
        self.control_arc_terminal_cross_track_emphasis = (
            control_arc_terminal_cross_track_emphasis
        )
        self.control_arc_terminal_vertical_emphasis = (
            control_arc_terminal_vertical_emphasis
        )
        self.control_terminal_position_weight = control_terminal_position_weight
        self.control_terminal_velocity_weight = control_terminal_velocity_weight
        self.control_terminal_position_scale_m = control_terminal_position_scale_m
        self.control_terminal_velocity_scale_mps = control_terminal_velocity_scale_mps
        self.control_terminal_clock = control_terminal_clock
        self.control_duration_parameterization = control_duration_parameterization
        self.control_dynamics_backend = control_dynamics_backend
        self.control_dynamics_model = control_dynamics_model
        self.control_bank_time_constant_s = control_bank_time_constant_s
        self.control_state_clock = control_state_clock
        self.control_state_loss_grid = control_state_loss_grid
        self.control_state_objective = control_state_objective
        self.control_state_duration_gradient = control_state_duration_gradient
        self.control_horizon_curriculum_s = tuple(control_horizon_curriculum_s)
        self.control_horizon_curriculum_stage_epochs = (
            control_horizon_curriculum_stage_epochs
        )
        self.control_gradient_clip_norm = control_gradient_clip_norm
        self.control_gradient_clip_policy = control_gradient_clip_policy
        self.control_rollout_dt = control_rollout_dt

        self.data_manifests = tuple(arrival_manifest_path(airport) for airport in self.airports)
        self.eligibility_rosters = tuple(
            default_lateral_pass_roster_path(manifest) for manifest in self.data_manifests
        )
        scope = self.airports[0] if training_mode == "per-airport" else "POOLED"
        suffix = (
            _prediction_output_tag(prediction_output)
            + _control_duration_tag(
                prediction_output, control_duration_parameterization
            )
            + _control_dynamics_filesystem_tag(
                prediction_output, control_dynamics_backend
            )
            + _control_clock_tag(prediction_output, control_state_clock)
            + _control_terminal_clock_filesystem_tag(
                prediction_output, control_terminal_clock
            )
            + _control_state_loss_grid_tag(prediction_output, control_state_loss_grid)
            + _control_objective_tag(prediction_output, control_state_objective)
            + _terminal_tracking_recipe_tag(
                prediction_output,
                control_state_objective,
                control_dense_state_weight,
                control_geometry_weight,
                control_arc_horizontal_velocity_weight,
                control_arc_vertical_velocity_weight,
                control_arc_horizontal_velocity_scale_mps,
                control_arc_vertical_velocity_scale_mps,
                control_arc_local_velocity,
                control_arc_tangent_weight,
                control_arc_position_end_weight,
                control_arc_terminal,
                control_arc_terminal_cross_track_emphasis,
                control_arc_terminal_vertical_emphasis,
                control_terminal_position_weight,
                control_terminal_velocity_weight,
                control_terminal_position_scale_m,
                control_terminal_velocity_scale_mps,
            )
            + _control_duration_gradient_tag(
                prediction_output, control_state_duration_gradient
            )
            + _control_horizon_curriculum_tag(
                control_horizon_curriculum_s,
                control_horizon_curriculum_stage_epochs,
            )
            + _control_gradient_clip_tag(
                control_gradient_clip_norm, control_gradient_clip_policy
            )
            + _aircraft_filter_tag(aircraft_filter)
            + _frame_tag(coordinate_frame)
            + _anchor_tag(random_train_anchor)
            + _training_cohort_tag(training_cohort_min_future_s)
            + _validation_selection_tag(checkpoint_selection_metric)
        )
        self.train_dir = (
            Path(output_dir)
            if output_dir is not None
            else OPT_OUTPUTS_ROOT
            / scope
            / _bounded_component(f"ts_{model}_{HORIZON_TAGS[horizon_mode]}{suffix}")
        )
        self.cv_dir = self.train_dir / "cross_validation"
        self.cv_results = self.cv_dir / CV_RESULTS_NAME
        self.best_config = self.cv_dir / BEST_CONFIG_NAME
        self.checkpoint = self.train_dir / CHECKPOINT_NAME
        self.checkpoint_metadata = self.train_dir / CHECKPOINT_METADATA_NAME
        self.test_release = self.train_dir / TEST_RELEASE_NAME

    @property
    def pooled(self) -> bool:
        return self.training_mode == "pooled"

    @property
    def label(self) -> str:
        return "POOLED[" + ",".join(self.airports) + "]" if self.pooled else self.airports[0]

    def _data_args(self) -> list[str]:
        data = [
            token for manifest in self.data_manifests for token in ("--data", str(manifest))
        ]
        eligibility = [
            token
            for roster in self.eligibility_rosters
            for token in ("--eligibility-roster", str(roster))
        ]
        return data + eligibility

    def _recipe_args(self, *, include_base_n_segments: bool = True) -> list[str]:
        args = [
            "--model", self.model,
            "--prediction-output", self.prediction_output,
            "--coordinate-frame", self.coordinate_frame,
            "--batch-size", self.batch_size,
            "--horizon-mode", self.horizon_mode,
        ]
        if self.full_horizon_steps is not None:
            args += ["--full-horizon-steps", str(self.full_horizon_steps)]
        if self.window_horizon_steps is not None:
            args += ["--window-horizon-steps", str(self.window_horizon_steps)]
        if self.random_train_anchor:
            args.append("--random-train-anchor")
            args += [
                "--random-train-anchor-min-future-s",
                str(self.random_train_anchor_min_future_s),
            ]
        if self.training_cohort_min_future_s > 0.0:
            args += [
                "--training-cohort-min-future-s",
                str(self.training_cohort_min_future_s),
            ]
        if self.checkpoint_selection_metric != CHECKPOINT_SELECTION_COMMON_GRID_ADE:
            args += ["--checkpoint-selection-metric", self.checkpoint_selection_metric]
        if self.validation_common_grid_points != DEFAULT_VALIDATION_COMMON_GRID_POINTS:
            args += [
                "--validation-common-grid-points",
                str(self.validation_common_grid_points),
            ]
        if self.n_segments is not None and include_base_n_segments:
            args += ["--n-segments", str(self.n_segments)]
        if self.seed is not None:
            args += ["--seed", str(self.seed)]
        if self.split_seed is not None:
            args += ["--split-seed", str(self.split_seed)]
        if self.device is not None:
            args += ["--device", self.device]
        if self.aircraft_type is not None:
            args += ["--aircraft-type", self.aircraft_type]
        args += ["--aircraft-filter", self.aircraft_filter]
        if self.control_effort_weight is not None:
            args += ["--control-effort-weight", str(self.control_effort_weight)]
        if self.control_smoothness_weight is not None:
            args += ["--control-smoothness-weight", str(self.control_smoothness_weight)]
        args += [
            "--control-dense-state-weight",
            str(self.control_dense_state_weight),
            "--control-geometry-weight",
            str(self.control_geometry_weight),
            "--control-arc-horizontal-velocity-weight",
            str(self.control_arc_horizontal_velocity_weight),
            "--control-arc-vertical-velocity-weight",
            str(self.control_arc_vertical_velocity_weight),
            "--control-arc-horizontal-velocity-scale-mps",
            str(self.control_arc_horizontal_velocity_scale_mps),
            "--control-arc-vertical-velocity-scale-mps",
            str(self.control_arc_vertical_velocity_scale_mps),
            "--control-arc-local-velocity",
            self.control_arc_local_velocity,
            "--control-arc-tangent-weight",
            str(self.control_arc_tangent_weight),
            "--control-arc-position-end-weight",
            str(self.control_arc_position_end_weight),
            "--control-arc-terminal",
            self.control_arc_terminal,
            "--control-arc-terminal-cross-track-emphasis",
            str(self.control_arc_terminal_cross_track_emphasis),
            "--control-arc-terminal-vertical-emphasis",
            str(self.control_arc_terminal_vertical_emphasis),
            "--control-terminal-position-weight",
            str(self.control_terminal_position_weight),
            "--control-terminal-velocity-weight",
            str(self.control_terminal_velocity_weight),
            "--control-terminal-position-scale-m",
            str(self.control_terminal_position_scale_m),
            "--control-terminal-velocity-scale-mps",
            str(self.control_terminal_velocity_scale_mps),
            "--control-terminal-clock",
            self.control_terminal_clock,
        ]
        args += [
            "--control-duration-parameterization",
            self.control_duration_parameterization,
            "--control-dynamics-backend",
            self.control_dynamics_backend,
            "--control-dynamics-model",
            self.control_dynamics_model,
        ]
        args += ["--control-state-clock", self.control_state_clock]
        args += ["--control-state-loss-grid", self.control_state_loss_grid]
        args += ["--control-state-objective", self.control_state_objective]
        if not self.control_state_duration_gradient:
            args.append("--no-control-state-duration-gradient")
        if self.control_horizon_curriculum_s:
            args += [
                "--control-horizon-curriculum",
                ",".join(f"{value:g}" for value in self.control_horizon_curriculum_s),
                "--control-horizon-stage-epochs",
                str(self.control_horizon_curriculum_stage_epochs),
            ]
        if self.control_gradient_clip_norm > 0.0:
            args += [
                "--control-gradient-clip-norm",
                f"{self.control_gradient_clip_norm:g}",
                "--control-gradient-clip-policy",
                self.control_gradient_clip_policy,
            ]
        if self.control_rollout_dt is not None:
            args += ["--control-rollout-dt", str(self.control_rollout_dt)]
        return args

    def checkpoint_reuse_error(self) -> str | None:
        if not self.checkpoint.is_file():
            return f"missing checkpoint {self.checkpoint}"
        if not self.checkpoint_metadata.is_file():
            return f"missing checkpoint metadata {self.checkpoint_metadata}"
        if any(not manifest.is_file() for manifest in self.data_manifests):
            return "one or more arrival manifests are missing"
        if any(not roster.is_file() for roster in self.eligibility_rosters):
            return "one or more lateral-pass eligibility rosters are missing"
        try:
            metadata = json.loads(self.checkpoint_metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return f"unreadable checkpoint metadata: {exc}"
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema_version") != CHECKPOINT_METADATA_SCHEMA
        ):
            return "checkpoint metadata has the wrong schema"
        if metadata.get("checkpoint_sha256") != _file_sha256(self.checkpoint):
            return "checkpoint failed SHA-256 validation"
        if metadata.get("arrival_manifests") != _manifest_digests(self.airports):
            return "checkpoint was trained against different arrival manifests"
        if metadata.get("eligibility_rosters") != _eligibility_digests(self.airports):
            return "checkpoint was trained against different eligibility rosters"
        if metadata.get("random_train_anchor") != self.random_train_anchor:
            return (
                "checkpoint random_train_anchor="
                f"{metadata.get('random_train_anchor')!r} does not match requested "
                f"{self.random_train_anchor!r}"
            )
        if (
            metadata.get("training_cohort_min_future_s")
            != self.training_cohort_min_future_s
        ):
            return "checkpoint training-cohort minimum future does not match the recipe"
        if (
            metadata.get("random_train_anchor_min_future_s")
            != self.random_train_anchor_min_future_s
        ):
            return "checkpoint random-anchor minimum future does not match the recipe"
        if metadata.get("checkpoint_selection_metric") != self.checkpoint_selection_metric:
            return "checkpoint validation selection metric does not match the recipe"
        if metadata.get("validation_common_grid_points") != self.validation_common_grid_points:
            return "checkpoint common-grid point count does not match the recipe"
        expected_config, _source = self.resolved_train_config(
            use_best_config=self.cv_reuse_error() is None
        )
        if metadata.get("prediction_output") != expected_config.prediction_output:
            return "checkpoint prediction output does not match the requested recipe"
        if metadata.get("aircraft_filter") != expected_config.aircraft_filter:
            return "checkpoint aircraft filter does not match the requested recipe"
        if metadata.get("horizon_mode") != expected_config.horizon_mode:
            return "checkpoint horizon mode does not match the requested recipe"
        if metadata.get("pred_len") != expected_config.pred_len:
            return "checkpoint output length does not match the requested recipe"
        expected_scheduler = {
            "name": "ReduceLROnPlateau",
            "factor": expected_config.lr_plateau_factor,
            "patience": expected_config.lr_plateau_patience,
        }
        if metadata.get("lr_scheduler") != expected_scheduler:
            return "checkpoint LR scheduler does not match the requested recipe"
        if (
            expected_config.horizon_mode == HORIZON_WINDOW
            and metadata.get("full_horizon_steps") != expected_config.full_horizon_steps
        ):
            return "checkpoint window rollout cap does not match the requested recipe"
        if uses_control_dynamics(expected_config.prediction_output):
            expected_control_recipe = control_recipe(expected_config)
            if metadata.get("control_recipe") != expected_control_recipe:
                return "checkpoint control recipe does not match the requested recipe"
        return None

    def cv_reuse_error(self) -> str | None:
        if not self.cv_results.is_file() or not self.best_config.is_file():
            return "missing cross-validation results or best_config.json"
        if any(not roster.is_file() for roster in self.eligibility_rosters):
            return "one or more lateral-pass eligibility rosters are missing"
        try:
            results = json.loads(self.cv_results.read_text(encoding="utf-8"))
            best = json.loads(self.best_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return f"unreadable cross-validation artifact: {exc}"
        if (
            not isinstance(results, dict)
            or results.get("schema_version") != CV_RESULTS_SCHEMA
            or not isinstance(best, dict)
        ):
            return "cross-validation artifact has the wrong schema"
        if results.get("best_overrides") != best:
            return "cross-validation best_config.json disagrees with cv_results.json"
        if results.get("arrival_manifests") != _manifest_digests(self.airports):
            return "cross-validation used different arrival manifests"
        if results.get("eligibility_rosters") != _eligibility_digests(self.airports):
            return "cross-validation used different eligibility rosters"
        base_config = results.get("base_config")
        expected_config = self._expected_cv_base_config()
        if base_config != expected_config:
            if isinstance(base_config, dict):
                differing = [
                    key for key in expected_config
                    if base_config.get(key) != expected_config[key]
                ]
                detail = differing[0] if differing else "fields"
            else:
                detail = "object"
            return f"cross-validation base_config {detail} does not match current recipe"
        expected_controls = {
            "n_splits": self.cv_folds,
            "search_strategy": "exhaustive_grid",
            "tuned_parameters": list(self.cv_parameters),
            "parameter_grid": parameter_grid(self.cv_parameters),
            "cv_epochs": self.cv_epochs,
            "cv_patience": self.cv_patience,
            "auto_batch_size": self.batch_size == "auto",
        }
        for field, expected in expected_controls.items():
            if results.get(field) != expected:
                return (
                    f"cross-validation {field}={results.get(field)!r} does not match "
                    f"current {expected!r}"
                )
        return None

    def _expected_cv_base_config(self) -> dict[str, object]:
        """Rebuild the exact base TSConfig produced by this plan's CV command."""
        overrides: dict[str, object] = {
            "model": self.model,
            "prediction_output": self.prediction_output,
            "coordinate_frame": self.coordinate_frame,
            "random_train_anchor": self.random_train_anchor,
            "training_cohort_min_future_s": self.training_cohort_min_future_s,
            "random_train_anchor_min_future_s": self.random_train_anchor_min_future_s,
            "checkpoint_selection_metric": self.checkpoint_selection_metric,
            "validation_common_grid_points": self.validation_common_grid_points,
            "horizon_mode": self.horizon_mode,
            "aircraft_filter": self.aircraft_filter,
            "control_state_supervision_clock": self.control_state_clock,
            "control_state_loss_grid": self.control_state_loss_grid,
            "control_state_objective": self.control_state_objective,
            "control_dense_state_loss_weight": self.control_dense_state_weight,
            "control_geometry_loss_weight": self.control_geometry_weight,
            "control_arc_horizontal_velocity_loss_weight": (
                self.control_arc_horizontal_velocity_weight
            ),
            "control_arc_vertical_velocity_loss_weight": (
                self.control_arc_vertical_velocity_weight
            ),
            "control_arc_horizontal_velocity_scale_mps": (
                self.control_arc_horizontal_velocity_scale_mps
            ),
            "control_arc_vertical_velocity_scale_mps": (
                self.control_arc_vertical_velocity_scale_mps
            ),
            "control_arc_local_velocity_parameterization": (
                self.control_arc_local_velocity
            ),
            "control_arc_tangent_loss_weight": self.control_arc_tangent_weight,
            "control_arc_position_end_weight": self.control_arc_position_end_weight,
            "control_arc_terminal_parameterization": self.control_arc_terminal,
            "control_arc_terminal_cross_track_emphasis": (
                self.control_arc_terminal_cross_track_emphasis
            ),
            "control_arc_terminal_vertical_emphasis": (
                self.control_arc_terminal_vertical_emphasis
            ),
            "control_terminal_position_loss_weight": (
                self.control_terminal_position_weight
            ),
            "control_terminal_velocity_loss_weight": (
                self.control_terminal_velocity_weight
            ),
            "control_terminal_position_scale_m": (
                self.control_terminal_position_scale_m
            ),
            "control_terminal_velocity_scale_mps": (
                self.control_terminal_velocity_scale_mps
            ),
            "control_terminal_supervision_clock": self.control_terminal_clock,
            "control_state_duration_gradient": self.control_state_duration_gradient,
            "control_horizon_curriculum_s": self.control_horizon_curriculum_s,
            "control_horizon_curriculum_stage_epochs": (
                self.control_horizon_curriculum_stage_epochs
            ),
            "control_gradient_clip_norm": self.control_gradient_clip_norm,
            "control_gradient_clip_policy": self.control_gradient_clip_policy,
            "control_duration_parameterization": self.control_duration_parameterization,
            "control_dynamics_backend": self.control_dynamics_backend,
        }
        if self.full_horizon_steps is not None:
            overrides["full_horizon_steps"] = self.full_horizon_steps
        if self.window_horizon_steps is not None:
            overrides["window_horizon_steps"] = self.window_horizon_steps
        if self.n_segments is not None:
            overrides["n_segments"] = self.n_segments
        if self.seed is not None:
            overrides["seed"] = self.seed
        if self.split_seed is not None:
            overrides["split_seed"] = self.split_seed
        if self.device is not None:
            overrides["device"] = self.device
        if self.aircraft_type is not None:
            overrides["aircraft_type"] = self.aircraft_type
        if self.control_effort_weight is not None:
            overrides["control_effort_loss_weight"] = self.control_effort_weight
        if self.control_smoothness_weight is not None:
            overrides["control_smoothness_loss_weight"] = self.control_smoothness_weight
        if self.control_rollout_dt is not None:
            overrides["control_rollout_integrator_dt_s"] = self.control_rollout_dt
        if self.batch_size != "auto":
            overrides["batch_size"] = int(self.batch_size)
        return TSConfig(**overrides).to_dict()

    def cv_step(self) -> tuple[str, list[str]]:
        """The isolated outer-train CV command for this training cell."""
        py = sys.executable
        return "cross validation (outer-train only)", [
            py, str(TS_SCRIPT), "cross-validate",
            *self._data_args(),
            "--output-dir", str(self.cv_dir),
            *self._recipe_args(),
            "--folds", str(self.cv_folds),
            "--cv-parameters", ",".join(self.cv_parameters),
            "--cv-epochs", str(self.cv_epochs),
            "--cv-patience", str(self.cv_patience),
        ]

    def train_step(self, *, use_best_config: bool) -> tuple[str, list[str]]:
        """The final-fit command, optionally consuming this cell's locked CV winner."""
        py = sys.executable
        train_command = [
            py, str(TS_SCRIPT), "train",
            *self._data_args(),
            "--output-dir", str(self.train_dir),
            *self._recipe_args(
                include_base_n_segments=(
                    not use_best_config or "n_segments" not in self.cv_parameters
                ),
            ),
        ]
        if self.epochs is not None:
            train_command += ["--epochs", str(self.epochs)]
        if use_best_config:
            train_command += ["--config-overrides", str(self.best_config)]
        return "final train (outer-val early stopping)", train_command

    def resolved_train_config(self, *, use_best_config: bool) -> tuple[TSConfig, str]:
        """Resolve the same final-training recipe encoded by :meth:`train_step`."""
        overrides: dict[str, object] = {}
        source = "TSConfig defaults"
        if use_best_config:
            overrides.update(json.loads(self.best_config.read_text(encoding="utf-8")))
            source = str(self.best_config)

        overrides.update({
            "model": self.model,
            "prediction_output": self.prediction_output,
            "coordinate_frame": self.coordinate_frame,
            "random_train_anchor": self.random_train_anchor,
            "training_cohort_min_future_s": self.training_cohort_min_future_s,
            "random_train_anchor_min_future_s": self.random_train_anchor_min_future_s,
            "checkpoint_selection_metric": self.checkpoint_selection_metric,
            "validation_common_grid_points": self.validation_common_grid_points,
            "horizon_mode": self.horizon_mode,
            "aircraft_filter": self.aircraft_filter,
            "control_state_supervision_clock": self.control_state_clock,
            "control_state_loss_grid": self.control_state_loss_grid,
            "control_state_objective": self.control_state_objective,
            "control_dense_state_loss_weight": self.control_dense_state_weight,
            "control_geometry_loss_weight": self.control_geometry_weight,
            "control_arc_horizontal_velocity_loss_weight": (
                self.control_arc_horizontal_velocity_weight
            ),
            "control_arc_vertical_velocity_loss_weight": (
                self.control_arc_vertical_velocity_weight
            ),
            "control_arc_horizontal_velocity_scale_mps": (
                self.control_arc_horizontal_velocity_scale_mps
            ),
            "control_arc_vertical_velocity_scale_mps": (
                self.control_arc_vertical_velocity_scale_mps
            ),
            "control_arc_local_velocity_parameterization": (
                self.control_arc_local_velocity
            ),
            "control_arc_tangent_loss_weight": self.control_arc_tangent_weight,
            "control_arc_position_end_weight": self.control_arc_position_end_weight,
            "control_arc_terminal_parameterization": self.control_arc_terminal,
            "control_arc_terminal_cross_track_emphasis": (
                self.control_arc_terminal_cross_track_emphasis
            ),
            "control_arc_terminal_vertical_emphasis": (
                self.control_arc_terminal_vertical_emphasis
            ),
            "control_terminal_position_loss_weight": (
                self.control_terminal_position_weight
            ),
            "control_terminal_velocity_loss_weight": (
                self.control_terminal_velocity_weight
            ),
            "control_terminal_position_scale_m": (
                self.control_terminal_position_scale_m
            ),
            "control_terminal_velocity_scale_mps": (
                self.control_terminal_velocity_scale_mps
            ),
            "control_terminal_supervision_clock": self.control_terminal_clock,
            "control_state_duration_gradient": self.control_state_duration_gradient,
            "control_horizon_curriculum_s": self.control_horizon_curriculum_s,
            "control_horizon_curriculum_stage_epochs": (
                self.control_horizon_curriculum_stage_epochs
            ),
            "control_gradient_clip_norm": self.control_gradient_clip_norm,
            "control_gradient_clip_policy": self.control_gradient_clip_policy,
            "control_duration_parameterization": self.control_duration_parameterization,
            "control_dynamics_backend": self.control_dynamics_backend,
        })
        if self.full_horizon_steps is not None:
            overrides["full_horizon_steps"] = self.full_horizon_steps
        if self.window_horizon_steps is not None:
            overrides["window_horizon_steps"] = self.window_horizon_steps
        if self.n_segments is not None and (
            not use_best_config or "n_segments" not in self.cv_parameters
        ):
            overrides["n_segments"] = self.n_segments
        if self.epochs is not None:
            overrides["epochs"] = self.epochs
        if self.seed is not None:
            overrides["seed"] = self.seed
        if self.split_seed is not None:
            overrides["split_seed"] = self.split_seed
        if self.device is not None:
            overrides["device"] = self.device
        if self.aircraft_type is not None:
            overrides["aircraft_type"] = self.aircraft_type
        if self.control_effort_weight is not None:
            overrides["control_effort_loss_weight"] = self.control_effort_weight
        if self.control_smoothness_weight is not None:
            overrides["control_smoothness_loss_weight"] = self.control_smoothness_weight
        if self.control_rollout_dt is not None:
            overrides["control_rollout_integrator_dt_s"] = self.control_rollout_dt
        if self.batch_size != "auto":
            overrides["batch_size"] = int(self.batch_size)
        return TSConfig(**overrides), source

    def steps(self, *, skip_cv: bool, reuse_checkpoint: bool) -> list[tuple[str, list[str]]]:
        if reuse_checkpoint:
            return []
        named: list[tuple[str, list[str]]] = []
        cv_available = self.cv_reuse_error() is None
        if not skip_cv:
            named.append(self.cv_step())
        # A CV stage in this run will create it; --skip-cv reuses it only when its manifest
        # provenance still matches. Otherwise final training deliberately uses base defaults.
        named.append(self.train_step(use_best_config=not skip_cv or cv_available))
        return named


class PredictionPlan:
    """Per-airport publication tail for a completed TrainingPlan."""

    def __init__(
        self,
        training: TrainingPlan,
        airport: str,
        outputs: tuple[str, ...],
        *,
        split: str = "val",
        experiment_tag: str | None = None,
    ) -> None:
        self.training = training
        self.airport = airport.upper()
        self.outputs = outputs
        if split not in PREDICTION_SPLITS:
            raise ValueError(f"unknown prediction split {split!r}")
        self.split = split
        self.data_manifest = arrival_manifest_path(self.airport)
        scope = "pooled_" if training.pooled else ""
        frame = _frame_tag(training.coordinate_frame)
        anchor = _anchor_tag(training.random_train_anchor)
        training_cohort = _training_cohort_tag(
            training.training_cohort_min_future_s
        )
        validation_selection = _validation_selection_tag(
            training.checkpoint_selection_metric
        )
        prediction_output = _prediction_output_tag(training.prediction_output)
        control_duration = _control_duration_tag(
            training.prediction_output, training.control_duration_parameterization
        )
        control_dynamics_filesystem = _control_dynamics_filesystem_tag(
            training.prediction_output, training.control_dynamics_backend
        )
        control_dynamics = _control_dynamics_tag(
            training.prediction_output, training.control_dynamics_backend
        )
        control_clock = _control_clock_tag(
            training.prediction_output, training.control_state_clock
        )
        control_terminal_clock_filesystem = (
            _control_terminal_clock_filesystem_tag(
                training.prediction_output, training.control_terminal_clock
            )
        )
        control_terminal_clock = _control_terminal_clock_tag(
            training.prediction_output, training.control_terminal_clock
        )
        control_state_loss_grid = _control_state_loss_grid_tag(
            training.prediction_output, training.control_state_loss_grid
        )
        control_objective = _control_objective_tag(
            training.prediction_output, training.control_state_objective
        )
        duration_gradient = _control_duration_gradient_tag(
            training.prediction_output, training.control_state_duration_gradient
        )
        horizon_curriculum = _control_horizon_curriculum_tag(
            training.control_horizon_curriculum_s,
            training.control_horizon_curriculum_stage_epochs,
        )
        gradient_clip = _control_gradient_clip_tag(
            training.control_gradient_clip_norm,
            training.control_gradient_clip_policy,
        )
        aircraft_filter = _aircraft_filter_tag(training.aircraft_filter)
        tag = f"_{experiment_tag}" if experiment_tag else ""
        horizon_tag = HORIZON_TAGS[training.horizon_mode]
        stem = (
            f"{scope}{training.model}{prediction_output}{control_duration}"
            f"{control_dynamics_filesystem}"
            f"{control_clock}{control_terminal_clock_filesystem}_{horizon_tag}"
            f"{control_state_loss_grid}{control_objective}{duration_gradient}"
            f"{horizon_curriculum}{gradient_clip}"
            f"{aircraft_filter}{frame}{anchor}{training_cohort}"
            f"{validation_selection}{tag}_{split}"
        )
        self.pred_dir = (
            OPT_OUTPUTS_ROOT / self.airport / _bounded_component(f"ts_pred_{stem}")
        )
        self.summary = self.pred_dir / "summary.json"
        self.report = self.pred_dir / "evaluation_report.json"
        self.report_html = self.pred_dir / "evaluation_report.html"
        category_scope = "pooled_" if training.pooled else ""
        self.category = (
            f"ts_{category_scope}{MODEL_SHORT[training.model]}{prediction_output}"
            f"{control_duration}{control_dynamics}"
            f"{control_terminal_clock}_{horizon_tag}"
            f"{control_state_loss_grid}{control_objective}{duration_gradient}"
            f"{horizon_curriculum}{gradient_clip}"
            f"{aircraft_filter}{frame}{anchor}{training_cohort}"
            f"{validation_selection}{tag}_{split}"
        )
        # The label is the canonical run grammar (run_naming, single source), derived
        # from the exact config this cell trains with; scope and tag ride as meta.
        scope_note = "pooled cohort" if training.pooled else f"{training.label} cohort"
        label_extra = [scope_note]
        if experiment_tag:
            label_extra.append(experiment_tag)
        self.label = category_display_label(
            self.split,
            run_display_name(
                training._expected_cv_base_config(), extra=tuple(label_extra)
            ),
        )
        self.comparison_dir = (
            COMPARISON_AIRPORTS_ROOT / self.airport / "comparison" / self.category
        )

    def steps(self) -> list[tuple[str, list[str]]]:
        py = sys.executable
        named: list[tuple[str, list[str]]] = []
        predict = [
            py, str(TS_SCRIPT), "predict",
            "--checkpoint", str(self.training.checkpoint),
            "--data", str(self.data_manifest),
            "--eligibility-roster",
            str(default_lateral_pass_roster_path(self.data_manifest)),
            "--output-dir", str(self.pred_dir),
            "--split", self.split,
        ]
        if self.split == "test":
            predict.append("--test-release")
        if self.training.device is not None:
            predict += ["--device", self.training.device]
        named.append((f"predict ({self.split} split)", predict))
        named.append(("evaluation report", [
            py, "-m", "evaluation",
            "--input", str(self.pred_dir),
            "--output", str(self.report),
        ]))
        if "eval" in self.outputs:
            named.append(("evaluation HTML", [
                py, "-m", "evaluation.visualize",
                "--input", str(self.pred_dir),
                "--output", str(self.report_html),
            ]))
        if "czml" in self.outputs:
            named.append(("comparison CZML", [
                py, str(CZML_SCRIPT),
                "--summary", str(self.summary),
                "--output-dir", str(self.comparison_dir),
                "--airport", self.airport,
                "--category", self.category,
                "--category-label", self.label,
                "--dataset-split", self.split,
                "--evaluation-report", str(self.report),
            ]))
        return named


def _run_steps(
    context: str,
    steps: list[tuple[str, list[str]]],
    *,
    dry_run: bool,
    before_step: Callable[[str, list[str]], None] | None = None,
) -> None:
    total = len(steps)
    for index, (label, command) in enumerate(steps, 1):
        qualified = f"{index}/{total} {label}"
        if before_step is not None:
            before_step(label, command)
        if dry_run:
            print(f"   [{qualified}] {' '.join(command)}")
            continue
        print(f"\n=== [{context} · {qualified}] ===\n{' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_training(
    plan: TrainingPlan, *, dry_run: bool, skip_cv: bool, skip_train: bool
) -> bool:
    missing = [manifest for manifest in plan.data_manifests if not manifest.exists()]
    if missing:
        print(f"   ⚠ skip {plan.label}: missing {missing[0]}")
        return False
    reuse_error = plan.checkpoint_reuse_error() if skip_train else None
    reuse = skip_train and reuse_error is None
    mode = "reuse checkpoint" if reuse else "train final checkpoint"
    horizon_label = HORIZON_LABELS[plan.horizon_mode]
    print(
        f"\n━━ {plan.label} [{plan.model} · {plan.prediction_output} · "
        f"{horizon_label} · {plan.coordinate_frame}] · {mode}"
    )
    print(f"   manifests : {len(plan.data_manifests)}")
    print(f"   CV        : {plan.cv_dir}")
    print(f"   training  : {plan.train_dir}")
    if skip_train and not reuse:
        print(f"   (checkpoint not reusable: {reuse_error} → rebuilding)")
    if skip_cv and not reuse and plan.cv_reuse_error() is not None:
        print("   (CV skipped and no reusable CV artifact → base hyperparameters)")

    def print_final_config(label: str, command: list[str]) -> None:
        if not label.startswith("final train"):
            return
        use_best_config = "--config-overrides" in command
        if not skip_cv and dry_run:
            print("   config    : pending CV selection")
            return
        config, source = plan.resolved_train_config(use_best_config=use_best_config)
        batch = "auto (GPU probe)" if plan.batch_size == "auto" else str(config.batch_size)
        anchor = "random" if config.random_train_anchor else "fixed L-1"
        print(f"   config    : {source}")
        print(
            f"   trajectory: dt={config.dt_s:g}s, L={config.seq_len}, "
            f"prediction_output={config.prediction_output}, mode={config.horizon_mode}, "
            f"output={config.pred_len}, "
            f"N={config.n_segments}, H_full={config.full_horizon_steps}, "
            f"H_window={config.window_horizon_steps}, "
            f"frame={config.coordinate_frame}, anchor={anchor}"
        )
        print(
            f"   network   : d_model={config.d_model}, d_ff={config.d_ff}, "
            f"heads={config.n_heads}, layers={config.e_layers}, dropout={config.dropout:g}"
        )
        print(
            f"   optimizer : lr={config.learning_rate:g}, weight_decay={config.weight_decay:g}, "
            f"epochs={config.epochs}, patience={config.patience}"
        )
        if config.prediction_output == PREDICTION_STATE:
            print(
                f"   loss      : true-time 3D position/{config.position_loss_scale_m:g}m "
                f"+ {config.state_endpoint_loss_weight:g}× output-endpoint position "
                f"+ final_time/{config.final_time_scale_s:g}s; "
                "future velocity derived from position"
            )
        else:
            print(
                f"   loss      : final_time={config.final_time_loss_weight:g}, "
                f"kinematic={config.kinematic_consistency_loss_weight:g}, "
                f"terminal={config.terminal_loss_weight:g}"
            )
        if uses_control_dynamics(config.prediction_output):
            print(
                f"   control   : effort={config.control_effort_loss_weight:g}, "
                f"smoothness={config.control_smoothness_loss_weight:g}, "
                f"duration={config.control_duration_parameterization}, "
                f"dynamics={config.control_dynamics_backend}, "
                f"state_clock={config.control_state_supervision_clock}, "
                f"rollout_dt={config.control_rollout_integrator_dt_s:g}s"
            )
            if (
                config.control_state_objective
                == CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY
            ):
                print(
                    "   arc-geometry: "
                    f"geometry={config.control_geometry_loss_weight:g}, "
                    "local_velocity="
                    f"{config.control_arc_horizontal_velocity_loss_weight:g}/"
                    f"{config.control_arc_horizontal_velocity_scale_mps:g}mps horiz, "
                    f"{config.control_arc_vertical_velocity_loss_weight:g}/"
                    f"{config.control_arc_vertical_velocity_scale_mps:g}mps vertical, "
                    f"local={config.control_arc_local_velocity_parameterization}, "
                    f"tangent={config.control_arc_tangent_loss_weight:g}, "
                    f"position_end={config.control_arc_position_end_weight:g}, "
                    f"terminal_mode={config.control_arc_terminal_parameterization}, "
                    "terminal_emphasis="
                    f"cross×{config.control_arc_terminal_cross_track_emphasis:g}/"
                    f"vertical×{config.control_arc_terminal_vertical_emphasis:g}, "
                    f"position={config.control_terminal_position_loss_weight:g}/"
                    f"{config.control_terminal_position_scale_m:g}m, "
                    f"velocity={config.control_terminal_velocity_loss_weight:g}/"
                    f"{config.control_terminal_velocity_scale_mps:g}mps"
                )
            if config.control_horizon_curriculum_s:
                horizons = "→".join(
                    f"{value:g}s" for value in config.control_horizon_curriculum_s
                )
                print(
                    f"   curriculum: {horizons} × "
                    f"{config.control_horizon_curriculum_stage_epochs} epochs -> full"
                )
            if config.control_gradient_clip_norm > 0.0:
                print(
                    f"   stability : gradient clip={config.control_gradient_clip_norm:g}, "
                    f"policy={config.control_gradient_clip_policy}"
                )
        print(
            f"   runtime   : batch={batch}, device={config.device}, seed={config.seed}, "
            f"aircraft={config.aircraft_type}, aircraft_filter={config.aircraft_filter}"
        )

    _run_steps(
        f"{plan.label} · {plan.model} · {horizon_label}",
        plan.steps(skip_cv=skip_cv, reuse_checkpoint=reuse),
        dry_run=dry_run,
        before_step=print_final_config,
    )
    return True


def run_prediction(plan: PredictionPlan, *, dry_run: bool) -> None:
    print(f"\n  ━━ publish {plan.airport}: {plan.pred_dir}")
    horizon_label = HORIZON_LABELS[plan.training.horizon_mode]
    _run_steps(
        f"{plan.airport} · {plan.training.model} · {horizon_label}",
        plan.steps(),
        dry_run=dry_run,
    )


def freeze_test_release(plan: TrainingPlan, *, dry_run: bool) -> None:
    """Make outer-test access explicit and bind it to this exact checkpoint/data roster."""
    command = [
        sys.executable,
        str(TS_SCRIPT),
        "freeze-test",
        "--checkpoint", str(plan.checkpoint),
        *plan._data_args(),
    ]
    _run_steps(
        f"{plan.label} · {plan.model} · outer-test release",
        [("freeze-test (irreversible)", command)],
        dry_run=dry_run,
    )


def _parse_csv(raw: str, allowed: tuple[str, ...], flag: str) -> tuple[str, ...]:
    tokens = tuple(token.strip() for token in raw.split(",") if token.strip())
    unknown = [token for token in tokens if token not in allowed]
    if unknown or not tokens:
        raise argparse.ArgumentTypeError(f"{flag} takes a comma list from {allowed}, got {raw!r}")
    return tokens


def _parse_positive_float_csv(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(token.strip()) for token in raw.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--control-horizon-curriculum takes comma-separated seconds"
        ) from exc
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError(
            "--control-horizon-curriculum requires positive seconds"
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-mode",
        choices=TRAINING_MODES,
        default="pooled",
        help="training scope (default: pooled)",
    )
    parser.add_argument("--airport", default=None,
                        help="optional single airport filter; otherwise discover every K-airport")
    parser.add_argument("--models", type=lambda raw: _parse_csv(raw, MODELS, "--models"),
                        default=MODELS, metavar=",".join(MODELS),
                        help="models to train (default: itransformer,patchtst)")
    parser.add_argument(
        "--prediction-output",
        choices=PREDICTION_OUTPUTS,
        default=PREDICTION_STATE,
        help="state baseline or bounded controls with differentiable rollout",
    )
    parser.add_argument("--n-segments", type=int, default=None,
                        help="base N for normalized progress; CV also tunes N")
    parser.add_argument(
        "--horizon-mode",
        choices=HORIZON_MODES,
        default=HORIZON_NORMALIZED,
        help="prediction horizon (default: normalized)",
    )
    parser.add_argument(
        "--full-horizon-steps",
        type=int,
        default=None,
        help="H_full physical-dt outputs and window recursion cap (default: 300)",
    )
    parser.add_argument(
        "--window-horizon-steps",
        type=int,
        default=None,
        help="H_window outputs per recursive pass (default: 30)",
    )
    parser.add_argument("--outputs", type=lambda raw: _parse_csv(raw, OUTPUT_KINDS, "--outputs"),
                        default=OUTPUT_KINDS, metavar="czml,eval")
    parser.add_argument(
        "--split", choices=(*PREDICTION_SPLITS, "development"), default="development",
        help="publication split; default development publishes train and validation only",
    )
    parser.add_argument(
        "--release-test",
        action="store_true",
        help="irreversibly freeze and evaluate outer-test; required with --split test",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--aircraft-type", default=None)
    parser.add_argument(
        "--aircraft-filter",
        choices=AIRCRAFT_FILTERS,
        default=AIRCRAFT_FILTER_ALL,
        help="fleet contract (openap-direct excludes synonyms, presets and fallbacks)",
    )
    parser.add_argument("--coordinate-frame", choices=COORDINATE_FRAMES, default="enu")
    parser.add_argument("--batch-size", default="2048",
                        help="positive integer or auto (default: 2048)")
    parser.add_argument("--control-effort-weight", type=float, default=None)
    parser.add_argument("--control-smoothness-weight", type=float, default=None)
    parser.add_argument("--control-dense-state-weight", type=float, default=0.25)
    parser.add_argument("--control-geometry-weight", type=float, default=0.75)
    parser.add_argument(
        "--control-arc-horizontal-velocity-weight", type=float, default=0.25
    )
    parser.add_argument(
        "--control-arc-vertical-velocity-weight", type=float, default=0.25
    )
    parser.add_argument(
        "--control-arc-horizontal-velocity-scale-mps", type=float, default=10.0
    )
    parser.add_argument(
        "--control-arc-vertical-velocity-scale-mps", type=float, default=2.0
    )
    parser.add_argument(
        "--control-arc-local-velocity",
        choices=CONTROL_ARC_LOCAL_VELOCITY_PARAMETERIZATIONS,
        default=CONTROL_ARC_LOCAL_VELOCITY_VECTOR,
    )
    parser.add_argument("--control-arc-tangent-weight", type=float, default=0.25)
    parser.add_argument("--control-arc-position-end-weight", type=float, default=4.0)
    parser.add_argument(
        "--control-arc-terminal",
        choices=CONTROL_ARC_TERMINAL_PARAMETERIZATIONS,
        default=CONTROL_ARC_TERMINAL_RUNWAY_COMPONENTS,
    )
    parser.add_argument(
        "--control-arc-terminal-cross-track-emphasis", type=float, default=3.0
    )
    parser.add_argument(
        "--control-arc-terminal-vertical-emphasis", type=float, default=5.0
    )
    parser.add_argument("--control-terminal-position-weight", type=float, default=1.0)
    parser.add_argument("--control-terminal-velocity-weight", type=float, default=1.0)
    parser.add_argument("--control-terminal-position-scale-m", type=float, default=100.0)
    parser.add_argument("--control-terminal-velocity-scale-mps", type=float, default=10.0)
    parser.add_argument(
        "--control-terminal-clock",
        choices=CONTROL_TERMINAL_CLOCKS,
        default=CONTROL_TERMINAL_CLOCK_STATE_SUPERVISION,
    )
    parser.add_argument(
        "--control-duration-parameterization",
        choices=CONTROL_DURATION_PARAMETERIZATIONS,
        default=CONTROL_DURATION_FACTORIZED,
    )
    parser.add_argument(
        "--control-dynamics-backend",
        choices=CONTROL_DYNAMICS_BACKENDS,
        default=CONTROL_DYNAMICS_REANCHORED_RK4,
    )
    parser.add_argument(
        "--control-dynamics-model",
        choices=CONTROL_DYNAMICS_MODELS,
        default=CONTROL_DYNAMICS_POINT_MASS,
    )
    parser.add_argument(
        "--control-state-clock",
        choices=CONTROL_STATE_CLOCKS,
        default=CONTROL_STATE_CLOCK_PREDICTED,
    )
    parser.add_argument(
        "--control-state-loss-grid",
        choices=CONTROL_STATE_LOSS_GRIDS,
        default=CONTROL_STATE_LOSS_GRID_NATIVE,
    )
    parser.add_argument(
        "--control-state-objective",
        choices=CONTROL_STATE_OBJECTIVES,
        default=CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    )
    parser.add_argument(
        "--control-state-duration-gradient",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--control-horizon-curriculum",
        type=_parse_positive_float_csv,
        default=(),
        metavar="SECONDS,...",
    )
    parser.add_argument(
        "--control-horizon-stage-epochs",
        type=int,
        default=DEFAULT_CONTROL_HORIZON_CURRICULUM_STAGE_EPOCHS,
    )
    parser.add_argument(
        "--control-gradient-clip-norm",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--control-gradient-clip-policy",
        choices=CONTROL_GRADIENT_CLIP_POLICIES,
        default=CONTROL_GRADIENT_CLIP_GLOBAL,
    )
    parser.add_argument("--control-rollout-dt", type=float, default=None)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument(
        "--cv-parameters",
        type=lambda raw: _parse_csv(raw, tuple(CV_PARAMETER_GRIDS), "--cv-parameters"),
        default=DEFAULT_CV_PARAMETERS,
        metavar=",".join(DEFAULT_CV_PARAMETERS),
        help="parameters included in the exhaustive CV grid",
    )
    parser.add_argument("--cv-epochs", type=int, default=DEFAULT_CV_EPOCHS)
    parser.add_argument("--cv-patience", type=int, default=DEFAULT_CV_PATIENCE)
    parser.add_argument(
        "--random-train-anchor",
        action="store_true",
        help="train rolling forecasts from random anchors; default is fixed anchor L-1",
    )
    parser.add_argument(
        "--training-cohort-min-future-s",
        type=float,
        default=0.0,
        help="train-only fixed-L-1 future-duration floor for controlled comparisons",
    )
    parser.add_argument(
        "--random-train-anchor-min-future-s",
        type=float,
        default=DEFAULT_RANDOM_TRAIN_ANCHOR_MIN_FUTURE_S,
    )
    parser.add_argument(
        "--checkpoint-selection-metric",
        choices=CHECKPOINT_SELECTION_METRICS,
        default=CHECKPOINT_SELECTION_COMMON_GRID_ADE,
    )
    parser.add_argument(
        "--validation-common-grid-points",
        type=int,
        default=DEFAULT_VALIDATION_COMMON_GRID_POINTS,
    )
    parser.add_argument(
        "--skip-cv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip CV by default; use --no-skip-cv to run it",
    )
    parser.add_argument("--skip-train", action="store_true",
                        help="reuse a checkpoint only when all selected manifest digests match")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.batch_size != "auto":
        try:
            if int(args.batch_size) <= 0:
                raise ValueError
        except ValueError:
            parser.error("--batch-size must be a positive integer or 'auto'")
    if args.split == "test" and not args.release_test:
        parser.error(
            "--split test requires --release-test after all model and hyperparameter "
            "decisions are frozen"
        )
    if args.split != "test" and args.release_test:
        parser.error("--release-test is valid only with --split test")

    if args.airport:
        airports = [args.airport.strip().upper()]
    else:
        airports = discover_k_airports()
        if not airports:
            parser.error(f"no K-prefixed airports with arrivals manifests under {HARVEST_ROOT}")

    if not args.dry_run:
        print("preparing evaluation-derived lateral-pass eligibility rosters")
        for airport in airports:
            roster = ensure_lateral_pass_roster(arrival_manifest_path(airport))
            print(f"  {airport}: {roster}")

    scopes = (
        [tuple(airports)]
        if args.training_mode == "pooled"
        else [(airport,) for airport in airports]
    )
    cells = [
        (scope, model)
        for scope in scopes
        for model in args.models
    ]
    print(f"{len(cells)} training cell(s), mode={args.training_mode}, airports={','.join(airports)}")

    publish_splits = ("train", "val") if args.split == "development" else (args.split,)
    completed = 0
    for scope, model in cells:
        training = TrainingPlan(
            scope,
            model,
            training_mode=args.training_mode,
            prediction_output=args.prediction_output,
            n_segments=args.n_segments,
            horizon_mode=args.horizon_mode,
            full_horizon_steps=args.full_horizon_steps,
            window_horizon_steps=args.window_horizon_steps,
            epochs=args.epochs,
            seed=args.seed,
            split_seed=args.split_seed,
            device=args.device,
            aircraft_type=args.aircraft_type,
            aircraft_filter=args.aircraft_filter,
            coordinate_frame=args.coordinate_frame,
            batch_size=args.batch_size,
            cv_folds=args.cv_folds,
            cv_parameters=args.cv_parameters,
            cv_epochs=args.cv_epochs,
            cv_patience=args.cv_patience,
            random_train_anchor=args.random_train_anchor,
            training_cohort_min_future_s=args.training_cohort_min_future_s,
            random_train_anchor_min_future_s=args.random_train_anchor_min_future_s,
            checkpoint_selection_metric=args.checkpoint_selection_metric,
            validation_common_grid_points=args.validation_common_grid_points,
            control_effort_weight=args.control_effort_weight,
            control_smoothness_weight=args.control_smoothness_weight,
            control_dense_state_weight=args.control_dense_state_weight,
            control_geometry_weight=args.control_geometry_weight,
            control_arc_horizontal_velocity_weight=(
                args.control_arc_horizontal_velocity_weight
            ),
            control_arc_vertical_velocity_weight=(
                args.control_arc_vertical_velocity_weight
            ),
            control_arc_horizontal_velocity_scale_mps=(
                args.control_arc_horizontal_velocity_scale_mps
            ),
            control_arc_vertical_velocity_scale_mps=(
                args.control_arc_vertical_velocity_scale_mps
            ),
            control_arc_local_velocity=args.control_arc_local_velocity,
            control_arc_tangent_weight=args.control_arc_tangent_weight,
            control_arc_position_end_weight=args.control_arc_position_end_weight,
            control_arc_terminal=args.control_arc_terminal,
            control_arc_terminal_cross_track_emphasis=(
                args.control_arc_terminal_cross_track_emphasis
            ),
            control_arc_terminal_vertical_emphasis=(
                args.control_arc_terminal_vertical_emphasis
            ),
            control_terminal_position_weight=args.control_terminal_position_weight,
            control_terminal_velocity_weight=args.control_terminal_velocity_weight,
            control_terminal_position_scale_m=args.control_terminal_position_scale_m,
            control_terminal_velocity_scale_mps=(
                args.control_terminal_velocity_scale_mps
            ),
            control_terminal_clock=args.control_terminal_clock,
            control_duration_parameterization=args.control_duration_parameterization,
            control_dynamics_backend=args.control_dynamics_backend,
            control_dynamics_model=args.control_dynamics_model,
            control_state_clock=args.control_state_clock,
            control_state_loss_grid=args.control_state_loss_grid,
            control_state_objective=args.control_state_objective,
            control_state_duration_gradient=args.control_state_duration_gradient,
            control_horizon_curriculum_s=args.control_horizon_curriculum,
            control_horizon_curriculum_stage_epochs=(
                args.control_horizon_stage_epochs
            ),
            control_gradient_clip_norm=args.control_gradient_clip_norm,
            control_gradient_clip_policy=args.control_gradient_clip_policy,
            control_rollout_dt=args.control_rollout_dt,
        )
        if not run_training(
            training,
            dry_run=args.dry_run,
            skip_cv=args.skip_cv,
            skip_train=args.skip_train,
        ):
            continue
        if args.split == "test":
            freeze_test_release(training, dry_run=args.dry_run)
        for airport in scope:
            for split in publish_splits:
                run_prediction(
                    PredictionPlan(training, airport, tuple(args.outputs), split=split),
                    dry_run=args.dry_run,
                )
        completed += 1

    verb = "previewed" if args.dry_run else "completed"
    print(f"\n✓ {verb} {completed}/{len(cells)} training cell(s) "
          f"[CV={'skip/reuse' if args.skip_cv else 'run'}, splits={','.join(publish_splits)}]")


if __name__ == "__main__":
    main()
