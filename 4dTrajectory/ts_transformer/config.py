"""The single configuration object for a ts_transformer run.

Both vendored architectures take one ``configs`` object and read attributes off it —
that is upstream's contract (they were driven by an argparse namespace), and this
dataclass is the drop-in. Keeping data, architecture and training knobs in ONE frozen
object is deliberate: the whole thing is serialised into every checkpoint, so a trained
artifact carries the exact recipe that produced it and inference never has to guess the
resample step, the channel order, or the prediction-time grid.

Fields are grouped by who reads them. The "read by both" and per-model groups are named
exactly as upstream expects — do not rename them without editing the vendored code, which
would break the byte-identical property PROVENANCE.md promises.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

# Channel order is a hard contract between the data build, the model, and the export.
# It lives in channels.py; imported here so the default cannot drift from it.
from channels import CHANNELS
from reference_velocity import (
    REFERENCE_VELOCITY_SOURCES,
    REFERENCE_VELOCITY_TRACK_FIT,
)

MODELS = ("itransformer", "patchtst")
COORDINATE_FRAMES = ("enu", "runway-aligned")
AIRCRAFT_FILTER_ALL = "all"
AIRCRAFT_FILTER_OPENAP_DIRECT = "openap-direct"
AIRCRAFT_FILTERS = (AIRCRAFT_FILTER_ALL, AIRCRAFT_FILTER_OPENAP_DIRECT)
PREDICTION_STATE = "state"
PREDICTION_CONTROL = "control"
PREDICTION_OUTPUTS = (PREDICTION_STATE, PREDICTION_CONTROL)
CONTROL_STATE_CLOCK_PREDICTED = "predicted"
CONTROL_STATE_CLOCK_OBSERVED = "observed"
CONTROL_STATE_CLOCKS = (
    CONTROL_STATE_CLOCK_PREDICTED,
    CONTROL_STATE_CLOCK_OBSERVED,
)
CONTROL_TERMINAL_CLOCK_STATE_SUPERVISION = "state-supervision"
CONTROL_TERMINAL_CLOCK_PREDICTED = "predicted"
CONTROL_TERMINAL_CLOCK_PREDICTED_DETACHED_TIME = "predicted-detached-time"
CONTROL_TERMINAL_CLOCKS = (
    CONTROL_TERMINAL_CLOCK_STATE_SUPERVISION,
    CONTROL_TERMINAL_CLOCK_PREDICTED,
    CONTROL_TERMINAL_CLOCK_PREDICTED_DETACHED_TIME,
)
CONTROL_STATE_LOSS_GRID_NATIVE = "native-segment-endpoints"
CONTROL_STATE_LOSS_GRID_FIXED_DT = "fixed-dt"
CONTROL_STATE_LOSS_GRIDS = (
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
)
CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE = "normalized-mse"
CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY = "arc-length-geometry"
CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION = "true-time-position"
CONTROL_STATE_OBJECTIVES = (
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
)
CONTROL_ARC_TERMINAL_VECTOR_NORM = "vector-norm"
CONTROL_ARC_TERMINAL_RUNWAY_COMPONENTS = "runway-components"
CONTROL_ARC_TERMINAL_PARAMETERIZATIONS = (
    CONTROL_ARC_TERMINAL_VECTOR_NORM,
    CONTROL_ARC_TERMINAL_RUNWAY_COMPONENTS,
)
CONTROL_ARC_LOCAL_VELOCITY_VECTOR = "vector-components"
CONTROL_ARC_LOCAL_VELOCITY_TANGENT_SPEED = "tangent-speed"
CONTROL_ARC_LOCAL_VELOCITY_PARAMETERIZATIONS = (
    CONTROL_ARC_LOCAL_VELOCITY_VECTOR,
    CONTROL_ARC_LOCAL_VELOCITY_TANGENT_SPEED,
)
CONTROL_DURATION_FACTORIZED = "factorized"
CONTROL_DURATION_UNIFORM = "uniform"
CONTROL_DURATION_PARAMETERIZATIONS = (
    CONTROL_DURATION_FACTORIZED,
    CONTROL_DURATION_UNIFORM,
)
# The flight model itself, orthogonal to the state representation below. ``point-mass``
# applies each piecewise-constant control instantly; ``first-order-lag`` makes the three
# controls states that chase their command with a time constant, so bank, thrust and load
# factor are continuous across a segment boundary instead of stepping.
CONTROL_DYNAMICS_POINT_MASS = "point-mass"
CONTROL_DYNAMICS_FIRST_ORDER_LAG = "first-order-lag"
CONTROL_DYNAMICS_MODELS = (
    CONTROL_DYNAMICS_POINT_MASS,
    CONTROL_DYNAMICS_FIRST_ORDER_LAG,
)

CONTROL_DYNAMICS_REANCHORED_RK4 = "reanchored-rk4"
CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY = "transport-chart-velocity"
CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY = (
    "scaled-transport-chart-velocity"
)
CONTROL_DYNAMICS_BACKENDS = (
    CONTROL_DYNAMICS_REANCHORED_RK4,
    CONTROL_DYNAMICS_TRANSPORT_CHART_VELOCITY,
    CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
)
CONTROL_GRADIENT_CLIP_GLOBAL = "global"
CONTROL_GRADIENT_CLIP_FINAL_TIME_DECOUPLED = "final-time-decoupled"
CONTROL_GRADIENT_CLIP_POLICIES = (
    CONTROL_GRADIENT_CLIP_GLOBAL,
    CONTROL_GRADIENT_CLIP_FINAL_TIME_DECOUPLED,
)

CONTROL_RECIPE_CUSTOM = "custom"
CONTROL_RECIPE_SIMPLE_V1 = "simple-v1"
# simple-v1 with the lagged flight model substituted and nothing else changed, so the two
# recipes differ by exactly one axis and a paired comparison measures the flight model
# rather than a bundle of choices. The three time constants are deliberately NOT frozen:
# tau_bank is the parameter the CV sweep resolves.
CONTROL_RECIPE_SIMPLE_V1_LAG = "simple-v1-lag"
CONTROL_RECIPE_NAMES = (
    CONTROL_RECIPE_CUSTOM,
    CONTROL_RECIPE_SIMPLE_V1,
    CONTROL_RECIPE_SIMPLE_V1_LAG,
)

CHECKPOINT_SELECTION_OBJECTIVE = "fixed-anchor-objective"
CHECKPOINT_SELECTION_COMMON_GRID_ADE = "fixed-anchor-common-grid-ade"
CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY = "fixed-anchor-arc-length-geometry"
CHECKPOINT_SELECTION_METRICS = (
    CHECKPOINT_SELECTION_OBJECTIVE,
    CHECKPOINT_SELECTION_COMMON_GRID_ADE,
    CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY,
)


def uses_control_dynamics(prediction_output: str) -> bool:
    """Whether an output strategy requires per-flight aircraft dynamics."""
    return prediction_output == PREDICTION_CONTROL


HORIZON_NORMALIZED = "normalized"
HORIZON_FULL = "full"
HORIZON_WINDOW = "window"
HORIZON_MODES = (HORIZON_NORMALIZED, HORIZON_FULL, HORIZON_WINDOW)

# Time grid. ADS-B arrives at ~1 Hz but irregularly; 2 s is the resample step — fine enough
# not to smooth away the turn onto final, coarse enough not to invent samples between
# reports.
#
# Everything below is sized from the MEASURED duration distribution of the 3747 harvested
# arrivals (5 airports, rostered by `arrivals/manifest.json`, truncated at the 25 km ring):
#
#     p5  235 s | p25 271 s | p50 328 s | p75 533 s | p90 607 s | p95 651 s | p99 920 s
#
# i.e. a median arrival is ~5.5 min and the long tail reaches ~15 min. Note this is much
# longer than the naive "25 km at 120 m/s = 3.5 min" estimate the first draft used — real
# arrivals are vectored (downwind legs, base turns, the occasional hold), so the flown path
# is far longer than the straight-line distance to the ring. Sizing off the straight-line
# guess made `full` mode cover barely half an approach.
DEFAULT_DT_S = 2.0
DEFAULT_PRED_LEN_FULL = 300
DEFAULT_PRED_LEN_WINDOW = 30

# Lookback. 60 steps x 2 s = 120 s of observed track — long enough to contain a vectoring
# turn rather than just the straight segment before it. Raising it costs anchors twice
# over: fewer per flight, AND whole short flights dropped (p5 is only 235 s).
DEFAULT_SEQ_LEN = 60

# Every remaining approach is mapped onto the same normalized progress domain [0, 1].
# For state output, N is the number of equal-progress future endpoints. For control output,
# N is the number of learned non-uniform piecewise-constant control segments; deployment
# samples the resulting dynamics densely rather than treating their endpoints as the path.
# It is deliberately independent of ``dt_s``. Held-out state-output
# The frozen state-output search selected N=16 for iTransformer and N=256 for PatchTST.
# Model-specific defaults live in one mapping so callers do not reproduce architecture
# branches or silently drift from the selected final-training recipes.
DEFAULT_N_SEGMENTS_BY_MODEL = {
    "itransformer": 16,
    "patchtst": 256,
}
# Public shorthand for the primary model; derived from the mapping so it cannot drift.
DEFAULT_N_SEGMENTS = DEFAULT_N_SEGMENTS_BY_MODEL[MODELS[0]]

# ``final_time_s`` is emitted in physical seconds.  The scale only nondimensionalizes its
# loss; it is not a duration cap and does not change the value returned at inference.
DEFAULT_FINAL_TIME_SCALE_S = 600.0
DEFAULT_POSITION_LOSS_SCALE_M = 10_000.0
DEFAULT_RANDOM_TRAIN_ANCHOR_MIN_FUTURE_S = 60.0
DEFAULT_VALIDATION_COMMON_GRID_POINTS = 64
DEFAULT_CONTROL_HORIZON_CURRICULUM_STAGE_EPOCHS = 10
DEFAULT_CONTROL_DURATION_UNIFORM_FLOOR = 0.8

# Fallback aircraft when a flight dict has no resolvable type or usable performance model.
# Not cosmetic: it sets the target state's Vref and threshold-crossing height — the ENU
# frame and the state the evaluation gates judge — which is why the resolved value is a
# config field (serialised into every checkpoint) and predict defaults to the checkpoint's
# value, not to this constant. Strict OpenAP-direct experiments reject those rows before
# scenario construction and therefore never use this fallback.
DEFAULT_AIRCRAFT_TYPE = "A320"


# The three actuator constants, named once. simple-v1-lag deliberately leaves them open
# (tau_bank is what the CV sweep resolves), and the checkpoint records the resolved values.
TIME_CONSTANT_FIELDS = frozenset(
    (
        "control_thrust_time_constant_s",
        "control_bank_time_constant_s",
        "control_load_time_constant_s",
    )
)


def control_recipe_overrides(name: str) -> dict[str, Any]:
    """Return the frozen field values a named recipe pins, or {} for ``custom``."""
    if name == CONTROL_RECIPE_CUSTOM:
        return {}
    overrides = control_simple_v1_overrides()
    if name == CONTROL_RECIPE_SIMPLE_V1_LAG:
        overrides["control_dynamics_model"] = CONTROL_DYNAMICS_FIRST_ORDER_LAG
    return overrides


def control_simple_v1_overrides() -> dict[str, Any]:
    """Return the frozen scientific definition of the minimal control recipe."""

    return {
        "model": "itransformer",
        "prediction_output": PREDICTION_CONTROL,
        "horizon_mode": HORIZON_NORMALIZED,
        "dt_s": DEFAULT_DT_S,
        "seq_len": DEFAULT_SEQ_LEN,
        "n_segments": 64,
        "channels": CHANNELS,
        "aircraft_type": DEFAULT_AIRCRAFT_TYPE,
        "aircraft_filter": AIRCRAFT_FILTER_OPENAP_DIRECT,
        "coordinate_frame": "enu",
        "reference_velocity_source": REFERENCE_VELOCITY_TRACK_FIT,
        "d_model": 512,
        "n_heads": 8,
        "d_ff": 1024,
        "e_layers": 4,
        "dropout": 0.1,
        "activation": "gelu",
        "use_norm": False,
        "batch_size": 512,
        "epochs": 180,
        "learning_rate": 3e-5,
        "weight_decay": 0.0,
        "lr_plateau_factor": 0.5,
        "lr_plateau_patience": 8,
        "patience": 20,
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "random_train_anchor": False,
        "training_cohort_min_future_s": 0.0,
        "random_train_anchor_min_future_s": DEFAULT_RANDOM_TRAIN_ANCHOR_MIN_FUTURE_S,
        "checkpoint_selection_metric": CHECKPOINT_SELECTION_COMMON_GRID_ADE,
        "validation_common_grid_points": DEFAULT_VALIDATION_COMMON_GRID_POINTS,
        "fitted_tail_position_weight": 0.25,
        "fitted_terminal_position_weight": 1.0,
        "position_loss_scale_m": DEFAULT_POSITION_LOSS_SCALE_M,
        "final_time_scale_s": DEFAULT_FINAL_TIME_SCALE_S,
        "final_time_loss_weight": 1.0,
        "state_endpoint_loss_weight": 0.25,
        "kinematic_consistency_loss_weight": 0.0,
        "terminal_loss_weight": 0.0,
        "control_effort_loss_weight": 0.0,
        "control_smoothness_loss_weight": 0.0,
        "control_duration_parameterization": CONTROL_DURATION_UNIFORM,
        "control_duration_uniform_floor": 0.0,
        "control_dynamics_backend": CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
        "control_dynamics_model": CONTROL_DYNAMICS_POINT_MASS,
        "control_state_supervision_clock": CONTROL_STATE_CLOCK_OBSERVED,
        "control_state_loss_grid": CONTROL_STATE_LOSS_GRID_NATIVE,
        "control_state_objective": CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        "control_velocity_loss_weight": 0.0,
        "control_velocity_loss_scale_mps": 10.0,
        "control_dense_state_loss_weight": 0.0,
        "control_geometry_loss_weight": 0.0,
        "control_arc_horizontal_velocity_loss_weight": 0.0,
        "control_arc_vertical_velocity_loss_weight": 0.0,
        "control_arc_tangent_loss_weight": 0.0,
        "control_terminal_position_loss_weight": 0.0,
        "control_terminal_velocity_loss_weight": 0.0,
        "control_terminal_supervision_clock": CONTROL_TERMINAL_CLOCK_STATE_SUPERVISION,
        "control_state_duration_gradient": False,
        "control_horizon_curriculum_s": (),
        "control_gradient_clip_norm": 20.0,
        "control_gradient_clip_policy": CONTROL_GRADIENT_CLIP_GLOBAL,
        "control_rollout_integrator_dt_s": 0.5,
    }


# Serialized fields a checkpoint MUST carry. Absence is an error rather than a default:
# taking this build's default would silently restate the recipe an artifact was trained
# under. The control list applies only to control-output checkpoints.
REQUIRED_SERIALIZED_FIELDS = ("channels", "reference_velocity_source")
REQUIRED_SERIALIZED_CONTROL_FIELDS = (
    "control_duration_parameterization",
    "control_duration_uniform_floor",
    "control_state_loss_grid",
    "control_state_objective",
    "control_state_duration_gradient",
    "control_horizon_curriculum_s",
    "control_horizon_curriculum_stage_epochs",
    "control_gradient_clip_norm",
    "control_gradient_clip_policy",
    "control_dynamics_backend",
    "control_dynamics_model",
    "control_thrust_time_constant_s",
    "control_bank_time_constant_s",
    "control_load_time_constant_s",
    # control_velocity_loss_weight / _scale_mps are deliberately NOT here: their defaults
    # (0.0 / 10.0) reproduce the behaviour of every checkpoint trained before the term
    # existed, which is exactly the "safe stand-in" test this list applies.
    "control_dense_state_loss_weight",
    "control_geometry_loss_weight",
    "control_arc_horizontal_velocity_loss_weight",
    "control_arc_vertical_velocity_loss_weight",
    "control_arc_horizontal_velocity_scale_mps",
    "control_arc_vertical_velocity_scale_mps",
    "control_arc_local_velocity_parameterization",
    "control_arc_tangent_loss_weight",
    "control_arc_position_end_weight",
    "control_arc_terminal_parameterization",
    "control_arc_terminal_cross_track_emphasis",
    "control_arc_terminal_vertical_emphasis",
    "control_terminal_position_loss_weight",
    "control_terminal_velocity_loss_weight",
    "control_terminal_position_scale_m",
    "control_terminal_velocity_scale_mps",
    "control_terminal_supervision_clock",
)


@dataclass(frozen=True)
class TSConfig:
    """Everything that defines a run. Serialised whole into each checkpoint."""

    # ── what to train ────────────────────────────────────────────────────────
    model: str = MODELS[0]
    prediction_output: str = PREDICTION_STATE
    # Named recipes freeze one complete scientific contract. ``custom`` preserves every
    # historical experiment mode and remains the default for existing callers/checkpoints.
    control_recipe_name: str = CONTROL_RECIPE_CUSTOM
    horizon_mode: str = HORIZON_NORMALIZED
    # ── the time grid + windowing (read by the data build AND both models) ──
    dt_s: float = DEFAULT_DT_S
    seq_len: int = DEFAULT_SEQ_LEN                  # L
    # None resolves once, during construction, through DEFAULT_N_SEGMENTS_BY_MODEL. After
    # __post_init__ every serialized/runtime config carries a concrete integer.
    n_segments: int | None = None                    # N normalized progress segments
    # The two physical-time modes keep their own horizon sizes. ``full`` emits H_full
    # fixed-dt states in one pass; ``window`` emits H_window states per pass and chains
    # passes up to H_full at inference. The vendored networks only consume ``pred_len``.
    full_horizon_steps: int = DEFAULT_PRED_LEN_FULL
    window_horizon_steps: int = DEFAULT_PRED_LEN_WINDOW
    channels: tuple[str, ...] = CHANNELS
    # The aircraft-type fallback the series were built with (target Vref / TCH -> the ENU
    # frame and the gate target). In the config so the checkpoint records it and predict
    # rebuilds series with the SAME frames the normalizer stats were fit under.
    aircraft_type: str = DEFAULT_AIRCRAFT_TYPE
    # Data-selection contract. ``openap-direct`` means: resolve identity to an ICAO Doc
    # 8643 designator, then retain it only when OpenAP has a native model under that exact
    # designator. OpenAP synonyms, presets and the fallback above are excluded.
    aircraft_filter: str = AIRCRAFT_FILTER_ALL
    # ``runway-aligned`` rotates the horizontal plane so every threshold course points
    # along the first axis. It keeps the six-channel tensor shape while removing a major
    # source of cross-airport orientation variance.
    coordinate_frame: str = "enu"
    # Velocity-state supervision may retain the upstream centred track fit or be rebuilt
    # causally from the uniform chart positions.  This changes both model inputs and
    # measured velocity targets, so it is an explicit checkpoint recipe field.
    reference_velocity_source: str = REFERENCE_VELOCITY_TRACK_FIT

    # ── architecture, shared by both models ─────────────────────────────────
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 512
    e_layers: int = 3 #number of encoder layers; why 3? 
    dropout: float = 0.1
    activation: str = "gelu"

    # ── iTransformer only ───────────────────────────────────────────────────
    # use_norm / PatchTST's revin below are the per-window instance normalisations, ON
    # upstream and OFF here. Both exist to strip a window's absolute level as nuisance and
    # keep its shape as signal. In a threshold-anchored ENU frame that is backwards:
    # absolute position IS the signal — it determines where the turn onto final happens,
    # when the descent starts, and where the approach ends. Measured cost of leaving them
    # on (synthetic KRDU, window mode): iTransformer ADE 288 -> 743 m, PatchTST 698 -> 910 m.
    # Re-ablate on real data with --instance-norm; see README "Instance normalisation".
    use_norm: bool = False
    output_attention: bool = False
    # Read by the vendored code but inert on this path — kept so the object stays a drop-in
    # for upstream's run.py. See vendor/itransformer/PROVENANCE.md "Config contract".
    embed: str = "timeF" # unused, ignored by iTransformer
    freq: str = "h"
    factor: int = 1
    class_strategy: str = "projection"

    # ── PatchTST only ───────────────────────────────────────────────────────
    patch_len: int = 16
    stride: int = 8
    padding_patch: str = "end"
    revin: bool = False             # per-window instance norm (RevIN); see use_norm above
    affine: bool = False
    subtract_last: bool = False
    decomposition: bool = False
    kernel_size: int = 25
    individual: bool = False
    fc_dropout: float = 0.1
    head_dropout: float = 0.0

    # ── training ────────────────────────────────────────────────────────────
    batch_size: int = 2048
    # Full pooled CUDA validation reached its minimum at epoch 161 and early-stopped at
    # 181. A 180-epoch cap contains the useful region; the retained checkpoint is still
    # the best epoch, not necessarily the last one.
    epochs: int = 180
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 3
    patience: int = 20              # early-stopping patience, in epochs without val improvement
    seed: int = 1337
    # ``seed`` controls model initialisation and epoch shuffling.  Leave this unset to
    # preserve the historical behaviour where the same seed also assigns outer splits;
    # set it explicitly when repeating an experiment with different training seeds so
    # outer-train/validation/test identities remain locked.
    split_seed: int | None = None
    device: str = "auto"            # "auto" -> cuda when available, else cpu
    val_fraction: float = 0.15      # split is BY FLIGHT, never by window — see dataset.py
    test_fraction: float = 0.15
    # One full-trajectory example per flight is the default: observe L samples, then predict
    # from anchor L-1 to the runway. Rolling/replanning experiments opt into later anchors.
    random_train_anchor: bool = False
    # Optional common train-roster floor used by controlled fixed-vs-random comparisons.
    # It is applied only after the by-flight split, so validation membership stays intact.
    training_cohort_min_future_s: float = 0.0
    # Random anchors with only a few seconds of future create a nearly constant normalized
    # target and do not represent the fixed-anchor deployment task. This train-only floor is
    # frozen before validation; fixed-anchor train/validation windows do not use it.
    random_train_anchor_min_future_s: float = DEFAULT_RANDOM_TRAIN_ANCHOR_MIN_FUTURE_S
    # One formal development score: fixed-anchor, common true-physical-time, airport-macro
    # 3D ADE.  It is shared by CV, the LR scheduler, early stopping and checkpointing.
    checkpoint_selection_metric: str = CHECKPOINT_SELECTION_COMMON_GRID_ADE
    validation_common_grid_points: int = DEFAULT_VALIDATION_COMMON_GRID_POINTS
    # Inferred final-approach geometry is weaker supervision than an observed ADS-B row.
    # These weights apply to POSITION channels only; fitted velocity channels are always
    # masked.  The terminal weight is added on the fitted crossing row so the endpoint is
    # not diluted by the rest of the short extrapolated tail.
    fitted_tail_position_weight: float = 0.25
    fitted_terminal_position_weight: float = 1.0
    final_time_loss_weight: float = 1.0
    # One explicit output-endpoint task prevents the last physical position from being
    # diluted to 1/N of the whole-path objective. It uses the same physical position scale
    # as the path loss; the 0.25 coefficient is frozen by the development Pareto audit.
    state_endpoint_loss_weight: float = 0.25
    # Control/oracle experiment compatibility knobs. The formal direct-state objective
    # ignores both: it predicts position+duration and derives future velocity from position.
    kinematic_consistency_loss_weight: float = 3.0
    terminal_loss_weight: float = 0.02
    # Nondimensionalizes the direct-state physical 3D position MSE. This is an optimizer
    # scale, not an aviation acceptance threshold; the checkpoint records it explicitly.
    position_loss_scale_m: float = DEFAULT_POSITION_LOSS_SCALE_M
    final_time_scale_s: float = DEFAULT_FINAL_TIME_SCALE_S
    # Control-output-only regularizers. Controls are scaled by each flight's own envelope
    # before these are evaluated, so mixed-aircraft batches share one dimensionless loss.
    control_effort_loss_weight: float = 1e-3
    control_smoothness_loss_weight: float = 1e-2
    # Duration-head ablation for the deterministic single-control strategy. ``factorized``
    # predicts one positive total time plus a softmax partition; ``direct`` predicts each
    # positive segment duration and derives total time by summation. Both emit the same
    # ControlPrediction contract, so rollout/loss/inference remain strategy-agnostic.
    control_duration_parameterization: str = CONTROL_DURATION_FACTORIZED
    # Reserve this fraction of total duration uniformly across control segments; only the
    # remainder is allocated by learned logits. At the 0.8 default no single segment can
    # exceed ``0.2 + 0.8/N`` of the horizon, eliminating the observed ~95% collapse while
    # preserving a learnable non-uniform partition.
    control_duration_uniform_floor: float = DEFAULT_CONTROL_DURATION_UNIFORM_FLOOR
    # Which total duration drives the differentiable rollout used by control state loss.
    # ``predicted`` preserves the original joint geometry/clock training. ``observed`` is
    # an explicit development candidate: controls and duration fractions receive state
    # supervision on the known training clock while the final-time head keeps its own loss.
    # Inference always uses predicted time, regardless of this training-only choice.
    control_state_supervision_clock: str = CONTROL_STATE_CLOCK_PREDICTED
    # State supervision can remain on learned segment endpoints or use every regular
    # reference-grid timestamp.  The fixed-dt strategy is isolated in its own data/loss
    # modules: segment durations still choose control-switch times, while state error is
    # evaluated independently every ``dt_s`` seconds on the observed training clock.
    control_state_loss_grid: str = CONTROL_STATE_LOSS_GRID_NATIVE
    # The default keeps the historical normalized-channel MSE. ``physical-criteria``
    # optimizes the smooth worst of fixed-dt 3-D ADE/100 m and terminal error/100 m.
    # ``terminal-state`` composes independently replaceable dense-state, terminal-position
    # and terminal-velocity terms. ``arc-length-geometry`` replaces only the dense term with
    # position SmoothL1 plus reliable local chart-velocity errors on one normalized
    # horizontal-arc grid. Orthogonal ablation fields below change terminal decomposition,
    # local-velocity decomposition or progress weighting without creating more objectives.
    control_state_objective: str = CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE
    # The true-time-position objective scores POSITION only, so a rollout may thread the
    # right places with the wrong heading and swing back between them — the measured
    # signature of that is a bank profile shared by every flight (see
    # docs/2026-08-19_control_bank_wiggle_diagnosis.zh.md). This term scores the chart
    # velocity at the same endpoints, on the same measured rows the position term uses:
    # supervision weights are already zero on fitted-tail velocities, so the placeholder
    # rows cannot enter. Zero keeps the frozen simple-v1 behaviour.
    control_velocity_loss_weight: float = 0.0
    control_velocity_loss_scale_mps: float = 10.0
    control_dense_state_loss_weight: float = 0.25
    control_geometry_loss_weight: float = 0.75
    control_arc_horizontal_velocity_loss_weight: float = 0.25
    control_arc_vertical_velocity_loss_weight: float = 0.25
    control_arc_horizontal_velocity_scale_mps: float = 10.0
    control_arc_vertical_velocity_scale_mps: float = 2.0
    control_arc_local_velocity_parameterization: str = (
        CONTROL_ARC_LOCAL_VELOCITY_VECTOR
    )
    control_arc_tangent_loss_weight: float = 0.25
    control_arc_position_end_weight: float = 4.0
    control_arc_terminal_parameterization: str = (
        CONTROL_ARC_TERMINAL_RUNWAY_COMPONENTS
    )
    control_arc_terminal_cross_track_emphasis: float = 3.0
    control_arc_terminal_vertical_emphasis: float = 5.0
    control_terminal_position_loss_weight: float = 1.0
    control_terminal_velocity_loss_weight: float = 1.0
    control_terminal_position_scale_m: float = 100.0
    control_terminal_velocity_scale_mps: float = 10.0
    # Terminal state may share the state-supervision rollout or use a second deployable
    # predicted-clock rollout. The latter keeps dense geometry on the observed clock while
    # full-horizon terminal errors train the inference clock.
    control_terminal_supervision_clock: str = (
        CONTROL_TERMINAL_CLOCK_STATE_SUPERVISION
    )
    # Whether state-rollout gradients may update the learned duration partition. Turning
    # this off leaves the final-time loss trainable while controls own geometry fitting.
    control_state_duration_gradient: bool = True
    # Optional physical-time single-shooting curriculum. Numeric stages are trained for
    # ``control_horizon_curriculum_stage_epochs`` each, followed by the full horizon for
    # the remaining epoch budget. Empty preserves the historical full-horizon training.
    control_horizon_curriculum_s: tuple[float, ...] = ()
    control_horizon_curriculum_stage_epochs: int = (
        DEFAULT_CONTROL_HORIZON_CURRICULUM_STAGE_EPOCHS
    )
    # Optional gradient-norm cap for deterministic control training. The default policy
    # applies one global cap. The opt-in ablation leaves only an isolated factorized
    # final-time head outside the combined backbone/control cap, so it requires observed
    # state clock and detached state-duration gradients. Zero keeps historical behavior.
    control_gradient_clip_norm: float = 0.0
    control_gradient_clip_policy: str = CONTROL_GRADIENT_CLIP_GLOBAL
    # Rollout state representation is independent of the model/data coordinate frame.
    # The baseline re-anchors a local ENU RK4 step into geodetic state every sub-step;
    # transport-chart-velocity integrates threshold-chart position plus moving-local-ENU
    # physical velocity with the full WGS84 transport rate.
    control_dynamics_backend: str = CONTROL_DYNAMICS_REANCHORED_RK4
    # Which flight model the rollout integrates. ``first-order-lag`` augments the state
    # with the three actual control values and drives them towards the model's commands;
    # it reduces to ``point-mass`` as the time constants go to zero, and reuses the same
    # force equations, so the two are comparable rather than two separate models.
    control_dynamics_model: str = CONTROL_DYNAMICS_POINT_MASS
    # Actuator/autopilot time constants, in the control contract's order
    # (thrust, bank, load factor). Bank is the slow one: rolling into and out of a
    # vectored turn is what the meeting identified as the discontinuity worth fixing,
    # and tau_bank is the parameter the CV sweep resolves. Thrust and load factor
    # respond closer to instantly at this scale and are held fixed.
    control_thrust_time_constant_s: float = 1.5
    control_bank_time_constant_s: float = 2.0
    control_load_time_constant_s: float = 0.8
    # Must match the high-fidelity replay integration cap. The Torch rollout subdivides every
    # learned non-uniform segment at this interval and is numerically contract-tested against
    # CasadiSimulator, rather than training on a cheaper second dynamics model.
    control_rollout_integrator_dt_s: float = 0.5

    # ── provenance (free-form; recorded in the checkpoint) ──────────────────
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.control_recipe_name not in CONTROL_RECIPE_NAMES:
            raise ValueError(
                f"unknown control_recipe_name {self.control_recipe_name!r}; expected one "
                f"of {CONTROL_RECIPE_NAMES}"
            )
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}; expected one of {MODELS}")
        if self.prediction_output not in PREDICTION_OUTPUTS:
            raise ValueError(
                f"unknown prediction_output {self.prediction_output!r}; "
                f"expected one of {PREDICTION_OUTPUTS}"
            )
        if self.n_segments is None:
            object.__setattr__(
                self, "n_segments", DEFAULT_N_SEGMENTS_BY_MODEL[self.model]
            )
        expected = control_recipe_overrides(self.control_recipe_name)
        if expected:
            mismatches = {
                name: (getattr(self, name), value)
                for name, value in expected.items()
                if getattr(self, name) != value
            }
            if mismatches:
                details = ", ".join(
                    f"{name}={actual!r} (expected {wanted!r})"
                    for name, (actual, wanted) in sorted(mismatches.items())
                )
                raise ValueError(
                    f"{self.control_recipe_name} recipe fields are frozen: {details}"
                )
        if self.horizon_mode not in HORIZON_MODES:
            raise ValueError(
                f"unknown horizon_mode {self.horizon_mode!r}; expected one of {HORIZON_MODES}"
            )
        if (
            uses_control_dynamics(self.prediction_output)
            and self.horizon_mode != HORIZON_NORMALIZED
        ):
            raise ValueError(
                "control output uses learned non-uniform segments and currently requires "
                "horizon_mode='normalized'; state output retains normalized/full/window"
            )
        if self.coordinate_frame not in COORDINATE_FRAMES:
            raise ValueError(
                f"unknown coordinate_frame {self.coordinate_frame!r}; "
                f"expected one of {COORDINATE_FRAMES}"
            )
        if self.reference_velocity_source not in REFERENCE_VELOCITY_SOURCES:
            raise ValueError(
                f"unknown reference_velocity_source {self.reference_velocity_source!r}; "
                f"expected one of {REFERENCE_VELOCITY_SOURCES}"
            )
        if self.aircraft_filter not in AIRCRAFT_FILTERS:
            raise ValueError(
                f"unknown aircraft_filter {self.aircraft_filter!r}; "
                f"expected one of {AIRCRAFT_FILTERS}"
            )
        if self.control_dynamics_backend not in CONTROL_DYNAMICS_BACKENDS:
            raise ValueError(
                f"unknown control_dynamics_backend {self.control_dynamics_backend!r}; "
                f"expected one of {CONTROL_DYNAMICS_BACKENDS}"
            )
        if (
            not uses_control_dynamics(self.prediction_output)
            and self.control_dynamics_backend != CONTROL_DYNAMICS_REANCHORED_RK4
        ):
            raise ValueError(
                "non-default control dynamics backend requires a control prediction output"
            )
        if self.control_dynamics_model not in CONTROL_DYNAMICS_MODELS:
            raise ValueError(
                f"unknown control_dynamics_model {self.control_dynamics_model!r}; "
                f"expected one of {CONTROL_DYNAMICS_MODELS}"
            )
        if (
            not math.isfinite(self.control_velocity_loss_weight)
            or self.control_velocity_loss_weight < 0.0
        ):
            raise ValueError("control_velocity_loss_weight must be finite and non-negative")
        if (
            not math.isfinite(self.control_velocity_loss_scale_mps)
            or self.control_velocity_loss_scale_mps <= 0.0
        ):
            raise ValueError("control_velocity_loss_scale_mps must be finite and positive")
        for name in (
            "control_thrust_time_constant_s",
            "control_bank_time_constant_s",
            "control_load_time_constant_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            # The actuator ODE is integrated by the same explicit RK4 as the rest of the
            # state, and explicit RK4 on y' = -y/tau is only stable for h/tau < 2.785.
            # Below that the rollout does not degrade, it produces NaN, so a swept time
            # constant shorter than the integrator step is refused at construction
            # rather than discovered as a dead training run.
            if (
                self.control_dynamics_model == CONTROL_DYNAMICS_FIRST_ORDER_LAG
                and value < self.control_rollout_integrator_dt_s
            ):
                raise ValueError(
                    f"{name}={value:g}s is shorter than the "
                    f"{self.control_rollout_integrator_dt_s:g}s integrator step; explicit "
                    "RK4 is unstable there"
                )
        if self.control_dynamics_model == CONTROL_DYNAMICS_FIRST_ORDER_LAG:
            if not uses_control_dynamics(self.prediction_output):
                raise ValueError(
                    "the lagged flight model requires prediction_output='control'"
                )
            # The lag is one RK4 over the coupled point-mass/actuator ODE, so it needs a
            # backend that exposes a continuous chart RHS. The re-anchored baseline is a
            # discrete map (it rebuilds a local ENU frame every substep) and has no such
            # RHS to augment.
            if self.control_dynamics_backend == CONTROL_DYNAMICS_REANCHORED_RK4:
                raise ValueError(
                    "the lagged flight model requires a transport-chart dynamics backend"
                )
        if self.control_state_supervision_clock not in CONTROL_STATE_CLOCKS:
            raise ValueError(
                "unknown control_state_supervision_clock "
                f"{self.control_state_supervision_clock!r}; expected one of "
                f"{CONTROL_STATE_CLOCKS}"
            )
        if self.control_terminal_supervision_clock not in CONTROL_TERMINAL_CLOCKS:
            raise ValueError(
                "unknown control_terminal_supervision_clock "
                f"{self.control_terminal_supervision_clock!r}; expected one of "
                f"{CONTROL_TERMINAL_CLOCKS}"
            )
        if self.control_state_loss_grid not in CONTROL_STATE_LOSS_GRIDS:
            raise ValueError(
                "unknown control_state_loss_grid "
                f"{self.control_state_loss_grid!r}; expected one of "
                f"{CONTROL_STATE_LOSS_GRIDS}"
            )
        if self.control_state_objective not in CONTROL_STATE_OBJECTIVES:
            raise ValueError(
                "unknown control_state_objective "
                f"{self.control_state_objective!r}; expected one of "
                f"{CONTROL_STATE_OBJECTIVES}"
            )
        if (
            self.control_arc_terminal_parameterization
            not in CONTROL_ARC_TERMINAL_PARAMETERIZATIONS
        ):
            raise ValueError(
                "unknown control_arc_terminal_parameterization "
                f"{self.control_arc_terminal_parameterization!r}; expected one of "
                f"{CONTROL_ARC_TERMINAL_PARAMETERIZATIONS}"
            )
        if (
            self.control_arc_local_velocity_parameterization
            not in CONTROL_ARC_LOCAL_VELOCITY_PARAMETERIZATIONS
        ):
            raise ValueError(
                "unknown control_arc_local_velocity_parameterization "
                f"{self.control_arc_local_velocity_parameterization!r}; expected one of "
                f"{CONTROL_ARC_LOCAL_VELOCITY_PARAMETERIZATIONS}"
            )
        if self.control_state_loss_grid == CONTROL_STATE_LOSS_GRID_FIXED_DT:
            if self.prediction_output != PREDICTION_CONTROL:
                raise ValueError(
                    "fixed-dt control state loss is supported only by "
                    "prediction_output='control'"
                )
            if self.control_state_supervision_clock != CONTROL_STATE_CLOCK_OBSERVED:
                raise ValueError(
                    "fixed-dt control state loss requires "
                    "control_state_supervision_clock='observed'"
                )
        if self.control_state_objective == CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY:
            if self.prediction_output != PREDICTION_CONTROL:
                raise ValueError(
                    f"{self.control_state_objective} control objective is supported only by "
                    "prediction_output='control'"
                )
            if self.control_state_loss_grid != CONTROL_STATE_LOSS_GRID_FIXED_DT:
                raise ValueError(
                    f"{self.control_state_objective} control objective requires "
                    "control_state_loss_grid='fixed-dt'"
                )
        if self.control_state_objective == CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION:
            if self.prediction_output != PREDICTION_CONTROL:
                raise ValueError(
                    "true-time-position control objective is supported only by "
                    "prediction_output='control'"
                )
            if self.control_state_loss_grid != CONTROL_STATE_LOSS_GRID_NATIVE:
                raise ValueError(
                    "true-time-position control objective requires "
                    "control_state_loss_grid='native-segment-endpoints'"
                )
            if self.control_duration_parameterization != CONTROL_DURATION_UNIFORM:
                raise ValueError(
                    "true-time-position control objective requires uniform control durations"
                )
            if self.control_state_supervision_clock != CONTROL_STATE_CLOCK_OBSERVED:
                raise ValueError(
                    "true-time-position control objective requires observed state supervision"
                )
        if (
            self.control_terminal_supervision_clock
            in (
                CONTROL_TERMINAL_CLOCK_PREDICTED,
                CONTROL_TERMINAL_CLOCK_PREDICTED_DETACHED_TIME,
            )
        ):
            if self.prediction_output != PREDICTION_CONTROL:
                raise ValueError(
                    "predicted terminal supervision clock requires "
                    "prediction_output='control'"
                )
            if self.control_state_supervision_clock != CONTROL_STATE_CLOCK_OBSERVED:
                raise ValueError(
                    "predicted terminal supervision clock requires observed dense-state "
                    "supervision"
                )
            if self.control_state_loss_grid != CONTROL_STATE_LOSS_GRID_FIXED_DT:
                raise ValueError(
                    "predicted terminal supervision clock requires fixed-dt state loss"
                )
            if (
                self.control_state_objective
                != CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY
            ):
                raise ValueError(
                    "predicted terminal supervision clock requires the "
                    "arc-length-geometry objective"
                )
        if (
            self.control_terminal_supervision_clock
            == CONTROL_TERMINAL_CLOCK_PREDICTED_DETACHED_TIME
            and self.control_duration_parameterization != CONTROL_DURATION_FACTORIZED
        ):
            raise ValueError(
                "predicted-detached-time terminal supervision requires factorized "
                "durations"
            )
        if (
            self.control_state_objective == CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY
            and self.checkpoint_selection_metric
            != CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY
        ):
            raise ValueError(
                "arc-length-geometry control objective requires "
                "checkpoint_selection_metric='fixed-anchor-arc-length-geometry'"
            )
        if (
            self.control_state_objective == CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY
            and self.n_segments < 2
        ):
            raise ValueError("arc-length-geometry requires n_segments >= 2")
        if (
            self.checkpoint_selection_metric == CHECKPOINT_SELECTION_ARC_LENGTH_GEOMETRY
            and self.control_state_objective
            != CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY
        ):
            raise ValueError(
                "fixed-anchor-arc-length-geometry checkpoint selection requires the "
                "arc-length-geometry control objective"
            )
        if not self.control_state_duration_gradient:
            if self.prediction_output != PREDICTION_CONTROL:
                raise ValueError(
                    "detached control-state duration gradients are supported only by "
                    "prediction_output='control'"
                )
            if self.control_state_supervision_clock != CONTROL_STATE_CLOCK_OBSERVED:
                raise ValueError(
                    "detached control-state duration gradients require "
                    "control_state_supervision_clock='observed'"
                )
        if self.control_horizon_curriculum_s:
            if self.prediction_output != PREDICTION_CONTROL:
                raise ValueError(
                    "control horizon curriculum is supported only by "
                    "prediction_output='control'"
                )
            if self.control_state_loss_grid != CONTROL_STATE_LOSS_GRID_FIXED_DT:
                raise ValueError(
                    "control horizon curriculum requires "
                    "control_state_loss_grid='fixed-dt'"
                )
            if (
                self.control_state_objective
                != CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY
            ):
                raise ValueError(
                    "control horizon curriculum requires a fixed-dt physical objective"
                )
            if self.control_state_duration_gradient:
                raise ValueError(
                    "control horizon curriculum requires detached state-duration gradients"
                )
            if self.control_duration_parameterization != CONTROL_DURATION_FACTORIZED:
                raise ValueError(
                    "control horizon curriculum currently requires factorized durations"
                )
            if self.random_train_anchor:
                raise ValueError("control horizon curriculum requires fixed train anchors")
            previous_horizon = 0.0
            for horizon_s in self.control_horizon_curriculum_s:
                if (
                    not isinstance(horizon_s, (int, float))
                    or not math.isfinite(horizon_s)
                    or not 0.0 < horizon_s
                ):
                    raise ValueError("control curriculum horizons must be positive seconds")
                if horizon_s <= previous_horizon:
                    raise ValueError(
                        "control curriculum horizons must be strictly increasing"
                    )
                grid_steps = round(horizon_s / self.dt_s)
                if abs(horizon_s - grid_steps * self.dt_s) > self.dt_s * 1e-7:
                    raise ValueError(
                        "control curriculum horizons must align with the fixed-dt grid"
                    )
                previous_horizon = float(horizon_s)
            if self.epochs <= (
                len(self.control_horizon_curriculum_s)
                * self.control_horizon_curriculum_stage_epochs
            ):
                raise ValueError(
                    "control horizon curriculum must leave at least one full-horizon epoch"
                )
        if (
            not math.isfinite(self.control_gradient_clip_norm)
            or self.control_gradient_clip_norm < 0.0
        ):
            raise ValueError("control_gradient_clip_norm must be finite and non-negative")
        if self.control_gradient_clip_policy not in CONTROL_GRADIENT_CLIP_POLICIES:
            raise ValueError(
                "unknown control_gradient_clip_policy "
                f"{self.control_gradient_clip_policy!r}; expected one of "
                f"{CONTROL_GRADIENT_CLIP_POLICIES}"
            )
        if (
            self.control_gradient_clip_norm > 0.0
            and self.prediction_output != PREDICTION_CONTROL
        ):
            raise ValueError(
                "control gradient clipping is supported only by "
                "prediction_output='control'"
            )
        if (
            self.control_gradient_clip_policy != CONTROL_GRADIENT_CLIP_GLOBAL
            and self.control_gradient_clip_norm <= 0.0
        ):
            raise ValueError(
                "non-global control gradient clip policy requires a positive clip norm"
            )
        if self.control_duration_parameterization not in CONTROL_DURATION_PARAMETERIZATIONS:
            raise ValueError(
                "unknown control_duration_parameterization "
                f"{self.control_duration_parameterization!r}; expected one of "
                f"{CONTROL_DURATION_PARAMETERIZATIONS}"
            )
        if self.control_gradient_clip_policy == CONTROL_GRADIENT_CLIP_FINAL_TIME_DECOUPLED:
            if self.control_duration_parameterization != CONTROL_DURATION_FACTORIZED:
                raise ValueError(
                    "final-time-decoupled clipping requires factorized durations"
                )
            if self.control_state_supervision_clock != CONTROL_STATE_CLOCK_OBSERVED:
                raise ValueError(
                    "final-time-decoupled clipping requires observed state clock"
                )
            if self.control_state_duration_gradient:
                raise ValueError(
                    "final-time-decoupled clipping requires detached state-duration gradients"
                )
        if (
            self.control_duration_parameterization == CONTROL_DURATION_UNIFORM
            and self.prediction_output != PREDICTION_CONTROL
        ):
            raise ValueError(
                "uniform control durations are supported only by prediction_output='control'"
            )
        if self.checkpoint_selection_metric not in CHECKPOINT_SELECTION_METRICS:
            raise ValueError(
                f"unknown checkpoint_selection_metric "
                f"{self.checkpoint_selection_metric!r}; expected one of "
                f"{CHECKPOINT_SELECTION_METRICS}"
            )
        for name in (
            "seq_len",
            "n_segments",
            "full_horizon_steps",
            "window_horizon_steps",
            "d_model",
            "n_heads",
            "d_ff",
            "e_layers",
            "patch_len",
            "stride",
            "kernel_size",
            "batch_size",
            "epochs",
            "lr_plateau_patience",
            "patience",
            "validation_common_grid_points",
            "control_horizon_curriculum_stage_epochs",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        if self.validation_common_grid_points <= 1:
            raise ValueError("validation_common_grid_points must be greater than one")
        for name in (
            "dt_s",
            "learning_rate",
            "position_loss_scale_m",
            "final_time_scale_s",
            "control_rollout_integrator_dt_s",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        if self.random_train_anchor_min_future_s < 0.0:
            raise ValueError("random_train_anchor_min_future_s must be non-negative")
        if self.training_cohort_min_future_s < 0.0:
            raise ValueError("training_cohort_min_future_s must be non-negative")
        if not 0.0 < self.lr_plateau_factor < 1.0:
            raise ValueError(
                "lr_plateau_factor must be between 0 and 1, got "
                f"{self.lr_plateau_factor!r}"
            )
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model={self.d_model} must divide evenly by n_heads={self.n_heads}"
            )
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError(
                f"val_fraction + test_fraction must leave a training split "
                f"(got {self.val_fraction} + {self.test_fraction})"
            )
        if self.fitted_tail_position_weight < 0.0:
            raise ValueError("fitted_tail_position_weight must be non-negative")
        if self.fitted_terminal_position_weight < 0.0:
            raise ValueError("fitted_terminal_position_weight must be non-negative")
        if self.final_time_loss_weight < 0.0:
            raise ValueError("final_time_loss_weight must be non-negative")
        if self.state_endpoint_loss_weight < 0.0:
            raise ValueError("state_endpoint_loss_weight must be non-negative")
        if self.kinematic_consistency_loss_weight < 0.0:
            raise ValueError("kinematic_consistency_loss_weight must be non-negative")
        if self.terminal_loss_weight < 0.0:
            raise ValueError("terminal_loss_weight must be non-negative")
        if self.control_effort_loss_weight < 0.0:
            raise ValueError("control_effort_loss_weight must be non-negative")
        if self.control_smoothness_loss_weight < 0.0:
            raise ValueError("control_smoothness_loss_weight must be non-negative")
        if not 0.0 <= self.control_duration_uniform_floor < 1.0:
            raise ValueError("control_duration_uniform_floor must be in [0, 1)")
        for name, value in (
            ("control_dense_state_loss_weight", self.control_dense_state_loss_weight),
            ("control_geometry_loss_weight", self.control_geometry_loss_weight),
            (
                "control_arc_horizontal_velocity_loss_weight",
                self.control_arc_horizontal_velocity_loss_weight,
            ),
            (
                "control_arc_vertical_velocity_loss_weight",
                self.control_arc_vertical_velocity_loss_weight,
            ),
            ("control_arc_tangent_loss_weight", self.control_arc_tangent_loss_weight),
            (
                "control_terminal_position_loss_weight",
                self.control_terminal_position_loss_weight,
            ),
            (
                "control_terminal_velocity_loss_weight",
                self.control_terminal_velocity_loss_weight,
            ),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name, value in (
            (
                "control_arc_horizontal_velocity_scale_mps",
                self.control_arc_horizontal_velocity_scale_mps,
            ),
            (
                "control_arc_vertical_velocity_scale_mps",
                self.control_arc_vertical_velocity_scale_mps,
            ),
            (
                "control_terminal_position_scale_m",
                self.control_terminal_position_scale_m,
            ),
            (
                "control_terminal_velocity_scale_mps",
                self.control_terminal_velocity_scale_mps,
            ),
            (
                "control_arc_terminal_cross_track_emphasis",
                self.control_arc_terminal_cross_track_emphasis,
            ),
            (
                "control_arc_terminal_vertical_emphasis",
                self.control_arc_terminal_vertical_emphasis,
            ),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.control_arc_position_end_weight)
            or self.control_arc_position_end_weight < 1.0
        ):
            raise ValueError("control_arc_position_end_weight must be finite and >= 1")
        if self.control_state_objective == CONTROL_STATE_OBJECTIVE_ARC_LENGTH_GEOMETRY:
            if (
                self.control_terminal_position_loss_weight
                <= self.control_geometry_loss_weight
            ):
                raise ValueError(
                    f"{self.control_state_objective} requires terminal position weight "
                    "greater than "
                    "geometry weight"
                )
            if (
                self.control_terminal_velocity_loss_weight
                <= self.control_geometry_loss_weight
            ):
                raise ValueError(
                    f"{self.control_state_objective} requires terminal velocity weight "
                    "greater than "
                    "geometry weight"
                )
            if self.control_terminal_velocity_loss_weight <= max(
                self.control_arc_horizontal_velocity_loss_weight,
                self.control_arc_vertical_velocity_loss_weight,
                self.control_arc_tangent_loss_weight,
            ):
                raise ValueError(
                    "arc-length-geometry requires terminal velocity weight greater "
                    "than local velocity weights"
                )

    # PatchTST reads configs.enc_in; iTransformer infers the count from the tensor.
    @property
    def enc_in(self) -> int:
        return len(self.channels)

    @property
    def pred_len(self) -> int:
        """Vendored model output length under the selected horizon contract."""
        return {
            HORIZON_NORMALIZED: int(self.n_segments),
            HORIZON_FULL: self.full_horizon_steps,
            HORIZON_WINDOW: self.window_horizon_steps,
        }[self.horizon_mode]

    @property
    def horizon_s(self) -> float | None:
        """Physical coverage of fixed-time modes; normalized time has no fixed cap."""
        if self.horizon_mode == HORIZON_NORMALIZED:
            return None
        steps = (
            self.full_horizon_steps
            if self.horizon_mode == HORIZON_FULL
            else self.window_horizon_steps
        )
        return steps * self.dt_s

    @property
    def lookback_s(self) -> float:
        """Wall-clock seconds of observed track the model is shown."""
        return self.seq_len * self.dt_s

    @property
    def control_time_constants_s(self) -> tuple[float, float, float]:
        """The three lag constants in the control contract's order."""
        return (
            self.control_thrust_time_constant_s,
            self.control_bank_time_constant_s,
            self.control_load_time_constant_s,
        )

    @property
    def resolved_split_seed(self) -> int:
        """Seed used only for the locked outer train/validation/test assignment."""
        return self.seed if self.split_seed is None else self.split_seed

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["channels"] = list(self.channels)  # JSON has no tuple
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TSConfig:
        """Rebuild a config, refusing a checkpoint that predates a recipe field.

        A missing key would otherwise take this build's DEFAULT, silently rewriting the
        recipe a trained artifact was produced under. Every field whose default is not a
        safe stand-in for "the old runs did this" is therefore required, not defaulted.
        """
        data = dict(data)
        missing = [name for name in REQUIRED_SERIALIZED_FIELDS if name not in data]
        if uses_control_dynamics(data.get("prediction_output", PREDICTION_STATE)):
            missing += [
                name for name in REQUIRED_SERIALIZED_CONTROL_FIELDS if name not in data
            ]
        if missing:
            raise ValueError(
                f"serialized config is missing {', '.join(sorted(missing))}; "
                "regenerate the derived checkpoint"
            )
        data["channels"] = tuple(data["channels"])
        data["control_horizon_curriculum_s"] = tuple(
            data["control_horizon_curriculum_s"]
        )
        return cls(**data)


def control_recipe(config: TSConfig) -> dict[str, Any]:
    """Serialize the complete recipe for a control-output strategy."""
    base: dict[str, Any] = {
        "reference_velocity_source": config.reference_velocity_source,
        "effort_loss_weight": config.control_effort_loss_weight,
        "smoothness_loss_weight": config.control_smoothness_loss_weight,
        "duration_parameterization": config.control_duration_parameterization,
        "duration_uniform_floor": config.control_duration_uniform_floor,
        "dynamics_backend": config.control_dynamics_backend,
        "dynamics_model": config.control_dynamics_model,
        "time_constants_s": list(config.control_time_constants_s),
        "rollout_integrator_dt_s": config.control_rollout_integrator_dt_s,
        "state_supervision_clock": config.control_state_supervision_clock,
        "state_loss_grid": config.control_state_loss_grid,
        "state_objective": config.control_state_objective,
        "velocity_loss_weight": config.control_velocity_loss_weight,
        "velocity_loss_scale_mps": config.control_velocity_loss_scale_mps,
        "dense_state_loss_weight": config.control_dense_state_loss_weight,
        "geometry_loss_weight": config.control_geometry_loss_weight,
        "arc_horizontal_velocity_loss_weight": (
            config.control_arc_horizontal_velocity_loss_weight
        ),
        "arc_vertical_velocity_loss_weight": (
            config.control_arc_vertical_velocity_loss_weight
        ),
        "arc_horizontal_velocity_scale_mps": (
            config.control_arc_horizontal_velocity_scale_mps
        ),
        "arc_vertical_velocity_scale_mps": (
            config.control_arc_vertical_velocity_scale_mps
        ),
        "arc_local_velocity_parameterization": (
            config.control_arc_local_velocity_parameterization
        ),
        "arc_tangent_loss_weight": config.control_arc_tangent_loss_weight,
        "arc_position_end_weight": config.control_arc_position_end_weight,
        "arc_terminal_parameterization": config.control_arc_terminal_parameterization,
        "arc_terminal_cross_track_emphasis": (
            config.control_arc_terminal_cross_track_emphasis
        ),
        "arc_terminal_vertical_emphasis": (
            config.control_arc_terminal_vertical_emphasis
        ),
        "terminal_position_loss_weight": config.control_terminal_position_loss_weight,
        "terminal_velocity_loss_weight": config.control_terminal_velocity_loss_weight,
        "terminal_position_scale_m": config.control_terminal_position_scale_m,
        "terminal_velocity_scale_mps": config.control_terminal_velocity_scale_mps,
        "terminal_supervision_clock": config.control_terminal_supervision_clock,
        "state_duration_gradient": config.control_state_duration_gradient,
        "horizon_curriculum_s": list(config.control_horizon_curriculum_s),
        "horizon_curriculum_stage_epochs": (
            config.control_horizon_curriculum_stage_epochs
        ),
        "gradient_clip_norm": config.control_gradient_clip_norm,
        "gradient_clip_policy": config.control_gradient_clip_policy,
    }
    if config.control_recipe_name != CONTROL_RECIPE_CUSTOM:
        base["name"] = config.control_recipe_name
    if not uses_control_dynamics(config.prediction_output):
        raise ValueError("state output has no control recipe")
    return base
