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
from coordinate_frames import (
    COORDINATE_FRAME_ENU, COORDINATE_FRAME_RUNWAY_ALIGNED, COORDINATE_FRAMES,
)
from target_conditioning import (
    TARGET_CONDITIONING_CHANNELS,
    TARGET_CONDITIONING_NONE,
    TARGET_CONDITIONINGS,
    conditioning_channel_names,
)
from reference_velocity import (
    REFERENCE_VELOCITY_SOURCES,
    REFERENCE_VELOCITY_TRACK_FIT,
)

MODELS = ("itransformer", "patchtst")
# How the state output's POSITION channels are read. ``absolute``: the network emits the
# chart position itself (state-v1). ``anchor-relative``: it emits the displacement from
# the anchor (the last observed row), which is added back in normalized space — so
# "start where the aircraft is" is the zero output rather than something the network
# must reconstruct from a 120 s history. Measured motivation: the absolute form put every
# KRDU forecast ~250 m NW of the aircraft from its first step
# (docs/2026-09-03_krdu_nw_endpoint_bias.md).
STATE_POSITION_ABSOLUTE = "absolute"
STATE_POSITION_ANCHOR_RELATIVE = "anchor-relative"
# The absolute output, with the position channels bounded to the final-approach corridor
# and glidepath window on the rows the output itself places on the final
# (final_approach_geometry): a hard constraint by construction, no weight to calibrate.
STATE_POSITION_CORRIDOR_BOUNDED = "corridor-bounded"
STATE_POSITION_REFERENCES = (
    STATE_POSITION_ABSOLUTE, STATE_POSITION_ANCHOR_RELATIVE, STATE_POSITION_CORRIDOR_BOUNDED,
)
# Which rows the corridor binds. ``on-final``: rows inside the full-scale cone and aligned
# with the course, read from the prediction itself (deployable). ``faf``: every row inside
# the coded FAF distance — the optimizer's convention and the ablation the measured join
# distances argue against (docs/2026-09-04_procedure_constraints_design.zh.md).
CORRIDOR_GATE_ON_FINAL = "on-final"
CORRIDOR_GATE_FAF = "faf"
CORRIDOR_GATES = (CORRIDOR_GATE_ON_FINAL, CORRIDOR_GATE_FAF)
# The scene / join-anchor design's Phase 0 upper bound (intent_conditioning.py): the
# TRUTH join point (``truth-join``) and, with it, the lead aircraft's TRUE landing time
# (``truth-join-lead``) as input-only constant channels after the target conditioning.
# Read from the future — a development measurement of what inferring the intent could
# be worth, never a deployable predictor. The channel names live here beside
# ``input_channels`` because the geometry module that computes them imports this one.
INTENT_CONDITIONING_NONE = "none"
INTENT_CONDITIONING_TRUTH_JOIN = "truth-join"
INTENT_CONDITIONING_TRUTH_JOIN_LEAD = "truth-join-lead"
# ``truth-join-duration``: the join point plus the flight's TRUE remaining time from the
# anchor — the duration head's own target handed to the input. The ceiling of a decision
# made of (where to join, when to land) with this decoder; measured after the join-only
# arms showed the residual is along-path timing.
INTENT_CONDITIONING_TRUTH_JOIN_DURATION = "truth-join-duration"
INTENT_CONDITIONINGS = (
    INTENT_CONDITIONING_NONE,
    INTENT_CONDITIONING_TRUTH_JOIN,
    INTENT_CONDITIONING_TRUTH_JOIN_LEAD,
    INTENT_CONDITIONING_TRUTH_JOIN_DURATION,
)
# The fields a named recipe leaves OPEN for the intent axis (the CLI's override check).
INTENT_FIELDS = ("intent_conditioning",)
# Order is load-bearing like channels.CHANNELS: serialised into every checkpoint
# (``input_channels``) and ``train.load_checkpoint`` refuses a mismatch.
INTENT_JOIN_CHANNELS: tuple[str, ...] = ("e_join", "n_join", "u_join")
INTENT_LEAD_CHANNELS: tuple[str, ...] = ("lead_eta",)
INTENT_DURATION_CHANNELS: tuple[str, ...] = ("remaining_time",)


def intent_channel_names(intent_conditioning: str) -> tuple[str, ...]:
    """The input-only channels an intent mode appends after the target conditioning."""
    if intent_conditioning == INTENT_CONDITIONING_NONE:
        return ()
    if intent_conditioning == INTENT_CONDITIONING_TRUTH_JOIN:
        return INTENT_JOIN_CHANNELS
    if intent_conditioning == INTENT_CONDITIONING_TRUTH_JOIN_LEAD:
        return INTENT_JOIN_CHANNELS + INTENT_LEAD_CHANNELS
    if intent_conditioning == INTENT_CONDITIONING_TRUTH_JOIN_DURATION:
        return INTENT_JOIN_CHANNELS + INTENT_DURATION_CHANNELS
    raise ValueError(
        f"unknown intent_conditioning {intent_conditioning!r}; expected one of "
        f"{INTENT_CONDITIONINGS}"
    )


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
# The production recipe as of 2026-08-20: simple-v1-lag plus the chart-velocity term that
# the bank-wiggle investigation settled on. Measured against simple-v1-lag on the same
# 1083 KSJC validation flights, it takes the flight-independent share of the predicted
# bank from 70.7 % to 17.3 % (flown tracks: 3.2 %), the bank on straight-in references
# from 3.65 to 0.79 deg (0.55), per-flight bank skill from -0.073 to +0.197, AND improves
# ADE on 77.8 % of flights (median -58.2 m, p=4.7e-79). Everything is frozen here,
# including the three time constants — a recipe that leaves a field open does not name one
# configuration. See docs/2026-08-19_control_bank_wiggle_diagnosis.zh.md.
CONTROL_RECIPE_SIMPLE_V2 = "simple-v2"
# simple-v2 plus direct supervision of the control schedule against the one
# control_inverse_dynamics reads off the flown track. simple-v2 scored position (order 0)
# and velocity (order 1) only, so bank -- an order-2 quantity -- was never named by the
# loss, and unsupervised it landed BELOW a trivial baseline: on KRDU the predicted bank
# carried less information about the flown bank than a randomly chosen other flight's did
# (per-flight skill 0.124 against a random-flight floor of 0.170). On 1404 KRDU validation
# flights this recipe takes that skill to 0.735, the flight-independent share of the bank
# from 49.0 % to 3.3 % (KRDU's own flown tracks: 1.8 %), the bank on straight-in references
# from 3.92 to 0.36 deg (0.41), sign reversals there from 5 to 0, AND improves ADE on 57.0 % of
# flights (median 656 -> 501 m, p=1.9e-7) with FDE unchanged. Unlike the velocity term,
# whose doses bought bank structure at 18-50 % of FDE, this one costs no accuracy.
# See docs/2026-08-19_control_bank_wiggle_diagnosis.zh.md section 12.
CONTROL_RECIPE_SIMPLE_V3 = "simple-v3"
CONTROL_RECIPE_NAMES = (
    CONTROL_RECIPE_CUSTOM,
    CONTROL_RECIPE_SIMPLE_V1,
    CONTROL_RECIPE_SIMPLE_V1_LAG,
    CONTROL_RECIPE_SIMPLE_V2,
    CONTROL_RECIPE_SIMPLE_V3,
)

# The velocity weight simple-v2 pins. Calibrated, not chosen: at the converged simple-v1
# operating point the raw velocity and position terms differ by 642x, so this puts the
# velocity term at ~2x the position term. The dose curve turns here — 8x and beyond keep
# reducing the bank (below the flown tracks' own 0.55 deg) while ADE, FDE and final-time
# all degrade, FDE from 818 to 1231 m at 128x.
SIMPLE_V2_VELOCITY_LOSS_WEIGHT = 0.003

# The imitation weight simple-v3 pins, as a weight and not a multiple: at the converged
# KRDU simple-v2 operating point (position term 0.0417, unweighted imitation term 0.0308)
# w = 1.36 is 1x the position term, so 64 is ~47x it. Selected from an eight-point
# geometric ladder, and the two neighbours are why it is this and not the extremes:
#   - Below 11.8x the ladder is a NOISY PLATEAU, not a ramp: the 1.47x arm came out worse
#     than the 0.74x arm on every metric. Sampling only that region would have concluded
#     the term barely works.
#   - At 188x the fit is saturating (unweighted term 0.00891 -> 0.00827, -7 %, against
#     -31 % over the previous step) and its straight-reference bank, 0.24 deg, is 41 % BELOW
#     what the flown tracks themselves fly (0.41) -- smoother than reality rather than
#     closer to it, where 47x sits 12 % below.
# 47x is chosen on that margin plus the saturation and the better FDE mean, NOT on the
# common-profile share, where 188x is marginally closer (3.0 vs 3.3 % against 1.8 %).
# NOTE both airports' flown values differ and neither dose reaches KRDU's 1.8 % share.
#
# **64.0 IS A KRDU-CALIBRATED NUMBER, NOT A UNIVERSAL ONE.** Replicated on KSJC the
# mechanism holds and is if anything stronger (bank skill 0.197 -> 0.678, past that
# airport's 0.543 twin), but the same weight overshoots there: straight-in bank 0.18 deg
# against a flown 0.53, and FDE degrades on 68 % of flights (p=9e-33) where KRDU paid
# nothing. The cause is already measured -- bank carries 18 % of the box-normalised term at
# KSJC against 41 % at KRDU -- so a new airport should recalibrate off that channel split
# rather than inherit 64.0.
SIMPLE_V3_IMITATION_LOSS_WEIGHT = 64.0

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
# The final-approach penalty's fields: an objective on BOTH output paths (the state rows,
# or the control rollout's segment endpoints), added after the control recipes were
# frozen, so a named recipe leaves them OPEN — the CLI accepts them as overrides and
# run_naming lists them as recipe edits. One source for both.
PROCEDURE_LOSS_FIELDS = (
    "procedure_loss_lateral_weight",
    "procedure_loss_vertical_weight",
    "procedure_loss_dual_step",
    "procedure_loss_epsilon",
    "procedure_loss_lateral_scale_m",
    "procedure_loss_vertical_scale_m",
)
# The rollout command hook: a constraint module that rewrites each control segment's
# command from the state at the segment's start (control/dynamics/hooks.py). ``barrier``
# is the per-step safety layer (a barrier on the corridor gives a bank interval the command
# is saturated into); ``nominal-residual`` is a fixed tracking law toward the centreline
# and glidepath with the command as a bounded residual around it
# (docs/2026-09-05_control_constraint_design.zh.md). Both act only where the corridor gate
# says the aircraft is on the final; ``soft`` saturation keeps gradients in the training
# loop, ``hard`` is for inference-only arms.
CONTROL_HOOK_OFF = "off"
CONTROL_HOOK_BARRIER = "barrier"
CONTROL_HOOK_NOMINAL_RESIDUAL = "nominal-residual"
CONTROL_HOOKS = (CONTROL_HOOK_OFF, CONTROL_HOOK_BARRIER, CONTROL_HOOK_NOMINAL_RESIDUAL)
HOOK_SATURATION_SOFT = "soft"
HOOK_SATURATION_HARD = "hard"
HOOK_SATURATIONS = (HOOK_SATURATION_SOFT, HOOK_SATURATION_HARD)
# The hook gates on the rollout state itself; the FAF gate is not carried by the control
# dynamics, so ``on-final`` is the only gate a hook can use.
HOOK_GATES = (CORRIDOR_GATE_ON_FINAL,)
CONTROL_HOOK_FIELDS = (
    "control_command_hook",
    "control_hook_gate",
    "control_hook_saturation",
    "control_barrier_alpha",
    "control_barrier_heading_gain",
    "control_nominal_l1_distance_m",
    "control_nominal_vertical_lookahead_m",
    "control_nominal_vertical_gain",
    "control_nominal_residual_bank_max_rad",
    "control_nominal_residual_load_max",
    "control_nominal_speed_gain",
)
# Tuple-valued fields. JSON (``--config-overrides``, ``from_dict``, a campaign's arm file)
# hands them back as lists; every reader that compares them against recipe content must
# coerce them first, through this one function, or ``[] != ()`` refuses a faithful copy.
SEQUENCE_FIELDS = ("channels", "control_horizon_curriculum_s")


def coerce_sequence_fields(settings: dict[str, Any]) -> dict[str, Any]:
    """Return ``settings`` with every ``SEQUENCE_FIELDS`` entry present as a tuple."""
    return {
        name: tuple(value) if name in SEQUENCE_FIELDS else value
        for name, value in settings.items()
    }


def recipe_settings(name: str, *, keep_name: bool) -> dict[str, Any]:
    """A named recipe's content as a complete override set for a campaign arm.

    ``keep_name=True`` runs under the recipe name (only its OPEN fields — the penalty,
    a lag recipe's time constants — may be overridden; run names read
    ``recipe+(edits)``); ``keep_name=False`` carries ``custom``, for an experiment that
    varies a field the recipe freezes. One helper for every arm runner.
    """
    settings = dict(control_recipe_overrides(name))
    settings["control_recipe_name"] = name if keep_name else CONTROL_RECIPE_CUSTOM
    return settings


def control_recipe_overrides(name: str) -> dict[str, Any]:
    """Return the frozen field values a named recipe pins, or {} for ``custom``."""
    if name == CONTROL_RECIPE_CUSTOM:
        return {}
    overrides = control_simple_v1_overrides()
    if name in (CONTROL_RECIPE_SIMPLE_V1_LAG, CONTROL_RECIPE_SIMPLE_V2,
                CONTROL_RECIPE_SIMPLE_V3):
        overrides["control_dynamics_model"] = CONTROL_DYNAMICS_FIRST_ORDER_LAG
    if name in (CONTROL_RECIPE_SIMPLE_V2, CONTROL_RECIPE_SIMPLE_V3):
        overrides["control_velocity_loss_weight"] = SIMPLE_V2_VELOCITY_LOSS_WEIGHT
        overrides["control_velocity_loss_scale_mps"] = 10.0
        # Frozen, unlike simple-v1-lag: the tau sweep came out unresolved (best-to-worst
        # 5.7 % against 11-23 % fold noise), so 2 s is a defensible default rather than a
        # selected value, and a production recipe must still name one number.
        overrides["control_thrust_time_constant_s"] = 1.5
        overrides["control_bank_time_constant_s"] = 2.0
        overrides["control_load_time_constant_s"] = 0.8
        # d_model stays 512. Widening to 1024 was tried WITH this term and added nothing
        # to the bank metrics (17.6 % vs 17.3 %) while costing ADE on 71.8 % of flights.
    if name == CONTROL_RECIPE_SIMPLE_V3:
        overrides["control_imitation_loss_weight"] = SIMPLE_V3_IMITATION_LOSS_WEIGHT
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
        "control_imitation_loss_weight": 0.0,
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
    # control_velocity_loss_weight / _scale_mps and control_imitation_loss_weight are
    # deliberately NOT here: their defaults (0.0 / 10.0 / 0.0) reproduce the behaviour of
    # every checkpoint trained before those terms existed, which is exactly the "safe
    # stand-in" test this list applies.
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
    # source of cross-airport orientation variance. ``airport-enu`` moves the ANCHOR to
    # the airport reference point: one chart per airport, shared by all its runways, in
    # which the target is an ordinary point rather than the origin (the target-
    # conditioning ablation, docs/2026-09-03_airport_frame_ablation_plan.md).
    coordinate_frame: str = "enu"
    # ``channels`` appends the target's chart position and runway course to the observed
    # history as INPUT-ONLY constant channels (target_conditioning.CONDITIONING_CHANNELS),
    # so a model whose chart no longer puts the target at the origin can still be told
    # which runway it is flying to. The OUTPUT contract stays ``channels``. iTransformer
    # only: a channel-independent backbone cannot route a conditioning token anywhere.
    target_conditioning: str = TARGET_CONDITIONING_NONE
    # The Phase 0 intent upper bound (INTENT_CONDITIONINGS above): the truth join point,
    # optionally with the lead's true landing time, appended after the target
    # conditioning as input-only constant channels. iTransformer only, for the same
    # reason; the lead channel is measured at the fixed anchor, so random train anchors
    # are refused with it.
    intent_conditioning: str = INTENT_CONDITIONING_NONE
    # State output only: position channels as absolute chart coordinates (state-v1), as
    # displacements from the anchor added back in normalized space, or absolute and
    # bounded to the final-approach corridor (see the constants).
    state_position_reference: str = STATE_POSITION_ABSOLUTE
    corridor_gate: str = CORRIDOR_GATE_ON_FINAL
    # State output only: the final-approach penalty (train.procedure_loss). Hinge² on the
    # metres outside the k-cone / glidepath window, on rows where the OBSERVED track is
    # established (final_approach_geometry.truth_final_gate), each family divided by its
    # runway-scale length. Weights are the multipliers λ: fixed when ``dual_step`` is 0,
    # else updated once per epoch, λ ← max(0, λ + dual_step·(violation rate − epsilon)),
    # i.e. the primal-dual recipe with the violation RATE as the constraint. Zero weights
    # and zero step = off (the state-v1 objective).
    procedure_loss_lateral_weight: float = 0.0
    procedure_loss_vertical_weight: float = 0.0
    procedure_loss_dual_step: float = 0.0
    procedure_loss_epsilon: float = 0.02
    procedure_loss_lateral_scale_m: float = 100.0
    procedure_loss_vertical_scale_m: float = 30.0
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
    # Direct supervision of the control schedule against the one inverted from the flown
    # track by control_inverse_dynamics -- the same registry the forward model dispatches
    # through, so the target can never be the solution of different equations. Position and
    # velocity supervision constrain derivative orders 0 and 1; bank lives at order 2 and is
    # otherwise never told what it should be. Measured on KSJC val, an unsupervised bank
    # carries LESS information about the flown bank than a randomly chosen other flight's
    # (per-flight skill +0.197 against +0.312), while a same-runway twin reaches +0.598 --
    # so the signal is there and only supervision was missing. Zero keeps simple-v1/v2.
    control_imitation_loss_weight: float = 0.0
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
    # The rollout command hook (see the CONTROL_HOOK_* constants). Open under every named
    # recipe like the procedure penalty; first-order-lag dynamics and the native state-loss
    # grid only (the hook rides the segmented endpoint rollout).
    control_command_hook: str = CONTROL_HOOK_OFF
    control_hook_gate: str = CORRIDOR_GATE_ON_FINAL
    control_hook_saturation: str = HOOK_SATURATION_SOFT
    # Barrier filter: the barrier's decay rate α (1/s; the allowed closing rate toward a
    # corridor edge is α × the remaining margin) and the heading gain that turns a heading
    # error outside the admissible interval into a turn-rate demand (1/s).
    control_barrier_alpha: float = 0.1
    control_barrier_heading_gain: float = 0.1
    # Nominal law + residual: L1 lateral lookahead, the vertical lookahead and gain of the
    # glidepath law, and the residual bounds around the nominal command.
    control_nominal_l1_distance_m: float = 3000.0
    control_nominal_vertical_lookahead_m: float = 2000.0
    control_nominal_vertical_gain: float = 0.2
    control_nominal_residual_bank_max_rad: float = math.radians(5.0)
    control_nominal_residual_load_max: float = 0.1
    # Thrust coordination of the nominal-law hook: T' = T + k·m·(V_reference − V), the
    # reference being the network's own unhooked rollout (1/k = 10 s).
    control_nominal_speed_gain: float = 0.1
    # Must match the high-fidelity replay integration cap. The Torch rollout subdivides every
    # learned non-uniform segment at this interval and is numerically contract-tested against
    # CasadiSimulator, rather than training on a cheaper second dynamics model.
    control_rollout_integrator_dt_s: float = 0.5

    # ── provenance (free-form; recorded in the checkpoint) ──────────────────
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Sequence fields arrive as lists from JSON; the contract is tuples (see
        # SEQUENCE_FIELDS / coerce_sequence_fields — the CLI's recipe check uses the same).
        for name in SEQUENCE_FIELDS:
            object.__setattr__(self, name, tuple(getattr(self, name)))
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
        if self.state_position_reference not in STATE_POSITION_REFERENCES:
            raise ValueError(
                f"unknown state_position_reference {self.state_position_reference!r}; "
                f"expected one of {STATE_POSITION_REFERENCES}"
            )
        if self.corridor_gate not in CORRIDOR_GATES:
            raise ValueError(
                f"unknown corridor_gate {self.corridor_gate!r}; expected one of {CORRIDOR_GATES}"
            )
        for name in ("procedure_loss_lateral_weight", "procedure_loss_vertical_weight",
                     "procedure_loss_dual_step"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)!r}")
        if not 0.0 <= self.procedure_loss_epsilon < 1.0:
            raise ValueError(
                f"procedure_loss_epsilon is a violation RATE in [0, 1), got "
                f"{self.procedure_loss_epsilon!r}"
            )
        for name in ("procedure_loss_lateral_scale_m", "procedure_loss_vertical_scale_m"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        if self.uses_final_approach_context and self.coordinate_frame != COORDINATE_FRAME_ENU:
            raise ValueError(
                "the final-approach corridor (corridor-bounded output / procedure loss) is "
                "written in the threshold-anchored ENU chart — the target at the origin, "
                f"east/north axes; coordinate_frame={self.coordinate_frame!r} would measure "
                "it from the wrong point or rotate it twice"
            )
        if (
            self.prediction_output != PREDICTION_STATE
            and self.state_position_reference != STATE_POSITION_ABSOLUTE
        ):
            raise ValueError(
                "state_position_reference belongs to the state output; "
                f"prediction_output={self.prediction_output!r} rolls its states out of "
                "controls and has no position channels to reparametrize"
            )
        if self.control_command_hook not in CONTROL_HOOKS:
            raise ValueError(
                f"unknown control_command_hook {self.control_command_hook!r}; expected one "
                f"of {CONTROL_HOOKS}"
            )
        if self.control_hook_saturation not in HOOK_SATURATIONS:
            raise ValueError(
                f"unknown control_hook_saturation {self.control_hook_saturation!r}; "
                f"expected one of {HOOK_SATURATIONS}"
            )
        if self.control_hook_gate not in HOOK_GATES:
            raise ValueError(
                f"unknown control_hook_gate {self.control_hook_gate!r}; a command hook gates "
                f"on the rollout state itself, expected one of {HOOK_GATES}"
            )
        if self.control_command_hook != CONTROL_HOOK_OFF:
            if self.prediction_output != PREDICTION_CONTROL:
                raise ValueError("a control command hook needs the control output")
            if self.control_dynamics_model != CONTROL_DYNAMICS_FIRST_ORDER_LAG:
                raise ValueError(
                    "the command hook is implemented on the first-order-lag dynamics (its "
                    f"state carries the actuators a hook reads); "
                    f"control_dynamics_model={self.control_dynamics_model!r}"
                )
            if self.control_state_loss_grid != CONTROL_STATE_LOSS_GRID_NATIVE:
                raise ValueError("the command hook rides the native segment-endpoint rollout")
            if self.coordinate_frame != COORDINATE_FRAME_ENU:
                raise ValueError("the command hook reads the threshold-anchored ENU chart")
            for name in ("control_barrier_alpha", "control_barrier_heading_gain",
                         "control_nominal_l1_distance_m", "control_nominal_vertical_lookahead_m",
                         "control_nominal_vertical_gain", "control_nominal_residual_bank_max_rad",
                         "control_nominal_residual_load_max", "control_nominal_speed_gain"):
                if getattr(self, name) <= 0.0:
                    raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        if (
            self.prediction_output == PREDICTION_CONTROL
            and self.procedure_loss_active
            and self.control_state_loss_grid != CONTROL_STATE_LOSS_GRID_NATIVE
        ):
            raise ValueError(
                "the procedure penalty on the control path is implemented on the native "
                "segment-endpoint rollout (its aligned targets carry the truth gate); "
                f"control_state_loss_grid={self.control_state_loss_grid!r} is not supported"
            )
        if self.target_conditioning not in TARGET_CONDITIONINGS:
            raise ValueError(
                f"unknown target_conditioning {self.target_conditioning!r}; "
                f"expected one of {TARGET_CONDITIONINGS}"
            )
        if (
            self.target_conditioning == TARGET_CONDITIONING_CHANNELS
            and self.model != "itransformer"
        ):
            raise ValueError(
                f"target_conditioning={TARGET_CONDITIONING_CHANNELS!r} requires the "
                f"itransformer backbone: {self.model!r} is channel-independent, so a "
                "conditioning channel could never reach the state channels"
            )
        if self.intent_conditioning not in INTENT_CONDITIONINGS:
            raise ValueError(
                f"unknown intent_conditioning {self.intent_conditioning!r}; "
                f"expected one of {INTENT_CONDITIONINGS}"
            )
        if self.intent_conditioning != INTENT_CONDITIONING_NONE:
            if self.model != "itransformer":
                raise ValueError(
                    f"intent_conditioning={self.intent_conditioning!r} requires the "
                    f"itransformer backbone: {self.model!r} is channel-independent, so a "
                    "conditioning channel could never reach the state channels"
                )
            if self.coordinate_frame == COORDINATE_FRAME_RUNWAY_ALIGNED:
                raise ValueError(
                    "the truth join point is gated on chart east/north against the world "
                    f"runway course; the {COORDINATE_FRAME_RUNWAY_ALIGNED!r} chart is "
                    "already rotated"
                )
            if (
                self.intent_conditioning in (
                    INTENT_CONDITIONING_TRUTH_JOIN_LEAD,
                    INTENT_CONDITIONING_TRUTH_JOIN_DURATION,
                )
                and self.random_train_anchor
            ):
                raise ValueError(
                    "the lead ETA and remaining-time channels are measured at the flight's "
                    "fixed anchor; random_train_anchor=True moves the anchor per sample"
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
        if (
            not math.isfinite(self.control_imitation_loss_weight)
            or self.control_imitation_loss_weight < 0.0
        ):
            raise ValueError("control_imitation_loss_weight must be finite and non-negative")
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

    @property
    def procedure_loss_active(self) -> bool:
        return (
            self.procedure_loss_lateral_weight > 0.0
            or self.procedure_loss_vertical_weight > 0.0
            or self.procedure_loss_dual_step > 0.0
        )

    @property
    def uses_final_approach_context(self) -> bool:
        """Whether batches carry the per-flight runway course / glidepath / FAF row."""
        return (
            self.state_position_reference == STATE_POSITION_CORRIDOR_BOUNDED
            or self.procedure_loss_active
        )

    @property
    def input_channels(self) -> tuple[str, ...]:
        """What the model SEES: the channel contract plus any input-only conditioning.

        Serialised into every checkpoint beside ``channels``; ``load_checkpoint`` refuses
        a mismatch, the same lock that keeps a renamed state channel from loading.
        """
        return (
            self.channels
            + conditioning_channel_names(self.target_conditioning)
            + intent_channel_names(self.intent_conditioning)
        )

    # The model INPUT width. PatchTST reads configs.enc_in; iTransformer infers the token
    # count from the tensor, but its duration head and the control feature head flatten
    # over exactly this many channels.
    @property
    def enc_in(self) -> int:
        return len(self.input_channels)

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
        "imitation_loss_weight": config.control_imitation_loss_weight,
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
