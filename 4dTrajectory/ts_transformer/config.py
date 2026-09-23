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

from dataclasses import MISSING, asdict, dataclass, field, fields
import json
import math
from typing import Any, Mapping

# Channel order is a hard contract between the data build, the model, and the export.
# It lives in channels.py; imported here so the default cannot drift from it.
from ts_transformer.data.channels import CHANNELS
from ts_transformer.data.coordinate_frames import (
    COORDINATE_FRAME_ENU, COORDINATE_FRAME_RUNWAY_ALIGNED, COORDINATE_FRAMES,
)
from ts_transformer.data.target_conditioning import (
    TARGET_CONDITIONING_CHANNELS,
    TARGET_CONDITIONING_NONE,
    TARGET_CONDITIONINGS,
    conditioning_channel_names,
)
from ts_transformer.data.reference_velocity import (
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
# VETOED (2026-09-03 state-v2 campaign, by that campaign's OWN pre-registered decision
# rule): it did not clear the bar it was registered against, and `corridor-bounded` did.
# The value STAYS only because a current-cohort artifact is stored under it
# (`4dTrajectory/outputs/*/experiments/state_v2_20260903/A_anchor_relative`), whose config
# must keep loading and naming. It cannot be SELECTED: it is absent from
# `STATE_POSITION_REFERENCES_AVAILABLE`, which is the CLI's choices and the boundary check
# `cli.common._refuse_unavailable_selection` applies to `--config-overrides` as well — the
# same mechanism `control_command_hook="nominal-residual"` uses.
STATE_POSITION_ANCHOR_RELATIVE = "anchor-relative"
# The absolute output, with the position channels bounded to the final-approach corridor
# and glidepath window on the rows the output itself places on the final
# (final_approach_geometry): a hard constraint by construction, no weight to calibrate.
STATE_POSITION_CORRIDOR_BOUNDED = "corridor-bounded"
#: What a STORED config may say.
STATE_POSITION_REFERENCES = (
    STATE_POSITION_ABSOLUTE, STATE_POSITION_ANCHOR_RELATIVE, STATE_POSITION_CORRIDOR_BOUNDED,
)
#: What a NEW run may select (the CLI's choices): the vetoed value is not one of them.
STATE_POSITION_REFERENCES_AVAILABLE = (
    STATE_POSITION_ABSOLUTE, STATE_POSITION_CORRIDOR_BOUNDED,
)
# Which rows the corridor binds. ``on-final``: rows inside the full-scale cone and aligned
# with the course, read from the prediction itself (deployable). ``faf``: every row inside
# the coded FAF distance — the optimizer's convention and the ablation the measured join
# distances argue against (docs/2026-09-04_procedure_constraints_design.zh.md).
# The ONE corridor gate (`final_approach_geometry.membership`): inside the membership cone
# AND aligned with the course. The FAF-distance gate (`faf`) was DELETED 2026-09-09 (review
# §5): never set in any stored run, and a FAF-gated projection wrecked vectored flights. The
# `corridor_gate` field went with it (`RETIRED_CONSTANT_FIELDS` below); the name survives as
# the label `predict --project-final` takes and `run_display_name` never spelled it.
CORRIDOR_GATE_ON_FINAL = "on-final"
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

# The latent-intent design's L3: the CONTROLLED time of arrival as a decoder input. Under
# ``given`` the flight's duration IS the CTA (the duration head is bypassed) and the network
# decides only the path that arrives then; training feeds the truth duration, prediction
# feeds truth + ``--cta-offset-s`` (the counterfactual a scheduler asks for). A ``given`` run
# READS THE FUTURE — the run name carries ``cta=given``, its ``final_time_error_s`` is an
# identity check, and it is a delivery-form demonstration, never a prediction result.
CTA_CONDITIONING_OFF = "off"
CTA_CONDITIONING_GIVEN = "given"
# B3 (`docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md` §三 3.3): the CTA
# is the model's OWN duration quantile, so the decoder reads no future. It is a PREDICT-TIME
# label — `predict --cta-from-quantiles` stamps it on the config written beside the records
# so every naming surface says `cta=self-q` — and never a value a training run may select
# (`CTA_CONDITIONINGS_AVAILABLE` below). §六 5: `cta=self-q` and `cta=given` are two ARMS,
# and only the first is quotable as a prediction.
CTA_CONDITIONING_SELF_QUANTILE = "self-q"
CTA_CONDITIONINGS = (
    CTA_CONDITIONING_OFF, CTA_CONDITIONING_GIVEN, CTA_CONDITIONING_SELF_QUANTILE
)
#: What a NEW run may select (`cli.common._NEW_RUN_VOCABULARIES`): `self-q` describes how a
#: prediction directory was decoded, and there is nothing to train under it.
CTA_CONDITIONINGS_AVAILABLE = (CTA_CONDITIONING_OFF, CTA_CONDITIONING_GIVEN)
CTA_FIELDS = ("cta_conditioning",)

# The plan token: one fused decoder input beside the aircraft condition and the CTA. Nothing
# builds one today — the axis is ``off`` alone; every stored control config carries it, and the
# retired token values below are refused by name. The masking share that used to sit beside the
# token (``plan_conditioning_dropout``) is RETIRED at its constant 0 (`RETIRED_CONSTANT_FIELDS`):
# 0.5 taught the head to ignore the token (v2 §10.8).
PLAN_CONDITIONING_OFF = "off"
PLAN_CONDITIONINGS = (PLAN_CONDITIONING_OFF,)
#: The values a stored control config may carry that no longer build a token, each with the
#: archive its builder moved to. Named separately from the vocabulary so the refusal reads as a
#: RETIREMENT with a pointer rather than as a corrupt value: ``truth-next`` / ``waypoints`` are
#: the two-tier v2 tokens (2026-09-18; 21 stored L1 control checkpoints carry one) and
#: ``manoeuvre-code`` the intent-code second layer (2026-09-20, plan v3 §10 audit — the stage A
#: and stage B executors that carry it no longer load).
PLAN_CONDITIONINGS_RETIRED = {
    "truth-next": "archive/two_tier_v2_2026_09/",
    "waypoints": "archive/two_tier_v2_2026_09/",
    "manoeuvre-code": "archive/manoeuvre_codes_2026_09/",
}
PLAN_CONDITIONING_FIELDS = ("plan_conditioning",)

# The duration head (B1, §三 3.1; B1.b, §三 3.1b). ``point`` is the package's original
# scalar ``FinalTimeHead``; ``quantile`` is ``outputs.duration_heads.QuantileFinalTimeHead`` —
# five monotone quantiles of the SAME quantity, trained by the sum of their pinball losses
# in place of the point head's squared error (the loss component keeps the name
# ``final_time``); ``two-head`` carries BOTH — the point head drives the rollout duration
# exactly as under ``point``, and the quantile head emits nothing but the published ETA
# distribution. `B1_point_matched` dissociated the two gains B1 delivered: the PATH gain is
# the duration term's weight (point head) and the ARRIVAL-TIME gain is the quantile head,
# and they do not overlap. ``two-head`` takes both without the quantile head's path cost
# being forced onto the rollout.
DURATION_HEAD_POINT = "point"
DURATION_HEAD_QUANTILE = "quantile"
DURATION_HEAD_TWO_HEAD = "two-head"
DURATION_HEADS = (DURATION_HEAD_POINT, DURATION_HEAD_QUANTILE, DURATION_HEAD_TWO_HEAD)
#: WHICH head exists under each value — the two predicates every consumer asks, written
#: once, so "does this checkpoint publish an interval" and "does a point estimate drive the
#: rollout" cannot drift apart from the table above.
DURATION_HEADS_WITH_QUANTILES = (DURATION_HEAD_QUANTILE, DURATION_HEAD_TWO_HEAD)
DURATION_HEADS_WITH_POINT = (DURATION_HEAD_POINT, DURATION_HEAD_TWO_HEAD)
# They must PARTITION the vocabulary, and this is where that is enforced: every head emits
# a duration, so `ControlFeatureModel.duration` reads "not point-bearing" as "take the
# median quantile". A fourth value forgotten in both tuples would raise inside the forward
# pass instead of at config construction, so it fails at import here instead.
assert set(DURATION_HEADS_WITH_QUANTILES) | set(DURATION_HEADS_WITH_POINT) == set(DURATION_HEADS), (
    "every duration_head must carry a point head, a quantile head, or both"
)
#: The quantile levels, in order. ONE definition: the head, the pinball loss, the record
#: field, `calibration.py` and every readout read this tuple, so "the five quantiles" cannot
#: become two different sets of five. §六 6: these are quantiles of the DURATION, never of
#: the trajectory.
DURATION_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
#: Which of them walks the existing ``final_time_s`` contract (the rollout's duration).
DURATION_MEDIAN_INDEX = DURATION_QUANTILES.index(0.5)
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
# Retired outputs (2026-09-18, `docs/2026-09-18_manoeuvre_token_plan.zh.md` §5): the closure
# arm (scene design P1.c), the plan-and-guidance head (design v5) and the two-tier v2
# segment-plan layer. Their code is under archive/ and their checkpoints no longer load;
# the NAMES stay so a published category or an old run directory still reads and names.
PREDICTION_OUTPUTS_RETIRED = ("closure", "plan", "segment-plan")
# Every output name a published category may carry — what the frontend mirrors
# (`tests/test_frontend_mirrors.py`): the live ones and the retired ones with published CZML.
PREDICTION_OUTPUTS_PUBLISHED = PREDICTION_OUTPUTS + PREDICTION_OUTPUTS_RETIRED
# What a NEW run may select. A value in the stored vocabulary but not here is FROZEN: its
# checkpoints load, predict and publish, and `cli.common._refuse_unavailable_selection`
# refuses it for training. Both live outputs are selectable.
PREDICTION_OUTPUTS_AVAILABLE = PREDICTION_OUTPUTS
# The truth-join oracles were the scene design's Phase 0 instrument; the L4 gate failed and
# the scene encoder is archived (archive/scene_encoder_2026_09/), so no new oracle arm.
INTENT_CONDITIONINGS_AVAILABLE = (INTENT_CONDITIONING_NONE,)
# The threshold-anchored chart is the model's runway knowledge (2026-09-03 frame ablation:
# the airport frame averages across parallel pairs); the two alternatives keep loading.
COORDINATE_FRAMES_AVAILABLE = (COORDINATE_FRAME_ENU,)
CONTROL_STATE_CLOCK_PREDICTED = "predicted"
CONTROL_STATE_CLOCK_OBSERVED = "observed"
CONTROL_STATE_CLOCKS = (
    CONTROL_STATE_CLOCK_PREDICTED,
    CONTROL_STATE_CLOCK_OBSERVED,
)
CONTROL_STATE_LOSS_GRID_NATIVE = "native-segment-endpoints"
CONTROL_STATE_LOSS_GRID_FIXED_DT = "fixed-dt"
CONTROL_STATE_LOSS_GRIDS = (
    CONTROL_STATE_LOSS_GRID_NATIVE,
    CONTROL_STATE_LOSS_GRID_FIXED_DT,
)
# `fixed-dt` is a 2026-08 arm family (26 stored runs): without the imitation term, which is
# not registered on it, it trips the straight-in veto and brings the bank wiggle back. Frozen
# (review §5, 2026-09-09) — the named recipes pin the native grid.
CONTROL_STATE_LOSS_GRIDS_AVAILABLE = (CONTROL_STATE_LOSS_GRID_NATIVE,)
CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE = "normalized-mse"
CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION = "true-time-position"
CONTROL_STATE_OBJECTIVES = (
    CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
)
# What the imitation term imitates (latent-intent design §六 L5.a). ``inverse-dynamics`` is
# the schedule `outputs.control.supervision.reference_control_supervision` inverts out of the flown track, and
# L0 measured what it is worth: flown open-loop it lands 2.5-7.8 km from the truth it was
# read off (N=4 7850 m, N=8 6381, N=16 4095, N=32 2537), so "imitating the teacher
# perfectly" is NOT "flying the truth track". ``fitted`` reads a per-flight control table
# fitted THROUGH the same differentiable rollout (`experiments/control_basis_oracle.py
# --checkpoint`), which reproduces the truth to 88-433 m at the same width — a strictly
# better teacher. The table is width-, anchor- and duration-specific, and it is the DATASET
# that checks all three: this config only carries the path.
CONTROL_IMITATION_TARGET_INVERSE_DYNAMICS = "inverse-dynamics"
CONTROL_IMITATION_TARGET_FITTED = "fitted"
CONTROL_IMITATION_TARGETS = (
    CONTROL_IMITATION_TARGET_INVERSE_DYNAMICS,
    CONTROL_IMITATION_TARGET_FITTED,
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
CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY = (
    "scaled-transport-chart-velocity"
)
# ``transport-chart-velocity`` — the same chart in PHYSICAL coordinates — was the third
# value until 2026-09-07 (package audit T2). It was a measured regression against
# `reanchored-rk4` (2026-07-31), the nondimensional variant replaced it on 2026-08-02, and
# every stored config that carried it is a 2026-07/08 artifact that `from_dict` already
# refuses for other reasons (13 on disk, 0 of them loading). `run_naming` still abbreviates
# it, because those runs' directory names are historical record.
CONTROL_DYNAMICS_BACKENDS = (
    CONTROL_DYNAMICS_REANCHORED_RK4,
    CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
)
# WHICH quantity the longitudinal control is (docs/2026-09-14_specific_force_control_design.md).
# ``thrust-fraction`` commands T / T_max — normalised by the actuator; every run before
# 2026-09-14 trained under it and every named recipe pins it. ``specific-force`` commands
# n_x = (T - D)/W — normalised by the EFFECT: the lag RHS re-solves the thrust at every RK4
# stage, the drag cancels, and the airframe enters only through the thrust clamp (the same
# [-0.2, 1] x T_max range at every state), the stall clamp and the exported thrust. The SAME
# thrust range is not the same dynamical system: with the drag cancelled the speed has no
# drag feedback, so a biased command drifts where the thrust-fraction law settles (design
# §2.1). ``speed-command`` (design §12) commands a target airspeed relative to the anchor's,
# Δv, flown by a first-order speed loop through the specific-force law's thrust: the same
# invariance, plus the restoring force the specific force lacks (``V' = (V₀ + Δv − V)/τ_V``
# wherever the clamp does not bind). Both non-default laws are first-order-lag only: the
# point-mass rows hold newton controls across a segment and have no per-stage thrust.
CONTROL_THRUST_FRACTION = "thrust-fraction"
CONTROL_SPECIFIC_FORCE = "specific-force"
CONTROL_SPEED_COMMAND = "speed-command"
# ``specific-force+path-angle`` (design §14) is the one value that also moves the VERTICAL
# column: the head commands the target path angle γ* and the lag RHS re-solves the load factor
# that flies it, `n = [cos γ + V(γ* − γ)/(g τ_γ)] / cos φ`, clipped to the load box with the
# stall clamp unchanged. It exists because the load-factor column is an open-loop integrator —
# a bias δn grows a height error like ½·g·δn·t² (design §7.5.3, measured: +0.004 of load is
# +92 m at the end) — while a path-angle target does not accumulate that way (+17 m for the
# same instantaneous error, §13.3). Only THIS combination is spelled, and the ``+`` is a
# LOOKUP, never a split, exactly as in CONTROL_HOOK_MEMBERS: the vertical law is
# energy-neutral (Ė = V·n_x, no γ) only under the specific force, and under a thrust fraction
# it would change the induced drag and need the second law the 2026-09-06 nominal hook needed.
# Every non-default law is first-order-lag only: the point-mass rows hold newton controls
# across a segment and have no per-stage re-solve.
CONTROL_SPECIFIC_FORCE_PATH_ANGLE = "specific-force+path-angle"
# Where each value may be used is ONE row of `CONTROL_PARAMETERIZATION_SCOPES` (below the hook
# vocabulary it names), and what it IS — box, law, teacher, identity — one row of
# `outputs/envelope.py`'s contract registry, which asserts its keys equal these.
# HOW the airframe is presented to the control head (`outputs/conditioning.py`; design N4).
# ``raw`` scales each quantity on its own (mass, installed thrust, wing area, the polar) —
# every run before 2026-09-15, pinned by every named recipe. ``ratios`` hands the head the
# dynamics' own groups in place of the thrust and the area: the thrust-to-weight ratio and the
# 1-g stall speed. The two sets carry the same information and have the same width, so an arm
# that moves only this field starts from the same weights and differs only in what it reads.
CONTROL_CONDITION_FEATURES_RAW = "raw"
CONTROL_CONDITION_FEATURES_RATIOS = "ratios"
CONTROL_CONDITION_FEATURES = (CONTROL_CONDITION_FEATURES_RAW, CONTROL_CONDITION_FEATURES_RATIOS)
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
# A1 (`docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`): the SAME
# common-grid ADE, averaged over five anchor sets — L−1 plus `anchor_grid`'s four
# remaining-path bins — instead of L−1 alone. For a random-anchor arm the fixed metric is
# blind to what the arm improves: it scores the one anchor such a model is LEAST
# specialised for, and froze `A0_random_hr8_tv1` at epoch 10 while that arm's geometry was
# better than the fixed arm's at every anchor on the re-anchoring grid.
CHECKPOINT_SELECTION_ANCHOR_GRID_ADE = "anchor-grid-common-grid-ade"
CHECKPOINT_SELECTION_METRICS = (
    CHECKPOINT_SELECTION_OBJECTIVE,
    CHECKPOINT_SELECTION_COMMON_GRID_ADE,
    CHECKPOINT_SELECTION_ANCHOR_GRID_ADE,
)
#: The metrics scored on a deployable common-grid REPLAY rather than on the training
#: objective. They share the lean ADE evaluator and the cached common-grid truth, so every
#: site that asks "does this run need the replay path" reads this tuple rather than one
#: metric name — which is how the anchor-grid metric would otherwise have silently taken
#: the full-diagnostic path and paid for twenty metrics it does not select on.
CHECKPOINT_SELECTION_COMMON_GRID_METRICS = (
    CHECKPOINT_SELECTION_COMMON_GRID_ADE,
    CHECKPOINT_SELECTION_ANCHOR_GRID_ADE,
)

# WHICH number `ReduceLROnPlateau` measures its plateau on. Checkpoint selection is
# `checkpoint_selection_metric` either way — this axis only decides when the learning rate
# is halved.
#
# `selection` is what every run before this axis existed did: the scheduler was stepped with
# the checkpoint-selection value, which is right while the two move together and wrong the
# moment they part. A0.b (`docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`
# §2.4c) measured them parting on a random-anchor arm: the validation OBJECTIVE improved to
# epoch 60 (1.147 -> 0.707) while the selection metric — a dense-grid ADE — stalled after
# epoch 8, so the LR was halved from epoch 20 on and reached 9.4e-7 by epoch 60. The model
# stopped training at the epoch the READOUT stalled, not the epoch the loss did. Every
# fixed-anchor arm improves on both for a hundred epochs, which is why this never showed.
LR_PLATEAU_METRIC_SELECTION = "selection"
LR_PLATEAU_METRIC_OBJECTIVE = "objective"
LR_PLATEAU_METRICS = (LR_PLATEAU_METRIC_SELECTION, LR_PLATEAU_METRIC_OBJECTIVE)

# HOW a random train anchor is drawn from the flight's admissible ones (A0.b, design §2.4c).
# Admissibility is NOT part of this axis: `eligible_random_train_anchors` and the
# `random_train_anchor_min_future_s` contract decide WHICH anchors exist, both policies draw
# from exactly that set, and the training cohort is therefore identical under either.
#
# `uniform` draws over the SAMPLES, i.e. uniformly in TIME. Pooled over flights that
# OVER-WEIGHTS THE NEAR END relative to the anchor population actually stored: every flight
# gets one draw whatever its length, and the aircraft is slow near the runway, so a kilometre
# there holds more samples than a kilometre at 25 km. Measured on the whole KRDU validation
# split (1404 flights, 181,906 admissible anchors, 200 epochs): the draws put 34.3 % under
# 6 km where the population has 25.1 %, and 17.9 % beyond 20 km where the population has
# 32.9 %. `remaining-path-uniform` places the draw uniformly across the flight's OWN
# admissible remaining-path span and takes the nearest admissible anchor — equal weight per
# kilometre rather than per sample — which moves those to 31.0 % and 21.0 %.
#
# NOT a stratum draw. A0.b's first draft drew an `anchor_strata` stratum uniformly and then a
# sample inside it; that moved training TOWARD the runway (review measurement, 700 KRDU val
# flights: mean remaining path 12.2 -> 8.0 km, share >= 20 km 16.9 % -> 4.1 %), because the
# grid cuts the near end into four 2-km strata and leaves the far end ONE open stratum
# spanning 20-123 km. The strata survive only as the histogram the draws are COUNTED in.
RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM = "uniform"
RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM = "remaining-path-uniform"
RANDOM_TRAIN_ANCHOR_SAMPLINGS = (
    RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM,
    RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM,
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
# The two duration-term weights, one per head. Named because the refusals below compare
# against them: a weight that has no term to weigh under the configured head is refused,
# and "non-default" has to mean exactly one number to both the field and the refusal.
DEFAULT_FINAL_TIME_LOSS_WEIGHT = 1.0
DEFAULT_DURATION_QUANTILE_LOSS_WEIGHT = 1.0
DEFAULT_POSITION_LOSS_SCALE_M = 10_000.0
DEFAULT_RANDOM_TRAIN_ANCHOR_MIN_FUTURE_S = 60.0
DEFAULT_VALIDATION_COMMON_GRID_POINTS = 64
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
)
# The rollout command hook: a constraint module that rewrites each control segment's
# command from the state at the segment's start (outputs/dynamics/hooks.py). ``barrier``
# is the per-step safety layer (a barrier on the corridor gives a bank interval the command
# is saturated into). It acts only where the corridor gate says the aircraft is on the
# final; ``soft`` saturation keeps gradients in the training loop, ``hard`` is for
# inference-only arms.
CONTROL_HOOK_OFF = "off"
CONTROL_HOOK_BARRIER = "barrier"
# ``speed-floor`` is the second constraint module (L3.d, 2026-09-08): a floor on the speed
# the rollout may fly, enforced THROUGH the thrust command, never by clipping V after the
# fact. The floor is the stall-margin speed
# ``control_speed_floor_margin x V_stall(n_commanded, mass, rho, Cl_max)``, i.e. exactly the
# ``flyability`` stall criterion read as a speed; it is UNGATED (a stall is a stall
# anywhere, and 77-94 % of the measured stall samples sit at >= 20 km remaining) and it
# leaves bank and load factor alone.
CONTROL_HOOK_SPEED_FLOOR = "speed-floor"
# The one COMBINATION, in the order it is applied: the barrier sets bank and re-coordinates
# the load factor, then the speed floor reads that load factor and sets thrust. The two
# channels are disjoint, so composing them is well defined; no other combination is, and
# this vocabulary is the only place a combination may be spelled.
CONTROL_HOOK_BARRIER_SPEED_FLOOR = "barrier+speed-floor"
# ``trombone`` is the third constraint module (L3.e, 2026-09-08): the delay the remaining
# path cannot absorb AT THE FLOOR SPEED, spent as a lateral extension of the pre-final path
# — a bounded, lag-compensated heading offset held on one side and mirrored back, with the
# load factor coordinated. It is the degree of freedom L3.d found missing: with the CTA
# fixing the total time and the path fixed, the mean speed is fixed and a floor alone has
# nowhere to put a late arrival. It acts ONLY where the predicted path is more than 30° off
# the runway course — strictly inside "the on-final gate is closed", so it can never act on
# the final and never fights the barrier over a state that is already lined up.
CONTROL_HOOK_TROMBONE = "trombone"
# The COMBINATIONS, each in the order it is applied. `barrier+speed-floor`: the barrier sets
# bank and re-coordinates the load factor, then the floor reads that load factor and sets
# thrust — disjoint channels. The two trombone stacks add a module that writes bank and load
# as well, which is well defined for the same reason and one more: the builder confines the
# barrier to the HARD on-final gate whenever a trombone is a member, and the trombone acts
# only where that gate has not opened, so the two never rewrite the same step
# (outputs/constraints/__init__.py; the soft gate alone was not that complement). The
# trombone comes last because that is where the value spells it; its 15° turn cap is what
# keeps the load factor it coordinates from eating the floor's stall margin
# (outputs/constraints/trombone.py). No other combination is registered, and this vocabulary
# is the only place a combination may be spelled.
CONTROL_HOOK_BARRIER_TROMBONE = "barrier+trombone"
CONTROL_HOOK_BARRIER_SPEED_FLOOR_TROMBONE = "barrier+speed-floor+trombone"
# ``nominal-residual`` — a fixed tracking law toward the centreline and glidepath with the
# command as a bounded residual — was NEVER ADOPTED
# (docs/2026-09-06_control_hooks_results.zh.md) and its code is archived
# (archive/nominal_law_hook_2026_09/). The VALUE stays because six 2026-09-06 configs, and
# the checkpoints `load_checkpoint` rebuilds from them, carry it; `build_command_hook`
# refuses to construct it, so it cannot be chosen for new work.
CONTROL_HOOK_NOMINAL_RESIDUAL = "nominal-residual"
#: What a STORED config may say.
CONTROL_HOOKS = (
    CONTROL_HOOK_OFF,
    CONTROL_HOOK_BARRIER,
    CONTROL_HOOK_NOMINAL_RESIDUAL,
    CONTROL_HOOK_SPEED_FLOOR,
    CONTROL_HOOK_BARRIER_SPEED_FLOOR,
    CONTROL_HOOK_BARRIER_TROMBONE,
    CONTROL_HOOK_BARRIER_SPEED_FLOOR_TROMBONE,
)
#: What a NEW run may select. ``off`` is in it — the flag's own choices drop that one,
#: because `--command-hook` exists to turn a hook ON, but a config saying ``off`` is the
#: default and must pass the boundary check in `cli.common`.
#: There is deliberately no SOLO ``trombone``: the hook hands the command back at the final
#: approach course and has nothing to hand it to unless the barrier is in the stack, and a
#: module that moves the aircraft laterally without the adopted corridor layer under it is
#: not an arm anyone should be able to spell.
CONTROL_HOOKS_AVAILABLE = (
    CONTROL_HOOK_OFF,
    CONTROL_HOOK_BARRIER,
    CONTROL_HOOK_SPEED_FLOOR,
    CONTROL_HOOK_BARRIER_SPEED_FLOOR,
    CONTROL_HOOK_BARRIER_TROMBONE,
    CONTROL_HOOK_BARRIER_SPEED_FLOOR_TROMBONE,
)
#: The modules each vocabulary value composes, in application order. One row per value; the
#: ``+`` spelling is a LOOKUP, never a split, so ``speed-floor+barrier`` (the same two
#: modules in the wrong order), ``trombone+barrier``, a bare ``trombone`` and
#: ``barrier+nominal-residual`` are simply not members and are refused by the vocabulary
#: check like any other unknown value.
CONTROL_HOOK_MEMBERS: dict[str, tuple[str, ...]] = {
    CONTROL_HOOK_BARRIER: (CONTROL_HOOK_BARRIER,),
    CONTROL_HOOK_SPEED_FLOOR: (CONTROL_HOOK_SPEED_FLOOR,),
    CONTROL_HOOK_BARRIER_SPEED_FLOOR: (CONTROL_HOOK_BARRIER, CONTROL_HOOK_SPEED_FLOOR),
    CONTROL_HOOK_BARRIER_TROMBONE: (CONTROL_HOOK_BARRIER, CONTROL_HOOK_TROMBONE),
    CONTROL_HOOK_BARRIER_SPEED_FLOOR_TROMBONE: (
        CONTROL_HOOK_BARRIER, CONTROL_HOOK_SPEED_FLOOR, CONTROL_HOOK_TROMBONE
    ),
}
# Fail at import, like every other table in this package: a value added to the selectable
# vocabulary without a members row would pass construction and die inside the rollout.
assert set(CONTROL_HOOK_MEMBERS) == set(CONTROL_HOOKS_AVAILABLE) - {CONTROL_HOOK_OFF}, (
    "CONTROL_HOOK_MEMBERS must name the modules of every selectable hook"
)
#: The modules that READ ``control_speed_floor_margin``. The floor holds the margin; the
#: trombone divides by the same ``V_floor`` to size the detour the delay needs, so the value
#: changes an answer under either — and a knob that cannot change an answer is refused.
CONTROL_SPEED_FLOOR_MARGIN_READERS = (CONTROL_HOOK_SPEED_FLOOR, CONTROL_HOOK_TROMBONE)



@dataclass(frozen=True)
class ControlParameterizationScope:
    """Where one ``control_thrust_parameterization`` value may be used — the config layer's
    half of the contract (config cannot import ``outputs``; the box, the law, the teacher and
    the checkpoint identity are the value's row in ``outputs/envelope.py``, whose registry
    asserts its keys equal this table's). `TSConfig` reads the row; it never compares the value.
    """

    #: The value's word in run names and slugs; empty for the default, which is never spelled.
    slug: str
    #: Whether the point-mass rows may fly it. They convert the schedule to newtons once per
    #: segment and hold the thrust, so a law that re-solves its thrust (or its load) at every
    #: RK4 stage exists on the first-order-lag model only.
    point_mass: bool
    #: Whether ``control_imitation_target=fitted`` may teach it: the fitted-teacher table's
    #: schema names no parameterisation, so another contract would imitate δ as its own.
    fitted_teacher: bool
    #: The command-hook modules built for this contract's columns.
    hook_modules: tuple[str, ...]
    #: Why the other modules are not, quoted in the refusal.
    hook_note: str = ""


#: Every hook module a stored config may name, including the retired ``nominal-residual``
#: (its own single module), which only thrust-fraction configs ever carried.
_THRUST_FRACTION_HOOK_MODULES = (
    CONTROL_HOOK_BARRIER, CONTROL_HOOK_SPEED_FLOOR, CONTROL_HOOK_TROMBONE,
    CONTROL_HOOK_NOMINAL_RESIDUAL,
)
CONTROL_PARAMETERIZATION_SCOPES: dict[str, ControlParameterizationScope] = {
    CONTROL_THRUST_FRACTION: ControlParameterizationScope(
        slug="", point_mass=True, fitted_teacher=True,
        hook_modules=_THRUST_FRACTION_HOOK_MODULES,
    ),
    CONTROL_SPECIFIC_FORCE: ControlParameterizationScope(
        slug="sf", point_mass=False, fitted_teacher=False,
        hook_modules=(CONTROL_HOOK_BARRIER, CONTROL_HOOK_SPEED_FLOOR, CONTROL_HOOK_TROMBONE),
        hook_note="the retired nominal-residual hook was only ever built for thrust-fraction",
    ),
    CONTROL_SPEED_COMMAND: ControlParameterizationScope(
        slug="sc", point_mass=False, fitted_teacher=False,
        hook_modules=(CONTROL_HOOK_BARRIER, CONTROL_HOOK_TROMBONE),
        # design §12.4
        hook_note="the speed floor writes the thrust channel by inverting a thrust or "
        "specific-force law, and under the speed loop its demand is a speed",
    ),
    CONTROL_SPECIFIC_FORCE_PATH_ANGLE: ControlParameterizationScope(
        slug="sfpa", point_mass=False, fitted_teacher=False,
        # Every module reads the load through the law (`outputs/constraints/vertical.py`, two-tier
        # design §10.8) and the speed floor inverts the specific force.
        hook_modules=(CONTROL_HOOK_BARRIER, CONTROL_HOOK_SPEED_FLOOR, CONTROL_HOOK_TROMBONE),
        hook_note="the retired nominal-residual hook was only ever built for thrust-fraction",
    ),
}
CONTROL_THRUST_PARAMETERIZATIONS = tuple(CONTROL_PARAMETERIZATION_SCOPES)
# WHAT the trombone measures its surplus against (L3.f, 2026-09-09). ``beeline`` is L3.e's
# estimator — the straight-line distance from here to the threshold — and it reads a
# VECTORED flight's long intended path (downwind, base) as surplus time: measured at the
# TRUE CTA, `tromboneDelayS` p50 391 s, endpoint |xt| p95 10 km, 46 % of flights not
# reaching the threshold by T_cta. ``reference-rollout`` measures against the remaining
# path of the HOOK-FREE reference rollout — what the network itself intends to fly — so a
# flight already planning a long detour has nothing to absorb. The default is ``beeline``
# so every L3.e artifact stays reproducible to the bit, records included.
TROMBONE_SURPLUS_BEELINE = "beeline"
TROMBONE_SURPLUS_REFERENCE_ROLLOUT = "reference-rollout"
TROMBONE_SURPLUS_REFERENCES = (
    TROMBONE_SURPLUS_BEELINE, TROMBONE_SURPLUS_REFERENCE_ROLLOUT
)
HOOK_SATURATION_SOFT = "soft"
HOOK_SATURATION_HARD = "hard"
HOOK_SATURATIONS = (HOOK_SATURATION_SOFT, HOOK_SATURATION_HARD)

# Fields removed from the contract after checkpoints that store them were written
# (2026-09-07 package audit). There are TWO kinds, and conflating them is how a stored
# value that DID change a run gets dropped without a word.
#
# `RETIRED_SERIALIZED_FIELDS` is the unread kind: the code that read the field is gone or
# never existed, so whatever a stored config says, it cannot have changed the run that
# produced the artifact. `from_dict` drops these silently. `control_hook_gate` had a
# one-member vocabulary nothing read and `control_dense_state_loss_weight` was a weight no
# loss read. `control_effort_loss_weight` / `control_smoothness_loss_weight` are weaker:
# every named recipe pins them to 0.0 and no arm file since 2026-08 set them, but the
# 2026-07-29 POOLED sweeps (`stage_c_effort`, `stage_c_smoothness`) DID — those ten
# first-generation runs lose the `custom(effort=…)` / `custom(smooth=…)` item from their
# recomputed name.
#: The intent-code plan token's fields (archived 2026-09-20, `archive/manoeuvre_codes_2026_09/`).
#: Under `plan_conditioning='off'` — the only value a config reaching the sweep can carry, the
#: retired plan values being refused by name before it — the tokenizer was never built and
#: nothing read these; every stage A checkpoint stores them at their defaults (`'learned'`,
#: `[]`, `''`) beside `off`. Named apart so the canary test can pick an artifact of their era.
PLAN_TOKEN_RETIRED_FIELDS = (
    "manoeuvre_tokenizer",
    "manoeuvre_fsq_levels",
    "manoeuvre_codebook",
    "manoeuvre_token_s",
    "manoeuvre_token_step_s",
)
RETIRED_SERIALIZED_FIELDS = PLAN_TOKEN_RETIRED_FIELDS + (
    "control_hook_gate",
    "control_dense_state_loss_weight",
    "control_effort_loss_weight",
    "control_smoothness_loss_weight",
    # The horizon curriculum (T1-10, 2026-09-07). Every recipe pinned it to ``()`` and no
    # arm ever set it, so dropping the two fields cannot change any stored run.
    "control_horizon_curriculum_s",
    "control_horizon_curriculum_stage_epochs",
    # The arc-length-geometry objective family and the dual terminal clock (T1-11 / T1-13,
    # 2026-09-07). No arm file has used the objective since 2026-08-02; its only setters
    # were three teacher runners whose checkpoints `load_checkpoint` already refuses (the
    # 2026-08-18 control-unit change). The 2026-08 runs that DID set these fields therefore
    # lose the corresponding items from their recomputed names, which is the grammar's
    # documented behaviour, not a silent rewrite.
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
    # The gradient-clip POLICY axis went with `final-time-decoupled`: one remaining member
    # is not a choice, it is the behaviour. `control_gradient_clip_norm` stays.
    "control_gradient_clip_policy",
    # The nominal-law hook's six gains (T2, 2026-09-07). Its code is archived
    # (archive/nominal_law_hook_2026_09/) and nothing else reads them — the P1.d closure
    # tracker that also used them went in T1-9. Measured on disk: not one stored config
    # sets any of the six away from its default, so no recomputed run name moves. The
    # `control_command_hook="nominal-residual"` VALUE is deliberately NOT retired; six
    # 2026-09-06 configs and their checkpoints would stop loading.
    "control_nominal_l1_distance_m",
    "control_nominal_vertical_lookahead_m",
    "control_nominal_vertical_gain",
    "control_nominal_residual_bank_max_rad",
    "control_nominal_residual_load_max",
    "control_nominal_speed_gain",
)

#: The metres the corridor and glidepath hinges are read in — the UNIT the squared
#: violation is expressed in, never a dose (`procedure_loss_lateral_weight` /
#: `procedure_loss_vertical_weight` and the dual step are the doses). Read by
#: `objective.procedure_loss`.
PROCEDURE_LATERAL_SCALE_M = 100.0
PROCEDURE_VERTICAL_SCALE_M = 30.0
#: The retired closure output's timing scale (its code is archived): kept ONLY as the value
#: `RETIRED_CONSTANT_FIELDS` drops from a stored state/control config written while it existed.
CLOSURE_TIMING_SCALE_S = 60.0

# The MEASURED-CONSTANT kind of retirement (T3-21, 2026-09-07). These three were live loss
# denominators: a stored value other than the constant WOULD have changed the run, so
# dropping it silently would rewrite history. They were retired because nothing ever moved
# them — measured over every stored config under 4dTrajectory/outputs/*/experiments/**:
# 143, 143 and 81 artifacts carry them, none at anything but the constant above.
# `from_dict` therefore drops each only when it EQUALS the constant, and refuses the config
# otherwise, naming the key and the value.
RETIRED_CONSTANT_FIELDS: dict[str, Any] = {
    # The corridor gate: 67 stored configs carry `on-final`, none ever carried `faf`, and the
    # FAF gate is deleted (2026-09-09) — so the field is a constant and is dropped on load.
    "corridor_gate": CORRIDOR_GATE_ON_FINAL,
    "procedure_loss_lateral_scale_m": PROCEDURE_LATERAL_SCALE_M,
    "procedure_loss_vertical_scale_m": PROCEDURE_VERTICAL_SCALE_M,
    "closure_timing_scale_s": CLOSURE_TIMING_SCALE_S,
    # The plan token's training-time masking share (two-tier v2, 2026-09-16 … 09-18): 0.5
    # taught the head to ignore its token (v2 §10.8), so the field is a constant. The stored
    # 0.5 configs are the archived L1 / L1b arms, whose plan value is refused anyway; every
    # other stored control config carries 0.0.
    "plan_conditioning_dropout": 0.0,
}

# The RETIRED-OUTPUT kind (2026-09-18, `docs/2026-09-18_manoeuvre_token_plan.zh.md` §5). These
# fields belonged to the closure, plan and segment-plan views, which left the contract with
# their outputs (`PREDICTION_OUTPUTS_RETIRED`, code under archive/). A state or control config
# written while they existed carries each at the default below — the ownership rule
# (`_OUTPUT_OWNED_FIELDS`) refused a foreign view's field anywhere else — so nothing read it
# there and dropping it cannot change that run. A config that MOVED one was produced by the
# retired output ITSELF, and `__post_init__` refuses that `prediction_output` with the reason;
# a live-output config with a moved value is named rather than dropped, the same rule
# `RETIRED_CONSTANT_FIELDS` follows. Measured 2026-09-18 over all 289 stored checkpoints under
# 4dTrajectory/outputs/**: 112 carry the closure fields, 70 the plan fields, 18 the
# segment-plan fields; all 48 off-default values sit in a checkpoint of their own retired
# output (2 closure, 40 plan, 6 segment-plan) and NONE in a state or control one.
RETIRED_OUTPUT_FIELDS: dict[str, dict[str, Any]] = {
    "closure": {
        "closure_labels_path": "",
        "closure_slowness_knots": 4,
        "closure_height_knots": 4,
        "closure_geometry_loss_weight": 1.0,
        "closure_timing_loss_weight": 1.0,
        "closure_height_loss_weight": 1.0,
    },
    "plan": {
        "plan_operating_loss_weight": 1.0,
        "plan_instruction_loss_weight": 1.0,
        "plan_rolled_windows_path": "",
        "plan_rolled_share": 0.0,
        "plan_fan_components": 0,
    },
    "segment-plan": {
        "segment_plan_segments": 10,
        # The attention vocabulary went with the layer; "channels" was its default member.
        "segment_plan_attention": "channels",
        "segment_plan_position_loss_weight": 1.0,
        "segment_plan_arrival_loss_weight": 1.0,
    },
}

#: Every field name a stored config may still carry that the contract no longer declares —
#: the three retirement kinds as one set, so a reader that only needs "was this retired?"
#: (the CLI's `--config-overrides` message) cannot learn about two kinds and miss the third.
RETIRED_FIELD_NAMES: frozenset[str] = (
    frozenset(RETIRED_SERIALIZED_FIELDS)
    | frozenset(RETIRED_CONSTANT_FIELDS)
    | frozenset(name for names in RETIRED_OUTPUT_FIELDS.values() for name in names)
)

CONTROL_HOOK_FIELDS = (
    "control_command_hook",
    "control_hook_saturation",
    "control_barrier_alpha",
    "control_barrier_heading_gain",
    "control_speed_floor_margin",
    "trombone_surplus_reference",
)

#: The speed floor's default margin above the stall speed. The COEFFICIENT is the package's
#: existing one, not a new number: the optimizer's NLP velocity floor
#: (``optimization/scenario_optimization._STALL_MARGIN``) and this package's control-anchor
#: eligibility gate (``outputs.control.strategy.CONTROL_ANCHOR_STALL_MARGIN``) are both 1.10.
#: **The SPEED it multiplies is not the same one**, and the difference matters when the
#: three are quoted together: those two use the 1-g stall speed at SEA-LEVEL density (the
#: optimizer's also capped at V_ref), while the hook uses ``V_stall(n_commanded)`` at the
#: LOCAL ISA density and applies no cap — because it exists to defend ``flyability``'s
#: criterion, which is evaluated at each sample's own altitude and inverted load factor. At
#: 8000 ft that is rho/rho0 = 0.79, so the hook's floor is ~12.5 % higher, i.e. an effective
#: margin near 1.24 (times sqrt(n) in a turn). The hook is therefore strictly the TIGHTEST
#: of the three: what it lets through, the optimizer's floor would also have admitted.
#: The three are deliberately NOT one symbol: the other two are frozen policy constants
#: (one of them is spelled into the stored policy string ``airborne-1.10-stall-margin-v1``)
#: while this one is a per-run field a campaign may raise, and aliasing them would let a
#: run's knob rewrite a data policy's identity.
CONTROL_SPEED_FLOOR_MARGIN_DEFAULT = 1.10
#: Both barrier gains share one default; the validation below reads it to refuse a gain set
#: on a hook with no barrier in it, so the number lives once.
CONTROL_BARRIER_GAIN_DEFAULT = 0.1

#: Explicit RK4's stability limit on the negative real axis: on y' = -y/tau the step is
#: stable iff h/tau <= this. The actuator-tau rule in `_validate_control_contract` is
#: exactly this bound, spelled once.
RK4_REAL_AXIS_STABILITY_LIMIT = 2.785
# Tuple-valued fields. JSON (``--config-overrides``, ``from_dict``, a campaign's arm file)
# hands them back as lists; every reader that compares them against recipe content must
# coerce them first, through this one function, or ``[] != ()`` refuses a faithful copy.
SEQUENCE_FIELDS = ("channels",)


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
    """Return the frozen scientific definition of the minimal control recipe.

    Every value is a LITERAL or a recipe-named constant (``SIMPLE_V*``), never a module
    ``DEFAULT_*``: a recipe is what a published paired comparison ("same recipe, one axis")
    was run under, and a default that later moves must not redefine it after the fact. A recipe that no longer matches the defaults is
    the recipe telling the truth, not a bug.
    """

    return {
        "model": "itransformer",
        "prediction_output": PREDICTION_CONTROL,
        "horizon_mode": HORIZON_NORMALIZED,
        "dt_s": 2.0,
        "seq_len": 60,
        "n_segments": 64,
        "channels": ("e", "n", "u", "edot", "ndot", "udot"),
        "aircraft_type": "A320",
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
        "lr_plateau_metric": LR_PLATEAU_METRIC_SELECTION,
        "patience": 20,
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "random_train_anchor": False,
        "training_cohort_min_future_s": 0.0,
        "random_train_anchor_min_future_s": 60.0,
        "random_train_anchor_sampling": RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM,
        # A named recipe is judged at L-1: a common floor (two-tier T0(c)) is a `custom` run.
        # Every stored recipe config carries the default 0, so pinning renames nothing.
        "anchor_floor_index": 0,
        # `random_train_anchor_l1_share` is deliberately NOT pinned beside it, for the same
        # reason as `duration_quantile_loss_weight` below: with `random_train_anchor` pinned
        # False a non-zero share is already refused outright, and a bound that cannot bind
        # reads as though it had.
        "checkpoint_selection_metric": CHECKPOINT_SELECTION_COMMON_GRID_ADE,
        "validation_common_grid_points": 64,
        "fitted_tail_position_weight": 0.25,
        "fitted_terminal_position_weight": 1.0,
        "position_loss_scale_m": 10_000.0,
        "final_time_scale_s": 600.0,
        "final_time_loss_weight": 1.0,
        "state_endpoint_loss_weight": 0.25,
        "kinematic_consistency_loss_weight": 0.0,
        "terminal_loss_weight": 0.0,
        "control_duration_parameterization": CONTROL_DURATION_UNIFORM,
        "control_duration_uniform_floor": 0.0,
        # A named recipe is a published DETERMINISTIC point-estimate arm: like the seven
        # `latent_*` fields, the duration head is pinned at its default here, so a QUANTILE
        # or TWO-HEAD run is `custom` and its name carries `T=q5` / `T=2h` instead of hiding
        # behind a recipe. `duration_quantile_loss_weight` is deliberately NOT pinned beside
        # it: with the head pinned at `point`, a non-default value is already refused
        # outright, and a bound that cannot bind reads as though it had.
        "duration_head": "point",
        "control_dynamics_backend": CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
        "control_dynamics_model": CONTROL_DYNAMICS_POINT_MASS,
        # Every published simple-v* comparison flew the thrust-fraction law; an n_x arm
        # varies a field the recipe freezes, so it is `custom` and its dynamics word says so.
        "control_thrust_parameterization": CONTROL_THRUST_FRACTION,
        # ...and read the airframe as raw quantities; a `ratios` arm is `custom` and wears
        # `airframe=ratios`.
        "control_condition_features": CONTROL_CONDITION_FEATURES_RAW,
        "control_state_supervision_clock": CONTROL_STATE_CLOCK_OBSERVED,
        "control_state_loss_grid": CONTROL_STATE_LOSS_GRID_NATIVE,
        "control_state_objective": CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        "control_velocity_loss_weight": 0.0,
        "control_velocity_loss_scale_mps": 10.0,
        "control_imitation_loss_weight": 0.0,
        # L1.b's two supervision terms are off in every named recipe: they are the
        # candidates measured AGAINST simple-v3's teacher, not part of it.
        "control_heading_rate_loss_weight": 0.0,
        "control_heading_rate_loss_scale_dps": 1.5,
        "control_bank_tv_loss_weight": 0.0,
        # Every published simple-v* comparison was run against the inverse-dynamics
        # teacher; a recipe names one configuration, so the target is frozen here too.
        "control_imitation_target": CONTROL_IMITATION_TARGET_INVERSE_DYNAMICS,
        # The WHOLE latent axis is pinned off: a named recipe is a published DETERMINISTIC
        # comparison arm, so `simple-v*` means "no latent intent" by definition and a latent
        # run is `custom` — which is what every latent arm file has always said. Pinning only
        # L2.f's two levers would have drawn an incoherent line: a `simple-v3` run could then
        # choose its beta but not its warm-up. Measured before adopting (2026-09-07): no
        # stored artifact changes name or slug, and none stops loading.
        "latent_dim": 0,
        "latent_prior_components": 1,
        "latent_beta": 1.0,
        "latent_free_bits_nats": 0.0,
        "latent_posterior_init_std": 1.0,
        "latent_beta_warmup_epochs": 0,
        "latent_aux_duration_weight": 0.0,
        # ...and so is the plan token: a recipe run is told no plan.
        "plan_conditioning": PLAN_CONDITIONING_OFF,
        # ...and the rollout horizon (two-tier L1): a recipe run predicts the whole remaining
        # approach. Every stored recipe config predates the field and reads as 0, so pinning
        # it renames nothing.
        "control_horizon_s": 0.0,
        "control_state_duration_gradient": False,
        "control_gradient_clip_norm": 20.0,
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
    "control_gradient_clip_norm",
    "control_dynamics_backend",
    "control_dynamics_model",
    "control_thrust_time_constant_s",
    "control_bank_time_constant_s",
    "control_load_time_constant_s",
    # `objective.target_contract` keys the stored contract on the clock: a checkpoint
    # missing only this key must fail with the curated message, not as an opaque
    # "target contract does not match" (review C-14).
    "control_state_supervision_clock",
    # control_velocity_loss_weight / _scale_mps and control_imitation_loss_weight are
    # deliberately NOT here: their defaults (0.0 / 10.0 / 0.0) reproduce the behaviour of
    # every checkpoint trained before those terms existed, which is exactly the "safe
    # stand-in" test this list applies.
)


# ── the typed views: who owns which field (package review §4.3, 2026-09-10) ──────────
#
# `TSConfig` below stays the FLAT dataclass every checkpoint serialises and every module
# reads (`config.seq_len`); the dataclasses here are what OWNS each field and its rules.
# `TSConfig.__post_init__` builds one of each from the flat values and exposes them as
# `config.cohort / .backbone / .training / .output`, so a rule that used to be one of some
# forty cross-field `if`s in an 800-line validator now sits inside the object whose fields
# it reads, and a per-output strategy can take `config.output` (typed) instead of the whole
# config. Two things fall out by construction:
#
# - a field OWNED by another output variant is refused when it is off its default
#   (`TSConfig._validate_ownership`) — the one rule that thirteen "belongs to output X" /
#   "supported only by prediction_output='control'" checks used to spell one field at a
#   time, and the class of hole review C-1 / C-2 / C-4 came from;
# - the serialized form does not change: `to_dict` is the flat dict and `from_dict` reads
#   it. A stored non-control run's control fields are unread by definition, so `from_dict`
#   normalises them to their defaults rather than refusing the artifact (measured
#   2026-09-10 on the 66 stored state/closure runs: every one already sits at its default).
#
# A rule that reads TWO views (the chart an output needs, a selection metric that re-reads
# the future an oracle input hands over) stays on `TSConfig._validate_cross`, spelled once.
# The partition is checked at import: every `TSConfig` field belongs to exactly one view
# (`notes` excepted — free text, read by nobody).


def _own(spec: type, config: Any) -> dict[str, Any]:
    """The flat values of ``spec``'s own fields (a nested view is built separately)."""
    return {
        f.name: getattr(config, f.name)
        for f in fields(spec)
        if f.name in _FLAT_FIELD_NAMES
    }


def _require_member(name: str, value: Any, vocabulary: tuple[Any, ...]) -> None:
    if value not in vocabulary:
        raise ValueError(f"unknown {name} {value!r}; expected one of {vocabulary}")


def _require_positive(owner: Any, names: tuple[str, ...]) -> None:
    for name in names:
        if getattr(owner, name) <= 0:
            raise ValueError(f"{name} must be positive, got {getattr(owner, name)!r}")


@dataclass(frozen=True)
class CohortSpec:
    """The data plane: what a flight series is, and which anchors a window may take."""

    dt_s: float
    seq_len: int
    channels: tuple[str, ...]
    aircraft_type: str
    aircraft_filter: str
    coordinate_frame: str
    target_conditioning: str
    intent_conditioning: str
    reference_velocity_source: str
    val_fraction: float
    test_fraction: float
    random_train_anchor: bool
    training_cohort_min_future_s: float
    random_train_anchor_min_future_s: float
    random_train_anchor_sampling: str
    random_train_anchor_l1_share: float
    anchor_floor_index: int

    def __post_init__(self) -> None:
        _require_member("coordinate_frame", self.coordinate_frame, COORDINATE_FRAMES)
        _require_member("target_conditioning", self.target_conditioning, TARGET_CONDITIONINGS)
        _require_member("intent_conditioning", self.intent_conditioning, INTENT_CONDITIONINGS)
        _require_member(
            "reference_velocity_source", self.reference_velocity_source,
            REFERENCE_VELOCITY_SOURCES,
        )
        _require_member("aircraft_filter", self.aircraft_filter, AIRCRAFT_FILTERS)
        _require_member(
            "random_train_anchor_sampling", self.random_train_anchor_sampling,
            RANDOM_TRAIN_ANCHOR_SAMPLINGS,
        )
        _require_positive(self, ("seq_len", "dt_s"))
        if self.random_train_anchor_min_future_s < 0.0:
            raise ValueError("random_train_anchor_min_future_s must be non-negative")
        if self.training_cohort_min_future_s < 0.0:
            raise ValueError("training_cohort_min_future_s must be non-negative")
        if self.anchor_floor_index < 0:
            raise ValueError(f"anchor_floor_index is a sample index, got {self.anchor_floor_index!r}")
        if not 0.0 <= self.random_train_anchor_l1_share <= 1.0:
            raise ValueError(
                "random_train_anchor_l1_share is a share of the per-flight draws and must "
                f"be between 0 and 1, got {self.random_train_anchor_l1_share!r}"
            )
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError(
                f"val_fraction + test_fraction must leave a training split "
                f"(got {self.val_fraction} + {self.test_fraction})"
            )
        if (
            self.random_train_anchor_sampling != RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM
            and not self.random_train_anchor
        ):
            raise ValueError(
                f"random_train_anchor_sampling={self.random_train_anchor_sampling!r} names "
                "HOW a random train anchor is drawn, and random_train_anchor=False draws "
                "none — the fixed policy anchors every flight at L-1"
            )
        # The L-1 share is a coin ON TOP of a draw, so it needs a draw to sit on. Refused
        # rather than ignored in both directions it can be meaningless: the fixed policy
        # already anchors every flight at L-1 (the share would be 1 by construction), and
        # under `uniform` the axis would silently do nothing to a run whose name carries it.
        if self.random_train_anchor_l1_share:
            if not self.random_train_anchor:
                raise ValueError(
                    f"random_train_anchor_l1_share={self.random_train_anchor_l1_share!r} "
                    "reserves a share of the RANDOM anchor draws for L-1, and "
                    "random_train_anchor=False draws none — the fixed policy already "
                    "anchors every flight at L-1"
                )
            if self.random_train_anchor_sampling != RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM:
                raise ValueError(
                    f"random_train_anchor_l1_share={self.random_train_anchor_l1_share!r} "
                    "is defined on top of "
                    f"random_train_anchor_sampling={RANDOM_TRAIN_ANCHOR_SAMPLING_PATH_UNIFORM!r}"
                    f", not {self.random_train_anchor_sampling!r}: the `uniform` law draws "
                    "over the samples, where L-1 is already one of them, and mixing a "
                    "reserved share into it is a second axis nobody has measured"
                )
        if self.intent_conditioning != INTENT_CONDITIONING_NONE:
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

    @property
    def input_channels(self) -> tuple[str, ...]:
        """What the model SEES: the channel contract plus any input-only conditioning."""
        return (
            self.channels
            + conditioning_channel_names(self.target_conditioning)
            + intent_channel_names(self.intent_conditioning)
        )

    @property
    def lookback_s(self) -> float:
        """Wall-clock seconds of observed track the model is shown."""
        return self.seq_len * self.dt_s


@dataclass(frozen=True)
class BackboneSpec:
    """The vendored network and its hyperparameters — what `models.build_model` hands the
    vendored ``configs`` namespace. One dataclass for both backbones: the flat schema
    serialises every knob for either, and which ones a backbone reads is its own business
    (`vendor/*/model.py`)."""

    model: str
    d_model: int
    n_heads: int
    d_ff: int
    e_layers: int
    dropout: float
    activation: str
    use_norm: bool
    output_attention: bool
    embed: str
    freq: str
    factor: int
    class_strategy: str
    patch_len: int
    stride: int
    padding_patch: str
    revin: bool
    affine: bool
    subtract_last: bool
    decomposition: bool
    kernel_size: int
    individual: bool
    fc_dropout: float
    head_dropout: float

    def __post_init__(self) -> None:
        _require_member("model", self.model, MODELS)
        _require_positive(
            self, ("d_model", "n_heads", "d_ff", "e_layers", "patch_len", "stride", "kernel_size")
        )
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model={self.d_model} must divide evenly by n_heads={self.n_heads}"
            )


@dataclass(frozen=True)
class TrainingSpec:
    """The optimiser, its schedule, the seeds, and which epoch is kept."""

    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    lr_plateau_factor: float
    lr_plateau_patience: int
    lr_plateau_metric: str
    patience: int
    seed: int
    split_seed: int | None
    device: str
    checkpoint_selection_metric: str
    validation_common_grid_points: int

    def __post_init__(self) -> None:
        _require_member(
            "checkpoint_selection_metric", self.checkpoint_selection_metric,
            CHECKPOINT_SELECTION_METRICS,
        )
        _require_member("lr_plateau_metric", self.lr_plateau_metric, LR_PLATEAU_METRICS)
        _require_positive(
            self,
            ("batch_size", "epochs", "lr_plateau_patience", "patience",
             "validation_common_grid_points", "learning_rate"),
        )
        if self.validation_common_grid_points <= 1:
            raise ValueError("validation_common_grid_points must be greater than one")
        if not 0.0 < self.lr_plateau_factor < 1.0:
            raise ValueError(
                "lr_plateau_factor must be between 0 and 1, got "
                f"{self.lr_plateau_factor!r}"
            )

    @property
    def resolved_split_seed(self) -> int:
        """Seed used only for the locked outer train/validation/test assignment."""
        return self.seed if self.split_seed is None else self.split_seed


@dataclass(frozen=True)
class OutputSpec:
    """What every prediction path shares: the horizon contract and the loss scales that
    the state objective, the control tracking terms and the procedure penalty all read.
    One subclass per ``prediction_output`` carries that path's own fields."""

    prediction_output: str
    horizon_mode: str
    n_segments: int
    full_horizon_steps: int
    window_horizon_steps: int
    final_time_scale_s: float
    position_loss_scale_m: float
    fitted_tail_position_weight: float
    fitted_terminal_position_weight: float
    final_time_loss_weight: float
    state_endpoint_loss_weight: float
    kinematic_consistency_loss_weight: float
    terminal_loss_weight: float
    procedure_loss_lateral_weight: float
    procedure_loss_vertical_weight: float
    procedure_loss_dual_step: float
    procedure_loss_epsilon: float

    def __post_init__(self) -> None:
        _require_member("horizon_mode", self.horizon_mode, HORIZON_MODES)
        _require_positive(
            self,
            ("n_segments", "full_horizon_steps", "window_horizon_steps",
             "position_loss_scale_m", "final_time_scale_s"),
        )
        for name in (
            "fitted_tail_position_weight", "fitted_terminal_position_weight",
            "final_time_loss_weight", "state_endpoint_loss_weight",
            "kinematic_consistency_loss_weight", "terminal_loss_weight",
            "procedure_loss_lateral_weight", "procedure_loss_vertical_weight",
            "procedure_loss_dual_step",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)!r}")
        if not 0.0 <= self.procedure_loss_epsilon < 1.0:
            raise ValueError(
                f"procedure_loss_epsilon is a violation RATE in [0, 1), got "
                f"{self.procedure_loss_epsilon!r}"
            )

    @property
    def procedure_loss_active(self) -> bool:
        return (
            self.procedure_loss_lateral_weight > 0.0
            or self.procedure_loss_vertical_weight > 0.0
            or self.procedure_loss_dual_step > 0.0
        )

    @property
    def pred_len(self) -> int:
        """Vendored model output length under the selected horizon contract."""
        return {
            HORIZON_NORMALIZED: int(self.n_segments),
            HORIZON_FULL: self.full_horizon_steps,
            HORIZON_WINDOW: self.window_horizon_steps,
        }[self.horizon_mode]


@dataclass(frozen=True)
class StateOutput(OutputSpec):
    """The purely kinematic path: channels in, channels out."""

    state_position_reference: str

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_member(
            "state_position_reference", self.state_position_reference,
            STATE_POSITION_REFERENCES,
        )


@dataclass(frozen=True)
class DurationSpec:
    """How the rollout's total duration is predicted and partitioned into segments."""

    duration_head: str
    duration_quantile_loss_weight: float
    control_duration_parameterization: str
    control_duration_uniform_floor: float

    def __post_init__(self) -> None:
        _require_member("duration_head", self.duration_head, DURATION_HEADS)
        _require_member(
            "control_duration_parameterization", self.control_duration_parameterization,
            CONTROL_DURATION_PARAMETERIZATIONS,
        )
        if self.duration_quantile_loss_weight < 0.0:
            raise ValueError("duration_quantile_loss_weight must be non-negative")
        if not 0.0 <= self.control_duration_uniform_floor < 1.0:
            raise ValueError("control_duration_uniform_floor must be in [0, 1)")
        # B1.b: `point` has no pinball sum to weigh (`two-head` is where both weights bind).
        if (self.duration_head == DURATION_HEAD_POINT
                and self.duration_quantile_loss_weight != DEFAULT_DURATION_QUANTILE_LOSS_WEIGHT):
            raise ValueError(
                f"duration_quantile_loss_weight={self.duration_quantile_loss_weight!r} "
                "weighs the QUANTILE head's pinball losses, and duration_head='point' has "
                f"no such head; select duration_head={DURATION_HEAD_QUANTILE!r} or "
                f"{DURATION_HEAD_TWO_HEAD!r}"
            )


@dataclass(frozen=True)
class DynamicsSpec:
    """Which flight model the rollout integrates, on which chart, and how finely."""

    control_dynamics_model: str
    control_dynamics_backend: str
    control_thrust_parameterization: str
    control_thrust_time_constant_s: float
    control_bank_time_constant_s: float
    control_load_time_constant_s: float
    control_rollout_integrator_dt_s: float

    def __post_init__(self) -> None:
        _require_member(
            "control_dynamics_backend", self.control_dynamics_backend,
            CONTROL_DYNAMICS_BACKENDS,
        )
        _require_member(
            "control_dynamics_model", self.control_dynamics_model, CONTROL_DYNAMICS_MODELS
        )
        _require_member(
            "control_thrust_parameterization", self.control_thrust_parameterization,
            CONTROL_THRUST_PARAMETERIZATIONS,
        )
        if (not CONTROL_PARAMETERIZATION_SCOPES[self.control_thrust_parameterization].point_mass
                and self.control_dynamics_model != CONTROL_DYNAMICS_FIRST_ORDER_LAG):
            # A scope statement, not physics (`ControlParameterizationScope.point_mass`).
            raise ValueError(
                f"control_thrust_parameterization={self.control_thrust_parameterization!r} "
                "is implemented on the first-order-lag flight model only; "
                f"control_dynamics_model={self.control_dynamics_model!r}"
            )
        for name in sorted(TIME_CONSTANT_FIELDS):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        _require_positive(self, ("control_rollout_integrator_dt_s",))
        if self.control_dynamics_model == CONTROL_DYNAMICS_FIRST_ORDER_LAG:
            # The lag is one RK4 over the coupled point-mass/actuator ODE, so it needs a
            # backend that exposes a continuous chart RHS. The re-anchored baseline is a
            # discrete map (it rebuilds a local ENU frame every substep) and has none.
            if self.control_dynamics_backend == CONTROL_DYNAMICS_REANCHORED_RK4:
                raise ValueError(
                    "the lagged flight model requires a transport-chart dynamics backend"
                )
            for name in sorted(TIME_CONSTANT_FIELDS):
                # The actuator ODE is integrated by the same explicit RK4 as the rest of
                # the state, and explicit RK4 on y' = -y/tau is stable only for
                # h/tau <= RK4_REAL_AXIS_STABILITY_LIMIT (the substep h is at most the
                # integrator step). Past it the rollout does not degrade, it produces NaN,
                # so a swept time constant that short is refused at construction rather
                # than discovered as a dead training run (review C-13).
                value = getattr(self, name)
                if self.control_rollout_integrator_dt_s > RK4_REAL_AXIS_STABILITY_LIMIT * value:
                    raise ValueError(
                        f"{name}={value:g}s puts the {self.control_rollout_integrator_dt_s:g}s "
                        f"integrator step at h/tau = "
                        f"{self.control_rollout_integrator_dt_s / value:.2f} > "
                        f"{RK4_REAL_AXIS_STABILITY_LIMIT}; explicit RK4 is unstable there"
                    )

    @property
    def time_constants_s(self) -> tuple[float, float, float]:
        """The three lag constants in the control contract's order."""
        return (
            self.control_thrust_time_constant_s,
            self.control_bank_time_constant_s,
            self.control_load_time_constant_s,
        )


@dataclass(frozen=True)
class ControlObjective:
    """What a control schedule is scored against, and how its gradients are shaped.

    **`control_state_loss_grid` cannot be derived from `control_state_objective`** —
    measured on disk 2026-09-07, not assumed. `true-time-position` does pin the native
    grid, but `normalized-mse` runs on BOTH: of the 233 stored configs that load, 159 are
    (`true-time-position`, native), 62 + 8 are (`normalized-mse`, native) on the state and
    closure outputs, and 4 are (`normalized-mse`, `fixed-dt`) on the control output. Two
    live values under one objective is not a function, so the grid stays a field.
    """

    control_state_objective: str
    control_state_loss_grid: str
    control_state_supervision_clock: str
    control_velocity_loss_weight: float
    control_velocity_loss_scale_mps: float
    control_imitation_loss_weight: float
    control_imitation_target: str
    control_fitted_teacher_path: str
    control_heading_rate_loss_weight: float
    control_heading_rate_loss_scale_dps: float
    control_bank_tv_loss_weight: float
    control_state_duration_gradient: bool
    control_gradient_clip_norm: float

    def __post_init__(self) -> None:
        _require_member(
            "control_state_objective", self.control_state_objective, CONTROL_STATE_OBJECTIVES
        )
        _require_member(
            "control_state_loss_grid", self.control_state_loss_grid, CONTROL_STATE_LOSS_GRIDS
        )
        _require_member(
            "control_state_supervision_clock", self.control_state_supervision_clock,
            CONTROL_STATE_CLOCKS,
        )
        _require_member(
            "control_imitation_target", self.control_imitation_target,
            CONTROL_IMITATION_TARGETS,
        )
        for name in (
            "control_velocity_loss_weight", "control_imitation_loss_weight",
            "control_heading_rate_loss_weight", "control_bank_tv_loss_weight",
            "control_gradient_clip_norm",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("control_velocity_loss_scale_mps", "control_heading_rate_loss_scale_dps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "control_velocity_loss_weight",
            "control_imitation_loss_weight",
            "control_heading_rate_loss_weight",
            "control_bank_tv_loss_weight",
        ):
            # All four terms are built by the true-time-position objective only
            # (`objective.loss_component_names`). Elsewhere the weight would be a number
            # that cannot change an answer — while still NAMING the run and, for the
            # imitation term, solving an inverse-dynamics teacher per sample that nothing
            # reads (review C-1).
            if (
                getattr(self, name)
                and self.control_state_objective != CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION
            ):
                raise ValueError(
                    f"{name} is only built by the "
                    f"{CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION} objective, not "
                    f"{self.control_state_objective!r}"
                )
        if self.control_imitation_target == CONTROL_IMITATION_TARGET_FITTED:
            if not self.control_fitted_teacher_path:
                raise ValueError(
                    "the fitted teacher imitates a per-flight table: set "
                    "control_fitted_teacher_path to the basis_fit.json written by "
                    "run_ts.py control_basis_oracle --checkpoint"
                )
            if not self.control_imitation_loss_weight:
                raise ValueError(
                    "control_imitation_target names a teacher for a term this run switches "
                    "off (control_imitation_loss_weight=0): the table would be loaded and "
                    "validated against the cohort, and never read"
                )
            if self.control_state_objective != CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION:
                # `objective.loss_component_names` registers `imitation` under the
                # true-time-position objective ONLY, so under any other objective the term
                # is not built at all. That objective in turn requires the native grid and
                # UNIFORM durations, so this one check also closes the factorized-duration
                # hole: a table of schedules spread uniformly over the total duration
                # cannot supervise a learned partition.
                raise ValueError(
                    "the imitation term is registered under control_state_objective="
                    f"{CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION!r} only; this run scores "
                    f"{self.control_state_objective!r} and would load the fitted teacher "
                    "and never read it"
                )
        elif self.control_fitted_teacher_path:
            raise ValueError(
                "control_fitted_teacher_path belongs to "
                f"control_imitation_target={CONTROL_IMITATION_TARGET_FITTED!r}; this run "
                f"imitates {self.control_imitation_target!r}"
            )
        if (
            self.control_state_loss_grid == CONTROL_STATE_LOSS_GRID_FIXED_DT
            and self.control_state_supervision_clock != CONTROL_STATE_CLOCK_OBSERVED
        ):
            raise ValueError(
                "fixed-dt control state loss requires "
                "control_state_supervision_clock='observed'"
            )
        if self.control_state_objective == CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION:
            if self.control_state_loss_grid != CONTROL_STATE_LOSS_GRID_NATIVE:
                raise ValueError(
                    "true-time-position control objective requires "
                    "control_state_loss_grid='native-segment-endpoints'"
                )
            if self.control_state_supervision_clock != CONTROL_STATE_CLOCK_OBSERVED:
                raise ValueError(
                    "true-time-position control objective requires observed state supervision"
                )
        if (
            not self.control_state_duration_gradient
            and self.control_state_supervision_clock != CONTROL_STATE_CLOCK_OBSERVED
        ):
            raise ValueError(
                "detached control-state duration gradients require "
                "control_state_supervision_clock='observed'"
            )

    @property
    def uses_fitted_teacher(self) -> bool:
        return self.control_imitation_target == CONTROL_IMITATION_TARGET_FITTED


@dataclass(frozen=True)
class HookSpec:
    """The predict-time command hook and the knobs of the modules it builds."""

    control_command_hook: str
    control_hook_saturation: str
    control_barrier_alpha: float
    control_barrier_heading_gain: float
    control_speed_floor_margin: float
    trombone_surplus_reference: str

    def __post_init__(self) -> None:
        _require_member("control_command_hook", self.control_command_hook, CONTROL_HOOKS)
        _require_member(
            "control_hook_saturation", self.control_hook_saturation, HOOK_SATURATIONS
        )
        _require_member(
            "trombone_surplus_reference", self.trombone_surplus_reference,
            TROMBONE_SURPLUS_REFERENCES,
        )
        # Each module's own knobs, checked against the modules the hook actually BUILDS.
        # `nominal-residual` builds nothing (it is load-only and archived) and is absent
        # from the members table, so its six stored 2026-09-06 configs are left exactly as
        # they are; `off` builds nothing either and IS held to the rule (review C-2).
        hook_modules = CONTROL_HOOK_MEMBERS.get(self.control_command_hook, ())
        builds_nothing = bool(hook_modules) or self.control_command_hook == CONTROL_HOOK_OFF
        if CONTROL_HOOK_BARRIER in hook_modules:
            _require_positive(self, ("control_barrier_alpha", "control_barrier_heading_gain"))
        elif builds_nothing:
            for name in ("control_barrier_alpha", "control_barrier_heading_gain"):
                if getattr(self, name) != CONTROL_BARRIER_GAIN_DEFAULT:
                    raise ValueError(
                        f"{name}={getattr(self, name)!r} needs a command hook that contains "
                        f"{CONTROL_HOOK_BARRIER!r} (control_command_hook="
                        f"{self.control_command_hook!r})"
                    )
        if (
            self.control_command_hook == CONTROL_HOOK_OFF
            and self.control_hook_saturation != HOOK_SATURATION_SOFT
        ):
            raise ValueError(
                f"control_hook_saturation={self.control_hook_saturation!r} softens a hook's "
                "bound and there is no hook (control_command_hook='off')"
            )
        # The margin is read by TWO modules: the floor holds it, and the trombone divides
        # by the same V_floor to size its detour, so it changes an answer under either.
        if any(name in hook_modules for name in CONTROL_SPEED_FLOOR_MARGIN_READERS):
            if not math.isfinite(self.control_speed_floor_margin) or (
                self.control_speed_floor_margin < 1.0
            ):
                raise ValueError(
                    "control_speed_floor_margin is a finite multiple of the stall speed and "
                    "a floor below it is not a floor; got "
                    f"{self.control_speed_floor_margin!r}"
                )
        elif self.control_speed_floor_margin != CONTROL_SPEED_FLOOR_MARGIN_DEFAULT:
            raise ValueError(
                f"control_speed_floor_margin={self.control_speed_floor_margin!r} needs a "
                f"command hook that contains one of {CONTROL_SPEED_FLOOR_MARGIN_READERS} "
                f"(control_command_hook={self.control_command_hook!r})"
            )
        # ...and the estimator axis is the trombone's alone: no other module measures a
        # surplus, so away from its default the value would change no trajectory.
        if (
            CONTROL_HOOK_TROMBONE not in hook_modules
            and self.trombone_surplus_reference != TROMBONE_SURPLUS_BEELINE
        ):
            raise ValueError(
                f"trombone_surplus_reference={self.trombone_surplus_reference!r} needs a "
                f"command hook that contains {CONTROL_HOOK_TROMBONE!r} "
                f"(control_command_hook={self.control_command_hook!r})"
            )

    @property
    def active(self) -> bool:
        return self.control_command_hook != CONTROL_HOOK_OFF


@dataclass(frozen=True)
class LatentSpec:
    """The latent intent z on the control output (L2). Every other knob means nothing at
    ``latent_dim == 0`` and is refused there rather than carried into the run name."""

    latent_dim: int
    latent_prior_components: int
    latent_beta: float
    latent_free_bits_nats: float
    latent_beta_warmup_epochs: int
    latent_aux_duration_weight: float
    latent_posterior_init_std: float

    def __post_init__(self) -> None:
        if self.latent_dim < 0 or self.latent_prior_components < 1:
            raise ValueError("latent_dim must be >= 0 and latent_prior_components >= 1")
        if self.latent_beta < 0.0 or self.latent_free_bits_nats < 0.0:
            raise ValueError("latent_beta and latent_free_bits_nats must be non-negative")
        if self.latent_beta_warmup_epochs < 0:
            raise ValueError("latent_beta_warmup_epochs must be >= 0 (0 = no warm-up)")
        if (
            not math.isfinite(self.latent_aux_duration_weight)
            or self.latent_aux_duration_weight < 0.0
        ):
            raise ValueError("latent_aux_duration_weight must be finite and non-negative")
        if self.latent_posterior_init_std <= 0.0:
            raise ValueError("latent_posterior_init_std must be positive")
        if self.latent_dim == 0 and (
            self.latent_prior_components != 1
            or self.latent_beta != 1.0
            or self.latent_free_bits_nats != 0.0
            or self.latent_posterior_init_std != 1.0
            or self.latent_beta_warmup_epochs != 0
            or self.latent_aux_duration_weight != 0.0
        ):
            raise ValueError(
                "latent_prior_components / latent_beta / latent_free_bits_nats / "
                "latent_posterior_init_std / latent_beta_warmup_epochs / "
                "latent_aux_duration_weight mean nothing "
                "without a latent (latent_dim == 0) and would still rename the run"
            )

    @property
    def active(self) -> bool:
        return self.latent_dim > 0


@dataclass(frozen=True)
class ControlOutput(OutputSpec):
    """Bounded controls rolled through the point-mass twin, with its five sub-axes.

    The axes are not independent: the objective constrains the duration parameterization,
    the command hook constrains the flight model and the grid, the latent constrains the
    duration head. Every rule here names the pair it binds; a rule inside ONE axis lives
    on that axis's own dataclass."""

    control_recipe_name: str
    cta_conditioning: str
    plan_conditioning: str
    control_horizon_s: float
    control_condition_features: str
    duration: DurationSpec
    dynamics: DynamicsSpec
    objective: ControlObjective
    hook: HookSpec
    latent: LatentSpec

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_member("control_recipe_name", self.control_recipe_name, CONTROL_RECIPE_NAMES)
        _require_member("cta_conditioning", self.cta_conditioning, CTA_CONDITIONINGS)
        if self.plan_conditioning in PLAN_CONDITIONINGS_RETIRED:
            raise ValueError(
                f"plan_conditioning={self.plan_conditioning!r} is retired: its token builder is "
                f"under {PLAN_CONDITIONINGS_RETIRED[self.plan_conditioning]} and the checkpoints "
                "that carry it no longer load"
            )
        _require_member("plan_conditioning", self.plan_conditioning, PLAN_CONDITIONINGS)
        if self.plan_conditioning != PLAN_CONDITIONING_OFF and self.latent.active:
            raise ValueError(
                "a plan token beside a latent intent is two answers to one question (what the "
                "flight is going to do); the latent line is its own axis"
            )
        # Two-tier L1 (`docs/2026-09-17_two_tier_plan_v2.zh.md` §3): a FIXED rollout horizon.
        # Under Δ > 0 nothing predicts the duration — the head emits its N segments over
        # exactly Δ seconds and the targets cover [0, Δ] (`dataset.target_horizon_s`) — so
        # every axis that would decide the duration has nothing to decide and is REFUSED
        # rather than left inert: the CTA (it would BE the duration), a quantile head
        # (quantiles of a constant), the point term's weight (an identically-zero residual)
        # and the latent (it reaches the duration through the point head's logit). The
        # imitation teacher is inverted over the whole remainder and is not built over a
        # window, so its weight is refused too.
        if not math.isfinite(self.control_horizon_s) or self.control_horizon_s < 0.0:
            raise ValueError(
                "control_horizon_s is the rollout horizon in seconds (0 = the whole remaining "
                f"approach), got {self.control_horizon_s!r}"
            )
        if self.control_horizon_s:
            if self.cta_conditioning != CTA_CONDITIONING_OFF:
                raise ValueError(
                    f"control_horizon_s={self.control_horizon_s:g} fixes the rollout duration; "
                    f"cta_conditioning={self.cta_conditioning!r} would hand it another one"
                )
            if self.duration.duration_head != DURATION_HEAD_POINT:
                raise ValueError(
                    f"control_horizon_s={self.control_horizon_s:g} predicts no duration; "
                    f"duration_head={self.duration.duration_head!r} would emit quantiles of a "
                    "constant"
                )
            if self.final_time_loss_weight:
                raise ValueError(
                    f"final_time_loss_weight={self.final_time_loss_weight!r} weighs the duration "
                    f"residual, which is identically zero under control_horizon_s="
                    f"{self.control_horizon_s:g}; set it to 0"
                )
            if self.latent.active:
                raise ValueError(
                    "a latent intent reaches the duration through the point head, which a "
                    f"control_horizon_s={self.control_horizon_s:g} run does not build"
                )
            if self.objective.control_imitation_loss_weight:
                raise ValueError(
                    "the imitation teacher is inverted over the whole remaining approach and is "
                    f"not built over a control_horizon_s={self.control_horizon_s:g} window"
                )
        _require_member(
            "control_condition_features", self.control_condition_features,
            CONTROL_CONDITION_FEATURES,
        )
        if self.horizon_mode != HORIZON_NORMALIZED:
            raise ValueError(
                "control output uses learned non-uniform segments and currently requires "
                "horizon_mode='normalized'; state output retains normalized/full/window"
            )
        if self.latent.latent_aux_duration_weight and self.cta_conditioning != CTA_CONDITIONING_OFF:
            raise ValueError(
                "latent_aux_duration_weight teaches z to carry the remaining duration, and "
                f"cta_conditioning={self.cta_conditioning!r} already hands that duration to "
                "the decoder — the auxiliary target would be supervising a known input"
            )
        if self.duration.duration_head in DURATION_HEADS_WITH_QUANTILES and self.latent.active:
            # Not a plumbing limitation. `outputs/control/latent.py` reaches the duration by
            # SHIFTING the head's single unconstrained logit (`latent_duration`), and a
            # cumulative-softplus head has five; shifting all five would move the spread as
            # well as the location. Worse, training decodes a POSTERIOR sample, so the five
            # would be quantiles of p(T | z ~ q(z | this flight's own future)) — an interval
            # conditioned on the answer, which B2 would then calibrate as if it were
            # p(T | history). `two-head` is refused for the same reason: its quantile head
            # is trained under the posterior sample exactly as the single one would be.
            raise ValueError(
                f"duration_head={self.duration.duration_head!r} and latent_dim > 0 are refused "
                "together: z reaches the duration by shifting the point head's single "
                "logit, and under a posterior sample the quantiles would be conditioned "
                "on the flight's own future — not a predictive interval to calibrate"
            )
        # B1.b: under `quantile` the pinball sum REPLACED the point term, so
        # `final_time_loss_weight` has nothing to weigh (every stored `quantile` run
        # carries the default); `two-head` is the one value where both weights bind.
        if (self.duration.duration_head == DURATION_HEAD_QUANTILE
                and self.final_time_loss_weight != DEFAULT_FINAL_TIME_LOSS_WEIGHT):
            raise ValueError(
                f"final_time_loss_weight={self.final_time_loss_weight!r} weighs the POINT "
                "head's squared duration residual, and duration_head='quantile' has no "
                "point term — the pinball sum replaced it under the same component name. "
                "Weigh the pinball with duration_quantile_loss_weight, or take the point "
                f"term back with duration_head={DURATION_HEAD_TWO_HEAD!r}"
            )
        law = self.dynamics.control_thrust_parameterization
        scope = CONTROL_PARAMETERIZATION_SCOPES[law]
        if (not scope.fitted_teacher
                and self.objective.control_imitation_target == CONTROL_IMITATION_TARGET_FITTED):
            raise ValueError(
                f"control_imitation_target={CONTROL_IMITATION_TARGET_FITTED!r} is not "
                f"built for control_thrust_parameterization={law!r}: a fitted-teacher "
                "table does not say which coordinate its schedules are in"
            )
        if self.hook.active:
            hook = self.hook.control_command_hook
            unbuilt = [
                module for module in CONTROL_HOOK_MEMBERS.get(hook, (hook,))
                if module not in scope.hook_modules
            ]
            if unbuilt:
                raise ValueError(
                    f"control_command_hook={hook!r}: {', '.join(map(repr, unbuilt))} is not "
                    f"built for control_thrust_parameterization={law!r} ({scope.hook_note})"
                )
            if self.dynamics.control_dynamics_model != CONTROL_DYNAMICS_FIRST_ORDER_LAG:
                raise ValueError(
                    "the command hook is implemented on the first-order-lag dynamics (its "
                    f"state carries the actuators a hook reads); "
                    f"control_dynamics_model={self.dynamics.control_dynamics_model!r}"
                )
            if self.objective.control_state_loss_grid != CONTROL_STATE_LOSS_GRID_NATIVE:
                raise ValueError("the command hook rides the native segment-endpoint rollout")
        if (
            self.objective.control_state_objective == CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION
            and self.duration.control_duration_parameterization != CONTROL_DURATION_UNIFORM
        ):
            raise ValueError(
                "true-time-position control objective requires uniform control durations"
            )
        if (
            self.procedure_loss_active
            and self.objective.control_state_loss_grid != CONTROL_STATE_LOSS_GRID_NATIVE
        ):
            raise ValueError(
                "the procedure penalty on the control path is implemented on the native "
                "segment-endpoint rollout (its aligned targets carry the truth gate); "
                f"control_state_loss_grid={self.objective.control_state_loss_grid!r} is not supported"
            )


#: The view each ``prediction_output`` builds.
_OUTPUT_VIEWS: dict[str, type[OutputSpec]] = {
    PREDICTION_STATE: StateOutput,
    PREDICTION_CONTROL: ControlOutput,
}
if set(_OUTPUT_VIEWS) != set(PREDICTION_OUTPUTS):
    raise RuntimeError("every prediction_output needs an OutputSpec view")

#: The nested axes of the control view, in build order.
_CONTROL_AXES: tuple[tuple[str, type], ...] = (
    ("duration", DurationSpec),
    ("dynamics", DynamicsSpec),
    ("objective", ControlObjective),
    ("hook", HookSpec),
    ("latent", LatentSpec),
)


def _view_fields(spec: type) -> tuple[str, ...]:
    """A view's flat field names, its nested axes expanded (built after ``_FLAT_FIELD_NAMES``
    exists for `_own`; here the axes are named explicitly so this runs at import)."""
    nested = dict(_CONTROL_AXES)
    names: list[str] = []
    for f in fields(spec):
        if f.name in nested:
            names.extend(g.name for g in fields(nested[f.name]))
        else:
            names.append(f.name)
    return tuple(names)


#: Which output VARIANT owns each output-specific field. A field owned by one variant is
#: refused off its default under any other (`TSConfig._validate_ownership`) and normalised
#: to its default when a stored config of another output is loaded (`TSConfig.from_dict`):
#: unread by definition, it cannot have changed that run.
_OUTPUT_OWNED_FIELDS: dict[str, tuple[str, ...]] = {
    output: tuple(
        name for name in _view_fields(view)
        if name not in {f.name for f in fields(OutputSpec)}
    )
    for output, view in _OUTPUT_VIEWS.items()
}


@dataclass(frozen=True)
class TSConfig:
    """Everything that defines a run. Serialised whole into each checkpoint.

    FLAT on purpose: the vendored networks read attributes off it, every module reads
    ``config.<field>``, and the checkpoint stores it as one dict. What owns each field —
    and validates it — are the views above (`CohortSpec`, `BackboneSpec`, `TrainingSpec`
    and one `OutputSpec` per prediction path), built in ``__post_init__`` and exposed as
    ``config.cohort / .backbone / .training / .output``. Rules that read two views live on
    ``_validate_cross``.
    """

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
    # bounded to the final-approach corridor (see the constants). ``anchor-relative`` is
    # VETOED for new arms — kept only so the 2026-09-03 artifact stored under it loads.
    state_position_reference: str = STATE_POSITION_ABSOLUTE
    # State output only: the final-approach penalty (objective.procedure_loss). Hinge² on the
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
    # WHICH validation number the plateau is measured on (`selection` = the checkpoint
    # metric, today's behaviour; `objective` = the macro validation objective the epoch
    # record already carries as `val_loss`). It never changes which epoch is KEPT.
    lr_plateau_metric: str = LR_PLATEAU_METRIC_SELECTION
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
    # HOW that random anchor is drawn from the admissible ones — uniformly over the samples
    # (today's, and uniform over TIME) or uniformly over the remaining-path strata. It
    # cannot change WHICH anchors are admissible, so both policies train the same cohort.
    random_train_anchor_sampling: str = RANDOM_TRAIN_ANCHOR_SAMPLING_UNIFORM
    # ...and how much of that draw is RESERVED for the L-1 anchor (A2b, design §2.4d): with
    # probability `l1_share` the epoch's draw for a flight IS `default_anchor(config)`,
    # exactly the anchor the fixed-anchor arms train at, and otherwise the unchanged
    # remaining-path-uniform draw. A MIXTURE, not a reweighting of the span — the draws the
    # coin does not take are the very anchors the share-0 arm saw, because the digest's
    # salt does not move with the share. WHY: A0.b's random-anchor arms beat the
    # fixed-anchor one at every re-anchored bin and still LOSE at L-1
    # (`A0b_lr_objective_path_uniform` 1782 m pooled ADE against native32's 1322 m),
    # because under a law spread over the whole approach L-1 is one point among many draws.
    # 0 is the pure draw; a non-zero share needs `remaining-path-uniform`, so it can never
    # be the silent difference between two runs of the default policy.
    random_train_anchor_l1_share: float = 0.0
    # A common FIXED anchor for arms whose lookbacks differ (two-tier T0(c), 2026-09-16):
    # `default_anchor` is `max(L-1, anchor_floor_index)`, so the fixed anchor every window
    # set, validation replay, cohort filter and `predict` reads moves to the floor together,
    # and arms at L = 60 / 90 / 120 with the floor at 119 train and are judged on the SAME
    # flights at the SAME anchor. 0 = L-1, every stored run. `run_ts.py history_ablation`'s
    # `minimum_anchor_index` is the same floor passed at call time; this one is the run's own.
    anchor_floor_index: int = 0
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
    # What weighs the POINT head's squared duration residual. Under `duration_head`
    # `quantile` there is no point term, and a non-default value here is refused rather
    # than silently weighing nothing.
    final_time_loss_weight: float = DEFAULT_FINAL_TIME_LOSS_WEIGHT
    # ...and what weighs the QUANTILE head's five pinball losses (B1.b). Under `quantile`
    # it multiplies the `final_time` component the pinball sum replaced (default 1.0, which
    # is the number that multiplied it before this field existed); under `two-head` it is
    # the weight of the SEPARATE `duration_quantile` component beside the point term. Under
    # `point` there is no such term and a non-default value is refused.
    duration_quantile_loss_weight: float = DEFAULT_DURATION_QUANTILE_LOSS_WEIGHT
    # One explicit output-endpoint task prevents the last physical position from being
    # diluted to 1/N of the whole-path objective. It uses the same physical position scale
    # as the path loss; the 0.25 coefficient is frozen by the development Pareto audit.
    state_endpoint_loss_weight: float = 0.25
    # Cross-output compatibility knobs the control path reads. The formal direct-state objective
    # ignores both: it predicts position+duration and derives future velocity from position.
    kinematic_consistency_loss_weight: float = 3.0
    terminal_loss_weight: float = 0.02
    # Nondimensionalizes the direct-state physical 3D position MSE. This is an optimizer
    # scale, not an aviation acceptance threshold; the checkpoint records it explicitly.
    position_loss_scale_m: float = DEFAULT_POSITION_LOSS_SCALE_M
    final_time_scale_s: float = DEFAULT_FINAL_TIME_SCALE_S
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
    # The default keeps the historical normalized-channel MSE. ``true-time-position`` is
    # the minimal physical objective simple-v1 froze: 3-D position on the rollout's own
    # clock plus a soft observed endpoint, with the velocity and imitation terms as
    # opt-in weights. (``physical-criteria``, ``terminal-state`` and
    # ``arc-length-geometry`` all lived here once and are retired.)
    control_state_objective: str = CONTROL_STATE_OBJECTIVE_NORMALIZED_MSE
    # The true-time-position objective scores POSITION only, so a rollout may thread the
    # right places with the wrong heading and swing back between them — the measured
    # signature of that is a bank profile shared by every flight (see
    # docs/2026-08-19_control_bank_wiggle_diagnosis.zh.md). This term scores the chart
    # velocity at the same endpoints, on the same measured rows the position term uses:
    # supervision weights are already zero on fitted-tail velocities, so the placeholder
    # rows cannot enter. Zero keeps the frozen simple-v1 behaviour.
    control_velocity_loss_weight: float = 0.0
    # A latent intent on the control output (latent-intent design §六 L2; outputs/control/latent.py).
    # latent_dim = 0 is the plain deterministic head. z is drawn from q(z | future) in
    # training and from the prior's top-1 component at inference; it is never an output.
    latent_dim: int = 0
    latent_prior_components: int = 1          # K in the mixture prior; 1 = a single Gaussian
    latent_beta: float = 1.0                  # weight of KL(q ‖ p) in the objective
    latent_free_bits_nats: float = 0.0        # per-dim KL below this is not charged
    # Epochs over which beta ramps LINEARLY from 0 to latent_beta (0 = no warm-up, the
    # weight is its full value from epoch 1). L2.f: the penalty's mean term is what kills
    # the posterior mean's information, and it does it in the first ten epochs, before the
    # decoder has any use for z; a warm-up lets the decoder's z-weights grow first, so the
    # mean has a reconstruction gradient holding it out when the penalty arrives. The
    # earlier "annealing only postpones collapse" reading (L2.c) was about the VARIANCE
    # term and does not carry over to the mean term.
    latent_beta_warmup_epochs: int = 0
    # A TRAINING-ONLY auxiliary target on z (L2.f, arm 2): weight of the MSE between a
    # linear read-out of the posterior sample and the normalized remaining duration
    # (truth_duration_s / final_time_scale_s). It says outright what z must carry — the
    # "when" half of the intent — and gives the posterior mean a restoring force that does
    # not depend on the decoder having learned to use z. The head is never called at
    # inference and nothing about z reaches a record. Refused under cta_conditioning=given,
    # where the CTA already hands the duration to the decoder.
    latent_aux_duration_weight: float = 0.0
    # The posterior's standard deviation at initialization. 1.0 is the init the 2026-09-07
    # L2 runs collapsed under: a mean of ≈0.2 inside a unit-variance sample is noise to the
    # decoder, which learns to ignore z before the posterior can become informative — and
    # the KL then sits under the free-bits floor, so beta never binds (L2_gauss and
    # L2b_beta0p1 had identical KL trajectories). A narrow start hands the decoder an
    # informative z from step 0 and lets the KL widen it at the configured beta.
    latent_posterior_init_std: float = 1.0
    # The CTA as a decoder input (CTA_CONDITIONINGS); the given arrival time replaces the
    # duration head's output outright.
    cta_conditioning: str = CTA_CONDITIONING_OFF
    # The plan as a decoder input (PLAN_CONDITIONINGS): `off` is the only value a run may
    # carry — the retired token values are refused by name (PLAN_CONDITIONINGS_RETIRED).
    plan_conditioning: str = PLAN_CONDITIONING_OFF
    # Two-tier L1: the rollout's FIXED horizon in seconds; 0 = the whole remaining approach
    # (every stored run). Under Δ > 0 the schedule is rolled over exactly Δ, the targets cover
    # [0, Δ] (`dataset.target_horizon_s`), every anchor needs Δ of truth after it
    # (`dataset.window_anchors`) and no duration head is built (`ControlOutput` refuses the
    # axes that would decide a duration). Named `ctrl-horizon=`.
    control_horizon_s: float = 0.0
    # How the airframe is written into the head's condition vector (CONTROL_CONDITION_FEATURES).
    # Not in REQUIRED_SERIALIZED_CONTROL_FIELDS: every checkpoint trained before the field
    # existed read the raw set, so absence reproduces it exactly.
    control_condition_features: str = CONTROL_CONDITION_FEATURES_RAW
    # WHAT the duration head emits (DURATION_HEADS, B1/B1.b): one point estimate, the five
    # DURATION_QUANTILES, or both. Every value but `point` belongs to the control output
    # only. Under `quantile` the median walks the existing `final_time_s` contract (it IS
    # the duration the rollout flies); under `two-head` the POINT head walks it and the
    # quantiles are published beside it. Either way every downstream reader is unchanged.
    duration_head: str = DURATION_HEAD_POINT
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
    # L1.b (latent-intent design §六): supervision that names the bank WITHOUT a teacher.
    # Naming the bank is the imitation term's only real job, and its target solves
    # equations the rollout does not fly (open-loop it drifts 2.5-7.8 km from the truth),
    # so the decoder is pinned to controls that do not reproduce the track. These two
    # price the same information THROUGH the rollout instead.
    #
    # ``control_heading_rate_loss_weight`` scores the rollout's own turn rate at the
    # segment endpoints against the flown track's, in deg/s divided by
    # ``control_heading_rate_loss_scale_dps``. That scale is the unit the residual is read
    # in, not the dose: 1.5 deg/s is half a standard-rate turn, so missing a standard-rate
    # turn entirely costs 4 before the weight. ``control_bank_tv_loss_weight`` is the
    # structural half — the mean absolute step between adjacent COMMANDED banks, in units
    # of half the bank box. Both register ONLY under the true-time-position objective,
    # beside the velocity and imitation terms; a non-zero weight under any other objective
    # is refused below rather than silently ignored.
    control_heading_rate_loss_weight: float = 0.0
    control_heading_rate_loss_scale_dps: float = 1.5
    control_bank_tv_loss_weight: float = 0.0
    # WHICH schedule that term imitates (CONTROL_IMITATION_TARGETS above), and, under
    # ``fitted``, the table it reads. The path is carried, never opened, here: the width,
    # the anchor and the per-flight duration are checked where the flights are known, at
    # the dataset build, which refuses a table that does not cover the cohort.
    control_imitation_target: str = CONTROL_IMITATION_TARGET_INVERSE_DYNAMICS
    control_fitted_teacher_path: str = ""
    # Whether state-rollout gradients may update the learned duration partition. Turning
    # this off leaves the final-time loss trainable while controls own geometry fitting.
    control_state_duration_gradient: bool = True
    # Optional gradient-norm cap for deterministic control training: one global cap over
    # every parameter. Zero keeps historical behavior.
    control_gradient_clip_norm: float = 0.0
    # Rollout state representation is independent of the model/data coordinate frame.
    # The baseline re-anchors a local ENU RK4 step into geodetic state every sub-step;
    # scaled-transport-chart-velocity integrates threshold-chart position plus
    # moving-local-ENU physical velocity with the full WGS84 transport rate, in
    # order-one internal coordinates.
    control_dynamics_backend: str = CONTROL_DYNAMICS_REANCHORED_RK4
    # Which flight model the rollout integrates. ``first-order-lag`` augments the state
    # with the three actual control values and drives them towards the model's commands;
    # it reduces to ``point-mass`` as the time constants go to zero, and reuses the same
    # force equations, so the two are comparable rather than two separate models.
    control_dynamics_model: str = CONTROL_DYNAMICS_POINT_MASS
    # Which quantity the longitudinal control is (see CONTROL_THRUST_PARAMETERIZATIONS).
    # Not in REQUIRED_SERIALIZED_CONTROL_FIELDS: every checkpoint trained before the field
    # existed ran the default, so absence reproduces it exactly.
    control_thrust_parameterization: str = CONTROL_THRUST_FRACTION
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
    control_hook_saturation: str = HOOK_SATURATION_SOFT
    # Barrier filter: the barrier's decay rate α (1/s; the allowed closing rate toward a
    # corridor edge is α × the remaining margin) and the heading gain that turns a heading
    # error outside the admissible interval into a turn-rate demand (1/s).
    control_barrier_alpha: float = CONTROL_BARRIER_GAIN_DEFAULT
    control_barrier_heading_gain: float = CONTROL_BARRIER_GAIN_DEFAULT
    # Speed floor: how far above the stall speed of the COMMANDED load factor the rollout
    # is held (V_floor = margin x V_stall). 1.10 is the package's existing stall margin —
    # see CONTROL_SPEED_FLOOR_MARGIN_DEFAULT for the two places it already appears.
    # Deliberately NOT in REQUIRED_SERIALIZED_CONTROL_FIELDS, like the barrier's gains: a
    # config without a speed-floor hook cannot read it (the validation below refuses a
    # non-default value there), so the default IS what every stored artifact ran under.
    control_speed_floor_margin: float = CONTROL_SPEED_FLOOR_MARGIN_DEFAULT
    # Trombone: WHAT the surplus is measured against (see TROMBONE_SURPLUS_REFERENCES).
    # Same reasoning as the two fields above for staying out of
    # REQUIRED_SERIALIZED_CONTROL_FIELDS: a config without a trombone cannot read it (the
    # validation below refuses a non-default value there), and the default is what every
    # stored artifact — the whole L3.e campaign included — ran under.
    trombone_surplus_reference: str = TROMBONE_SURPLUS_BEELINE
    # Must match the high-fidelity replay integration cap. The Torch rollout subdivides every
    # learned non-uniform segment at this interval and is numerically contract-tested against
    # CasadiSimulator, rather than training on a cheaper second dynamics model.
    control_rollout_integrator_dt_s: float = 0.5

    # ── provenance (free-form; recorded in the checkpoint) ──────────────────
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize the two under-specified fields, then build the views.

        The order is the only one that works: the output must be a known one before the
        ownership rule can ask which fields are foreign to it; the backbone must be a known
        one before ``n_segments`` can be resolved through the per-model table; every view
        must exist before the recipe freeze and the cross-view rules can read them.
        """
        # Sequence fields arrive as lists from JSON; the contract is tuples (see
        # SEQUENCE_FIELDS / coerce_sequence_fields — the CLI's recipe check uses the same).
        for name in SEQUENCE_FIELDS:
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.prediction_output in PREDICTION_OUTPUTS_RETIRED:
            raise ValueError(
                f"prediction_output={self.prediction_output!r} is retired (2026-09-18): its code is "
                "under archive/ and its checkpoints no longer load "
                "(docs/2026-09-18_manoeuvre_token_plan.zh.md §5)"
            )
        # A sequence field arrives as a tuple, a JSON list or an arm file's list: ONE form.
        for name in SEQUENCE_FIELDS:
            object.__setattr__(self, name, tuple(getattr(self, name)))
        _require_member("prediction_output", self.prediction_output, PREDICTION_OUTPUTS)
        self._validate_ownership()
        cohort = CohortSpec(**_own(CohortSpec, self))
        backbone = BackboneSpec(**_own(BackboneSpec, self))
        # After the backbone's vocabulary, not before: the table is keyed by the model name.
        if self.n_segments is None:
            object.__setattr__(
                self, "n_segments", DEFAULT_N_SEGMENTS_BY_MODEL[self.model]
            )
        training = TrainingSpec(**_own(TrainingSpec, self))
        view = _OUTPUT_VIEWS[self.prediction_output]
        axes = (
            {name: axis(**_own(axis, self)) for name, axis in _CONTROL_AXES}
            if view is ControlOutput else {}
        )
        output = view(**_own(view, self), **axes)
        object.__setattr__(self, "_views", (cohort, backbone, training, output))
        self._validate_recipe()
        self._validate_cross()

    # ── the views ────────────────────────────────────────────────────────────
    # Not dataclass fields (they would serialise): built once in __post_init__ and read
    # through these. `dataclasses.replace` rebuilds them with the instance.

    @property
    def cohort(self) -> CohortSpec:
        return self._views[0]

    @property
    def backbone(self) -> BackboneSpec:
        return self._views[1]

    @property
    def training(self) -> TrainingSpec:
        return self._views[2]

    @property
    def output(self) -> OutputSpec:
        """The typed view of this run's prediction path: `StateOutput` or `ControlOutput`
        by ``prediction_output``."""
        return self._views[3]

    def _validate_ownership(self) -> None:
        """A field owned by ANOTHER output variant must sit at its default.

        It would be unread under this output and still serialized into the checkpoint and
        carried into the run name — a value that cannot change an answer while claiming
        to (review C-1 / C-2 / C-4 were three instances of the hole this closes). One rule
        for all three variants replaces the per-field "belongs to output X" checks.
        """
        for owner, names in _OUTPUT_OWNED_FIELDS.items():
            if owner == self.prediction_output:
                continue
            for name in names:
                value = getattr(self, name)
                if value != _OWNED_FIELD_DEFAULTS[name]:
                    raise ValueError(
                        f"{name}={value!r} belongs to the {owner} output; "
                        f"prediction_output={self.prediction_output!r} never reads it and "
                        "would still carry it into the checkpoint and the run name"
                    )

    def _validate_recipe(self) -> None:
        """A named recipe's fields are frozen at the values that define it.

        `simple-v1/v2/v3` are published comparison arms, so a run that says it is one and
        differs anywhere is a different experiment wearing the same name.
        """
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

    def _validate_cross(self) -> None:
        """The rules that read TWO views. Each names the pair it binds; a rule inside one
        view lives on that view's dataclass."""
        # training × output — a scheduler that watches the objective must watch a
        # COMPARABLE objective. Two things move the number under the model's feet, and
        # under either the plateau scheduler could halve the learning rate straight through
        # a schedule that is still ramping: the KL warm-up (`effective_latent_beta`
        # reweights the objective every epoch until the ramp ends) and the procedure
        # penalty's dual step (λ is updated once per epoch, so the same trajectory is priced
        # differently each time).
        if self.lr_plateau_metric == LR_PLATEAU_METRIC_OBJECTIVE:
            if self.latent_beta_warmup_epochs > 0:
                raise ValueError(
                    "lr_plateau_metric='objective' with "
                    f"latent_beta_warmup_epochs={self.latent_beta_warmup_epochs}: the "
                    "validation objective is scored under THIS epoch's beta, so it rises "
                    "with the ramp and the plateau scheduler would cut the learning rate "
                    "through the warm-up. Select on the objective only at a fixed beta"
                )
            if self.procedure_loss_dual_step > 0.0:
                raise ValueError(
                    "lr_plateau_metric='objective' with "
                    f"procedure_loss_dual_step={self.procedure_loss_dual_step:g}: the "
                    "penalty multipliers move once per epoch, so the objective prices the "
                    "same trajectory differently each epoch and its plateau is not the "
                    "model's. Use fixed multipliers, or step the scheduler on the "
                    "selection metric"
                )
        # cohort × output — the charts an output is written in.
        if self.uses_final_approach_context and self.coordinate_frame != COORDINATE_FRAME_ENU:
            raise ValueError(
                "the final-approach corridor (corridor-bounded output / procedure loss) is "
                "written in the threshold-anchored ENU chart — the target at the origin, "
                f"east/north axes; coordinate_frame={self.coordinate_frame!r} would measure "
                "it from the wrong point or rotate it twice"
            )
        if self.control_command_hook != CONTROL_HOOK_OFF and self.coordinate_frame != COORDINATE_FRAME_ENU:
            raise ValueError("the command hook reads the threshold-anchored ENU chart")
        if self.plan_conditioning != PLAN_CONDITIONING_OFF and self.coordinate_frame != COORDINATE_FRAME_ENU:
            raise ValueError(
                "the plan token is written in the threshold-anchored ENU chart; "
                f"coordinate_frame={self.coordinate_frame!r}"
            )
        # cohort × output — under a fixed horizon the targets cover [0, Δ], so a train anchor
        # needs Δ of truth after it. `dataset.window_anchors` raises every window set's floor
        # to Δ; a stated random-anchor floor BELOW it would name a population the run never
        # trains on, so it is refused rather than silently raised.
        # ...and the horizon is cut into `dt_s` rows (the segment rows, the rollout's query grid):
        # a horizon that is not a whole number of steps would be refused at model build, past
        # the campaign runner's dry run.
        if self.control_horizon_s and not _whole_multiple(self.control_horizon_s, self.dt_s):
            raise ValueError(
                f"control_horizon_s={self.control_horizon_s:g} is not a whole number of "
                f"dt_s={self.dt_s:g} steps"
            )
        if (
            self.control_horizon_s
            and self.random_train_anchor
            and self.random_train_anchor_min_future_s < self.control_horizon_s
        ):
            raise ValueError(
                f"random_train_anchor_min_future_s={self.random_train_anchor_min_future_s:g} is "
                f"below control_horizon_s={self.control_horizon_s:g}: the targets cover the whole "
                "horizon, so every train anchor needs at least that much truth after it"
            )
        # ...and the one intent channel that carries the truth's remaining DURATION is the leak
        # the CTA refusal under Δ exists to prevent, one input over.
        if self.control_horizon_s and self.intent_conditioning == INTENT_CONDITIONING_TRUTH_JOIN_DURATION:
            raise ValueError(
                f"intent_conditioning={self.intent_conditioning!r} feeds the truth's remaining "
                f"duration to a control_horizon_s={self.control_horizon_s:g} run, which predicts "
                "no duration and must not read one"
            )
        if self.uses_fitted_teacher and self.random_train_anchor:
            raise ValueError(
                "the fitted teacher is a table of schedules fitted AT the fixed anchor; "
                "random_train_anchor would supervise other anchors with it"
            )
        # cohort × backbone — a channel-independent backbone cannot route a conditioning
        # token anywhere.
        if (
            self.target_conditioning == TARGET_CONDITIONING_CHANNELS
            and self.model != "itransformer"
        ):
            raise ValueError(
                f"target_conditioning={TARGET_CONDITIONING_CHANNELS!r} requires the "
                f"itransformer backbone: {self.model!r} is channel-independent, so a "
                "conditioning channel could never reach the state channels"
            )
        if self.intent_conditioning != INTENT_CONDITIONING_NONE and self.model != "itransformer":
            raise ValueError(
                f"intent_conditioning={self.intent_conditioning!r} requires the "
                f"itransformer backbone: {self.model!r} is channel-independent, so a "
                "conditioning channel could never reach the state channels"
            )
        # training × output — which epoch is kept, against what the objective reads.
        if self.latent_dim > 0 and self.checkpoint_selection_metric == CHECKPOINT_SELECTION_OBJECTIVE:
            raise ValueError(
                "a latent control run cannot select its checkpoint on the validation objective: "
                "that objective decodes a posterior sample (it reads the future, and is "
                "stochastic); select on a fixed-anchor replay metric instead"
            )
        # The anchor-grid metric re-anchors the validation replay at every bin, so an
        # oracle input is re-read from the future AT EACH ANCHOR — the same reason
        # `experiments/anytime_curve.py` refuses these two checkpoints outright. Selecting an
        # epoch on that is selecting on how fast the oracle converges.
        if self.checkpoint_selection_metric == CHECKPOINT_SELECTION_ANCHOR_GRID_ADE:
            if self.cta_conditioning == CTA_CONDITIONING_GIVEN:
                raise ValueError(
                    "cta_conditioning=given reads the future — the CTA IS the truth "
                    f"duration — and {CHECKPOINT_SELECTION_ANCHOR_GRID_ADE} would re-read "
                    "it at every bin anchor, so the metric would be selecting on an answer "
                    "it was handed, not on a prediction"
                )
            if self.intent_conditioning != INTENT_CONDITIONING_NONE:
                raise ValueError(
                    f"intent_conditioning={self.intent_conditioning!r} reads the FUTURE "
                    "(the truth join point / the lead's true landing time) and "
                    f"{CHECKPOINT_SELECTION_ANCHOR_GRID_ADE} re-reads it afresh at every "
                    "bin anchor, so the metric would be selecting on the oracle, not the model"
                )
            if self.plan_conditioning != PLAN_CONDITIONING_OFF:
                raise ValueError(
                    f"plan_conditioning={self.plan_conditioning!r} reads the truth's plan and "
                    f"{CHECKPOINT_SELECTION_ANCHOR_GRID_ADE} re-reads it at every bin anchor, so "
                    "the metric would be selecting on the oracle, not the model"
                )

    # ── derived values, kept on the flat config for every existing reader ────

    @property
    def procedure_loss_active(self) -> bool:
        return self.output.procedure_loss_active

    @property
    def uses_fitted_teacher(self) -> bool:
        """Whether the dataset must load the per-flight teacher table and supervise from it."""
        return self.control_imitation_target == CONTROL_IMITATION_TARGET_FITTED

    @property
    def uses_final_approach_context(self) -> bool:
        """Whether batches carry the per-flight runway course / glidepath row."""
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
        return self.cohort.input_channels

    # The model INPUT width. PatchTST reads configs.enc_in; iTransformer infers the token
    # count from the tensor, but its duration head and the control feature head flatten
    # over exactly this many channels.
    @property
    def enc_in(self) -> int:
        return len(self.input_channels)

    @property
    def pred_len(self) -> int:
        """Vendored model output length under the selected horizon contract."""
        return self.output.pred_len

    @property
    def horizon_s(self) -> float | None:
        """Physical coverage of fixed-time modes; normalized time has no fixed cap."""
        if self.horizon_mode == HORIZON_NORMALIZED:
            return None
        return self.pred_len * self.dt_s

    @property
    def lookback_s(self) -> float:
        """Wall-clock seconds of observed track the model is shown."""
        return self.cohort.lookback_s

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
        return self.training.resolved_split_seed

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for name in SEQUENCE_FIELDS:
            data[name] = list(getattr(self, name))  # JSON has no tuple
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TSConfig:
        """Rebuild a config, refusing a checkpoint that predates a recipe field.

        A missing key would otherwise take this build's DEFAULT, silently rewriting the
        recipe a trained artifact was produced under. Every field whose default is not a
        safe stand-in for "the old runs did this" is therefore required, not defaulted.
        """
        data = dict(data)
        missing = missing_required_fields(data)
        if missing:
            raise ValueError(
                f"serialized config is missing {', '.join(sorted(missing))}; "
                "regenerate the derived checkpoint"
            )
        # A retired PLAN value is named before the retired-constant sweep below: the archived
        # L1 / L1b arms carry `plan_conditioning_dropout=0.5` beside it, and the constant's
        # refusal would otherwise fire first and point nowhere.
        if data.get("plan_conditioning") in PLAN_CONDITIONINGS_RETIRED:
            raise ValueError(
                f"plan_conditioning={data['plan_conditioning']!r} is retired: its token builder is "
                f"under {PLAN_CONDITIONINGS_RETIRED[data['plan_conditioning']]} and the checkpoints "
                "that carry it no longer load"
            )
        # An UNREAD retired field is dropped by name: nothing read it, so the stored value
        # could not have changed the run, and refusing the artifact would be a contract
        # change in the wrong direction.
        for name in RETIRED_SERIALIZED_FIELDS:
            data.pop(name, None)
        # A MEASURED-CONSTANT retired field is dropped only when it agrees with the constant
        # that replaced it. It was a live loss denominator, so a different stored value
        # describes a run this build cannot reproduce — that is a loud failure, not a
        # silent drop.
        for name, constant in RETIRED_CONSTANT_FIELDS.items():
            if name not in data:
                continue
            stored = data.pop(name)
            if stored != constant:
                raise ValueError(
                    f"serialized config sets {name}={stored!r}, but the field was retired "
                    f"to the constant {constant!r} (this build has no other value of it); "
                    "this artifact was produced under a value this build cannot reproduce"
                )
        output = data.get("prediction_output", PREDICTION_STATE)
        # A field of a RETIRED OUTPUT is dropped at the default it held under a live output,
        # where the ownership rule pinned it and nothing read it. A moved value belongs to
        # that output's own run: under its own `prediction_output` the config is refused
        # below by `__post_init__`, which gives the reason, so only a state/control config
        # carrying a moved value is named here.
        for retired_output, retired_fields in RETIRED_OUTPUT_FIELDS.items():
            for name, default in retired_fields.items():
                if name not in data:
                    continue
                stored = data.pop(name)
                if stored != default and output not in PREDICTION_OUTPUTS_RETIRED:
                    raise ValueError(
                        f"serialized config sets {name}={stored!r}, a field of the retired "
                        f"{retired_output!r} output (2026-09-18) whose code is under archive/; "
                        f"only the default {default!r} it holds under a live output is dropped, "
                        "and no stored state or control artifact moved it"
                    )
        # A field owned by ANOTHER output variant was unread under this run's output, so
        # its stored value could not have changed the run: it is normalised to the default
        # the ownership rule requires (`_OUTPUT_OWNED_FIELDS`) rather than refused.
        # Measured 2026-09-10 across every stored config that loads: none moves.
        for owner, names in _OUTPUT_OWNED_FIELDS.items():
            if owner != output:
                for name in names:
                    if name in data:
                        data[name] = _OWNED_FIELD_DEFAULTS[name]
        for name in SEQUENCE_FIELDS:
            if name in data:
                data[name] = tuple(data[name])
        return cls(**data)


#: The flat schema's field names — what `_own` selects a view's slice from.
_FLAT_FIELD_NAMES: frozenset[str] = frozenset(f.name for f in fields(TSConfig))
#: The default every output-owned field must sit at under another output.
_OWNED_FIELD_DEFAULTS: dict[str, Any] = {
    f.name: f.default
    for f in fields(TSConfig)
    if any(f.name in names for names in _OUTPUT_OWNED_FIELDS.values())
}
#: Free text read by nobody: the one field outside every view.
_UNVIEWED_FIELDS: frozenset[str] = frozenset({"notes"})


def owned_field_defaults(prediction_output: str) -> dict[str, Any]:
    """The defaults of the fields ``prediction_output``'s view OWNS — what a config must
    carry for them to be read as another output's (`_validate_ownership` refuses them off
    their defaults there). A rollout flown under a CONTROL config derived from another
    output's run must not carry that run's own fields along into it."""
    return {name: _OWNED_FIELD_DEFAULTS[name] for name in _OUTPUT_OWNED_FIELDS[prediction_output]}


def _check_view_partition() -> None:
    """Every `TSConfig` field belongs to exactly one view — checked at import so a field
    added to the flat schema without an owner (or to two) fails the first test run."""
    groups: dict[str, tuple[str, ...]] = {
        "CohortSpec": _view_fields(CohortSpec),
        "BackboneSpec": _view_fields(BackboneSpec),
        "TrainingSpec": _view_fields(TrainingSpec),
        "OutputSpec": _view_fields(OutputSpec),
        **{f"{owner} output": names for owner, names in _OUTPUT_OWNED_FIELDS.items()},
    }
    owners: dict[str, list[str]] = {}
    for group, names in groups.items():
        for name in names:
            owners.setdefault(name, []).append(group)
    twice = {name: who for name, who in owners.items() if len(who) > 1}
    unknown = sorted(set(owners) - _FLAT_FIELD_NAMES)
    orphans = sorted(_FLAT_FIELD_NAMES - set(owners) - _UNVIEWED_FIELDS)
    if twice or unknown or orphans:
        raise RuntimeError(
            "TSConfig views do not partition the flat schema: "
            f"owned twice {twice}, not a TSConfig field {unknown}, no view {orphans}"
        )
    missing_default = [name for name, d in _OWNED_FIELD_DEFAULTS.items() if d is MISSING]
    if missing_default:
        raise RuntimeError(
            f"output-owned fields need a plain default to be normalised to: {missing_default}"
        )


_check_view_partition()


def _required_serialized_fields(data: Mapping[str, Any]) -> tuple[str, ...]:
    """The fields `TSConfig.from_dict` REQUIRES of a stored config of this output."""
    output = data.get("prediction_output", PREDICTION_STATE)
    return (
        *REQUIRED_SERIALIZED_FIELDS,
        *(REQUIRED_SERIALIZED_CONTROL_FIELDS if uses_control_dynamics(output) else ()),
    )


def missing_required_fields(data: Mapping[str, Any]) -> list[str]:
    """The required fields a stored config lacks — what `TSConfig.from_dict` refuses."""
    return [name for name in _required_serialized_fields(data) if name not in data]


def absent_field_defaults(data: Mapping[str, Any]) -> dict[str, Any]:
    """What `TSConfig.from_dict` reads for every field a stored config does NOT carry: this
    build's default, for each field it does not require (a missing REQUIRED field is refused
    there, so it is left out here and still reads as absent).

    This is the ABSENT-field half of `from_dict`'s rule (it does not normalise fields that
    are present but owned by another output), for readers that compare a stored config field
    by field. The campaign runner's resume check read an absent field as ``None``, so the
    first field a recipe pins AFTER an arm was trained (``control_thrust_parameterization``,
    2026-09-14) refused every such arm's resume although the arm flew exactly the default.
    """
    required = set(_required_serialized_fields(data))
    return {
        name: value
        for name, value in json.loads(json.dumps(TSConfig().to_dict())).items()
        if name not in data and name not in required
    }


def lookback_anchor(config: TSConfig) -> int:
    """The earliest anchor with a complete observed lookback: ``L-1``, what the label "L-1"
    names wherever an artifact says which anchor it was taken at."""
    return config.seq_len - 1


def fixed_anchor_label(config: TSConfig, anchor: int | None = None) -> str:
    """What an artifact says about the anchor it was taken at (``anchor``, by default the
    run's fixed one): ``fixed L-1`` where it IS L-1, ``fixed index N`` elsewhere — one
    rendering, so a floored run's fit evaluation, checkpoint and cohort audit agree."""
    anchor = default_anchor(config) if anchor is None else int(anchor)
    return "fixed L-1" if anchor == lookback_anchor(config) else f"fixed index {anchor}"


def default_anchor(config: TSConfig) -> int:
    """The fixed anchor: the earliest with a complete lookback, or the run's common floor.

    It is `L-1` unless `anchor_floor_index` sets a later one (two-tier T0(c): arms with
    different lookbacks judged at one anchor): the anchor every fixed-anchor arm trains at,
    the anchor every prediction defaults to, and — since A2b — the anchor a random-anchor
    run reserves a share of its draws for. It lives HERE rather than in `forecast` because
    `dataset` needs the same index and `forecast` imports `dataset`; re-exported there so
    every existing call site is unchanged.
    """
    return max(lookback_anchor(config), config.anchor_floor_index)


#: The tolerance a horizon, span or step is judged against a time grid on (seconds) — the row
#: tolerance every grid test in the package uses (`data/time_grids.ROW_TOLERANCE_S`, which this
#: leaf cannot import; the two must stay equal).
_GRID_TOLERANCE_S = 1e-6


def _whole_multiple(value: float, unit: float) -> bool:
    return abs(round(value / unit) * unit - value) <= _GRID_TOLERANCE_S


def control_recipe(config: TSConfig) -> dict[str, Any]:
    """Serialize the complete recipe for a control-output strategy."""
    base: dict[str, Any] = {
        "reference_velocity_source": config.reference_velocity_source,
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
        "heading_rate_loss_weight": config.control_heading_rate_loss_weight,
        "heading_rate_loss_scale_dps": config.control_heading_rate_loss_scale_dps,
        "bank_tv_loss_weight": config.control_bank_tv_loss_weight,
        "state_duration_gradient": config.control_state_duration_gradient,
        "gradient_clip_norm": config.control_gradient_clip_norm,
    }
    if config.control_recipe_name != CONTROL_RECIPE_CUSTOM:
        base["name"] = config.control_recipe_name
    # Present only off the default: every stored checkpoint's metadata predates the field,
    # and `experiments.pipeline` compares this dict for reuse — writing the default too
    # would refuse every one of them.
    if config.control_thrust_parameterization != CONTROL_THRUST_FRACTION:
        base["thrust_parameterization"] = config.control_thrust_parameterization
    if config.control_condition_features != CONTROL_CONDITION_FEATURES_RAW:
        base["condition_features"] = config.control_condition_features
    if config.control_horizon_s:
        base["horizon_s"] = config.control_horizon_s
    if config.plan_conditioning != PLAN_CONDITIONING_OFF:
        base["plan"] = config.plan_conditioning
    if not uses_control_dynamics(config.prediction_output):
        raise ValueError("state output has no control recipe")
    return base
