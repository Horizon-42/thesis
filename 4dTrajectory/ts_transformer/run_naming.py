"""Canonical display names and slugs for ts_transformer runs.

One grammar for every surface that names a trained run (frontend category labels, the
experiment picker, publication manifests, future run directories):

    <output> · <backbone> · <dynamics> · <loss design> · <meta, …>

- **output** — ``prediction_output``: ``state`` (kinematic baseline) or ``control``.
- **backbone** — ``model``: ``iTransformer`` / ``PatchTST``; unknown future backbones pass
  through verbatim.
- **dynamics** — ``kinematic`` for the state output (no dynamics attached, by design);
  for control: ``point-mass`` (commands applied instantly — no control derivative) or
  ``first-order-lag`` (controls integrated through a first-order actuator ODE — with
  control derivative), with non-default time constants and a non-default rollout backend
  appended.
- **loss design** — the named control recipe (``simple-v1`` … ``simple-v3``) when one is
  set; a ``custom`` run is named against its NEAREST recipe — the recipe whose frozen
  loss fields need the fewest edits to reproduce the run — e.g. ``simple-v2+(imit=16)``.
  More than ``_MAX_LISTED_DIFFS`` residual edits collapse to a stable content hash
  (``custom-3f2a91bc``) — the "version" form for loss designs too complex to spell out.
- **meta** — possibly empty: non-default horizon mode, then every ``META_FIELDS`` entry
  that deviates from its default (capacity, budget, cohort, seed …) up to
  ``_MAX_LISTED_META`` (the rest fold into ``+N more``), then free-text extras from the
  caller (run id, campaign/arm, cohort scope).

Everything is derived from the run's serialized config dict — the exact object stored in
``history.json['config']``, checkpoint metadata, and experiment/publication manifests —
so a name can always be recomputed for any run ever trained, without touching artifacts.

The same grammar has a STRUCTURED form, :func:`run_parameter_rows`: every part as a named
row, the loss design's edits and every non-default setting listed in full (no ``+N more``, no
hash) under a section — what the frontend Experiments picker renders as the run's parameters.

Names describe a config relative to TODAY'S defaults: when a default changes, old runs'
names gain (or lose) a meta item. That is deliberate — the name answers "what was
special about this run", and "special" is defined by the current baseline.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

from ts_transformer.config import (
    CONTROL_HOOK_FIELDS,
    CTA_FIELDS,
    PLAN_CONDITIONING_FIELDS,
    DURATION_HEAD_QUANTILE,
    DURATION_HEAD_TWO_HEAD,
    DURATION_QUANTILES,
    INTENT_FIELDS,
    PROCEDURE_LOSS_FIELDS,
    CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    CONTROL_DYNAMICS_POINT_MASS,
    CONTROL_DYNAMICS_REANCHORED_RK4,
    CONTROL_RECIPE_CUSTOM,
    CONTROL_RECIPE_NAMES,
    CONTROL_PARAMETERIZATION_SCOPES,
    CONTROL_THRUST_FRACTION,
    DEFAULT_N_SEGMENTS_BY_MODEL,
    HORIZON_FULL,
    HORIZON_WINDOW,
    PREDICTION_CLOSURE,
    PREDICTION_CONTROL,
    PREDICTION_PLAN,
    TSConfig,
    control_recipe_overrides,
)


_BACKBONE_DISPLAY = {"itransformer": "iTransformer", "patchtst": "PatchTST"}
_BACKBONE_SLUG = {"itransformer": "itr", "patchtst": "ptst"}
_DYNAMICS_SLUG = {
    "kinematic": "kin",
    CONTROL_DYNAMICS_POINT_MASS: "pm",
    CONTROL_DYNAMICS_FIRST_ORDER_LAG: "lag",
}
_BACKEND_SLUG = {
    # `transport-chart-velocity` is RETIRED from the config vocabulary (T2, 2026-09-07) and
    # stays here on purpose: 13 stored 2026-07/08 configs carry it, their on-disk run
    # directories are named `…_tcv`, and on-disk names are historical record. A grammar
    # that could no longer name them would rename them.
    "transport-chart-velocity": "tcv",
    "scaled-transport-chart-velocity": "stcv",
}

#: Base name for the state output's loss design (the formal direct-state objective at
#: its frozen coefficients). Bump when that objective itself is redesigned.
STATE_LOSS_BASE = "state-v1"

#: Diffs listed inline up to this count; past it the loss name collapses to a hash.
_MAX_LISTED_DIFFS = 4

#: Meta items listed inline up to this count; the rest fold into ``+N more``.
_MAX_LISTED_META = 6

# The three field groups the grammar reads. Guarded below against TSConfig drift.
CONTROL_LOSS_FIELDS = (
    "control_state_objective",
    "control_state_loss_grid",
    "control_state_supervision_clock",
    "control_state_duration_gradient",
    "control_velocity_loss_weight",
    "control_velocity_loss_scale_mps",
    "control_imitation_loss_weight",
    # L1.b's teacherless bank supervision (heading rate through the rollout, bank total
    # variation). Every stored config predates them and carries the defaults, so adding
    # them here renames nothing.
    "control_heading_rate_loss_weight",
    "control_heading_rate_loss_scale_dps",
    "control_bank_tv_loss_weight",
    # WHICH teacher that weight imitates: only the non-default `fitted` names a run, and it
    # has to, because two runs on the same weight and different teachers are different runs.
    "control_imitation_target",
    "final_time_loss_weight",
    # B1.b: what weighs the quantile head's pinball sum. Under `point` a non-default value
    # is refused outright, so no stored config can carry one and adding it here renames
    # nothing (recounted on disk); the named recipes leave it open for the same reason.
    "duration_quantile_loss_weight",
    "final_time_scale_s",
    "position_loss_scale_m",
    # The final-approach penalty is an objective on BOTH paths (it acts on the control
    # rollout's segment endpoints too): a control run that carries it is a recipe edit.
    *PROCEDURE_LOSS_FIELDS,
    # The latent intent's objective knobs (outputs/control/latent.py); last, so tests that slice
    # the recipe fields off the front keep reading recipe fields.
    "latent_beta",
    "latent_free_bits_nats",
    "latent_posterior_init_std",
    # L2.f: the KL weight's warm-up and the auxiliary intent target on z. Every stored
    # config predates them and carries the defaults, so adding them here renames nothing
    # (recounted on disk).
    "latent_beta_warmup_epochs",
    "latent_aux_duration_weight",
)
#: The closure output's objective fields; its base name bumps when the regression
#: itself is redesigned.
CLOSURE_LOSS_BASE = "closure-v1"
CLOSURE_LOSS_FIELDS = (
    "closure_slowness_knots",
    "closure_height_knots",
    "closure_geometry_loss_weight",
    "closure_timing_loss_weight",
    "closure_height_loss_weight",
)
#: The plan output's objective fields; its base name bumps when the regression itself
#: is redesigned.
PLAN_LOSS_BASE = "plan-v1"
PLAN_LOSS_FIELDS = ("plan_operating_loss_weight", "plan_instruction_loss_weight", "plan_fan_components")
# Fields whose value is a path: rendered as the file's parent/name (two label generations
# in different directories must not read as one).
_PATH_FIELDS = frozenset({"closure_labels_path", "control_fitted_teacher_path", "plan_rolled_windows_path"})
STATE_LOSS_FIELDS = (
    "fitted_tail_position_weight",
    "fitted_terminal_position_weight",
    "state_endpoint_loss_weight",
    "kinematic_consistency_loss_weight",
    "terminal_loss_weight",
    "final_time_loss_weight",
    "final_time_scale_s",
    "position_loss_scale_m",
    *PROCEDURE_LOSS_FIELDS,
)
META_FIELDS = (
    # Tuple order is display priority: the first _MAX_LISTED_META deviations are spelled
    # out, the rest fold into "+N more" — keep identity-bearing fields (seed) up front.
    # The intent axis is identity-bearing too: a truth-conditioned run reads the future,
    # and its name is what keeps it from being quoted as a predictor, so it never folds.
    "seed",
    "split_seed",
    # The supervision target of a closure run: two runs on different label files are
    # different runs, whatever else matches.
    "closure_labels_path",
    # ...and the same for the imitation term's fitted teacher: `imit-target=fitted` says
    # WHICH KIND of teacher, this says which one. A table is width-, anchor- and
    # cohort-specific, so two generations of it are two different runs.
    "control_fitted_teacher_path",
    # ...and the plan head's rolled-window table (v5.2): which lockstep flights' windows it
    # trained on, and at what share of its draws — a second table is a second run.
    "plan_rolled_windows_path",
    "plan_rolled_share",
    *INTENT_FIELDS,
    # The CTA axis reads the future the same way: a given-CTA run must wear it.
    *CTA_FIELDS,
    # ...and the plan token (two-tier T1): a `plan=truth-next` run reads the truth's plan.
    *PLAN_CONDITIONING_FIELDS,
    # Two-tier L1: a fixed rollout horizon predicts a different thing (Δ seconds of the
    # approach, no duration head), so it is spelled out ahead of the backbone knobs that fold.
    # Every stored config predates it and carries 0, so adding it renames nothing.
    "control_horizon_s",
    # B1 / B1.b: which duration head. The interval a run publishes is part of what it IS,
    # and the named recipes pin this at `point`, so a `quantile` or `two-head` run is
    # `custom` and this item shows (`T=q5` / `T=2h`).
    "duration_head",
    # N4: how the airframe is written into the head's condition vector. It is the whole
    # difference between an N4 arm and its twin, so it sits ahead of the backbone knobs that
    # fold. Every stored config predates it and reads as the default, so adding it renames
    # nothing.
    "control_condition_features",
    "d_model",
    "n_heads",
    "d_ff",
    "e_layers",
    "dropout",
    "patch_len",
    "stride",
    "n_segments",
    "seq_len",
    # Two-tier T0(c): a common fixed anchor moves every window, the cohort and the anchor a
    # run is judged at, so two runs differing only here are different experiments. Beside
    # `seq_len`, which it is compared with, so it is spelled out rather than folded. Every
    # stored config predates it and carries the default 0, so adding it renames nothing.
    "anchor_floor_index",
    "dt_s",
    "full_horizon_steps",
    "window_horizon_steps",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "epochs",
    "patience",
    "val_fraction",
    "test_fraction",
    "aircraft_filter",
    "aircraft_type",
    "coordinate_frame",
    "target_conditioning",
    "state_position_reference",
    "reference_velocity_source",
    "use_norm",
    "revin",
    "random_train_anchor",
    # A0.b: HOW the random anchor is drawn. Two runs that differ only here see different
    # anchor distributions every epoch, so it is as identity-bearing as the flag above it.
    # Every stored config predates it and carries the default (recounted on disk).
    "random_train_anchor_sampling",
    # A2b: how much of that draw is RESERVED for L-1. A mixture is a third law, not a
    # tuning knob — two runs that differ only here see a different anchor distribution
    # every epoch. Every stored config predates it and carries the default 0, so adding
    # it renames nothing (recounted on disk).
    "random_train_anchor_l1_share",
    # The future floor a random train anchor must leave: it decides WHICH anchors are
    # admissible, so two runs differing only here train on different anchor populations.
    # Named 2026-09-09 (review C-3, decided): 7 stored runs at 20 s gain `anchor-min-future=20`.
    "random_train_anchor_min_future_s",
    "training_cohort_min_future_s",
    "checkpoint_selection_metric",
    # The common-grid resolution the selection metric is SCORED on: two runs differing
    # only here keep different epochs. Every stored config carries the default 64, so
    # adding it renames nothing (recounted on disk 2026-09-09; review C-3).
    "validation_common_grid_points",
    # A0.b: WHICH validation number the LR scheduler measures its plateau on. Two runs that
    # differ only here train under different learning-rate schedules from the epoch the two
    # numbers part, so it is an identity field. Every stored config predates it and carries
    # the default, so adding it renames nothing (recounted on disk).
    "lr_plateau_metric",
    # ...and how long it waits and how hard it cuts: two runs differing only here train under
    # different learning-rate schedules. Named 2026-09-09 (review C-3, decided): the named
    # recipes pin both, so recipe runs are unchanged; custom runs off the defaults (3 / 0.5)
    # gain `lr-patience=` / `lr-factor=`. Measured before landing it, together with the anchor
    # floor below: 146 of the 219 stored runs' names / slugs moved (75 spelled, 71 only the
    # folded `+N more` count and hash), 0 loadability changes.
    "lr_plateau_patience",
    "lr_plateau_factor",
    "control_duration_parameterization",
    "control_duration_uniform_floor",
    "control_gradient_clip_norm",
    "control_rollout_integrator_dt_s",
    *CONTROL_HOOK_FIELDS,
)

_TAU_FIELDS = (
    ("control_thrust_time_constant_s", "τ-thrust"),
    ("control_bank_time_constant_s", "τ-bank"),
    ("control_load_time_constant_s", "τ-load"),
)

#: Values whose display form is not the value itself. One field, and it exists because the
#: design names the item ``T=q5``: the 5 is ``len(DURATION_QUANTILES)``, so the name moves
#: with the tuple instead of restating its length. ``two-head`` is ``T=2h`` — the two heads,
#: short enough to sit in a run name beside the weight that tells them apart.
_VALUE_ABBREV: dict[str, dict[Any, str]] = {
    "duration_head": {
        DURATION_HEAD_QUANTILE: f"q{len(DURATION_QUANTILES)}",
        DURATION_HEAD_TWO_HEAD: "2h",
    },
}

_ABBREV = {
    "duration_head": "T",
    "latent_posterior_init_std": "q-std",
    "latent_beta_warmup_epochs": "beta-warmup",
    "latent_aux_duration_weight": "aux-T",
    "control_imitation_loss_weight": "imit",
    "control_imitation_target": "imit-target",
    "control_velocity_loss_weight": "vel",
    "control_velocity_loss_scale_mps": "vel-scale",
    "control_heading_rate_loss_weight": "hr",
    "control_heading_rate_loss_scale_dps": "hr-scale",
    "control_bank_tv_loss_weight": "bank-tv",
    "control_state_objective": "obj",
    "control_state_loss_grid": "grid",
    "control_state_supervision_clock": "clock",
    "control_state_duration_gradient": "duration-grad",
    "kinematic_consistency_loss_weight": "kinematic",
    "state_endpoint_loss_weight": "endpoint",
    "terminal_loss_weight": "terminal",
    "fitted_tail_position_weight": "tail",
    "fitted_terminal_position_weight": "fitted-terminal",
    "final_time_loss_weight": "final-time",
    "duration_quantile_loss_weight": "pinball",
    "final_time_scale_s": "time-scale",
    "position_loss_scale_m": "pos-scale",
    "learning_rate": "lr",
    "batch_size": "batch",
    "e_layers": "layers",
    "n_segments": "N",
    "seq_len": "L",
    "aircraft_filter": "fleet",
    "aircraft_type": "type",
    "coordinate_frame": "frame",
    "target_conditioning": "target",
    "intent_conditioning": "intent",
    "cta_conditioning": "cta",
    "plan_conditioning": "plan",
    "plan_conditioning_dropout": "plan-drop",
    "control_condition_features": "airframe",
    "closure_labels_path": "labels",
    "control_fitted_teacher_path": "teacher",
    "plan_rolled_windows_path": "rolled",
    "plan_rolled_share": "rolled-share",
    "plan_fan_components": "fan",
    "state_position_reference": "pos-ref",
    "procedure_loss_lateral_weight": "proc-lat",
    "procedure_loss_vertical_weight": "proc-vert",
    "procedure_loss_dual_step": "proc-dual",
    "procedure_loss_epsilon": "proc-eps",
    "reference_velocity_source": "ref-vel",
    "checkpoint_selection_metric": "select",
    "validation_common_grid_points": "grid-points",
    "lr_plateau_metric": "lr-metric",
    "lr_plateau_patience": "lr-patience",
    "lr_plateau_factor": "lr-factor",
    "training_cohort_min_future_s": "min-future",
    "random_train_anchor_min_future_s": "anchor-min-future",
    "random_train_anchor": "random-anchor",
    "random_train_anchor_sampling": "anchors",
    "random_train_anchor_l1_share": "l1-share",
    "anchor_floor_index": "anchor-floor",
    "control_duration_parameterization": "duration",
    "control_duration_uniform_floor": "duration-floor",
    "control_gradient_clip_norm": "grad-clip",
    "control_rollout_integrator_dt_s": "rollout-dt",
    "control_horizon_s": "horizon",
    "control_command_hook": "hook",
    "control_hook_saturation": "hook-sat",
    "control_barrier_alpha": "barrier-alpha",
    "control_barrier_heading_gain": "barrier-gain",
    "control_speed_floor_margin": "floor-margin",
    "trombone_surplus_reference": "trombone-surplus",
}

#: The validation days of a day partition (the runway-intent R series, `experiments.runway_intent_r1`
#: `day_folds`), flown by a checkpoint trained and selected on that partition's training days only —
#: none of the checkpoint's own splits, and not the sealed outer test. Records under it come from a
#: runner (`runway_intent_r3 --write-records`), never from `predict`.
SPLIT_DAYVAL = "dayval"
#: Split prefixes shared by every category-label producer (publisher, pipeline,
#: relabeler). The wording is asserted by frontend fixtures — change with care.
SPLIT_DISPLAY = {
    "train": "Training split (in-sample)",
    "val": "Validation split (model selection)",
    "test": "Test split (held-out)",
    SPLIT_DAYVAL: "Held-out days (a day partition's validation days)",
}

_DEFAULTS: dict[str, Any] = TSConfig().to_dict()

#: The fields the output word reads for a latent run; guarded like every other field.
LATENT_OUTPUT_FIELDS = ("latent_dim", "latent_prior_components")

_unknown = [
    name
    for name in (*CONTROL_LOSS_FIELDS, *STATE_LOSS_FIELDS, *META_FIELDS,
                 *(field for field, _ in _TAU_FIELDS), *LATENT_OUTPUT_FIELDS)
    if name not in _DEFAULTS
]
if _unknown:  # fail at import: a renamed TSConfig field must rename here too
    raise AssertionError(f"run_naming references unknown TSConfig fields: {_unknown}")

#: Fields the grammar reads WITHOUT a naming list: the output and dynamics words, the
#: recipe, the horizon and the seed are spelled by their own functions below.
_DIRECTLY_NAMED_FIELDS = frozenset({
    "model", "prediction_output", "control_recipe_name", "horizon_mode", "seed",
    "control_dynamics_model", "control_dynamics_backend", "control_thrust_parameterization",
    "latent_dim", "latent_prior_components",
})

#: TSConfig fields that deliberately NAME NOTHING, each with its reason. The reverse of the
#: guard above: a CLI-settable field that no list carries lets two runs differing only in
#: it share a name and a slug (review C-3 — five such fields on 2026-09-09). A new field
#: goes into a naming list or, with its reason, here; never silently into neither.
KNOWN_UNNAMED_FIELDS: dict[str, str] = {
    # execution, not identity
    "device": "where a run executes changes no number",
    "notes": "free text, never identity",
    # the channel contract is fixed (channels.CHANNELS; load_checkpoint refuses a mismatch)
    "channels": "one contract, refused on mismatch at load",
    # backbone knobs no run has ever set off their default (recounted on disk 2026-09-09:
    # 219 stored runs); a first run that moves one must add it to META_FIELDS.
    "activation": "backbone knob, never set",
    "affine": "backbone knob, never set",
    "class_strategy": "backbone knob, never set",
    "decomposition": "backbone knob, never set",
    "embed": "backbone knob, never set",
    "factor": "backbone knob, never set",
    "fc_dropout": "backbone knob, never set",
    "freq": "backbone knob, never set",
    "head_dropout": "backbone knob, never set",
    "individual": "backbone knob, never set",
    "kernel_size": "backbone knob, never set",
    "output_attention": "backbone knob, never set",
    "padding_patch": "backbone knob, never set",
    "subtract_last": "backbone knob, never set",
}
_named = (
    set(CONTROL_LOSS_FIELDS) | set(STATE_LOSS_FIELDS) | set(CLOSURE_LOSS_FIELDS) | set(PLAN_LOSS_FIELDS)
    | set(META_FIELDS) | {field for field, _ in _TAU_FIELDS} | set(LATENT_OUTPUT_FIELDS)
    | _DIRECTLY_NAMED_FIELDS
)
_unlisted = sorted(set(_DEFAULTS) - _named - set(KNOWN_UNNAMED_FIELDS))
if _unlisted:  # fail at import: a new field must be named, or excused here by name
    raise AssertionError(
        f"TSConfig fields in no naming list and not in KNOWN_UNNAMED_FIELDS: {_unlisted}"
    )
_both = sorted(_named & set(KNOWN_UNNAMED_FIELDS))
if _both:
    raise AssertionError(f"fields both named and excused from naming: {_both}")

#: The structured view's sections for the META_FIELDS deviations (``run_parameter_rows``),
#: in display order. ``seed`` is not here: it is always shown, in the ``Model`` rows.
SETTING_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Architecture", (
        "d_model", "n_heads", "d_ff", "e_layers", "dropout", "patch_len", "stride",
        "n_segments", "seq_len", "dt_s", "full_horizon_steps", "window_horizon_steps",
        "use_norm", "revin", "duration_head",
    )),
    ("Training", (
        "epochs", "patience", "batch_size", "learning_rate", "weight_decay",
        "lr_plateau_metric", "lr_plateau_patience", "lr_plateau_factor",
        "checkpoint_selection_metric", "validation_common_grid_points",
        "control_gradient_clip_norm",
    )),
    ("Data & anchors", (
        "split_seed", "val_fraction", "test_fraction", "aircraft_filter", "aircraft_type",
        "coordinate_frame", "state_position_reference", "reference_velocity_source",
        "random_train_anchor", "random_train_anchor_sampling", "random_train_anchor_l1_share",
        "random_train_anchor_min_future_s", "training_cohort_min_future_s", "anchor_floor_index",
    )),
    ("Supervision sources", (
        "closure_labels_path", "control_fitted_teacher_path",
        "plan_rolled_windows_path", "plan_rolled_share",
    )),
    ("Conditioning", (
        "target_conditioning", *INTENT_FIELDS, *CTA_FIELDS, *PLAN_CONDITIONING_FIELDS,
        "control_condition_features",
    )),
    ("Control rollout", (
        "control_duration_parameterization", "control_duration_uniform_floor",
        "control_rollout_integrator_dt_s", "control_horizon_s", *CONTROL_HOOK_FIELDS,
    )),
)
_SECTION_OF = {field: section for section, fields in SETTING_SECTIONS for field in fields}
_sectioned = [field for _, fields in SETTING_SECTIONS for field in fields]
if len(_sectioned) != len(set(_sectioned)) or set(_sectioned) != set(META_FIELDS) - {"seed"}:
    # fail at import: a META field added without a section would vanish from the picker
    raise AssertionError(
        "SETTING_SECTIONS must place every META_FIELDS entry but seed exactly once: missing "
        f"{sorted(set(META_FIELDS) - {'seed'} - set(_sectioned))}, unknown "
        f"{sorted(set(_sectioned) - set(META_FIELDS))}"
    )


def _norm(value: Any) -> Any:
    return tuple(_norm(item) for item in value) if isinstance(value, (list, tuple)) else value


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "/".join(_fmt(item) for item in value)
    if value is None:
        return "none"
    return str(value)


def _abbrev(field: str) -> str:
    if field in _ABBREV:
        return _ABBREV[field]
    trimmed = field
    if trimmed.startswith("control_"):
        trimmed = trimmed[len("control_"):]
    for suffix in ("_loss_weight", "_weight", "_parameterization", "_loss"):
        if trimmed.endswith(suffix):
            trimmed = trimmed[: -len(suffix)]
            break
    return trimmed.replace("_", "-")


def _default_for(field: str, config: Mapping[str, Any]) -> Any:
    if field == "n_segments":
        model = config.get("model")
        return DEFAULT_N_SEGMENTS_BY_MODEL.get(model, _DEFAULTS[field])
    return _DEFAULTS[field]


def _field_diffs(
    config: Mapping[str, Any],
    fields: Sequence[str],
    *,
    exclude: frozenset[str] = frozenset(),
) -> list[tuple[str, Any]]:
    """(field, value) for every listed field present in ``config`` and non-default."""
    return [
        (field, config[field])
        for field in fields
        if field not in exclude
        and field in config
        and _norm(config[field]) != _norm(_default_for(field, config))
    ]


def _diff_items(diffs: list[tuple[str, Any]]) -> list[str]:
    return [f"{_abbrev(field)}={_display_value(field, value)}" for field, value in diffs]


def _display_value(field: str, value: Any) -> str:
    if field in _PATH_FIELDS:
        return _fmt_path(value)
    # A META_FIELDS value can be a list (`channels`-shaped fields), which is unhashable and
    # would raise on the dict lookup rather than falling through to `_fmt`.
    spellings = _VALUE_ABBREV.get(field, {})
    if spellings and isinstance(value, Hashable):
        return spellings.get(value) or _fmt(value)
    return _fmt(value)


def _fmt_path(value: Any) -> str:
    parts = str(value).replace("\\", "/").rstrip("/").split("/")
    return "/".join(parts[-2:])


def _diff_hash(diffs: list[tuple[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(dict(diffs), sort_keys=True, default=str).encode()
    ).hexdigest()[:8]


def _loss_diffs_against(
    config: Mapping[str, Any], baseline: Mapping[str, Any]
) -> list[tuple[str, Any]]:
    """Loss-field edits needed to turn ``baseline`` into this run.

    A field absent from an old config means the code had no such term — it reads as the
    dataclass default, so a recipe that overrides it shows up as an explicit edit.
    """
    diffs = []
    for field in CONTROL_LOSS_FIELDS:
        actual = config.get(field, _DEFAULTS[field])
        expected = baseline.get(field, _DEFAULTS[field])
        if _norm(actual) != _norm(expected):
            diffs.append((field, actual))
    return diffs


def _non_loss_recipe_mismatches(
    config: Mapping[str, Any], overrides: Mapping[str, Any]
) -> int:
    """How many fields a recipe freezes OUTSIDE the loss design this run disagrees with."""
    return sum(
        1
        for field, expected in overrides.items()
        if field not in CONTROL_LOSS_FIELDS
        and _norm(config.get(field, _DEFAULTS.get(field, expected))) != _norm(expected)
    )


def loss_design_parts(config: Mapping[str, Any]) -> tuple[str, list[tuple[str, Any]]]:
    """Field 4 decomposed: the base design and EVERY loss-field edit from it, unfolded.

    The base is the output's own versioned objective (``state-v1`` / ``closure-v1`` /
    ``plan-v1``), the named recipe, or — for a ``custom`` control run — its NEAREST recipe
    (``custom`` itself when no recipe is nearer than the plain defaults). ``loss_design_name``
    renders the name from this; the structured parameter view lists the edits in full.
    """
    if config.get("prediction_output") == PREDICTION_CLOSURE:
        return CLOSURE_LOSS_BASE, _field_diffs(config, CLOSURE_LOSS_FIELDS)
    if config.get("prediction_output") == PREDICTION_PLAN:
        return PLAN_LOSS_BASE, _field_diffs(config, PLAN_LOSS_FIELDS)
    if config.get("prediction_output") != PREDICTION_CONTROL:
        return STATE_LOSS_BASE, _field_diffs(config, STATE_LOSS_FIELDS)
    recipe = config.get("control_recipe_name") or CONTROL_RECIPE_CUSTOM
    if recipe != CONTROL_RECIPE_CUSTOM:
        # A named recipe freezes its own fields, but leaves later-added objective fields
        # (the final-approach penalty) open: a run that sets one is the recipe plus that
        # edit, and must not wear the bare name.
        return recipe, _loss_diffs_against(config, control_recipe_overrides(recipe))
    # Name the custom run against its nearest recipe: fewest loss-field edits wins, then
    # fewest edits among the NON-loss fields the recipe also freezes, then a later recipe
    # (CONTROL_RECIPE_NAMES is oldest→newest, custom first). The second key exists because
    # `simple-v1` and `simple-v1-lag` are the SAME loss design — they differ only in the
    # flight model — so on loss fields alone every simple-v1 run would wear the `-lag` name
    # next to a `point-mass` dynamics field.
    best_name, best_diffs = CONTROL_RECIPE_CUSTOM, _loss_diffs_against(config, {})
    best_rank = (len(best_diffs), 0)
    for candidate in CONTROL_RECIPE_NAMES:
        if candidate == CONTROL_RECIPE_CUSTOM:
            continue
        overrides = control_recipe_overrides(candidate)
        diffs = _loss_diffs_against(config, overrides)
        rank = (len(diffs), _non_loss_recipe_mismatches(config, overrides))
        if rank <= best_rank:
            best_name, best_diffs, best_rank = candidate, diffs, rank
    return best_name, best_diffs


def loss_design_name(config: Mapping[str, Any]) -> str:
    """Field 4: the named recipe, or nearest-recipe + edits, or a hash version."""
    base, diffs = loss_design_parts(config)
    if not diffs:
        return base
    if config.get("prediction_output") != PREDICTION_CONTROL:
        # state / closure / plan: the output's own objective, edits inline up to the cap.
        if len(diffs) <= _MAX_LISTED_DIFFS:
            return f"{base}({', '.join(_diff_items(diffs))})"
        return f"{base}-{_diff_hash(diffs)}"
    if (config.get("control_recipe_name") or CONTROL_RECIPE_CUSTOM) != CONTROL_RECIPE_CUSTOM:
        return f"{base}+({', '.join(_diff_items(diffs))})"
    if len(diffs) <= _MAX_LISTED_DIFFS:
        joined = ", ".join(_diff_items(diffs))
        if base == CONTROL_RECIPE_CUSTOM:
            return f"custom({joined})"
        return f"{base}+({joined})"
    # Too complex to spell out: a stable content version, hashed over the edits
    # relative to the plain defaults so the name is baseline-independent.
    return f"custom-{_diff_hash(_loss_diffs_against(config, {}))}"


def dynamics_name(config: Mapping[str, Any]) -> str:
    """Field 3: ``kinematic`` for state output, ``closed-form`` for closure, ``guidance``
    for the plan (the guidance layer on the lagged rollout); flight model (+τ, +backend)
    for control."""
    if config.get("prediction_output") == PREDICTION_CLOSURE:
        return "closed-form"
    if config.get("prediction_output") == PREDICTION_PLAN:
        return "guidance"
    if config.get("prediction_output") != PREDICTION_CONTROL:
        return "kinematic"
    model = config.get("control_dynamics_model") or CONTROL_DYNAMICS_POINT_MASS
    name = str(model)
    if model == CONTROL_DYNAMICS_FIRST_ORDER_LAG:
        taus = [
            f"{label}={_fmt(config[field])}s"
            for field, label in _TAU_FIELDS
            if field in config and _norm(config[field]) != _norm(_DEFAULTS[field])
        ]
        if taus:
            name += f"({', '.join(taus)})"
    thrust = _thrust_parameterization(config)
    if thrust != CONTROL_THRUST_FRACTION:
        name += f"+{thrust}"
    backend = config.get("control_dynamics_backend") or CONTROL_DYNAMICS_REANCHORED_RK4
    if backend != CONTROL_DYNAMICS_REANCHORED_RK4:
        name += f" @{backend}"
    return name


def _thrust_parameterization(config: Mapping[str, Any]) -> str:
    """A stored config predating the field ran the default law."""
    return config.get("control_thrust_parameterization") or CONTROL_THRUST_FRACTION


def _meta_diffs(
    config: Mapping[str, Any], *, include_recipe_frozen: bool = False
) -> list[tuple[str, Any]]:
    """The META_FIELDS deviations, in display-priority order.

    A named recipe's frozen fields are part of its name, so the NAME skips them; the
    structured rows (``include_recipe_frozen``) list them, because there the question is
    "what does this run use", not "what is special about it".
    """
    exclude: frozenset[str] = frozenset()
    if not include_recipe_frozen:
        recipe = config.get("control_recipe_name") or CONTROL_RECIPE_CUSTOM
        exclude = frozenset(control_recipe_overrides(recipe))
    diffs = _field_diffs(config, META_FIELDS, exclude=exclude)
    # split_seed defaults to None = "use seed"; recording it equal to seed is a spelling
    # of the default, not a deviation.
    return [
        (field, value)
        for field, value in diffs
        if not (field == "split_seed" and value == config.get("seed", _DEFAULTS["seed"]))
    ]


def meta_items(config: Mapping[str, Any]) -> list[str]:
    """Field 5, config-derived part: non-default horizon + non-default META_FIELDS.

    At most ``_MAX_LISTED_META`` field diffs are spelled out (the tuple order of
    ``META_FIELDS`` decides which); the remainder fold into ``+N more``.
    """
    items: list[str] = []
    horizon = config.get("horizon_mode")
    if horizon == HORIZON_FULL:
        items.append("full horizon")
    elif horizon == HORIZON_WINDOW:
        items.append("recursive window")
    diffs = _meta_diffs(config)
    listed = _diff_items(diffs[:_MAX_LISTED_META])
    if len(diffs) > _MAX_LISTED_META:
        listed.append(f"+{len(diffs) - _MAX_LISTED_META} more")
    items.extend(listed)
    return items


def dropped_meta_diffs(config: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """The META_FIELDS deviations `meta_items` folded into ``+N more``.

    The display name may fold them: a human reading a table wants six items, not twenty.
    A DIRECTORY name may not — two runs whose only difference is past the cut would share
    one, and the second would write over the first. `run_slug` therefore hashes these.
    ``CONTROL_HOOK_FIELDS`` are last in ``META_FIELDS``, so the hook's own knobs — the two
    barrier gains and, since 2026-09-08, ``control_speed_floor_margin`` behind them — are the
    first things to fold.
    """
    return _meta_diffs(config)[_MAX_LISTED_META:]


def output_name(config: Mapping[str, Any]) -> str:
    """Field 1: the output contract; a control output with a latent intent says so
    (``control+z8``, ``control+z8k4`` for a K=4 mixture prior) — a latent run is a
    different model from the deterministic head, not a loss edit on it."""
    output = str(config.get("prediction_output") or "state")
    latent_dim = int(config.get("latent_dim") or 0)
    if output != PREDICTION_CONTROL or latent_dim <= 0:
        return output
    components = int(config.get("latent_prior_components") or 1)
    return f"{output}+z{latent_dim}" + (f"k{components}" if components > 1 else "")


def run_display_name(config: Mapping[str, Any], *, extra: Sequence[str] = ()) -> str:
    """The canonical human name: output · backbone · dynamics · loss · meta."""
    backbone = str(config.get("model") or "?")
    parts = [
        output_name(config),
        _BACKBONE_DISPLAY.get(backbone, backbone),
        dynamics_name(config),
        loss_design_name(config),
    ]
    meta = [*meta_items(config), *extra]
    if meta:
        parts.append(", ".join(meta))
    return " · ".join(parts)


def run_parameter_rows(config: Mapping[str, Any]) -> list[dict[str, str]]:
    """The grammar in structured form: ``{section, name, value[, field]}`` rows, in order.

    ``Model`` names the grammar's parts (output, backbone, dynamics, loss design) plus the
    horizon and the seed, always. ``Loss edits vs <base>`` lists every loss-field edit from the
    loss design's base, and each setting section every non-default META field — ALL of them,
    recipe-frozen ones included: the display name folds past six items and hashes a long loss
    design, this view never does. A settings row's ``name`` IS the config field; a ``Model``
    row names the field it chiefly reads as ``field`` — none for the dynamics and the loss
    design, which are composed from several (flight model + τ + backend; recipe + edits).
    """
    backbone = str(config.get("model") or "?")
    model_rows: list[tuple[str, str, str | None]] = [
        ("Output", output_name(config), "prediction_output"),
        ("Backbone", _BACKBONE_DISPLAY.get(backbone, backbone), "model"),
        ("Dynamics", dynamics_name(config), None),
        ("Loss design", loss_design_name(config), None),
        ("Horizon", _fmt(config.get("horizon_mode", _DEFAULTS["horizon_mode"])), "horizon_mode"),
        ("Seed", _fmt(config.get("seed", _DEFAULTS["seed"])), "seed"),
    ]
    rows: list[dict[str, str]] = []
    for name, value, field in model_rows:
        row = {"section": "Model", "name": name, "value": value}
        if field is not None:
            row["field"] = field
        rows.append(row)
    base, loss_edits = loss_design_parts(config)
    section = f"Loss edits vs {base}"
    rows.extend(
        {"section": section, "name": field, "value": _display_value(field, value)}
        for field, value in loss_edits
    )
    settings = [
        (field, value)
        for field, value in _meta_diffs(config, include_recipe_frozen=True)
        if field != "seed"
    ]
    for section, _fields in SETTING_SECTIONS:
        rows.extend(
            {"section": section, "name": field, "value": _display_value(field, value)}
            for field, value in settings
            if _SECTION_OF[field] == section
        )
    return rows


def category_display_label(split: str, display_name: str, *, kind: str = "Predicted") -> str:
    """A comparison-category label: split prefix + kind + canonical run name. An unknown split raises: a
    label without its prefix would read as some other split's."""
    return f"{SPLIT_DISPLAY[split]} — {kind}: {display_name}"


def _slugify(text: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", text.lower().replace(".", "p")).strip("-")
    return token or "x"


def run_slug(config: Mapping[str, Any], *, extra: Sequence[str] = ()) -> str:
    """Filesystem form of the same grammar, for FUTURE run/category directories.

    Existing directories are historical record — never rename them to match.
    """
    backbone = str(config.get("model") or "x")
    model = (
        config.get("control_dynamics_model") or CONTROL_DYNAMICS_POINT_MASS
        if config.get("prediction_output") == PREDICTION_CONTROL
        else "kinematic"
    )
    dyn = _DYNAMICS_SLUG.get(model, _slugify(str(model)))
    thrust = _thrust_parameterization(config)
    if config.get("prediction_output") == PREDICTION_CONTROL and thrust != CONTROL_THRUST_FRACTION:
        # The control contract off its default is physics — WHICH quantities the rollout
        # integrates the head's columns as — so it sits in the always-shown dynamics word, never
        # in the foldable meta list; its word is the config scope row's.
        dyn += f"-{CONTROL_PARAMETERIZATION_SCOPES[thrust].slug}"
    backend = config.get("control_dynamics_backend") or CONTROL_DYNAMICS_REANCHORED_RK4
    if config.get("prediction_output") == PREDICTION_CONTROL and (
        backend != CONTROL_DYNAMICS_REANCHORED_RK4
    ):
        dyn += f"-{_BACKEND_SLUG.get(backend, _slugify(str(backend)))}"
    dropped = dropped_meta_diffs(config)
    tokens = [
        _slugify(output_name(config)),
        _BACKBONE_SLUG.get(backbone, _slugify(backbone)),
        dyn,
        _slugify(loss_design_name(config)),
        *(_slugify(item) for item in (*meta_items(config), *extra)),
        # The folded tail, as a hash: a slug names a FUTURE directory, and two runs that
        # differ only past `_MAX_LISTED_META` must not be handed the same one.
        *([_diff_hash(dropped)] if dropped else []),
    ]
    return "_".join(tokens)
