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

Names describe a config relative to TODAY'S defaults: when a default changes, old runs'
names gain (or lose) a meta item. That is deliberate — the name answers "what was
special about this run", and "special" is defined by the current baseline.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from config import (
    CONTROL_HOOK_FIELDS,
    INTENT_FIELDS,
    PROCEDURE_LOSS_FIELDS,
    CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    CONTROL_DYNAMICS_POINT_MASS,
    CONTROL_DYNAMICS_REANCHORED_RK4,
    CONTROL_RECIPE_CUSTOM,
    CONTROL_RECIPE_NAMES,
    DEFAULT_N_SEGMENTS_BY_MODEL,
    HORIZON_FULL,
    HORIZON_WINDOW,
    PREDICTION_CLOSURE,
    PREDICTION_CONTROL,
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
    "control_terminal_supervision_clock",
    "control_state_duration_gradient",
    "control_velocity_loss_weight",
    "control_velocity_loss_scale_mps",
    "control_imitation_loss_weight",
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
    "control_effort_loss_weight",
    "control_smoothness_loss_weight",
    "final_time_loss_weight",
    "final_time_scale_s",
    "position_loss_scale_m",
    # The final-approach penalty is an objective on BOTH paths (it acts on the control
    # rollout's segment endpoints too): a control run that carries it is a recipe edit.
    *PROCEDURE_LOSS_FIELDS,
)
#: The closure output's objective fields; its base name bumps when the regression
#: itself is redesigned.
CLOSURE_LOSS_BASE = "closure-v1"
CLOSURE_LOSS_FIELDS = (
    "closure_slowness_knots",
    "closure_height_knots",
    "closure_geometry_loss_weight",
    "closure_timing_loss_weight",
    "closure_timing_scale_s",
    "closure_height_loss_weight",
)
# Fields whose value is a path: rendered as the file's parent/name (two label generations
# in different directories must not read as one).
_PATH_FIELDS = frozenset({"closure_labels_path"})
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
    *INTENT_FIELDS,
    "d_model",
    "n_heads",
    "d_ff",
    "e_layers",
    "dropout",
    "patch_len",
    "stride",
    "n_segments",
    "seq_len",
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
    "corridor_gate",
    "reference_velocity_source",
    "use_norm",
    "revin",
    "random_train_anchor",
    "training_cohort_min_future_s",
    "checkpoint_selection_metric",
    "control_duration_parameterization",
    "control_duration_uniform_floor",
    "control_horizon_curriculum_s",
    "control_horizon_curriculum_stage_epochs",
    "control_gradient_clip_norm",
    "control_gradient_clip_policy",
    "control_rollout_integrator_dt_s",
    *CONTROL_HOOK_FIELDS,
)

_TAU_FIELDS = (
    ("control_thrust_time_constant_s", "τ-thrust"),
    ("control_bank_time_constant_s", "τ-bank"),
    ("control_load_time_constant_s", "τ-load"),
)

_ABBREV = {
    "control_imitation_loss_weight": "imit",
    "control_velocity_loss_weight": "vel",
    "control_velocity_loss_scale_mps": "vel-scale",
    "control_dense_state_loss_weight": "dense",
    "control_geometry_loss_weight": "geom",
    "control_effort_loss_weight": "effort",
    "control_smoothness_loss_weight": "smooth",
    "control_state_objective": "obj",
    "control_state_loss_grid": "grid",
    "control_state_supervision_clock": "clock",
    "control_terminal_supervision_clock": "terminal-clock",
    "control_state_duration_gradient": "duration-grad",
    "kinematic_consistency_loss_weight": "kinematic",
    "state_endpoint_loss_weight": "endpoint",
    "terminal_loss_weight": "terminal",
    "fitted_tail_position_weight": "tail",
    "fitted_terminal_position_weight": "fitted-terminal",
    "final_time_loss_weight": "final-time",
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
    "closure_labels_path": "labels",
    "state_position_reference": "pos-ref",
    "corridor_gate": "gate",
    "procedure_loss_lateral_weight": "proc-lat",
    "procedure_loss_vertical_weight": "proc-vert",
    "procedure_loss_dual_step": "proc-dual",
    "procedure_loss_epsilon": "proc-eps",
    "procedure_loss_lateral_scale_m": "proc-lat-scale",
    "procedure_loss_vertical_scale_m": "proc-vert-scale",
    "reference_velocity_source": "ref-vel",
    "checkpoint_selection_metric": "select",
    "training_cohort_min_future_s": "min-future",
    "random_train_anchor": "random-anchor",
    "control_duration_parameterization": "duration",
    "control_duration_uniform_floor": "duration-floor",
    "control_horizon_curriculum_s": "curriculum",
    "control_horizon_curriculum_stage_epochs": "curriculum-epochs",
    "control_gradient_clip_norm": "grad-clip",
    "control_gradient_clip_policy": "grad-clip-policy",
    "control_rollout_integrator_dt_s": "rollout-dt",
    "control_command_hook": "hook",
    "control_hook_gate": "hook-gate",
    "control_hook_saturation": "hook-sat",
    "control_barrier_alpha": "barrier-alpha",
    "control_barrier_heading_gain": "barrier-gain",
    "control_nominal_l1_distance_m": "l1",
    "control_nominal_vertical_lookahead_m": "vert-look",
    "control_nominal_vertical_gain": "vert-gain",
    "control_nominal_residual_bank_max_rad": "res-bank",
    "control_nominal_residual_load_max": "res-load",
    "control_nominal_speed_gain": "speed-gain",
}

#: Split prefixes shared by every category-label producer (publisher, pipeline,
#: relabeler). The wording is asserted by frontend fixtures — change with care.
SPLIT_DISPLAY = {
    "train": "Training split (in-sample)",
    "val": "Validation split (model selection)",
    "test": "Test split (held-out)",
}

_DEFAULTS: dict[str, Any] = TSConfig().to_dict()

_unknown = [
    name
    for name in (*CONTROL_LOSS_FIELDS, *STATE_LOSS_FIELDS, *META_FIELDS,
                 *(field for field, _ in _TAU_FIELDS))
    if name not in _DEFAULTS
]
if _unknown:  # fail at import: a renamed TSConfig field must rename here too
    raise AssertionError(f"run_naming references unknown TSConfig fields: {_unknown}")


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
    return [f"{_abbrev(field)}={_fmt_path(value) if field in _PATH_FIELDS else _fmt(value)}"
            for field, value in diffs]


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


def loss_design_name(config: Mapping[str, Any]) -> str:
    """Field 4: the named recipe, or nearest-recipe + edits, or a hash version."""
    if config.get("prediction_output") == PREDICTION_CLOSURE:
        return _with_diffs(config, CLOSURE_LOSS_FIELDS, CLOSURE_LOSS_BASE)
    if config.get("prediction_output") != PREDICTION_CONTROL:
        return _with_state_diffs(config)
    recipe = config.get("control_recipe_name") or CONTROL_RECIPE_CUSTOM
    if recipe != CONTROL_RECIPE_CUSTOM:
        # A named recipe freezes its own fields, but leaves later-added objective fields
        # (the final-approach penalty) open: a run that sets one is the recipe plus that
        # edit, and must not wear the bare name.
        edits = _loss_diffs_against(config, control_recipe_overrides(recipe))
        if not edits:
            return recipe
        return f"{recipe}+({', '.join(_diff_items(edits))})"
    # Name the custom run against its nearest recipe: fewest loss-field edits wins,
    # a later recipe wins ties (CONTROL_RECIPE_NAMES is oldest→newest, custom first).
    best_name, best_diffs = CONTROL_RECIPE_CUSTOM, _loss_diffs_against(config, {})
    for candidate in CONTROL_RECIPE_NAMES:
        if candidate == CONTROL_RECIPE_CUSTOM:
            continue
        diffs = _loss_diffs_against(config, control_recipe_overrides(candidate))
        if len(diffs) <= len(best_diffs):
            best_name, best_diffs = candidate, diffs
    if not best_diffs:
        return best_name
    if len(best_diffs) <= _MAX_LISTED_DIFFS:
        joined = ", ".join(_diff_items(best_diffs))
        if best_name == CONTROL_RECIPE_CUSTOM:
            return f"custom({joined})"
        return f"{best_name}+({joined})"
    # Too complex to spell out: a stable content version, hashed over the edits
    # relative to the plain defaults so the name is baseline-independent.
    return f"custom-{_diff_hash(_loss_diffs_against(config, {}))}"


def _with_state_diffs(config: Mapping[str, Any]) -> str:
    return _with_diffs(config, STATE_LOSS_FIELDS, STATE_LOSS_BASE)


def _with_diffs(config: Mapping[str, Any], fields: tuple[str, ...], base: str) -> str:
    diffs = _field_diffs(config, fields)
    if not diffs:
        return base
    if len(diffs) <= _MAX_LISTED_DIFFS:
        return f"{base}({', '.join(_diff_items(diffs))})"
    return f"{base}-{_diff_hash(diffs)}"


def dynamics_name(config: Mapping[str, Any]) -> str:
    """Field 3: ``kinematic`` for state output, ``closed-form`` for closure; flight model
    (+τ, +backend) for control."""
    if config.get("prediction_output") == PREDICTION_CLOSURE:
        return "closed-form"
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
    backend = config.get("control_dynamics_backend") or CONTROL_DYNAMICS_REANCHORED_RK4
    if backend != CONTROL_DYNAMICS_REANCHORED_RK4:
        name += f" @{backend}"
    return name


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
    recipe = config.get("control_recipe_name") or CONTROL_RECIPE_CUSTOM
    frozen = frozenset(control_recipe_overrides(recipe))
    diffs = _field_diffs(config, META_FIELDS, exclude=frozen)
    # split_seed defaults to None = "use seed"; recording it equal to seed is a spelling
    # of the default, not a deviation.
    diffs = [
        (field, value)
        for field, value in diffs
        if not (field == "split_seed" and value == config.get("seed", _DEFAULTS["seed"]))
    ]
    listed = _diff_items(diffs[:_MAX_LISTED_META])
    if len(diffs) > _MAX_LISTED_META:
        listed.append(f"+{len(diffs) - _MAX_LISTED_META} more")
    items.extend(listed)
    return items


def run_display_name(config: Mapping[str, Any], *, extra: Sequence[str] = ()) -> str:
    """The canonical human name: output · backbone · dynamics · loss · meta."""
    backbone = str(config.get("model") or "?")
    parts = [
        str(config.get("prediction_output") or "state"),
        _BACKBONE_DISPLAY.get(backbone, backbone),
        dynamics_name(config),
        loss_design_name(config),
    ]
    meta = [*meta_items(config), *extra]
    if meta:
        parts.append(", ".join(meta))
    return " · ".join(parts)


def category_display_label(split: str, display_name: str, *, kind: str = "Predicted") -> str:
    """A comparison-category label: split prefix + kind + canonical run name."""
    prefix = SPLIT_DISPLAY.get(split)
    head = f"{kind}: {display_name}"
    return f"{prefix} — {head}" if prefix else head


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
    backend = config.get("control_dynamics_backend") or CONTROL_DYNAMICS_REANCHORED_RK4
    if config.get("prediction_output") == PREDICTION_CONTROL and (
        backend != CONTROL_DYNAMICS_REANCHORED_RK4
    ):
        dyn += f"-{_BACKEND_SLUG.get(backend, _slugify(str(backend)))}"
    tokens = [
        str(config.get("prediction_output") or "state"),
        _BACKBONE_SLUG.get(backbone, _slugify(backbone)),
        dyn,
        _slugify(loss_design_name(config)),
        *(_slugify(item) for item in (*meta_items(config), *extra)),
    ]
    return "_".join(tokens)
