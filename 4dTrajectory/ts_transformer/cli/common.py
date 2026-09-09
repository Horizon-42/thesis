"""Argument groups, config assembly and cohort loading shared by the subcommands.

The one thing worth stating up front: **an argparse flag is named after the ``TSConfig``
field it sets**, so ``CLI_CONFIG_FIELDS`` is a list of field names rather than a
hand-written flag→field mapping, and the assertion under it fails at import if a field is
renamed in ``config.py`` and not here. Fifteen flags were renamed on 2026-09-07 to make
that true; the two remaining exceptions are named where they occur.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from ts_transformer.config import (
    AIRCRAFT_FILTER_OPENAP_DIRECT,
    AIRCRAFT_FILTERS,
    CHECKPOINT_SELECTION_METRICS,
    CONTROL_DYNAMICS_BACKENDS,
    CONTROL_DYNAMICS_MODELS,
    CONTROL_DURATION_PARAMETERIZATIONS,
    CONTROL_HOOKS_AVAILABLE,
    CTA_CONDITIONINGS_AVAILABLE,
    CONTROL_HOOK_FIELDS,
    CONTROL_IMITATION_TARGETS,
    CONTROL_RECIPE_CUSTOM,
    CONTROL_RECIPE_NAMES,
    CONTROL_RECIPE_SIMPLE_V1_LAG,
    CONTROL_STATE_CLOCKS,
    CONTROL_STATE_LOSS_GRIDS_AVAILABLE,
    CONTROL_STATE_OBJECTIVES,
    COORDINATE_FRAMES_AVAILABLE,
    RETIRED_CONSTANT_FIELDS,
    RETIRED_SERIALIZED_FIELDS,
    DEFAULT_AIRCRAFT_TYPE,
    HORIZON_MODES,
    INTENT_CONDITIONINGS_AVAILABLE,
    INTENT_FIELDS,
    LR_PLATEAU_METRICS,
    MODELS,
    PREDICTION_OUTPUTS_AVAILABLE,
    PROCEDURE_LOSS_FIELDS,
    RANDOM_TRAIN_ANCHOR_SAMPLINGS,
    STATE_POSITION_REFERENCES_AVAILABLE,
    TARGET_CONDITIONINGS,
    TIME_CONSTANT_FIELDS,
    TSConfig,
    coerce_sequence_fields,
    control_recipe_overrides,
)
from ts_transformer.cross_validation import validate_cv_parameters
from ts_transformer.data_provenance import arrival_data_provenance, eligibility_sources
from ts_transformer.dataset import build_series, load_flight_dicts
from ts_transformer.development_cohorts import development_cohort_audit, load_development_cohort
from ts_transformer.experiment_index import begin_run, finish_run
from ts_transformer.reference_velocity import REFERENCE_VELOCITY_SOURCES
from ts_transformer.splits import data_selection_audit, flight_keys_by_split

#: The repo root, for the experiment manifest's provenance.
REPO_ROOT = Path(__file__).resolve().parents[3]


def add_eligibility_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--eligibility-roster",
        action="append",
        default=None,
        help=(
            "pre-split flight eligibility JSON; repeat once per --data manifest. "
            "The top-level pipeline supplies evaluation-derived lateral-pass rosters."
        ),
    )


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data", required=True, action="append",
        help="airport harvest directory or arrivals/manifest.json; repeat for pooled training",
    )
    add_eligibility_arg(parser)
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
    parser.add_argument("--output-dir", required=True, type=Path)


def refuse_airport_override_for_pooled_data(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """``--airport`` re-homes every flight that carries no ``arr_airport``; over several
    manifests it would put another airport's flights on this one's thresholds. Train and
    cross-validate always refused it; predict did not (review C-19)."""
    if len(args.data) > 1 and args.airport:
        parser.error("--airport cannot override flights when multiple --data manifests are used")


def provenance_from_args(args: argparse.Namespace) -> dict[str, object]:
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


def cv_parameters_type(value: str) -> tuple[str, ...]:
    try:
        return validate_cv_parameters(
            name.strip() for name in value.split(",") if name.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--control-recipe-name",
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
        choices=PREDICTION_OUTPUTS_AVAILABLE,
        default=None,
        help="predict state endpoints (default) or bounded controls with dynamics rollout "
             "(the closure output is frozen: its checkpoints load, no new run trains it)",
    )
    parser.add_argument(
        "--closure-labels-path", default=None, metavar="JSON",
        help="closure output: the per-flight labels written by docs/p1_closure_oracle.py labels",
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
    parser.add_argument("--dt-s", type=float, default=None, help="resample step, seconds")
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
    parser.add_argument(
        "--lr-plateau-metric",
        choices=LR_PLATEAU_METRICS,
        default=None,
        help=(
            "which validation number the LR scheduler measures its plateau on: the "
            "checkpoint-selection value ('selection', the default) or the macro validation "
            "objective ('objective'). Never changes which epoch is kept"
        ),
    )
    parser.add_argument("--fitted-tail-position-weight", type=float, default=None,
                        help="position-only weight for fitted ADS-B tail rows (default: 0.25)")
    parser.add_argument("--fitted-terminal-position-weight", type=float, default=None,
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
    parser.add_argument("--kinematic-consistency-loss-weight", type=float, default=None,
                        help="control-path compatibility weight; direct state ignores it")
    parser.add_argument("--terminal-loss-weight", type=float, default=None,
                        help="control-path compatibility weight; direct state ignores it")
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
    parser.add_argument("--control-thrust-time-constant-s", type=float, default=None)
    parser.add_argument(
        "--control-bank-time-constant-s",
        type=float,
        default=None,
        help="bank-angle time constant; the lagged model's swept parameter",
    )
    parser.add_argument("--control-load-time-constant-s", type=float, default=None)
    parser.add_argument(
        "--control-state-supervision-clock",
        choices=CONTROL_STATE_CLOCKS,
        default=None,
        help=(
            "total duration used for control state-loss rollout: model prediction "
            "(default) or observed train/validation duration"
        ),
    )
    parser.add_argument(
        "--control-state-loss-grid",
        choices=CONTROL_STATE_LOSS_GRIDS_AVAILABLE,
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
        "--control-imitation-target",
        choices=CONTROL_IMITATION_TARGETS,
        default=None,
        help=(
            "what the imitation term imitates: the schedule inverted out of the flown "
            "track (default), or a per-flight table fitted through the rollout "
            "(--control-fitted-teacher-path). The inverted schedule flown open-loop lands "
            "2.5-7.8 km from the truth it was read off; the fitted one lands 88-433 m"
        ),
    )
    parser.add_argument(
        "--control-fitted-teacher-path", default=None, metavar="JSON",
        help=(
            "--control-imitation-target fitted: the basis_fit.json written by "
            "run_ts_control_basis_oracle.py --checkpoint. The dataset build refuses a "
            "table whose width, anchor or per-flight duration is not this run's"
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
        "--control-heading-rate-loss-weight",
        type=float,
        default=None,
        help=(
            "weight of the heading-rate term: the rollout's own turn rate at the segment "
            "endpoints against the flown track's (default: 0, off). Built by the "
            "true-time-position objective only"
        ),
    )
    parser.add_argument(
        "--control-heading-rate-loss-scale-dps",
        type=float,
        default=None,
        help=(
            "deg/s the heading-rate residual is read in — the unit, not the dose "
            "(default: 1.5, half a standard-rate turn)"
        ),
    )
    parser.add_argument(
        "--control-bank-tv-loss-weight",
        type=float,
        default=None,
        help=(
            "weight of the COMMANDED bank's total variation: mean |step| between adjacent "
            "segments, in half-box units (default: 0, off)"
        ),
    )
    # The adopted command hook's two gains. Design doc §三.8 asks for them in the
    # checkpoint; until 2026-09-07 the only way to set either was a hand-written
    # --config-overrides JSON, which is how six 2026-09-06 configs ended up carrying the
    # first campaign's heading gain.
    parser.add_argument(
        "--control-barrier-alpha",
        type=float,
        default=None,
        help=(
            "barrier command hook: the class-K gain on the corridor margins, capped by the "
            "segment's own 1/hold rate (default: 0.1). A held command cannot close more "
            "than the whole margin in one hold, so the cap is not a taste knob"
        ),
    )
    parser.add_argument(
        "--control-barrier-heading-gain",
        type=float,
        default=None,
        help=(
            "barrier command hook: the gain on the heading-alignment barrier (default: "
            "0.1; the first 2026-09-06 campaign ran at 0.3 and limit-cycled on 7 s holds)"
        ),
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
        "--control-rollout-integrator-dt-s",
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
    parser.add_argument("--coordinate-frame", choices=COORDINATE_FRAMES_AVAILABLE, default=None)
    parser.add_argument(
        "--state-position-reference",
        choices=STATE_POSITION_REFERENCES_AVAILABLE,
        default=None,
        help=(
            "state output only: 'corridor-bounded' binds the position channels to the "
            "final-approach corridor on the rows the output places on the final; default: "
            "absolute chart coordinates. ('anchor-relative' is vetoed — a stored config may "
            "carry it, a new run may not select it.)"
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
        "--intent-conditioning",
        choices=INTENT_CONDITIONINGS_AVAILABLE,
        default=None,
        help=(
            "Phase 0 intent upper bound: feed the TRUTH join point ('truth-join'), or that "
            "plus the lead's TRUE landing time relative to the anchor ('truth-join-lead'), "
            "or plus the flight's TRUE remaining time ('truth-join-duration'), as input-only "
            "constant channels (iTransformer only). Reads the future — a development "
            "measurement, never a deployable model; default: none"
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
        "--random-train-anchor-sampling",
        choices=RANDOM_TRAIN_ANCHOR_SAMPLINGS,
        default=None,
        help=(
            "how that anchor is drawn from the admissible ones: uniformly over the samples "
            "('uniform', the default, which is uniform in TIME and, pooled over flights, "
            "over-weights the near end) or uniformly across the flight's own remaining-path "
            "span ('remaining-path-uniform'). Refused without --random-train-anchor"
        ),
    )
    parser.add_argument(
        "--random-train-anchor-l1-share",
        type=float,
        default=None,
        help=(
            "share of the per-flight per-epoch draws RESERVED for the L-1 anchor the "
            "fixed-anchor arms train at, mixed into the remaining-path-uniform law "
            "(default: 0, the pure law). Refused under --random-train-anchor-sampling "
            "uniform and without --random-train-anchor"
        ),
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


#: Every ``TSConfig`` field a training flag can set, spelled exactly as the field — which
#: is also exactly the flag, minus ``--`` and with dashes for underscores. Two fields are
#: not in the list and cannot be: ``batch_size`` (the flag also accepts ``"auto"``, which is
#: not a config value) and ``use_norm`` / ``revin`` (one ``--instance-norm`` flag, two
#: backbone-specific fields). ``control_recipe_name`` is a third exception in the other
#: direction: its flag IS its name, but it has to be resolved against
#: ``--config-overrides`` before the rest, so ``config_from_args`` sets it explicitly.
CLI_CONFIG_FIELDS = (
    "model",
    "prediction_output",
    "closure_labels_path",
    "seq_len",
    "n_segments",
    "horizon_mode",
    "full_horizon_steps",
    "window_horizon_steps",
    "dt_s",
    "epochs",
    "reference_velocity_source",
    "learning_rate",
    "lr_plateau_factor",
    "lr_plateau_patience",
    "lr_plateau_metric",
    "patience",
    "fitted_tail_position_weight",
    "fitted_terminal_position_weight",
    "state_endpoint_loss_weight",
    "kinematic_consistency_loss_weight",
    "terminal_loss_weight",
    "control_duration_parameterization",
    "control_duration_uniform_floor",
    "control_dynamics_backend",
    "control_dynamics_model",
    "control_thrust_time_constant_s",
    "control_bank_time_constant_s",
    "control_load_time_constant_s",
    "control_state_supervision_clock",
    "control_state_loss_grid",
    "control_state_objective",
    "control_imitation_target",
    "control_fitted_teacher_path",
    "control_state_duration_gradient",
    "control_heading_rate_loss_weight",
    "control_heading_rate_loss_scale_dps",
    "control_bank_tv_loss_weight",
    "control_gradient_clip_norm",
    "control_barrier_alpha",
    "control_barrier_heading_gain",
    "control_rollout_integrator_dt_s",
    "d_model",
    "e_layers",
    "n_heads",
    "seed",
    "split_seed",
    "device",
    "aircraft_type",
    "coordinate_frame",
    "target_conditioning",
    "intent_conditioning",
    "state_position_reference",
    "aircraft_filter",
    "random_train_anchor",
    "training_cohort_min_future_s",
    "random_train_anchor_min_future_s",
    "random_train_anchor_sampling",
    "random_train_anchor_l1_share",
    "checkpoint_selection_metric",
    "validation_common_grid_points",
)

_TSCONFIG_FIELDS = {field.name for field in fields(TSConfig)}
_unknown = [name for name in CLI_CONFIG_FIELDS if name not in _TSCONFIG_FIELDS]
if _unknown:  # fail at import: a renamed TSConfig field must rename its flag too
    raise AssertionError(f"the CLI sets unknown TSConfig fields: {_unknown}")


#: (field, what a NEW run may select, why the rest are there). A stored config may carry any
#: value the vocabulary allows — `TSConfig.from_dict` and `load_checkpoint` must keep working
#: on artifacts trained under a value that has since been archived or vetoed. Selecting one
#: for a NEW run is a different act, and this is where it is refused.
_NEW_RUN_VOCABULARIES = (
    ("cta_conditioning", CTA_CONDITIONINGS_AVAILABLE,
     "`self-q` is what `predict --cta-from-quantiles` stamps on the config it writes beside "
     "the records (B3) — the CTA is the model's own quantile, decided at prediction time; "
     "there is nothing to train under it"),
    ("control_command_hook", CONTROL_HOOKS_AVAILABLE,
     "the nominal-law hook is archived (archive/nominal_law_hook_2026_09/); its numbers are "
     "in docs/2026-09-06_control_hooks_results.zh.md"),
    ("state_position_reference", STATE_POSITION_REFERENCES_AVAILABLE,
     "anchor-relative was VETOED by the 2026-09-03 state-v2 campaign's own pre-registered "
     "rule; the value exists so that campaign's artifact still loads"),
    # Frozen 2026-09-09 (package review §5): axes with published numbers whose checkpoints
    # keep loading, predicting and publishing, and which no new run trains.
    ("prediction_output", PREDICTION_OUTPUTS_AVAILABLE,
     "the closure output is a comparison arm (2026-09-05/06) whose tracker was deleted; "
     "its two stored runs load and predict"),
    ("intent_conditioning", INTENT_CONDITIONINGS_AVAILABLE,
     "the truth-join oracles were the scene design's Phase 0 instrument; the L4 gate failed "
     "and the scene encoder is archived (archive/scene_encoder_2026_09/)"),
    ("control_state_loss_grid", CONTROL_STATE_LOSS_GRIDS_AVAILABLE,
     "fixed-dt is the 2026-08 arm family that trips the straight-in veto without the "
     "imitation term (which is not registered on it); the named recipes pin the native grid"),
    ("coordinate_frame", COORDINATE_FRAMES_AVAILABLE,
     "the 2026-09-03 frame ablation kept the threshold-anchored ENU chart (the airport frame "
     "averages across parallel pairs); its arms load"),
)


def _refuse_unavailable_selection(config: TSConfig, parser: argparse.ArgumentParser) -> None:
    """Refuse a value a stored config may carry but a new run may not select.

    The flags' own ``choices`` already refuse these; ``--config-overrides`` is the second
    door into the same fields, and without this the run gets a dataset build and a formal
    experiment manifest (``begin_run``) before the training loop dies on it.
    """
    for field, available, why in _NEW_RUN_VOCABULARIES:
        value = getattr(config, field)
        if value not in available:
            parser.error(
                f"{field}={value!r} cannot be selected for a new run: {why}. "
                f"Available: {', '.join(available)}"
            )


def config_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[TSConfig, bool]:
    overrides: dict[str, object] = {}
    if args.config_overrides:
        try:
            loaded = json.loads(Path(args.config_overrides).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read --config-overrides: {exc}")
        if not isinstance(loaded, dict):
            parser.error("--config-overrides must be a JSON object of TSConfig fields")
        allowed = {field.name for field in fields(TSConfig)}
        unknown = sorted(key for key in loaded if key not in allowed)
        if unknown:
            retired = [key for key in unknown
                       if key in RETIRED_SERIALIZED_FIELDS or key in RETIRED_CONSTANT_FIELDS]
            why = (
                f" ({', '.join(retired)} was retired from the contract; a stored config may "
                "still carry it, a new run may not set it)"
                if retired else ""
            )
            parser.error(
                f"--config-overrides names {len(unknown)} field(s) TSConfig does not have: "
                f"{', '.join(unknown)}{why}"
            )
        # Tuples came back as lists; the frozen-recipe comparison below is on raw values.
        overrides.update(coerce_sequence_fields(loaded))

    batch_auto = args.batch_size == "auto"
    overrides.update({
        name: value
        for name in CLI_CONFIG_FIELDS
        if (value := getattr(args, name)) is not None
    })
    # The three fields whose flag is not simply their name (see CLI_CONFIG_FIELDS).
    if isinstance(args.batch_size, int):
        overrides["batch_size"] = args.batch_size
    if args.instance_norm is not None:
        overrides["use_norm"] = overrides["revin"] = args.instance_norm

    override_recipe = overrides.get("control_recipe_name")
    if (
        args.control_recipe_name is not None
        and override_recipe is not None
        and args.control_recipe_name != override_recipe
    ):
        parser.error(
            "--control-recipe-name conflicts with control_recipe_name in --config-overrides"
        )
    requested_recipe = args.control_recipe_name or override_recipe or CONTROL_RECIPE_CUSTOM
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
        # …and the final-approach penalty, the command hooks and the intent axis, which
        # every recipe leaves open.
        open_fields = (
            set(PROCEDURE_LOSS_FIELDS) | set(CONTROL_HOOK_FIELDS) | set(INTENT_FIELDS) | (
                set(TIME_CONSTANT_FIELDS)
                if requested_recipe == CONTROL_RECIPE_SIMPLE_V1_LAG
                else set()
            )
        )
        unsupported = sorted(
            set(overrides) - set(frozen) - runtime_fields - open_fields
        )
        if unsupported:
            parser.error(
                f"{requested_recipe} allows only seed, split_seed, device, run identity, "
                f"the final-approach penalty, command-hook and intent fields outside its "
                f"frozen fields; unsupported override {unsupported[0]!r}"
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
    config = TSConfig(**overrides)
    _refuse_unavailable_selection(config, parser)
    return config, batch_auto


def build_series_or_exit(args: argparse.Namespace, config: TSConfig,
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


@dataclass(frozen=True)
class TrainingRun:
    """Everything ``train`` and ``cross-validate`` assemble identically before they part.

    They share the cohort: the same eligibility, the same by-flight split, the same
    development roster, the same audit and the same experiment manifest. Only what they do
    with `series` differs, so only that is in the two command modules.
    """

    config: TSConfig
    batch_auto: bool
    series: list
    data_provenance: dict[str, Any]
    outer_split_keys: dict[str, list[str]]
    data_selection: dict[str, Any]
    development_cohort: Any | None
    experiment_manifest: Any | None

    def finish(self, failure: BaseException | None = None) -> None:
        """Close the formal experiment manifest, if this run opened one."""
        if self.experiment_manifest is None:
            return
        if failure is None:
            finish_run(self.experiment_manifest)
        else:
            finish_run(
                self.experiment_manifest,
                failure=f"{type(failure).__name__}: {failure}",
            )


def add_training_run_arguments(parser: argparse.ArgumentParser, *, cohort_help: str) -> None:
    """The data, config and development-cohort flags shared by train and cross-validate."""
    add_data_args(parser)
    add_training_args(parser)
    parser.add_argument("--development-cohort", default=None, help=cohort_help)


def prepare_training_run(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    argv: list[str] | None,
) -> TrainingRun:
    """Resolve the config, load the development cohort and open the experiment manifest."""
    refuse_airport_override_for_pooled_data(args, parser)
    config, batch_auto = config_from_args(args, parser)
    if bool(args.campaign_id) != bool(args.experiment_id):
        parser.error("--campaign-id and --experiment-id must be supplied together")
    data_provenance = provenance_from_args(args)
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
    series, build_report = build_series_or_exit(
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
    # The compared identity (the eligible SET) plus the roster's byte facts, which are
    # recorded HERE and nowhere that is compared: `sources.evaluation_report_sha256` moves
    # whenever the observed evaluation is regenerated over an identical eligible set.
    roster_sources = {
        source["airport"]: source
        for source in eligibility_sources(getattr(args, "eligibility_roster", None))
    }
    data_selection["pre_split_eligibility"] = [
        {
            "airport": entry["airport"],
            "arrival_candidates": entry.get("arrival_candidate_count"),
            **(
                entry["eligibility"]
                if isinstance(entry.get("eligibility"), dict)
                else {"policy": "none"}
            ),
            **(
                {"roster_sources": roster_sources[entry["airport"]]}
                if entry["airport"] in roster_sources
                else {}
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
            repo_root=REPO_ROOT,
            campaign_id=args.campaign_id,
            run_id=args.experiment_id,
            config=config.to_dict(),
            data_selection=data_selection,
            command=sys.argv if argv is None else ["ts_transformer", *argv],
        )

    return TrainingRun(
        config=config,
        batch_auto=batch_auto,
        series=series,
        data_provenance=data_provenance,
        outer_split_keys=outer_split_keys,
        data_selection=data_selection,
        development_cohort=development_cohort,
        experiment_manifest=experiment_manifest,
    )
