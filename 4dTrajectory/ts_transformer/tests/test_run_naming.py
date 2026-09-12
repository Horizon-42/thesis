"""The canonical run-name grammar: output · backbone · dynamics · loss · meta."""

import sys
from pathlib import Path

_TS_DIR = Path(__file__).resolve().parents[1]
if str(_TS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_TS_DIR.parent))

from ts_transformer.config import (  # noqa: E402
    CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    CONTROL_RECIPE_SIMPLE_V1,
    CONTROL_RECIPE_SIMPLE_V1_LAG,
    CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    CONTROL_RECIPE_SIMPLE_V2,
    CONTROL_RECIPE_SIMPLE_V3,
    TSConfig,
    control_recipe_overrides,
)
from ts_transformer.run_naming import (  # noqa: E402
    CLOSURE_LOSS_FIELDS,
    CONTROL_LOSS_FIELDS,
    META_FIELDS,
    STATE_LOSS_FIELDS,
    category_display_label,
    dynamics_name,
    loss_design_name,
    run_display_name,
    run_slug,
)


def _state_defaults() -> dict:
    return TSConfig().to_dict()


def _control_config(**overrides) -> dict:
    config = TSConfig().to_dict()
    config["prediction_output"] = "control"
    config.update(overrides)
    return config


def test_every_named_field_exists_on_tsconfig():
    fields = set(TSConfig().to_dict())
    for group in (CONTROL_LOSS_FIELDS, STATE_LOSS_FIELDS, CLOSURE_LOSS_FIELDS, META_FIELDS):
        assert set(group) <= fields


def test_default_state_run_has_no_meta_tail():
    assert run_display_name(_state_defaults()) == (
        "state · iTransformer · kinematic · state-v1"
    )


def test_state_horizon_and_deviations_land_in_meta():
    config = _state_defaults()
    config["horizon_mode"] = "full"
    config["seed"] = 2024
    name = run_display_name(config)
    assert "full horizon" in name
    assert "seed=2024" in name


def test_frame_and_target_conditioning_land_in_meta():
    config = _state_defaults()
    config["coordinate_frame"] = "airport-enu"
    config["target_conditioning"] = "channels"
    name = run_display_name(config)
    assert "frame=airport-enu" in name and "target=channels" in name
    assert run_slug(config).endswith("_frame-airport-enu_target-channels")
    config = _state_defaults()
    config["state_position_reference"] = "anchor-relative"
    assert "pos-ref=anchor-relative" in run_display_name(config)


def test_named_recipe_is_the_loss_field():
    config = _control_config(control_recipe_name=CONTROL_RECIPE_SIMPLE_V3)
    config.update(control_recipe_overrides(CONTROL_RECIPE_SIMPLE_V3))
    assert loss_design_name(config) == "simple-v3"


def test_custom_run_is_named_against_its_nearest_recipe():
    # simple-v3 content trained as a CV candidate: recipe field says custom, the loss
    # fields say v3 with one edited weight.
    config = _control_config(control_recipe_name="custom")
    config.update(control_recipe_overrides(CONTROL_RECIPE_SIMPLE_V3))
    config["control_recipe_name"] = "custom"
    config["control_imitation_loss_weight"] = 16.0
    assert loss_design_name(config) == "simple-v3+(imit=16)"

    config["control_imitation_loss_weight"] = control_recipe_overrides(
        CONTROL_RECIPE_SIMPLE_V3
    )["control_imitation_loss_weight"]
    assert loss_design_name(config) == "simple-v3"


def test_a_loss_field_tie_is_broken_by_the_flight_model_the_recipe_freezes():
    """simple-v1 and simple-v1-lag are the same loss design (zero loss-field diffs for both);
    the second rank key — fewest edits among the non-loss fields the recipe also freezes —
    is what keeps a point-mass run from being named after the lag recipe (T1-11)."""
    for recipe in (CONTROL_RECIPE_SIMPLE_V1, CONTROL_RECIPE_SIMPLE_V1_LAG):
        config = _control_config(control_recipe_name="custom")
        config.update(control_recipe_overrides(recipe))
        config["control_recipe_name"] = "custom"
        assert loss_design_name(config) == recipe


def test_exact_v2_content_reads_as_v2():
    config = _control_config(control_recipe_name="custom")
    config.update(control_recipe_overrides(CONTROL_RECIPE_SIMPLE_V2))
    config["control_recipe_name"] = "custom"
    assert loss_design_name(config) == "simple-v2"


def test_the_l1b_supervision_terms_are_loss_fields_with_short_names():
    """A run trained THROUGH the heading-rate or bank-TV term is a different objective and
    must not share a name with one that was not. Every stored config predates both and
    carries their defaults, so adding them renames nothing that exists."""
    from ts_transformer.run_naming import _ABBREV

    for field, abbreviation in (
        ("control_heading_rate_loss_weight", "hr"),
        ("control_heading_rate_loss_scale_dps", "hr-scale"),
        ("control_bank_tv_loss_weight", "bank-tv"),
    ):
        assert field in CONTROL_LOSS_FIELDS
        assert _ABBREV[field] == abbreviation

    config = _control_config(control_recipe_name="custom")
    config.update(control_recipe_overrides(CONTROL_RECIPE_SIMPLE_V3))
    config["control_recipe_name"] = "custom"
    assert loss_design_name(config) == "simple-v3"
    config["control_heading_rate_loss_weight"] = 8.0
    assert loss_design_name(config) == "simple-v3+(hr=8)"
    config["control_bank_tv_loss_weight"] = 1.0
    # Diffs are listed in CONTROL_LOSS_FIELDS order, so the heading rate leads.
    assert loss_design_name(config) == "simple-v3+(hr=8, bank-tv=1)"


def test_deeply_custom_loss_collapses_to_a_stable_hash():
    config = _control_config(control_recipe_name="custom")
    for field in CONTROL_LOSS_FIELDS[:8]:
        default = config[field]
        config[field] = (default + 1.0) if isinstance(default, float) else "different"
    first = loss_design_name(config)
    assert first.startswith("custom-") and len(first.split("-")[1]) == 8
    assert loss_design_name(dict(config)) == first


def test_dynamics_distinguishes_derivative_and_backend():
    assert dynamics_name(_state_defaults()) == "kinematic"
    assert dynamics_name(_control_config()) == "point-mass"
    lagged = _control_config(
        control_dynamics_model=CONTROL_DYNAMICS_FIRST_ORDER_LAG,
        control_dynamics_backend=CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    )
    assert dynamics_name(lagged) == (
        "first-order-lag @scaled-transport-chart-velocity"
    )
    lagged["control_bank_time_constant_s"] = 4.0
    assert "τ-bank=4s" in dynamics_name(lagged)


def test_the_retired_transport_chart_backend_still_names_its_stored_runs():
    """`transport-chart-velocity` left the config vocabulary in T2 (2026-09-07).

    Thirteen stored 2026-07/08 configs carry it and their on-disk directories end in
    `_tcv`. The grammar reads stored DICTS, not `TSConfig`, so it must keep naming them —
    a grammar that fell back to a slugified spelling would rename historical record.
    """
    stored = _control_config(control_dynamics_backend="transport-chart-velocity")
    assert dynamics_name(stored) == "point-mass @transport-chart-velocity"
    assert "_tcv" in run_slug(stored).replace("-", "_")


def test_meta_folds_past_the_cap_and_keeps_seed_first():
    config = _control_config(
        seed=2024,
        d_model=512,
        d_ff=1024,
        e_layers=6,
        batch_size=128,
        learning_rate=1e-4,
        dropout=0.3,
        weight_decay=0.01,
    )
    name = run_display_name(config)
    assert "seed=2024" in name
    assert "+2 more" in name


def test_extra_meta_is_appended_verbatim():
    name = run_display_name(_state_defaults(), extra=("campaign/arm",))
    assert name.endswith("· campaign/arm")


def test_unknown_backbone_passes_through():
    config = _state_defaults()
    config["model"] = "informer"
    assert "· informer ·" in run_display_name(config)


def test_category_label_prefixes_the_split():
    label = category_display_label("val", "state · iTransformer · kinematic · state-v1")
    assert label == (
        "Validation split (model selection) — "
        "Predicted: state · iTransformer · kinematic · state-v1"
    )
    assert category_display_label("nope", "x", kind="Experiment") == "Experiment: x"


def test_slug_is_filesystem_safe():
    config = _control_config(
        control_dynamics_model=CONTROL_DYNAMICS_FIRST_ORDER_LAG,
        control_dynamics_backend=CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
        control_imitation_loss_weight=16.0,
    )
    slug = run_slug(config)
    assert slug.startswith("control_itr_lag-stcv_")
    assert all(c.isalnum() or c in "_-" for c in slug)


def test_a_named_recipe_with_an_open_field_edit_is_named_recipe_plus_edits():
    """The final-approach penalty is open under every recipe; a run that sets it must not
    wear the bare recipe name (two different objectives would share one name)."""
    from ts_transformer.config import TSConfig, recipe_settings

    plain = TSConfig(**recipe_settings("simple-v3", keep_name=True))
    assert run_display_name(plain.to_dict()).split(" · ")[3] == "simple-v3"
    edited = TSConfig(**recipe_settings("simple-v3", keep_name=True),
                      procedure_loss_lateral_weight=0.001, procedure_loss_vertical_weight=0.001)
    assert run_display_name(edited.to_dict()).split(" · ")[3] == "simple-v3+(proc-lat=0.001, proc-vert=0.001)"
    assert run_slug(edited.to_dict()) != run_slug(plain.to_dict())
    # ``custom`` semantics for a runner that varies a frozen field are unchanged.
    assert recipe_settings("simple-v3", keep_name=False)["control_recipe_name"] == "custom"


def test_the_recipe_definitions_are_literals_and_match_the_defaults_today():
    """A recipe is a frozen definition: it must not be spelled with module defaults (a
    default that moves would redefine every published paired comparison). Today the
    literals equal the defaults — this pins that equality so a divergence is a
    deliberate, visible act."""
    import inspect
    import re
    from ts_transformer.config import TSConfig, control_recipe_overrides, control_simple_v1_overrides, CONTROL_RECIPE_SIMPLE_V1
    for function in (control_simple_v1_overrides, control_recipe_overrides):
        source = inspect.getsource(function)
        leaked = re.findall(r"\bDEFAULT_[A-Z0-9_]+\b", source) + re.findall(r"\bCHANNELS\b", source)
        assert not leaked, f"{function.__name__} spells a mutable default: {leaked}"
    defaults = TSConfig()
    recipe = control_recipe_overrides(CONTROL_RECIPE_SIMPLE_V1)
    for field in ("dt_s", "seq_len", "channels", "aircraft_type", "random_train_anchor_min_future_s",
                  "validation_common_grid_points", "position_loss_scale_m", "final_time_scale_s"):
        assert recipe[field] == getattr(defaults, field), field


# ── review 2026-09-09 C-3 ────────────────────────────────────────────────────

def test_the_common_grid_resolution_names_the_run():
    """Two custom arms differing only in `--validation-common-grid-points 64|128` keep
    different epochs; they used to share a name and a slug."""
    from ts_transformer.run_naming import run_slug
    base = _state_defaults()
    finer = {**base, "validation_common_grid_points": 128}
    assert "grid-points=128" in run_display_name(finer)
    assert "grid-points" not in run_display_name(base)
    assert run_slug(finer) != run_slug(base)


def test_every_tsconfig_field_is_named_or_excused_by_name():
    """The reverse of the import-time guard: a CLI-settable field in no naming list lets
    two runs differing only in it share a name. `KNOWN_UNNAMED_FIELDS` holds the ones
    that deliberately name nothing, each with its reason."""
    from ts_transformer.run_naming import KNOWN_UNNAMED_FIELDS
    for field, reason in KNOWN_UNNAMED_FIELDS.items():
        assert field in TSConfig().to_dict(), field
        assert reason
    assert "device" in KNOWN_UNNAMED_FIELDS
    for named in ("validation_common_grid_points", "lr_plateau_patience", "lr_plateau_factor",
                  "random_train_anchor_min_future_s"):
        assert named not in KNOWN_UNNAMED_FIELDS, named


def test_the_scheduler_patience_factor_and_anchor_floor_name_a_custom_run():
    """Review C-3, decided 2026-09-09: the three identity-bearing fields name the run. A
    named recipe pins the scheduler pair, so a recipe run is unchanged; a custom run off the
    defaults spells them, and the slug moves with them."""
    from ts_transformer.config import CONTROL_RECIPE_SIMPLE_V3, recipe_settings
    from ts_transformer.run_naming import run_slug
    base = _state_defaults()
    custom = {**base, "lr_plateau_patience": 8, "lr_plateau_factor": 0.3,
              "random_train_anchor": True, "random_train_anchor_min_future_s": 20.0}
    name = run_display_name(custom)
    for token in ("lr-patience=8", "lr-factor=0.3", "anchor-min-future=20"):
        assert token in name, (token, name)
    assert run_slug(custom) != run_slug({**custom, "lr_plateau_patience": 12})
    assert "lr-patience" not in run_display_name(base)
    # simple-v3 pins lr_plateau_patience=8: frozen by the recipe, so it is not a deviation.
    recipe = recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=True)
    recipe_name = run_display_name({**TSConfig().to_dict(), **recipe})
    assert recipe["lr_plateau_patience"] == 8 and "lr-patience" not in recipe_name


# ── the structured form (the frontend Experiments picker's parameter rows) ─────────────────

def _rows_by_section(rows: list[dict]) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    for row in rows:
        sections.setdefault(row["section"], {})[row["name"]] = row["value"]
    return sections


def test_parameter_rows_name_every_grammar_part_and_never_fold():
    """The display name folds past six meta items; the rows list every one, each under the
    config field it is, and the grammar's own parts under a named `Model` row."""
    from ts_transformer.run_naming import run_parameter_rows
    config = _control_config(
        seed=2024, d_model=512, d_ff=1024, e_layers=6, batch_size=128,
        learning_rate=1e-4, dropout=0.3, weight_decay=0.01,
    )
    assert "+2 more" in run_display_name(config)

    rows = run_parameter_rows(config)
    sections = _rows_by_section(rows)
    assert list(sections)[0] == "Model"
    assert sections["Model"] == {
        "Output": "control", "Backbone": "iTransformer", "Dynamics": "point-mass",
        "Loss design": loss_design_name(config), "Horizon": "normalized", "Seed": "2024",
    }
    assert sections["Architecture"] == {
        "d_model": "512", "d_ff": "1024", "e_layers": "6", "dropout": "0.3",
    }
    assert sections["Training"] == {
        "batch_size": "128", "learning_rate": "0.0001", "weight_decay": "0.01",
    }
    # the seed is a Model row, never a second settings row
    assert all(row["name"] != "seed" for row in rows if row["section"] != "Model")
    fields = {row["name"]: row.get("field") for row in rows if row["section"] == "Model"}
    assert fields["Output"] == "prediction_output" and fields["Dynamics"] is None


def test_parameter_rows_spell_out_a_loss_design_the_name_hashes():
    from ts_transformer.run_naming import loss_design_parts, run_parameter_rows
    config = _control_config(control_recipe_name="custom")
    for field in CONTROL_LOSS_FIELDS[:8]:
        default = config[field]
        config[field] = (default + 1.0) if isinstance(default, float) else "different"
    assert loss_design_name(config).startswith("custom-")

    base, edits = loss_design_parts(config)
    sections = _rows_by_section(run_parameter_rows(config))
    assert sections["Model"]["Loss design"] == loss_design_name(config)
    assert list(sections[f"Loss edits vs {base}"]) == [field for field, _ in edits]
    assert len(edits) > 4


def test_parameter_rows_of_a_named_recipe_list_the_settings_the_recipe_freezes():
    """The NAME skips a recipe's frozen fields (they are the recipe); the rows answer "what
    does this run use", so they list them."""
    from ts_transformer.config import CONTROL_RECIPE_SIMPLE_V3, recipe_settings
    from ts_transformer.run_naming import run_parameter_rows
    recipe = recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=True)
    config = {**TSConfig().to_dict(), **recipe}
    assert "lr-patience" not in run_display_name(config)

    sections = _rows_by_section(run_parameter_rows(config))
    assert sections["Model"]["Loss design"] == "simple-v3"
    assert sections["Training"]["lr_plateau_patience"] == "8"
    assert not any(section.startswith("Loss edits") for section in sections)
