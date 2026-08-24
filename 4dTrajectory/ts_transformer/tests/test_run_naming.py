"""The canonical run-name grammar: output · backbone · dynamics · loss · meta."""

import sys
from pathlib import Path

_TS_DIR = Path(__file__).resolve().parents[1]
if str(_TS_DIR) not in sys.path:
    sys.path.insert(0, str(_TS_DIR))

from config import (  # noqa: E402
    CONTROL_DYNAMICS_FIRST_ORDER_LAG,
    CONTROL_DYNAMICS_SCALED_TRANSPORT_CHART_VELOCITY,
    CONTROL_RECIPE_SIMPLE_V2,
    CONTROL_RECIPE_SIMPLE_V3,
    TSConfig,
    control_recipe_overrides,
)
from run_naming import (  # noqa: E402
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
    for group in (CONTROL_LOSS_FIELDS, STATE_LOSS_FIELDS, META_FIELDS):
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


def test_exact_v2_content_reads_as_v2():
    config = _control_config(control_recipe_name="custom")
    config.update(control_recipe_overrides(CONTROL_RECIPE_SIMPLE_V2))
    config["control_recipe_name"] = "custom"
    assert loss_design_name(config) == "simple-v2"


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
