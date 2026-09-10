"""The multi-flight capacity report.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""

import json

import pytest

import ts_transformer.data.channels as ch
import ts_transformer.inference.build_multiflight_capacity_report as capacity_report


@pytest.mark.parametrize(
    ("field", "different"),
    [
        ("control_imitation_loss_weight", 0.75),
        ("learning_rate", 5e-4),
        ("d_model", 128),
    ],
)
def test_capacity_report_recipe_detects_every_config_difference(
    tmp_path, field, different
):
    first = {"config": {"notes": {}, field: 0.25}}
    second = {"config": {"notes": {"comment": "display only"}, field: different}}
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")

    assert capacity_report._recipe(first) != capacity_report._recipe(second)
    with pytest.raises(ValueError, match=field):
        capacity_report._load_results([first_path, second_path])


def test_capacity_report_masks_unsupervised_reference_velocity_placeholders():
    diagnostics = {
        "channel_names": list(ch.CHANNELS),
        "anchor_state": [0.0, 0.0, 0.0, 10.0, 0.0, -1.0],
        "fixed_dt": {
            "offset_s": [2.0, 4.0, 6.0],
            "reference_state": [
                [20.0, 0.0, -2.0, 10.0, 0.0, -1.0],
                [40.0, 0.0, -4.0, 999.0, 999.0, 999.0],
                [60.0, 0.0, -6.0, 999.0, 999.0, 999.0],
            ],
            "predicted_state": [
                [20.0, 0.0, -2.0, 10.0, 0.0, -1.0],
                [40.0, 0.0, -4.0, 10.0, 0.0, -1.0],
                [60.0, 0.0, -6.0, 10.0, 0.0, -1.0],
            ],
            "reference_fully_measured": [True, False, False],
        },
    }

    charts = capacity_report._chart_diagnostics(diagnostics)

    assert charts["reference_horizontal_speed_mps"] == [10.0, None, None]
    assert charts["reference_vertical_speed_mps"] == [-1.0, None, None]
    assert charts["reference_consistency_mps"] == [0.0, None, None]
    assert charts["reference_acceleration_mps2"] == [0.0, None, None]
