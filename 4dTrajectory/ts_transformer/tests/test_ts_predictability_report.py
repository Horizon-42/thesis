from __future__ import annotations

import pytest

from ts_transformer.config import HORIZON_FULL, HORIZON_WINDOW, TSConfig
from ts_transformer.data.time_grids import output_time_grid


def test_full_grid_shortens_the_last_segment_and_masks_padding():
    config = TSConfig(
        horizon_mode=HORIZON_FULL,
        full_horizon_steps=4,
        dt_s=2.0,
    )
    grid = output_time_grid(5.0, config)

    assert grid.offsets_s == pytest.approx([2.0, 4.0, 5.0, 5.0])
    assert grid.segment_durations_s == pytest.approx([2.0, 2.0, 1.0, 0.0])
    assert grid.active.tolist() == [True, True, True, False]
    assert not grid.horizon_capped


def test_window_grid_reports_a_capped_physical_horizon():
    config = TSConfig(
        horizon_mode=HORIZON_WINDOW,
        window_horizon_steps=3,
        dt_s=2.0,
    )
    grid = output_time_grid(8.0, config)

    assert grid.offsets_s == pytest.approx([2.0, 4.0, 6.0])
    assert grid.active.all()
    assert grid.horizon_capped
