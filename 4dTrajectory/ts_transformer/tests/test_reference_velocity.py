from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


TS_DIR = Path(__file__).resolve().parents[1]
if str(TS_DIR) not in sys.path:
    sys.path.insert(0, str(TS_DIR))

from config import TSConfig  # noqa: E402
from reference_velocity import (  # noqa: E402
    REFERENCE_VELOCITY_POSITION_DIFFERENCE,
    REFERENCE_VELOCITY_SMOOTHED_POSITION_DIFFERENCE,
    rebuild_reference_velocities,
)


def _linear_values() -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(0.0, 12.0, 2.0)
    values = np.zeros((len(times), 6), dtype=np.float64)
    values[:, :3] = times[:, None] * np.array([3.0, -2.0, 0.5])
    values[:, 3:] = 999.0
    return times, values


@pytest.mark.parametrize(
    "source",
    [
        REFERENCE_VELOCITY_POSITION_DIFFERENCE,
        REFERENCE_VELOCITY_SMOOTHED_POSITION_DIFFERENCE,
    ],
)
def test_position_derived_reference_velocity_recovers_linear_motion(source: str):
    times, values = _linear_values()
    rebuilt = rebuild_reference_velocities(
        times,
        values,
        source=source,
        valid_rows=np.ones(len(times), dtype=bool),
    )

    np.testing.assert_allclose(rebuilt[:, :3], values[:, :3])
    expected = np.tile([3.0, -2.0, 0.5], (len(times), 1))
    np.testing.assert_allclose(rebuilt[:, 3:], expected, atol=1e-12)


def test_reference_velocity_rebuild_leaves_masked_tail_placeholders_unchanged():
    times, values = _linear_values()
    valid = np.array([True, True, True, True, False, False])
    rebuilt = rebuild_reference_velocities(
        times,
        values,
        source=REFERENCE_VELOCITY_POSITION_DIFFERENCE,
        valid_rows=valid,
    )

    expected = np.tile([3.0, -2.0, 0.5], (4, 1))
    np.testing.assert_allclose(rebuilt[:4, 3:], expected)
    np.testing.assert_allclose(rebuilt[4:, 3:], 999.0)


def test_smoothed_position_velocity_does_not_use_future_positions():
    times, values = _linear_values()
    changed = values.copy()
    changed[4:, :3] += 10_000.0
    valid = np.ones(len(times), dtype=bool)

    original = rebuild_reference_velocities(
        times,
        values,
        source=REFERENCE_VELOCITY_SMOOTHED_POSITION_DIFFERENCE,
        valid_rows=valid,
    )
    perturbed = rebuild_reference_velocities(
        times,
        changed,
        source=REFERENCE_VELOCITY_SMOOTHED_POSITION_DIFFERENCE,
        valid_rows=valid,
    )

    np.testing.assert_allclose(perturbed[:4, 3:], original[:4, 3:])


def test_reference_velocity_source_is_required_in_serialized_config():
    document = TSConfig().to_dict()
    document.pop("reference_velocity_source")

    with pytest.raises(ValueError, match="missing reference_velocity_source"):
        TSConfig.from_dict(document)
