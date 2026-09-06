"""The L0 runner's joins: three key spaces meet, and a stratum must follow its flight.

The cohort's readout key (`id_runway_icao24_landing`), the compact
`flight_scenarios.identity.flight_key` the dataset builds under, and the ORDER
`build_series` returns are three different things. The strata masks are aligned to the
cohort's order; the scored rows arrive in the series' order. Zipping the masks against the
wrong one is size-matched and silent, and it turns the gated stratum into a random subset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from approach_difficulty import STRATUM_ALL, STRATUM_VECTORED, strata_masks
from control.oracle.basis import inverse_dynamics_seed
from run_ts_control_basis_oracle import summarise, summary_row_key


def _reference() -> dict[str, dict]:
    """Three flights whose readout keys sort into a different order than they are listed."""
    rows = [
        {"id": "ZZZ1", "runway": "05L", "icao24": "aaa001", "landing_time_utc": "2026-01-01T00:00:00Z",
         "route_tortuosity": 2.4, "established_at_anchor": False, "remaining_path_m": 30_000.0},
        {"id": "AAA2", "runway": "05L", "icao24": "aaa002", "landing_time_utc": "2026-01-01T00:10:00Z",
         "route_tortuosity": 1.01, "established_at_anchor": True, "remaining_path_m": 9_000.0},
        {"id": "MMM3", "runway": "23R", "icao24": "aaa003", "landing_time_utc": "2026-01-01T00:20:00Z",
         "route_tortuosity": 1.9, "established_at_anchor": False, "remaining_path_m": 25_000.0},
    ]
    return {summary_row_key(row): row for row in rows}


def _row(key: str, ade: float) -> dict:
    return {
        "flight_key": key, "ade_m": ade, "fde_m": ade, "fixed_dt_ade_m": ade,
        "seed_fixed_dt_ade_m": ade * 2, "chamfer_m": ade, "frechet_m": ade,
        "saturated_fraction": 0.0, "tail_gain": 0.0,
    }


def test_a_stratum_follows_its_flight_whatever_order_the_rows_arrive_in():
    reference = _reference()
    cohort_keys = sorted(reference)                     # what the masks are built against
    masks = strata_masks(reference, cohort_keys)
    vectored_keys = {key for key, member in zip(cohort_keys, masks[STRATUM_VECTORED]) if member}
    assert len(vectored_keys) == 2                      # the two tortuous, unestablished flights

    # The rows come back in the series' order, which is neither the cohort's nor stable.
    ades = {key: 100.0 * (index + 1) for index, key in enumerate(cohort_keys)}
    for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
        rows = [_row(cohort_keys[index], ades[cohort_keys[index]]) for index in order]
        strata = summarise(rows, masks, cohort_keys)
        assert strata[STRATUM_ALL]["n"] == 3
        assert strata[STRATUM_VECTORED]["n"] == 2
        assert strata[STRATUM_VECTORED]["ade_mean_m"] == pytest.approx(
            float(np.mean([ades[key] for key in vectored_keys]))
        )


def test_summarise_refuses_a_row_outside_the_cohort_the_masks_cover():
    reference = _reference()
    cohort_keys = sorted(reference)
    masks = strata_masks(reference, cohort_keys)
    with pytest.raises(RuntimeError, match="outside the cohort"):
        summarise([_row("not_a_cohort_key", 100.0)], masks, cohort_keys)


def test_summary_row_key_maps_a_null_field_to_the_empty_string():
    """A present-but-null field must read as absent, not as the string "None"."""
    assert summary_row_key({"id": "AAA", "runway": None, "icao24": "x", "landing_time_utc": "t"}) == (
        "AAA__x_t"
    )


@dataclass
class _Series:
    flight_id: str
    values: np.ndarray
    times: np.ndarray
    frame: object


def test_the_seed_refuses_a_batch_that_does_not_cover_the_same_flights():
    dynamics = {
        "control_lower": torch.zeros(2, 3),
        "control_upper": torch.ones(2, 3),
        "initial_state": torch.zeros(2, 7),
        "aero_params": torch.zeros(2, 4),
        "max_thrust_n": torch.ones(2),
    }
    series = [_Series("only-one", np.zeros((2, 6)), np.zeros(2), None)]
    with pytest.raises(ValueError, match="same flights"):
        inverse_dynamics_seed(
            series, 0, dynamics, config=None, n_segments=2, final_time_s=np.array([10.0])
        )
