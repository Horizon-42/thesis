"""The paired plan-oracle reader (`experiments/plan_oracle_pair`): two artifacts joined
flight by flight — an identical arm reads as zero differing rows, a moved flight as one,
a different cohort is refused, and an artifact written before the order hold reads `n/a`
on the hold's columns."""

from __future__ import annotations

import copy

import pytest

from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED
from ts_transformer.experiments.plan_oracle_pair import IDENTITY_TOLERANCE_M, format_table, pair


def _row(dataset_id: str, *, tortuosity: float, ade_m: float, held: float | None = 2.0) -> dict:
    hook = {"planBankCappedSteps": 0.1, "planSteps": 10.0}
    if held is not None:
        hook.update({"planHeldSteps": held, "planOrderChanges": 1.0})
    return {
        "dataset_id": dataset_id,
        "difficulty": {"route_tortuosity": tortuosity, "established_at_anchor": False, "remaining_path_m": 20_000.0},
        "prediction": {"ade_m": ade_m, "fde_m": 50.0, "final_time_error_s": -3.0},
        "geometry": {"chamfer_m": 40.0, "frechet_m": 120.0},
        "reference": {"established": True, "fully_flyable": True, "lateral_violation": False, "glidepath_violation": False},
        "route": {"capped_by": None, "turns_incomplete": 0, "legs": 2},
        "hook": hook,
    }


def _rows(**kw) -> dict[str, dict]:
    rows = [_row("a", tortuosity=1.0, ade_m=300.0, **kw), _row("b", tortuosity=1.3, ade_m=2_000.0, **kw)]
    return {row["dataset_id"]: row for row in rows}


def test_an_identical_arm_reads_as_zero_differing_rows_and_a_moved_flight_as_one():
    base = _rows()
    same = pair(base, copy.deepcopy(base))
    assert same["identity"] == {"rows_differing": 0, "max_abs_ade_delta_m": 0.0, "established_differing": 0}
    assert same["strata"][STRATUM_ALL]["ade_m"]["delta_p50"] == 0.0
    assert same["strata"][STRATUM_STRAIGHT_IN]["flights"] == 1 and same["strata"][STRATUM_VECTORED]["flights"] == 1
    moved = copy.deepcopy(base)
    moved["b"]["prediction"]["ade_m"] -= 500.0
    moved["b"]["reference"]["established"] = False
    result = pair(base, moved)
    assert result["identity"]["rows_differing"] == 1
    assert result["identity"]["max_abs_ade_delta_m"] == pytest.approx(500.0)
    assert result["identity"]["established_differing"] == 1
    vectored = result["strata"][STRATUM_VECTORED]
    assert vectored["ade_m"]["delta_p50"] == pytest.approx(-500.0) and vectored["ade_m"]["arm_lower_share"] == 1.0
    assert vectored["established"] == {"base": 1.0, "arm": 0.0}
    assert result["strata"][STRATUM_ALL]["orders_held"]["arm"] == pytest.approx(0.2)
    assert IDENTITY_TOLERANCE_M < 1.0
    text = format_table(result, "base", "arm")
    assert "1 of 2 rows differ" in text and "orders held (of steps)" in text


def test_an_artifact_without_the_hold_reads_n_a_and_a_different_cohort_is_refused():
    base = _rows(held=None)
    result = pair(base, _rows())
    assert result["strata"][STRATUM_ALL]["orders_held"]["base"] != result["strata"][STRATUM_ALL]["orders_held"]["base"]  # nan
    assert "n/a" in format_table(result, "old", "new")
    other = _rows()
    other["c"] = _row("c", tortuosity=1.0, ade_m=1.0)
    with pytest.raises(SystemExit, match="different flights"):
        pair(base, other)
