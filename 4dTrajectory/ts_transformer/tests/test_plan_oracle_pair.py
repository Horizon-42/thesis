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
    # a pooled artifact against a single-airport one: the common flights, the counts said
    common = pair(base, other, common=True)
    assert common["flights"] == 2 and common["cohort"] == {"base_flights": 2, "arm_flights": 3, "common": 2}
    assert "the common 2 of base 2 / arm 3" in format_table(common, "krdu", "pooled")
    with pytest.raises(SystemExit, match="base inside the arm"):
        pair(other, base, common=True)
    with pytest.raises(SystemExit, match="base inside the arm"):
        pair(base, {"z": _row("z", tortuosity=1.0, ade_m=1.0)}, common=True)



def test_the_oracle_summary_cuts_by_airport():
    """`plan_oracle.summarize_by_airport`: the flight key's prefix is the airport; each block
    is the full summary over that airport's rows alone."""
    from ts_transformer.data.approach_difficulty import STRATUM_ALL
    from ts_transformer.experiments.plan_oracle import summarize, summarize_by_airport

    def oracle_row(dataset_id: str, ade: float) -> dict:
        return {
            "dataset_id": dataset_id,
            "difficulty": {"route_tortuosity": 1.0, "established_at_anchor": False, "remaining_path_m": 20_000.0},
            "prediction": {"ade_m": ade, "fde_m": 10.0, "final_time_error_s": 1.0, "true_final_time_s": 300.0},
            "geometry": {"chamfer_m": 5.0, "frechet_m": 9.0},
            "reference": {"fully_flyable": True, "established": True, "lateral_violation": False,
                          "glidepath_violation": False, "floor_violation": False},
            "truth": {"lateral_violation": False, "glidepath_violation": False, "floor_violation": False},
            "route": {"route_time_s": 290.0, "shortfall_m": 0.0, "intercept_deg": 0.0, "kind": "final"},
            "T_s": 300.0, "labels": {"waypoints": [], "waypoints_dropped": 0}, "eta_predicted_s": None,
            "hook": {"planBankCappedSteps": 0.0, "planThrustSaturatedSteps": 0.0, "planThrustIdleSteps": 0.0,
                     "planLoadClampedSteps": 0.0, "planRouteCrossTrackM": 1.0, "gatedSteps": 0.0, "clampedSteps": 0.0,
                     "planCaptureHeightClamped": 0.0},
        }

    rows = [oracle_row("KRDU:a_05L_x_t", 100.0), oracle_row("KRDU:b_05L_y_t", 300.0), oracle_row("KSJC:c_30L_z_t", 500.0)]
    by_airport = summarize_by_airport(rows)
    assert sorted(by_airport) == ["KRDU", "KSJC"]
    assert by_airport["KRDU"][STRATUM_ALL]["ade_mean_m"] == pytest.approx(200.0)
    assert by_airport["KSJC"][STRATUM_ALL]["flights"] == 1
    assert summarize(rows)[STRATUM_ALL]["ade_mean_m"] == pytest.approx(300.0)
