"""The gates (`manoeuvre/gates.py`): every verdict reads the lockstep artefacts, refuses a single
seed, and applies the pre-registered numbers exactly."""

from __future__ import annotations

import pytest

from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_ESTABLISHED, STRATUM_STRAIGHT_IN, STRATUM_VECTORED
from ts_transformer.manoeuvre import gates as g


def _payload(*, vectored_ade: float, straight_ade: float, flyable: float, established: float,
             at_vectored=(500.0, 900.0), at_straight=(200.0, 400.0), name="arm", protocol="A") -> dict:
    def cell(ade, at):
        return {"n": 10, "ade_mean_m": ade, "ade_p50_m": ade * 0.8, "fde_p50_m": ade, "fully_flyable_share": flyable,
                "established_share": established, "at_p50_m": {"60": at[0] / 2, "120": at[0], "180": at[1], "300": None},
                "ended": {"crossed": 8, "horizon": 2}, "predictions_p50": 4.0, "e_track_by_round_p50_m": {}, "e_plan_by_round_p50_m": {}}
    return {"executor_name": name, "protocol": protocol, "strata": {
        STRATUM_ALL: cell((vectored_ade + straight_ade) / 2, at_vectored), STRATUM_VECTORED: cell(vectored_ade, at_vectored),
        STRATUM_STRAIGHT_IN: cell(straight_ade, at_straight), STRATUM_ESTABLISHED: {"n": 0},
    }}


def test_gate_x_reads_flyable_established_against_the_guidance_and_the_two_ade_lines():
    good = _payload(vectored_ade=1200.0, straight_ade=200.0, flyable=0.97, established=0.85, protocol="C")
    verdict = g.gate_x({1337: good, 2024: good}, guidance_established=0.9)
    assert verdict["passes"] and verdict["established_floor"] == pytest.approx(0.81)
    weak = _payload(vectored_ade=1200.0, straight_ade=200.0, flyable=0.97, established=0.80, protocol="C")
    assert not g.gate_x({1337: good, 2024: weak}, guidance_established=0.9)["passes"]
    with pytest.raises(ValueError, match="two seeds"):
        g.gate_x({1337: good}, guidance_established=0.9)
    with pytest.raises(ValueError, match="protocol"):
        g.gate_x({1337: good, 2024: _payload(vectored_ade=1.0, straight_ade=1.0, flyable=1.0, established=1.0)}, guidance_established=0.9)


def test_gate_e_target_and_progress_are_read_seed_by_seed():
    target = _payload(vectored_ade=2700.0, straight_ade=400.0, flyable=0.96, established=0.95)
    verdict = g.gate_e({1337: target, 2024: target})
    assert verdict["target"] and verdict["progress"]
    progress_only = {1337: _payload(vectored_ade=2900.0, straight_ade=500.0, flyable=0.9, established=0.70),
                     2024: _payload(vectored_ade=2940.0, straight_ade=500.0, flyable=0.9, established=0.70)}
    verdict = g.gate_e(progress_only)
    assert not verdict["target"] and verdict["progress"]
    stalled = {1337: _payload(vectored_ade=2950.0, straight_ade=500.0, flyable=0.9, established=0.70), 2024: progress_only[2024]}
    assert not g.gate_e(stalled)["progress"]          # 2950 is not 125 m below 3044
    with pytest.raises(ValueError, match="two seeds"):
        g.gate_e({2024: target})
    # a seed without a v2 progress line is not applicable, never a failure
    third = g.gate_e({7: target, 9: target})
    assert third["target"] and third["progress"] is None and third["progress_seeds"] == []
    with pytest.raises(ValueError, match="protocol"):
        g.gate_e({1337: target, 2024: _payload(vectored_ade=1.0, straight_ade=1.0, flyable=1.0, established=1.0, protocol="C")})


def test_gate_p_open_loop_and_the_discrete_vs_continuous_control():
    open_loop = _payload(vectored_ade=0, straight_ade=0, flyable=1, established=1, at_vectored=(800.0, 1500.0), at_straight=(400.0, 600.0), protocol="A-truth")
    assert g.gate_p_open_loop({1337: open_loop, 2024: open_loop})["passes"]
    late = _payload(vectored_ade=0, straight_ade=0, flyable=1, established=1, at_vectored=(800.0, 1530.0), at_straight=(400.0, 600.0), protocol="A-truth")
    assert not g.gate_p_open_loop({1337: open_loop, 2024: late})["passes"]
    with pytest.raises(ValueError, match="protocol"):
        g.gate_p_open_loop({1337: open_loop, 2024: _payload(vectored_ade=0, straight_ade=0, flyable=1, established=1)})
    discrete = _payload(vectored_ade=2600.0, straight_ade=400.0, flyable=0.95, established=0.80)
    continuous = _payload(vectored_ade=2500.0, straight_ade=400.0, flyable=0.95, established=0.82)
    assert g.gate_p_discrete_vs_continuous({1337: discrete, 2024: discrete}, {1337: continuous, 2024: continuous})["passes"]   # a tie
    better_continuous = _payload(vectored_ade=2400.0, straight_ade=400.0, flyable=0.95, established=0.82)
    assert not g.gate_p_discrete_vs_continuous({1337: discrete, 2024: discrete}, {1337: continuous, 2024: better_continuous})["passes"]
    with pytest.raises(ValueError, match="fewer than two"):
        g.gate_p_discrete_vs_continuous({1337: discrete, 2024: discrete}, {1337: continuous, 7: continuous})


def test_gate_s_picks_the_best_on_both_seeds_lets_established_decide_and_breaks_ties_longer():
    def cand(ade, est):
        return {1337: _payload(vectored_ade=ade, straight_ade=300.0, flyable=0.95, established=est),
                2024: _payload(vectored_ade=ade + 10.0, straight_ade=300.0, flyable=0.95, established=est - 0.01)}
    verdict = g.gate_s({30.0: cand(2600.0, 0.80), 60.0: cand(2400.0, 0.85), 90.0: cand(2700.0, 0.70)})
    assert verdict["selected_segment_s"] == 60.0 and verdict["note"] is None
    # ADE and established disagree: established decides
    verdict = g.gate_s({30.0: cand(2200.0, 0.80), 60.0: cand(2400.0, 0.90)})
    assert verdict["best_by_ade"] == 30.0 and verdict["best_by_established"] == 60.0 and verdict["selected_segment_s"] == 60.0
    # a longer candidate within the seed line on both metrics wins the tie
    verdict = g.gate_s({60.0: cand(2400.0, 0.85), 90.0: cand(2450.0, 0.84)})
    assert verdict["selected_segment_s"] == 90.0 and "longer" in verdict["note"]
    with pytest.raises(ValueError, match="at least two"):
        g.gate_s({60.0: cand(2400.0, 0.85)})
    with pytest.raises(ValueError, match="two seeds"):
        g.gate_s({60.0: cand(2400.0, 0.85), 90.0: {1337: cand(2450.0, 0.84)[1337]}})


def test_gate_t_is_the_readouts_own_verdict():
    assert g.gate_t({"gate_t": {"selected_k": 32}}) == {"selected_k": 32}


def _none_payload(*, established_all: float, established_vectored: float, flyable: float = 0.99, vectored_ade: float = 2000.0, n: int = 1400) -> dict:
    """A protocol-none lockstep payload with the cells the grid gate reads."""
    def cell(established, ade):
        return {"n": 10, "ade_mean_m": ade, "ade_p50_m": ade * 0.8, "fde_p50_m": ade, "fully_flyable_share": flyable,
                "established_share": established}
    return {"executor_name": "twin", "protocol": "none", "segment_s": 20.0, "flights": n, "strata": {
        STRATUM_ALL: cell(established_all, vectored_ade * 0.6), STRATUM_VECTORED: cell(established_vectored, vectored_ade),
        STRATUM_STRAIGHT_IN: cell(1.0, 300.0), STRATUM_ESTABLISHED: {"n": 0},
    }}


def test_the_grid_gate_reads_the_seed_line_off_the_grid_and_breaks_ties_shorter():
    """Two-tier v3 A1 (§3.3, D9): established over all flights and over the vectored group,
    both seeds; fully flyable ≥ 0.95 on both seeds; the seed line is the grid's own p75 of the
    seed differences; ties within it go to the shorter segment, then the shorter lookback."""
    def grid(values):
        return {cell: {1337: _none_payload(established_all=a1, established_vectored=v1),
                       2024: _none_payload(established_all=a2, established_vectored=v2)}
                for cell, (a1, a2, v1, v2) in values.items()}
    cells = grid({
        (30.0, 20.0): (0.70, 0.71, 0.30, 0.31),
        (60.0, 20.0): (0.70, 0.72, 0.30, 0.32),      # ties the best within the line: shorter L wins the tie
        (30.0, 60.0): (0.75, 0.77, 0.40, 0.42),      # the best on both seeds, both primaries
        (60.0, 60.0): (0.74, 0.76, 0.39, 0.41),      # within 0.01 of the best: a tie; longer L loses
        (30.0, 90.0): (0.76, 0.75, 0.41, 0.40),      # within the line of the best on both seeds; longer Δ loses
    })
    verdict = g.gate_grid(cells)
    assert verdict["seeds"] == [1337, 2024] and verdict["seed_line"]["established_all"]["p75"] == pytest.approx(0.02)
    assert verdict["eligible"] == ["L30_D20", "L60_D20", "L30_D60", "L60_D60", "L30_D90"] and verdict["ineligible"] == []
    assert verdict["leaders"]["established_all"] == ["L30_D60", "L60_D60", "L30_D90"]
    assert verdict["winners"] == ["L30_D60", "L60_D60", "L30_D90"]
    assert verdict["selected"] == {"cell": "L30_D60", "lookback_s": 30.0, "segment_s": 60.0} and verdict["decisive"]
    assert verdict["best_on_both_seeds"]["established_all"] is None      # seed 1337's best is L30_D90, seed 2024's L30_D60
    # the flyable constraint removes a cell on ONE seed's failure
    cells[(30.0, 60.0)][2024] = _none_payload(established_all=0.77, established_vectored=0.42, flyable=0.90)
    verdict = g.gate_grid(cells)
    assert verdict["ineligible"] == ["L30_D60"] and verdict["selected"]["cell"] == "L60_D60"
    # the primaries disagree: all-flights established decides, and the note says so
    cells = grid({(30.0, 20.0): (0.80, 0.80, 0.20, 0.20), (30.0, 60.0): (0.70, 0.70, 0.40, 0.40)})
    verdict = g.gate_grid(cells)
    assert verdict["selected"]["cell"] == "L30_D20" and "decides" in verdict["note"]
    # every cell within the seed line: not decisive, the shortest cell is taken
    cells = grid({(30.0, 20.0): (0.70, 0.72, 0.30, 0.32), (60.0, 60.0): (0.71, 0.71, 0.31, 0.31), (120.0, 120.0): (0.71, 0.73, 0.30, 0.32)})
    verdict = g.gate_grid(cells)
    assert not verdict["decisive"] and verdict["selected"]["cell"] == "L30_D20" and "not the deciding" in verdict["note"]
    with pytest.raises(ValueError, match="two seeds"):
        g.gate_grid({(30.0, 20.0): {1337: _none_payload(established_all=0.7, established_vectored=0.3)}})
    with pytest.raises(ValueError, match="protocol"):
        g.gate_grid({(30.0, 20.0): {1337: _payload(vectored_ade=1.0, straight_ade=1.0, flyable=1.0, established=1.0),
                                   2024: _payload(vectored_ade=1.0, straight_ade=1.0, flyable=1.0, established=1.0)}})
