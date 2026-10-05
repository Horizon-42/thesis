"""The M1 loop's control flow, on a scripted judge and a fake solver (no NLP, no data)."""

from types import SimpleNamespace

import numpy as np
import pytest

from traffic import loop, rules
from traffic.check import Check, Conflict

LOSS = dict(kind=rules.RADAR_OR_VERTICAL, required_m=5556.0, distance_m=4000.0, vertical_m=100.0, above=True,
            responsible=True)


def _check(*conflicts):
    return Check(np.arange(0.0, 10.0), tuple(conflicts), 0)


def _run(monkeypatch, visual_checks, *, fail_solve_at=None, max_rounds=5, baseline_error=None):
    """Run the loop with ``visual_checks`` as the judge's successive VISUAL answers (IFR: no loss)."""
    solves = []

    def shortest(*_a, **_k):
        if baseline_error:
            raise ValueError(baseline_error)
        return SimpleNamespace(pc="IAF", dense_times=[1.0, 2.0], decision_vector="x0", final_time=100.0)

    def solve(pc, *_a, initial_guess, extra_rows, **_k):
        solves.append(initial_guess)
        if fail_solve_at == len(solves):
            raise ValueError("collocation free-time optimization failed: Infeasible_Problem_Detected")
        return SimpleNamespace(pc=pc, dense_times=[1.0, 2.0], decision_vector=f"x{len(solves)}",
                               final_time=100.0 + len(solves))

    fake_so = SimpleNamespace(
        iaf_setup=lambda scenario, root: ("target", ["IAF"], "aircraft", 60.0),
        shortest_iaf_solve=shortest, solve_iaf=solve,
        iaf_result=lambda solve, *a, **k: SimpleNamespace(final_time_s=solve.final_time, simulator_states=[]),
    )
    window = SimpleNamespace(recorded=[SimpleNamespace(flight_key="OTHER", lat_deg=np.array([35.0]), runway="09",
                                                       end_utc_s=100.0)],
                             flight_key="OWN", uncategorised_types=(), runways={}, runway="09", t0_utc_s=0.0,
                             rules=SimpleNamespace(along_nm={"09": 0.0}, speed_mps=70.0),
                             frame=SimpleNamespace(east_scale_error=lambda lats: 0.002))
    answers = iter(visual_checks)
    monkeypatch.setattr(loop, "so", fake_so)
    monkeypatch.setattr(loop, "make_window", lambda *a, **k: window)
    monkeypatch.setattr(loop, "FlownTrack", SimpleNamespace(from_samples=lambda s: SimpleNamespace(lat_deg=np.array([35.0]))))
    monkeypatch.setattr(loop, "check", lambda w, f, *, reading, step_s: next(answers) if reading == rules.VISUAL else _check())
    monkeypatch.setattr(loop, "extra_rows", lambda specs, *a, **k: specs)
    result, side = loop.fly_in_traffic(
        SimpleNamespace(), None, procedure_root="root", settings=loop.LoopSettings(max_rounds=max_rounds),
        max_duration=2000.0, rollout_dt_s=0.5, solve_options={})
    return result, side, solves


def test_a_loss_removed_by_one_re_solve_is_separated(monkeypatch):
    result, side, solves = _run(monkeypatch, [_check(Conflict(5.0, 0, **LOSS)), _check()])
    assert side["outcome"] == loop.SEPARATED and len(side["rounds"]) == 2
    assert solves == ["x0"]                                         # warm-started from the baseline
    assert result.final_time_s == 101.0
    assert side["rounds"][0]["rows_next"][0]["family"] == "horizontal"   # 4000/5556 > 100/305: horizontal
    assert side["branches"] == {"OTHER": {"family": "horizontal", "sign": 1.0}}


def test_a_loss_that_comes_back_asks_for_twice_its_margin(monkeypatch):
    loss = Conflict(5.0, 0, **LOSS)
    _result, side, solves = _run(monkeypatch, [_check(loss), _check(loss), _check()])
    assert side["outcome"] == loop.SEPARATED and len(solves) == 2
    bounds = [r["rows_next"][0]["bound"] for r in side["rounds"][:2]]
    assert bounds == pytest.approx([5556.0 * 1.01, 5556.0 * 1.02])


def test_a_moving_loss_is_unresolved_after_max_rounds(monkeypatch):
    checks = [_check(Conflict(float(t), 0, **LOSS)) for t in range(1, 5)]
    _result, side, solves = _run(monkeypatch, checks, max_rounds=2)
    assert side["outcome"] == loop.UNRESOLVED and len(solves) == 2 and len(side["rounds"]) == 3


def test_a_failed_re_solve_keeps_the_last_good_record(monkeypatch):
    result, side, _solves = _run(monkeypatch, [_check(Conflict(5.0, 0, **LOSS))], fail_solve_at=1)
    assert side["outcome"] == loop.SOLVE_FAILED and result.final_time_s == 100.0
    assert "Infeasible" in side["rounds"][0]["next_solve_error"]


def test_a_failed_baseline_is_its_own_error(monkeypatch):
    with pytest.raises(loop.BaselineFailed, match="all 2 IAF"):
        _run(monkeypatch, [], baseline_error="all 2 IAF(s) infeasible")


def test_an_aircraft_in_loss_at_the_start_gets_rows_only_after_its_first_free_instant(monkeypatch):
    at_start = [Conflict(t, 0, **LOSS) for t in (0.0, 1.0, 2.0)]
    assert loop.free_from(_check(*at_start), 1) == [3.0]
    assert loop.free_from(_check(Conflict(4.0, 0, **LOSS)), 1) == [0.0]
    _result, side, solves = _run(monkeypatch, [_check(*at_start)])
    assert side["outcome"] == loop.SEPARATED_AT_BASELINE and solves == []
    assert side["starts_in_loss"] == {"OTHER": 3.0}
    assert not any(loss[7] for loss in side["rounds"][0]["losses"])


def test_an_aircraft_in_loss_at_every_instant_is_null_in_the_sidecar(monkeypatch):
    import json
    always = [Conflict(float(t), 0, **LOSS) for t in range(10)]
    assert loop.free_from(_check(*always), 1) == [float("inf")]
    _result, side, solves = _run(monkeypatch, [_check(*always)])
    assert side["starts_in_loss"] == {"OTHER": None} and solves == []
    json.dumps(side, allow_nan=False)                       # the batch writes it so


def test_two_kinds_at_one_instant_both_get_rows_and_double_once_a_round(monkeypatch):
    trail = Conflict(5.0, 0, rules.IN_TRAIL, 5556.0, 4000.0, 100.0, True, True)
    wake = Conflict(5.0, 0, rules.AT_THRESHOLD, 9260.0, 4000.0, 100.0, True, True)
    _result, side, _solves = _run(monkeypatch, [_check(trail, wake), _check(trail, wake), _check()])
    first, second = side["rounds"][0]["rows_next"], side["rounds"][1]["rows_next"]
    assert sorted(r["family"] for r in first) == ["horizontal", "landing_after"]
    assert next(r["bound"] for r in first if r["family"] == "horizontal") == pytest.approx(5556.0 * 1.01)
    assert next(r["bound"] for r in second if r["family"] == "horizontal") == pytest.approx(5556.0 * 1.02)


def test_a_branch_is_kept_across_rounds_until_an_in_trail_loss(monkeypatch):
    vertical = dict(LOSS, distance_m=1000.0, vertical_m=250.0)          # vertical margin larger: vertical
    horizontal = dict(LOSS, distance_m=5000.0, vertical_m=10.0)         # would prefer horizontal alone
    trail = Conflict(7.0, 0, rules.IN_TRAIL, 5556.0, 4000.0, 100.0, True, True)
    _result, side, _solves = _run(monkeypatch, [_check(Conflict(5.0, 0, **vertical)),
                                                _check(Conflict(6.0, 0, **horizontal)), _check(trail), _check()])
    families = [{r["family"] for r in rnd["rows_next"]} for rnd in side["rounds"][:3]]
    assert families == [{"vertical"}, {"vertical"}, {"horizontal"}]
    assert side["branches"] == {"OTHER": {"family": "horizontal", "sign": 1.0}}
