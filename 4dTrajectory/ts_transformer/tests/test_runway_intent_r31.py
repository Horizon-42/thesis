"""Runway-intent R3.1's error model and predictions (`experiments.runway_intent_r31`) and its gates (`..._r31_readout`)."""

import numpy as np
import pytest

from ts_transformer.experiments.runway_intent_r31 import POOLED, ErrorModel, error_grid, predictions
from ts_transformer.experiments.runway_intent_r31_readout import gates
from ts_transformer.inference.calibration import MIN_CALIBRATION_FLIGHTS
from ts_transformer.inference.runway_schedule import Arrival, SampledSlots, Separation, Slot, sample_schedules


def test_the_error_model_cuts_at_the_terciles_and_falls_back_to_the_pool_when_thin():
    predicted = np.arange(300.0)                       # terciles at ~100 and ~200 s
    errors = np.where(predicted < 100, -5.0, np.where(predicted < 200, 0.0, 20.0))
    model = ErrorModel.fit(predicted, errors)
    assert model.edges_s == pytest.approx((99.67, 199.33), abs=0.01)
    assert model.stratum(50.0) == "T tercile 1" and model.stratum(150.0) == "T tercile 2" and model.stratum(250.0) == "T tercile 3"
    assert np.median(model.residuals[model.stratum(250.0)]) == 20.0
    # a stratum under MIN_CALIBRATION_FLIGHTS is dropped and its flights draw from the pooled errors
    thin = ErrorModel.fit(np.arange(250.0), np.zeros(250), bins=10)      # 25 flights a bin
    assert set(thin.residuals) == {POOLED} and thin.stratum(260.0) == POOLED and len(thin.residuals[POOLED]) == 250
    assert 250 // 10 < MIN_CALIBRATION_FLIGHTS
    partial = ErrorModel((100.0, 200.0), {POOLED: np.zeros(3), "T tercile 1": np.zeros(40), "T tercile 2": np.zeros(40)})
    assert partial.stratum(250.0) == POOLED and partial.stratum(150.0) == "T tercile 2"


def test_predictions_read_the_median_the_modal_runway_and_the_calibrated_eta():
    model = ErrorModel((100.0, 200.0), {POOLED: np.array([10.0]), "T tercile 3": np.array([-10.0, 20.0, 30.0])})
    row = {"flight_key": "a", "anchor_s": 1000.0, "etas_s": {"17L": 1300.0, "17R": 1310.0},
           "independent": {"runway": "17L", "time_s": 1300.0}, "truth": {"runway": "17L", "time_s": 1290.0}}
    samples = {"a": SampledSlots(("17R", "17L", "17R", "17L"), np.array([1280.0, 1300.0, 1330.0, 1290.0]),
                                 np.array([0.0, 0.0, 20.0, 12.0]))}
    predictions([row], samples, samples, {"a": np.array([30.0, 20.0, -10.0, 10.0])}, model,
                {"a": Slot("a", "17L", 1300.0, 1300.0, "F")})
    s = row["stochastic"]
    assert s["runway"] == "17L"                    # a 2-2 tie goes to the head's top runway
    assert s["time_s"] == pytest.approx(1295.0) and s["delay_s"] == pytest.approx(6.0) and s["p_delayed"] == 0.5
    assert row["stochastic_causal"] == s
    assert row["error_stratum"] == "T tercile 3"   # predicted remaining 300 s
    assert row["calibrated"]["time_s"] == pytest.approx(1300.0 - 15.0)   # the median of the flight's own draws
    assert row["deterministic"] == {"runway": "17L", "time_s": 1300.0, "delay_s": 0.0}


def test_a_flight_nobody_interacts_with_gets_exactly_the_calibrated_prediction():
    # the draws are the stratum's quantile grid in a random order: the stochastic median of a flight
    # alone is its calibrated prediction, whatever the permutation (with draws taken with replacement
    # the two differed by 1-2.5 s and the noise read as an interaction effect)
    rng = np.random.default_rng(3)
    residuals = rng.normal(2.0, 20.0, size=400)
    model = ErrorModel((1e9, 2e9), {POOLED: residuals, "T tercile 1": residuals})
    rules = Separation(same_nm=3.0, speed_mps=70.0)
    arrivals = [Arrival("a", {"17L": 1300.0}, {"17L": 0.0}, "F"), Arrival("b", {"17L": 9000.0}, {"17L": 0.0}, "F")]
    draws = {a.key: error_grid(residuals, 200, rng) for a in arrivals}
    assert np.allclose(np.sort(draws["a"]), np.sort(draws["b"]))       # one grid, two orders
    samples = sample_schedules(arrivals, draws, rules, delay_weight_per_s=1 / 60, min_probability=0.01)
    rows = [{"flight_key": a.key, "anchor_s": 1000.0, "etas_s": dict(a.etas), "independent": {"runway": "17L", "time_s": a.etas["17L"]},
             "truth": {"runway": "17L", "time_s": 0.0}} for a in arrivals]
    predictions(rows, samples, samples, draws, model, {a.key: Slot(a.key, "17L", a.etas["17L"], a.etas["17L"], "F") for a in arrivals})
    for r in rows:
        assert r["stochastic"]["time_s"] == pytest.approx(r["calibrated"]["time_s"], abs=1e-6)
        assert r["stochastic"]["delay_s"] == 0.0


def test_the_causal_form_places_the_flights_in_the_order_they_became_known():
    rules = Separation(same_nm=3.0, speed_mps=70.0)
    early = Arrival("early", {"17L": 100.0}, {"17L": 0.0}, "F")
    late = Arrival("late", {"17L": 130.0}, {"17L": 0.0}, "F")
    zero = {"early": np.zeros(2), "late": np.zeros(2)}
    fcfs = sample_schedules([late, early], zero, rules, delay_weight_per_s=1 / 60, min_probability=0.01)
    causal = sample_schedules([late, early], zero, rules, delay_weight_per_s=1 / 60, min_probability=0.01,
                              known_at={"late": 0.0, "early": 10.0})
    assert fcfs["early"].delays_s[0] == 0.0 and fcfs["late"].delays_s[0] > 0.0
    assert causal["late"].delays_s[0] == 0.0 and causal["early"].delays_s[0] > 0.0


def _artifact(busy: tuple[float, float] | None, every: tuple[float, float], moved: tuple[int, float, float]) -> dict:
    def cell(stoch: float, calib: float) -> dict:
        return {"stochastic": {"time_error_s": {"p50": stoch}}, "calibrated": {"time_error_s": {"p50": calib}}}

    summary = {"all": {**cell(*every), "moved": {"flights": moved[0], "stochastic": {"p50": moved[1]}, "calibrated": {"p50": moved[2]}}}}
    if busy is not None:
        summary["busy"] = cell(*busy)
    return {"summary": summary}


def test_the_gates_read_the_calibrated_baseline_and_leave_an_unmeasured_gate_open():
    verdicts = gates({
        "PASS": _artifact((11.0, 12.0), (12.5, 12.0), (20, 15.0, 16.0)),
        "WORSE": _artifact((11.0, 12.0), (13.5, 12.0), (20, 15.0, 16.0)),     # all hours 1.5 s worse: over the 1 s tolerance
        "MOVED": _artifact((11.0, 12.0), (12.0, 12.0), (20, 17.0, 16.0)),
        "NOBUSY": _artifact(None, (12.0, 12.0), (20, 15.0, 16.0)),
        "NOMOVED": _artifact((11.0, 12.0), (12.0, 12.0), (0, float("nan"), float("nan"))),
    })
    assert verdicts["PASS"]["passed"] is True
    assert verdicts["WORSE"]["passed"] is False and verdicts["MOVED"]["passed"] is False
    assert verdicts["NOBUSY"]["passed"] is None and verdicts["NOMOVED"]["passed"] is None
