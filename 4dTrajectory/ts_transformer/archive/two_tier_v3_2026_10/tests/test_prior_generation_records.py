"""A free-generation readout's sentences as evaluation records (`experiments.prior_generation_records`): the stored words
flown again are the closed loop's flight, an augmented start's record starts where the executor flew from, and the
readout counts every sentence."""

from __future__ import annotations

import math
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ts_transformer.autopilot.flights import FlightInputs
from ts_transformer.autopilot.frame import AirportCharts
from ts_transformer.autopilot.lateral import Runways
from ts_transformer.data.channels import states_from_channels
from ts_transformer.data.dataset import FlightSeries
from ts_transformer.experiments.prior_free_generation import speak_and_fly
from ts_transformer.experiments.prior_generation_records import (
    augmented_series, best_of_samples, fly_said, reproduced, summarise,
)
from ts_transformer.instructions.signals import signals_from_series
from ts_transformer.instructions.words import UNCHANGED, Words
from ts_transformer.prior import data as prior_data
from ts_transformer.prior.augment import Augmentation, augment_signals, augment_state
from ts_transformer.prior.data import column_classes
from ts_transformer.prior.masks import ProcedureMasks
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.tests.support import fly_legs, instruction_airport, instruction_flight, instruction_spec as spec
from ts_transformer.tests.test_autopilot import DOWNWIND_BASE_FINAL, _params, _physics, vertical_paths
from ts_transformer.tests.test_instruction_vocabulary import _series


def _loop(seed: int, limits: list[float]):
    """Closed loops from one start to each of ``limits`` (s), the prior speaking with ``seed``: ``(what they flew, what was said, and
    the batch's physics)``."""
    one, geometry = spec(), instruction_airport()
    words, params = Words(one), _params()
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    torch.manual_seed(0)
    model = Prior(PriorConfig(classes=column_classes(words, 1), airports=("KXXX",), candidate_slots=1,
                              variant="no-context", d_model=32, layers=2, heads=4, feedforward=64, dropout=0.0),
                  torch.as_tensor(prior_data.candidate_table({"KXXX": geometry}, ("KXXX",), 1))).eval()
    start = replace(signals, **{name: getattr(signals, name)[N_LOOK:] for name in
                                ("time_s", "e_m", "n_m", "altitude_m", "track_deg", "ground_speed_mps",
                                 "vertical_rate_mps")})
    flights = len(limits)
    one_inputs, _, _, one_approach = _physics(start, geometry)
    inputs = FlightInputs(*(torch.cat([getattr(one_inputs, f.name)] * flights) for f in fields(FlightInputs)))
    runways = Runways.of([geometry] * flights, [vertical_paths(geometry)] * flights, dtype=torch.float64,
                         device=torch.device("cpu"))
    charts = AirportCharts.of([geometry] * flights, dtype=torch.float64, device=torch.device("cpu"))
    approach = torch.cat([one_approach] * flights)
    physics = (inputs, runways, charts, approach, limits, words, params)
    flown, said, _, _ = speak_and_fly(model, [signals] * flights, [geometry] * flights, inputs, runways, charts,
                                      approach, limits, words, params, None,
                                      generator=torch.Generator().manual_seed(seed), temperature=1.0,
                                      procedure_masks=ProcedureMasks.none())
    return flown, said, physics


def _same_to(again, flown, j, last):
    assert torch.equal(again.states[j, : last + 1], flown.states[j, : last + 1])
    assert torch.equal(again.commands[j, :last], flown.commands[j, :last])


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_a_said_sentence_flown_again_is_the_flight_the_closed_loop_flew(seed):
    flown, said, physics = _loop(seed, [90.0])
    again = fly_said(physics[0], [said[0]], *physics[1:])
    assert torch.equal(again.done_cycle, flown.done_cycle)
    _same_to(again, flown, 0, int(flown.done_cycle[0]) + 1)


def test_a_batch_flown_again_is_each_flight_the_loop_flew_and_a_cut_sentence_is_until_its_cut():
    flown, said, physics = _loop(5, [90.0, 40.0, 60.0, 75.0])
    assert len(set(flown.done_cycle.tolist())) > 1                       # flights done at different cycles
    step_rows = 2
    # as the readout stores a sentence the glidepath edge stopped: nothing said after the stop's row
    cut = said.copy()
    row = 3
    cut[0, row + 1:] = UNCHANGED
    again = fly_said(physics[0], list(cut), *physics[1:])
    for j in range(1, 4):
        assert int(again.done_cycle[j]) == int(flown.done_cycle[j])
        _same_to(again, flown, j, int(flown.done_cycle[j]) + 1)
    # the cut sentence flies as said up to the end of the stopping step, the span its record keeps
    _same_to(again, flown, 0, min((row + 1) * step_rows, int(flown.done_cycle[0]) + 1))


def _turn():
    geometry = instruction_airport()
    plain, samples = _series(geometry, float(math.radians(0.0)))
    series = FlightSeries(flight_id="KXXX:turn", scenario=plain.scenario, frame=plain.frame, times=plain.times,
                          values=plain.values)
    return geometry, plain, series


def test_an_augmented_series_starts_where_the_executor_flies_from_and_its_rows_are_the_moved_signals():
    geometry, plain, series = _turn()
    move = Augmentation(rotation_deg=11.0, altitude_m=-90.0, speed_scale=1.04)
    moved = augmented_series(series, geometry, move)
    # its row N_LOOK is the executor's augmented start
    state = states_from_channels(moved.times[N_LOOK: N_LOOK + 1], moved.values[N_LOOK: N_LOOK + 1], moved.frame,
                                 mass_kg=float(series.scenario.initial.m))[0][1]
    own = states_from_channels(series.times[N_LOOK: N_LOOK + 1], series.values[N_LOOK: N_LOOK + 1], series.frame,
                               mass_kg=float(series.scenario.initial.m))[0][1]
    lat, lon, altitude, speed, psi = augment_state(own.latitude, own.longitude, own.altitude, own.V, own.psi, geometry,
                                                   move)
    assert (state.latitude, state.longitude) == pytest.approx((lat, lon), abs=1e-9)
    assert (state.altitude, state.V, state.psi) == pytest.approx((altitude, speed, psi), abs=1e-9)
    assert state.gamma == pytest.approx(own.gamma, abs=1e-12)
    # every row is the moved signals' row: rotated, stretched about row N_LOOK, raised, sped up, turned
    expected = augment_signals(signals_from_series(plain, geometry), move)
    again = signals_from_series(SimpleNamespace(**{**vars(plain), "values": moved.values}), geometry)
    for name in ("e_m", "n_m", "altitude_m", "ground_speed_mps", "vertical_rate_mps"):
        assert np.allclose(getattr(again, name), getattr(expected, name), rtol=0.0, atol=1e-6), name
    # (a track is unwrapped from its own first row: the moved one may start a turn away)
    turned = (again.track_deg - expected.track_deg + 180.0) % 360.0 - 180.0
    assert np.allclose(turned, 0.0, rtol=0.0, atol=1e-6)
    # the supervision rows move alike (here they are the observed rows)
    assert np.allclose(moved.supervision_values, moved.values, rtol=0.0, atol=1e-9)


def test_an_unmoved_series_is_the_series():
    geometry, _, series = _turn()
    same = augmented_series(series, geometry, Augmentation(rotation_deg=0.0, altitude_m=0.0, speed_scale=1.0))
    assert np.allclose(same.values, series.values, rtol=0.0, atol=1e-6)


def test_a_re_flown_row_that_differs_from_the_readout_is_named_field_by_field():
    stored = [{"dataset_id": "A", "sample": 0, "outcome": "landed", "end_s": 100.0, "forbidden_mass": {"runway": 0.1}}]
    assert reproduced(stored, [{**stored[0], "forbidden_mass": {}}], ("forbidden_mass",)) == []
    wrong = reproduced(stored, [{**stored[0], "outcome": "timeout", "end_s": 101.0}], ("forbidden_mass",))
    assert wrong == ["A sample 0: outcome 'landed' → 'timeout', end_s 100.0 → 101.0"]


def _row(flight, sample, verdict, ade, fde, outcome="landed", recorded=True, runway=0):
    return {"dataset_id": flight, "sample": sample, "verdict": verdict, "ade_m": ade, "fde_m": fde,
            "outcome": outcome, "recorded": recorded, "observed_runway": 0, "last_runway": runway}


def test_the_pass_rate_counts_every_sentence_and_the_best_of_the_samples_is_per_flight():
    rows = [_row("A", 0, "pass", 300.0, 50.0), _row("A", 1, "fail", 200.0, 900.0, outcome="timeout"),
            _row("B", 0, "not recorded", None, None, outcome="dynamics_failure", recorded=False),
            _row("B", 1, "fail", 800.0, 400.0, runway=1)]
    summary = summarise(rows, errors=True)
    assert summary["sentences"] == 4 and summary["recorded"] == 3
    assert summary["pass_rate"] == 0.25                                   # one not recorded does not pass
    # the verdict judges the observed runway: a sentence ending on another is counted apart
    assert summary["ending_on_observed_runway"] == {"sentences": 3, "pass_rate": 1 / 3}
    assert summary["ending_on_another_runway"] == {"sentences": 1, "verdicts": {"fail": 1}}
    assert summary["ade_m"]["median"] == 300.0 and summary["ade_m"]["count"] == 3
    assert summary["landed_ade_m"]["count"] == 2 and summary["landed_fde_m"]["median"] == 225.0
    best = best_of_samples(rows, errors=True)
    assert best["flights"] == 2 and best["any_sample_passes"] == 0.5
    # A's smallest ADE (200, sample 1) and smallest FDE (50, sample 0) are taken apart; B's only recorded sample
    assert best["min_ade_m"]["median"] == 500.0 and best["min_fde_m"]["median"] == 225.0
    # an augmented start has no truth: no errors at all
    assert "ade_m" not in summarise(rows, errors=False)
    assert "min_ade_m" not in best_of_samples(rows, errors=False)
