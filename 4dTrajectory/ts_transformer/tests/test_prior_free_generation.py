"""Single-aircraft free generation (`experiments.prior_free_generation`, prior design §9.1): the prior speaks, the
executor flies, step by step — on one synthetic flight."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from ts_transformer.autopilot.frame import ALT, AirportCharts, read_state
from ts_transformer.autopilot.judge import OUTCOMES, outcome_of
from ts_transformer.experiments.prior_free_generation import reference_grid, speak_and_fly
from ts_transformer.instructions.grammar import step_allowed
from ts_transformer.instructions.words import ANGLE, APPROACH, HEADING, RUNWAY, UNCHANGED, Words
from ts_transformer.prior import data as prior_data
from ts_transformer.prior.data import column_classes
from ts_transformer.prior.masks import ProcedureMasks
from ts_transformer.prior.model import Prior, PriorConfig
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.tests.support import fly_legs, instruction_airport, instruction_flight, instruction_spec as spec
from ts_transformer.tests.test_autopilot import DOWNWIND_BASE_FINAL, _params, _physics


def _speak(seed: int):
    one, geometry = spec(), instruction_airport()
    words, params = Words(one), _params()
    signals = instruction_flight(*fly_legs(DOWNWIND_BASE_FINAL, 270.0, 1110.0, -400.0, 0.0))
    torch.manual_seed(0)
    model = Prior(PriorConfig(classes=column_classes(words, 1), airports=("KXXX",), candidate_slots=1,
                              variant="no-context", d_model=32, layers=2, heads=4, feedforward=64, dropout=0.0),
                  torch.as_tensor(prior_data.candidate_table({"KXXX": geometry}, ("KXXX",), 1))).eval()
    # the executor starts where the prior first speaks: the observed state at row N_LOOK
    start = replace(signals, **{name: getattr(signals, name)[N_LOOK:] for name in
                                ("time_s", "e_m", "n_m", "altitude_m", "track_deg", "ground_speed_mps",
                                 "vertical_rate_mps")})
    inputs, runways, charts, approach = _physics(start, geometry)
    flown, said, forbidden, speaker = speak_and_fly(model, [signals], [geometry], inputs, runways, charts, approach,
                                                    [60.0], words, params, None,
                                                    generator=torch.Generator().manual_seed(seed), temperature=1.0,
                                                    procedure_masks=ProcedureMasks.none())
    return flown, said, geometry, one, inputs, forbidden, signals, speaker


def test_the_prior_speaks_and_the_executor_flies_until_it_is_done():
    flown, said, geometry, one, inputs, forbidden, signals, speaker = _speak(3)
    assert said.shape[0] == 1 and said.shape[2] == 6
    assert (said[0, 0] != UNCHANGED).all()                               # the first step says every column
    assert said.shape[1] <= 60 / one.step_s + 1                          # no step past the time limit
    assert torch.equal(flown.states[0, 0], inputs.initial_state[0])      # flown from the observed state at N_LOOK
    outcome = outcome_of(flown, 0, geometry, int(said[0, 0, RUNWAY]), one)
    assert outcome.outcome in OUTCOMES
    # it stops when the executor is done or at its time limit (60 s: 60 cycles)
    assert int(flown.done_cycle[0]) < 60 and flown.states.shape[1] - 1 <= 60
    # the rows the prior read after the first predicted step are where the executor was at the start of each step
    charts = AirportCharts.of([geometry], dtype=torch.float64, device=torch.device("cpu"))
    for k in range(1, said.shape[1]):
        at = read_state(flown.states[:, 2 * k], charts)
        assert speaker.e[0, N_LOOK + k] == pytest.approx(float(at.e_m[0]))
        assert speaker.h[0, N_LOOK + k] == pytest.approx(float(at.height_m[0]))
    # every word said is one the vocabulary has
    words = Words(one)
    heading = said[0, :, HEADING]
    assert ((heading == UNCHANGED) | ((heading >= 0) & (heading < words.n_heading))).all()
    assert set(said[0, :, APPROACH].tolist()) <= {UNCHANGED, 0, 1, 2}
    # every step said is one the grammar allows at the altitude flown (the mask), and the mass it removed is recorded
    in_force = None
    altitude = flown.states[0, :, ALT].numpy()
    for step, row in enumerate(said[0]):
        height = signals.altitude_m[N_LOOK] if step == 0 else altitude[2 * step]
        assert step_allowed(in_force, row, float(height), one, words), step
        in_force = row.copy() if in_force is None else np.where(row != UNCHANGED, row, in_force)
    assert set(forbidden) == {RUNWAY, APPROACH, ANGLE}
    for mass in forbidden.values():
        assert mass.shape == (1, said.shape[1]) and ((mass >= 0) & (mass <= 1)).all()
    # the same seed, the same flight
    again, same, *_ = _speak(3)
    assert np.array_equal(same, said) and torch.equal(again.states, flown.states)


def test_the_reference_starts_from_the_words_in_force_at_the_first_predicted_step():
    grid = np.full((N_LOOK + 4, 6), UNCHANGED, dtype=np.int64)
    grid[0] = [0, 0, 10, 20, 1, 30]
    grid[3, HEADING] = 12
    grid[N_LOOK, APPROACH] = 1
    grid[N_LOOK + 2, HEADING] = 14
    out = reference_grid(grid)
    assert out[0].tolist() == [0, 1, 12, 20, 1, 30]
    assert np.array_equal(out[1:], grid[N_LOOK + 1:])


def test_the_approach_column_says_only_its_transitions():
    """Multi-aircraft design §6.6 step 8 item 6 (the user's 2026-10-02 decision): not cleared → cleared → go-around →
    cleared; no clearance withdrawn, no go-around from "not cleared" nor at the first step, no word said again — a
    decoding rule beside the labeller's (a runway changed under a clearance still takes the approach with it)."""
    from ts_transformer.instructions.grammar import approach_words_allowed
    from ts_transformer.instructions.words import APPROACH_CLEARED, APPROACH_GO_AROUND, APPROACH_NOT_CLEARED
    from ts_transformer.prior.generate import vocabulary_allowed

    one = spec()
    words = Words(one)
    assert approach_words_allowed(None).tolist() == [True, True, False]
    assert approach_words_allowed(APPROACH_NOT_CLEARED).tolist() == [False, True, False]
    assert approach_words_allowed(APPROACH_CLEARED).tolist() == [False, False, True]
    assert approach_words_allowed(APPROACH_GO_AROUND).tolist() == [False, True, False]
    in_force = np.array([0, APPROACH_CLEARED, 10, words.altitude_land, 3, 5])
    value = np.tile(in_force + 1, (5, 1))                       # classes in force: the word + 1
    value[1, APPROACH], value[3, APPROACH] = APPROACH_NOT_CLEARED + 1, APPROACH_GO_AROUND + 1
    chosen = np.zeros((5, 6), dtype=np.int64)                   # nothing said yet this step
    chosen[4, RUNWAY] = 2                                       # the last: another runway, under the clearance
    opening = np.array([True, False, False, False, False])
    out = vocabulary_allowed(APPROACH, chosen, opening, 4, np.zeros(5, dtype=bool), value, np.full(5, 400.0), words)
    assert out[0, 1:].tolist() == [True, True, False]           # the first step: not cleared or cleared
    assert out[1:4].tolist() == [[True, False, True, False],    # unchanged, or the one transition
                                 [True, False, False, True],
                                 [True, False, True, False]]
    assert out[4].tolist() == [False, False, False, True]       # a runway changed under a clearance: only a go-around


def test_the_grammar_asks_the_labellers_own_rules_of_one_step():
    one = spec()
    words = Words(one)
    land, level, descent = words.altitude_land, 0, 3
    assert not words.is_descent(level) and words.is_descent(descent)
    first = [0, 0, 10, land, descent, 5]
    assert step_allowed(None, first, 900.0, one, words)
    assert not step_allowed(None, [0, 0, 10, land, level, 5], 900.0, one, words)     # rule 3: land needs a descent
    in_force = np.array([0, 1, 10, land, descent, 5])                                 # cleared (1) to land
    still = [UNCHANGED] * 6
    assert step_allowed(in_force, still, 400.0, one, words)
    # a runway changed under the clearance must take the approach with it
    assert not step_allowed(in_force, [1, UNCHANGED, UNCHANGED, UNCHANGED, UNCHANGED, UNCHANGED], 400.0, one, words)
    assert step_allowed(in_force, [1, 0, UNCHANGED, UNCHANGED, UNCHANGED, UNCHANGED], 400.0, one, words)
    # an altitude target below the aircraft with a level angle in force needs a descent class in the same step
    low = words.altitude_index(300.0)
    level_flight = np.array([0, 0, 10, words.altitude_index(900.0), level, 5])
    assert not step_allowed(level_flight, [UNCHANGED, UNCHANGED, UNCHANGED, low, UNCHANGED, UNCHANGED], 900.0, one,
                            words)
    assert step_allowed(level_flight, [UNCHANGED, UNCHANGED, UNCHANGED, low, descent, UNCHANGED], 900.0, one, words)
