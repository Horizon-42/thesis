"""Landing-reward fine-tuning (prior design §9.3, `experiments.prior_landing_reward`): the reward, the flights' own
comparison, the loss terms, the trainer's direction and the guarded choice."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ts_transformer.autopilot import replay
from ts_transformer.experiments.prior_free_generation import speak_and_fly, steps_said
from ts_transformer.autopilot.judge import OUTCOMES
from ts_transformer.experiments.prior_landing_reward import (
    Sentences, against_the_direction, guarded_choice, join, reward_summary, sentence_flights, write_sentences,
)
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.words import UNCHANGED, Words
from ts_transformer.prior import data as prior_data
from ts_transformer.prior.data import Split, column_classes
from ts_transformer.prior.landing_reward import LANDED, group_advantages, landing_direction, rewards
from ts_transformer.prior.scene import CONTEXT_WINDOW_S, N_LOOK, utc_s
from ts_transformer.prior.train import (
    RewardConfig, RewardTuner, batch_logits, column_nll, flight_kl, flight_nll, to_batch,
)
from ts_transformer.tests.support import instruction_airport
from ts_transformer.tests.test_autopilot import _params
from ts_transformer.tests.test_prior import _landings, _signals, _two_runways
from ts_transformer.tests.test_prior_closed_loop import _flight, _model

CPU = torch.device("cpu")


def test_the_landing_direction_is_the_runways_landed_on_in_the_half_hour_before_and_their_parallels():
    signals, geometry = _signals(), _two_runways()                       # candidates 09 (090°) and 27 (270°)
    first = utc_s(signals.entry_time_utc) + float(signals.time_s[N_LOOK])
    assert landing_direction(signals, geometry, _landings(times_27=[first - 600.0])).tolist() == [False, True]
    assert landing_direction(signals, geometry, _landings(times_09=[first - 60.0])).tolist() == [True, False]
    # outside the window, or at the step itself: no landing in the window — any runway
    quiet = _landings(times_27=[first - CONTEXT_WINDOW_S - 1.0, first])
    assert landing_direction(signals, geometry, quiet).tolist() == [True, True]
    # the flight's own landing is never counted (it must be in the pool, as in the prior's input)
    with pytest.raises(ValueError, match="the flight's own must be there"):
        landing_direction(signals, geometry, _landings(own=False))


def test_a_sentence_earns_its_reward_by_landing_in_the_direction_and_is_compared_with_its_flight_s_others():
    allowed = [np.array([False, True])] * 4
    earned = rewards(["landed", "landed", "timeout", "crossed_off_runway"], [1, 0, 1, 1], allowed)
    assert earned.tolist() == [1.0, 0.0, 0.0, 0.0]                        # landed against the direction earns nothing
    advantages = group_advantages(np.array([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]))
    assert np.allclose(advantages, [[0.5, -0.5, 0.5, -0.5], [0, 0, 0, 0], [-0.25, -0.25, -0.25, 0.75]])
    assert LANDED in OUTCOMES                                             # the mirror names the judge's outcome


def _sentences(samples=3, seed=2):
    """``samples`` sentences said to the synthetic flight, as the runner keeps them."""
    one, geometry, signals, (inputs, runways, charts, approach) = _flight()
    words, params = Words(one), _params()
    index = torch.zeros(samples, dtype=torch.long)
    flown, said, _, speaker = speak_and_fly(_model(words), [signals] * samples, [geometry] * samples,
                                            inputs.take(index), runways.take(index), charts.take(index),
                                            approach[index], [40.0] * samples, words, params, None,
                                            generator=torch.Generator().manual_seed(seed), temperature=1.0)
    steps = [steps_said(flown, j, said.shape[1], round(one.step_s / flown.cycle_s)) for j in range(samples)]
    reading = read_flight(signals, geometry, one, words)
    batch = replay.Batch(signals=[signals], series=[], readings=[reading], geometries=[geometry], crossing_heights=[],
                         approach_ias_mps=[], groups=[], drawn={})
    sentences = Sentences(np.zeros(samples, dtype=np.int64),
                          [np.column_stack((speaker.e[j, :N_LOOK + s], speaker.n[j, :N_LOOK + s],
                                            speaker.h[j, :N_LOOK + s])) for j, s in enumerate(steps)],
                          [said[j, :s] for j, s in enumerate(steps)], ["landed"] * samples, np.zeros(samples, int),
                          np.zeros(samples, int))
    return one, geometry, batch, sentences


def _cut(flight, rows):
    """The flight's first ``rows`` rows."""
    from dataclasses import replace
    return replace(flight, **{name: getattr(flight, name)[:rows] for name in
                              ("features", "relative", "in_force", "since", "targets", "asked")})


def _split(flights, words):
    table = prior_data.candidate_table({"KXXX": instruction_airport()}, ("KXXX",), 1)
    return Split(flights, ("KXXX",), table, (("09",),), ((90.0,),), column_classes(words, 1), "no-context")


def test_a_sentence_trains_on_its_own_words_and_chunks_join_in_flight_order():
    one, geometry, batch, sentences = _sentences()
    flights = sentence_flights(batch, sentences, np.arange(3), ("KXXX",), one.step_s, None)
    for flight, said in zip(flights, sentences.said):
        assert flight.rows == N_LOOK + len(said) and flight.asked[N_LOOK:].all()
        assert np.array_equal(flight.targets[N_LOOK:], np.where(said != UNCHANGED, said + 1, 0))
    joined = join([sentences, sentences], [0, 5])
    assert joined.flight.tolist() == [0, 0, 0, 5, 5, 5] and len(joined.said) == 6


def test_the_per_flight_terms_add_up_to_the_columns_and_the_kl_is_zero_at_the_reference():
    one, geometry, batch, sentences = _sentences()
    words = Words(one)
    split = _split(sentence_flights(batch, sentences, np.arange(3), ("KXXX",), one.step_s, None), words)
    model, batch_ = _model(words), to_batch(split, [0, 1, 2], CPU)
    logits = batch_logits(model, batch_)
    per_flight = flight_nll(logits, batch_["targets"], batch_["present"], batch_["asked"])
    per_column = column_nll(logits, batch_["targets"], batch_["present"], batch_["asked"])
    assert per_flight.shape == (3,) and torch.allclose(per_flight.sum(), per_column.sum())
    # scene by scene, on sentences of unequal length (padding in the batch): each is its own single-scene total
    uneven = _split([_cut(f, n) for f, n in zip(split.flights, (N_LOOK + 5, N_LOOK + 9, N_LOOK + 13))], words)
    both = to_batch(uneven, [0, 1, 2], CPU)
    together = flight_nll(batch_logits(model, both), both["targets"], both["present"], both["asked"])
    for b in range(3):
        alone = to_batch(uneven, [b], CPU)
        single = column_nll(batch_logits(model, alone), alone["targets"], alone["present"], alone["asked"]).sum()
        assert float(together[b].detach()) == pytest.approx(float(single.detach()), rel=1e-5)
    assert torch.allclose(flight_kl(logits, logits, batch_["targets"], batch_["present"], batch_["asked"]),
                          torch.zeros(3), atol=1e-6)
    moved = _model(words)
    with torch.no_grad():
        for parameter in moved.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    kl = flight_kl(batch_logits(moved, batch_), logits, batch_["targets"], batch_["present"], batch_["asked"])
    assert (kl > 0).all()


def test_the_reward_term_raises_a_better_sentence_s_words_and_lowers_a_worse_one_s():
    one, geometry, batch, sentences = _sentences()
    words = Words(one)
    split = _split(sentence_flights(batch, sentences, np.arange(3), ("KXXX",), one.step_s, None), words)
    config = RewardConfig(learning_rate=1e-3, warmup_steps=1, kl_weight=0.0, data_weight=0.0)

    def nll_after(advantage: float) -> float:
        model = _model(words)
        RewardTuner(model, _model(words), config, CPU, seed=0).one_pass(
            split.subset([0]), np.array([advantage]), split)
        with torch.no_grad():
            b = to_batch(split, [0], CPU)
            return float(flight_nll(batch_logits(model, b), b["targets"], b["present"], b["asked"])[0])

    with torch.no_grad():
        b = to_batch(split, [0], CPU)
        before = float(flight_nll(batch_logits(_model(words), b), b["targets"], b["present"], b["asked"])[0])
    assert nll_after(1.0) < before < nll_after(-1.0)
    # dropout is off: at the first update the model is its reference and the pull is 0
    dropping = _model(words)
    dropping.layers[0].dropout.p = 0.5
    first = RewardTuner(dropping, _model(words), RewardConfig(warmup_steps=1, learning_rate=0.0), CPU,
                        seed=0).one_pass(split.subset([0]), np.array([1.0]), split)
    assert first["kl_mean"] == pytest.approx(0.0, abs=1e-7)
    with pytest.raises(ValueError, match="advantages for"):
        RewardTuner(_model(words), _model(words), config, CPU, seed=0).one_pass(split, np.zeros(1), split)
    with pytest.raises(ValueError, match="no sentence to train on"):
        RewardTuner(_model(words), _model(words), config, CPU, seed=0).one_pass(split.subset([]), np.zeros(0), split)


def test_the_round_kept_is_the_best_within_the_guards():
    def row(number, landed, on_runway, heading):
        return {"round": number, "select_landed": landed,
                "select": {"landed_on_observed_runway": on_runway, "words_after_first_per_flight": {"heading": heading}}}

    history = [row(0, 0.90, 0.85, 18.0), row(1, 0.93, 0.84, 17.0), row(2, 0.97, 0.80, 18.0), row(3, 0.96, 0.85, 25.0),
               row(4, 0.935, 0.86, 20.0)]
    kept, excluded = guarded_choice(history)
    assert excluded == [2, 3]                          # the runway guard, then the heading guard
    assert kept == 1                                   # 0.935 best; 0.93 within the tie, earlier
    assert guarded_choice([*history, row(5, 0.0, None, 30.0)]) == (1, [2, 3, 5])   # landed nothing: excluded


def test_a_round_s_sentences_are_summarised_and_stored(tmp_path):
    one, geometry, batch, sentences = _sentences(samples=4)
    sentences = Sentences(sentences.flight, sentences.positions, sentences.said,
                          ["landed", "landed", "timeout", "landed"], np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]))
    earned = rewards(sentences.outcomes, sentences.runway, [np.array([True])] * 4)
    summary = reward_summary(batch, sentences, earned, 4)
    assert summary["reward_mean"] == 0.75 and summary["flights_with_contrast"] == 1
    assert summary["rewards_per_flight"] == {"3": 1} and summary["landed_on_observed_runway"] == 1.0
    advantages = group_advantages(earned.reshape(-1, 4)).reshape(-1)
    write_sentences(tmp_path / "sentences.npz", batch, sentences, earned, advantages)
    stored = np.load(tmp_path / "sentences.npz")
    assert np.array_equal(stored["reward"], earned) and np.array_equal(stored["advantage"], advantages)
    assert stored["step_offsets"][-1] == len(stored["said"]) == sum(len(s) for s in sentences.said)
    rows = [{"dataset_id": "a", "outcome": "landed", "last_runway": 1}, {"dataset_id": "a", "outcome": "landed",
            "last_runway": 0}, {"dataset_id": "b", "outcome": "timeout", "last_runway": 1}]
    assert against_the_direction(rows, {"a": np.array([True, False]), "b": np.array([True, False])}) == 1 / 3
