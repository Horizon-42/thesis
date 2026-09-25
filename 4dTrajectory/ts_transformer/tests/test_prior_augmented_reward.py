"""Post-training stage 2 (post-training design §4, §5): the augmented start (`prior.augment`, the executor's state moved
alike), the masks the speaker records and the trainer scores under (`Speaker.allowed`, `train.allowed_tensors`), the
tuner without a data term, and the runner's start picking and guarded choice (`experiments.prior_augmented_reward`)."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aerodynamic_model.torch_dynamics import isa_density
from flight_scenarios.fas_geometry import FasCourseGeometry
from ts_transformer.autopilot import replay
from ts_transformer.autopilot.frame import ALT, LAT, LON, MASS, PSI, SPEED, compass_deg
from ts_transformer.autopilot.plant import EXECUTOR_DYNAMICS
from ts_transformer.experiments import prior_augmented_reward as runner
from ts_transformer.experiments.prior_augmented_reward import (
    GUARD_NLL_GROWTH, augmentation_summary, guarded_choice, round_starts, speak_augmented,
)
from ts_transformer.experiments.prior_free_generation import (
    AUGMENT_TRIES, BELOW_GLIDEPATH, augmented_inputs, augmented_starts, speak_and_fly,
)
from ts_transformer.experiments.prior_landing_reward import sentence_flights
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.outputs.constraints.speed_floor import stall_speed_mps
from ts_transformer.instructions.words import ALTITUDE, ANGLE, APPROACH, RUNWAY, UNCHANGED, Words
from ts_transformer.prior.augment import Augmentation, Limits, augment_signals, augment_state, draw, rotate
from ts_transformer.prior.generate import allowed_classes
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.prior.train import RewardConfig, RewardTuner, allowed_tensors, masked
from ts_transformer.tests.test_autopilot import _params
from ts_transformer.tests.test_prior_landing_reward import _sentences, _split
from ts_transformer.tests.test_prior_procedure import _final
from ts_transformer.tests.test_prior_speaker import _flight, _model, _repeated

CPU = torch.device("cpu")
MOVE = Augmentation(rotation_deg=12.0, altitude_m=-90.0, speed_scale=1.04)


def test_a_rotation_turns_bearings_clockwise_and_keeps_distances():
    e, n = rotate(np.array([0.0, 1_000.0]), np.array([1_000.0, 0.0]), 90.0)      # north → east, east → south
    assert np.allclose([e[0], n[0], e[1], n[1]], [1_000.0, 0.0, 0.0, -1_000.0], atol=1e-9)
    e, n = rotate(np.array([3_000.0]), np.array([-4_000.0]), 37.0)
    assert math.hypot(e[0], n[0]) == pytest.approx(5_000.0)


def test_the_draws_stay_inside_their_limits():
    limits, rng = Limits(), np.random.default_rng(0)
    moves = [draw(rng, limits) for _ in range(500)]
    assert max(abs(m.rotation_deg) for m in moves) <= limits.rotation_deg
    assert max(abs(m.altitude_m) for m in moves) <= limits.altitude_m
    assert max(abs(m.speed_scale - 1.0) for m in moves) <= limits.speed_fraction
    assert min(m.rotation_deg for m in moves) < -0.8 * limits.rotation_deg < 0.8 * limits.rotation_deg \
        < max(m.rotation_deg for m in moves)


def test_the_observed_rows_and_the_executor_s_state_move_alike():
    one, geometry, signals, (inputs, *_rest) = _flight()
    moved = augment_signals(signals, MOVE)
    # the start row: rotated only (the stretch's centre), raised; its track turned; distances to the airport kept
    e, n = rotate(signals.e_m[N_LOOK:N_LOOK + 1], signals.n_m[N_LOOK:N_LOOK + 1], MOVE.rotation_deg)
    assert (moved.e_m[N_LOOK], moved.n_m[N_LOOK]) == pytest.approx((e[0], n[0]))
    assert math.hypot(moved.e_m[N_LOOK], moved.n_m[N_LOOK]) == pytest.approx(
        math.hypot(signals.e_m[N_LOOK], signals.n_m[N_LOOK]))
    assert moved.altitude_m[N_LOOK] == pytest.approx(signals.altitude_m[N_LOOK] + MOVE.altitude_m)
    assert moved.track_deg[3] == pytest.approx(signals.track_deg[3] + MOVE.rotation_deg)
    # heights stretch like positions: a climb the prior reads from the rows scales like the state's speed
    sloped = replace(signals, altitude_m=signals.altitude_m + 7.0 * np.arange(len(signals.altitude_m)))
    lifted = augment_signals(sloped, MOVE)
    assert np.allclose(np.diff(lifted.altitude_m), MOVE.speed_scale * np.diff(sloped.altitude_m))
    assert lifted.altitude_m[N_LOOK] == pytest.approx(sloped.altitude_m[N_LOOK] + MOVE.altitude_m)
    # the earlier rows stretched about it: steps 4 % longer, speeds 4 % faster, the times kept
    step = math.hypot(moved.e_m[1] - moved.e_m[0], moved.n_m[1] - moved.n_m[0])
    assert step == pytest.approx(MOVE.speed_scale * math.hypot(signals.e_m[1] - signals.e_m[0],
                                                               signals.n_m[1] - signals.n_m[0]))
    assert moved.ground_speed_mps[2] == pytest.approx(MOVE.speed_scale * signals.ground_speed_mps[2])
    assert np.array_equal(moved.time_s, signals.time_s) and moved.entry_time_utc == signals.entry_time_utc
    # the executor starts where the moved row is, flying the moved track, faster by the same factor
    state = augmented_inputs(inputs, [geometry], [MOVE]).initial_state[0]
    e0, n0 = geometry.frame.horizontal_from_latlon(float(inputs.initial_state[0, LAT]),
                                                   float(inputs.initial_state[0, LON]))
    e1, n1 = geometry.frame.horizontal_from_latlon(float(state[LAT]), float(state[LON]))
    assert (e1, n1) == pytest.approx(tuple(float(x[0]) for x in rotate(np.array([e0]), np.array([n0]),
                                                                        MOVE.rotation_deg)), abs=1e-6)
    assert float(state[ALT]) == pytest.approx(float(inputs.initial_state[0, ALT]) + MOVE.altitude_m)
    assert float(state[SPEED]) == pytest.approx(float(inputs.initial_state[0, SPEED]) * MOVE.speed_scale)
    # ψ is the dynamics' math angle: a clockwise turn takes it off
    assert float(state[PSI]) == pytest.approx(float(inputs.initial_state[0, PSI]) - math.radians(MOVE.rotation_deg))
    # … which in compass degrees is the moved row's track (the fixture's state flies the row's track)
    assert float(compass_deg(state[PSI])) == pytest.approx(moved.track_deg[N_LOOK] % 360.0)
    lat, lon, *_ = augment_state(float(inputs.initial_state[0, LAT]), float(inputs.initial_state[0, LON]), 0.0, 1.0,
                                 0.0, geometry, Augmentation(0.0, 0.0, 1.0))
    assert (lat, lon) == pytest.approx((float(inputs.initial_state[0, LAT]), float(inputs.initial_state[0, LON])))


def _floor(inputs, altitude_m):
    aero = inputs.aero_params[0]
    return EXECUTOR_DYNAMICS.control_speed_floor_margin * float(stall_speed_mps(
        1.0, inputs.initial_state[0, MASS], isa_density(torch.tensor(altitude_m, dtype=torch.float64)), aero[0],
        aero[1]))


def test_a_start_is_drawn_until_it_is_plausible_and_given_up_after_the_tries():
    one, geometry, signals, (inputs, *_rest) = _flight()
    here = float(signals.altitude_m[N_LOOK])
    wide = {"KXXX": (here - 1_000.0, here + 1_000.0)}
    starts = augmented_starts([signals], inputs, np.random.default_rng(0), wide)
    assert starts.draws == [1] and starts.moves[0] is not None
    # a window only the upper half of the altitude draws reaches: redrawn until one does, and every kept one is inside
    upper = {"KXXX": (here + 100.0, here + 1_000.0)}
    starts = augmented_starts([signals] * 40, _repeated(inputs, torch.zeros(40, dtype=torch.long)),
                              np.random.default_rng(1), upper)
    kept = [m for m in starts.moves if m is not None]
    assert all(m.altitude_m >= 100.0 for m in kept) and max(starts.draws) > 1
    assert all(d == AUGMENT_TRIES for m, d in zip(starts.moves, starts.draws) if m is None)
    # outside every window: given up after the tries
    starts = augmented_starts([signals], inputs, np.random.default_rng(0), {"KXXX": (here + 500.0, here + 600.0)})
    assert starts.moves == [None] and starts.draws == [AUGMENT_TRIES]
    # below the executor's stall floor at the new altitude even 5 % faster: given up; at the floor, only a fast enough
    # draw is kept
    slow = replace(inputs, initial_state=inputs.initial_state.clone())
    slow.initial_state[0, SPEED] = 0.9 * _floor(inputs, float(inputs.initial_state[0, ALT]))
    assert augmented_starts([signals], slow, np.random.default_rng(0), wide).moves == [None]
    slow.initial_state[0, SPEED] = _floor(inputs, float(inputs.initial_state[0, ALT]) + 150.0)
    move = augmented_starts([signals], slow, np.random.default_rng(0), wide).moves[0]
    assert move is not None and float(slow.initial_state[0, SPEED]) * move.speed_scale >= _floor(
        inputs, float(inputs.initial_state[0, ALT]) + move.altitude_m)


def test_the_augmentation_readout_counts_the_draws_and_the_sources_given_up():
    batch = SimpleNamespace(signals=[SimpleNamespace(airport=a) for a in "AABAAB"])
    starts = SimpleNamespace(moves=[MOVE, None, MOVE, MOVE, MOVE, None], draws=[1, 10, 3, 2, 1, 10])
    summary = augmentation_summary(batch, starts, [0, 2, 3])            # A's pool ran to flight 3, B's to flight 2
    assert summary["A"] == {"starts": 2, "draws_per_start": 1.5, "redrawn_share": 0.5, "sources_given_up": 1}
    assert summary["B"] == {"starts": 1, "draws_per_start": 3.0, "redrawn_share": 1.0, "sources_given_up": 0}


def test_augmented_sentences_read_the_moved_rows_carry_their_masks_and_train_under_them(monkeypatch):
    """The runner's closed loop from an augmented start: the prior reads the moved rows, a sentence stopped below the
    edge ends at its stop with its masks cut alike, and one masked pass over them is finite."""
    one, geometry, signals, (inputs, runways, charts, approach) = _flight()
    words, samples = Words(one), 4
    index = torch.zeros(samples, dtype=torch.long)
    # the fixture's physics stand in for the runner's (its batch has no series)
    monkeypatch.setattr(runner, "flight_inputs", lambda series, device, anchor: _repeated(inputs, index))
    monkeypatch.setattr(runner, "_physics", lambda batch, device: (_repeated(runways, index), _repeated(charts, index),
                                                                   approach[index]))
    reading = read_flight(signals, geometry, one, words)
    batch = replay.Batch(signals=[signals], series=[None], readings=[reading], geometries=[geometry],
                         crossing_heights=[()], approach_ias_mps=[0.0], groups=["own"], drawn={})
    # an edge high above the start and wide enough to bind wherever the moved start takes the aircraft
    final = _final(crossing_m=1_500.0, faf_d_m=40_000.0, cone=FasCourseGeometry(40_000.0, 41_000.0, 80_000.0))
    model = _model(words)
    sentences = speak_augmented(model, batch, [MOVE], samples, words, _params(), None, {geometry.code: (final,)},
                                generator=torch.Generator().manual_seed(2))
    moved = augment_signals(signals, MOVE)
    assert sentences.flight.tolist() == [0] * samples
    for position, said, allowed in zip(sentences.positions, sentences.said, sentences.allowed):
        assert np.allclose(position[:N_LOOK, 0], moved.e_m[:N_LOOK], atol=1e-3)
        assert np.allclose(position[:N_LOOK, 2], moved.altitude_m[:N_LOOK], atol=1e-3)
        assert len(position) == N_LOOK + len(said)
        assert set(allowed) == {RUNWAY, APPROACH, ANGLE, ALTITUDE}
        assert all(len(codes) == len(said) for codes in allowed.values())
    assert BELOW_GLIDEPATH in sentences.outcomes
    augmented = replace(batch, signals=[moved])
    keep = np.arange(samples)
    split = _split(sentence_flights(augmented, sentences, keep, ("KXXX",), one.step_s, None), words)
    tuner = RewardTuner(_model(words), _model(words), RewardConfig(learning_rate=1e-3, warmup_steps=1, data_weight=0.0),
                        CPU, seed=0)
    passed = tuner.one_pass(split, np.array([1.0, -1.0, 0.5, -0.5]), None, sentences.allowed)
    assert all(math.isfinite(passed[k]) for k in ("reward_mean", "kl_mean"))


def test_the_trainer_scores_a_sentence_under_the_masks_it_was_said_under():
    """What `Speaker.allowed` records unpacks to the classes the masks left, every word said is one of them, and the
    masked scores are the unmasked renormalised over them."""
    one, geometry, signals, (inputs, runways, charts, approach) = _flight()
    words = Words(one)
    model = _model(words)
    final = _final(crossing_m=2_000.0, faf_d_m=12_000.0)            # an edge that binds on the final
    _, said, _, speaker = speak_and_fly(model, [signals], [geometry], inputs, runways, charts, approach, [200.0], words,
                                        _params(), None, generator=torch.Generator().manual_seed(2), temperature=1.0,
                                        finals=[(final,)])
    grid = said[0]
    assert set(speaker.allowed) == {RUNWAY, APPROACH, ANGLE, ALTITUDE}
    packed = {c: np.stack(steps, axis=1)[0] for c, steps in speaker.allowed.items()}
    classes = model.config.classes
    for c, codes in packed.items():
        allowed = allowed_classes(codes, classes[c])
        assert allowed.shape == (len(grid), classes[c])
        spoken = np.where(grid[:, c] != UNCHANGED, grid[:, c] + 1, 0)
        assert allowed[np.arange(len(grid)), spoken].all()           # every word said was allowed
    assert not allowed_classes(packed[ALTITUDE], classes[ALTITUDE]).all()     # and the edge did remove some
    tensors = allowed_tensors([packed], N_LOOK + len(grid) + 3, classes, CPU)
    assert tensors[3].shape == (1, 1, N_LOOK + len(grid) + 3, classes[ALTITUDE])
    assert tensors[3][0, 0, :N_LOOK].all() and tensors[3][0, 0, N_LOOK + len(grid):].all()   # nothing masked there
    assert tensors[HEADING_COLUMN] is None
    logits = [torch.randn(1, 1, N_LOOK + len(grid) + 3, k) for k in classes]
    scored = masked(logits, tensors)
    row, allowed = N_LOOK + 1, tensors[3][0, 0, N_LOOK + 1]
    expected = torch.log_softmax(logits[3][0, 0, row][allowed], dim=-1)
    assert torch.allclose(torch.log_softmax(scored[3][0, 0, row], dim=-1)[allowed], expected)


HEADING_COLUMN = 2


def test_without_a_data_term_the_pass_needs_no_data_and_trains_under_the_masks():
    one, geometry, batch, sentences = _sentences()
    words = Words(one)
    from ts_transformer.experiments.prior_landing_reward import sentence_flights

    split = _split(sentence_flights(batch, sentences, np.arange(3), ("KXXX",), one.step_s, None), words)
    config = RewardConfig(learning_rate=1e-3, warmup_steps=1, data_weight=0.0)
    classes = _model(words).config.classes
    everything = [{c: np.packbits(np.ones((len(s), classes[c]), dtype=bool), axis=1, bitorder="little")
                   for c in (RUNWAY, ALTITUDE)} for s in sentences.said]
    free = RewardTuner(_model(words), _model(words), config, CPU, seed=0).one_pass(split, np.array([1.0, -1.0, 0.0]),
                                                                                   None)
    masked_all = RewardTuner(_model(words), _model(words), config, CPU, seed=0).one_pass(
        split, np.array([1.0, -1.0, 0.0]), None, everything)
    assert free["data_mean"] == 0.0
    # masks that allow everything score as no masks
    assert masked_all["reward_mean"] == pytest.approx(free["reward_mean"]) and masked_all["kl_mean"] == pytest.approx(
        free["kl_mean"])
    with pytest.raises(ValueError, match="sentences' masks for"):
        RewardTuner(_model(words), _model(words), config, CPU, seed=0).one_pass(split, np.zeros(3), None, everything[:1])


def test_a_round_takes_each_airport_s_first_plausible_starts_from_its_pool():
    pool = SimpleNamespace(signals=[SimpleNamespace(airport=a) for a in "AABBABB"])
    moves = [MOVE, None, MOVE, MOVE, MOVE, None, MOVE]
    keep, chosen = round_starts(pool, moves, 2)
    assert keep == [0, 2, 3, 4] and chosen == [MOVE] * 4
    with pytest.raises(ValueError, match="too few plausible starts"):
        round_starts(pool, moves, 3)


def _row(round_number, real, augmented, runway=0.85, heading=15.0, nll=0.28):
    return {"round": round_number, "real_landed": real, "augmented_landed": augmented, "select_tf_nll": nll,
            "real": {"landed_on_observed_runway": runway, "words_after_first_per_flight": {"heading": heading}}}


def test_the_choice_keeps_the_best_augmented_round_within_round_0_s_guards():
    history = [_row(0, 0.92, 0.80), _row(1, 0.93, 0.86), _row(2, 0.905, 0.95),            # 2: real landed dropped
               _row(3, 0.93, 0.95, nll=0.28 * GUARD_NLL_GROWTH + 0.01),                     # 3: drifted off the data
               _row(4, 0.93, 0.87)]
    kept, excluded = guarded_choice(history)
    assert excluded == [2, 3] and kept == 1                    # 4 is within the tie of 1: the earlier
