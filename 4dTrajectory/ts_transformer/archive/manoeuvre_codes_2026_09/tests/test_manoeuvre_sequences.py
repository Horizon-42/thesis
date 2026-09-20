"""Code sequences (`manoeuvre/sequences.py`): a flight's full segments from the anchor through
a frozen codebook, the state token at every boundary, what remains as the landed fraction, and
the operating-day split."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ts_transformer.config import TSConfig
from ts_transformer.data.channels import POSITION_IDX
from ts_transformer.data.dataset import build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.manoeuvre import sequences as sq
from ts_transformer.manoeuvre import tokenizer as tok
from ts_transformer.manoeuvre.segments import segment_start_times, state_row
from ts_transformer.tests.support import AIRPORT, RUNWAY

DT_S = 2.0
IDENTITY = {"eligible_set_sha256": {"KRDU": "b" * 64}}


@pytest.fixture(scope="module")
def cohort():
    config = TSConfig(seq_len=8, n_segments=4, d_model=16, n_heads=4, d_ff=32, e_layers=1, device="cpu",
                      horizon_mode="normalized", epochs=1, patience=1, batch_size=8, val_fraction=0.25, test_fraction=0.25)
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=6, seed=3), config, airport=AIRPORT)
    return series


@pytest.fixture(scope="module")
def codebook(tmp_path_factory):
    torch.manual_seed(0)
    tokenizer = tok.tokenizer_for("learned", levels=(4, 4), segment_s=30.0, dt_s=DT_S)
    return tok.write_codebook(tmp_path_factory.mktemp("cb") / "cb", tokenizer, segment_s=30.0, dt_s=DT_S,
                              data_identity=IDENTITY, source={})


def test_the_state_token_is_position_speed_and_the_course_as_a_unit_vector():
    row = np.array([4000.0, -2000.0, 600.0, 60.0, 80.0, -3.0])
    token = sq.state_token(row)
    np.testing.assert_allclose(token, [0.2, -0.1, 0.3, 1.0, 0.6, 0.8], atol=1e-6)
    assert token.dtype == np.float32 and len(sq.STATE_TOKEN_FEATURES) == 6


def test_a_flight_becomes_its_full_segments_codes_and_boundary_states(cohort, codebook):
    anchor = 7
    sequences = sq.flight_sequences(cohort, codebook, anchor, batch_size=4)
    assert len(sequences) == len(cohort)
    for item, sequence in zip(cohort, sequences, strict=True):
        starts = segment_start_times(item, anchor, 30.0)
        assert sequence.length == len(starts) and sequence.codes.shape == (len(starts),) and sequence.z.shape == (len(starts), 2)
        assert sequence.states.shape == (len(starts) + 1, 6)
        np.testing.assert_allclose(sequence.states[0], sq.state_token(item.values[anchor]), atol=1e-6)
        np.testing.assert_allclose(sequence.states[-1], sq.state_token(state_row(item, float(starts[-1]) + 30.0)), atol=1e-6)
        remaining = float(item.supervision_times[-1]) - (float(item.times[anchor]) + 30.0 * len(starts))
        assert sequence.landed_fraction == pytest.approx(remaining / 30.0) and 0.0 <= sequence.landed_fraction < 1.0
        assert bool((sequence.codes >= 0).all()) and bool((sequence.codes < 16).all())
        assert sequence.typecode == str(item.scenario.aircraft.code) and sequence.runway_course_rad == pytest.approx(float(item.scenario.target.psi))
        # the same codes one flight at a time, and the codes are the codebook's own reading
        alone = sq.flight_sequence(item, codebook, anchor)
        assert np.array_equal(alone.codes, sequence.codes) and np.allclose(alone.z, sequence.z)
        assert np.allclose(codebook.tokenizer.codes_to_z(torch.from_numpy(sequence.codes)).numpy(), sequence.z)


def test_the_continuous_coordinates_round_to_the_codes_z(cohort, codebook):
    item = cohort[0]
    sequence = sq.flight_sequence(item, codebook, 7)
    from ts_transformer.manoeuvre.segments import truth_segment_rows
    rows = np.stack([truth_segment_rows(item, float(start), 30.0, DT_S) for start in sequence.start_times]).astype(np.float32)
    states = np.stack([state_row(item, float(start)) for start in sequence.start_times]).astype(np.float32)
    continuous = codebook.encode_continuous(rows, states)
    assert continuous.shape == sequence.z.shape and bool((np.abs(continuous) <= 1.0).all())
    # rounding the continuous coordinates to the grid gives the sequence's codes
    rounded = codebook.tokenizer.z_to_codes(torch.from_numpy(continuous))
    assert np.array_equal(rounded.numpy(), sequence.codes)
    assert np.allclose(codebook.encode_continuous(rows[0], states[0]), continuous[0])
    command = tok.write_codebook(codebook.path.parent / "cv", tok.tokenizer_for("command-vocabulary", levels=(), segment_s=30.0, dt_s=DT_S),
                                 segment_s=30.0, dt_s=DT_S, data_identity=IDENTITY, source={})
    with pytest.raises(ValueError, match="continuous"):
        command.encode_continuous(rows, states)


def test_the_operating_day_split_keeps_a_day_together_and_is_a_function_of_the_day(cohort):
    days = [sq.operational_day_of(item) for item in cohort]
    assert all(len(day) == 10 for day in days)
    train, val = sq.split_by_operational_day(cohort, val_fraction=0.5, seed=7)
    assert len(train) + len(val) == len(cohort)
    train_days = {sq.operational_day_of(item) for item in train}
    assert not train_days & {sq.operational_day_of(item) for item in val}
    again_train, _again_val = sq.split_by_operational_day(list(reversed(cohort)), val_fraction=0.5, seed=7)
    assert {item.dataset_id for item in again_train} == {item.dataset_id for item in train}
    with pytest.raises(ValueError, match="val_fraction"):
        sq.split_by_operational_day(cohort, val_fraction=1.0, seed=7)


def test_continuous_targets_align_with_the_sequences(cohort, codebook):
    sequences = sq.flight_sequences(cohort, codebook, 7)
    targets = sq.continuous_targets(cohort, sequences, codebook, batch_size=3)
    assert len(targets) == len(sequences)
    for sequence, target in zip(sequences, targets, strict=True):
        assert target.shape == sequence.z.shape
        assert np.array_equal(codebook.tokenizer.z_to_codes(torch.from_numpy(target)).numpy(), sequence.codes)
    with pytest.raises(ValueError, match="is not"):
        sq.continuous_targets(list(reversed(cohort)), sequences, codebook)
