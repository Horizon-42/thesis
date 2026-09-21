"""The instruction token (two-tier v3 stage B, B′-dev2): the config binds `plan_conditioning=
'instruction'` to a vocabulary artefact, the token is the words in force over the executor's
segment at the vocabulary's bin centres, ONE source feeds the training row, `predict` and the
closed loop (by flown position there), the head fuses it, the checkpoint binds to the
vocabulary's sha, and the closed-loop protocol must agree with the executor. Untrained models
on synthetic data: the mechanics, never a number."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from ts_transformer.backbone.adapters import build_model
from ts_transformer.config import (
    CONTROL_RECIPE_SIMPLE_V3, PLAN_CONDITIONING_INSTRUCTION, PLAN_CONDITIONING_OFF, PREDICTION_CONTROL, TSConfig,
    recipe_settings,
)
from ts_transformer.data.batch_contract import unpack_batch
from ts_transformer.data.dataset import FixedAnchorTrajectoryWindows, Normalizer, build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.manoeuvre import instructions as ins
from ts_transformer.manoeuvre import lockstep as ls
from ts_transformer.outputs.control import forecast as fc
from ts_transformer.outputs.control import instruction_token as tok
from ts_transformer.outputs.control.forecast import forecast_control_batch
from ts_transformer.outputs.control.strategy import ControlStrategy
from ts_transformer.run_naming import run_display_name
from ts_transformer.tests.support import AIRPORT, OTHER_RUNWAY_WORD, RUNWAY, RUNWAY_WORD, fake_data_provenance
from ts_transformer.training.train import load_checkpoint, train

SEGMENT_S = 20.0
#: The runway word's classes the artefact carries beside the spec (D62); these flights land on RUNWAY.
RUNWAYS = ins.RunwayVocabulary.from_idents([RUNWAY_WORD, OTHER_RUNWAY_WORD])


def _settings(**overrides) -> dict:
    settings = recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False)
    settings.update(dict(
        prediction_output=PREDICTION_CONTROL, control_horizon_s=SEGMENT_S, n_segments=2, control_imitation_loss_weight=0.0,
        final_time_loss_weight=0.0, state_endpoint_loss_weight=0.0, seq_len=8, d_model=16, n_heads=4, d_ff=32,
        e_layers=1, dropout=0.0, device="cpu", epochs=1, patience=1,
    ))
    settings.update(overrides)
    return settings


@pytest.fixture(scope="module")
def vocabulary_path(tmp_path_factory) -> str:
    folder = tmp_path_factory.mktemp("vocabulary")
    path = ins.write_vocabulary(folder, ins.Vocabulary(), runway_vocabulary=RUNWAYS,
                                cohort_identity={"name": "synthetic"}, counts={}, source={})
    return str(path)


@pytest.fixture(scope="module")
def world(vocabulary_path):
    config = TSConfig(**_settings(plan_conditioning=PLAN_CONDITIONING_INSTRUCTION, instruction_vocabulary=vocabulary_path))
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=3, seed=3), config, airport=AIRPORT)
    torch.manual_seed(1)
    model = build_model(config).eval()
    with torch.no_grad():
        model.control_head.control_projection.weight.normal_(std=0.05)
    return config, series, ls.Executor(model=model, config=config, normalizer=Normalizer.fit(series))


def test_the_config_binds_the_vocabulary_path_to_the_instruction_plan(vocabulary_path):
    with pytest.raises(ValueError, match="set instruction_vocabulary"):
        TSConfig(**_settings(plan_conditioning=PLAN_CONDITIONING_INSTRUCTION))
    with pytest.raises(ValueError, match="belongs to plan_conditioning"):
        TSConfig(**_settings(plan_conditioning=PLAN_CONDITIONING_OFF, instruction_vocabulary=vocabulary_path))
    with pytest.raises(ValueError, match="control_horizon_s > 0"):
        TSConfig(**_settings(plan_conditioning=PLAN_CONDITIONING_INSTRUCTION, instruction_vocabulary=vocabulary_path,
                             control_horizon_s=0.0, n_segments=8))
    config = TSConfig(**_settings(plan_conditioning=PLAN_CONDITIONING_INSTRUCTION, instruction_vocabulary=vocabulary_path))
    assert TSConfig.from_dict(config.to_dict()) == config
    name = run_display_name(config.to_dict())
    assert "plan=instruction" in name and "vocab=" in name and "plan=" not in run_display_name(TSConfig(**_settings()).to_dict())


def test_the_artefact_is_read_for_the_instruction_plan_only_and_the_segment_must_be_whole_steps(vocabulary_path):
    config = TSConfig(**_settings(plan_conditioning=PLAN_CONDITIONING_INSTRUCTION, instruction_vocabulary=vocabulary_path))
    vocabulary, runways = tok.load_vocabulary_for(config)
    # the artefact hands back the spec AND the cohort's runway classes; only the spec is hashed
    assert vocabulary == ins.Vocabulary() and runways == RUNWAYS and tok.instruction_positions(config, vocabulary) == 2
    assert tok.instruction_token_width(config, vocabulary) == 2 * ins.Vocabulary.CONDITIONING_WIDTH
    with pytest.raises(ValueError, match="reads no instruction vocabulary"):
        tok.load_vocabulary_for(TSConfig(**_settings()))
    with pytest.raises(ValueError, match="whole number"):
        tok.load_vocabulary_for(replace(config, control_horizon_s=25.0, n_segments=5))


def test_the_token_is_the_words_in_force_over_the_segment_at_the_bin_centres(world):
    config, series, _executor = world
    vocabulary, runways = tok.load_vocabulary_for(config)
    item = series[0]
    reading = ins.read_instructions(item, vocabulary, runways)
    start = float(item.times[10])
    token = tok.instruction_context(reading, start, config, vocabulary)[tok.INSTRUCTION_KEY]
    assert token.shape == (2, ins.Vocabulary.CONDITIONING_WIDTH) and token.dtype == np.float32
    expected = vocabulary.conditioning(reading.words_at(np.array([start, start + vocabulary.token_step_s])))
    assert np.array_equal(token, expected)
    # the nearest truth row: a flown row ON row k reads row k's time at distance 0; a row pushed
    # 300 m sideways reads the nearest row (at most 300 m away, since row k itself is) and how far
    off = np.array(item.values[12], dtype=np.float64)
    off[0] += 300.0
    time_s, distance_m = tok.nearest_truth_time_s(item, off)
    assert tok.nearest_truth_time_s(item, item.values[12]) == (float(item.times[12]), 0.0)
    assert 0.0 < distance_m <= 300.0 and time_s in set(np.asarray(item.times, dtype=np.float64).tolist())
    with pytest.raises(FileNotFoundError, match="instruction_vocabulary"):
        tok.load_vocabulary_for(replace(config, instruction_vocabulary=str(config.instruction_vocabulary) + ".missing"))


def test_a_training_batch_and_the_probe_carry_the_token(world):
    config, series, _executor = world
    context = unpack_batch(FixedAnchorTrajectoryWindows(series, config, Normalizer.fit(series)).batch([0]))[5]
    assert tok.INSTRUCTION_KEY in context and tuple(context[tok.INSTRUCTION_KEY].shape) == (1, 2, ins.Vocabulary.CONDITIONING_WIDTH)
    probe = ControlStrategy().probe_context(3, torch.device("cpu"), config)
    assert tuple(probe[tok.INSTRUCTION_KEY].shape) == (3, 2, ins.Vocabulary.CONDITIONING_WIDTH)
    assert set(probe) == set(context), "the probe's key set is a real training batch's"
    # a no-token run's batch carries nothing of it
    plain = TSConfig(**_settings())
    plain_context = unpack_batch(FixedAnchorTrajectoryWindows(series, plain, Normalizer.fit(series)).batch([0]))[5]
    assert tok.INSTRUCTION_KEY not in plain_context


def test_the_head_fuses_the_token_and_predict_reads_the_truth_s_words(world):
    config, series, executor = world
    model = executor.model                                                             # `ControlOutput` mixes the fusion in
    assert model.instruction_given and model.instruction_vocabulary_sha256 == ins.Vocabulary().sha256
    assert model.feature_fusion[0].in_features == (config.enc_in + 2) * config.d_model
    anchor = int(config.seq_len) - 1
    forecasts = forecast_control_batch(executor.model, series, config, executor.normalizer, anchor, torch.device("cpu"))
    assert len(forecasts) == len(series) and all(f.values.shape[1] == len(config.channels) for f in forecasts)


def test_the_checkpoint_binds_to_the_vocabulary_s_sha(world, tmp_path):
    config, _series, executor = world
    strategy = ControlStrategy()
    extras = strategy.checkpoint_payload_extras(config, executor.model)                  # the sha the encoder was sized against
    assert extras == {"instruction_vocabulary_sha256": ins.Vocabulary().sha256}
    strategy.verify_checkpoint_payload(config, extras)                                   # the artefact still matches
    with pytest.raises(ValueError, match="is not the one this executor trained against"):
        strategy.verify_checkpoint_payload(config, {"instruction_vocabulary_sha256": "0" * 64})
    other = ins.write_vocabulary(tmp_path, ins.Vocabulary(token_step_s=5.0), runway_vocabulary=RUNWAYS, cohort_identity={}, counts={}, source={})
    moved = replace(config, instruction_vocabulary=str(other))
    with pytest.raises(ValueError, match="is not the one this executor trained against"):
        strategy.verify_checkpoint_payload(moved, extras)
    assert strategy.checkpoint_payload_extras(TSConfig(**_settings()), executor.model) == {}


def test_a_trained_instruction_executor_round_trips_through_the_checkpoint(tmp_path, vocabulary_path):
    """`train` stores the model's vocabulary sha in the payload; `load_checkpoint` rebuilds the
    head from the artefact and refuses one whose sha moved or that is gone."""
    config = TSConfig(**_settings(plan_conditioning=PLAN_CONDITIONING_INSTRUCTION, instruction_vocabulary=vocabulary_path,
                                  epochs=1, patience=1, batch_size=4))
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=6, seed=3), config, airport=AIRPORT)
    train(series, config, output_dir=tmp_path, data_provenance=fake_data_provenance(), verbose=False)
    model, loaded, _normalizer, payload = load_checkpoint(tmp_path / "checkpoint.pt")
    assert payload["instruction_vocabulary_sha256"] == ins.Vocabulary().sha256 == model.instruction_vocabulary_sha256
    assert loaded.plan_conditioning == PLAN_CONDITIONING_INSTRUCTION and loaded.instruction_vocabulary == vocabulary_path
    # the artefact rewritten under other bins: refused by sha; gone: refused by name
    moved = tmp_path / "moved"
    moved.mkdir()
    other = ins.write_vocabulary(moved, ins.Vocabulary(token_step_s=5.0), runway_vocabulary=RUNWAYS, cohort_identity={}, counts={}, source={})
    stored = dict(payload)
    stored["config"] = {**payload["config"], "instruction_vocabulary": str(other)}
    torch.save(stored, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="is not the one this executor trained against"):
        load_checkpoint(tmp_path / "other.pt")
    stored["config"] = {**payload["config"], "instruction_vocabulary": str(moved / "gone.json")}
    torch.save(stored, tmp_path / "gone.pt")
    with pytest.raises(FileNotFoundError, match="instruction_vocabulary"):
        load_checkpoint(tmp_path / "gone.pt")


def test_the_closed_loop_feeds_the_truth_s_words_by_flown_position_and_the_protocol_must_fit(world):
    """The feed is the whole-record reading of every flight; round 0 hands the executor exactly
    the token `predict` builds at the same anchor (the ONE source), every round records where its
    words came from, and the executor, the protocol and the feed must agree."""
    config, series, executor = world
    vocabulary, runways = tok.load_vocabulary_for(config)
    feed = ls.InstructionFeed.read(vocabulary, runways, series)
    cpu = torch.device("cpu")
    a0 = int(config.seq_len) - 1
    # round 0 == predict: the same reading, the anchor's own row, distance 0
    runs = [ls.FlightRun(series=item, anchor=a0, horizon_s=100.0) for item in series]
    dynamics, sources = ls._dynamics(executor, runs, series, a0, cpu, feed)
    predict = fc._dynamics_batch(series, a0, cpu, config)
    assert torch.equal(dynamics[tok.INSTRUCTION_KEY], predict[tok.INSTRUCTION_KEY])
    assert all(s["instruction_truth_distance_m"] == 0.0 and s["instruction_truth_time_s"] == float(item.times[a0]) for s, item in zip(sources, series, strict=True))
    lines = []
    runs = ls.fly(executor, series, device=cpu, batch_size=2, log=lines.append, protocol=ls.PROTOCOL_TRUTH_INSTRUCTION, feed=feed)
    assert len(runs) == 3 and all(run.ended in (ls.ENDED_CROSSED, ls.ENDED_HORIZON) for run in runs)
    assert lines and lines[0].startswith(f"    {ls.PROTOCOL_TRUTH_INSTRUCTION} round 0:")
    for run in runs:
        assert run.predictions == len(run.legs) == len(run.rounds)
        for leg in run.legs[:-1]:
            assert float(np.sum(leg.sample_durations_s)) == pytest.approx(SEGMENT_S)
        assert all({"instruction_truth_time_s", "instruction_truth_distance_m"} <= set(record) for record in run.rounds)
        assert run.rounds[0]["instruction_truth_distance_m"] == 0.0
    # a cohort cut at a later first row flies under the SAME whole-record feed: the words at the
    # cut row are the whole record's, not a cut record's
    cut, first_rows = ls.from_row(series, config, a0 + 5)
    assert cut and all(first_rows[item.dataset_id] == a0 + 5 for item in cut)
    runs = [ls.FlightRun(series=item, anchor=a0, horizon_s=100.0) for item in cut]
    dynamics, _sources = ls._dynamics(executor, runs, cut, a0, cpu, feed)
    whole = np.stack([tok.instruction_context(feed.readings[item.dataset_id], float(item.times[a0]), config, vocabulary)[tok.INSTRUCTION_KEY] for item in cut])
    assert np.array_equal(dynamics[tok.INSTRUCTION_KEY].numpy(), whole)
    ls.fly(executor, cut, device=cpu, batch_size=2, protocol=ls.PROTOCOL_TRUTH_INSTRUCTION, feed=feed)
    # refusals: the wrong protocol, no feed, a feed under `none`, a feed missing a flight, another vocabulary
    with pytest.raises(ValueError, match="does not fit an executor"):
        ls.fly(executor, series, device=cpu, batch_size=2, protocol=ls.PROTOCOL_NONE)
    with pytest.raises(ValueError, match="needs instruction feed"):
        ls.fly(executor, series, device=cpu, batch_size=2, protocol=ls.PROTOCOL_TRUTH_INSTRUCTION)
    with pytest.raises(ValueError, match="holds no reading"):
        ls.fly(executor, series, device=cpu, batch_size=2, protocol=ls.PROTOCOL_TRUTH_INSTRUCTION, feed=ls.InstructionFeed.read(vocabulary, runways, series[:1]))
    with pytest.raises(ValueError, match="another vocabulary"):
        ls.fly(executor, series, device=cpu, batch_size=2, protocol=ls.PROTOCOL_TRUTH_INSTRUCTION,
               feed=ls.InstructionFeed(ins.Vocabulary(token_step_s=5.0), runways, feed.readings))
    # other runway CLASSES are another vocabulary too: the same word index would mean another threshold
    with pytest.raises(ValueError, match="another vocabulary"):
        ls.fly(executor, series, device=cpu, batch_size=2, protocol=ls.PROTOCOL_TRUTH_INSTRUCTION,
               feed=ls.InstructionFeed(vocabulary, ins.RunwayVocabulary.from_idents([RUNWAY_WORD]), feed.readings))
    plain = TSConfig(**_settings())
    torch.manual_seed(1)
    no_token = ls.Executor(model=build_model(plain).eval(), config=plain, normalizer=executor.normalizer)
    with pytest.raises(ValueError, match="does not fit an executor"):
        ls.fly(no_token, series, device=cpu, batch_size=2, protocol=ls.PROTOCOL_TRUTH_INSTRUCTION, feed=feed)
    with pytest.raises(ValueError, match="takes no instruction feed"):
        ls.fly(no_token, series, device=cpu, batch_size=2, protocol=ls.PROTOCOL_NONE, feed=feed)
    with pytest.raises(ValueError, match="not one of"):
        ls.fly(no_token, series, device=cpu, batch_size=2, protocol="prior")
