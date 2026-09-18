"""The executor side of the manoeuvre-code token (plan §2.6, P1.3): the config's rules, the
run name, the context row (the truth segment + the anchor state), the two z sources, the joint
gradient into the encoder, a train → export codebook → retrain-against-it → refusal chain,
and the record fields."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ts_transformer.backbone.adapters import build_model
from ts_transformer.config import (
    CONTROL_RECIPE_SIMPLE_V3,
    PLAN_CONDITIONING_MANOEUVRE_CODE,
    PREDICTION_CONTROL,
    TSConfig,
    recipe_settings,
)
from ts_transformer.data.batch_contract import model_forward
from ts_transformer.data.channels import POSITION_IDX, VELOCITY_IDX
from ts_transformer.data.dataset import FixedAnchorTrajectoryWindows, Normalizer, build_series
from ts_transformer.data.synthetic import synthetic_arrivals
from ts_transformer.experiments.manoeuvre_codebook import export_codebook
from ts_transformer.inference.export import build_prediction_record
from ts_transformer.inference.forecast import forecast_approaches
from ts_transformer.manoeuvre import tokenizer as tok
from ts_transformer.outputs.control.forecast import forecast_control_batch
from ts_transformer.outputs.control.plan_token import (
    MANOEUVRE_SEGMENT_KEY,
    MANOEUVRE_STATE_KEY,
    MANOEUVRE_Z_KEY,
    manoeuvre_code_count,
    manoeuvre_tokenizer_for,
    plan_token_width,
    probe_plan_context,
)
from ts_transformer.outputs.control.supervision import probe_dynamics
from ts_transformer.run_naming import run_display_name
from ts_transformer.tests.support import AIRPORT, RUNWAY, fake_data_provenance
from ts_transformer.training.train import load_checkpoint, train

HORIZON_S = 20.0


def _config(**overrides) -> TSConfig:
    """The L1c executor recipe (plan §2.6) shrunk to a smoke size: the simple-v3 base with the
    duration-deciding terms off, a fixed 20 s horizon of two 10 s segments, a learned K = 16
    tokenizer trained jointly."""
    settings = recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False)
    settings.update(dict(
        prediction_output=PREDICTION_CONTROL,
        plan_conditioning=PLAN_CONDITIONING_MANOEUVRE_CODE,
        manoeuvre_fsq_levels=(4, 4),
        control_horizon_s=HORIZON_S,
        n_segments=2,
        control_imitation_loss_weight=0.0,
        control_heading_rate_loss_weight=0.0,
        control_bank_tv_loss_weight=0.0,
        final_time_loss_weight=0.0,
        state_endpoint_loss_weight=0.0,
        checkpoint_selection_metric="fixed-anchor-common-grid-ade",
        lr_plateau_metric="objective",
        seq_len=8, d_model=16, n_heads=4, d_ff=32, e_layers=1, dropout=0.0,
        device="cpu", horizon_mode="normalized", epochs=1, patience=1, batch_size=8,
        val_fraction=0.25, test_fraction=0.25,
    ))
    settings.update(overrides)
    return TSConfig(**settings)


@pytest.fixture(scope="module")
def cohort():
    config = _config()
    series, _report = build_series(synthetic_arrivals(AIRPORT, RUNWAY, n_flights=8, seed=3), config, airport=AIRPORT)
    return series


# ── the config ───────────────────────────────────────────────────────────────

def test_the_manoeuvre_code_token_needs_a_segment_and_a_tokenizer_and_nothing_else_may_carry_them():
    with pytest.raises(ValueError, match="control_horizon_s"):
        _config(control_horizon_s=0.0)
    with pytest.raises(ValueError, match="FSQ levels"):
        _config(manoeuvre_fsq_levels=())
    with pytest.raises(ValueError, match="must be empty"):
        _config(manoeuvre_tokenizer="command-vocabulary")
    _config(manoeuvre_tokenizer="command-vocabulary", manoeuvre_fsq_levels=())      # baseline B
    with pytest.raises(ValueError, match="unknown manoeuvre_tokenizer"):
        _config(manoeuvre_tokenizer="k-means")
    for moved in ({"manoeuvre_fsq_levels": (4, 4)}, {"manoeuvre_tokenizer": "command-vocabulary"},
                  {"manoeuvre_codebook": "/cb"}):
        with pytest.raises(ValueError, match="belong"):
            _config(**{"plan_conditioning": "off", "manoeuvre_fsq_levels": (), **moved})
    with pytest.raises(ValueError, match="belongs to the control output"):
        TSConfig(prediction_output="state", manoeuvre_fsq_levels=(4, 4))


def test_the_levels_are_one_form_and_round_trip_through_json():
    config = _config(manoeuvre_fsq_levels=[8, 4])            # an arm file's list
    assert config.manoeuvre_fsq_levels == (8, 4)
    stored = json.loads(json.dumps(config.to_dict()))
    assert stored["manoeuvre_fsq_levels"] == [8, 4]
    assert TSConfig.from_dict(stored).manoeuvre_fsq_levels == (8, 4)
    assert manoeuvre_code_count(config) == 32 and plan_token_width(config) == 2
    command = _config(manoeuvre_tokenizer="command-vocabulary", manoeuvre_fsq_levels=())
    assert manoeuvre_code_count(command) == 63 and plan_token_width(command) == 63
    off = _config(plan_conditioning="off", manoeuvre_fsq_levels=())
    assert manoeuvre_code_count(off) == 0 and plan_token_width(off) == 0


def test_the_masking_share_is_retired_at_zero():
    """`plan_conditioning_dropout` is a constant now: a stored 0.0 is dropped, a stored 0.5
    (the archived L1 / L1b arms) is refused by name."""
    stored = _config().to_dict()
    assert "plan_conditioning_dropout" not in stored
    assert TSConfig.from_dict({**stored, "plan_conditioning_dropout": 0.0}).plan_conditioning == PLAN_CONDITIONING_MANOEUVRE_CODE
    with pytest.raises(ValueError, match="retired to the constant"):
        TSConfig.from_dict({**stored, "plan_conditioning_dropout": 0.5})
    with pytest.raises(TypeError):
        _config(plan_conditioning_dropout=0.0)


def test_the_run_name_says_which_intent_space():
    name = run_display_name(_config().to_dict())
    assert "plan=manoeuvre-code" in name and "fsq=4/4" in name and "ctrl-horizon=20" in name
    assert "tok=" not in name                                              # the default is silent
    command = run_display_name(_config(manoeuvre_tokenizer="command-vocabulary", manoeuvre_fsq_levels=()).to_dict())
    assert "tok=command-vocabulary" in command and "fsq=" not in command


# ── the context row ──────────────────────────────────────────────────────────

def test_the_window_set_carries_the_truth_segment_and_the_anchor_state(cohort):
    config = _config()
    windows = FixedAnchorTrajectoryWindows(cohort, config, Normalizer.fit(cohort))
    _x, _y, _w, final_time, _fw, dynamics = windows.batch(np.array([0, 1, 2]))
    rows = int(HORIZON_S / config.dt_s) + 1
    segment, state = dynamics[MANOEUVRE_SEGMENT_KEY], dynamics[MANOEUVRE_STATE_KEY]
    assert segment.shape == (3, rows, 6) and segment.dtype == torch.float32
    assert state.shape == (3, 6) and state.dtype == torch.float32
    assert torch.allclose(final_time, torch.full((3,), HORIZON_S))
    for row, index in enumerate([0, 1, 2]):
        s_idx, anchor = windows.index[index]
        item = cohort[s_idx]
        np.testing.assert_allclose(state[row].numpy(), item.values[anchor], rtol=1e-6)
        speed = np.hypot(*item.values[anchor, list(VELOCITY_IDX[:2])])
        np.testing.assert_allclose(segment[row, 0].numpy(), [0.0, 0.0, 0.0, speed, 0.0, item.values[anchor, VELOCITY_IDX[2]]], atol=1e-3)
        assert segment[row, -1, POSITION_IDX[0]] > 0.0                  # flies ahead
    # the batch-size probe carries the same keys and shapes
    probe = probe_dynamics(4, torch.device("cpu"), config)
    assert probe[MANOEUVRE_SEGMENT_KEY].shape == (4, rows, 6) and probe[MANOEUVRE_STATE_KEY].shape == (4, 6)
    assert set(probe_plan_context(config, 2, torch.device("cpu"))) == {MANOEUVRE_SEGMENT_KEY, MANOEUVRE_STATE_KEY}


# ── the executor ─────────────────────────────────────────────────────────────

def test_the_executor_reads_z_from_the_truth_segment_or_from_a_given_z(cohort):
    torch.manual_seed(0)
    config = _config()
    model = build_model(config).eval()
    assert isinstance(model.manoeuvre_tokenizer, tok.LearnedTokenizer) and model.codebook_sha256 is None
    windows = FixedAnchorTrajectoryWindows(cohort, config, Normalizer.fit(cohort))
    x, _y, _w, _t, _fw, dynamics = windows.batch(np.array([0, 1, 2]))
    with torch.no_grad():
        model.control_head.control_projection.weight.normal_(std=0.1)   # starts at zero
        truth = model_forward(model, x, dynamics)
        z, codes = model.manoeuvre_tokenizer(dynamics[MANOEUVRE_SEGMENT_KEY], dynamics[MANOEUVRE_STATE_KEY])
        given = {k: v for k, v in dynamics.items() if k not in (MANOEUVRE_SEGMENT_KEY, MANOEUVRE_STATE_KEY)}
        given[MANOEUVRE_Z_KEY] = z
        handed = model_forward(model, x, given)
    assert truth.manoeuvre_code.shape == (3,) and torch.equal(truth.manoeuvre_code, codes)
    assert torch.equal(handed.manoeuvre_code, codes) and torch.allclose(handed.controls, truth.controls)
    # another code changes the schedule: the executor is conditioned on z
    other = {**given, MANOEUVRE_Z_KEY: model.manoeuvre_tokenizer.codes_to_z((codes + 5) % 16)}
    with torch.no_grad():
        assert not torch.allclose(model_forward(model, x, other).controls, truth.controls)
    with pytest.raises(ValueError, match="one z source"):
        model_forward(model, x, {**dynamics, MANOEUVRE_Z_KEY: z})
    with pytest.raises(ValueError, match="one z source"):
        model_forward(model, x, {k: v for k, v in given.items() if k != MANOEUVRE_Z_KEY})


def test_joint_training_reaches_the_encoder_through_z(cohort):
    torch.manual_seed(0)
    config = _config()
    model = build_model(config).train()
    windows = FixedAnchorTrajectoryWindows(cohort, config, Normalizer.fit(cohort))
    x, _y, _w, _t, _fw, dynamics = windows.batch(np.array([0, 1, 2, 3]))
    with torch.no_grad():
        model.control_head.control_projection.weight.normal_(std=0.1)
    prediction = model_forward(model, x, dynamics)
    prediction.controls.square().sum().backward()
    encoder_grads = [p.grad for p in model.manoeuvre_tokenizer.encoder.parameters() if p.grad is not None]
    assert encoder_grads and any(bool((g != 0).any()) for g in encoder_grads)
    # the command vocabulary has nothing to train and no gradient to receive
    command = build_model(_config(manoeuvre_tokenizer="command-vocabulary", manoeuvre_fsq_levels=()))
    assert not list(command.manoeuvre_tokenizer.parameters())
    assert model_forward(command, x, dynamics).manoeuvre_code.shape == (4,)


# ── train, export, retrain against the codebook, refuse a swapped one ───────

def test_train_export_the_codebook_retrain_against_it_and_refuse_a_swapped_one(tmp_path: Path, cohort):
    config = _config()
    provenance = fake_data_provenance()
    train(cohort, config, output_dir=tmp_path / "joint", data_provenance=provenance, verbose=False)
    history = json.loads((tmp_path / "joint" / "history.json").read_text())["history"]
    usage = history[0]["control_training_diagnostics"]["manoeuvre_codes"]
    assert usage["count"] == 16 and usage["used"] + usage["unused"] == 16 and sum(usage["counts"]) > 0
    assert history[0]["control_training_diagnostics"]["gradient_norm_pre_clip"]["mean"]["manoeuvre_tokenizer"] >= 0.0
    metadata = json.loads((tmp_path / "joint" / "checkpoint_metadata.json").read_text())
    assert "codebook_sha256" not in metadata and metadata["control_recipe"]["plan"] == PLAN_CONDITIONING_MANOEUVRE_CODE
    joint_model, loaded, normalizer, _payload = load_checkpoint(tmp_path / "joint" / "checkpoint.pt")
    assert loaded.manoeuvre_fsq_levels == (4, 4)

    # predict: every forecast says which code it flew and that it came from the truth
    forecasts = forecast_approaches(joint_model, cohort[:2], loaded, normalizer, device=torch.device("cpu"))
    assert all(f.manoeuvre_code_source == "truth" and 0 <= f.manoeuvre_code < 16 for f in forecasts)
    assert all(f.final_time_s == pytest.approx(HORIZON_S) for f in forecasts)
    record = build_prediction_record(cohort[0], forecasts[0], index=0, model_name=loaded.model, horizon_mode=loaded.horizon_mode)
    assert record.source["manoeuvreCode"] == forecasts[0].manoeuvre_code
    assert record.source["manoeuvreCodeSource"] == "truth"
    # ...or from a z handed in (the prior's, in protocol A)
    anchor = loaded.seq_len - 1
    z = joint_model.manoeuvre_tokenizer.codes_to_z(torch.tensor([3, 9]))
    given = forecast_control_batch(
        joint_model, cohort[:2], loaded, normalizer, anchor, torch.device("cpu"),
        dynamics={**{k: v for k, v in _truth_dynamics(joint_model, cohort[:2], loaded, normalizer, anchor).items()
                     if k not in (MANOEUVRE_SEGMENT_KEY, MANOEUVRE_STATE_KEY)}, MANOEUVRE_Z_KEY: z},
    )
    assert [f.manoeuvre_code for f in given] == [3, 9] and all(f.manoeuvre_code_source == "given" for f in given)

    # export the codebook, then train an executor AGAINST it
    codebook = export_codebook(tmp_path / "joint" / "checkpoint.pt", tmp_path / "cb")
    assert codebook.levels == (4, 4) and codebook.segment_s == HORIZON_S
    assert codebook.source["checkpoint_sha256"] == metadata["checkpoint_sha256"]
    assert codebook.data_identity["eligible_set_sha256"] == {}                     # the fake provenance has none
    with pytest.raises(FileExistsError):
        export_codebook(tmp_path / "joint" / "checkpoint.pt", tmp_path / "cb")
    frozen = _config(manoeuvre_codebook=str(tmp_path / "cb"))
    tokenizer, sha = manoeuvre_tokenizer_for(frozen)
    assert sha == codebook.sha256 and not any(p.requires_grad for p in tokenizer.parameters())
    with pytest.raises(ValueError, match="config says"):
        manoeuvre_tokenizer_for(_config(manoeuvre_codebook=str(tmp_path / "cb"), manoeuvre_fsq_levels=(8, 4)))
    train(cohort, frozen, output_dir=tmp_path / "against", data_provenance=provenance, verbose=False)
    against = json.loads((tmp_path / "against" / "checkpoint_metadata.json").read_text())
    assert against["codebook_sha256"] == codebook.sha256
    model, loaded_against, _n, _p = load_checkpoint(tmp_path / "against" / "checkpoint.pt")
    assert loaded_against.manoeuvre_codebook == str(tmp_path / "cb")
    for key, value in codebook.tokenizer.state_dict().items():
        assert torch.equal(model.manoeuvre_tokenizer.state_dict()[key], value)       # frozen, unchanged
    with pytest.raises(ValueError, match="nothing new to export"):
        export_codebook(tmp_path / "against" / "checkpoint.pt", tmp_path / "cb2")
    assert "codebook" in run_display_name(loaded_against.to_dict())

    # swap the codebook under the same path: the executor no longer loads
    torch.manual_seed(99)
    other = tok.tokenizer_for("learned", levels=(4, 4), segment_s=HORIZON_S, dt_s=config.dt_s)
    import shutil
    shutil.rmtree(tmp_path / "cb")
    tok.write_codebook(tmp_path / "cb", other, segment_s=HORIZON_S, dt_s=config.dt_s,
                       data_identity={"eligible_set_sha256": {"KRDU": "e" * 64}}, source={})
    with pytest.raises(ValueError, match="differ from the codebook"):
        load_checkpoint(tmp_path / "against" / "checkpoint.pt")


def _truth_dynamics(model, series, config, normalizer, anchor):
    from ts_transformer.outputs.control.forecast import _dynamics_batch
    return _dynamics_batch(series, anchor, torch.device("cpu"), config)
