"""The two-head duration (anytime / calibrated-ETA design §三 3.1b, B1.b).

``duration_head='two-head'`` carries BOTH heads at once. The POINT head drives the rollout
duration, exactly as under ``point`` — that is where B1_point_matched's path gain came from
(the duration term's weight, not the quantile head) — and the QUANTILE head emits nothing
but the published ETA distribution B2 calibrates and B3 decodes, which is where B1's
arrival-time gain came from. The two gains do not overlap, and this is the head that takes
both without the quantile head's path cost being forced onto the trajectory.

What is asserted here: the point head is what the rollout flies (never q50); both weights
bind and price their own term; every refusal names its reason; and the record, the
calibration runner, ``predict --cta-from-quantiles`` and the ETA-error readout all work on a
two-head checkpoint. `point` / `quantile` are unchanged — that is a bit-exactness harness
run, reported in the commit, not something a unit test can see.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from batch_contract import model_forward
from config import (
    CONTROL_DURATION_UNIFORM,
    CONTROL_STATE_CLOCK_OBSERVED,
    CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
    CTA_CONDITIONING_GIVEN,
    DEFAULT_DURATION_QUANTILE_LOSS_WEIGHT,
    DURATION_HEAD_POINT,
    DURATION_HEAD_QUANTILE,
    DURATION_HEAD_TWO_HEAD,
    DURATION_MEDIAN_INDEX,
    DURATION_QUANTILES,
    PREDICTION_CONTROL,
    PREDICTION_STATE,
    TSConfig,
)
from control.envelope import CONTROL_LOWER, CONTROL_UPPER
from data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from dataset import build_series
from forecast import duration_quantile_predictions
from io_utils import file_sha256
from models import build_model
import objective
from objective import DURATION_QUANTILE_COMPONENT, loss_component_names
from prediction_outputs import ControlPrediction, QuantileFinalTimeHead, pinball_duration_loss
from run_naming import run_display_name, run_slug
from synthetic import synthetic_arrivals
from train import load_checkpoint, train

from tests.test_duration_quantiles import _config, _dynamics

AIRPORT, RUNWAY = "KRDU", "05L"

PROVENANCE = {
    "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
    "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64,
                   "source_records": []}],
}


def _two_head(**overrides) -> TSConfig:
    return _config(duration_head=DURATION_HEAD_TWO_HEAD, **overrides)


def _history(config: TSConfig, batch: int = 4) -> torch.Tensor:
    return torch.randn(batch, config.seq_len, config.enc_in)


# ── the two heads ───────────────────────────────────────────────────────────

def test_two_head_builds_both_heads_and_the_other_values_build_one():
    """`final_time_head` stays the POINT head's name and shape under `two-head`, so the
    state-dict keys a `point` run trains are the keys this run trains; the quantile head is
    a second module beside it, and neither `point` nor `quantile` gains a parameter."""
    torch.manual_seed(0)
    two = build_model(_two_head())
    point = build_model(_config(duration_head=DURATION_HEAD_POINT))
    quantile = build_model(_config(duration_head=DURATION_HEAD_QUANTILE))

    assert two.final_time_head.network[-1].out_features == 1
    assert two.duration_quantile_head.network[-1].out_features == len(DURATION_QUANTILES)
    assert isinstance(two.duration_quantile_head, QuantileFinalTimeHead)
    assert point.duration_quantile_head is None and quantile.duration_quantile_head is None
    # The point head's own keys are untouched; the second head adds keys of its own.
    point_keys = {k for k in point.state_dict() if k.startswith("final_time_head.")}
    assert {k for k in two.state_dict() if k.startswith("final_time_head.")} == point_keys
    assert any(k.startswith("duration_quantile_head.") for k in two.state_dict())
    assert not any(k.startswith("duration_quantile_head.") for k in quantile.state_dict())
    # ...and the quantile head is bound ONCE: binding it under a second name would publish
    # every one of its tensors twice in the state dict.
    assert sum(1 for k in quantile.state_dict() if "final_time_head" in k) == len(point_keys)


def test_the_rollout_flies_the_point_head_and_never_the_median():
    """The whole construction: the duration the schedule is rolled over is the POINT head's
    answer, and the quantile head rides beside it without touching it."""
    torch.manual_seed(0)
    config = _two_head()
    model = build_model(config).eval()
    with torch.no_grad():
        # Move the two heads apart — at initialization the median sits exactly on the point
        # head's start (B1's rule, asserted below), which would make this test vacuous.
        model.final_time_head.network[-1].weight.normal_(std=0.5)
        model.duration_quantile_head.network[-1].bias.add_(0.7)
    history = _history(config)
    prediction = model_forward(model, history, _dynamics(len(history)))

    with torch.no_grad():
        point = model.final_time_head(history)
        quantiles = model.duration_quantile_head(history)
    assert torch.allclose(prediction.final_time_s, point)
    assert torch.allclose(prediction.duration_quantiles_s, quantiles)
    # The rollout's own clock agrees with the point head, and the median is a DIFFERENT
    # number — a test that could not tell the two apart would pass under `quantile` too.
    assert torch.allclose(prediction.segment_durations.sum(dim=1), point)
    median = prediction.duration_quantiles_s[:, DURATION_MEDIAN_INDEX]
    assert not torch.allclose(prediction.final_time_s, median)
    assert torch.all(torch.diff(prediction.duration_quantiles_s, dim=-1) > 0.0)
    assert prediction.duration_quantiles_s.shape == (len(history), len(DURATION_QUANTILES))


def test_both_heads_start_where_the_point_head_starts():
    """B1's initialization rule holds per HEAD, so a `two-head` run and a `point` run make
    the same first prediction and differ only in what they learn."""
    torch.manual_seed(0)
    two = build_model(_two_head()).eval()
    torch.manual_seed(0)
    point = build_model(_config(duration_head=DURATION_HEAD_POINT)).eval()
    history = _history(_two_head(), batch=3)
    with torch.no_grad():
        assert torch.allclose(two.final_time_head(history), point.final_time_head(history))
        assert torch.allclose(
            two.duration_quantile_head(history)[:, DURATION_MEDIAN_INDEX],
            point.final_time_head(history),
            atol=1e-6,
        )


def test_a_given_cta_leaves_the_point_head_inert_and_still_trains_the_quantiles():
    """Same rule as `quantile` (L3 + B3): the CTA is the duration, so the point head is
    never read, and the quantile head is read and trained on purpose."""
    torch.manual_seed(0)
    config = _two_head(cta_conditioning=CTA_CONDITIONING_GIVEN)
    model = build_model(config).eval()
    cta = torch.tensor([200.0, 250.0])
    prediction = model_forward(model, _history(config, batch=2), _dynamics(2, cta))
    assert torch.allclose(prediction.final_time_s, cta)
    assert prediction.duration_quantiles_s.shape == (2, len(DURATION_QUANTILES))


# ── the loss ────────────────────────────────────────────────────────────────

def _loss_components(config: TSConfig, monkeypatch, *, predicted_s: float, truth_s: float,
                     quantiles_s: list[float] | None):
    """The objective on one flight whose rollout is pinned at zero position error.

    Only the duration terms are under test, so the rollout is replaced by a perfect one and
    every position/terminal contribution is zero by construction.
    """
    import control.dynamics.rollout as control_rollout_module

    monkeypatch.setattr(
        control_rollout_module, "rollout_control_endpoints",
        lambda _controls, _durations, _dynamics, _config, command_hook=None: SimpleNamespace(
            channels=torch.zeros(1, config.n_segments, config.enc_in, dtype=torch.float64),
            geodetic_states=torch.zeros(1, config.n_segments, 7, dtype=torch.float64),
        ),
    )
    from dataset import Normalizer

    normalizer = Normalizer(mean=np.zeros(config.enc_in), std=np.ones(config.enc_in))
    prediction = ControlPrediction(
        controls=torch.zeros(1, config.n_segments, 3),
        segment_durations=torch.full((1, config.n_segments), predicted_s / config.n_segments),
        final_time_s=torch.tensor([predicted_s]),
        duration_quantiles_s=(None if quantiles_s is None
                              else torch.tensor([quantiles_s], dtype=torch.float32)),
    )
    target = torch.zeros(1, config.n_segments, config.enc_in)
    return objective.control_prediction_loss_components(
        prediction,
        torch.zeros(1, config.enc_in),
        target,
        torch.full_like(target, 1.0 / config.enc_in),
        torch.tensor([truth_s]),
        torch.ones(1),
        config,
        normalizer,
        {"control_lower": torch.tensor([CONTROL_LOWER], dtype=torch.float32),
         "control_upper": torch.tensor([CONTROL_UPPER], dtype=torch.float32)},
    )


def _loss_config(**overrides) -> TSConfig:
    """The minimal true-time-position control objective, as `test_ts_transformer` builds it
    for the same purpose: the position terms are zero by construction above, so what comes
    out of it is the duration terms and nothing else."""
    settings = dict(
        prediction_output=PREDICTION_CONTROL,
        duration_head=DURATION_HEAD_TWO_HEAD,
        control_duration_parameterization=CONTROL_DURATION_UNIFORM,
        control_state_supervision_clock=CONTROL_STATE_CLOCK_OBSERVED,
        control_state_objective=CONTROL_STATE_OBJECTIVE_TRUE_TIME_POSITION,
        control_state_duration_gradient=False,
        n_segments=2, final_time_scale_s=600.0, position_loss_scale_m=10_000.0,
        state_endpoint_loss_weight=0.25,
    )
    settings.update(overrides)
    return TSConfig(**settings)


def test_each_head_is_priced_by_its_own_weight(monkeypatch):
    """The point term keeps `final_time` and `final_time_loss_weight`; the pinball sum is a
    component of its own at `duration_quantile_loss_weight`. Neither can be read out of the
    other's weight, which is exactly what B1_quantile could not do."""
    config = _loss_config(final_time_loss_weight=26.0, duration_quantile_loss_weight=3.0)
    quantiles = [200.0, 260.0, 300.0, 340.0, 400.0]
    components = _loss_components(
        config, monkeypatch, predicted_s=280.0, truth_s=320.0, quantiles_s=quantiles,
    )
    assert float(components.final_time) == pytest.approx(
        26.0 * ((280.0 - 320.0) / 600.0) ** 2, rel=1e-6
    )
    expected_pinball = float(pinball_duration_loss(
        torch.tensor([quantiles]), torch.tensor([320.0]), 600.0
    ))
    assert float(components.extras[DURATION_QUANTILE_COMPONENT]) == pytest.approx(
        3.0 * expected_pinball, rel=1e-6
    )


def test_the_point_term_is_the_one_a_point_run_would_get(monkeypatch):
    """The `final_time` component under `two-head` is bit-for-bit the one `point` computes
    at the same weight: the second head adds a term, it does not change the first."""
    quantiles = [200.0, 260.0, 300.0, 340.0, 400.0]
    two = _loss_components(
        _loss_config(final_time_loss_weight=26.0), monkeypatch,
        predicted_s=280.0, truth_s=320.0, quantiles_s=quantiles,
    )
    point = _loss_components(
        _loss_config(duration_head=DURATION_HEAD_POINT, final_time_loss_weight=26.0),
        monkeypatch, predicted_s=280.0, truth_s=320.0, quantiles_s=None,
    )
    assert float(two.final_time) == float(point.final_time)
    assert DURATION_QUANTILE_COMPONENT not in point.extras


def test_the_quantile_weight_is_what_multiplies_the_pinball_under_quantile(monkeypatch):
    """Under `quantile` the pinball sum still rides in `final_time` (B1's contract, which
    every stored history row keys on) — but the number multiplying it is now the quantile
    weight, and its default is the 1.0 that multiplied it before."""
    quantiles = [200.0, 260.0, 300.0, 340.0, 400.0]
    expected = float(pinball_duration_loss(
        torch.tensor([quantiles]), torch.tensor([320.0]), 600.0
    ))
    for weight in (DEFAULT_DURATION_QUANTILE_LOSS_WEIGHT, 4.0):
        components = _loss_components(
            _loss_config(duration_head=DURATION_HEAD_QUANTILE,
                         duration_quantile_loss_weight=weight),
            monkeypatch, predicted_s=280.0, truth_s=320.0, quantiles_s=quantiles,
        )
        assert float(components.final_time) == pytest.approx(weight * expected, rel=1e-6)
        assert DURATION_QUANTILE_COMPONENT not in components.extras


def test_only_two_head_adds_a_loss_component():
    """`point` and `quantile` keep the component list B1 froze; `two-head` extends it,
    because with both terms alive neither can be read out of the other."""
    for head in (DURATION_HEAD_POINT, DURATION_HEAD_QUANTILE):
        names = loss_component_names(_config(duration_head=head))
        assert DURATION_QUANTILE_COMPONENT not in names and "final_time" in names
    names = loss_component_names(_two_head())
    assert names[-1] == DURATION_QUANTILE_COMPONENT and "final_time" in names


# ── refusals ────────────────────────────────────────────────────────────────

def test_two_head_is_refused_with_a_latent():
    """Same reason as `quantile`: under a posterior sample the quantiles would be
    conditioned on the flight's own future, and B2 would calibrate them as p(T | history)."""
    with pytest.raises(ValueError, match="refused together"):
        _two_head(latent_dim=4)


def test_two_head_is_refused_off_the_control_output():
    with pytest.raises(ValueError, match="control path"):
        TSConfig(prediction_output=PREDICTION_STATE, duration_head=DURATION_HEAD_TWO_HEAD)


def test_the_point_weight_is_refused_where_there_is_no_point_term():
    """`quantile` replaced the point term outright, so weighing it weighs nothing. Every
    stored `quantile` run carries the default, so nothing on disk is refused by this."""
    with pytest.raises(ValueError, match="no point term"):
        _config(duration_head=DURATION_HEAD_QUANTILE, final_time_loss_weight=26.0)
    # ...and the refusal names the value that takes the point term back.
    _two_head(final_time_loss_weight=26.0)


def test_the_quantile_weight_is_refused_where_there_is_no_quantile_head():
    with pytest.raises(ValueError, match="no such head"):
        _config(duration_head=DURATION_HEAD_POINT, duration_quantile_loss_weight=3.0)
    _config(duration_head=DURATION_HEAD_QUANTILE, duration_quantile_loss_weight=3.0)
    _two_head(duration_quantile_loss_weight=3.0)


def test_a_negative_quantile_weight_is_refused():
    with pytest.raises(ValueError, match="duration_quantile_loss_weight must be non-negative"):
        _two_head(duration_quantile_loss_weight=-1.0)


def test_a_named_recipe_cannot_wear_two_heads():
    """The recipes pin `duration_head='point'`, so a two-head run is `custom` and its name
    says which head it has instead of hiding behind a recipe."""
    with pytest.raises(ValueError, match="recipe fields are frozen"):
        TSConfig(**{**_two_head().to_dict(), "control_recipe_name": "simple-v3"})


# ── the name ────────────────────────────────────────────────────────────────

def test_the_run_name_carries_the_two_head_token_and_only_a_non_default_weight():
    two = _two_head().to_dict()
    assert "T=2h" in run_display_name(two) and "t-2h" in run_slug(two)
    assert "pinball" not in run_display_name(two)
    weighted = _two_head(duration_quantile_loss_weight=3.0).to_dict()
    assert "pinball=3" in run_display_name(weighted)
    # The other two values are untouched by the new token.
    assert f"T=q{len(DURATION_QUANTILES)}" in run_display_name(
        _config(duration_head=DURATION_HEAD_QUANTILE).to_dict()
    )
    assert "T=" not in run_display_name(_config(duration_head=DURATION_HEAD_POINT).to_dict())


def test_a_config_that_predates_the_new_weight_is_named_as_it_was():
    """The recount's invariant, in the suite: a stored config carries no
    `duration_quantile_loss_weight`, and the grammar reads the absent field as its default,
    so adding it renames nothing."""
    stored = _config(duration_head=DURATION_HEAD_POINT).to_dict()
    assert "duration_quantile_loss_weight" in stored
    older = {k: v for k, v in stored.items() if k != "duration_quantile_loss_weight"}
    assert run_display_name(older) == run_display_name(stored)
    assert run_slug(older) == run_slug(stored)


# ── the record, end to end ──────────────────────────────────────────────────

def _trained_two_head(tmp_path: Path, monkeypatch, **overrides):
    import cli.predict as predict_module

    config = _two_head(final_time_loss_weight=26.0, **overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3)
    series, _report = build_series(flights, config, airport=AIRPORT)
    run = tmp_path / "run"
    train(series, config, output_dir=run, data_provenance=PROVENANCE, verbose=False)
    monkeypatch.setattr(predict_module, "provenance_from_args", lambda _args: PROVENANCE)
    monkeypatch.setattr(
        predict_module, "load_flight_dicts", lambda _path, include_flight_keys=None: flights
    )
    return run / "checkpoint.pt", series


def _cli():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "__main__.py"
    spec = importlib.util.spec_from_file_location("ts_cli_two_head_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predict(checkpoint: Path, out: Path, tmp_path: Path, *extra: str) -> int:
    return _cli().main([
        "predict", "--checkpoint", str(checkpoint), "--data", str(tmp_path / "manifest.json"),
        "--airport", AIRPORT, "--output-dir", str(out), "--split", "val", "--device", "cpu",
        *extra,
    ])


def test_the_record_carries_the_point_duration_and_the_quantile_interval(
    tmp_path: Path, monkeypatch
):
    """Both fields, and they are two DIFFERENT heads' answers: `durationHeadFinalTimeS` is
    the duration the states were rolled over, `durationQuantilesS` is what B2 calibrates."""
    checkpoint, _series = _trained_two_head(tmp_path, monkeypatch)
    out = tmp_path / "pred"
    assert _predict(checkpoint, out, tmp_path) == 0

    summary = json.loads((out / "summary.json").read_text())
    assert summary["config"]["duration_head"] == DURATION_HEAD_TWO_HEAD
    model, loaded, normalizer, _payload = load_checkpoint(checkpoint)
    rows = summary["results"]
    for row in rows:
        states = json.loads((out / row["states_file"]).read_text())
        source = states["source"]
        published = source["durationQuantilesS"]
        assert len(published) == len(DURATION_QUANTILES) and published == sorted(published)
        assert row["duration_quantiles_s"] == published
        # The rollout's duration is the POINT head's, so the record's own two numbers are
        # not required to agree — under `quantile` they would be the same number twice.
        assert source["durationHeadFinalTimeS"] == pytest.approx(
            row["predicted_final_time_s"]
        )
        assert source["calibrated"] is False

    # ...and the quantiles in the record ARE the quantile head's own answer. The rows are
    # the checkpoint's val split, a subset of the cohort in the checkpoint's own order, so
    # this matches each published row against the whole cohort's quantile-head forward
    # rather than restating how the split was cut.
    cohort, _report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3), loaded, airport=AIRPORT
    )
    predicted = duration_quantile_predictions(
        model, cohort, loaded, normalizer, device=torch.device("cpu")
    )
    for row in rows:
        published = np.array(row["duration_quantiles_s"], dtype=np.float64)
        assert any(np.allclose(published, candidate, rtol=1e-6) for candidate in predicted)


def test_the_calibration_runner_takes_a_two_head_checkpoint(tmp_path: Path, monkeypatch):
    """The runner reads the QUANTILE head (one forward, no rollout, no CTA) — so which head
    drives the rollout is none of its business, and only a head with no quantiles at all is
    refused."""
    from tests.test_eta_calibration import _cohort, _metadata, _stubbed_runner

    samples = _cohort(400, narrow_s=10.0)
    runner = _stubbed_runner(monkeypatch, samples, split_seed=1,
                             duration_head=DURATION_HEAD_TWO_HEAD)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    _metadata(tmp_path, sha=file_sha256(checkpoint))
    assert runner.main(["--checkpoint", str(checkpoint), "--out", str(tmp_path / "out"),
                        "--device", "cpu"]) == 0
    table = json.loads((tmp_path / "out" / runner.JSON_NAME).read_text())["conformal"]
    assert table["quantiles"] == list(DURATION_QUANTILES)

    point = _stubbed_runner(monkeypatch, samples, split_seed=1,
                            duration_head=DURATION_HEAD_POINT)
    with pytest.raises(SystemExit, match="no interval to calibrate"):
        point.main(["--checkpoint", str(checkpoint), "--out", str(tmp_path / "refused"),
                    "--device", "cpu"])


def test_the_quantile_fan_decodes_a_two_head_checkpoint(tmp_path: Path, monkeypatch):
    """B3 on a B1.b arm: training's rollout was CTA-driven and the point head inert, and the
    fan is decoded at the QUANTILE head's five levels."""
    from calibration import QUANTILE_DIR_NAME, quantile_directory_name

    checkpoint, _series = _trained_two_head(
        tmp_path, monkeypatch, cta_conditioning=CTA_CONDITIONING_GIVEN
    )
    out = tmp_path / "fan"
    assert _predict(checkpoint, out, tmp_path, "--cta-from-quantiles") == 0
    leaves = sorted(path.name for path in (out / QUANTILE_DIR_NAME).iterdir())
    assert leaves == sorted(quantile_directory_name(tau) for tau in DURATION_QUANTILES)
    for row in json.loads((out / "summary.json").read_text())["results"]:
        assert row["cta_from_quantiles"] is True and row["cta_quantile"] == 0.5
        assert row["cta_s"] == pytest.approx(
            row["duration_quantiles_s"][DURATION_MEDIAN_INDEX]
        )


def test_the_eta_error_readout_reports_the_quantile_head_beside_the_rollout(
    tmp_path: Path, monkeypatch, capsys
):
    """Gate 1 of §三 3.1b is read on two different heads: ADE and the rollout's own duration
    on the point head, the arrival-time MAE on q50. The readout prints both blocks and
    derives q50's error from the row's own published quantiles."""
    import importlib.util

    checkpoint, _series = _trained_two_head(tmp_path, monkeypatch)
    out = tmp_path / "pred"
    assert _predict(checkpoint, out, tmp_path) == 0

    path = Path(__file__).resolve().parents[3] / "run_ts_eta_error_readout.py"
    spec = importlib.util.spec_from_file_location("run_ts_eta_error_readout_test", path)
    readout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(readout)

    payload = readout.readout("B1b", out)
    assert payload["duration_head"] == DURATION_HEAD_TWO_HEAD
    assert readout.Q50_METRIC in payload["metrics"]
    rows = json.loads((out / "summary.json").read_text())["results"]
    expected = np.abs(np.array(
        [row["duration_quantiles_s"][DURATION_MEDIAN_INDEX] - row["true_final_time_s"]
         for row in rows], dtype=np.float64
    )).mean()
    pooled = next(iter(payload["strata"].values()))
    assert pooled["n"] == len(rows)
    assert pooled[readout.Q50_METRIC]["mae"] == pytest.approx(expected)
    # ...and the point head's own MAE is the block that was already there, on the duration
    # the rollout actually flew.
    assert pooled["final_time_error_s"]["mae"] == pytest.approx(
        np.abs(np.array([row["final_time_error_s"] for row in rows])).mean()
    )
    text = readout.render({"schema": readout.RESULT_SCHEMA, "arms": [payload]})
    assert "GATE 1" in text and "POINT head" in text
    assert readout.DURATION_MEDIAN_INDEX == DURATION_MEDIAN_INDEX


def test_a_point_head_directory_has_no_q50_block(tmp_path: Path, monkeypatch):
    """The block exists only where a quantile head wrote the rows, so a point-head arm's
    readout is the one B0 published."""
    import importlib.util

    import cli.predict as predict_module

    config = _config(duration_head=DURATION_HEAD_POINT)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=12, seed=3)
    series, _report = build_series(flights, config, airport=AIRPORT)
    run = tmp_path / "run"
    train(series, config, output_dir=run, data_provenance=PROVENANCE, verbose=False)
    monkeypatch.setattr(predict_module, "provenance_from_args", lambda _args: PROVENANCE)
    monkeypatch.setattr(
        predict_module, "load_flight_dicts", lambda _path, include_flight_keys=None: flights
    )
    out = tmp_path / "pred"
    assert _predict(run / "checkpoint.pt", out, tmp_path) == 0

    path = Path(__file__).resolve().parents[3] / "run_ts_eta_error_readout.py"
    spec = importlib.util.spec_from_file_location("run_ts_eta_error_readout_point", path)
    readout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(readout)
    payload = readout.readout("point", out)
    assert readout.Q50_METRIC not in payload["metrics"]
    assert "GATE 1" not in readout.render(
        {"schema": readout.RESULT_SCHEMA, "arms": [payload]}
    )
