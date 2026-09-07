"""`lr_plateau_metric`: WHICH number the learning-rate plateau is measured on.

A0.b of `docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md` §2.4c. On the
`A0_random_hr8_tv1_grid` arm the validation OBJECTIVE improved to epoch 60 while the
checkpoint-selection value — a dense-grid ADE — stalled after epoch 8, so
`ReduceLROnPlateau`, stepped with the selection value, halved the learning rate from epoch
20 and reached 9.4e-7 by epoch 60: the model stopped training at the epoch the READOUT
stalled. The axis lets the scheduler watch the objective instead.

What the tests hold:

* the value handed to `scheduler.step` is one of the two numbers the epoch record already
  carries, and WHICH one is exactly `lr_plateau_metric` — a spy compares against
  `history.json`, so a third number computed for the scheduler alone would fail here;
* checkpoint selection is untouched by the axis (the kept epoch is still the selection
  metric's best);
* the run wears the non-default value, and the recipes pin the default as a literal.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from config import (
    CONTROL_RECIPE_SIMPLE_V3,
    LR_PLATEAU_METRIC_OBJECTIVE,
    LR_PLATEAU_METRIC_SELECTION,
    LR_PLATEAU_METRICS,
    TSConfig,
    control_recipe_overrides,
    control_simple_v1_overrides,
    recipe_settings,
)
from data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from dataset import build_series
from run_naming import run_display_name, run_slug
from synthetic import synthetic_arrivals
from train import train

AIRPORT, RUNWAY = "KRDU", "05L"
SELECTION_METRIC = "fixed-anchor-common-grid-ade"
_REAL_SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau


def test_the_axis_has_exactly_two_values_and_defaults_to_todays_behaviour():
    assert LR_PLATEAU_METRICS == (LR_PLATEAU_METRIC_SELECTION, LR_PLATEAU_METRIC_OBJECTIVE)
    assert TSConfig().lr_plateau_metric == LR_PLATEAU_METRIC_SELECTION


def test_an_unknown_metric_is_refused_at_construction():
    with pytest.raises(ValueError, match="unknown lr_plateau_metric"):
        TSConfig(lr_plateau_metric="val_loss")


def test_every_named_recipe_pins_the_default_as_a_literal():
    """A recipe is a frozen definition: a later default change must not redefine it."""
    assert control_simple_v1_overrides()["lr_plateau_metric"] == LR_PLATEAU_METRIC_SELECTION
    for recipe in ("simple-v1", "simple-v1-lag", "simple-v2", CONTROL_RECIPE_SIMPLE_V3):
        assert control_recipe_overrides(recipe)["lr_plateau_metric"] == (
            LR_PLATEAU_METRIC_SELECTION
        )


def test_only_the_non_default_metric_names_the_run():
    plain = TSConfig()
    assert "lr-metric" not in run_display_name(plain.to_dict())
    watching_objective = replace(plain, lr_plateau_metric=LR_PLATEAU_METRIC_OBJECTIVE)
    assert "lr-metric=objective" in run_display_name(watching_objective.to_dict())
    assert run_slug(watching_objective.to_dict()) != run_slug(plain.to_dict())


class _SchedulerSpy(_REAL_SCHEDULER):
    """The real scheduler, recording every value it was stepped with."""

    stepped: list[float]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).stepped = []

    def step(self, metrics=None, epoch=None):
        type(self).stepped.append(float(metrics))
        return super().step(metrics, epoch)


def _two_epoch_run(tmp_path: Path, name: str, *, epochs: int = 2, **overrides) -> list[dict]:
    settings = dict(recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False))
    settings.update(
        n_segments=8, control_imitation_loss_weight=0.0,
        seq_len=8, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        final_time_scale_s=2.0, device="cpu",
        epochs=epochs, patience=epochs, batch_size=8, dropout=0.0,
        control_rollout_integrator_dt_s=0.5,
        checkpoint_selection_metric=SELECTION_METRIC,
    )
    settings.update(overrides)
    config = TSConfig(**settings)
    series, report = build_series(
        synthetic_arrivals(AIRPORT, RUNWAY, n_flights=8, seed=3), config, airport=AIRPORT
    )
    assert report.built == 8, report.format()
    torch.manual_seed(0)
    train(
        series, config, output_dir=tmp_path / name,
        data_provenance={
            "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
            "manifests": [{"airport": AIRPORT, "arrival_manifest_sha256": "a" * 64,
                           "source_records": []}],
        },
        verbose=False,
    )
    return json.loads((tmp_path / name / "history.json").read_text())["history"]


@pytest.mark.parametrize(
    "metric, recorded_field",
    [
        (LR_PLATEAU_METRIC_SELECTION, "validation_selection_value"),
        (LR_PLATEAU_METRIC_OBJECTIVE, "val_loss"),
    ],
)
def test_the_scheduler_steps_on_the_number_the_axis_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metric: str, recorded_field: str
):
    """And on the number the RECORD writes, not a second one computed for the scheduler."""
    monkeypatch.setattr(torch.optim.lr_scheduler, "ReduceLROnPlateau", _SchedulerSpy)
    history = _two_epoch_run(tmp_path, metric, lr_plateau_metric=metric)
    assert _SchedulerSpy.stepped == [epoch[recorded_field] for epoch in history]
    # ...and the two numbers really are different here, so the assertion above discriminates.
    assert history[0]["val_loss"] != history[0]["validation_selection_value"]


def test_the_scheduler_block_says_which_metric_it_stepped_on(tmp_path: Path):
    _two_epoch_run(tmp_path, "named", lr_plateau_metric=LR_PLATEAU_METRIC_OBJECTIVE)
    metadata = json.loads((tmp_path / "named" / "checkpoint_metadata.json").read_text())
    assert metadata["lr_scheduler"]["metric"] == LR_PLATEAU_METRIC_OBJECTIVE


#: The staged selection values: one improvement, then a STALL — the A0-random `_grid` arm's
#: shape, small enough to write down. The objective meanwhile improves every epoch by more
#: than `ReduceLROnPlateau`'s relative threshold, which is what makes the two arms diverge.
_STALLED_SELECTION = (100.0, 99.0, 99.0, 99.0, 99.0, 99.0)


def _stall_the_selection_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real metric's machinery, replace only the NUMBER it reports."""
    import train as train_module

    real = train_module.VALIDATION_SELECTIONS[SELECTION_METRIC]
    staged = iter(_STALLED_SELECTION)

    def stalled(**kwargs):
        return replace(real(**kwargs), value=next(staged))

    monkeypatch.setitem(train_module.VALIDATION_SELECTIONS, SELECTION_METRIC, stalled)


def test_the_axis_moves_the_learning_rate_and_not_the_kept_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Both arms over four epochs of a STALLED selection metric and an improving objective —
    the situation the axis exists for. The learning-rate columns must DIFFER (the axis does
    something) while each run still keeps the epoch its selection metric scored best (the
    axis does only that)."""
    rates = {}
    for metric in (LR_PLATEAU_METRIC_SELECTION, LR_PLATEAU_METRIC_OBJECTIVE):
        with monkeypatch.context() as patch:
            _stall_the_selection_metric(patch)
            history = _two_epoch_run(
                tmp_path, metric, epochs=len(_STALLED_SELECTION), lr_plateau_patience=1,
                learning_rate=1e-2, lr_plateau_metric=metric,
            )
        assert [epoch["validation_selection_value"] for epoch in history] == list(
            _STALLED_SELECTION
        )
        # The objective really is improving, or the comparison below proves nothing.
        losses = [epoch["val_loss"] for epoch in history]
        assert losses == sorted(losses, reverse=True)
        summary = json.loads((tmp_path / metric / "history.json").read_text())
        assert summary["validation_selection"]["best_value"] == pytest.approx(
            min(_STALLED_SELECTION)
        )
        assert summary["config"]["lr_plateau_metric"] == metric
        rates[metric] = [epoch["learning_rate"] for epoch in history]
    # The stalled metric cuts the rate; the improving objective does not.
    assert rates[LR_PLATEAU_METRIC_SELECTION] != rates[LR_PLATEAU_METRIC_OBJECTIVE], rates
    assert rates[LR_PLATEAU_METRIC_OBJECTIVE] == [1e-2] * len(_STALLED_SELECTION)
    assert rates[LR_PLATEAU_METRIC_SELECTION][-1] < 1e-2


def test_the_objective_is_refused_while_the_kl_warm_up_reweights_it(tmp_path: Path):
    """A ramping beta rescales the objective every epoch, so its plateau is not the model's."""
    with pytest.raises(ValueError, match="latent_beta_warmup_epochs"):
        TSConfig(**{
            **recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False),
            "latent_dim": 4, "latent_beta_warmup_epochs": 10,
            "lr_plateau_metric": LR_PLATEAU_METRIC_OBJECTIVE,
        })


def test_the_objective_is_refused_while_the_procedure_multipliers_move():
    with pytest.raises(ValueError, match="procedure_loss_dual_step"):
        TSConfig(
            procedure_loss_lateral_weight=1.0,
            procedure_loss_dual_step=0.5,
            lr_plateau_metric=LR_PLATEAU_METRIC_OBJECTIVE,
        )


def test_a_fixed_beta_latent_run_may_still_watch_the_objective():
    """The refusal is about a MOVING objective, not about the latent path as such."""
    config = TSConfig(**{
        **recipe_settings(CONTROL_RECIPE_SIMPLE_V3, keep_name=False),
        "latent_dim": 4, "latent_beta": 0.01,
        "lr_plateau_metric": LR_PLATEAU_METRIC_OBJECTIVE,
    })
    assert config.lr_plateau_metric == LR_PLATEAU_METRIC_OBJECTIVE
