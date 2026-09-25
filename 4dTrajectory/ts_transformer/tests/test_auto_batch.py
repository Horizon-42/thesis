"""`--batch-size auto`: the probe, its heterogeneous control partitions, its margin, its diagnostics.

Split from `test_ts_transformer.py` on 2026-09-10 (review §4.6).
"""


import torch

import ts_transformer.data.channels as ch
import ts_transformer.training.batching as batching
import ts_transformer.training.objective as objective
import ts_transformer.outputs.control.strategy as control_strategy
from ts_transformer.training.batching import resolve_batch_size
from ts_transformer.config import PREDICTION_CONTROL, TSConfig
from ts_transformer.outputs.control.training.diagnostics import ControlTrainingDiagnosticsAccumulator
from ts_transformer.outputs.control.heads import ControlPrediction


def test_auto_batch_uses_config_default_without_cuda():
    config = TSConfig(batch_size=37)
    assert resolve_batch_size(
        config, torch.device("cpu"), auto=True, verbose=False
    ) == 37


def test_auto_batch_selects_2048_when_2048_probe_succeeds(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: None)
    monkeypatch.setattr(batching, "_probe_training_step", lambda *_args: None)

    assert batching.resolve_batch_size(
        TSConfig(), torch.device("cuda"), auto=True, verbose=False
    ) == 2048


def test_control_auto_batch_probe_uses_heterogeneous_duration_partitions():
    batch_size, n_segments = 8, 8
    final_time = torch.full((batch_size,), 80.0)
    prediction = ControlPrediction(
        controls=torch.zeros(batch_size, n_segments, 3),
        segment_durations=torch.full((batch_size, n_segments), 10.0),
        final_time_s=final_time,
    )

    probed = control_strategy.heterogeneous_control_probe_prediction(prediction)
    fractions = probed.segment_durations / probed.final_time_s[:, None]
    baseline_steps = int(torch.ceil(
        prediction.segment_durations.max(dim=0).values / 0.5
    ).sum())
    heterogeneous_steps = int(torch.ceil(
        probed.segment_durations.max(dim=0).values / 0.5
    ).sum())

    assert torch.allclose(probed.segment_durations.sum(dim=-1), final_time)
    assert torch.unique(fractions, dim=0).shape[0] == batch_size
    assert heterogeneous_steps >= 2 * baseline_steps


def test_control_auto_batch_retains_one_power_of_two_safety_margin(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: None)
    monkeypatch.setattr(batching, "_probe_training_step", lambda *_args: None)

    assert batching.resolve_batch_size(
        TSConfig(prediction_output=PREDICTION_CONTROL),
        torch.device("cuda"),
        auto=True,
        verbose=False,
    ) == 1024


def test_control_auto_batch_training_probe_applies_heterogeneous_partition(monkeypatch):
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        device="cpu",
        seq_len=4,
        n_segments=2,
        d_model=8,
        d_ff=16,
        n_heads=2,
        e_layers=1,
        final_time_scale_s=2.0,
        control_rollout_integrator_dt_s=0.5,
    )
    original = control_strategy.heterogeneous_control_probe_prediction
    calls: list[torch.Size] = []

    def tracked(prediction):
        calls.append(prediction.segment_durations.shape)
        return original(prediction)

    monkeypatch.setattr(control_strategy, "heterogeneous_control_probe_prediction", tracked)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    batching._probe_training_step(config, 2, torch.device("cpu"))

    assert calls == [torch.Size([2, 2])]


def test_control_auto_batch_training_probe_executes_clip_diagnostics(monkeypatch):
    config = TSConfig(
        prediction_output=PREDICTION_CONTROL,
        device="cpu",
        seq_len=4,
        n_segments=2,
        d_model=8,
        d_ff=16,
        n_heads=2,
        e_layers=1,
        final_time_scale_s=2.0,
        control_rollout_integrator_dt_s=0.5,
        control_gradient_clip_norm=20.0,
    )
    events: list[str] = []

    class TrackedDiagnostics(ControlTrainingDiagnosticsAccumulator):
        def record_prediction(self, prediction, dynamics):
            events.append("prediction")
            return super().record_prediction(prediction, dynamics)

        def record_gradients_and_clip(self, model):
            assert any(parameter.grad is not None for parameter in model.parameters())
            events.append("gradients")
            return super().record_gradients_and_clip(model)

    monkeypatch.setattr(
        control_strategy,
        "ControlTrainingDiagnosticsAccumulator",
        TrackedDiagnostics,
        raising=False,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    batching._probe_training_step(config, 2, torch.device("cpu"))

    assert events == ["prediction", "gradients"]


def test_auto_batch_probe_executes_the_shared_physics_loss(monkeypatch):
    config = TSConfig(
        device="cpu",
        seq_len=4,
        n_segments=4,
        d_model=8,
        d_ff=16,
        n_heads=2,
        e_layers=1,
    )
    # "Shared" is the claim under test: the probe must run the package's ONE objective,
    # not a copy of it that can drift.
    assert batching.prediction_loss is objective.prediction_loss
    original_loss = objective.prediction_loss
    calls: list[tuple[torch.Size, torch.Size]] = []

    def tracked_loss(prediction, anchor, target, *args):
        calls.append((anchor.shape, target.shape))
        return original_loss(prediction, anchor, target, *args)

    monkeypatch.setattr(batching, "prediction_loss", tracked_loss)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    batching._probe_training_step(config, 2, torch.device("cpu"))

    assert calls == [
        (torch.Size([2, len(ch.CHANNELS)]), torch.Size([2, 4, len(ch.CHANNELS)]))
    ]


def test_the_probe_builds_a_latent_model_s_posterior_and_kl(monkeypatch):
    """A latent control model consumes the future (its posterior encoder and the KL): the probe hands it one, as the
    training loop does, so it measures the graph training builds — without it the KL had no posterior and raised."""
    from ts_transformer.tests.test_latent_control import _config as latent_config

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)   # the probe step, run on the CPU
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    batching._probe_training_step(latent_config(), 4, torch.device("cpu"))
