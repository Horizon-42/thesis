"""Phase 0 intent conditioning: the truth join point and the lead's landing time as input."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

_TS_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _TS_DIR.parents[1]
for path in (_TS_DIR, _REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import channels as ch  # noqa: E402
import dataset as dataset_module  # noqa: E402
import final_approach_geometry as fag  # noqa: E402
import intent_conditioning as ic  # noqa: E402
from config import (  # noqa: E402
    INTENT_JOIN_CHANNELS, INTENT_LEAD_CHANNELS, PREDICTION_CONTROL, TSConfig,
    intent_channel_names,
)
from dataset import (  # noqa: E402
    ARRIVAL_DATA_PROVENANCE_SCHEMA, FixedAnchorTrajectoryWindows, Normalizer, build_series,
)
from forecast import _history_at_anchor, forecast_approach  # noqa: E402
from models import build_model  # noqa: E402
from run_naming import run_display_name  # noqa: E402
from synthetic import synthetic_arrivals  # noqa: E402
from target_conditioning import CONDITIONING_CHANNELS  # noqa: E402
from train import load_checkpoint, train  # noqa: E402

AIRPORT, RUNWAY = "KRDU", "05L"
ENTRY_UTC = "2026-07-01T00:00:00Z"


def _series(n_flights=3, **config_overrides):
    """Synthetic arrivals as FlightSeries, each stamped with an entry time and a lead."""
    config = TSConfig(**config_overrides)
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=n_flights, seed=3)
    for index, flight in enumerate(flights):
        flight["lead_landing"] = ic.LeadLanding(f"2026-07-01T00:{index:02d}:30Z")
    series, report = build_series(flights, config, airport=AIRPORT)
    assert report.built == n_flights, report.format()
    for item in series:
        item.scenario.source["entry_time_utc"] = ENTRY_UTC
    return series, config


def _fake_data_provenance(airport: str = AIRPORT):
    return {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [{
            "airport": airport,
            "arrival_manifest_sha256": "a" * 64,
            "source_records": [],
        }],
    }


# ── Config contract ──────────────────────────────────────────────────────────

def test_intent_channels_are_input_only_and_follow_the_target_conditioning():
    joined = TSConfig(intent_conditioning="truth-join")
    assert joined.input_channels == ch.CHANNELS + INTENT_JOIN_CHANNELS
    assert joined.channels == ch.CHANNELS  # the OUTPUT contract does not grow
    both = TSConfig(coordinate_frame="airport-enu", target_conditioning="channels",
                    intent_conditioning="truth-join-lead")
    assert both.input_channels == (
        ch.CHANNELS + CONDITIONING_CHANNELS + INTENT_JOIN_CHANNELS + INTENT_LEAD_CHANNELS
    )
    assert both.enc_in == len(both.input_channels)
    assert intent_channel_names("none") == ()
    with pytest.raises(ValueError, match="requires the itransformer backbone"):
        TSConfig(model="patchtst", intent_conditioning="truth-join")
    with pytest.raises(ValueError, match="already rotated"):
        TSConfig(coordinate_frame="runway-aligned", intent_conditioning="truth-join")
    # The lead ETA is measured at the fixed anchor; a moving anchor has no one ETA.
    with pytest.raises(ValueError, match="fixed anchor"):
        TSConfig(random_train_anchor=True, intent_conditioning="truth-join-lead")
    TSConfig(random_train_anchor=True, intent_conditioning="truth-join")
    with pytest.raises(ValueError, match="unknown intent_conditioning"):
        TSConfig(intent_conditioning="join")
    with pytest.raises(ValueError, match="unknown intent_conditioning"):
        intent_channel_names("join")


def test_run_name_marks_a_truth_conditioned_checkpoint():
    assert "intent=truth-join-lead" in run_display_name(
        TSConfig(intent_conditioning="truth-join-lead").to_dict()
    )
    assert "intent" not in run_display_name(TSConfig().to_dict())


# ── The join point ───────────────────────────────────────────────────────────

def _chart_track(d_m: np.ndarray, xt_m: np.ndarray, psi: float, target: np.ndarray):
    """Chart positions for ``(d, xt)`` about ``target`` along course ``psi``."""
    e, n = fag.chart_from_axes(
        torch.tensor(d_m)[None], torch.tensor(xt_m)[None], torch.tensor([psi])
    )
    u = d_m * math.tan(math.radians(3.0))
    return np.stack([e[0].numpy(), n[0].numpy(), u], axis=1) + target


def test_truth_join_index_is_the_first_row_that_stays_inside_the_cone():
    psi, target = 0.7, np.array([120.0, -40.0, 15.0])
    d = np.linspace(20_000.0, 0.0, 41)          # 500 m steps to the threshold
    xt = np.zeros_like(d)
    xt[:10] = 3_000.0                            # a downwind offset, far outside the cone
    positions = _chart_track(d, xt, psi, target)
    assert ic.truth_join_index(positions, target, psi) == 10
    # Already established from the first row: the join is the first row.
    assert ic.truth_join_index(_chart_track(d, np.zeros_like(d), psi, target), target, psi) == 0
    # Outside the cone on the last row before the 300 m near-zone: never establishes.
    xt_late = np.zeros_like(d)
    xt_late[-2] = 3_000.0                        # d = 500 m > NEAR_THRESHOLD_M
    assert ic.truth_join_index(_chart_track(d, xt_late, psi, target), target, psi) is None
    with pytest.raises(ValueError, match=r"\[N, 3\]"):
        ic.truth_join_index(positions[:, :2], target, psi)


def test_truth_join_point_lies_on_the_final_and_falls_back_to_the_target():
    series, _config = _series(n_flights=2)
    for item in series:
        join = ic.truth_join_point(item)
        relative = torch.tensor(join[None, None, :2] - item.target_chart[None, None, :2])
        d, xt = fag.runway_axes(
            relative[..., 0], relative[..., 1], torch.tensor([item.scenario.target.psi])
        )
        assert d.item() > fag.NEAR_THRESHOLD_M
        assert abs(xt.item()) <= fag.K_MARGIN * fag.corridor_halfwidth(d).item()
    # A track that never establishes carries the target itself.
    never = replace(series[0], values=series[0].values.copy())
    never.values[:, list(ch.POSITION_IDX)[1]] += 5_000.0   # shove the whole track sideways
    assert np.allclose(ic.truth_join_point(never), series[0].target_chart)


# ── The lead ─────────────────────────────────────────────────────────────────

def _write_roster(tmp_path: Path, tracks: list[dict], arrivals: list[dict]) -> Path:
    (tmp_path / "tracks").mkdir()
    (tmp_path / "arrivals").mkdir()
    (tmp_path / "tracks" / "manifest.json").write_text(json.dumps({"records": tracks}))
    manifest = tmp_path / "arrivals" / "manifest.json"
    manifest.write_text(json.dumps({
        "source_manifest": "../tracks/manifest.json", "records": arrivals,
    }))
    return manifest


def test_lead_is_the_previous_assigned_landing_on_the_same_runway(tmp_path):
    def track(key, runway, landing, outcome="assigned"):
        return {"flight_key": key, "runway": runway, "outcome": outcome,
                "landing_time_utc": landing}

    tracks = [
        track("A", "05L", "2026-05-01T00:00:00Z"),
        track("X", "05L", "2026-05-01T00:03:00Z"),          # excluded from arrivals, still landed
        track("B", "05L", "2026-05-01T00:05:00Z"),
        track("C", "23R", "2026-05-01T00:04:00Z"),          # another runway
        track("N", "05L", "2026-05-01T00:04:30Z", outcome="not_landing"),
        {"flight_key": "U", "runway": None, "outcome": "not_landing", "landing_time_utc": None},
        track("S", "05L", "2026-05-01T00:05:00Z"),          # same second as B: not earlier
    ]
    arrivals = [
        {"flight_key": "A", "runway": "05L", "landing_time_utc": "2026-05-01T00:00:00Z"},
        {"flight_key": "B", "runway": "05L", "landing_time_utc": "2026-05-01T00:05:00Z"},
        {"flight_key": "C", "runway": "23R", "landing_time_utc": "2026-05-01T00:04:00Z"},
    ]
    manifest_path = _write_roster(tmp_path, tracks, arrivals)
    manifest = json.loads(manifest_path.read_text())
    leads = ic.lead_landings(manifest, manifest_path=manifest_path, flight_keys=["A", "B", "C"])
    assert leads == {
        "A": ic.LeadLanding(None),
        "B": ic.LeadLanding("2026-05-01T00:03:00Z"),
        "C": ic.LeadLanding(None),
    }
    # Only the requested flights are looked up.
    assert set(ic.lead_landings(manifest, manifest_path=manifest_path, flight_keys=["B"])) == {"B"}


def test_lead_eta_is_measured_from_the_anchor_and_clipped():
    series, config = _series(n_flights=1)
    anchor = config.seq_len - 1
    anchor_time_s = float(series[0].times[anchor])    # 118 s after the first sample
    assert anchor_time_s == pytest.approx((config.seq_len - 1) * config.dt_s)
    item = replace(series[0], lead_landing=ic.LeadLanding("2026-07-01T00:03:00Z"))
    assert ic.lead_eta_s(item, anchor_time_s=anchor_time_s) == pytest.approx(180.0 - anchor_time_s)
    late = replace(series[0], lead_landing=ic.LeadLanding("2026-07-01T02:00:00Z"))
    assert ic.lead_eta_s(late, anchor_time_s=anchor_time_s) == ic.LEAD_ETA_CLIP_S
    early = replace(series[0], lead_landing=ic.LeadLanding("2026-06-30T20:00:00Z"))
    assert ic.lead_eta_s(early, anchor_time_s=anchor_time_s) == -ic.LEAD_ETA_CLIP_S
    # Consulted, no earlier landing: the negative clip. Never consulted: refused.
    none = replace(series[0], lead_landing=ic.LeadLanding(None))
    assert ic.lead_eta_s(none, anchor_time_s=anchor_time_s) == -ic.LEAD_ETA_CLIP_S
    with pytest.raises(ValueError, match="no scene context"):
        ic.lead_eta_s(replace(series[0], lead_landing=None), anchor_time_s=anchor_time_s)
    del item.scenario.source["entry_time_utc"]
    with pytest.raises(ValueError, match="entry_time_utc"):
        ic.lead_eta_s(item, anchor_time_s=anchor_time_s)


def test_build_series_carries_the_lead_from_the_flight_dict():
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=2, seed=3)
    flights[0]["lead_landing"] = ic.LeadLanding("2026-07-01T00:01:00Z")
    series, _report = build_series(flights, TSConfig(), airport=AIRPORT)
    assert [s.lead_landing for s in series] == [ic.LeadLanding("2026-07-01T00:01:00Z"), None]


# ── The conditioning row through the windows and the model ───────────────────

def test_intent_windows_carry_one_constant_row_and_the_model_predicts_six_channels():
    series, config = _series(
        n_flights=3, seq_len=20, n_segments=4, intent_conditioning="truth-join-lead",
        d_model=16, n_heads=4, d_ff=32, e_layers=1,
    )
    normalizer = Normalizer.fit(series)
    windows = FixedAnchorTrajectoryWindows(series, config, normalizer)
    extra = len(INTENT_JOIN_CHANNELS) + len(INTENT_LEAD_CHANNELS)
    x, y, _weights, _final_time_s, _flight_weight = windows[0]
    assert x.shape == (config.seq_len, len(ch.CHANNELS) + extra)
    assert y.shape == (config.pred_len, len(ch.CHANNELS))
    row = windows.conditioning[0]
    assert np.allclose(x[:, len(ch.CHANNELS):].numpy(), row[None, :])
    position = list(ch.POSITION_IDX)
    assert np.allclose(
        row[:3],
        (ic.truth_join_point(series[0]) - normalizer.mean[position]) / normalizer.std[position],
    )
    anchor = windows.index[0][1]
    assert anchor == config.seq_len - 1
    eta = ic.lead_eta_s(series[0], anchor_time_s=float(series[0].times[anchor]))
    assert row[3] == pytest.approx(eta / ic.LEAD_ETA_SCALE_S)
    # Flights with different leads get different rows: the channel carries information.
    assert windows.conditioning[0][3] != windows.conditioning[1][3]
    batch = windows.batch(np.arange(len(windows)))
    prediction = build_model(config)(batch[0])
    assert prediction.states.shape == (len(windows), config.pred_len, len(ch.CHANNELS))
    # Inference builds the SAME augmented history the training windows carried.
    history = _history_at_anchor(series[0], config, normalizer, anchor)
    assert np.allclose(history, x.numpy())
    # Off: no row at all.
    assert dataset_module.series_conditioning(
        series[0], TSConfig(), normalizer, anchor=anchor
    ) is None


def test_the_row_is_built_at_the_windows_actual_anchor():
    series, config = _series(n_flights=2, seq_len=20, intent_conditioning="truth-join-lead")
    normalizer = Normalizer.fit(series)
    later = config.seq_len + 9
    windows = FixedAnchorTrajectoryWindows(
        series, config, normalizer, minimum_anchor_index=later
    )
    assert windows.index[0][1] == later
    eta = ic.lead_eta_s(series[0], anchor_time_s=float(series[0].times[later]))
    assert windows.conditioning[0][3] == pytest.approx(eta / ic.LEAD_ETA_SCALE_S)
    # Twenty seconds later on the clock, the lead is twenty seconds nearer.
    default = FixedAnchorTrajectoryWindows(series, config, normalizer)
    assert (default.conditioning[0][3] - windows.conditioning[0][3]) * ic.LEAD_ETA_SCALE_S == (
        pytest.approx((later - (config.seq_len - 1)) * config.dt_s)
    )


def test_duration_mode_hands_the_duration_head_its_own_target():
    from config import INTENT_DURATION_CHANNELS

    series, config = _series(n_flights=2, seq_len=20, intent_conditioning="truth-join-duration")
    assert config.input_channels == ch.CHANNELS + INTENT_JOIN_CHANNELS + INTENT_DURATION_CHANNELS
    normalizer = Normalizer.fit(series)
    windows = FixedAnchorTrajectoryWindows(series, config, normalizer)
    _x, _y, _w, final_time_s, _fw = windows[0]
    s_idx, anchor = windows.index[0]
    row = windows.conditioning[s_idx]
    # The channel is the duration head's own target, on the loss's own time scale.
    assert row[3] == pytest.approx(float(final_time_s) / config.final_time_scale_s)
    assert ic.remaining_time_s(
        series[s_idx], anchor_time_s=float(series[s_idx].times[anchor])
    ) == pytest.approx(float(final_time_s))
    with pytest.raises(ValueError, match="fixed anchor"):
        TSConfig(random_train_anchor=True, intent_conditioning="truth-join-duration")


def test_target_and_intent_rows_concatenate_in_input_channel_order():
    series, config = _series(
        n_flights=1, coordinate_frame="airport-enu", target_conditioning="channels",
        intent_conditioning="truth-join",
    )
    normalizer = Normalizer.fit(series)
    row = dataset_module.series_conditioning(
        series[0], config, normalizer, anchor=config.seq_len - 1
    )
    assert row.shape == (len(CONDITIONING_CHANNELS) + len(INTENT_JOIN_CHANNELS),)
    psi = series[0].scenario.target.psi
    assert row[3:5] == pytest.approx([math.cos(psi), math.sin(psi)])
    position = list(ch.POSITION_IDX)
    assert np.allclose(
        row[5:],
        (ic.truth_join_point(series[0]) - normalizer.mean[position]) / normalizer.std[position],
    )


def test_control_training_step_and_checkpoint_round_trip_at_the_intent_width(tmp_path):
    # The control loss validates the anchor width [B, C]; a training loop that handed it
    # the whole conditioned row died on the first batch (found in review).
    series, config = _series(
        n_flights=12, prediction_output=PREDICTION_CONTROL, intent_conditioning="truth-join-lead",
        epochs=1, patience=1, batch_size=32, d_model=16, n_heads=4, d_ff=32, e_layers=1,
        seq_len=20, n_segments=4, device="cpu",
    )
    train(
        series, config, output_dir=tmp_path,
        data_provenance=_fake_data_provenance(), verbose=False,
    )
    model, loaded, normalizer, payload = load_checkpoint(tmp_path / "checkpoint.pt")
    assert loaded.intent_conditioning == "truth-join-lead"
    assert payload["input_channels"] == list(config.input_channels)
    forecast = forecast_approach(
        model, series[0], loaded, normalizer, device=torch.device("cpu")
    )
    assert forecast.values.shape[1] == len(ch.CHANNELS)
