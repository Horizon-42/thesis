"""The closure tracker: the point-mass rollout flying a drawn closure reference."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

TS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TS_DIR.parents[1]
for path in (TS_DIR, TS_DIR / "tests", REPO_ROOT, REPO_ROOT / "geokit" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import channels as ch  # noqa: E402
import closure_geometry as cg  # noqa: E402
import closure_output as co  # noqa: E402
import geometric_metrics as gm  # noqa: E402
from control.constraints.closure_tracking import HOOK_NAME, reference_path  # noqa: E402
from control.envelope import CONTROL_LOWER, CONTROL_UPPER, fraction_controls  # noqa: E402
from export import build_prediction_record, observed_series_metrics, write_batch  # noqa: E402
from forecast import forecast_closure_from_labels  # noqa: E402
from test_closure_output import _closure_series  # noqa: E402


def test_reference_path_reads_course_curvature_and_slope():
    # A left-turning arc of radius 2 km at 80 m/s descending 1 m per 20 m.
    theta = np.linspace(0.0, np.pi / 2, 200)
    xy = np.stack([2_000.0 * np.sin(theta), 2_000.0 * (1.0 - np.cos(theta))], 1)
    s = 2_000.0 * theta
    ref = reference_path(xy, 900.0 - s / 20.0, np.full(200, 80.0), s / 80.0)
    assert np.allclose(ref.course[1:-1], theta[1:-1], atol=0.02)
    assert np.allclose(ref.arc, s, atol=1.0) and ref.times[-1] == pytest.approx(s[-1] / 80.0)
    assert np.allclose(ref.curvature[2:-2], 1.0 / 2_000.0, rtol=0.05)
    assert np.allclose(ref.slope[1:-1], -1.0 / 20.0, rtol=0.05)


def test_tracked_forecast_flies_the_reference_within_the_envelope(tmp_path):
    # 32 held segments: a hook steers once per segment, and a 100 s hold (the tiny 4-segment
    # config) cannot track anything.
    series, config, labels = _closure_series(tmp_path, n_flights=3, n_segments=32)
    drawn = forecast_closure_from_labels(series, config, labels)
    flown = forecast_closure_from_labels(series, config, labels, track=True, device=torch.device("cpu"))
    for item, reference, tracked in zip(series, drawn, flown):
        assert tracked.closure_tracked and tracked.closure_from_labels and tracked.command_hook == HOOK_NAME
        assert tracked.controls is not None and tracked.controls.shape == (config.n_segments, 3)
        assert tracked.values.shape[1] == len(ch.CHANNELS) and tracked.geodetic_values is not None
        # The rollout runs for the reference's own duration, its clock strictly increasing.
        assert tracked.final_time_s == pytest.approx(reference.final_time_s, rel=1e-6)
        assert np.all(tracked.sample_durations_s > 0.0)
        # The controls flown sit inside the envelope (fractions of installed thrust).
        max_thrust = float(item.scenario.aircraft.engine.max_thrust_total_n)
        fractions = fraction_controls(tracked.controls, np.array(max_thrust))
        assert np.all(fractions >= CONTROL_LOWER - 1e-6) and np.all(fractions <= CONTROL_UPPER + 1e-6)
        # The dynamics stay close to the drawn reference in geometry (chamfer well under
        # the family's own residuals); how much the speed hold lags the reference at the
        # end is the quantity the P1.d campaign measures, not a spec — only bounded here.
        assert gm.chamfer_m(tracked.values[:, :2], reference.values[:, :2]) < 150.0
        assert np.hypot(*tracked.values[-1, :2]) < 1_500.0
        metrics = observed_series_metrics(item, tracked)
        assert np.isfinite(metrics["ade_m"]) and metrics["ade_m"] < 1_000.0
    # The record contract: controls flown in newtons, the closure provenance in source.
    records = [build_prediction_record(item, f, index=i, model_name=config.model, horizon_mode=config.horizon_mode)
               for i, (item, f) in enumerate(zip(series, flown))]
    write_batch(records, output_dir=tmp_path / "pred", config_dict=config.to_dict(),
                flight_metrics=[observed_series_metrics(item, f) for item, f in zip(series, flown)])
    summary = json.loads((tmp_path / "pred" / "summary.json").read_text())
    states = json.loads((tmp_path / "pred" / summary["results"][0]["states_file"]).read_text())
    assert states["source"]["closureTracked"] is True and states["source"]["commandHook"] == HOOK_NAME
    assert len(states["control_segments"]) == config.n_segments
