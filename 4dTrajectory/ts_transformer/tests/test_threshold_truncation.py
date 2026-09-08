"""``--truncate-at-threshold``: reading the approach, not the flying after it.

L3.d's geometry readout is the reason. With the speed floor holding the commanded speed up,
a rollout asked to arrive late reaches the threshold EARLY, flies on and turns — endpoint
``|xt|`` p95 43-63 km, pooled ADE 840 -> 3443 m at offset 0 — and every flyability, corridor
and CTA number taken over the whole record is scoring that tail. Cut at the crossing and the
same rollout reads on the approach proper (+60 s: fully flyable 1.35 % -> 48.9 %).

The rule under test: the cut is the closest horizontal approach to the target among the rows
at or past the threshold plane (``d <= 0``), scoped to the FIRST such run; a forecast that
never gets there is returned whole and still says ``truncatedAtThreshold: false``, while one
that crosses on its last row is whole with the flag TRUE (the flag means "this record ends at
the threshold"); and the two clocks a control record carries stay aligned, because ``export``
refuses them otherwise.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

TS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TS_DIR.parents[1]
for path in (TS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    CONTROL_DYNAMICS_FIRST_ORDER_LAG, PREDICTION_CONTROL, TSConfig,
)
from dataset import Normalizer, build_series  # noqa: E402
from evaluation.records import record_from_dict  # noqa: E402
from export import build_prediction_record  # noqa: E402
from final_approach_geometry import chart_from_axes  # noqa: E402
from forecast import (  # noqa: E402
    Forecast, cut_at_threshold_crossing, forecast_approach, forecast_approaches,
)
from prediction_outputs import ControlPrediction  # noqa: E402
from synthetic import synthetic_arrivals  # noqa: E402

AIRPORT, RUNWAY = "KRDU", "05L"
STEP_S = 5.0


def _config(**overrides) -> TSConfig:
    settings = {
        "prediction_output": PREDICTION_CONTROL,
        "control_dynamics_model": CONTROL_DYNAMICS_FIRST_ORDER_LAG,
        "control_dynamics_backend": "scaled-transport-chart-velocity",
        "seq_len": 8, "n_segments": 4,
    }
    settings.update(overrides)
    return TSConfig(**settings)


def _series(config: TSConfig):
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=1, seed=5)
    series, _ = build_series(flights, config, airport=AIRPORT)
    return series[0]


def _forecast_along(series, d_m: np.ndarray, xt_m: np.ndarray, *, segments: int) -> Forecast:
    """A control forecast whose rows walk the runway axes ``(d, xt)``, one step each.

    The dense samples and the control segments are two clocks, as they are on the real
    thing: ``segments`` control segments spanning exactly the same span as the samples.
    """
    psi = torch.tensor([float(series.scenario.target.psi)], dtype=torch.float64)
    e, n = chart_from_axes(torch.from_numpy(d_m)[None], torch.from_numpy(xt_m)[None], psi)
    values = np.zeros((len(d_m), 6), dtype=np.float64)
    values[:, 0] = e[0].numpy() + series.target_chart[0]
    values[:, 1] = n[0].numpy() + series.target_chart[1]
    values[:, 2] = np.linspace(400.0, 0.0, len(d_m))
    sample_durations = np.full(len(d_m), STEP_S)
    total = float(sample_durations.sum())
    geodetic = np.zeros((len(d_m), 7), dtype=np.float64)
    geodetic[:, 3] = 70.0
    return Forecast(
        times=float(series.times[0]) + np.cumsum(sample_durations),
        values=values,
        normalized_progress=np.cumsum(sample_durations) / total,
        anchor=0,
        final_time_s=total,
        predicted_final_time_s=total,
        horizon_mode="normalized",
        passes=1,
        truncated_at_threshold=False,
        horizon_capped=False,
        sample_durations_s=sample_durations,
        segment_durations_s=np.full(segments, total / segments),
        controls=np.tile(np.array([0.2, 0.0, 1.0]), (segments, 1)),
        geodetic_values=geodetic,
        prediction_output=PREDICTION_CONTROL,
    )


def test_a_forecast_that_flies_past_the_threshold_is_cut_at_the_crossing():
    """Straight in from 6 km and 2 km out the other side: the record stops at the runway."""
    series = _series(_config())
    d_m = np.linspace(6_000.0, -2_000.0, 41)
    forecast = _forecast_along(series, d_m, np.zeros_like(d_m), segments=8)
    cut = cut_at_threshold_crossing(forecast, series)
    assert cut.truncated_at_threshold
    crossing = int(np.argmax(d_m <= 0.0))
    assert cut.n_steps == crossing + 1
    assert cut.final_time_s == pytest.approx((crossing + 1) * STEP_S)
    assert cut.times[-1] == forecast.times[crossing]
    assert cut.normalized_progress[-1] == pytest.approx(1.0)
    # The control clock is cut to the segment holding the new end, and that segment is
    # shortened to land exactly on it — the two clocks have to end together.
    assert cut.segment_durations_s.sum() == pytest.approx(cut.final_time_s)
    assert len(cut.controls) == len(cut.segment_durations_s) < len(forecast.controls)
    assert cut.geodetic_values is not None and len(cut.geodetic_values) == cut.n_steps
    # The forecast itself is untouched: the cut returns a new one.
    assert not forecast.truncated_at_threshold and forecast.n_steps == len(d_m)


def test_a_forecast_that_never_reaches_the_threshold_is_left_whole_and_says_so():
    """No crossing, no cut — and no claim of one. Inventing an arrival would be worse."""
    series = _series(_config())
    d_m = np.linspace(9_000.0, 1_500.0, 25)
    forecast = _forecast_along(series, d_m, np.zeros_like(d_m), segments=5)
    cut = cut_at_threshold_crossing(forecast, series)
    assert cut is forecast
    assert not cut.truncated_at_threshold


def test_the_cut_is_the_closest_approach_of_the_FIRST_crossing():
    """A rollout that overshoots, wanders and comes back is cut on its real arrival.

    Eleven hand-placed rows, so the answer is a literal rather than a second copy of the
    rule. Rows 3-5 are the first run past the plane and row 4 is the nearest of them (403 m
    against 640 m and 510 m); rows 8-10 are a SECOND pass that comes nearer still (102 m at
    row 9), and row 2 — the last row before the crossing — is nearer than rows 3 and 5. So a
    plain global ``argmin`` answers 9, "the first row at or past the plane" answers 3, and
    "closest approach ignoring the plane, up to the crossing" answers 2. The rule answers 4.
    (No row sits at ``d`` exactly 0: the fixture round-trips through `chart_from_axes` and
    back, and a row placed on the plane could land either side of it by an ULP.)
    """
    series = _series(_config())
    d_m = np.array([2_000.0, 900.0, 300.0, -400.0, -50.0, -500.0,
                    600.0, 1_500.0, -100.0, -20.0, -300.0])
    xt_m = np.array([1_500.0, 700.0, 150.0, 500.0, 400.0, 100.0,
                     400.0, 900.0, 300.0, 100.0, 200.0])
    assert list(np.round(np.hypot(d_m, xt_m)[2:6])) == [335.0, 640.0, 403.0, 510.0]
    forecast = _forecast_along(series, d_m, xt_m, segments=4)
    cut = cut_at_threshold_crossing(forecast, series)
    assert cut.n_steps - 1 == 4                              # the hand-computed answer
    assert int(np.argmin(np.hypot(d_m, xt_m))) == 9          # not the global closest
    assert int(np.argmax(d_m <= 0.0)) == 3                   # not the first row past the plane


def test_a_forecast_that_crosses_on_its_LAST_row_is_whole_but_still_says_it_arrived():
    """Nothing to cut, but the record does end at the threshold — and must not read as one
    that never got there. The flag means "this record ends at the crossing"; only a forecast
    whose rows never reach ``d <= 0`` is whole AND false."""
    series = _series(_config())
    d_m = np.linspace(5_000.0, -50.0, 21)
    assert (d_m <= 0.0).sum() == 1 and d_m[-1] <= 0.0
    forecast = _forecast_along(series, d_m, np.zeros_like(d_m), segments=7)
    cut = cut_at_threshold_crossing(forecast, series)
    assert cut.truncated_at_threshold
    assert cut.n_steps == forecast.n_steps
    assert np.array_equal(cut.values, forecast.values)
    assert cut.final_time_s == forecast.final_time_s
    assert len(cut.segment_durations_s) == len(forecast.segment_durations_s)


def test_a_cut_that_lands_exactly_on_a_control_boundary_keeps_both_clocks():
    """The common case in production, and the one `searchsorted(side="left")` is chosen for.

    The dense query grid carries every control boundary, so the cut frequently falls exactly
    ON one. `side="left"` must then keep that segment whole rather than open a zero-length
    one after it.
    """
    series = _series(_config())
    rows, segments = 24, 6                    # 4 samples per segment: boundaries at 4, 8, ...
    d_m = np.linspace(6_000.0, -1_400.0, rows)
    crossing = int(np.argmax(d_m <= 0.0))
    assert (crossing + 1) % (rows // segments) == 0, crossing   # lands on a boundary
    forecast = _forecast_along(series, d_m, np.zeros_like(d_m), segments=segments)
    cut = cut_at_threshold_crossing(forecast, series)
    assert cut.n_steps == crossing + 1
    assert len(cut.segment_durations_s) == (crossing + 1) // (rows // segments)
    assert cut.segment_durations_s.sum() == pytest.approx(cut.final_time_s)
    # No zero-length tail segment, and the untouched segments keep their original duration.
    assert (cut.segment_durations_s > 0.0).all()
    assert cut.segment_durations_s[-1] == pytest.approx(forecast.segment_durations_s[0])


def test_a_state_forecast_is_cut_on_its_single_clock():
    """Every non-control forecast carries one clock: segments ARE samples, so both cut 1:1."""
    series = _series(_config())
    d_m = np.linspace(5_000.0, -1_000.0, 31)
    forecast = replace(
        _forecast_along(series, d_m, np.zeros_like(d_m), segments=31),
        controls=None, geodetic_values=None,
        segment_durations_s=np.full(31, STEP_S),
    )
    cut = cut_at_threshold_crossing(forecast, series)
    assert cut.controls is None
    assert len(cut.segment_durations_s) == len(cut.sample_durations_s) == cut.n_steps


def test_the_cut_record_still_satisfies_the_export_and_evaluation_contracts():
    """`export` refuses a record whose dense clock does not end at the control horizon, and
    the evaluation record contract requires `final_time_s == states[-1].t` with the controls
    1:1 ZOH-aligned. That pair is what the segment shortening exists for, and the only way to
    test it is to build the record and parse it back: a cut that broke either would raise.
    """
    series = _series(_config())
    d_m = np.linspace(6_000.0, -2_000.0, 41)
    cut = cut_at_threshold_crossing(
        _forecast_along(series, d_m, np.zeros_like(d_m), segments=8), series
    )
    record = build_prediction_record(
        series, cut, index=0, model_name="itransformer", horizon_mode="normalized"
    )
    assert record.source["truncatedAtThreshold"] is True
    assert record.eval_record["final_time_s"] == pytest.approx(cut.final_time_s)
    segments = record.states_payload["control_segments"]
    assert segments[-1]["end_t"] == pytest.approx(cut.final_time_s)
    assert all(row["duration_s"] > 0.0 for row in segments)
    parsed = record_from_dict(record.eval_record)
    assert parsed.states[-1]["t"] == pytest.approx(cut.final_time_s)
    assert len(parsed.controls) == len(parsed.states)


class _StraightControlModel(torch.nn.Module):
    """Two long segments of near-trim thrust: enough to fly the anchor past the threshold."""

    def forward(self, history, dynamics):
        controls = torch.tensor(
            [[[0.10, 0.0, 0.999], [0.10, 0.0, 0.999]]], dtype=history.dtype
        ).expand(len(history), -1, -1)
        durations = torch.tensor([[200.0, 200.0]], dtype=history.dtype).expand(len(history), -1)
        return ControlPrediction(
            controls=controls, segment_durations=durations, final_time_s=durations.sum(dim=-1)
        )


def test_predict_is_bit_identical_without_the_flag_and_cuts_with_it():
    """The flag is the only thing that changes: default off, and off means untouched."""
    config = _config(n_segments=2)
    series = _series(config)
    normalizer = Normalizer.fit([series])
    model, device = _StraightControlModel(), torch.device("cpu")
    default = forecast_approach(model, series, config, normalizer, device=device)
    off = forecast_approaches(model, [series], config, normalizer, device=device,
                              truncate_at_threshold=False)[0]
    assert np.array_equal(default.values, off.values)
    assert default.final_time_s == off.final_time_s
    assert not default.truncated_at_threshold
    cut = forecast_approaches(model, [series], config, normalizer, device=device,
                              truncate_at_threshold=True)[0]
    # The fixture is only worth anything if the rollout DOES cross; assert that it does.
    assert cut.truncated_at_threshold
    assert np.array_equal(cut.values, cut_at_threshold_crossing(default, series).values)
    assert cut.final_time_s < default.final_time_s
    assert cut.n_steps < default.n_steps
    assert math.isclose(cut.segment_durations_s.sum(), cut.final_time_s, rel_tol=1e-9)
