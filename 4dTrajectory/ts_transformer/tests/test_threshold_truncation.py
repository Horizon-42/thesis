"""``--truncate-at-threshold``: reading the approach, not the flying after it.

L3.d's geometry readout is the reason. With the speed floor holding the commanded speed up,
a rollout asked to arrive late reaches the threshold EARLY, flies on and turns — endpoint
``|xt|`` p95 43-63 km, pooled ADE 840 -> 3443 m at offset 0 — and every flyability, corridor
and CTA number taken over the whole record is scoring that tail. Cut at the crossing and the
same rollout reads on the approach proper (+60 s: fully flyable 1.35 % -> 48.9 %).

The rule under test (``final_approach_geometry.threshold_crossing_index``, shared with the
trombone's reference path): the crossing is the first row at or past the threshold plane
(``d <= 0``) that is also ON the final there — inside the on-final gate's membership cone and
aligned with the course — and the cut is the closest horizontal approach to the target within
that FIRST crossing run. A forecast that never crosses on the final is returned whole and
still says ``truncatedAtThreshold: false``, while one that crosses on its last row is whole
with the flag TRUE (the flag means "this record ends at the threshold"); and the two clocks a
control record carries stay aligned, because ``export`` refuses them otherwise.

The plane ALONE was the rule until 2026-09-09, and it fired ABEAM: a vectored flight's
downwind runs parallel to the course and opposite it, several kilometres wide, and passes
``d = 0`` out there. Measured on ``L3e_stack_p60s_pred_val``, 96.5 % of the vectored cuts lay
more than 1 km from the threshold (median ``|xt|`` 8.7 km, a median 1.75 km above it) against
a straight-in ``|xt|`` p95 of 50 m — so the L3.e/L3.f flyability and geometry of every
vectored arm were read on a window that ended out on the downwind.
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
for path in (TS_DIR.parent, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ts_transformer.channels import IDX  # noqa: E402
from ts_transformer.config import (  # noqa: E402
    CONTROL_DYNAMICS_FIRST_ORDER_LAG, PREDICTION_CONTROL, TSConfig,
)
from ts_transformer.dataset import Normalizer, build_series  # noqa: E402
from evaluation.records import record_from_dict  # noqa: E402
from ts_transformer.export import build_prediction_record  # noqa: E402
from ts_transformer.final_approach_geometry import chart_from_axes, runway_axes  # noqa: E402
from ts_transformer.forecast import (  # noqa: E402
    Forecast, cut_at_threshold_crossing, forecast_approach, forecast_approaches,
)
from ts_transformer.outputs import ForecastOptions  # noqa: E402
from ts_transformer.outputs.control.heads import ControlPrediction  # noqa: E402
from ts_transformer.synthetic import synthetic_arrivals  # noqa: E402

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


def _series(config: TSConfig, *, entry_offset_deg: float = 50.0):
    """One synthetic arrival. ``entry_offset_deg`` is `synthetic_arrivals`' own default —
    0 gives a straight-in, the package default an entry up to 50° off the reciprocal."""
    flights = synthetic_arrivals(AIRPORT, RUNWAY, n_flights=1, seed=5,
                                 entry_offset_deg=entry_offset_deg)
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
    """Straight in from 6 km and 2 km out the other side: the record stops at the runway.

    Also the no-change case for the 2026-09-09 rule: a straight-in is inside the cone and
    aligned at the plane, so the on-final crossing IS the first row past the plane — the row
    the plane-only rule answered. Every straight-in record cuts exactly where it did.
    """
    series = _series(_config())
    d_m = np.linspace(6_000.0, -2_000.0, 41)
    forecast = _forecast_along(series, d_m, np.zeros_like(d_m), segments=8)
    cut = cut_at_threshold_crossing(forecast, series)
    assert cut.truncated_at_threshold
    crossing = int(np.argmax(d_m <= 0.0))                     # the plane-only rule's answer
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
    """No crossing, no cut — and no claim of one. Inventing an arrival would be worse.

    "Whole" is every row and both clocks untouched; the flag is CLEARED rather than left,
    because on a fixed-time state forecast the postprocessor's own closest-approach rule may
    have set it on a record that ends nowhere near the threshold.
    """
    series = _series(_config())
    d_m = np.linspace(9_000.0, 1_500.0, 25)
    forecast = _forecast_along(series, d_m, np.zeros_like(d_m), segments=5)
    cut = cut_at_threshold_crossing(replace(forecast, truncated_at_threshold=True), series)
    assert not cut.truncated_at_threshold
    assert np.array_equal(cut.values, forecast.values)
    assert cut.n_steps == forecast.n_steps and cut.final_time_s == forecast.final_time_s
    assert np.array_equal(cut.segment_durations_s, forecast.segment_durations_s)


def _plane_only_cut(d_m: np.ndarray, xt_m: np.ndarray) -> int:
    """What the rule answered before 2026-09-09: the closest approach among the rows past
    the plane, scoped to the first such run — the threshold PLANE and nothing else."""
    past = d_m <= 0.0
    first = int(np.argmax(past))
    run = past[first:]
    end = len(past) if run.all() else first + int(np.argmin(run))
    return first + int(np.argmin(np.hypot(d_m, xt_m)[first:end]))


def test_a_downwind_that_passes_the_plane_abeam_is_not_a_crossing():
    """The defect, as its own fixture: parallel to the course, opposite it, 8 km abeam.

    A vectored flight's downwind flies the reciprocal of the landing direction, so its ``d``
    RISES through zero — out where the runway is a fly-past, not a landing. The plane-only
    rule cut the record there, 8 km from the threshold; the cone refuses it (8 km against a
    500 m floor) and so does the alignment (180° off), so this forecast never crossed and is
    returned whole.
    """
    series = _series(_config())
    d_m = np.linspace(-8_000.0, 8_000.0, 32)
    xt_m = np.full_like(d_m, 8_000.0)
    forecast = _forecast_along(series, d_m, xt_m, segments=8)
    cut = cut_at_threshold_crossing(forecast, series)
    assert not cut.truncated_at_threshold
    assert cut.n_steps == forecast.n_steps and np.array_equal(cut.values, forecast.values)
    # ...and the fixture really does cross the plane, which is the whole point: the rule that
    # read the plane alone cut it 8 km out and called that an arrival.
    assert (d_m <= 0.0).any() and (d_m > 0.0).any()
    assert np.hypot(d_m, xt_m)[_plane_only_cut(d_m, xt_m)] > 8_000.0


def test_a_downwind_base_and_final_is_cut_on_the_final_not_on_the_abeam_pass():
    """The vectored arrival in full: downwind 8 km right, a base turn, then the final.

    The same track carries BOTH crossings of the plane — the abeam one on the downwind (row
    3, 8 km out) and the real one on the final — and they are 8 km and most of the record
    apart. The rule takes the second, because the first is not a landing.
    """
    series = _series(_config())
    legs, rows = ((-3_000.0, 8_000.0), (9_000.0, 8_000.0), (6_000.0, 0.0), (-1_500.0, 0.0)), 12
    d_m, xt_m = np.array([legs[0][0]]), np.array([legs[0][1]])
    for start, end in zip(legs, legs[1:]):
        d_m = np.concatenate((d_m, np.linspace(start[0], end[0], rows + 1)[1:]))
        xt_m = np.concatenate((xt_m, np.linspace(start[1], end[1], rows + 1)[1:]))
    # Row 34: the final leg runs from row 25 (d = 6 km) in 625 m steps, so its eleventh row
    # is the first at or past the plane — a literal, as the sibling test's is.
    expected = 34
    assert len(d_m) == 37 and d_m[expected] < 0.0 <= d_m[expected - 1]
    forecast = _forecast_along(series, d_m, xt_m, segments=6)
    cut = cut_at_threshold_crossing(forecast, series)
    assert cut.truncated_at_threshold
    assert cut.n_steps - 1 == expected
    # The abeam pass is where the plane-only rule cut: row 3, exactly abeam the threshold
    # and 8 km wide of it, most of the record before the real arrival.
    assert _plane_only_cut(d_m, xt_m) == 3
    assert np.hypot(d_m, xt_m)[3] == 8_000.0 and expected - 3 > 25


def test_the_cut_is_the_closest_approach_of_the_FIRST_crossing():
    """A rollout that overshoots, goes around and comes back is cut on its real arrival.

    Eleven hand-placed rows, so the answer is a literal rather than a second copy of the
    rule. Rows 3-4 are the first crossing run — past the plane, inside the cone and aligned,
    the direction being the central difference over the neighbouring rows — and row 4 is the
    nearer of the two (372 m against 400 m). Row 5 leaves the final, rows 6-7 fly out, and
    rows 9-10 are a SECOND
    crossing that comes nearer still (72 m at row 9). Row 2, the last row before the
    crossing, is nearer than either row of the first run.

    So a plain global ``argmin`` answers 9, "the first row at or past the plane" answers 3,
    and "closest approach ignoring the plane, up to the crossing" answers 2. The rule
    answers 4. The margins are deliberately small and are stated here so a mirror change is
    read as one: row 4's direction is 28.2° against ``ALIGNMENT_MAX_DEG`` = 30, and the run
    ends at row 5 because the go-around's turn puts it at 154°. (No row sits at ``d`` exactly
    0: the fixture round-trips through `chart_from_axes` and back, and a row placed on the
    plane could land either side of it by an ULP.)
    """
    series = _series(_config())
    d_m = np.array([2_000.0, 900.0, 180.0, -20.0, -220.0, -300.0,
                    400.0, 1_200.0, 300.0, -40.0, -250.0])
    xt_m = np.array([1_500.0, 700.0, 150.0, 400.0, 300.0, 250.0,
                     600.0, 900.0, 200.0, 60.0, 30.0])
    assert list(np.round(np.hypot(d_m, xt_m)[2:6])) == [234.0, 400.0, 372.0, 391.0]
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


def _crossing_geometry(series, forecast):
    """``(d, xt)`` of a forecast's rows about the target — the axes the rule reads."""
    psi = torch.tensor([float(series.scenario.target.psi)], dtype=torch.float64)
    e = torch.from_numpy(forecast.values[:, IDX["e"]] - series.target_chart[0])[None]
    n = torch.from_numpy(forecast.values[:, IDX["n"]] - series.target_chart[1])[None]
    d, xt = runway_axes(e, n, psi)
    return d[0].numpy(), xt[0].numpy()


def test_predict_is_bit_identical_without_the_flag_and_cuts_with_it():
    """The flag is the only thing that changes: default off, and off means untouched.

    A STRAIGHT-IN synthetic arrival (`entry_offset_deg=0`), because a rollout has to reach
    the final for there to be a cut at all — see the vectored case below, which is the same
    model on the package's default 50°-offset entry and does not.
    """
    config = _config(n_segments=2)
    series = _series(config, entry_offset_deg=0.0)
    normalizer = Normalizer.fit([series])
    model, device = _StraightControlModel(), torch.device("cpu")
    default = forecast_approach(model, series, config, normalizer, device=device)
    off = forecast_approaches(model, [series], config, normalizer, device=device,
                              options=ForecastOptions(truncate_at_threshold=False))[0]
    assert np.array_equal(default.values, off.values)
    assert default.final_time_s == off.final_time_s
    assert not default.truncated_at_threshold
    cut = forecast_approaches(model, [series], config, normalizer, device=device,
                              options=ForecastOptions(truncate_at_threshold=True))[0]
    # The fixture is only worth anything if the rollout DOES cross; assert that it does.
    assert cut.truncated_at_threshold
    assert np.array_equal(cut.values, cut_at_threshold_crossing(default, series).values)
    assert cut.final_time_s < default.final_time_s
    assert cut.n_steps < default.n_steps
    # ...and it cuts where it always did: on a straight-in the on-final crossing IS the
    # first row past the plane, so the 2026-09-09 rule moves this record by nothing.
    d, xt = _crossing_geometry(series, default)
    assert cut.n_steps - 1 == int(np.argmax(d <= 0.0))
    assert abs(xt[cut.n_steps - 1]) < 300.0
    assert math.isclose(cut.segment_durations_s.sum(), cut.final_time_s, rel_tol=1e-9)


def test_a_rollout_that_passes_the_plane_ABEAM_is_not_an_arrival():
    """The defect, end to end and on the package's own synthetic fleet.

    The same straight-flying model on the DEFAULT synthetic entry (up to 50° off the
    reciprocal) flies a track that never turns onto the final: it passes the threshold plane
    8.7 km abeam, 45° off the course — the shape of a downwind, and of 96.5 % of the vectored
    cuts the plane-only rule made on ``L3e_stack_p60s_pred_val``. It is not an arrival, so
    the record is left WHOLE and says so; the flag is what a reader checks.
    """
    config = _config(n_segments=2)
    series = _series(config)                      # entry_offset_deg=50, the default
    model, device = _StraightControlModel(), torch.device("cpu")
    forecast = forecast_approaches(
        model, [series], config, Normalizer.fit([series]), device=device,
        options=ForecastOptions(truncate_at_threshold=True),
    )[0]
    assert not forecast.truncated_at_threshold
    assert forecast.n_steps == forecast_approach(
        model, series, config, Normalizer.fit([series]), device=device).n_steps
    # The fixture is only worth anything if it DOES cross the plane: it is the cone and the
    # alignment that refuse it, not a rollout that stopped short.
    d, xt = _crossing_geometry(series, forecast)
    abeam = int(np.argmax(d <= 0.0))
    assert (d <= 0.0).any() and abs(xt[abeam]) > 8_000.0
