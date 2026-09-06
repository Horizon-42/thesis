"""The scene context: who is around the ego at t₀, with only what was observable then."""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "geokit" / "src", str(Path(__file__).resolve().parent)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from geokit import compass_bearing_to_math_enu_rad  # noqa: E402
from scene_fixture import PSI, RUNWAY, T0, TARGET, axes_to_chart, chart_to_latlon, standard_scene  # noqa: E402
from trajectory_data_process.scene_index import build_scene_index  # noqa: E402
import flight_scenarios.scene_context as sc  # noqa: E402


def _ego_kwargs(keys):
    e, n = axes_to_chart(15_000.0, 0.0)
    lat, lon = chart_to_latlon(e, n)
    return dict(ego_flight_key=keys["EGO1"], ego_runway=RUNWAY, ego_target=TARGET, t0_utc_s=T0,
                ego_lat=lat, ego_lon=lon, ego_alt_hae_m=TARGET["elevation_hae_m"] + 800.0, ego_ground_speed_mps=80.0)


def test_runway_axes_match_the_ts_chart_convention():
    frame = sc.ego_frame(RUNWAY, TARGET)
    e, n = axes_to_chart(6_000.0, 300.0)
    lat, lon = chart_to_latlon(e, n)
    d, xt, h = sc.runway_axes(frame, lat, lon, TARGET["elevation_hae_m"] + 250.0)
    assert d == pytest.approx(6_000.0, abs=1.0) and xt == pytest.approx(300.0, abs=1.0) and h == pytest.approx(250.0)
    # The inbound course in math-ENU is geokit's conversion of the compass course.
    assert compass_bearing_to_math_enu_rad(math.radians(TARGET["course_deg"])) == pytest.approx(PSI)


def test_the_scene_reads_the_past_only(tmp_path):
    paths, keys = standard_scene(tmp_path)
    index = build_scene_index(paths, verbose=False)
    scene = sc.scene_context(paths, index, **_ego_kwargs(keys))
    by_key = {nb.flight_key: nb for nb in scene.neighbours}
    # The ego is not its own neighbour; B (samples only after t₀) and D (60 km) are absent;
    # E and F landed before the window. A and C remain.
    assert set(by_key) == {keys["AHEAD"], keys["DWIND"]}
    assert scene.candidates_in_window == 3 and scene.in_radius == 2
    for nb in scene.neighbours:
        assert np.all(nb.observed.t_rel_s <= 0.0) and np.all(nb.observed.t_rel_s >= -scene.window_s)
    # A: 6 km out on the final, inbound, established, ahead of the ego by ETA.
    a = by_key[keys["AHEAD"]].observed
    assert a.d_m == pytest.approx(13_000.0 * 90.0 / 190.0, abs=5.0) and abs(a.xt_m - 30.0) < 5.0   # 13 km at T0−100 → 0 at T0+90
    assert a.established and abs(sc_wrap(a.heading_rad - PSI)) < math.radians(2.0)
    assert a.ground_speed_mps == pytest.approx(13_000.0 / 190.0, rel=0.05) and a.eta_s < scene.ego_eta_s
    assert a.height_m == pytest.approx(400.0 - TARGET["elevation_hae_m"], abs=1.0)
    # C: on the downwind, outbound — not established, far by ETA.
    c = by_key[keys["DWIND"]].observed
    assert not c.established and abs(c.xt_m - 8_000.0) < 50.0 and math.cos(c.heading_rad - PSI) < -0.9
    # Nearest to the ego first.
    assert [nb.flight_key for nb in scene.neighbours] == sorted(by_key, key=lambda k: by_key[k].distance_to_ego_m)
    # The future lives in future_label only: A's landing time is after t₀ and is not in Observed.
    assert by_key[keys["AHEAD"]].future_label.landing_utc_s == T0 + 90.0 > T0
    assert by_key[keys["AHEAD"]].future_label.runway == RUNWAY
    assert not any(field for field in vars(a) if "landing" in field or "runway" in field or "outcome" in field)
    # Scalars come from the PAST landings: E 300 s ago on the ego's runway, F 1000 s ago on 23R.
    s = scene.scalars
    assert s.since_last_landing_same_runway_s == pytest.approx(300.0)
    assert s.landings_recent == 2 and s.landings_recent_same_runway == 1 and s.same_runway_share_recent == 0.5
    assert s.airborne_in_radius == 2 and s.established_on_ego_final == 1 and s.ahead_by_eta == 1
    assert s.lead_eta_s == pytest.approx(a.eta_s) and s.lead_gap_s == pytest.approx(scene.ego_eta_s - a.eta_s)
    assert s.hour_utc == 12.0 and s.weekday == 4


def test_window_radius_and_n_max_are_applied_and_stated(tmp_path):
    paths, keys = standard_scene(tmp_path)
    index = build_scene_index(paths, verbose=False)
    wide = sc.scene_context(paths, index, radius_m=80_000.0, **_ego_kwargs(keys))
    assert {nb.flight_key for nb in wide.neighbours} >= {keys["FAR"]} and wide.radius_m == 80_000.0
    one = sc.scene_context(paths, index, n_max=1, **_ego_kwargs(keys))
    assert len(one.neighbours) == 1 and one.n_max == 1 and one.in_radius == 2
    # A shorter window drops a neighbour whose last sample is older than it.
    short = sc.scene_context(paths, index, window_s=1.0, **_ego_kwargs(keys))
    assert len(short.neighbours) == 0 and short.window_s == 1.0


def test_the_scene_is_deterministic_and_the_reader_cache_changes_nothing(tmp_path):
    paths, keys = standard_scene(tmp_path)
    index = build_scene_index(paths, verbose=False)
    reader = sc._Reader(paths)
    first = sc.scene_context(paths, index, reader=reader, **_ego_kwargs(keys))
    second = sc.scene_context(paths, index, reader=reader, **_ego_kwargs(keys))
    third = sc.scene_context(paths, index, **_ego_kwargs(keys))
    for a, b in ((first, second), (first, third)):
        assert [nb.flight_key for nb in a.neighbours] == [nb.flight_key for nb in b.neighbours]
        assert a.scalars == b.scalars
        for x, y in zip(a.neighbours, b.neighbours):
            assert np.array_equal(x.observed.e_m, y.observed.e_m) and x.observed.eta_s == y.observed.eta_s


def sc_wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
