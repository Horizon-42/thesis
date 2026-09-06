"""Scene context → arrays: the ego grid, the masks, and the wall between observed and future."""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

TS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TS_DIR.parents[1]
for path in (TS_DIR, REPO_ROOT, REPO_ROOT / "geokit" / "src", REPO_ROOT / "flight_scenarios" / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import final_approach_geometry as fag  # noqa: E402
import flight_scenarios.scene_context as sc  # noqa: E402
from scene.features import SCALAR_NAMES, STATIC_NAMES, NO_LEAD_SENTINEL_S, scene_arrays  # noqa: E402
from scene_fixture import RUNWAY, T0, TARGET, axes_to_chart, chart_to_latlon, standard_scene  # noqa: E402
from trajectory_data_process.scene_index import build_scene_index  # noqa: E402


def _scene(tmp_path):
    paths, keys = standard_scene(tmp_path)
    index = build_scene_index(paths, verbose=False)
    e, n = axes_to_chart(15_000.0, 0.0)
    lat, lon = chart_to_latlon(e, n)
    return sc.scene_context(paths, index, ego_flight_key=keys["EGO1"], ego_runway=RUNWAY, ego_target=TARGET, t0_utc_s=T0,
                            ego_lat=lat, ego_lon=lon, ego_alt_hae_m=TARGET["elevation_hae_m"] + 800.0, ego_ground_speed_mps=80.0), keys


def test_on_final_membership_mirrors_the_package_gate():
    """flight_scenarios.scene_context restates the membership rule (it cannot import this
    package); the two must agree, constant for constant and verdict for verdict."""
    assert (sc.MEMBERSHIP_K, sc.MEMBERSHIP_FLOOR_M, sc.ALIGNMENT_MAX_DEG) == (fag.MEMBERSHIP_K, fag.MEMBERSHIP_FLOOR_M, fag.ALIGNMENT_MAX_DEG)
    rng = np.random.default_rng(0)
    d = rng.uniform(-500.0, 20_000.0, 500); xt = rng.uniform(-2_000.0, 2_000.0, 500); err = rng.uniform(-math.pi, math.pi, 500)
    ours = np.array([sc.on_final(float(a), float(b), float(c)) for a, b, c in zip(d, xt, err)])
    theirs = fag.hard_on_final(torch.tensor(d)[None], torch.tensor(xt)[None], torch.cos(torch.tensor(err))[None])[0].numpy()
    # The package gate has no upstream condition of its own (d > 0 is applied by its callers).
    assert np.array_equal(ours, theirs & (d > 0.0))


def test_arrays_follow_the_ego_grid_and_mask_where_the_neighbour_has_no_sample(tmp_path):
    scene, keys = _scene(tmp_path)
    arrays = scene_arrays(scene, seq_len=60, dt_s=2.0)
    assert arrays.neighbours.shape == (16, 60, 6) and arrays.neighbour_mask.shape == (16, 60)
    assert arrays.neighbour_valid.sum() == 2 and arrays.neighbour_static.shape == (16, len(STATIC_NAMES))
    assert arrays.scalars.shape == (len(SCALAR_NAMES),)
    slot = [nb.flight_key for nb in scene.neighbours].index(keys["AHEAD"])
    # A has samples over the whole window (started 100 s before t₀): the grid's last 100 s are inside.
    grid = -2.0 * np.arange(59, -1, -1)
    assert np.array_equal(arrays.neighbour_mask[slot], grid >= -100.0)
    assert not arrays.neighbours[slot][~arrays.neighbour_mask[slot]].any()
    # Its velocity on the grid is the inbound course at ~68 m/s; its last position is the observed one.
    v = arrays.neighbours[slot, -1, 3:5]
    o = scene.neighbours[slot].observed
    assert np.hypot(*v) == pytest.approx(o.ground_speed_mps, rel=0.1)
    assert np.allclose(arrays.neighbours[slot, -1, :2], [o.e_m[-1], o.n_m[-1]], atol=1.0)
    assert arrays.neighbour_static[slot, STATIC_NAMES.index("established")] == 1.0
    assert arrays.neighbour_static[slot, STATIC_NAMES.index("eta_lead_s")] == pytest.approx(scene.ego_eta_s - o.eta_s, rel=1e-5)
    # Empty slots are all zero and invalid.
    assert not arrays.neighbours[2:].any() and not arrays.neighbour_mask[2:].any() and not arrays.neighbour_valid[2:].any()
    assert arrays.scalars[SCALAR_NAMES.index("lead_gap_s")] == pytest.approx(scene.scalars.lead_gap_s, rel=1e-5)


def test_the_arrays_never_read_the_future(tmp_path):
    scene, _keys = _scene(tmp_path)
    scrambled = replace(scene, neighbours=tuple(
        replace(nb, future_label=sc.FutureLabel("not_landing", None, None)) for nb in scene.neighbours))
    a, b = scene_arrays(scene, seq_len=30, dt_s=2.0), scene_arrays(scrambled, seq_len=30, dt_s=2.0)
    assert np.array_equal(a.neighbours, b.neighbours) and np.array_equal(a.neighbour_static, b.neighbour_static)
    assert np.array_equal(a.scalars, b.scalars) and np.array_equal(a.neighbour_mask, b.neighbour_mask)
    # A scene with nobody ahead states so through the sentinel, never a NaN.
    alone = replace(scene, neighbours=(), scalars=replace(scene.scalars, lead_eta_s=None, lead_gap_s=None, ahead_by_eta=0))
    s = scene_arrays(alone, seq_len=30, dt_s=2.0).scalars
    assert s[SCALAR_NAMES.index("lead_eta_s")] == NO_LEAD_SENTINEL_S and np.isfinite(s).all()
