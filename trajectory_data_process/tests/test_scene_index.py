"""The scene index: built once from the roster, cached under a contract, queried by time."""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "geokit" / "src", REPO_ROOT / "flight_scenarios" / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scene_fixture import RUNWAY, T0, standard_scene  # noqa: E402
from trajectory_data_process.scene_index import INDEX_NAME, SCENE_INDEX_SCHEMA, build_scene_index, load_scene_index  # noqa: E402


def test_index_is_built_from_the_roster_and_cached_under_its_contract(tmp_path):
    paths, keys = standard_scene(tmp_path)
    index = build_scene_index(paths, verbose=False)
    assert len(index) == 7 and index.airport == "KRDU"
    cache = json.loads((paths.tracks / INDEX_NAME).read_text())
    assert cache["schema"] == SCENE_INDEX_SCHEMA and cache["record_count"] == 7
    ego = index.entry(keys["EGO1"])
    assert ego.start_utc_s == T0 - 130.0 and ego.end_utc_s == T0 + 200.0 and ego.landing_utc_s == T0 + 200.0
    assert index.entry(keys["FAR"]).landing_utc_s is None and index.entry(keys["FAR"]).runway is None
    # A cache that matches is reused; one that does not (a changed roster) is rebuilt.
    reloaded = load_scene_index(paths, verbose=False)
    assert reloaded.manifest_sha256 == index.manifest_sha256 and len(reloaded) == 7
    manifest = json.loads(paths.manifest.read_text())
    manifest["records"] = manifest["records"][:-1]
    paths.manifest.write_text(json.dumps(manifest))
    rebuilt = load_scene_index(paths, verbose=False)
    assert len(rebuilt) == 6 and rebuilt.manifest_sha256 != index.manifest_sha256
    assert json.loads((paths.tracks / INDEX_NAME).read_text())["record_count"] == 6


def test_airborne_and_landing_queries_read_the_past_only(tmp_path):
    paths, keys = standard_scene(tmp_path)
    index = build_scene_index(paths, verbose=False)
    airborne = {e.flight_key for e in index.airborne_at(T0, 120.0)}
    # In the window: the ego, A, C, D. Not: B (starts after T0), E and F (ended before it).
    assert airborne == {keys["EGO1"], keys["AHEAD"], keys["DWIND"], keys["FAR"]}
    assert {e.flight_key for e in index.airborne_at(T0 - 290.0, 120.0)} >= {keys["LANDED"]}
    landings = index.landings_before(T0)
    assert [e.flight_key for _, _, e in landings] == [keys["OTHER"], keys["LANDED"]]
    recent = index.landings_before(T0, since_s=600.0)
    assert [(runway, e.flight_key) for _, runway, e in recent] == [(RUNWAY, keys["LANDED"])]
    # A landing at or after T0 is never "before" it.
    assert all(t <= T0 for t, _, _ in landings)
