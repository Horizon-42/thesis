"""The separation rules the optimizer reads from ts_transformer, pinned on fixed scenes (design §6.1 rules 3, 4).

A change of the rules on the two-tier side fails here, in the optimizer's suite, instead of moving the
optimizer's results silently. The expected values are the FAA minima (7110.65BB), in metres.
"""

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from geokit import FT_M, NM_M, METRES_PER_DEG_LAT

from traffic import rules

TRAFFIC_DIR = Path(__file__).resolve().parents[1]
# Two parallel runways 3,000 ft (914 m) apart, course 090: dependent parallels (5-9-6 a2, 1.0 NM diagonal).
TARGETS = {
    "09L": {"lat": 35.0 + 457.2 / METRES_PER_DEG_LAT, "lon": -78.0, "course_deg": 90.0},
    "09R": {"lat": 35.0 - 457.2 / METRES_PER_DEG_LAT, "lon": -78.0, "course_deg": 90.0},
}
SEPARATION = rules.separation(TARGETS, 70.0)


def _scene(n, e, h, runway, established, category, *, along=None, right=None, off=None):
    count = len(n)
    return rules.Scene(
        e_m=np.array(e, float), n_m=np.array(n, float), height_m=np.array(h, float), runway=tuple(runway),
        along_m=np.array(along if along is not None else e, float),       # course 090: along the landing direction is east
        track_minus_course_deg=np.array(off if off is not None else [0.0] * count, float),
        right_of_course_m=np.array(right if right is not None else [0.0] * count, float),
        established=np.array(established, bool), category=tuple(category),
    )


def test_in_trail_on_one_final_is_the_radar_or_the_wake_minimum_and_the_follower_answers():
    # both established on 09L, 2.5 NM apart: under the 3 NM radar minimum; the one behind answers
    e = [-5000.0, -5000.0 - 2.5 * NM_M]
    found = rules.judge(_scene([0, 0], e, [500, 600], ["09L", "09L"], [True, True], ["F", "F"]),
                        SEPARATION, rules.VISUAL, [])
    assert [(f.kind, f.responsible) for f in found] == [(rules.IN_TRAIL, (1,))]
    assert found[0].required_m == pytest.approx(3.0 * NM_M)
    # a heavy (B) ahead of a light (F): the directly-behind wake minimum, 5 NM (TBL 5-5-1)
    found = rules.judge(_scene([0, 0], [-5000.0, -5000.0 - 4.0 * NM_M], [500, 600], ["09L", "09L"],
                               [True, True], ["B", "F"]), SEPARATION, rules.VISUAL, [])
    assert found[0].required_m == pytest.approx(5.0 * NM_M)


def test_not_established_needs_the_radar_minimum_or_1000_ft():
    scene = _scene([0, 0], [-20000.0, -20000.0 - 2.0 * NM_M], [1500, 1500 + 800 * FT_M], ["09L", "09L"],
                   [False, False], ["F", "F"])
    found = rules.judge(scene, SEPARATION, rules.VISUAL, [])
    assert [(f.kind, f.responsible) for f in found] == [(rules.RADAR_OR_VERTICAL, (0, 1))]
    apart = _scene([0, 0], [-20000.0, -20000.0 - 2.0 * NM_M], [1500, 1500 + 1100 * FT_M], ["09L", "09L"],
                   [False, False], ["F", "F"])
    assert rules.judge(apart, SEPARATION, rules.VISUAL, []) == []


def test_dependent_parallels_are_free_under_visual_and_diagonal_under_ifr():
    # both established, on their own finals, 0.5 NM along-track apart, same height
    scene = _scene([457.2, -457.2], [-6000.0, -6000.0 - 0.5 * NM_M], [600, 600], ["09L", "09R"],
                   [True, True], ["F", "F"], right=[0.0, 0.0])
    assert rules.judge(scene, SEPARATION, rules.VISUAL, []) == []
    found = rules.judge(scene, SEPARATION, rules.IFR, [])
    assert [f.kind for f in found] == [rules.DIAGONAL]
    assert found[0].required_m == pytest.approx(1.0 * NM_M)


def test_crossing_finals_both_established_are_judged_only_under_ifr():
    targets = {"09": {"lat": 35.0, "lon": -78.0, "course_deg": 90.0},
               "36": {"lat": 35.0 - 3000.0 / METRES_PER_DEG_LAT, "lon": -78.0 + 0.02, "course_deg": 0.0}}
    crossing = rules.separation(targets, 70.0)
    scene = _scene([0, -3500], [-3000.0, -1200.0], [400, 420], ["09", "36"], [True, True], ["F", "F"],
                   along=[0.0, 0.0])
    assert rules.judge(scene, crossing, rules.VISUAL, []) == []
    assert [f.kind for f in rules.judge(scene, crossing, rules.IFR, [])] == [rules.RADAR_OR_VERTICAL]


def test_the_wake_at_the_threshold_behind_a_leader_over_it():
    # a B over the 09L threshold, an F established 4 NM behind: TBL 5-5-2 asks 5 NM
    scene = _scene([0, 0], [0.0, -4.0 * NM_M], [120, 500], ["09L", "09L"], [True, True], ["B", "F"],
                   along=[SEPARATION.along_nm["09L"] * NM_M, SEPARATION.along_nm["09L"] * NM_M - 4.0 * NM_M])
    found = [f for f in rules.judge(scene, SEPARATION, rules.VISUAL, [0]) if f.kind == rules.AT_THRESHOLD]
    assert len(found) == 1 and found[0].responsible == (1,)
    assert found[0].required_m == pytest.approx(5.0 * NM_M)


def test_category_reads_none_for_no_type_and_for_an_unlisted_type():
    assert rules.category(None) is None
    assert rules.category("NOT-A-TYPE") is None
    assert rules.category("A320") in set("ABCDEFGHI")


def test_only_the_adapter_imports_ts_transformer_and_nothing_imports_torch():
    for path in sorted(TRAFFIC_DIR.glob("*.py")):
        imported = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported.add(node.module)
        ts = {m for m in imported if m.split(".")[0] == "ts_transformer"}
        assert not {m for m in imported if m.split(".")[0] == "torch"}, path.name
        if path.name == "rules.py":
            assert ts == {"ts_transformer.inference.runway_schedule", "ts_transformer.inference.separation"}
        else:
            assert not ts, path.name


def test_importing_the_adapter_does_not_load_torch():
    code = ("import sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]; import traffic.rules; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code, str(TRAFFIC_DIR.parent), str(TRAFFIC_DIR.parents[2])],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
