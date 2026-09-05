"""The closure decoder's geometry, on poses whose paths are known before any fit runs."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

TS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TS_DIR.parents[1]
for path in (TS_DIR, REPO_ROOT, REPO_ROOT / "geokit" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import closure_geometry as cg  # noqa: E402
import final_approach_geometry as fag  # noqa: E402

PSI = math.radians(50.0)      # an inbound course, math-ENU


def _anchor(d: float, xt: float, heading: float, speed: float = 75.0) -> cg.AnchorPose:
    e, n = cg.chart_from_axes_np(d, xt, PSI)
    return cg.AnchorPose.from_state([float(e), float(n), 900.0, speed * math.cos(heading), speed * math.sin(heading), -2.0], PSI)


def _heading_at(points: np.ndarray, index: int) -> float:
    step = points[index + 1] - points[index] if index + 1 < len(points) else points[index] - points[index - 1]
    return math.atan2(step[1], step[0])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _at_join(path: cg.ClosurePath) -> tuple[float, float, float]:
    """(d, xt, heading error) of the path where it reaches its join."""
    i = int(np.searchsorted(path.arc, path.s_join))
    d, xt = cg.runway_axes_np(*path.horizontal[i], PSI)
    return float(d), float(xt), abs(_wrap(_heading_at(path.horizontal, i) - PSI))


def _no_repeated_nodes(path: cg.ClosurePath) -> bool:
    return bool(np.hypot(*np.diff(path.horizontal, axis=0).T).min() > 1e-9)


def test_numpy_axes_mirror_the_torch_ones():
    e = np.array([1_000.0, -2_500.0, 300.0]); n = np.array([-400.0, 800.0, 2_000.0])
    d, xt = cg.runway_axes_np(e, n, PSI)
    d_t, xt_t = fag.runway_axes(torch.tensor(e)[None], torch.tensor(n)[None], torch.tensor([PSI]))
    assert np.allclose(d, d_t[0].numpy()) and np.allclose(xt, xt_t[0].numpy())
    e2, n2 = cg.chart_from_axes_np(d, xt, PSI)
    assert np.allclose(e2, e) and np.allclose(n2, n)


def test_anchor_pose_reads_the_state_channels():
    anchor = _anchor(12_000.0, -3_000.0, PSI + math.pi, speed=80.0)
    assert anchor.d == pytest.approx(12_000.0) and anchor.xt == pytest.approx(-3_000.0)
    assert anchor.speed_mps == pytest.approx(80.0) and _wrap(anchor.heading - (PSI + math.pi)) == pytest.approx(0.0)
    assert anchor.radius == pytest.approx(80.0 ** 2 / (cg.GRAVITY_MPS2 * math.tan(cg.BANK_RAD)))
    assert cg.turn_radius_m(150.0) == cg.turn_radius_m(cg.TURN_SPEED_CAP_MPS)
    assert anchor.pose == (anchor.position[0], anchor.position[1], anchor.heading)


def test_dubins_path_joins_the_two_poses_with_their_headings_and_fine_steps():
    p0, h0 = np.array([0.0, 0.0]), math.radians(90.0)
    p1, h1 = np.array([6_000.0, 2_000.0]), math.radians(0.0)
    path = cg.dubins_csc(p0, h0, p1, h1, radius=1_000.0)
    assert np.allclose(path[0], p0) and np.allclose(path[-1], p1, atol=1e-6)
    assert abs(_wrap(_heading_at(path, 0) - h0)) < math.radians(3.0)
    assert abs(_wrap(_heading_at(path, len(path) - 1) - h1)) < math.radians(3.0)
    steps = np.hypot(*np.diff(path, axis=0).T)
    assert steps.max() <= cg.PATH_STEP_M + 1e-6 and steps.min() > 1e-9
    # Collinear poses: the CSC path IS the straight line (zero-sweep arcs add no nodes).
    straight = cg.dubins_csc(p0, 0.0, np.array([5_000.0, 0.0]), 0.0, radius=1_000.0)
    assert np.hypot(*np.diff(straight, axis=0).T).sum() == pytest.approx(5_000.0, abs=1.0)
    assert np.hypot(*np.diff(straight, axis=0).T).min() > 1e-9
    # Identical poses: the single point; the same position with another heading is a
    # loop (a CSC path, ~2πr long), never a straight; an absent candidate costs infinity.
    assert cg.dubins_csc(p0, h0, p0, h0, radius=1_000.0).shape == (1, 2)
    loop = cg.dubins_csc(p0, h0, p0, h0 + 0.1, radius=1_000.0)
    assert np.hypot(*np.diff(loop, axis=0).T).sum() > 2 * math.pi * 1_000.0 * 0.9
    assert cg._cost(None, np.zeros((2, 2))) == math.inf


@pytest.mark.parametrize("side", (1.0, -1.0))
def test_trombone_ends_on_the_localizer_at_the_join_heading_inbound(side):
    # Outbound on the downwind, 4 km off the course on either side, 5 km out: the join at
    # 6 km is 1 km further downwind (a trombone joins at or beyond the anchor's distance).
    anchor = _anchor(5_000.0, side * 4_000.0, PSI + math.pi)
    path = cg.rule_template(anchor, PSI, d_join=6_000.0)
    assert path.kind == cg.KIND_TROMBONE and path.d_join == 6_000.0
    d, xt, heading_error = _at_join(path)
    assert d == pytest.approx(6_000.0, abs=1.0) and xt == pytest.approx(0.0, abs=1.0) and heading_error < math.radians(1.0)
    assert np.allclose(path.horizontal[-1], 0.0) and _no_repeated_nodes(path)
    assert abs(_wrap(_heading_at(path.horizontal, len(path.horizontal) - 1) - PSI)) < math.radians(1.0)
    tail = path.horizontal[path.arc > path.s_join]
    assert np.abs(cg.runway_axes_np(tail[:, 0], tail[:, 1], PSI)[1]).max() < 1e-6
    # The path never crosses the centreline before the join.
    before = path.horizontal[path.arc < path.s_join - 1.0]
    assert np.all(side * cg.runway_axes_np(before[:, 0], before[:, 1], PSI)[1] > 0.0)
    # The via is the base-turn point: 1 km down the downwind, still heading outbound.
    assert path.via is not None
    assert cg.runway_axes_np(path.via[0], path.via[1], PSI)[0] == pytest.approx(6_000.0, abs=1.0)
    assert abs(_wrap(path.via[2] - (PSI + math.pi))) < 1e-9
    # Already past the join: the trombone turns at once and reports its own join.
    past = cg.rule_template(_anchor(9_000.0, side * 4_000.0, PSI + math.pi), PSI, d_join=6_000.0)
    assert past.kind == cg.KIND_TROMBONE_PAST_JOIN and past.d_join == pytest.approx(9_000.0)


def test_rule_template_picks_straight_and_dubins_by_the_anchor_pose():
    straight = cg.rule_template(_anchor(10_000.0, 200.0, PSI), PSI, d_join=8_000.0)
    assert straight.kind == cg.KIND_STRAIGHT and np.allclose(straight.horizontal[-1], 0.0)
    # The anchor is the join: the whole path flies the glidepath, ending at 0 at the threshold.
    assert straight.s_join == 0.0 and straight.via is None
    tan_gpa = math.tan(math.radians(3.0))
    u = cg.vertical_profile(straight, anchor_u=900.0, tan_gpa=tan_gpa)
    assert u[0] == 900.0 and u[-1] == 0.0
    assert np.allclose(u[1:], (straight.length - straight.arc[1:]) * tan_gpa)
    # The second straight-in clause: about to join (d_join within 500 m of the anchor's
    # distance) and within 1 km of the centreline, whatever the heading.
    near = _anchor(10_000.0, 800.0, PSI + math.radians(60.0))
    assert cg.rule_template(near, PSI, d_join=9_800.0).kind == cg.KIND_STRAIGHT
    assert cg.rule_template(near, PSI, d_join=8_000.0).kind == cg.KIND_DUBINS
    crossing = cg.rule_template(_anchor(14_000.0, 9_000.0, PSI - math.pi / 2), PSI, d_join=8_000.0)
    assert crossing.kind == cg.KIND_DUBINS and crossing.d_join == 8_000.0 and crossing.via is None
    d, xt, heading_error = _at_join(crossing)
    assert d == pytest.approx(8_000.0, abs=1.0) and xt == pytest.approx(0.0, abs=1.0) and heading_error < math.radians(1.5)
    assert np.allclose(crossing.horizontal[-1], 0.0) and _no_repeated_nodes(crossing)


def test_dubins_join_holds_the_heading_first_and_via_dubins_passes_the_via_pose():
    anchor = _anchor(9_000.0, 4_000.0, PSI + math.pi)
    plain = cg.dubins_join(anchor, PSI, 6_000.0, 0.0)
    extended = cg.dubins_join(anchor, PSI, 6_000.0, 3_000.0)
    assert plain.kind == extended.kind == cg.KIND_DOWNWIND_DUBINS
    assert extended.length > plain.length + 2_000.0
    # The first 3 km of the extended path is the anchor heading, straight; the via is its end.
    head = extended.horizontal[extended.arc <= 3_000.0]
    assert np.allclose(np.diff(head, axis=0) / np.linalg.norm(np.diff(head, axis=0), axis=1)[:, None],
                       cg._unit(anchor.heading))
    assert np.allclose(extended.via[:2], anchor.position + 3_000.0 * cg._unit(anchor.heading))
    for path in (plain, extended):
        d, xt, heading_error = _at_join(path)
        assert d == pytest.approx(6_000.0, abs=1.0) and xt == pytest.approx(0.0, abs=1.0) and heading_error < math.radians(1.5)
        assert np.allclose(path.horizontal[-1], 0.0) and _no_repeated_nodes(path)
    via = np.array(cg.chart_from_axes_np(14_000.0, 2_000.0, PSI))
    path = cg.via_dubins(anchor, PSI, 6_000.0, via[0], via[1], PSI + math.pi / 2)
    assert path.kind == cg.KIND_VIA_DUBINS and path.via == (via[0], via[1], PSI + math.pi / 2)
    assert np.min(np.hypot(*(path.horizontal - via).T)) < 1.0
    # The label carries the via in runway axes with its heading relative to the course, wrapped.
    assert path.params["d_join"] == 6_000.0 and (path.params["via_e"], path.params["via_n"]) == (via[0], via[1])
    assert path.params["via_d"] == pytest.approx(14_000.0) and path.params["via_xt"] == pytest.approx(2_000.0)
    assert path.params["via_heading_rel"] == pytest.approx(math.pi / 2)
    wrapped = cg.via_dubins(anchor, PSI, 6_000.0, via[0], via[1], PSI + math.pi / 2 + 4 * math.pi)
    assert wrapped.params["via_heading"] == pytest.approx(PSI + math.pi / 2) and -math.pi <= wrapped.params["via_heading"] < math.pi
    assert cg.wrap_angle(3 * math.pi) == pytest.approx(-math.pi) and cg.wrap_angle(-math.pi - 0.1) == pytest.approx(math.pi - 0.1)
    d, xt, heading_error = _at_join(path)
    assert d == pytest.approx(6_000.0, abs=1.0) and xt == pytest.approx(0.0, abs=1.0) and heading_error < math.radians(1.5)


def test_vertical_profile_meets_the_glidepath_at_the_join_and_zero_at_the_threshold():
    anchor = _anchor(5_000.0, 4_000.0, PSI + math.pi)
    path = cg.rule_template(anchor, PSI, d_join=6_000.0)
    tan_gpa = math.tan(math.radians(3.0))
    u = cg.vertical_profile(path, anchor_u=900.0, tan_gpa=tan_gpa)
    assert u[0] == pytest.approx(900.0) and u[-1] == pytest.approx(0.0)
    at_join = np.searchsorted(path.arc, path.s_join)
    assert u[at_join] == pytest.approx(6_000.0 * tan_gpa, abs=tan_gpa * cg.PATH_STEP_M)
    tail = path.arc > path.s_join
    assert np.allclose(u[tail], (path.length - path.arc[tail]) * tan_gpa)


def test_timings():
    anchor = _anchor(5_000.0, 4_000.0, PSI + math.pi, speed=90.0)
    path = cg.rule_template(anchor, PSI, d_join=6_000.0)
    # Truth timing: the truth flies its own (different) path at a constant 100 m/s; the
    # template gets the truth's time at the same arc FRACTION, so its end time is the truth's.
    truth_xy = np.stack([np.linspace(anchor.position[0], 0.0, 200), np.linspace(anchor.position[1], 0.0, 200)], 1)
    truth_len = np.hypot(*(truth_xy[-1] - truth_xy[0]))
    truth_t = np.linspace(0.0, truth_len / 100.0, 200)
    t = cg.truth_timed(path, truth_xy, truth_t)
    assert t[0] == 0.0 and t[-1] == pytest.approx(truth_t[-1], abs=1e-3) and np.all(np.diff(t) > 0)
    mid = np.searchsorted(path.arc, 0.5 * path.length)
    assert t[mid] == pytest.approx(0.5 * truth_t[-1], abs=0.5)
    # Naive timing: linear deceleration from the anchor speed to the threshold speed;
    # strictly increasing even across a repeated node.
    n = cg.naive_timed(path, 90.0)
    assert n[0] == 0.0 and np.all(np.diff(n) > 0)
    assert n[-1] == pytest.approx(path.length / (0.5 * (90.0 + cg.THRESHOLD_SPEED_MPS)), rel=0.05)
    repeated = np.array([0.0, 100.0, 100.0, 200.0])
    assert np.all(np.diff(cg.naive_times(repeated, 90.0)) > 0)
    offsets, values = cg.path_record(path, n, anchor_u=900.0, tan_gpa=math.tan(math.radians(3.0)))
    assert offsets.shape == (len(path.horizontal) - 1,) and values.shape == (len(path.horizontal) - 1, 6)
    assert np.allclose(values[-1, :3], 0.0)


def test_a_join_at_the_threshold_still_yields_a_valid_record():
    """A truth that never establishes reports the target as its join (d_join = 0): the
    path must not repeat the origin, and its naive clock must stay strictly increasing."""
    anchor = _anchor(14_000.0, 9_000.0, PSI - math.pi / 2)
    path = cg.rule_template(anchor, PSI, d_join=0.0)
    assert np.allclose(path.horizontal[-1], 0.0) and _no_repeated_nodes(path)
    assert np.all(np.diff(cg.naive_timed(path, 75.0)) > 0)


def test_localizer_entry_is_where_the_path_settles_on_the_final():
    anchor = _anchor(6_000.0, 8_000.0, PSI + math.pi)
    trombone = cg.rule_template(anchor, PSI, d_join=7_500.0)
    assert cg.localizer_entry(trombone, PSI)[1] == pytest.approx(7_500.0, abs=1.0)
    # A straight path from off the centreline never settles on the localizer: no entry,
    # and the F1 fit keeps the straight path it found instead of relabelling it away.
    off = _anchor(10_000.0, 800.0, PSI + math.radians(60.0))
    straight = cg.rule_template(off, PSI, d_join=9_800.0)
    assert straight.kind == cg.KIND_STRAIGHT and cg.localizer_entry(straight, PSI) is None
    f1 = cg.fit_rule_template(off, PSI, straight.horizontal, d_join0=9_800.0)
    assert f1.kind == cg.KIND_STRAIGHT and cg.path_error_m(f1.horizontal, straight.horizontal) < 1.0
    # An anchor already on the localizer 14 km out, asked to "join" at 6 km: the CSC is
    # the localizer itself, so the identifiable join is the anchor's own distance and
    # the path joining there is the same path.
    aligned = _anchor(14_000.0, 0.0, PSI)
    far = cg.dubins_join(aligned, PSI, 6_000.0)
    entry = cg.localizer_entry(far, PSI)[1]
    assert entry == pytest.approx(14_000.0, abs=1.0)
    assert cg.path_error_m(cg.dubins_join(aligned, PSI, entry).horizontal, far.horizontal) < 1.0


@pytest.mark.parametrize("side", (1.0, -1.0))
@pytest.mark.parametrize("xt", (5_000.0, 8_000.0, 14_000.0))
def test_fits_recover_a_trombone_the_family_generated_with_nested_residuals(side, xt):
    anchor = _anchor(6_000.0, side * xt, PSI + math.pi)
    truth = cg.rule_template(anchor, PSI, d_join=7_500.0)
    assert truth.kind == cg.KIND_TROMBONE
    f1 = cg.fit_rule_template(anchor, PSI, truth.horizontal, d_join0=5_000.0)
    assert f1.d_join == pytest.approx(7_500.0, abs=1.0)
    e1 = cg.path_error_m(f1.horizontal, truth.horizontal)
    f2 = cg.fit_dubins_join(anchor, PSI, truth.horizontal, d_join0=5_000.0, seed=f1)
    e2 = cg.path_error_m(f2.horizontal, truth.horizontal)
    f3, spread = cg.fit_via_dubins(anchor, PSI, truth.horizontal, d_join0=5_000.0, seeds=(f1, f2))
    e3 = cg.path_error_m(f3.horizontal, truth.horizontal)
    # Each family is seeded with the previous one's solution, so the residuals nest; the
    # F3 label is the canonical (earliest reproducing) via, within its tolerance.
    assert e1 < 1.0 and e2 <= e1 + 1e-6 and f3.params["fit_error_m"] <= e2 + 1e-6
    assert e3 < cg.CANONICAL_VIA_TOLERANCE_M + 1.0
    assert f2.params["d_join"] == pytest.approx(7_500.0, abs=1.0) and f2.params["d_downwind"] == pytest.approx(1_500.0, abs=10.0)
    # The canonical via of a trombone is its base-turn point: 1.5 km along the downwind.
    assert f3.params["d_join"] == pytest.approx(7_500.0, abs=1.0) and f3.params["canonical"] is True
    assert np.hypot(f3.via[0] - truth.via[0], f3.via[1] - truth.via[1]) < 60.0
    assert 0.0 < f3.params["via_fraction"] < 0.25
    assert spread == 0.0 or math.isnan(spread)
    for fitted in (f1, f2, f3):
        assert np.allclose(fitted.horizontal[-1], 0.0) and fitted.params["d_join"] >= cg.D_JOIN_MIN_M


def test_fit_via_on_a_single_csc_truth_labels_the_anchor_as_the_via():
    """Any pose on a single CSC path reproduces it; the canonical label is the earliest —
    the anchor itself, via fraction 0 — and the join is the localizer entry."""
    anchor = _anchor(14_000.0, 9_000.0, PSI - math.pi / 2)
    truth = cg.rule_template(anchor, PSI, d_join=8_000.0)
    assert truth.kind == cg.KIND_DUBINS
    f2 = cg.fit_dubins_join(anchor, PSI, truth.horizontal, d_join0=6_000.0)
    f3, _spread = cg.fit_via_dubins(anchor, PSI, truth.horizontal, d_join0=6_000.0, seeds=(f2,))
    assert cg.path_error_m(f3.horizontal, truth.horizontal) < 1.0
    assert f3.params["d_join"] == pytest.approx(8_000.0, abs=1.0) and f3.params["via_fraction"] == 0.0
    assert np.allclose(f3.via[:2], anchor.position)


def test_fits_on_a_truth_outside_every_family_still_nest():
    """A noisy, lengthened trombone: no family reproduces it, the residuals still nest and
    F1 never loses to the unfitted rule template at the truth gate's join."""
    rng = np.random.default_rng(3)
    anchor = _anchor(6_000.0, 9_000.0, PSI + math.pi)
    base = cg.rule_template(anchor, PSI, d_join=8_000.0).horizontal
    truth = base + np.cumsum(rng.normal(scale=15.0, size=base.shape), axis=0)
    truth[-1] = 0.0
    f0 = cg.rule_template(anchor, PSI, d_join=7_000.0)
    f1 = cg.fit_rule_template(anchor, PSI, truth, d_join0=7_000.0)
    f2 = cg.fit_dubins_join(anchor, PSI, truth, d_join0=7_000.0, seed=f1)
    f3, _ = cg.fit_via_dubins(anchor, PSI, truth, d_join0=7_000.0, seeds=(f1, f2))
    e0, e1, e2 = (cg.path_error_m(p.horizontal, truth) for p in (f0, f1, f2))
    assert e1 <= e0 + 1e-6 and e2 <= e1 + 1e-6 and f3.params["fit_error_m"] <= e2 + 1e-6
    assert e1 > 50.0                                  # genuinely outside the family
