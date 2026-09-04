"""The learned corridor is the optimizer's corridor: one geometry, checked side by side."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TS_DIR.parents[1]
OPTIMIZATION_DIR = REPO_ROOT / "4dTrajectory" / "optimization"
for path in (TS_DIR, REPO_ROOT, OPTIMIZATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import final_approach_geometry as fag  # noqa: E402
from config import (  # noqa: E402
    CORRIDOR_GATE_FAF, CORRIDOR_GATE_ON_FINAL, STATE_POSITION_CORRIDOR_BOUNDED, TSConfig,
)
from approach_constraints import lateral, segments, vertical  # noqa: E402
from approach_constraints.segments import LpvFinalSpec  # noqa: E402
from run_naming import run_display_name  # noqa: E402


def _lpv(psi: float) -> LpvFinalSpec:
    # approach_constraints works in (n, e); the GARP/FPAP sit PAST the LTP along the course.
    inbound_ne = np.array([math.sin(psi), math.cos(psi)])
    return LpvFinalSpec(
        ltp_ne=np.zeros(2),
        fpap_ne=fag.FAS.d_fpap_m * inbound_ne,
        garp_ne=fag.FAS.d_garp_m * inbound_ne,
        course_width_m=fag.FAS.course_width_m,
        tdze_m=100.0 - 17.0,
        tch_m=17.0,
        gpa_deg=3.0,
    )


def test_the_mirrored_constants_equal_the_optimizer_defaults():
    assert fag.K_MARGIN == lateral.DEFAULT_K_MARGIN
    assert fag.GLIDEPATH_BELOW_M == segments.DEFAULT_GLIDEPATH_BELOW_M
    assert fag.GLIDEPATH_ABOVE_M == segments.DEFAULT_GLIDEPATH_ABOVE_M
    assert fag.ALIGNMENT_MAX_DEG == segments.STANDARD_INTERCEPT_MAX_DEG


@pytest.mark.parametrize("psi_deg", [45.0, -135.0, 0.0, 100.0])
def test_axes_halfwidth_and_glidepath_agree_with_approach_constraints(psi_deg):
    psi = math.radians(psi_deg)
    lpv = _lpv(psi)
    rng = np.random.default_rng(int(psi_deg) + 1000)
    e = rng.uniform(-20_000, 20_000, size=200)
    n = rng.uniform(-20_000, 20_000, size=200)
    d_torch, xt_torch = fag.runway_axes(
        torch.tensor(e)[None], torch.tensor(n)[None], torch.tensor([psi], dtype=torch.float64)
    )
    d_ref = lateral.fac_distance_to_ltp(n, e, lpv)
    xt_ref = lateral.fac_cross_track(n, e, lpv)
    assert np.allclose(d_torch[0].numpy(), d_ref, atol=1e-6)
    # approach_constraints measures cross-track from the GARP→LTP axis (which runs
    # opposite to the inbound course), so its sign convention lands on the same side as
    # this chart's and the readouts': positive to the RIGHT of the inbound course.
    assert np.allclose(xt_torch[0].numpy(), xt_ref, atol=1e-6)

    upstream = d_ref >= 0.0
    hw_ref = lateral.lpv_course_halfwidth(n[upstream], e[upstream], lpv)
    hw_torch = fag.corridor_halfwidth(d_torch[0][torch.tensor(upstream)])
    assert np.allclose(hw_torch.numpy(), hw_ref, rtol=1e-9)

    gp_ref = vertical.glidepath_altitude(d_ref[upstream], lpv) - (lpv.tdze_m + lpv.tch_m)
    gp_torch = fag.glidepath_height(
        d_torch[0][torch.tensor(upstream)][None],
        torch.tensor([math.tan(math.radians(3.0))], dtype=torch.float64),
    )
    assert np.allclose(gp_torch[0].numpy(), gp_ref, rtol=1e-9)

    # The corridor and window violations reproduce the optimizer's rows: g ≤ 0 ⇔ satisfied.
    right, left = lateral.lpv_corridor_violation(n, e, lpv, k=fag.K_MARGIN)
    lat_viol, _ = fag.corridor_violations(
        d_torch, xt_torch, torch.zeros_like(d_torch),
        torch.tensor([math.tan(math.radians(3.0))], dtype=torch.float64),
    )
    outside_ref = np.maximum(right, left)[upstream]
    assert np.allclose(lat_viol[0].numpy()[upstream], np.maximum(outside_ref, 0.0), atol=1e-6)


def test_axes_round_trip_and_clamp_past_the_threshold():
    psi = torch.tensor([0.7, -2.1])
    e = torch.randn(2, 5) * 5_000
    n = torch.randn(2, 5) * 5_000
    d, xt = fag.runway_axes(e, n, psi)
    e2, n2 = fag.chart_from_axes(d, xt, psi)
    assert torch.allclose(e, e2, atol=1e-3) and torch.allclose(n, n2, atol=1e-3)
    past = torch.tensor([[-500.0, 0.0, 500.0]])
    assert torch.allclose(fag.corridor_halfwidth(past)[0, :2], torch.full((2,), fag.FAS.course_width_m))
    assert fag.glidepath_height(past, torch.tensor([0.05]))[0, 0] == 0.0


def test_bounds_saturate_inside_the_window_with_unit_slope_at_zero():
    hw = torch.tensor([[400.0]])
    x = torch.linspace(-3000, 3000, 601)[None]
    bounded = fag.bounded_cross_track(x, hw.expand_as(x))
    assert torch.all(bounded.abs() <= fag.K_MARGIN * 400.0)
    assert torch.all(torch.diff(bounded[0]) >= 0)                      # monotone
    assert torch.all(torch.diff(bounded[0, 250:351]) > 0)              # strictly, near the centre
    assert bounded[0, 300] == 0.0
    assert torch.allclose(bounded[0, 301] - bounded[0, 300], x[0, 301] - x[0, 300], rtol=1e-2)
    r = torch.linspace(-1000, 1000, 2001)[None]
    res = fag.bounded_height_residual(r)
    assert torch.all(res >= -fag.GLIDEPATH_BELOW_M) and torch.all(res <= fag.GLIDEPATH_ABOVE_M)
    assert torch.all(torch.diff(res[0]) >= 0)
    assert torch.allclose(res[0, 1001] - res[0, 999], r[0, 1001] - r[0, 999], rtol=1e-2)


def test_bound_to_final_hard_clamps_and_soft_stays_inside_and_weight_zero_is_identity():
    d = torch.tensor([[12_000.0, 6_000.0, 1_000.0]])
    xt = torch.tensor([[2_000.0, -50.0, 900.0]])
    u = torch.tensor([[900.0, 100.0, 400.0]])          # above / below / above the window
    tan_gpa = torch.tensor([math.tan(math.radians(3.0))])
    one = torch.ones_like(d)
    xt_h, u_h = fag.bound_to_final(d=d, xt=xt, u=u, weight=one, tan_gpa=tan_gpa, hard=True)
    bound = fag.K_MARGIN * fag.corridor_halfwidth(d)
    gp = fag.glidepath_height(d, tan_gpa)
    assert torch.allclose(xt_h, torch.tensor([[bound[0, 0], -50.0, bound[0, 2]]]))
    assert torch.allclose(u_h, torch.tensor([[gp[0, 0] + fag.GLIDEPATH_ABOVE_M, gp[0, 1] - fag.GLIDEPATH_BELOW_M, gp[0, 2] + fag.GLIDEPATH_ABOVE_M]]))
    xt_s, u_s = fag.bound_to_final(d=d, xt=xt, u=u, weight=one, tan_gpa=tan_gpa, hard=False)
    assert torch.all(xt_s.abs() <= bound + 1e-3)
    assert torch.all(u_s - gp >= -fag.GLIDEPATH_BELOW_M - 1e-3) and torch.all(u_s - gp <= fag.GLIDEPATH_ABOVE_M + 1e-3)
    lat, vert = fag.corridor_violations(d, xt_s, u_s, tan_gpa)
    assert torch.all(lat <= 1e-3) and torch.all(vert <= 1e-3)
    xt_0, u_0 = fag.bound_to_final(d=d, xt=xt, u=u, weight=torch.zeros_like(d), tan_gpa=tan_gpa, hard=False)
    assert torch.equal(xt_0, xt) and torch.equal(u_0, u)


def test_on_final_membership_wants_the_cone_and_the_heading():
    psi = torch.tensor([0.0])                       # course along +e
    d = torch.tensor([[10_000.0, 10_000.0, 10_000.0, 10_000.0]])
    hw = fag.corridor_halfwidth(d)
    xt = torch.tensor([[0.0, 0.0, 3.0 * hw[0, 0], 0.0]])
    v_e = torch.tensor([[70.0, -70.0, 70.0, 70.0]])
    v_n = torch.tensor([[0.0, 0.0, 0.0, 70.0]])     # 4th row: 45° off course
    cos_align = fag.alignment_cosine(v_e, v_n, psi)
    hard = fag.hard_on_final(d, xt, cos_align)
    assert hard.tolist() == [[True, False, False, False]]
    soft = fag.soft_on_final(d, xt, cos_align)
    assert soft[0, 0] > 0.98 and soft[0, 1] < 0.01 and soft[0, 2] < 0.01 and soft[0, 3] < 0.01
    assert torch.equal(
        fag.membership(CORRIDOR_GATE_ON_FINAL, d=d, xt=xt, cos_align=cos_align, d_faf=torch.tensor([float("nan")]), hard=True),
        hard,
    )
    with pytest.raises(ValueError, match="FAF distance"):
        fag.membership(CORRIDOR_GATE_FAF, d=d, xt=xt, cos_align=cos_align, d_faf=torch.tensor([float("nan")]), hard=True)
    faf = fag.membership(CORRIDOR_GATE_FAF, d=torch.tensor([[12_000.0, 8_000.0]]), xt=xt[:, :2], cos_align=cos_align[:, :2], d_faf=torch.tensor([10_000.0]), hard=True)
    assert faf.tolist() == [[False, True]]


def test_truth_gate_is_the_established_tail_beyond_the_last_300_m():
    d = torch.tensor([[20_000.0, 15_000.0, 12_000.0, 8_000.0, 4_000.0, 200.0, 0.0]])
    hw = fag.corridor_halfwidth(d)
    # inside, OUT, inside, inside, inside, (near: ignored, even if off), (near)
    xt = torch.tensor([[10.0, 2.0 * hw[0, 1], 10.0, 20.0, 5.0, 500.0, 1.0]])
    valid = torch.ones_like(d, dtype=torch.bool)
    gate = fag.truth_final_gate(d, xt, valid)
    assert gate.tolist() == [[False, False, True, True, True, False, False]]
    # A padded (invalid) suffix neither opens nor closes the gate.
    valid[0, 4:] = False
    xt[0, 4] = 5_000.0
    assert fag.truth_final_gate(d, xt, valid).tolist() == [[False, False, True, True, False, False, False]]
    assert fag.stays_mask(torch.tensor([[True, False, True]]), torch.tensor([[True, True, True]])).tolist() == [[False, False, True]]


def test_config_guards_and_naming():
    config = TSConfig(state_position_reference=STATE_POSITION_CORRIDOR_BOUNDED, seq_len=4, n_segments=3)
    assert config.uses_final_approach_context and not config.procedure_loss_active
    penalised = TSConfig(procedure_loss_lateral_weight=4.0, seq_len=4, n_segments=3)
    assert penalised.procedure_loss_active and penalised.uses_final_approach_context
    assert not TSConfig(seq_len=4, n_segments=3).uses_final_approach_context
    with pytest.raises(ValueError, match="unknown corridor_gate"):
        TSConfig(corridor_gate="never")
    with pytest.raises(ValueError, match="violation RATE"):
        TSConfig(procedure_loss_epsilon=1.5)
    with pytest.raises(ValueError, match="state output"):
        TSConfig(prediction_output="control", state_position_reference=STATE_POSITION_CORRIDOR_BOUNDED)
    with pytest.raises(ValueError, match="state output"):
        TSConfig(prediction_output="control", procedure_loss_dual_step=0.1)
    name = run_display_name(
        TSConfig(state_position_reference=STATE_POSITION_CORRIDOR_BOUNDED, corridor_gate=CORRIDOR_GATE_FAF).to_dict()
    )
    assert "pos-ref=corridor-bounded" in name and "gate=faf" in name
    assert "proc-lat" in run_display_name(penalised.to_dict())
