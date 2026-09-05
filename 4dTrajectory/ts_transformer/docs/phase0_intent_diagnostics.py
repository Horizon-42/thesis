#!/usr/bin/env python
"""Diagnostics behind the Phase 0 (truth-intent) reading — what the join point is worth.

The Phase 0 arms (``docs/experiments/scene_phase0_arms.json``) feed the TRUTH join point and
the lead's TRUE landing time to simple-v3. Their stratified ADE alone cannot say whether a
shortfall against the pre-registered gate is the information or the decoder, so five
readings sit beside it, every one on the same validation flights and the same
common-true-time metric the campaign is scored with:

  residual     per arm, on the vectored stratum: time-free path error (symmetric chamfer,
               100 m resampling, against the post-anchor OBSERVED rows — the ADE column is
               scored against the supervision rows incl. the fitted tail, a 6 s / ~400 m
               longer truth), duration error, and whether the predicted path ever
               establishes on the final — judged with the MEMBERSHIP gate
               (``hard_on_final``: the full-scale cone floored at 500 m, aligned within 30°,
               from some row to the end), NOT the k=0.5 truth gate, which the documented
               250–350 m endpoint translation saturates for every arm
  sensitivity  the intent-conditioned checkpoint re-run with the join input moved ±5 km
               along the localizer, the cross-track offset and the height above the
               glidepath of the truth join row kept (one point family; shift 0 reproduces
               the truth row exactly): does the predicted duration and path move, and does
               the path pass through the point it is told about
  template     a no-learning predictor from the truth join point: the standard trombone
               (continue the downwind, 90° base turn at the join distance plus one turn
               radius, base leg, 90° onto the final at the join), the localizer and the
               glidepath; Dubins CSC when the anchor is not on a downwind; straight to the
               threshold when already on the final. A downwind anchor already past the
               join distance turns at once and joins at its own distance — counted and
               printed. Timing either a naive speed profile (anchor ground speed falling
               linearly with distance to 70 m/s) or the truth's timing by arc length. Plus
               the decomposition: truth path + naive timing (timing-only error), truth path
               + constant speed over the truth duration
  context      the design's Phase 1 second gate in scratch form: 5-fold gradient boosting
               of the truth join distance from the ego anchor state, causal traffic counts
               from the manifests (in-TMA arrivals not yet landed at t0, time since the
               last landing, landings in the last 30 min, hour, weekday, runway), and the
               TRUTH lead ETA (all arrivals of the airport). Everything is conditioned on
               the ASSIGNED runway, itself an ATC decision.
  timing       how much duration uncertainty remains once the join distance is known

``context``/``timing`` read the raw harvest tracks: their "join before anchor" is the truth
gate opening at or before the anchor (NOT ``approach_difficulty.established_at_anchor``,
the summaries' 500 m / 30° rule) and their path length is the raw track's (NOT the
supervision path). Development scope only. Run from the repository root::

    python 4dTrajectory/ts_transformer/docs/phase0_intent_diagnostics.py residual \
        A=<baseline_pred_val> O_join_lead=<arm_pred_val> [--airport KRDU]
    python ... sensitivity --checkpoint <arm>/checkpoint.pt --summary <arm_pred_val>/summary.json
    python ... template  A=<baseline_pred_val> O_join_lead=<arm_pred_val>
    python ... context   [--airport KRDU]
    python ... timing    [--airport KRDU]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TS_DIR = HERE.parent
REPO_ROOT = TS_DIR.parents[1]
for path in (HERE, TS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import compare_frame_arms as cfa  # noqa: E402
import final_approach_geometry as fag  # noqa: E402
import intent_conditioning as ic  # noqa: E402
import geometric_metrics as gm  # noqa: E402
from config import DEFAULT_DT_S, DEFAULT_SEQ_LEN, TSConfig  # noqa: E402
from coordinate_frames import COORDINATE_FRAME_ENU  # noqa: E402
from dataset import build_series, load_flight_dicts  # noqa: E402
from flight_scenarios.identity import flight_key  # noqa: E402
from geokit import compass_bearing_to_math_enu_rad  # noqa: E402
from metrics import common_physical_time_flight_metrics  # noqa: E402
from trajectory_data_process.harvest.arrivals import load_arrival_flights  # noqa: E402

HARVEST_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"
GRAVITY = 9.80665
TEMPLATE_BANK_RAD = math.radians(25.0)
TEMPLATE_TURN_SPEED_CAP_MPS = 100.0      # the turn radius is sized at approach speed
TEMPLATE_THRESHOLD_SPEED_MPS = 70.0
# The population readings anchor raw tracks where the package anchors its windows.
ANCHOR_S = (DEFAULT_SEQ_LEN - 1) * DEFAULT_DT_S


# ── shared helpers ───────────────────────────────────────────────────────────

def _utc(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def _manifest(airport: str) -> Path:
    return HARVEST_ROOT / airport / "arrivals" / "manifest.json"


def _load_arms(specs: list[str]) -> dict[str, tuple[Path, dict[str, dict]]]:
    arms = {}
    for spec in specs:
        label, _, path = spec.partition("=")
        pred_dir = Path(path)
        pred_dir = pred_dir if pred_dir.is_absolute() else REPO_ROOT / pred_dir
        # The observed truth: this script's chamfer convention (the readouts default to
        # the closed one; the two differ by the stop-short gap, see geometric_metrics).
        arms[label] = (pred_dir, cfa.load_arm(pred_dir, geometry_truth=gm.GEOMETRY_TRUTH_OBSERVED))
    return arms


def _stratum(masks: dict[str, np.ndarray], prefix: str) -> str:
    """The strata_masks label starting with ``prefix`` (the labels carry their rule)."""
    matches = [label for label in masks if label.startswith(prefix)]
    if len(matches) != 1:
        raise KeyError(f"expected one stratum starting with {prefix!r}, got {matches}")
    return matches[0]


def _series_for(keys: list[str], airport: str, config: TSConfig):
    flights = load_flight_dicts(
        _manifest(airport), include_flight_keys={f"{airport}:{k}" for k in keys}, verbose=False
    )
    series, report = build_series(flights, config, airport=airport)
    if report.built != len(keys):
        raise RuntimeError(f"built {report.built} of {len(keys)} flights:\n{report.format()}")
    return series


def _en(rows: list[dict], lat0: float, lon0: float) -> np.ndarray:
    return np.stack(gm.chart_en([r["lat"] for r in rows], [r["lon"] for r in rows], lat0, lon0), 1)


def _axes(xy: np.ndarray, psi: float) -> tuple[torch.Tensor, torch.Tensor]:
    return fag.runway_axes(
        torch.tensor(xy[:, 0])[None], torch.tensor(xy[:, 1])[None], torch.tensor([psi])
    )


def _truth_join_distance(xy: np.ndarray, psi: float) -> float:
    """The truth gate on an observed path: the distance at which it establishes."""
    d, xt = _axes(xy, psi)
    gate = fag.truth_final_gate(d, xt, torch.ones_like(d, dtype=torch.bool))[0].numpy()
    opened = np.flatnonzero(gate)
    return float(d[0, opened[0]]) if len(opened) else math.nan


def _prediction_join_distance(xy: np.ndarray, psi: float) -> float:
    """The MEMBERSHIP gate on a predicted path (``hard_on_final`` from some row to the end):
    the distance at which the prediction claims the final, NaN if it never does."""
    d, xt = _axes(xy, psi)
    step = np.gradient(xy, axis=0)
    cos_align = fag.alignment_cosine(
        torch.tensor(step[:, 0])[None], torch.tensor(step[:, 1])[None], torch.tensor([psi])
    )
    inside = fag.hard_on_final(d, xt, cos_align)
    gate = fag.stays_mask(inside, torch.ones_like(d, dtype=torch.bool))[0].numpy()
    opened = np.flatnonzero(gate)
    return float(d[0, opened[0]]) if len(opened) else math.nan


def _p(values, q):
    return np.nanpercentile(np.asarray(values, dtype=float), q)


# ── residual ─────────────────────────────────────────────────────────────────

def cmd_residual(args: argparse.Namespace) -> None:
    arms = _load_arms(args.arms)
    reference = next(iter(arms.values()))[1]
    keys = sorted(reference)
    masks = cfa.strata_masks(reference, keys)
    vectored = _stratum(masks, "vectored")
    keys = [k for k, m in zip(keys, masks[vectored]) if m and all(k in rows for _d, rows in arms.values())]
    print(f"{vectored}, paired: n={len(keys)}")
    stats = {label: {"ade": [], "chamfer": [], "duration": [], "pred_join": []} for label in arms}
    truth_join = []
    for key in keys:
        for index, (label, (pred_dir, rows)) in enumerate(arms.items()):
            row = rows[key]
            record = json.loads((pred_dir / row["eval_file"]).read_text())
            states = json.loads((pred_dir / row["states_file"]).read_text())
            target = record["target_state"]
            lat0, lon0, psi = float(target["lat"]), float(target["lon"]), float(target["psi"])
            truth = [r for r in states["observed_states"] if r["t"] >= 0.0]
            pxy, txy = _en(states["predicted_states"], lat0, lon0), _en(truth, lat0, lon0)
            if index == 0:
                truth_join.append(_truth_join_distance(txy, psi))
            stats[label]["ade"].append(row["ade_m"])
            stats[label]["chamfer"].append(row["chamfer_m"])
            stats[label]["duration"].append(row["predicted_final_time_s"] - row["true_final_time_s"])
            stats[label]["pred_join"].append(_prediction_join_distance(pxy, psi))
    truth_join = np.array(truth_join)
    print(f"truth d_join (k=0.5 gate) km p25/p50/p75 = {np.round(_p(truth_join / 1e3, [25, 50, 75]), 1)}; "
          f"never established {int(np.isnan(truth_join).sum())}")
    print("prediction join = membership gate (full-scale cone floored 500 m, aligned 30°, to the end)")
    print(f"{'arm':14s} {'ADE':>6s} {'chamfer p50':>12s} {'chamfer mean':>13s} {'|dur err| p50':>14s} "
          f"{'dur bias p50':>13s} {'pred never claims final':>24s} {'pred join km p50':>17s} {'|Δ join| km p50':>16s}")
    for label, v in stats.items():
        dur = np.array(v["duration"])
        pred_join = np.array(v["pred_join"])
        never = int(np.isnan(pred_join).sum())
        print(f"{label:14s} {np.mean(v['ade']):6.0f} {np.median(v['chamfer']):12.0f} "
              f"{np.mean(v['chamfer']):13.0f} {np.median(np.abs(dur)):14.1f} {np.median(dur):13.1f} "
              f"{never:13d} / {len(keys):<6d} {np.nanmedian(pred_join) / 1e3:17.1f} "
              f"{np.nanmedian(np.abs(pred_join - truth_join)) / 1e3:16.2f}")
    first = next(iter(stats))
    for label in list(stats)[1:]:
        delta = np.array(stats[label]["chamfer"]) - np.array(stats[first]["chamfer"])
        ade, dur, ch = (np.array(stats[label][k]) for k in ("ade", "duration", "chamfer"))
        print(f"{label} vs {first}: chamfer better on {(delta < 0).mean():.1%} (median Δ {np.median(delta):.0f} m); "
              f"corr(ADE, |dur err|) = {np.corrcoef(ade, np.abs(dur))[0, 1]:.2f}, "
              f"corr(ADE, chamfer) = {np.corrcoef(ade, ch)[0, 1]:.2f}")


# ── sensitivity ──────────────────────────────────────────────────────────────

def cmd_sensitivity(args: argparse.Namespace) -> None:
    from forecast import forecast_approach
    from train import load_checkpoint

    model, config, normalizer, _payload = load_checkpoint(args.checkpoint)
    model = model.to("cpu").eval()
    summary = json.loads(Path(args.summary).read_text())
    vectored = [
        r for r in summary["results"]
        if r.get("route_tortuosity") is not None and r["route_tortuosity"] >= cfa.STRAIGHT_TORTUOSITY
        and not r["established_at_anchor"]
    ]
    rng = np.random.default_rng(args.seed)
    picked = [vectored[i] for i in rng.choice(len(vectored), args.flights, replace=False)]
    series = _series_for([flight_key(r, 0) for r in picked], args.airport, config)
    anchor = config.seq_len - 1
    shifts = (-args.shift_m, 0.0, args.shift_m)
    results = {shift: [] for shift in shifts}
    original = ic.truth_join_point
    try:
        for item in series:
            psi = float(item.scenario.target.psi)
            tan_gpa = math.tan(-float(item.scenario.target.gamma))
            truth_xy = item.values[anchor:, :2]
            base = original(item)
            d0, xt0 = (float(x) for x in fag.runway_axes(
                torch.tensor([[base[0]]]), torch.tensor([[base[1]]]), torch.tensor([psi])
            ))
            forecasts = {}
            for shift in shifts:
                # One point family: the truth row moved along the localizer, its cross-track
                # offset and its height above the glidepath kept (shift 0 == the truth row).
                e, n = fag.chart_from_axes(
                    torch.tensor([[d0 + shift]]), torch.tensor([[xt0]]), torch.tensor([psi])
                )
                point = np.array([float(e), float(n), base[2] + shift * tan_gpa])
                ic.truth_join_point = lambda _s, p=point: p
                with torch.no_grad():
                    forecast = forecast_approach(
                        model, item, config, normalizer, device=torch.device("cpu")
                    )
                forecasts[shift] = (forecast.values[:, :2], forecast.predicted_final_time_s, point)
            ref_xy, ref_t, _ = forecasts[0.0]
            for shift, (xy, t, point) in forecasts.items():
                nearest = float(np.min(np.hypot(xy[:, 0] - point[0], xy[:, 1] - point[1])))
                results[shift].append((t - ref_t, gm.chamfer_m(xy, ref_xy), gm.chamfer_m(xy, truth_xy), nearest))
    finally:
        ic.truth_join_point = original
    print(f"{len(series)} vectored flights, checkpoint {args.checkpoint}")
    for shift, rows_ in results.items():
        v = np.array(rows_)
        print(f"join shifted {shift:+7.0f} m: Δduration p50 {np.median(v[:, 0]):+6.1f} s | "
              f"path change vs unshifted p50 {np.median(v[:, 1]):5.0f} m | chamfer to truth p50 "
              f"{np.median(v[:, 2]):5.0f} m | closest approach to the join point p50 {np.median(v[:, 3]):5.0f} m")


# ── template ─────────────────────────────────────────────────────────────────

def _arc(centre: np.ndarray, radius: float, start: float, sweep: float, step: float = 50.0) -> np.ndarray:
    n = max(2, int(abs(radius * sweep) / step) + 1)
    return np.array([
        centre + radius * np.array([math.cos(start + k), math.sin(start + k)])
        for k in np.linspace(0.0, sweep, n)
    ])


def _dubins(p0, h0, p1, h1, radius, step: float = 50.0) -> np.ndarray:
    """Shortest turn-straight-turn path between two poses (LSL/RSR/LSR/RSL)."""
    best = None
    for s0, s1 in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        c0 = p0 + radius * np.array([-s0 * math.sin(h0), s0 * math.cos(h0)])
        c1 = p1 + radius * np.array([-s1 * math.sin(h1), s1 * math.cos(h1)])
        dc = c1 - c0
        distance = float(np.hypot(*dc))
        theta = math.atan2(dc[1], dc[0])
        if s0 == s1:
            if distance < 1e-6:
                continue
            psi = theta
        else:
            if distance < 2 * radius:
                continue
            psi = theta + s0 * math.asin(2 * radius / distance)
        a0, a1 = psi - s0 * math.pi / 2, psi - s1 * math.pi / 2
        t0 = c0 + radius * np.array([math.cos(a0), math.sin(a0)])
        t1 = c1 + radius * np.array([math.cos(a1), math.sin(a1)])
        straight = t1 - t0
        if straight @ np.array([math.cos(psi), math.sin(psi)]) < 0:
            continue
        f0, f1 = h0 - s0 * math.pi / 2, h1 - s1 * math.pi / 2
        d0 = (s0 * (a0 - f0)) % (2 * math.pi)
        d1 = (s1 * (f1 - a1)) % (2 * math.pi)
        length = radius * (d0 + d1) + float(np.hypot(*straight))
        if best is None or length < best[0]:
            best = (length, s0, s1, c0, c1, f0, a0, a1, d0, d1, t0, t1)
    if best is None:
        raise RuntimeError("no Dubins CSC path")
    _length, s0, s1, c0, c1, f0, a0, a1, d0, d1, t0, t1 = best
    points = list(_arc(c0, radius, f0, s0 * d0, step))
    n = max(2, int(np.hypot(*(t1 - t0)) / step) + 1)
    points.extend(t0 + f * (t1 - t0) for f in np.linspace(0.0, 1.0, n)[1:])
    points.extend(_arc(c1, radius, a1, s1 * d1, step)[1:])
    return np.array(points)


def _trombone(p0, psi, d0, xt0, d_join, radius, step: float = 50.0) -> tuple[np.ndarray, float]:
    """Downwind → 90° base turn → base leg → 90° turn onto the final at ``d_join``.

    Runway axes: ``d`` upstream (against the course), ``xt`` to the right of it. The
    downwind continues to ``d = d_join``; the base leg lies at ``d_join + radius``; the
    turn onto the final ends on the localizer at ``d_join`` heading for the threshold. An
    anchor already past ``d_join`` turns at once and joins at its own distance (returned).
    """
    ud = -np.array([math.cos(psi), math.sin(psi)])
    ux = np.array([math.sin(psi), -math.cos(psi)])
    side = 1.0 if xt0 > 0 else -1.0
    points = [p0.copy()]
    if d0 < d_join:
        n = max(2, int((d_join - d0) / step) + 1)
        points.extend(p0 + f * (d_join - d0) * ud for f in np.linspace(0.0, 1.0, n)[1:])
    d_turn = max(d0, d_join)
    start = points[-1]
    centre1 = start - side * radius * ux
    heading_base = -side * ux
    cross = ud[0] * heading_base[1] - ud[1] * heading_base[0]
    sweep = (math.pi / 2) * (1.0 if cross > 0 else -1.0)
    points.extend(_arc(centre1, radius, math.atan2(*(start - centre1)[::-1]), sweep, step)[1:])
    base_start = points[-1]
    base_end = (d_turn + radius) * ud + side * radius * ux
    n = max(2, int(np.hypot(*(base_end - base_start)) / step) + 1)
    points.extend(base_start + f * (base_end - base_start) for f in np.linspace(0.0, 1.0, n)[1:])
    centre2 = d_turn * ud + side * radius * ux
    heading_final = -ud
    cross = heading_base[0] * heading_final[1] - heading_base[1] * heading_final[0]
    sweep = (math.pi / 2) * (1.0 if cross > 0 else -1.0)
    points.extend(_arc(centre2, radius, math.atan2(*(base_end - centre2)[::-1]), sweep, step)[1:])
    return np.array(points), d_turn


def template_path(series, anchor: int) -> tuple[np.ndarray, float, float, str]:
    """Horizontal template ``[N, 2]`` from the anchor to the threshold, its along-path
    distance at the join, the join distance it uses, and which construction it took
    (``straight`` / ``trombone`` / ``trombone-past-join`` / ``dubins``)."""
    psi = float(series.scenario.target.psi)
    a = series.values[anchor]
    p0 = a[:2].copy()
    v0 = float(np.hypot(a[3], a[4]))
    h0 = math.atan2(a[4], a[3])
    radius = min(v0, TEMPLATE_TURN_SPEED_CAP_MPS) ** 2 / (GRAVITY * math.tan(TEMPLATE_BANK_RAD))
    join = ic.truth_join_point(series)
    d_join = float(fag.runway_axes(
        torch.tensor([[join[0]]]), torch.tensor([[join[1]]]), torch.tensor([psi])
    )[0])
    d0, xt0 = (float(x) for x in fag.runway_axes(
        torch.tensor([[p0[0]]]), torch.tensor([[p0[1]]]), torch.tensor([psi])
    ))
    cos_align = math.cos(h0 - psi)
    if (abs(xt0) < 2.5 * radius and cos_align > 0.7) or (d_join >= d0 - 500.0 and abs(xt0) < 1000.0):
        n = max(2, int(np.hypot(*p0) / 50.0) + 1)     # the origin IS the threshold (enu)
        horiz = np.array([p0 * (1.0 - f) for f in np.linspace(0.0, 1.0, n)])
        return horiz, 0.0, d0, "straight"
    if cos_align < -0.3 and abs(xt0) >= 2.0 * radius:
        horiz, d_used = _trombone(p0, psi, d0, xt0, d_join, radius)
        kind = "trombone" if d_used == d_join else "trombone-past-join"
        d_join = d_used
    else:
        e_j, n_j = fag.chart_from_axes(
            torch.tensor([[d_join]]), torch.tensor([[0.0]]), torch.tensor([psi])
        )
        horiz = _dubins(p0, h0, np.array([float(e_j), float(n_j)]), psi, radius)
        kind = "dubins"
    d_final = np.arange(d_join - 50.0, 0.0, -50.0)
    e_f, n_f = fag.chart_from_axes(
        torch.tensor(d_final)[None], torch.zeros(1, len(d_final)), torch.tensor([psi])
    )
    horiz = np.concatenate([horiz, np.stack([e_f[0].numpy(), n_f[0].numpy()], 1), np.zeros((1, 2))])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(horiz, axis=0).T))])
    return horiz, float(s[-1] - d_join), d_join, kind


def _template_record(series, anchor: int, timing: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Template as ``(offsets_s, values[N, 6], kind)`` after the anchor, with a vertical
    profile (linear to the glidepath at the join, then the glidepath — the chart origin is
    the threshold-crossing aim point, so the glidepath height is ``d·tan(GPA)`` with no
    TCH) and the chosen timing."""
    a = series.values[anchor]
    horiz, s_join, d_join, kind = template_path(series, anchor)
    tan_gpa = math.tan(-float(series.scenario.target.gamma))
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(horiz, axis=0).T))])
    length = s[-1]
    u = np.where(
        s <= s_join,
        a[2] + (s / max(s_join, 1.0)) * (d_join * tan_gpa - a[2]),
        (length - s) * tan_gpa,
    )
    if timing == "truth":
        truth_xy = np.concatenate([a[None, :2], series.supervision_values[anchor + 1:, :2]], 0)
        s_truth = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(truth_xy, axis=0).T))])
        t_truth = np.concatenate([[0.0], series.supervision_times[anchor + 1:] - series.times[anchor]])
        t = np.interp(s / length * s_truth[-1], s_truth, t_truth)
        t = np.maximum.accumulate(t + np.arange(len(t)) * 1e-6)
    else:
        v0 = float(np.hypot(a[3], a[4]))
        speed = v0 + (TEMPLATE_THRESHOLD_SPEED_MPS - v0) * s / length
        t = np.concatenate([[0.0], np.cumsum(np.diff(s) / (0.5 * (speed[1:] + speed[:-1])))])
    values = np.zeros((len(s), 6))
    values[:, 0], values[:, 1], values[:, 2] = horiz[:, 0], horiz[:, 1], u
    return t[1:], values[1:], kind


def _score(series, anchor: int, offsets: np.ndarray, values: np.ndarray) -> dict:
    truth_t = series.supervision_times[anchor + 1:] - series.times[anchor]
    return common_physical_time_flight_metrics(
        anchor_values=series.values[anchor], predicted_values=values, predicted_offsets_s=offsets,
        predicted_final_time_s=float(offsets[-1]), truth_values=series.supervision_values[anchor + 1:],
        truth_offsets_s=truth_t, true_final_time_s=float(truth_t[-1]),
    )


def cmd_template(args: argparse.Namespace) -> None:
    arms = _load_arms(args.arms)
    reference_dir, reference = next(iter(arms.values()))
    # The flights' own config (the arm's summary carries it whole), not a recipe re-derived
    # here: the population and the anchor must be the campaign's.
    config = TSConfig(**json.loads((reference_dir / "summary.json").read_text())["config"])
    if config.coordinate_frame != COORDINATE_FRAME_ENU:
        raise ValueError("the template assumes the threshold-anchored enu chart")
    keys = sorted(reference)
    masks = cfa.strata_masks(reference, keys)
    anchor = config.seq_len - 1
    for prefix in ("vectored", "straight-in"):
        stratum = _stratum(masks, prefix)
        chosen = [k for k, m in zip(keys, masks[stratum]) if m and all(k in rows for _d, rows in arms.values())]
        # compare_frame_arms keys rows by the ISO landing time; the manifest (and
        # FlightSeries.flight_id) use flight_scenarios' compact flight_key.
        by_compact = {flight_key(reference[k], 0): k for k in chosen}
        if len(by_compact) != len(chosen):
            raise RuntimeError("compact flight keys collide")
        series = _series_for(list(by_compact), args.airport, config)
        columns = {**{label: [] for label in arms}, "template + naive timing": [],
                   "template + truth timing by arc length": []}
        fde = {k: [] for k in columns}
        duration = {k: [] for k in columns}
        kinds: dict[str, int] = {}
        decomposition = {"truth path + naive timing": [], "truth path + constant speed, truth duration": []}
        for item in series:
            for label, (_dir, rows) in arms.items():
                row = rows[by_compact[item.flight_id]]
                columns[label].append(row["ade_m"])
                fde[label].append(row["fde_m"])
                duration[label].append(row["final_time_error_s"])
            for timing, label in (("naive", "template + naive timing"), ("truth", "template + truth timing by arc length")):
                t, v, kind = _template_record(item, anchor, timing)
                m = _score(item, anchor, t, v)
                columns[label].append(m["ade_m"])
                fde[label].append(m["fde_m"])
                duration[label].append(m["final_time_error_s"])
            kinds[kind] = kinds.get(kind, 0) + 1
            if prefix == "vectored":
                a = item.values[anchor]
                truth = np.concatenate([a[None, :3], item.supervision_values[anchor + 1:, :3]], 0)
                s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(truth[:, :2], axis=0).T))])
                truth_t = np.concatenate([[0.0], item.supervision_times[anchor + 1:] - item.times[anchor]])
                v0 = float(np.hypot(a[3], a[4]))
                speed = v0 + (TEMPLATE_THRESHOLD_SPEED_MPS - v0) * s / s[-1]
                naive = np.concatenate([[0.0], np.cumsum(np.diff(s) / (0.5 * (speed[1:] + speed[:-1])))])
                constant = s / s[-1] * truth_t[-1]
                values = np.zeros((len(truth), 6))
                values[:, :3] = truth
                for label, t in (("truth path + naive timing", naive), ("truth path + constant speed, truth duration", constant)):
                    decomposition[label].append(_score(item, anchor, t[1:], values[1:])["ade_m"])
        print(f"== {stratum} (n={len(series)}; template constructions {kinds}) ==")
        for label, ade in columns.items():
            print(f"  {label:40s} ADE mean {np.mean(ade):5.0f} p50 {np.median(ade):5.0f} | FDE mean "
                  f"{np.mean(fde[label]):5.0f} p50 {np.median(fde[label]):5.0f} | |dur err| p50 "
                  f"{np.median(np.abs(duration[label])):5.1f} s")
        for label, ade in decomposition.items():
            if ade:
                print(f"  {label:40s} ADE mean {np.mean(ade):5.0f} p50 {np.median(ade):5.0f}")


# ── context / timing (whole-manifest population readings) ────────────────────

CONTEXT_NAMES = ("since_last_landing_s", "airborne_same_runway", "airborne_other_runway",
                 "landings_last_30min", "hour_utc", "weekday", "runway")


def _population(airport: str) -> list[dict]:
    """Per arrival of the manifest (raw harvest track): truth join distance, whether the
    gate opened at or before the anchor, raw-track duration and path length after the
    anchor, ego anchor state, causal traffic context, and the TRUTH lead ETA."""
    manifest = json.loads(_manifest(airport).read_text())
    tracks = json.loads((_manifest(airport).parent / manifest["source_manifest"]).resolve().read_text())
    flights = load_arrival_flights(_manifest(airport))
    targets = manifest["runway_targets"]
    landings = sorted(
        (_utc(r["landing_time_utc"]), r["runway"]) for r in tracks["records"] if r["outcome"] == "assigned"
    )
    land_t = np.array([x[0] for x in landings])
    land_rw = np.array([x[1] for x in landings])
    entries = [(_utc(r["entry_time_utc"]), _utc(r["landing_time_utc"]), r["runway"]) for r in manifest["records"]]
    ent_e = np.array([x[0] for x in entries])
    ent_l = np.array([x[1] for x in entries])
    ent_rw = np.array([x[2] for x in entries])
    runways = sorted({r["runway"] for r in manifest["records"]})
    clip = ic.LEAD_ETA_CLIP_S
    rows = []
    dropped = 0
    for flight in flights:
        runway = flight["runway"]
        target = targets[runway]
        psi = compass_bearing_to_math_enu_rad(math.radians(target["course_deg"]))
        wp = np.array(flight["waypoints"])
        e, n = gm.chart_en(wp[:, 2], wp[:, 1], target["lat"], target["lon"])
        d, xt = _axes(np.stack([e, n], 1), psi)
        gate = fag.truth_final_gate(d, xt, torch.ones_like(d, dtype=torch.bool))[0].numpy()
        opened = np.flatnonzero(gate)
        ia = int(np.searchsorted(wp[:, 0], ANCHOR_S))
        if not len(opened) or ia < 1 or ia >= len(wp) - 2:
            dropped += 1
            continue
        d, xt = d[0].numpy(), xt[0].numpy()
        t0 = _utc(flight["entry_time_utc"]) + ANCHOR_S
        tl = _utc(flight["landing_time_utc"])
        de, dn = e[ia + 1] - e[ia - 1], n[ia + 1] - n[ia - 1]
        speed = math.hypot(de, dn) / (wp[ia + 1, 0] - wp[ia - 1, 0])
        heading = math.atan2(dn, de)
        same = land_rw == runway
        earlier = land_t[same][land_t[same] < t0]
        since_last = (t0 - earlier[-1]) if len(earlier) else clip
        before_own = land_t[same][land_t[same] < tl]
        lead_eta_truth = (before_own[-1] - t0) if len(before_own) else -clip
        when = datetime.fromtimestamp(t0, tz=timezone.utc)
        rows.append({
            "d_join": float(d[opened[0]]),
            "join_before_anchor": bool(opened[0] <= ia),
            "raw_duration_s": float(wp[-1, 0] - wp[ia, 0]),
            "raw_track_path_m": float(np.sum(np.hypot(np.diff(e[ia:]), np.diff(n[ia:])))),
            # Raw harvest altitude is HAE: the height above the threshold uses its HAE elevation.
            "ego": [float(d[ia]), float(xt[ia]), math.cos(heading - psi), math.sin(heading - psi), speed,
                    float(wp[ia, 3]) - target["elevation_hae_m"]],
            "context": [min(since_last, clip),
                        int(((ent_e <= t0) & (ent_l > t0) & (ent_rw == runway)).sum()) - 1,
                        int(((ent_e <= t0) & (ent_l > t0) & (ent_rw != runway)).sum()),
                        int(((land_t > t0 - 1800.0) & (land_t <= t0)).sum()),
                        when.hour + when.minute / 60.0, when.weekday(), runways.index(runway)],
            "lead": [float(np.clip(lead_eta_truth, -clip, clip))],
        })
    print(f"{airport}: {len(rows)} arrivals with an open gate and a full anchor window; "
          f"{dropped} dropped (never established, or too short)")
    return rows


def _boosting(n_columns: int, categorical: list[int]):
    from sklearn.ensemble import HistGradientBoostingRegressor

    mask = np.zeros(n_columns, dtype=bool)
    mask[categorical] = True
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, random_state=0, categorical_features=mask
    )


def _cv_r2(X: np.ndarray, y: np.ndarray, categorical: list[int]) -> tuple[float, float]:
    from sklearn.model_selection import KFold, cross_val_predict

    pred = cross_val_predict(
        _boosting(X.shape[1], categorical), X, y, cv=KFold(5, shuffle=True, random_state=0)
    )
    return 1.0 - np.mean((pred - y) ** 2) / np.var(y), float(np.median(np.abs(pred - y)))


def cmd_context(args: argparse.Namespace) -> None:
    rows = _population(args.airport)
    y = np.array([r["d_join"] for r in rows])
    before = np.array([r["join_before_anchor"] for r in rows])
    ego = np.array([r["ego"] for r in rows])
    context = np.array([r["context"] for r in rows])
    lead = np.array([r["lead"] for r in rows])
    runway_column = ego.shape[1] + CONTEXT_NAMES.index("runway")
    print(f"join before the anchor {before.mean():.1%}; d_join km p25/50/75 {np.round(_p(y / 1e3, [25, 50, 75]), 1)}")
    for name, mask in (("ALL flights", np.ones(len(y), bool)), ("join AFTER the anchor", ~before)):
        print(f"d_join, 5-fold gradient boosting, {name} (n={int(mask.sum())}):")
        baseline = np.median(np.abs(y[mask] - np.median(y[mask])))
        for label, X, categorical in (
            ("ego anchor state only", ego, []),
            ("ego + causal traffic context", np.hstack([ego, context]), [runway_column]),
            ("ego + causal context + TRUTH lead ETA", np.hstack([ego, context, lead]), [runway_column]),
        ):
            r2, err = _cv_r2(X[mask], y[mask], categorical)
            print(f"  {label:42s} R2 {r2:5.2f} | median |err| {err / 1e3:4.2f} km "
                  f"(constant baseline {baseline / 1e3:4.2f} km)")


def cmd_timing(args: argparse.Namespace) -> None:
    rows = [r for r in _population(args.airport) if not r["join_before_anchor"]]
    ego = np.array([r["ego"] for r in rows])
    d_join = np.array([[r["d_join"]] for r in rows])
    print(f"{args.airport} join after the anchor: n={len(rows)}")
    for target, unit in (("raw_duration_s", "s"), ("raw_track_path_m", "m")):
        y = np.array([r[target] for r in rows])
        baseline = np.median(np.abs(y - np.median(y)))
        for label, X in (("anchor state only", ego), ("anchor state + TRUTH d_join", np.hstack([ego, d_join]))):
            r2, err = _cv_r2(X, y, [])
            print(f"  {target:16s} from {label:28s} R2 {r2:5.2f} | median |err| {err:7.1f} {unit} "
                  f"(constant baseline {baseline:7.1f} {unit})")
    duration = np.array([r["raw_duration_s"] for r in rows])
    print(f"corr(d_join, duration) = {np.corrcoef(d_join[:, 0], duration)[0, 1]:.2f}; "
          f"corr(raw path length, duration) = {np.corrcoef([r['raw_track_path_m'] for r in rows], duration)[0, 1]:.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    residual = sub.add_parser("residual")
    residual.add_argument("arms", nargs="+", help="label=prediction_dir; the first is the reference")
    sensitivity = sub.add_parser("sensitivity")
    sensitivity.add_argument("--checkpoint", required=True)
    sensitivity.add_argument("--summary", required=True, help="the arm's prediction summary.json")
    sensitivity.add_argument("--flights", type=int, default=24)
    sensitivity.add_argument("--shift-m", type=float, default=5000.0)
    sensitivity.add_argument("--seed", type=int, default=0)
    template = sub.add_parser("template")
    template.add_argument("arms", nargs="+", help="label=prediction_dir; the first is the reference")
    for p in (residual, sensitivity, template, sub.add_parser("context"), sub.add_parser("timing")):
        p.add_argument("--airport", default="KRDU")
    args = parser.parse_args(argv)
    {"residual": cmd_residual, "sensitivity": cmd_sensitivity, "template": cmd_template,
     "context": cmd_context, "timing": cmd_timing}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
