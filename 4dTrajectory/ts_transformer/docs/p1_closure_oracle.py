#!/usr/bin/env python
"""P1 oracle studies for the closure decoder (scene design §五 P1.a / P1.b).

``geometry`` (P1.a) — how well each closed-form path FAMILY (``closure_geometry``) can
reproduce the truth path when its parameters are fitted to that truth, scored with the
geometry-only timing (the truth's time at the same arc fraction). Per stratum and
family: chamfer / Fréchet against the truth, ADE on the package's common-true-time grid,
the predicted/true length ratio, and the share of flights whose fit stays above the
fallback residual. F0 (the rule template at the truth join) reproduces Phase 0's
"template + truth timing" 1688 m; the fitted families say how much of that is the join
distance (F1), a downwind decision (F2) or a free via pose (F3). F3 is seeded with F1's
and F2's solutions, so its FITTED residual never exceeds theirs beyond re-expressing them;
the labels are then canonicalised (``closure_geometry``: join at the localizer entry,
F3's via the earliest reproducing pose), which costs residual on a looping fit — the
readout prints the share of flights on which each nesting fails, and how identifiable
F3's via is (the distance between the canonical vias of its two best starts when both are
within 10 %). Straight-in flights (the rule
template's ``straight`` branch) are not fitted — every family uses that branch — and the
``fitted`` column counts the flights that were. Every fit is written to
``closure_labels.json`` (per flight_key, per family: kind, parameters, residuals) — the
closure decoder's labels and P1.d's fallback criterion.

``labels`` (P1.c-1) — the closure decoder's labels for EVERY rostered flight of the
airport (training and validation alike; straight-in flights fitted too): the canonical F3
geometry, the K=4 / K=8 profiles with their residuals, ``valid`` (canonical and within the
residual cap — the flights the regression loss should use), and the difficulty
covariates. One JSON file, keyed by the compact flight key ``FlightSeries.flight_id``.

``speed`` (P1.b) — on the TRUTH path (geometry error zero), the ADE each speed / height
profile parametrisation (``closure_profile``) can reach: the naive profile (Phase 0's
1308 m), the naive shape stretched onto the truth duration, slowness knots fitted by
least squares (K = 2 / 4 / 8 / 16), the geometry's closure height profile against
height knots, and the two fitted together. Labels at ``LABEL_KNOTS`` go to
``profile_labels.json``.

Truth for both = the post-anchor SUPERVISION rows (observed rows plus the fitted tail to
the threshold): the truth the ADE grid scores against, as in Phase 0's template study.
It is NOT the readouts' ``closed`` truth (observed rows closed by one straight node —
they differ by that tail, ~380 m / 6 s at KRDU); the P1.c campaign gate is read by the
readouts on ``closed``. P1.a gate (§五): vectored chamfer p50 < 500 m AND Fréchet p50 <
1.5 km (order-preserving — chamfer alone lets detours through) AND truth-timed ADE mean
< 1.0 km. Development scope only; run from the repository root::

    python 4dTrajectory/ts_transformer/docs/p1_closure_oracle.py geometry --airport KRDU \\
        --reference 4dTrajectory/outputs/KRDU/experiments/control_procedure_20260905/A_control_v3_pred_val \\
        --out 4dTrajectory/outputs/KRDU/experiments/closure_p1_20260905 [--limit N] [--workers 8]
    python 4dTrajectory/ts_transformer/docs/p1_closure_oracle.py speed ... (same arguments)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
TS_DIR = HERE.parent
REPO_ROOT = TS_DIR.parents[1]
for path in (HERE, TS_DIR, REPO_ROOT, REPO_ROOT / "geokit" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import closure_geometry as cg  # noqa: E402
import closure_profile as cp  # noqa: E402
import compare_frame_arms as cfa  # noqa: E402
import geometric_metrics as gm  # noqa: E402
import intent_conditioning as ic  # noqa: E402
import phase0_intent_diagnostics as pid  # noqa: E402
from approach_difficulty import approach_difficulty  # noqa: E402
from config import TSConfig  # noqa: E402
from coordinate_frames import COORDINATE_FRAME_ENU  # noqa: E402
from dataset import build_series, load_flight_dicts  # noqa: E402
from flight_scenarios.identity import flight_key  # noqa: E402
from metrics import common_physical_time_flight_metrics  # noqa: E402

FAMILIES = ("F0 rule@truth join", "F1 rule, d_join fitted", "F2 downwind+Dubins fitted", "F3 via-pose Dubins fitted")
FALLBACK_CHAMFER_M = 1_000.0     # placeholder for P1.d's criterion; flags 0.6 % of F3 fits at KRDU
GATE = {"chamfer_p50_m": 500.0, "frechet_p50_m": 1_500.0, "ade_mean_m": 1_000.0}
MIN_FLIGHTS_FOR_DISTRIBUTION = 20


def _truth(series, anchor: int):
    """Anchor state, truth horizontal path and times (anchor first, ``t[0] == 0``)."""
    a = series.values[anchor]
    xy = np.concatenate([a[None, :2], series.supervision_values[anchor + 1:, :2]], 0)
    t = np.concatenate([[0.0], series.supervision_times[anchor + 1:] - series.times[anchor]])
    return a, xy, t


def _score(series, anchor: int, offsets: np.ndarray, values: np.ndarray, truth_t: np.ndarray) -> dict:
    m = common_physical_time_flight_metrics(
        anchor_values=series.values[anchor], predicted_values=values, predicted_offsets_s=offsets,
        predicted_final_time_s=float(offsets[-1]), truth_values=series.supervision_values[anchor + 1:],
        truth_offsets_s=truth_t[1:], true_final_time_s=float(truth_t[-1]),
    )
    return {"ade_m": m["ade_m"], "duration_error_s": float(offsets[-1] - truth_t[-1])}


def _residuals(path: cg.ClosurePath, truth_xy: np.ndarray) -> dict:
    return {"kind": path.kind, "params": path.params, "d_join_m": path.d_join,
            "error_m": cg.path_error_m(path.horizontal, truth_xy),
            "chamfer_m": gm.chamfer_m(path.horizontal, truth_xy),
            "frechet_m": gm.discrete_frechet_m(path.horizontal, truth_xy),
            "length_ratio": path.length / float(gm.cumulative_arc_m(truth_xy)[-1])}


def fit_flight(item) -> dict:
    """Every family on one flight: kind, parameters, residuals, and the truth-timed ADE
    (plus F0 with the naive timing, Phase 0's other column)."""
    series, anchor = item
    psi = float(series.scenario.target.psi)
    tan_gpa = math.tan(-float(series.scenario.target.gamma))
    a, truth_xy, truth_t = _truth(series, anchor)
    anchor_pose = cg.AnchorPose.from_state(a, psi)
    join = ic.truth_join_point(series)
    d_join0 = float(cg.runway_axes_np(join[0], join[1], psi)[0])
    f0 = cg.rule_template(anchor_pose, psi, d_join0)
    fitted = f0.kind != cg.KIND_STRAIGHT
    if fitted:
        f1 = cg.fit_rule_template(anchor_pose, psi, truth_xy, d_join0)
        f2 = cg.fit_dubins_join(anchor_pose, psi, truth_xy, d_join0, seed=f1)
        f3, spread = cg.fit_via_dubins(anchor_pose, psi, truth_xy, d_join0, seeds=(f1, f2))
        paths = dict(zip(FAMILIES, (f0, f1, f2, f3)))
    else:
        paths, spread = {name: f0 for name in FAMILIES}, math.nan
    out = {"flight_id": series.flight_id, "fitted": fitted, "d_join_truth_m": d_join0,
           "anchor_d_m": anchor_pose.d, "anchor_xt_m": anchor_pose.xt, "families": {}}
    scored: dict[int, dict] = {}
    for name, path in paths.items():
        if id(path) not in scored:
            offsets, values = cg.path_record(path, cg.truth_timed(path, truth_xy, truth_t), float(a[2]), tan_gpa)
            scored[id(path)] = {**_residuals(path, truth_xy),
                                "ade_truth_timing_m": _score(series, anchor, offsets, values, truth_t)["ade_m"]}
        out["families"][name] = dict(scored[id(path)])
    out["families"][FAMILIES[3]]["via_label_spread_m"] = spread
    offsets, values = cg.path_record(f0, cg.naive_timed(f0, anchor_pose.speed_mps), float(a[2]), tan_gpa)
    out["families"][FAMILIES[0]]["ade_naive_timing_m"] = _score(series, anchor, offsets, values, truth_t)["ade_m"]
    return out


def _p(values, q):
    return float(np.nanpercentile(np.asarray(values, dtype=float), q))


def _reference_rows(pred_dir: Path) -> tuple[dict[str, dict], dict]:
    """The reference arm's scored flights from its summary alone (the strata need only
    the difficulty covariates; no per-flight record is read)."""
    summary = json.loads((pred_dir / "summary.json").read_text())
    rows = {}
    for row in summary["results"]:
        if row.get("ade_m") is not None and row.get("route_tortuosity") is not None:
            rows[cfa.flight_key(row)] = row
    return rows, summary["config"]


def _flights(args: argparse.Namespace):
    """The reference arm's flights, config, anchor, strata, and the series to score."""
    reference_dir = args.reference if args.reference.is_absolute() else REPO_ROOT / args.reference
    reference, config_dict = _reference_rows(reference_dir)
    config = TSConfig(**config_dict)
    if config.coordinate_frame != COORDINATE_FRAME_ENU:
        raise ValueError("the closure geometry assumes the threshold-anchored enu chart")
    anchor = config.seq_len - 1
    keys = sorted(reference)
    masks = cfa.strata_masks(reference, keys)
    by_compact = {flight_key(reference[k], 0): k for k in keys}
    if len(by_compact) != len(keys):
        raise RuntimeError("compact flight keys collide")
    chosen = list(by_compact)
    if args.limit:
        rng = np.random.default_rng(0)
        chosen = sorted(str(k) for k in rng.choice(chosen, size=min(args.limit, len(chosen)), replace=False))
        print(f"limited to {len(chosen)} flights (seed 0)")
    series = pid._series_for(chosen, args.airport, config)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    return reference_dir, config, anchor, keys, masks, by_compact, series, out


def _stratum_lines(masks, keys, by_key):
    for stratum, mask in masks.items():
        selected = [k for k, m in zip(keys, mask) if m and k in by_key]
        if selected:
            yield stratum, selected


def cmd_geometry(args: argparse.Namespace) -> None:
    reference_dir, config, anchor, keys, masks, by_compact, series, out = _flights(args)
    print(f"{args.airport}: {len(series)} flights, anchor index {anchor}, fitting {len(FAMILIES)} families "
          f"with {args.workers} workers")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(fit_flight, [(item, anchor) for item in series], chunksize=4))
    by_key = {by_compact[r["flight_id"]]: r for r in results}
    (out / "closure_labels.json").write_text(json.dumps(
        {"airport": args.airport, "reference": str(reference_dir), "anchor_index": anchor,
         "truth": "post-anchor supervision rows", "objective": "closure_geometry.path_error_m",
         "fallback_chamfer_m": FALLBACK_CHAMFER_M, "families": FAMILIES, "flights": by_key}, indent=1))
    lines = [f"# {args.airport} closure geometry oracle: {len(by_key)} flights; truth = post-anchor SUPERVISION rows "
             f"(not the readouts' closed truth); timing = truth time at the same arc fraction; fits minimise "
             f"the arc-aligned horizontal error; straight-in flights are not fitted (every family flies the "
             f"straight branch); fallback residual {FALLBACK_CHAMFER_M:.0f} m is a placeholder.",
             f"# gate (vectored): chamfer p50 < {GATE['chamfer_p50_m']:.0f} m AND Fréchet p50 < "
             f"{GATE['frechet_p50_m']:.0f} m AND truth-timed ADE mean < {GATE['ade_mean_m']:.0f} m"]
    for stratum, selected in _stratum_lines(masks, keys, by_key):
        kinds: dict[str, int] = {}
        for k in selected:
            kind = by_key[k]["families"][FAMILIES[0]]["kind"]
            kinds[kind] = kinds.get(kind, 0) + 1
        fitted = [k for k in selected if by_key[k]["fitted"]]
        lines.append(f"\n## {stratum} (n={len(selected)}; fitted {len(fitted)}; F0 constructions {kinds})\n")
        lines.append("| family | chamfer p50 / mean | Fréchet p50 | arc-aligned error p50 | ADE truth-timed mean / p50 | "
                     "len ratio p50 | above fallback | gate |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for name in FAMILIES:
            rows = [by_key[k]["families"][name] for k in selected]
            chamfer = np.array([r["chamfer_m"] for r in rows])
            frechet_p50 = _p([r["frechet_m"] for r in rows], 50)
            ade = np.array([r["ade_truth_timing_m"] for r in rows])
            passes = (np.median(chamfer) < GATE["chamfer_p50_m"] and frechet_p50 < GATE["frechet_p50_m"]
                      and ade.mean() < GATE["ade_mean_m"])
            lines.append(
                f"| {name} | {np.median(chamfer):.0f} / {chamfer.mean():.0f} | {frechet_p50:.0f} | "
                f"{_p([r['error_m'] for r in rows], 50):.0f} | {ade.mean():.0f} / {np.median(ade):.0f} | "
                f"{_p([r['length_ratio'] for r in rows], 50):.2f} | {(chamfer > FALLBACK_CHAMFER_M).mean():.1%} | "
                f"{'pass' if passes else 'fail'} |")
        naive = [by_key[k]["families"][FAMILIES[0]]["ade_naive_timing_m"] for k in selected]
        lines.append(f"\nF0 + naive timing (Phase 0's other column): ADE mean {np.mean(naive):.0f} / p50 {np.median(naive):.0f}")
        if len(fitted) >= MIN_FLIGHTS_FOR_DISTRIBUTION:
            for name in FAMILIES[1:]:
                deltas = np.array([by_key[k]["families"][name]["d_join_m"] - by_key[k]["d_join_truth_m"] for k in fitted])
                lines.append(f"{name}: fitted d_join − truth-gate d_join, km p25/p50/p75 = "
                             f"{np.round(np.percentile(deltas / 1e3, [25, 50, 75]), 2)}")
            f3 = [by_key[k]["families"][FAMILIES[3]] for k in fitted]
            spread = np.array([r["via_label_spread_m"] for r in f3])
            fit_error = np.array([r["params"]["fit_error_m"] for r in f3])
            error = np.array([r["error_m"] for r in f3])
            for earlier in FAMILIES[1:3]:
                excess = np.array([by_key[k]["families"][FAMILIES[3]]["error_m"] - by_key[k]["families"][earlier]["error_m"] for k in fitted])
                worse = excess > 10.0
                lines.append(f"F3 worse than {earlier.split(' ')[0]} by > 10 m on {worse.mean():.1%} of fitted flights"
                             + (f" (excess p50 {np.median(excess[worse]):.0f} m, max {excess.max():.0f} m)" if worse.any() else ""))
            f2_vs_f1 = np.array([by_key[k]["families"][FAMILIES[2]]["error_m"] - by_key[k]["families"][FAMILIES[1]]["error_m"] for k in fitted])
            lines.append(f"F2 worse than F1 by > 10 m on {(f2_vs_f1 > 10.0).mean():.1%} (F2's downwind runs along the "
                         f"anchor heading; it contains a trombone only when the anchor already points down the reciprocal course)")
            canonical = np.array([r["params"]["canonical"] for r in f3])
            lines.append(f"F3 labels not canonical (the fit loops; fitted via kept, not identifiable): "
                         f"{(~canonical).sum()} of {len(fitted)} ({(~canonical).mean():.1%})")
            lines.append(f"F3 labels ({len(fitted)} fitted): via fraction of the pre-final path p25/p50/p75 = "
                         f"{np.round(np.percentile([r['params']['via_fraction'] for r in f3], [25, 50, 75]), 2)}; "
                         f"via label spread (two best starts within 10 %): defined on {np.isfinite(spread).mean():.0%}, "
                         f"p50 {np.nanmedian(spread) if np.isfinite(spread).any() else math.nan:.0f} m, "
                         f"p90 {np.nanpercentile(spread, 90) if np.isfinite(spread).any() else math.nan:.0f} m; "
                         f"canonicalisation cost (labelled − fitted arc-aligned error) p50 {np.median(error - fit_error):.0f} m, "
                         f"p90 {np.percentile(error - fit_error, 90):.0f} m")
    text = "\n".join(lines)
    print(text)
    (out / "oracle_geometry.txt").write_text(text + "\n")
    print(f"\nwrote {out / 'oracle_geometry.txt'} and closure_labels.json")


# ── P1.b: speed / height profiles on the truth path ──────────────────────────

SLOWNESS_KNOTS = (2, 4, 8, 16)
HEIGHT_KNOTS = (2, 4, 8)
LABEL_KNOTS = (4, 8)


def _timing_variants(f, length, truth_t, speed0):
    """Time profiles on the truth path: name → times (``t[0] == 0``)."""
    naive = cg.naive_times(f * length, speed0)
    variants = {"naive (anchor speed → 70 m/s)": naive,
                "naive shape × truth duration": cp.scale_to_duration(naive, float(truth_t[-1]))}
    for k in SLOWNESS_KNOTS:
        variants[f"slowness knots K={k} (LSQ)"] = cp.times_from_slowness(f, length, cp.fit_slowness_knots(f, length, truth_t, k))
    return variants


def _height_variants(f, truth_u, s, s_join, d_join, tan_gpa, anchor_u):
    length = s[-1]
    closure = np.where(s <= s_join, anchor_u + (s / max(s_join, 1.0)) * (d_join * tan_gpa - anchor_u), (length - s) * tan_gpa)
    variants = {"closure (linear to glidepath at join)": closure}
    for k in HEIGHT_KNOTS:
        variants[f"height knots K={k} (LSQ)"] = cp.height_from_knots(f, cp.fit_height_knots(f, truth_u, k))
    return variants


def profile_flight(item) -> dict:
    series, anchor = item
    a, truth_xy, truth_t = _truth(series, anchor)
    truth_u = np.concatenate([[a[2]], series.supervision_values[anchor + 1:, 2]])
    f, length = cp.progress(truth_xy)
    s = f * length
    psi = float(series.scenario.target.psi)
    tan_gpa = math.tan(-float(series.scenario.target.gamma))
    join = ic.truth_join_point(series)
    d_join = float(cg.runway_axes_np(join[0], join[1], psi)[0])
    # The truth's own join on its own path: the first post-anchor row within 200 m of the join point.
    near = np.flatnonzero(np.hypot(*(truth_xy - join[:2]).T) < 200.0)
    s_join = float(s[near[0]]) if len(near) else 0.0
    speed0 = float(np.hypot(a[3], a[4]))

    def score(times, u):
        # A repeated truth row (zero-length step) repeats a fitted time; the record
        # contract wants a strictly increasing clock.
        times = cg.strictly_increasing(np.asarray(times, dtype=np.float64))
        values = np.zeros((len(truth_xy), 6))
        values[:, :2], values[:, 2] = truth_xy, u
        m = common_physical_time_flight_metrics(
            anchor_values=a, predicted_values=values[1:], predicted_offsets_s=times[1:],
            predicted_final_time_s=float(times[-1]), truth_values=series.supervision_values[anchor + 1:],
            truth_offsets_s=truth_t[1:], true_final_time_s=float(truth_t[-1]))
        return {"ade_m": m["ade_m"], "duration_error_s": float(times[-1] - truth_t[-1])}

    out = {"flight_id": series.flight_id, "timing": {}, "height": {}, "combined": {}}
    timings = _timing_variants(f, length, truth_t, speed0)
    heights = _height_variants(f, truth_u, s, s_join, d_join, tan_gpa, float(a[2]))
    for name, times in timings.items():
        out["timing"][name] = score(times, truth_u)                       # truth height: timing alone
    for name, u in heights.items():
        out["height"][name] = score(truth_t, u)                           # truth timing: height alone
    for k in SLOWNESS_KNOTS:
        if k in HEIGHT_KNOTS:
            out["combined"][f"slowness + height knots K={k}"] = score(
                timings[f"slowness knots K={k} (LSQ)"], heights[f"height knots K={k} (LSQ)"])
    # Labels at each K the design considers, with the residual the pair reaches on this
    # flight (its own ADE, and the fitted duration's error) so a fallback rule can read it.
    out["labels"] = {
        "slowness_knots": {str(k): cp.fit_slowness_knots(f, length, truth_t, k).tolist() for k in LABEL_KNOTS},
        "height_knots": {str(k): cp.fit_height_knots(f, truth_u, k).tolist() for k in LABEL_KNOTS},
        "combined_ade_m": {str(k): out["combined"][f"slowness + height knots K={k}"]["ade_m"] for k in LABEL_KNOTS},
        "combined_duration_error_s": {str(k): out["combined"][f"slowness + height knots K={k}"]["duration_error_s"] for k in LABEL_KNOTS},
        "duration_s": float(truth_t[-1]), "naive_duration_s": float(timings["naive (anchor speed → 70 m/s)"][-1]),
        "path_length_m": length, "anchor_speed_mps": speed0,
    }
    return out


def cmd_speed(args: argparse.Namespace) -> None:
    reference_dir, config, anchor, keys, masks, by_compact, series, out = _flights(args)
    print(f"{args.airport}: {len(series)} flights, anchor index {anchor}; profiles on the TRUTH path "
          f"(slowness knots {SLOWNESS_KNOTS}, height knots {HEIGHT_KNOTS}, labels at K in {LABEL_KNOTS})")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(profile_flight, [(item, anchor) for item in series], chunksize=8))
    by_key = {by_compact[r["flight_id"]]: r for r in results}
    (out / "profile_labels.json").write_text(json.dumps(
        {"airport": args.airport, "reference": str(reference_dir), "anchor_index": anchor,
         "truth": "post-anchor supervision rows", "label_knots": LABEL_KNOTS,
         "flights": {k: r["labels"] for k, r in by_key.items()}}, indent=1))
    lines = [f"# {args.airport} closure profile oracle: {len(by_key)} flights on the TRUTH path (geometry error zero); "
             f"ADE on the common-true-time grid. The 'closure' height row approximates closure_geometry.vertical_profile "
             f"on the truth path (join = first truth row within 200 m of the truth join point; glidepath from the truth's "
             f"arc length), the timing rows use the truth height, the height rows the truth timing."]
    for stratum, selected in _stratum_lines(masks, keys, by_key):
        lines.append(f"\n## {stratum} (n={len(selected)})\n")
        lines.append("| variant | ADE mean / p50 | abs Δdur p50 |")
        lines.append("|---|---:|---:|")
        for group in ("timing", "height", "combined"):
            for name in by_key[selected[0]][group]:
                rows = [by_key[k][group][name] for k in selected]
                ade = [r["ade_m"] for r in rows]
                lines.append(f"| {group}: {name} | {np.mean(ade):.0f} / {np.median(ade):.0f} | "
                             f"{np.median(np.abs([r['duration_error_s'] for r in rows])):.1f} |")
    text = "\n".join(lines)
    print(text)
    (out / "oracle_profile.txt").write_text(text + "\n")
    print(f"\nwrote {out / 'oracle_profile.txt'} and profile_labels.json")


# ── P1.c-1: labels for every flight of the cohort ────────────────────────────

LABEL_SCHEMA = "closure-labels-v1"
LABEL_RESIDUAL_MAX_M = 1_000.0     # above this F3 residual the flight is a fallback, not a label


def label_flight(item) -> dict:
    """The closure decoder's label for one flight: the canonical F3 geometry (join at the
    localizer entry, via in the chart and in runway axes), the K=4 / K=8 slowness and
    height knots on the truth path, their residuals, and the difficulty covariates."""
    series, anchor = item
    psi = float(series.scenario.target.psi)
    a, truth_xy, truth_t = _truth(series, anchor)
    truth_u = np.concatenate([[a[2]], series.supervision_values[anchor + 1:, 2]])
    anchor_pose = cg.AnchorPose.from_state(a, psi)
    join = ic.truth_join_point(series)
    d_join0 = float(cg.runway_axes_np(join[0], join[1], psi)[0])
    f0 = cg.rule_template(anchor_pose, psi, d_join0)
    f1 = cg.fit_rule_template(anchor_pose, psi, truth_xy, d_join0)
    f2 = cg.fit_dubins_join(anchor_pose, psi, truth_xy, d_join0, seed=f1)
    f3, spread = cg.fit_via_dubins(anchor_pose, psi, truth_xy, d_join0, seeds=(f1, f2))
    f, length = cp.progress(truth_xy)
    profile = {str(k): {"slowness_knots": cp.fit_slowness_knots(f, length, truth_t, k).tolist(),
                        "height_knots": cp.fit_height_knots(f, truth_u, k).tolist()} for k in LABEL_KNOTS}
    for k, block in profile.items():
        times = cg.strictly_increasing(cp.times_from_slowness(f, length, np.array(block["slowness_knots"])))
        values = np.zeros((len(truth_xy), 6))
        values[:, :2], values[:, 2] = truth_xy, cp.height_from_knots(f, np.array(block["height_knots"]))
        block["ade_m"] = _score(series, anchor, times[1:], values[1:], truth_t)["ade_m"]
        block["duration_error_s"] = float(times[-1] - truth_t[-1])
    geometry = {**_residuals(f3, truth_xy), "via_label_spread_m": spread, "rule_kind": f0.kind}
    valid = bool(f3.params["canonical"] and geometry["error_m"] <= LABEL_RESIDUAL_MAX_M)
    return {"flight_id": series.flight_id, "runway": series.scenario.source.get("runway"),
            "anchor": {"d_m": anchor_pose.d, "xt_m": anchor_pose.xt, "heading_rad": anchor_pose.heading,
                       "speed_mps": anchor_pose.speed_mps, "u_m": float(a[2])},
            "d_join_truth_m": d_join0, "geometry": geometry, "profile": profile,
            "duration_s": float(truth_t[-1]), "path_length_m": length, "valid": valid,
            "difficulty": approach_difficulty(series, anchor).to_dict()}


def cmd_labels(args: argparse.Namespace) -> None:
    reference_dir = args.reference if args.reference.is_absolute() else REPO_ROOT / args.reference
    config = TSConfig(**_reference_rows(reference_dir)[1])
    if config.coordinate_frame != COORDINATE_FRAME_ENU:
        raise ValueError("the closure geometry assumes the threshold-anchored enu chart")
    anchor = config.seq_len - 1
    manifest = pid._manifest(args.airport)
    flights = load_flight_dicts(manifest, verbose=False)
    if args.limit:
        flights = flights[:args.limit]
    series, report = build_series(flights, config, airport=args.airport)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{args.airport}: {len(flights)} rostered arrivals, {report.built} built "
          f"(skipped {dict(report.skipped)}); anchor index {anchor}; labelling with {args.workers} workers")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(label_flight, [(item, anchor) for item in series], chunksize=4))
    by_id = {r["flight_id"]: r for r in results}
    out.write_text(json.dumps({
        "schema": LABEL_SCHEMA, "airport": args.airport, "manifest": str(manifest), "config_source": str(reference_dir),
        "anchor_index": anchor, "truth": "post-anchor supervision rows", "objective": "closure_geometry.path_error_m",
        "label_knots": LABEL_KNOTS, "residual_max_m": LABEL_RESIDUAL_MAX_M, "flights": by_id}, indent=1))
    valid = np.array([r["valid"] for r in results])
    canonical = np.array([r["geometry"]["params"]["canonical"] for r in results])
    error = np.array([r["geometry"]["error_m"] for r in results])
    vectored = np.array([r["difficulty"]["route_tortuosity"] >= cfa.STRAIGHT_TORTUOSITY
                         and not r["difficulty"]["established_at_anchor"] for r in results])
    print(f"labels: {len(results)} flights; valid {valid.mean():.1%} (canonical {canonical.mean():.1%}, "
          f"residual <= {LABEL_RESIDUAL_MAX_M:.0f} m {(error <= LABEL_RESIDUAL_MAX_M).mean():.1%}); "
          f"F3 arc-aligned error p50 all {np.median(error):.0f} m, vectored (n={vectored.sum()}) "
          f"{np.median(error[vectored]) if vectored.any() else math.nan:.0f} m, straight-in "
          f"{np.median(error[~vectored]) if (~vectored).any() else math.nan:.0f} m")
    for k in LABEL_KNOTS:
        ade = np.array([r["profile"][str(k)]["ade_m"] for r in results])
        print(f"  profile K={k}: ADE on the truth path mean {ade.mean():.0f} / p50 {np.median(ade):.0f} m "
              f"(vectored {ade[vectored].mean() if vectored.any() else math.nan:.0f} m)")
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func, help_text in (("geometry", cmd_geometry, "fit every path family to the truth, score with truth timing"),
                                  ("speed", cmd_speed, "speed / height profile parametrisations on the truth path"),
                                  ("labels", cmd_labels, "the closure decoder's labels for EVERY rostered flight (--out is the JSON file)")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--airport", required=True)
        p.add_argument("--reference", type=Path, required=True, help="prediction dir whose summary fixes the config (and, for geometry/speed, the flights)")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--limit", type=int, default=0, help="random subset of flights (smoke runs)")
        p.add_argument("--workers", type=int, default=8)
        p.set_defaults(func=func)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
