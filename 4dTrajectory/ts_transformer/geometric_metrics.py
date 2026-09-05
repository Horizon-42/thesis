"""Time-free path metrics for the readouts: what a prediction's GEOMETRY got wrong, apart
from when it flew it.

The package's ADE/FDE compare positions at the same TRUE time (``metrics.py``), so a
prediction that flies the right path at the wrong speed is charged the full along-path
displacement — on vectored flights that timing term dominates (Phase 0: the truth path
with a naive speed profile still scores 1.3 km; docs/2026-09-05_scene_phase0_results.zh.md).
The readouts therefore print both families side by side, and the scene design doc (§一)
rules that a conclusion drawn from one family alone does not count.

Three geometric distances, on the HORIZONTAL path unless stated:

* ``chamfer_m`` — symmetric mean nearest-point distance after resampling both paths every
  ``RESAMPLE_STEP_M``. Order-blind: a path that sweeps the same area scores well.
* ``discrete_frechet_m`` — the order-preserving "dog-leash" distance on the same
  resampling; a doubled-back or region-filling path cannot cheat it.
* ``arc_aligned_ade_m`` — mean 3D distance between the paths at the same FRACTION of
  their own horizontal arc length (``ARC_POINTS`` fractions, the anchor excluded, the
  ADE grid's convention): order-preserving and speed-free, the closest cousin of ADE.

and two timing quantities that carry the rest of the ADE:

* ``along_path_lag_s`` — at each arc fraction, the time the prediction reaches it minus
  the time the truth does (positive = late).
* ``duration_error_s`` — the lag's last value: the exported path's end time minus the
  truth path's. On the exported states' clock, so a horizon-capped flight (states cut at
  H) reads the cap here while ``summary.json``'s ``final_time_error_s`` reads the duration
  head; they agree everywhere else.

**The arc family is parametrised by each path's OWN arc length, which the state output's
node-scale saw-tooth doubles** (review 2026-09-05: its exported polyline reverses heading
by more than 90° at half of its 2 s nodes — p50 0.50 of nodes, every flight above 0.05 —
where control rollouts and the observed truth reverse at none, max 0.008; raw length ratio
≈ 2 against 1.01). Chamfer, Fréchet and the time-aligned ADE are immune. So each flight
carries ``reversal_share`` and is ``arc_family_valid`` only below
``ARC_FAMILY_MAX_REVERSAL_SHARE``; a block aggregates arc-ADE and the lag over its valid
flights only, states their share, and the table prints ``n/a`` below
``ARC_FAMILY_MIN_SHARE`` — nothing smooths the path silently. The length ratio is printed
as information (a genuinely short or long prediction is an error, not an artifact).

Truth definition (``GEOMETRY_TRUTHS``). The exported states file carries the OBSERVED rows
only; the fitted tail that closes the supervision path onto the threshold (the truth the
4D metrics use, ending at ``true_final_time_s``) is not exported. ``closed`` (the
readouts' default) appends the target threshold at ``true_final_time_s`` as the truth's
last node — a straight closure of the gap the observed track stops short (KRDU val:
median 380 m / 6 s, p95 660 m / 10 s), so both metric families end at the same point
and time. ``observed`` scores the observed rows as they are (the Phase 0 diagnostics'
convention); the endpoint-sensitive metrics then carry that gap, and the duration error
is read against the last observed time. The readouts print which one they used and the
median closure.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from arc_length_geometry import resample_horizontal_arc_length_numpy
from geokit import METRES_PER_DEG_LAT, metres_per_deg_lon

RESAMPLE_STEP_M = 100.0
ARC_POINTS = 64
# A polyline whose consecutive steps reverse heading (> 90°) at more than this share of its
# nodes is a saw-tooth, not a route, and its arc length parametrises nothing (measured
# 2026-09-05: state arms p50 0.50, control arms and the truth 0.000, truth max 0.008).
ARC_FAMILY_MAX_REVERSAL_SHARE = 0.05
# Below this share of valid flights a block's arc-ADE / lag are not printed.
ARC_FAMILY_MIN_SHARE = 0.95

GEOMETRY_TRUTH_CLOSED = "closed"
GEOMETRY_TRUTH_OBSERVED = "observed"
GEOMETRY_TRUTHS = (GEOMETRY_TRUTH_CLOSED, GEOMETRY_TRUTH_OBSERVED)


def cumulative_arc_m(xy: np.ndarray) -> np.ndarray:
    """Horizontal mileage of every row from the first (``[0, ...]``)."""
    xy = np.asarray(xy, dtype=np.float64)
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy[:, :2], axis=0).T))])


def resample_by_step(xy: np.ndarray, step: float = RESAMPLE_STEP_M) -> np.ndarray:
    """Points along a horizontal path every ``step`` metres, both endpoints included: the
    last grid point snaps onto the end when it falls within half a step of it (that final
    interval is then up to 1.5 steps), otherwise the end is appended. A path shorter than
    one step is returned with its rows as they are."""
    xy = np.asarray(xy, dtype=np.float64)[:, :2]
    s = cumulative_arc_m(xy)
    if s[-1] < step:
        return xy
    grid = np.arange(0.0, s[-1], step)
    # s[-1] >= step here, so grid has at least two points and the snap never touches the start.
    if s[-1] - grid[-1] < 0.5 * step:
        grid[-1] = s[-1]
    else:
        grid = np.append(grid, s[-1])
    return np.stack([np.interp(grid, s, xy[:, 0]), np.interp(grid, s, xy[:, 1])], 1)


def chamfer_m(a: np.ndarray, b: np.ndarray, *, step: float = RESAMPLE_STEP_M) -> float:
    """Symmetric mean nearest-point distance between two horizontal paths."""
    a, b = resample_by_step(a, step), resample_by_step(b, step)
    return float(0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()))


def _frechet_from_distances(d: np.ndarray) -> float:
    """Discrete Fréchet distance from the pairwise distance matrix ``d[i, j]``.

    The classic recurrence ``ca[i, j] = max(d[i, j], min(ca[i-1, j], ca[i-1, j-1],
    ca[i, j-1]))`` filled along anti-diagonals ``i + j = k``, whose cells depend only on
    the two previous diagonals, so each diagonal is one vectorised step."""
    n, m = d.shape
    ca = np.empty((n, m), dtype=np.float64)
    ca[0, :] = np.maximum.accumulate(d[0, :])
    ca[:, 0] = np.maximum.accumulate(d[:, 0])
    for k in range(2, n + m - 1):
        lo, hi = max(1, k - (m - 1)), min(n - 1, k - 1)
        if lo > hi:
            continue
        i = np.arange(lo, hi + 1)
        j = k - i
        best = np.minimum(np.minimum(ca[i - 1, j], ca[i - 1, j - 1]), ca[i, j - 1])
        ca[i, j] = np.maximum(d[i, j], best)
    return float(ca[-1, -1])


def discrete_frechet_m(a: np.ndarray, b: np.ndarray, *, step: float = RESAMPLE_STEP_M) -> float:
    """Discrete Fréchet distance between two horizontal paths, resampled every ``step``."""
    a, b = resample_by_step(a, step), resample_by_step(b, step)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return _frechet_from_distances(d)


def _arc_fractions(points: int) -> np.ndarray:
    return np.arange(1, points + 1, dtype=np.float64) / points


def arc_aligned_ade_m(pred_xyz: np.ndarray, truth_xyz: np.ndarray, *, points: int = ARC_POINTS) -> float:
    """Mean 3D distance at the same fractions of each path's own horizontal arc length
    (``1/points … 1``; the shared anchor at fraction 0 is not scored)."""
    pred = resample_horizontal_arc_length_numpy(np.asarray(pred_xyz)[:, :3], points=points + 1)[1:]
    truth = resample_horizontal_arc_length_numpy(np.asarray(truth_xyz)[:, :3], points=points + 1)[1:]
    return float(np.linalg.norm(pred - truth, axis=1).mean())


def _time_at_fractions(xyt: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    xyt = np.asarray(xyt, dtype=np.float64)
    s = cumulative_arc_m(xyt[:, :2])
    return np.interp(fractions * s[-1], s, xyt[:, -1])


def along_path_lag_s(pred_xyt: np.ndarray, truth_xyt: np.ndarray, *, points: int = ARC_POINTS) -> np.ndarray:
    """``(points,)`` predicted minus true time at the same fraction of each path's own
    horizontal arc length (columns ``[e, n, t]``; positive = the prediction is late).
    The last entry is the end-time difference of the two paths."""
    fractions = _arc_fractions(points)
    return _time_at_fractions(pred_xyt, fractions) - _time_at_fractions(truth_xyt, fractions)


def reversal_share(xy: np.ndarray) -> float:
    """Share of a polyline's interior nodes at which the heading reverses (consecutive
    non-zero steps more than 90° apart) — ~0 on a route, ~0.5 on a node-scale saw-tooth."""
    step = np.diff(np.asarray(xy, dtype=np.float64)[:, :2], axis=0)
    step = step[np.hypot(*step.T) > 0.0]
    heading = np.arctan2(step[:, 1], step[:, 0])
    turn = np.abs((np.diff(heading) + np.pi) % (2.0 * np.pi) - np.pi)
    return float((turn > 0.5 * np.pi).mean()) if len(turn) else 0.0


def path_metrics(pred_xyzt: np.ndarray, truth_xyzt: np.ndarray) -> dict[str, float | bool]:
    """Every time-free metric plus the timing for one ``[N, 4]`` pair ``(e, n, u, t)``,
    both in the same chart. Per-flight values; ``summarize`` aggregates them."""
    pred, truth = np.asarray(pred_xyzt, dtype=np.float64), np.asarray(truth_xyzt, dtype=np.float64)
    lag = along_path_lag_s(pred[:, [0, 1, 3]], truth[:, [0, 1, 3]])
    # Both paths are arc-parametrised, so both must be routes (the truth is, max 0.008).
    reversal, truth_reversal = reversal_share(pred), reversal_share(truth)
    return {
        "chamfer_m": chamfer_m(pred, truth),
        "frechet_m": discrete_frechet_m(pred, truth),
        "arc_ade_m": arc_aligned_ade_m(pred, truth),
        "path_length_ratio": float(cumulative_arc_m(pred)[-1] / cumulative_arc_m(truth)[-1]),
        "reversal_share": reversal,
        "truth_reversal_share": truth_reversal,
        "arc_family_valid": bool(max(reversal, truth_reversal) <= ARC_FAMILY_MAX_REVERSAL_SHARE),
        "along_path_lag_median_s": float(np.median(lag)),
        "along_path_lag_abs_mean_s": float(np.abs(lag).mean()),
        "duration_error_s": float(pred[-1, 3] - truth[-1, 3]),
    }


# ── the exported record contract ────────────────────────────────────────────

def chart_en(lat, lon, lat0: float, lon0: float):
    """Local east/north metres of ``(lat, lon)`` about ``(lat0, lon0)`` — the readouts'
    one equirectangular chart (scalars or arrays)."""
    return (np.asarray(lon) - lon0) * metres_per_deg_lon(lat0), (np.asarray(lat) - lat0) * METRES_PER_DEG_LAT


def chart_rows(rows: list[dict], target: dict) -> np.ndarray:
    """``[N, 4]`` ``(e, n, u, t)`` of state rows in the threshold-anchored chart of ``target``."""
    lat0, lon0, alt0 = float(target["lat"]), float(target["lon"]), float(target["alt"])
    e, n = chart_en([r["lat"] for r in rows], [r["lon"] for r in rows], lat0, lon0)
    u = np.array([r["alt"] for r in rows], dtype=np.float64) - alt0
    t = np.array([r["t"] for r in rows], dtype=np.float64)
    return np.stack([e, n, u, t], 1)


def truth_path(states: dict, target: dict, *, geometry_truth: str, true_final_time_s: float) -> np.ndarray:
    """The truth as ``[N, 4]`` ``(e, n, u, t)``: the post-anchor observed rows, and under
    ``closed`` the threshold (chart origin) at ``true_final_time_s`` appended as the last node."""
    if geometry_truth not in GEOMETRY_TRUTHS:
        raise ValueError(f"geometry_truth must be one of {GEOMETRY_TRUTHS}, got {geometry_truth!r}")
    observed = chart_rows([r for r in states["observed_states"] if r["t"] >= 0.0], target)
    if geometry_truth == GEOMETRY_TRUTH_OBSERVED:
        return observed
    return np.concatenate([observed, [[0.0, 0.0, 0.0, float(true_final_time_s)]]], axis=0)


def record_geometry(eval_record: dict, states: dict, row: dict, *, geometry_truth: str) -> dict[str, float | bool]:
    """``path_metrics`` for one exported flight (its ``*_eval.json`` + ``*_states.json`` and
    the ``summary.json`` row), plus the closure the truth needed (0 under ``observed``)."""
    target = eval_record["target_state"]
    truth = truth_path(states, target, geometry_truth=geometry_truth,
                       true_final_time_s=float(row["true_final_time_s"]))
    pred = chart_rows(states["predicted_states"], target)
    metrics = path_metrics(pred, truth)
    closed = geometry_truth == GEOMETRY_TRUTH_CLOSED
    last_observed = truth[-2] if closed else truth[-1]
    metrics["truth_closure_m"] = float(np.hypot(last_observed[0], last_observed[1])) if closed else 0.0
    metrics["truth_closure_s"] = float(truth[-1, 3] - last_observed[3]) if closed else 0.0
    return metrics


def summarize(rows: list[dict]) -> dict[str, float | bool | None]:
    """Stratum aggregates of ``record_geometry`` rows: medians where the readouts print
    medians, means where they print means. The arc family (arc-ADE, lag) is aggregated over
    the ``arc_family_valid`` flights only — ``arc_family_flights`` / ``arc_family_share`` say
    how many that is — and is ``None`` (JSON null) wherever the table prints ``n/a``
    (``arc_family_printed``, below ``ARC_FAMILY_MIN_SHARE``), so a two-flight mean of a
    saw-tooth arm never sits in the JSON under the key that means "this arm's arc-ADE"."""
    if not rows:
        raise ValueError("summarize needs at least one flight")
    chamfer = np.array([r["chamfer_m"] for r in rows])
    frechet = np.array([r["frechet_m"] for r in rows])
    duration = np.array([r["duration_error_s"] for r in rows])
    valid = [r for r in rows if r["arc_family_valid"]]
    share = len(valid) / len(rows)
    printed = share >= ARC_FAMILY_MIN_SHARE

    def over_valid(key, reduce):
        return float(reduce([r[key] for r in valid])) if printed else None

    return {
        "chamfer_median_m": float(np.median(chamfer)), "chamfer_mean_m": float(chamfer.mean()),
        "frechet_median_m": float(np.median(frechet)), "frechet_mean_m": float(frechet.mean()),
        "path_length_ratio_median": float(np.median([r["path_length_ratio"] for r in rows])),
        "reversal_share_median": float(np.median([r["reversal_share"] for r in rows])),
        "arc_family_flights": len(valid),
        "arc_family_share": share,
        "arc_family_printed": printed,
        "arc_ade_mean_m": over_valid("arc_ade_m", np.mean),
        "arc_ade_median_m": over_valid("arc_ade_m", np.median),
        # Across valid flights: the median of the per-flight lag medians, the mean of the
        # per-flight mean |lag|.
        "along_path_lag_flight_median_s": over_valid("along_path_lag_median_s", np.median),
        "along_path_lag_abs_mean_s": over_valid("along_path_lag_abs_mean_s", np.mean),
        "duration_error_abs_median_s": float(np.median(np.abs(duration))),
        "duration_error_median_s": float(np.median(duration)),
        "truth_closure_median_m": float(np.median([r["truth_closure_m"] for r in rows])),
        "truth_closure_median_s": float(np.median([r["truth_closure_s"] for r in rows])),
    }


GEOMETRY_TABLE_HEADER = ["chamfer p50", "Fréchet p50", "arc-ADE", "len ratio", "abs Δdur p50", "lag p50"]


def _arc_cell(value: float | None, spec: str, block: dict) -> str:
    if value is None:
        return "n/a"
    share = block["arc_family_share"]
    return f"{value:{spec}}" if share == 1.0 else f"{value:{spec}} ({share:.1%})"


def geometry_table_cells(block: dict[str, float | bool | None]) -> list[str]:
    """The geometry columns every readout prints, from a ``summarize`` block. The arc
    family prints ``n/a`` where too few of the block's polylines are routes, and carries
    the valid share whenever it is below 100 %."""
    return [
        f"{block['chamfer_median_m']:.0f}", f"{block['frechet_median_m']:.0f}",
        _arc_cell(block["arc_ade_mean_m"], ".0f", block),
        f"{block['path_length_ratio_median']:.2f}",
        f"{block['duration_error_abs_median_s']:.1f}",
        _arc_cell(block["along_path_lag_flight_median_s"], "+.1f", block),
    ]


def geometry_truth_notice(geometry_truth: str, block: dict, flights: int) -> str:
    """One line for the readout header: the truth definition, its closure, and every
    parameter the geometry columns depend on."""
    if geometry_truth == GEOMETRY_TRUTH_OBSERVED:
        truth = ("geometry truth: OBSERVED post-anchor rows as exported (no closure to the threshold; "
                 "Fréchet / arc-ADE carry the observed track's stop-short gap and Δdur is read against "
                 "the last observed time)")
    else:
        truth = (f"geometry truth: observed post-anchor rows CLOSED to the threshold at true_final_time_s "
                 f"(straight closure, median {block['truth_closure_median_m']:.0f} m / "
                 f"{block['truth_closure_median_s']:.0f} s over {flights} flights)")
    return (f"{truth}; chamfer / Fréchet horizontal at {RESAMPLE_STEP_M:.0f} m steps, arc-ADE 3D at "
            f"{ARC_POINTS} fractions of each path's own length, Δdur on the exported states' clock "
            f"(summary.json's final_time_error_s / time MAE read the duration head); arc-ADE / lag "
            f"are aggregated over the flights whose exported polyline is a route (heading reversals "
            f"at <= {ARC_FAMILY_MAX_REVERSAL_SHARE:.0%} of nodes; the state output's node-scale "
            f"saw-tooth reverses at ~50 % and doubles its arc length) and print n/a where fewer than "
            f"{ARC_FAMILY_MIN_SHARE:.0%} of a block qualify; the len ratio column is information, not a gate")
