#!/usr/bin/env python
"""A0: how a prediction gets better as the aircraft flies — the re-anchoring curve.

Anytime-prediction design
(`4dTrajectory/ts_transformer/docs/2026-09-07_anytime_prediction_and_calibrated_eta_design.zh.md`)
§二 2.1–2.4. Every evaluation in this package anchors at L−1 (`seq_len − 1`, 120 s after the
25 km slice starts), which is the moment the ego history knows LEAST about where the
controller is going to send the aircraft. The intent that error budget is made of is not a
property of the flight, it is a property of that instant: it gets exposed as the flight
proceeds. This runner measures how much, by replaying the SAME checkpoint from a grid of
later anchors::

    python run_ts_anytime_curve.py \\
        --checkpoint L1_native32=4dTrajectory/outputs/KRDU/experiments/l1_lowdim_20260907/L1_native32/checkpoint.pt \\
        --out 4dTrajectory/outputs/KRDU/experiments/anytime_a0_20260907

**Anchor grid** (§2.3). Bins are REMAINING PATH, not time: it is the covariate the NEAR /
FAR strata are already cut on (`approach_difficulty.remaining_path_m` — this runner imports
its arithmetic, `remaining_path_profile_m`, rather than restating it) and it is comparable
across flights of different speeds. In each bin a flight's anchor is the observed sample
whose remaining path is CLOSEST to the bin value; the bin is empty for that flight when
that sample has no full lookback (`anchor < seq_len − 1`) or less than `--min-future-s` of
truth after it. Every bin prints its flight count and its coverage of the stratum (§六 2).

**Strata are fixed at L−1** (§六 1): a flight that is vectored at the evaluation anchor and
established at 8 km stays in the vectored stratum at 8 km. Recomputing the label per bin
would make the curve a survivor curve — the hard flights would leave the stratum exactly as
they became easy.

**Bins hold different flights, so the curve is read PAIRED**: the monotonicity verdict
compares adjacent bins over the flights present in BOTH, and states that count. The
per-flight rows stay in the artifact so any other paired reading can be taken from it.

**Both metric families, always** (package rule): the time-aligned ADE / FDE / |Δt| sit
beside the time-free chamfer and Fréchet. A closure arm at 12 km scores ADE 6706 m with a
172 s duration error — reading either family alone gets that flight wrong.

Each checkpoint's arm is named from its own `random_train_anchor`: **A0-fixed** (trained at
L−1, so every later anchor is out of distribution and the curve carries that cost) or
**A0-random**. §六 3 requires the two to be reported together; a run that mixes them says so.

A `cta_conditioning=given` or `intent_conditioning=truth-…` checkpoint is REFUSED: both read
the future at every anchor, so their curves would improve for free. The output directory is
an immutable artifact, is written only once the measurement has succeeded, and nothing is
written back to the data plane.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
for path in (TS_DIR, REPO_ROOT / "geokit" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import geometric_metrics as gm  # noqa: E402
from approach_difficulty import (  # noqa: E402
    STRATUM_ALL,
    STRATUM_VECTORED,
    approach_difficulty,
    remaining_path_profile_m,
    strata_masks,
)
from channels import POSITION_IDX  # noqa: E402
from config import (  # noqa: E402
    CONTROL_HOOKS_AVAILABLE,
    CONTROL_HOOK_OFF,
    CTA_CONDITIONING_GIVEN,
    HOOK_SATURATIONS,
    INTENT_CONDITIONING_NONE,
    PREDICTION_CONTROL,
    TSConfig,
)
from data_provenance import (  # noqa: E402
    checkpoint_data_provenance,
    provenance_manifest_digests,
    require_matching_data_provenance,
)
from dataset import (  # noqa: E402
    Normalizer,
    build_series,
    load_flight_dicts,
    truth_duration_s,
)
from export import observed_series_metrics  # noqa: E402
from forecast import forecast_approaches  # noqa: E402
from io_utils import file_sha256  # noqa: E402
from models import resolve_device  # noqa: E402
from train import load_checkpoint, usable_series  # noqa: E402
import run_ts_pipeline as pipeline  # noqa: E402

RESULT_SCHEMA = "ts-anytime-curve-a0-v2"

DEFAULT_BINS_KM = "20,16,12,8,6,4,2"
DEFAULT_MIN_FUTURE_S = 60.0
# The outer-test split stays sealed (repo experiment rule): a curve read on it would spend
# the one-shot ledger on a measurement that has no gate.
FORBIDDEN_SPLIT = "test"
# ``test`` is offered so the refusal can say WHY rather than "invalid choice".
SPLITS = ("val", "train", FORBIDDEN_SPLIT)

# The two arms of §2.2, named per checkpoint from its own training anchor policy.
ARM_FIXED = "A0-fixed"
ARM_RANDOM = "A0-random"

# §六 2: a bin holding under half of its stratum is marked `partial` and does not enter a
# verdict — the flights it lost are the ones whose geometry never reached that bin.
PARTIAL_COVERAGE = 0.5
# §2.4's three readings. The tolerance is the frame-arm seed floor the package already
# quotes (5–22 m pooled ADE); the 1.5 km / 4 km pair is Phase 0's gate asked at a later
# anchor; the freeze point is reported, never gated.
MONOTONE_TOLERANCE_M = 22.0
GATE_ADE_M = 1500.0
GATE_MIN_REMAINING_M = 4000.0
FREEZE_TIME_ERROR_P80_S = 30.0
# The duration head cannot predict below ~125 s against a true range starting at 21 s
# (package trap, unfixed and present in every flight model). Inside that radius the
# duration error therefore RISES toward the runway however good the geometry gets, which is
# why the freeze block prints this number and every cell prints its predicted-duration p50.
DURATION_HEAD_FLOOR_S = 125.0

ANCHOR_DEFINITION = (
    "per flight and bin, the observed sample whose remaining path "
    "(approach_difficulty.remaining_path_profile_m — the horizontal arc length the truth "
    "still flies) is closest to the bin value; empty when that sample has no full lookback "
    "(anchor < seq_len - 1) or less than --min-future-s of truth after it"
)
GEOMETRY_TRUTH = (
    "the post-anchor supervision rows — the same curve ADE is scored against, not the "
    "threshold-closed one (geometric_metrics' `observed` truth)"
)


@dataclass(frozen=True)
class Grid:
    """What the curve is measured on: parsed once, then read.

    A parsed value rather than attributes grafted onto the argparse namespace, so every
    reader sees the same types and nothing downstream has to know which flag it came from.
    """

    split: str
    bins_m: tuple[float, ...]
    min_future_s: float
    batch_size: int | None
    limit: int


# ── the cell: one table drives the JSON, the text and the empty cell ─────────

def _p(values, quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


@dataclass(frozen=True)
class Metric:
    """One published number: where it comes from, how it reduces, how it prints."""

    key: str
    field: str
    reduce: Callable[[np.ndarray], float]
    header: str
    width: int
    digits: int
    scale: float = 1.0


CELL_METRICS: tuple[Metric, ...] = (
    # The bin is the NEAREST sample, not an exact distance: this says how near it got.
    Metric("remaining_path_p50_m", "remaining_path_m", lambda v: _p(v, 50), "s med km", 9, 1, 1e-3),
    Metric("ade_mean_m", "ade_m", lambda v: float(np.mean(v)), "ADE mean", 9, 1),
    # The MEDIAN is what the ±22 m monotonicity rule reads (the package's convention for a
    # heavy-tailed error); the mean and p95 are published beside it, never instead of it.
    Metric("ade_p50_m", "ade_m", lambda v: _p(v, 50), "ADE p50", 9, 1),
    Metric("ade_p95_m", "ade_m", lambda v: _p(v, 95), "ADE p95", 9, 1),
    Metric("fde_p50_m", "fde_m", lambda v: _p(v, 50), "FDE p50", 9, 1),
    # Time-free geometry beside the time-aligned errors: a timing-only improvement reads as
    # a model improvement on ADE alone (Phase 0, 2026-09-05).
    Metric("chamfer_p50_m", "chamfer_m", lambda v: _p(v, 50), "chamfer", 9, 1),
    Metric("frechet_p50_m", "frechet_m", lambda v: _p(v, 50), "Fréchet", 9, 1),
    Metric("abs_final_time_error_p50_s", "final_time_error_s",
           lambda v: _p(np.abs(v), 50), "|dt| p50", 9, 1),
    Metric("abs_final_time_error_p80_s", "final_time_error_s",
           lambda v: _p(np.abs(v), 80), "|dt| p80", 9, 1),
    # Against DURATION_HEAD_FLOOR_S: inside the floor the duration error is the head's
    # floor, not the geometry's error.
    Metric("predicted_final_time_p50_s", "predicted_final_time_s",
           lambda v: _p(v, 50), "pred T p50", 10, 1),
)
METRIC = {metric.key: metric for metric in CELL_METRICS}


# ── the cohort: the checkpoint's own split, rebuilt as `predict` rebuilds it ──

@dataclass(frozen=True)
class Arm:
    """One loaded checkpoint and the data plane its split will be rebuilt from."""

    label: str
    path: Path
    model: nn.Module
    config: TSConfig
    normalizer: Normalizer
    payload: dict
    airports: tuple[str, ...]
    manifests: list[Path]

    @property
    def arm(self) -> str:
        """Which of §2.2's two arms this checkpoint IS — its own training anchor policy,
        never a run-level claim: one invocation may hold both."""
        return ARM_RANDOM if self.config.random_train_anchor else ARM_FIXED


def load_arm(label: str, path: Path, grid: Grid, device: torch.device,
             *, command_hook: str | None = None, hook_saturation: str | None = None) -> Arm:
    """Every refusal this runner can make about a checkpoint, before it reads one track.

    The airports come from the checkpoint's own provenance — never chosen here, which is
    why the fingerprint is compared strictly rather than as the airport SUBSET `predict`
    allows for a pooled checkpoint narrowed by ``--data``.
    """
    model, config, normalizer, payload = load_checkpoint(path)
    if config.cta_conditioning == CTA_CONDITIONING_GIVEN:
        raise SystemExit(
            f"{label} ({path}): cta_conditioning=given reads the future — the CTA IS the "
            "truth duration — so its anytime curve would be an identity check that gets "
            "better for free, not a prediction that gets better with observation"
        )
    if config.intent_conditioning != INTENT_CONDITIONING_NONE:
        raise SystemExit(
            f"{label} ({path}): intent_conditioning={config.intent_conditioning!r} reads the "
            "FUTURE (the truth join point / the lead's true landing time), and it reads it "
            "afresh at every anchor — the curve would measure how fast the oracle's own "
            "answer arrives, not how fast intent is exposed to the aircraft"
        )
    if grid.split not in payload["split"]:
        raise SystemExit(f"{label} ({path}): the checkpoint has no {grid.split!r} split")
    if command_hook is not None:
        if config.prediction_output != PREDICTION_CONTROL:
            raise SystemExit(
                f"{label} ({path}): --command-hook applies to the control output only, and "
                f"this checkpoint predicts {config.prediction_output!r}"
            )
        # The remaining hook preconditions (lag dynamics, native grid, ENU) are TSConfig's,
        # and it refuses with the reason.
        config = replace(
            config, control_command_hook=command_hook, control_hook_saturation=hook_saturation
        )
    airports = tuple(entry["airport"] for entry in payload["data_provenance"]["manifests"])
    manifests = [pipeline.arrival_manifest_path(item) for item in airports]
    # The package helper owns the roster rule (the fingerprint reads the pre-split
    # lateral-pass roster iff the checkpoint recorded one); every replaying runner goes
    # through it, which is what the L5.a fitter was missing when it died at startup.
    require_matching_data_provenance(payload, checkpoint_data_provenance(payload, manifests))
    return Arm(label, path, model.to(device), config, normalizer, payload, airports, manifests)


def cohort_series(arm: Arm, grid: Grid) -> list:
    """The arm's own ``split`` flights, in the checkpoint's order.

    Every flight must rebuild: a curve over a silent subset of the split is a different
    cohort, and the strata sizes the coverage is read against would be that subset's.
    ``--limit`` narrows the split DELIBERATELY (a smoke test on a real checkpoint) and the
    denominators then come from the flights actually built — the artifact says both counts.
    """
    wanted = arm.payload["split"][grid.split]
    if grid.limit:
        wanted = wanted[: grid.limit]
    built, report = build_series(
        load_flight_dicts(arm.manifests, include_flight_keys=set(wanted), verbose=False),
        arm.config,
        aircraft_type=arm.config.aircraft_type,
    )
    print(f"  {report.format()}", flush=True)
    by_id = {
        item.dataset_id: item
        for item in usable_series(built, arm.config, verbose=False)
    }
    missing = [key for key in wanted if key not in by_id]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(wanted)} checkpoint flights could not be rebuilt; the "
            f"curve must cover the whole cohort. First missing: {missing[0]!r}"
        )
    return [by_id[key] for key in wanted]


def bin_anchor(series, profile: np.ndarray, target_m: float, *,
               seq_len: int, min_future_s: float) -> int | None:
    """The flight's anchor for one remaining-path bin, or None when the bin is empty.

    The sample is chosen on remaining path ALONE and only then tested for eligibility: a
    flight whose closest sample cannot be an anchor has no reading at that bin, rather than
    a reading taken somewhere else on its track.
    """
    anchor = int(np.argmin(np.abs(profile - target_m)))
    if anchor < seq_len - 1:
        return None
    if truth_duration_s(series, anchor) < min_future_s:
        return None
    return anchor


# ── one bin: forecast at each flight's own anchor, score after it ────────────

def _geometry(series, forecast) -> dict[str, float]:
    """The time-free metrics for one forecast against the truth AFTER its anchor.

    Both paths are ``[N, 4]`` ``(e, n, u, t)`` in the flight's own chart, which is all
    `geometric_metrics` needs — chamfer and Fréchet are relative, so the chart origin
    (threshold or airport) does not enter.
    """
    anchor_time = float(series.times[forecast.anchor])
    future = series.supervision_times > anchor_time
    truth = np.column_stack([
        np.asarray(series.supervision_values, dtype=np.float64)[future][:, list(POSITION_IDX)],
        np.asarray(series.supervision_times, dtype=np.float64)[future] - anchor_time,
    ])
    predicted = np.column_stack([
        np.asarray(forecast.values, dtype=np.float64)[:, list(POSITION_IDX)],
        np.cumsum(forecast.sample_durations_s),
    ])
    return gm.path_metrics(predicted, truth)


def measure_bin(model, series, profiles, keys, target_m, *, config, normalizer, device,
                batch_size: int, min_future_s: float) -> dict[str, dict]:
    """Every flight that HAS an anchor at ``target_m``, scored from it.

    Flights are grouped by anchor index because that is what one forecast call takes; the
    groups are then chunked so a bin never builds a batch bigger than the checkpoint's own.
    On real data the anchors are nearly all distinct, so the effective batch is small (≈2.7
    at KRDU) — the grouping is for correctness, not for speed.
    """
    groups: dict[int, list[int]] = {}
    for index, (item, profile) in enumerate(zip(series, profiles)):
        anchor = bin_anchor(
            item, profile, target_m, seq_len=config.seq_len, min_future_s=min_future_s
        )
        if anchor is not None:
            groups.setdefault(anchor, []).append(index)

    rows: dict[str, dict] = {}
    for anchor in sorted(groups):
        members = groups[anchor]
        for start in range(0, len(members), batch_size):
            chunk = members[start : start + batch_size]
            batch = [series[index] for index in chunk]
            forecasts = forecast_approaches(
                model, batch, config, normalizer, anchor=anchor, device=device
            )
            for index, item, forecast in zip(chunk, batch, forecasts, strict=True):
                metrics = observed_series_metrics(
                    item, forecast, points=config.validation_common_grid_points
                )
                geometry = _geometry(item, forecast)
                rows[keys[index]] = {
                    "anchor_index": anchor,
                    "remaining_path_m": float(profiles[index][anchor]),
                    "ade_m": float(metrics["ade_m"]),
                    "fde_m": float(metrics["fde_m"]),
                    "final_time_error_s": float(metrics["final_time_error_s"]),
                    "predicted_final_time_s": float(forecast.predicted_final_time_s),
                    "chamfer_m": float(geometry["chamfer_m"]),
                    "frechet_m": float(geometry["frechet_m"]),
                }
    return rows


def stratum_block(selected: list[dict], stratum_size: int) -> dict:
    """One (bin, stratum) cell. ``n`` and ``coverage`` are published, never implied."""
    coverage = len(selected) / stratum_size if stratum_size else 0.0
    values = {metric.key: None for metric in CELL_METRICS} if not selected else {
        metric.key: metric.reduce(
            np.array([row[metric.field] for row in selected], dtype=np.float64)
        )
        for metric in CELL_METRICS
    }
    return {
        "n": len(selected),
        "stratum_n_at_l1": stratum_size,
        "coverage": coverage,
        "partial": coverage < PARTIAL_COVERAGE,
        **values,
    }


def summarise_bin(rows: dict[str, dict], masks: dict[str, np.ndarray],
                  keys: list[str], stratum_size: dict[str, int]) -> dict[str, dict]:
    return {
        stratum: stratum_block(
            [rows[key] for key, member in zip(keys, mask) if member and key in rows],
            stratum_size[stratum],
        )
        for stratum, mask in masks.items()
    }


# ── §2.4's three readings ───────────────────────────────────────────────────

def _paired_step(near_rows: dict[str, dict], far_rows: dict[str, dict],
                 members: set[str]) -> dict | None:
    """One adjacent-bin step read over the flights present in BOTH bins.

    Bins hold different populations — a flight only appears where its geometry put a sample
    — so an unpaired difference of two medians mixes "the model got better" with "the bin
    got easier flights". None when no flight reached both.
    """
    shared = sorted((set(near_rows) & set(far_rows)) & members)
    if not shared:
        return None
    far = np.array([far_rows[key]["ade_m"] for key in shared], dtype=np.float64)
    near = np.array([near_rows[key]["ade_m"] for key in shared], dtype=np.float64)
    return {
        "paired_n": len(shared),
        "ade_p50_from_m": _p(far, 50),
        "ade_p50_to_m": _p(near, 50),
        "delta_p50_m": _p(near, 50) - _p(far, 50),
    }


def verdicts(grid: Grid, curve: dict[float, dict], masks: dict[str, np.ndarray],
             keys: list[str]) -> dict:
    """The vectored stratum's monotonicity, its 1.5 km crossing, and the freeze point.

    Read over the bins that are NOT `partial` only (§六 2), listed by name so the reader
    sees which points the verdicts were denied, and PAIRED across adjacent bins.
    """
    descending = sorted(grid.bins_m, reverse=True)
    usable = [
        value for value in descending
        if curve[value]["strata"][STRATUM_VECTORED]["n"] > 0
        and not curve[value]["strata"][STRATUM_VECTORED]["partial"]
    ]
    cell = {value: curve[value]["strata"][STRATUM_VECTORED] for value in usable}
    members = {key for key, member in zip(keys, masks[STRATUM_VECTORED]) if member}

    steps = []
    for far, near in zip(usable, usable[1:]):
        step = _paired_step(curve[near]["flights"], curve[far]["flights"], members)
        if step is not None:
            steps.append({"from_m": far, "to_m": near, **step})
    inversions = [step for step in steps if step["delta_p50_m"] > MONOTONE_TOLERANCE_M]
    reachable = [
        value for value in usable
        if value > GATE_MIN_REMAINING_M and cell[value]["ade_p50_m"] < GATE_ADE_M
    ]
    freeze = next(
        (value for value in usable
         if cell[value]["abs_final_time_error_p80_s"] < FREEZE_TIME_ERROR_P80_S),
        None,
    )
    return {
        "stratum": STRATUM_VECTORED,
        "statistic": "ade_p50_m",
        "bins_read_m": usable,
        "bins_skipped_partial_m": [value for value in descending if value not in usable],
        "monotone": {
            "rule": f"paired vectored ADE p50 never rises by more than "
                    f"{MONOTONE_TOLERANCE_M:g} m as remaining path shrinks",
            "steps": steps,
            "inversions": inversions,
            # "unread" is not "pass": with no pair to compare there is no reading, and a
            # gate that answers on no data is worse than one that says so.
            "status": "unread" if not steps else ("pass" if not inversions else "fail"),
        },
        "reachable": {
            "rule": f"the largest remaining path above {GATE_MIN_REMAINING_M / 1000:g} km "
                    f"whose vectored ADE p50 is below {GATE_ADE_M:g} m",
            "remaining_path_m": max(reachable) if reachable else None,
            "ade_p50_m": cell[max(reachable)]["ade_p50_m"] if reachable else None,
            "status": "pass" if reachable else ("fail" if usable else "unread"),
        },
        # Reported, not gated (§2.4 item 3).
        "freeze": {
            "rule": f"the first bin (from far to near) whose vectored |final-time error| "
                    f"p80 is below {FREEZE_TIME_ERROR_P80_S:g} s",
            "freeze_remaining_path_m": freeze,
            "abs_final_time_error_p80_s": (
                cell[freeze]["abs_final_time_error_p80_s"] if freeze is not None else None
            ),
            "duration_head_floor_s": DURATION_HEAD_FLOOR_S,
            "predicted_final_time_p50_s": {
                str(value): cell[value]["predicted_final_time_p50_s"] for value in usable
            },
        },
    }


# ── rendering ───────────────────────────────────────────────────────────────

def _cell(value, metric: Metric) -> str:
    if value is None:
        return f"{'n/a':>{metric.width}s}"
    return f"{value * metric.scale:>{metric.width}.{metric.digits}f}"


def _banner(arms: list[str]) -> list[str]:
    """What the reading is worth, said per arm — a mixed run says it is mixed."""
    lines = []
    if ARM_FIXED in arms:
        lines.append(
            f"{ARM_FIXED}: the checkpoint was trained at L-1, so every later anchor is OUT "
            "OF DISTRIBUTION and this curve carries that cost as well as the information "
            "gain. It must be read against an A0-random arm, never alone."
        )
    if ARM_RANDOM in arms:
        lines.append(
            f"{ARM_RANDOM}: the checkpoint was trained with random anchors, so a later "
            "anchor is in distribution; its difference from A0-fixed IS the out-of-"
            "distribution cost."
        )
    if len(arms) > 1:
        lines.append("This run holds BOTH arms; compare them bin by bin, not run to run.")
    return lines


def render(payload: dict) -> str:
    grid = payload["grid"]
    bins_m = grid["bins_m"]
    lines = [
        f"A0 anytime curve — split {grid['split']}, bins "
        f"{', '.join(f'{value / 1000:g}' for value in bins_m)} km, "
        f"min future {grid['min_future_s']:g} s"
        + ("" if not grid["limit"] else
           f", --limit {grid['limit']} (a SMOKE TEST over the first flights of the split, "
           "not a measurement)"),
        f"anchor: {ANCHOR_DEFINITION}",
        "strata are computed ONCE at the L-1 anchor and fixed for every bin, so the curve is "
        "not a survivor curve; `cov` is the bin's share of that stratum and `partial` "
        f"(cov < {PARTIAL_COVERAGE:g}) marks a point no verdict reads.",
        f"geometry truth: {GEOMETRY_TRUTH}. Read both families — a timing-only change moves "
        "ADE and leaves chamfer/Fréchet where they were.",
        *_banner(sorted({arm["arm"] for arm in payload["checkpoints"].values()})),
    ]
    header = (
        f"   {'s km':>6s} {'n':>6s} {'cov':>5s} {'partial':>8s} "
        + " ".join(f"{metric.header:>{metric.width}s}" for metric in CELL_METRICS)
    )
    for label, arm in payload["checkpoints"].items():
        lines += [
            "",
            f"── {label} [{arm['arm']}] — {'/'.join(arm['airports'])}, {arm['flights']} "
            f"flights of {arm['split_flights']} in the split, {arm['prediction_output']} "
            f"output, anchor L-1 = {arm['anchor_l1']}"
            + ("" if arm["command_hook"] is None else f", hook {arm['command_hook']}"),
            f"   {arm['checkpoint']}",
        ]
        for stratum, size in arm["stratum_n_at_l1"].items():
            lines += ["", f"   {stratum}  [{size} flights at L-1]", header]
            for value in bins_m:
                cell = arm["bins"][str(value)]["strata"][stratum]
                lines.append(
                    f"   {value / 1000:>6.1f} {cell['n']:>6d} {cell['coverage']:>5.2f} "
                    f"{'partial' if cell['partial'] else '':>8s} "
                    + " ".join(_cell(cell[metric.key], metric) for metric in CELL_METRICS)
                )
        lines += _render_verdicts(arm["verdicts"])
    return "\n".join(lines) + "\n"


def _render_verdicts(verdict: dict) -> list[str]:
    monotone, reachable, freeze = verdict["monotone"], verdict["reachable"], verdict["freeze"]
    steps = ", ".join(
        f"{step['from_m'] / 1000:g}->{step['to_m'] / 1000:g} km "
        f"{step['delta_p50_m']:+.0f} m (n={step['paired_n']})"
        for step in monotone["steps"]
    )
    predicted = ", ".join(
        f"{float(value) / 1000:g} km {seconds:.0f} s"
        for value, seconds in freeze["predicted_final_time_p50_s"].items()
    )
    return [
        "",
        f"   verdicts on {verdict['stratum']}, statistic {verdict['statistic']}, over "
        f"{[f'{value / 1000:g}' for value in verdict['bins_read_m']]} km"
        + (f" (skipped as partial: "
           f"{[f'{value / 1000:g}' for value in verdict['bins_skipped_partial_m']]} km)"
           if verdict["bins_skipped_partial_m"] else ""),
        f"   1. {monotone['rule']}: {monotone['status'].upper()}"
        + ("  [no adjacent pair shares a flight — nothing to read]"
           if monotone["status"] == "unread" else f" — steps: {steps}"),
        f"   2. {reachable['rule']}: "
        + ("none" if reachable["remaining_path_m"] is None else
           f"{reachable['remaining_path_m'] / 1000:g} km "
           f"(ADE p50 {reachable['ade_p50_m']:.0f} m)"),
        f"   3. s_freeze — {freeze['rule']}: "
        + ("not reached in the bins read" if freeze["freeze_remaining_path_m"] is None else
           f"{freeze['freeze_remaining_path_m'] / 1000:g} km "
           f"(p80 {freeze['abs_final_time_error_p80_s']:.1f} s)")
        + "  [reported, not a gate]",
        f"      the duration head cannot predict below ~{freeze['duration_head_floor_s']:g} s "
        "(package trap), so inside that radius |dt| RISES toward the runway however good the "
        f"geometry gets. Predicted duration p50 per bin: {predicted or 'n/a'}.",
    ]


# ── the run ─────────────────────────────────────────────────────────────────

def measure_checkpoint(arm: Arm, series: list, grid: Grid, device: torch.device) -> dict:
    """One arm's whole curve: the strata fixed at L−1, then every bin scored from its own
    anchor."""
    started = time.time()
    config = arm.config
    anchor_l1 = config.seq_len - 1
    keys = [item.dataset_id for item in series]
    # §六 1: ONE label per flight, taken at the evaluation anchor and reused at every bin.
    # Relabelling per bin would drop each flight out of the vectored stratum exactly when
    # it rolled out on the centreline, and the curve would measure the survivors.
    difficulty = {
        key: approach_difficulty(item, anchor_l1).to_dict()
        for key, item in zip(keys, series)
    }
    masks = strata_masks(difficulty, keys)
    stratum_size = {stratum: int(mask.sum()) for stratum, mask in masks.items()}
    profiles = [remaining_path_profile_m(item) for item in series]
    batch_size = grid.batch_size or config.batch_size
    print(f"{arm.label} [{arm.arm}]: {'/'.join(arm.airports)} {grid.split} split, "
          f"{len(series)} flights, anchor L-1 {anchor_l1}, batch {batch_size}, "
          f"device {device}", flush=True)

    curve: dict[float, dict] = {}
    for target_m in grid.bins_m:
        rows = measure_bin(
            arm.model, series, profiles, keys, target_m, config=config,
            normalizer=arm.normalizer, device=device, batch_size=batch_size,
            min_future_s=grid.min_future_s,
        )
        curve[target_m] = {
            "strata": summarise_bin(rows, masks, keys, stratum_size),
            # The per-flight rows stay in the artifact: bins hold different populations, so
            # any paired reading (this runner's monotonicity verdict included) needs them.
            "flights": rows,
        }
        everything = curve[target_m]["strata"][STRATUM_ALL]
        vectored = curve[target_m]["strata"][STRATUM_VECTORED]
        print(f"  {target_m / 1000:>5.1f} km: {everything['n']:>5d} flights "
              f"(cov {everything['coverage']:.2f}) | all ADE p50 "
              f"{_cell(everything['ade_p50_m'], METRIC['ade_p50_m'])} m | vectored "
              f"{_cell(vectored['ade_p50_m'], METRIC['ade_p50_m'])} m (n {vectored['n']})",
              flush=True)

    return {
        "arm": arm.arm,
        "checkpoint": str(arm.path),
        "checkpoint_sha256": file_sha256(arm.path),
        "airports": list(arm.airports),
        "prediction_output": config.prediction_output,
        "latent_dim": config.latent_dim,
        "model": config.model,
        "horizon_mode": config.horizon_mode,
        "random_train_anchor": bool(config.random_train_anchor),
        "command_hook": (
            None if config.control_command_hook == CONTROL_HOOK_OFF
            else f"{config.control_command_hook}/{config.control_hook_saturation}"
        ),
        "anchor_l1": anchor_l1,
        "flights": len(series),
        "split_flights": len(arm.payload["split"][grid.split]),
        "stratum_n_at_l1": stratum_size,
        # The digests the CHECKPOINT carries, not a fresh hash of today's files: the
        # provenance check has already established they agree, and re-hashing here would
        # have to repeat the roster rule to stay comparable.
        "manifests": provenance_manifest_digests(arm.payload["data_provenance"]),
        "seconds": time.time() - started,
        # Keyed by the bin's metres as a string: JSON object keys are strings, and a float
        # key silently becomes one anyway.
        "bins": {str(value): curve[value] for value in grid.bins_m},
        "verdicts": verdicts(grid, curve, masks, keys),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", action="append", default=None, metavar="LABEL=PATH",
                        help="a trained checkpoint and the name it is reported under; "
                             "repeatable, at least one. A relative PATH resolves against "
                             "the repository root")
    parser.add_argument("--out", type=Path, required=True,
                        help="must not exist (immutable artifact); written only once the "
                             "measurement has succeeded. A relative path resolves against "
                             "the repository root")
    parser.add_argument("--split", choices=SPLITS, default="val",
                        help=f"which checkpoint split to replay (default: val; 'train' is "
                             f"IN-SAMPLE and its curve is not a prediction result; "
                             f"{FORBIDDEN_SPLIT!r} is sealed and refused)")
    parser.add_argument("--bins-km", default=DEFAULT_BINS_KM,
                        help=f"remaining-PATH bins in km (default: {DEFAULT_BINS_KM})")
    parser.add_argument("--min-future-s", type=float, default=DEFAULT_MIN_FUTURE_S,
                        help=f"truth required after an anchor (default: {DEFAULT_MIN_FUTURE_S:g}); "
                             "a bin nearer than this many seconds of flight is empty by "
                             "construction and prints n=0")
    parser.add_argument("--limit", type=int, default=0,
                        help="measure only the first N flights of each split — a smoke test "
                             "on a real checkpoint; coverage denominators come from the "
                             "flights actually built and the artifact carries both counts")
    parser.add_argument("--command-hook",
                        choices=[hook for hook in CONTROL_HOOKS_AVAILABLE
                                 if hook != CONTROL_HOOK_OFF],
                        default=None, metavar="HOOK",
                        help="run the control rollout through this command hook, as "
                             "`predict --command-hook` does (the adopted deployment form); "
                             "the checkpoint's own hook applies otherwise")
    parser.add_argument("--hook-saturation", choices=HOOK_SATURATIONS, default=None,
                        help="with --command-hook: soft (tanh / softplus) or hard (clamp)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="default: the checkpoint's own. Flights are batched by ANCHOR, "
                             "so on real data the effective batch is small whatever this says")
    return parser


def parse_arms(parser: argparse.ArgumentParser, entries: list[str] | None) -> dict[str, Path]:
    if not entries:
        parser.error("--checkpoint LABEL=PATH is required (repeat it for more arms)")
    arms: dict[str, Path] = {}
    for entry in entries:
        label, separator, raw = entry.partition("=")
        if not separator or not label.strip() or not raw.strip():
            parser.error(f"--checkpoint takes LABEL=PATH, got {entry!r}")
        if label.strip() in arms:
            parser.error(f"--checkpoint label {label.strip()!r} is used twice")
        path = Path(raw.strip())
        arms[label.strip()] = path if path.is_absolute() else REPO_ROOT / path
    return arms


def parse_grid(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Grid:
    if args.split == FORBIDDEN_SPLIT:
        parser.error(
            f"the {FORBIDDEN_SPLIT} split is sealed: A0 is a measurement with no gate, and "
            "replaying it here would spend the one-shot ledger on one"
        )
    bins_km = [float(token) for token in args.bins_km.split(",") if token.strip()]
    if not bins_km or any(value <= 0.0 for value in bins_km):
        parser.error("--bins-km must be a non-empty list of positive distances")
    if len(set(bins_km)) != len(bins_km):
        parser.error("--bins-km repeats a bin")
    if args.min_future_s < 0.0:
        parser.error("--min-future-s must be non-negative")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.limit < 0:
        parser.error("--limit must be non-negative (0 = the whole split)")
    if (args.command_hook is None) != (args.hook_saturation is None):
        parser.error("--command-hook and --hook-saturation are given together")
    return Grid(
        split=args.split,
        bins_m=tuple(value * 1000.0 for value in bins_km),
        min_future_s=args.min_future_s,
        batch_size=args.batch_size,
        limit=args.limit,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    arms = parse_arms(parser, args.checkpoint)
    grid = parse_grid(parser, args)

    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        raise FileExistsError(f"{out} exists; the anytime curve is an immutable artifact")
    device = resolve_device(args.device)
    # Load every checkpoint FIRST: the CTA and intent refusals, the missing split, the hook
    # preconditions and the provenance check all live there, and a rejected invocation must
    # leave nothing behind. Only then is a track read.
    loaded = [
        load_arm(label, path, grid, device,
                 command_hook=args.command_hook, hook_saturation=args.hook_saturation)
        for label, path in arms.items()
    ]

    payload = {
        "schema": RESULT_SCHEMA,
        "grid": {
            "split": grid.split,
            "bins_m": list(grid.bins_m),
            "min_future_s": grid.min_future_s,
            "limit": grid.limit,
        },
        "anchor_definition": ANCHOR_DEFINITION,
        "strata_anchor": "L-1 (seq_len - 1), computed once and fixed for every bin",
        "geometry_truth": GEOMETRY_TRUTH,
        "partial_coverage_threshold": PARTIAL_COVERAGE,
        "device": str(device),
        "checkpoints": {
            arm.label: measure_checkpoint(arm, cohort_series(arm, grid), grid, device)
            for arm in loaded
        },
    }
    text = render(payload)
    # The artifact appears whole: a run that dies mid-measurement leaves a `.partial-*`
    # directory next to it, never a half-written curve under the name a reader will cite.
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(dir=out.parent, prefix=f"{out.name}.partial-"))
    (staged / "anytime_curve.json").write_text(json.dumps(payload, indent=2))
    (staged / "anytime_curve.txt").write_text(text)
    staged.chmod(0o755)                     # mkdtemp is 0700; the artifact is readable
    staged.rename(out)
    print()
    print(text, end="")
    print(f"wrote {out / 'anytime_curve.txt'} and {out / 'anytime_curve.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
