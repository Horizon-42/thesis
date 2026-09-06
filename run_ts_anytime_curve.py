#!/usr/bin/env python
"""A0-fixed: how a prediction gets better as the aircraft flies — the re-anchoring curve.

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

This is the FIXED arm: every checkpoint here was trained at L−1, so a later anchor is
out-of-distribution for it and the reading contains that cost. The A0-random arm
(`random_train_anchor=True`) is what separates the two, and §六 3 requires both to be
reported together — this runner measures one of them and says so in its own output.

A `cta_conditioning=given` checkpoint is refused: its duration IS the truth's, so its curve
would be an identity check that improves for free. The output directory is an immutable
artifact and must not exist; nothing is written back to the data plane.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
TS_DIR = REPO_ROOT / "4dTrajectory" / "ts_transformer"
for path in (TS_DIR, REPO_ROOT / "geokit" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from approach_difficulty import (  # noqa: E402
    STRATUM_ALL,
    STRATUM_VECTORED,
    approach_difficulty,
    remaining_path_profile_m,
    strata_masks,
)
from config import CTA_CONDITIONING_GIVEN, TSConfig  # noqa: E402
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

RESULT_SCHEMA = "ts-anytime-curve-a0-v1"

DEFAULT_BINS_KM = "20,16,12,8,6,4,2"
DEFAULT_MIN_FUTURE_S = 60.0
# The outer-test split stays sealed (repo experiment rule): a curve read on it would spend
# the one-shot ledger on a measurement that has no gate.
FORBIDDEN_SPLIT = "test"
# ``test`` is offered so the refusal can say WHY rather than "invalid choice".
SPLITS = ("val", "train", FORBIDDEN_SPLIT)

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

ANCHOR_DEFINITION = (
    "per flight and bin, the observed sample whose remaining path "
    "(approach_difficulty.remaining_path_profile_m — the horizontal arc length the truth "
    "still flies) is closest to the bin value; empty when that sample has no full lookback "
    "(anchor < seq_len - 1) or less than --min-future-s of truth after it"
)


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


def load_arm(label: str, path: Path, split: str, device: torch.device) -> Arm:
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
    if split not in payload["split"]:
        raise SystemExit(f"{label} ({path}): the checkpoint has no {split!r} split")
    airports = tuple(entry["airport"] for entry in payload["data_provenance"]["manifests"])
    manifests = [pipeline.arrival_manifest_path(item) for item in airports]
    # The package helper owns the roster rule (the fingerprint reads the pre-split
    # lateral-pass roster iff the checkpoint recorded one); every replaying runner goes
    # through it, which is what the L5.a fitter was missing when it died at startup.
    require_matching_data_provenance(payload, checkpoint_data_provenance(payload, manifests))
    return Arm(label, path, model.to(device), config, normalizer, payload, airports, manifests)


def cohort_series(arm: Arm, split: str) -> list:
    """The arm's own ``split`` flights, in the checkpoint's order.

    Every flight in the split must rebuild: a curve over a subset of the split is a
    different cohort, and the strata sizes the coverage is read against would be that
    subset's rather than the split's.
    """
    wanted = arm.payload["split"][split]
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
            f"curve must cover the whole {split!r} split. First missing: {missing[0]!r}"
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

def measure_bin(model, series, profiles, keys, target_m, *, config, normalizer, device,
                batch_size: int, min_future_s: float) -> dict[str, dict]:
    """Every flight that HAS an anchor at ``target_m``, scored from it.

    Flights are grouped by anchor index because that is what one forecast call takes; the
    groups are then chunked so a bin never builds a batch bigger than the checkpoint's own.
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
                rows[keys[index]] = {
                    "anchor_index": anchor,
                    "remaining_path_m": float(profiles[index][anchor]),
                    "ade_m": float(metrics["ade_m"]),
                    "fde_m": float(metrics["fde_m"]),
                    "final_time_error_s": float(metrics["final_time_error_s"]),
                }
    return rows


def _p(values, quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def stratum_block(selected: list[dict], stratum_size: int) -> dict:
    """One (bin, stratum) cell. ``n`` and ``coverage`` are published, never implied."""
    coverage = len(selected) / stratum_size if stratum_size else 0.0
    block = {
        "n": len(selected),
        "stratum_n_at_l1": stratum_size,
        "coverage": coverage,
        "partial": coverage < PARTIAL_COVERAGE,
    }
    if not selected:
        return {**block, "ade_mean_m": None, "fde_p50_m": None,
                "abs_final_time_error_p50_s": None, "abs_final_time_error_p80_s": None,
                "remaining_path_p50_m": None}
    absolute = np.abs([row["final_time_error_s"] for row in selected])
    return {
        **block,
        "ade_mean_m": float(np.mean([row["ade_m"] for row in selected])),
        "fde_p50_m": _p([row["fde_m"] for row in selected], 50),
        "abs_final_time_error_p50_s": _p(absolute, 50),
        "abs_final_time_error_p80_s": _p(absolute, 80),
        # The bin is the NEAREST sample, not an exact distance: this says how near.
        "remaining_path_p50_m": _p([row["remaining_path_m"] for row in selected], 50),
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

def verdicts(bins_m: list[float], curve: dict[float, dict[str, dict]]) -> dict:
    """The vectored stratum's monotonicity, its 1.5 km crossing, and the freeze point.

    Read over the bins that are NOT `partial` only (§六 2), listed by name so the reader
    sees which points the verdicts were denied.
    """
    descending = sorted(bins_m, reverse=True)
    usable = [
        value for value in descending
        if curve[value][STRATUM_VECTORED]["n"] > 0
        and not curve[value][STRATUM_VECTORED]["partial"]
    ]
    ade = {value: curve[value][STRATUM_VECTORED]["ade_mean_m"] for value in usable}
    p80 = {value: curve[value][STRATUM_VECTORED]["abs_final_time_error_p80_s"] for value in usable}

    inversions = [
        {"from_m": far, "to_m": near, "delta_m": ade[near] - ade[far]}
        for far, near in zip(usable, usable[1:])
        if ade[near] - ade[far] > MONOTONE_TOLERANCE_M
    ]
    reachable = [value for value in usable if value > GATE_MIN_REMAINING_M and ade[value] < GATE_ADE_M]
    freeze = next((value for value in usable if p80[value] < FREEZE_TIME_ERROR_P80_S), None)
    return {
        "stratum": STRATUM_VECTORED,
        "bins_read_m": usable,
        "bins_skipped_partial_m": [value for value in descending if value not in usable],
        "monotone": {
            "rule": f"vectored ADE never rises by more than {MONOTONE_TOLERANCE_M:g} m as "
                    "remaining path shrinks",
            "inversions": inversions,
            "status": "pass" if not inversions else "fail",
        },
        "reachable": {
            "rule": f"the largest remaining path above {GATE_MIN_REMAINING_M / 1000:g} km "
                    f"whose vectored ADE is below {GATE_ADE_M:g} m",
            "remaining_path_m": max(reachable) if reachable else None,
            "ade_m": ade[max(reachable)] if reachable else None,
            "status": "pass" if reachable else "fail",
        },
        # Reported, not gated (§2.4 item 3).
        "freeze": {
            "rule": f"the first bin (from far to near) whose vectored |final-time error| "
                    f"p80 is below {FREEZE_TIME_ERROR_P80_S:g} s",
            "freeze_remaining_path_m": freeze,
            "abs_final_time_error_p80_s": p80[freeze] if freeze is not None else None,
        },
    }


# ── rendering ───────────────────────────────────────────────────────────────

def _cell(value, width: int, digits: int) -> str:
    return f"{'n/a':>{width}s}" if value is None else f"{value:>{width}.{digits}f}"


def render(payload: dict) -> str:
    lines = [
        f"A0-fixed anytime curve — split {payload['split']}, bins "
        f"{', '.join(f'{value / 1000:g}' for value in payload['bins_m'])} km, "
        f"min future {payload['min_future_s']:g} s",
        f"anchor: {ANCHOR_DEFINITION}",
        "strata are computed ONCE at the L-1 anchor and fixed for every bin, so the curve is "
        "not a survivor curve; `cov` is the bin's share of that stratum and `partial` "
        f"(cov < {PARTIAL_COVERAGE:g}) marks a point no verdict reads.",
        "FIXED arm: every checkpoint was trained at L-1, so a later anchor is out of "
        "distribution and this curve carries that cost. The A0-random arm is what separates "
        "the two, and both must be reported together.",
    ]
    for label, arm in payload["checkpoints"].items():
        lines += [
            "",
            f"── {label} — {'/'.join(arm['airports'])}, {arm['flights']} flights, "
            f"{arm['prediction_output']} output, anchor L-1 = {arm['anchor_l1']}",
            f"   {arm['checkpoint']}",
        ]
        for stratum, size in arm["stratum_n_at_l1"].items():
            lines += [
                "",
                f"   {stratum}  [{size} flights at L-1]",
                f"   {'s km':>6s} {'s med km':>9s} {'n':>6s} {'cov':>5s} {'partial':>8s} "
                f"{'ADE mean':>9s} {'FDE p50':>9s} {'|dt| p50':>9s} {'|dt| p80':>9s}",
            ]
            for value in payload["bins_m"]:
                cell = arm["bins"][str(value)][stratum]
                median = cell["remaining_path_p50_m"]
                lines.append(
                    f"   {value / 1000:>6.1f} "
                    f"{'n/a' if median is None else f'{median / 1000:.1f}':>9s} "
                    f"{cell['n']:>6d} {cell['coverage']:>5.2f} "
                    f"{'partial' if cell['partial'] else '':>8s} "
                    f"{_cell(cell['ade_mean_m'], 9, 1)} {_cell(cell['fde_p50_m'], 9, 1)} "
                    f"{_cell(cell['abs_final_time_error_p50_s'], 9, 1)} "
                    f"{_cell(cell['abs_final_time_error_p80_s'], 9, 1)}"
                )
        verdict = arm["verdicts"]
        monotone = verdict["monotone"]
        reachable = verdict["reachable"]
        freeze = verdict["freeze"]
        lines += [
            "",
            f"   verdicts on {verdict['stratum']} over "
            f"{[f'{value / 1000:g}' for value in verdict['bins_read_m']]} km"
            + (f" (skipped as partial: "
               f"{[f'{value / 1000:g}' for value in verdict['bins_skipped_partial_m']]} km)"
               if verdict["bins_skipped_partial_m"] else ""),
            f"   1. {monotone['rule']}: {monotone['status'].upper()}"
            + ("" if not monotone["inversions"] else
               " — " + ", ".join(
                   f"{row['from_m'] / 1000:g}->{row['to_m'] / 1000:g} km +{row['delta_m']:.0f} m"
                   for row in monotone["inversions"])),
            f"   2. {reachable['rule']}: "
            + ("none" if reachable["remaining_path_m"] is None else
               f"{reachable['remaining_path_m'] / 1000:g} km "
               f"(ADE {reachable['ade_m']:.0f} m)"),
            f"   3. s_freeze — {freeze['rule']}: "
            + ("never" if freeze["freeze_remaining_path_m"] is None else
               f"{freeze['freeze_remaining_path_m'] / 1000:g} km "
               f"(p80 {freeze['abs_final_time_error_p80_s']:.1f} s)")
            + "  [reported, not a gate]",
        ]
    return "\n".join(lines) + "\n"


# ── the run ─────────────────────────────────────────────────────────────────

def measure_checkpoint(arm: Arm, series: list, args: argparse.Namespace,
                       device: torch.device) -> dict:
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
    batch_size = args.batch_size or config.batch_size
    print(f"{arm.label}: {'/'.join(arm.airports)} {args.split} split, {len(series)} flights, "
          f"anchor L-1 {anchor_l1}, batch {batch_size}, device {device}", flush=True)

    curve: dict[float, dict[str, dict]] = {}
    for target_m in args.bins_m:
        rows = measure_bin(
            arm.model, series, profiles, keys, target_m, config=config,
            normalizer=arm.normalizer, device=device, batch_size=batch_size,
            min_future_s=args.min_future_s,
        )
        curve[target_m] = summarise_bin(rows, masks, keys, stratum_size)
        everything = curve[target_m][STRATUM_ALL]
        vectored = curve[target_m][STRATUM_VECTORED]
        print(f"  {target_m / 1000:>5.1f} km: {everything['n']:>5d} flights "
              f"(cov {everything['coverage']:.2f}) | all ADE "
              f"{_cell(everything['ade_mean_m'], 8, 1)} m | vectored "
              f"{_cell(vectored['ade_mean_m'], 8, 1)} m (n {vectored['n']})", flush=True)

    return {
        "checkpoint": str(arm.path),
        "checkpoint_sha256": file_sha256(arm.path),
        "airports": list(arm.airports),
        "prediction_output": config.prediction_output,
        "latent_dim": config.latent_dim,
        "model": config.model,
        "horizon_mode": config.horizon_mode,
        "anchor_l1": anchor_l1,
        "flights": len(series),
        "stratum_n_at_l1": stratum_size,
        # The digests the CHECKPOINT carries, not a fresh hash of today's files: the
        # provenance check above has already established they agree, and re-hashing here
        # would have to repeat the roster rule to stay comparable.
        "manifests": provenance_manifest_digests(arm.payload["data_provenance"]),
        "seconds": time.time() - started,
        # Keyed by the bin's metres as a string: JSON object keys are strings, and a float
        # key silently becomes one anyway.
        "bins": {str(value): curve[value] for value in args.bins_m},
        "verdicts": verdicts(args.bins_m, curve),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", action="append", default=None, metavar="LABEL=PATH",
                        help="a trained checkpoint and the name it is reported under; "
                             "repeatable, at least one")
    parser.add_argument("--out", type=Path, required=True,
                        help="must not exist (immutable artifact)")
    parser.add_argument("--split", choices=SPLITS, default="val",
                        help=f"which checkpoint split to replay (default: val; "
                             f"{FORBIDDEN_SPLIT!r} is sealed and refused)")
    parser.add_argument("--bins-km", default=DEFAULT_BINS_KM,
                        help=f"remaining-PATH bins in km (default: {DEFAULT_BINS_KM})")
    parser.add_argument("--min-future-s", type=float, default=DEFAULT_MIN_FUTURE_S,
                        help=f"truth required after an anchor (default: {DEFAULT_MIN_FUTURE_S:g})")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="default: the checkpoint's own")
    args = parser.parse_args(argv)

    if args.split == FORBIDDEN_SPLIT:
        parser.error(
            f"the {FORBIDDEN_SPLIT} split is sealed: A0 is a measurement with no gate, and "
            "replaying it here would spend the one-shot ledger on one"
        )
    if not args.checkpoint:
        parser.error("--checkpoint LABEL=PATH is required (repeat it for more arms)")
    arms: dict[str, Path] = {}
    for entry in args.checkpoint:
        label, separator, raw = entry.partition("=")
        if not separator or not label.strip() or not raw.strip():
            parser.error(f"--checkpoint takes LABEL=PATH, got {entry!r}")
        if label.strip() in arms:
            parser.error(f"--checkpoint label {label.strip()!r} is used twice")
        path = Path(raw.strip())
        arms[label.strip()] = path if path.is_absolute() else REPO_ROOT / path
    bins_km = [float(token) for token in args.bins_km.split(",") if token.strip()]
    if not bins_km or any(value <= 0.0 for value in bins_km):
        parser.error("--bins-km must be a non-empty list of positive distances")
    if len(set(bins_km)) != len(bins_km):
        parser.error("--bins-km repeats a bin")
    if args.min_future_s < 0.0:
        parser.error("--min-future-s must be non-negative")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    args.bins_m = [value * 1000.0 for value in bins_km]

    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    device = resolve_device(args.device)
    # Load every checkpoint FIRST: the CTA refusal, the missing split and the provenance
    # check all live there, and a rejected invocation must leave nothing behind. Only then
    # is the immutable directory claimed, and only then is a track read.
    loaded = [load_arm(label, path, args.split, device) for label, path in arms.items()]
    out.mkdir(parents=True, exist_ok=False)

    payload = {
        "schema": RESULT_SCHEMA,
        "arm": "A0-fixed",
        "split": args.split,
        "bins_m": args.bins_m,
        "min_future_s": args.min_future_s,
        "anchor_definition": ANCHOR_DEFINITION,
        "strata_anchor": "L-1 (seq_len - 1), computed once and fixed for every bin",
        "partial_coverage_threshold": PARTIAL_COVERAGE,
        "device": str(device),
        "checkpoints": {
            arm.label: measure_checkpoint(arm, cohort_series(arm, args.split), args, device)
            for arm in loaded
        },
    }
    (out / "anytime_curve.json").write_text(json.dumps(payload, indent=2))
    text = render(payload)
    (out / "anytime_curve.txt").write_text(text)
    print()
    print(text, end="")
    print(f"wrote {out / 'anytime_curve.txt'} and {out / 'anytime_curve.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
