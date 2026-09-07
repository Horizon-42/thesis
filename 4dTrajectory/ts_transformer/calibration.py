"""Split-conformal calibration of the predicted arrival-time interval (B2, design §三 3.2).

The quantile duration head (B1) publishes five levels of ``p(T | history)``. Nothing makes
them CALIBRATED: a head trained by the pinball loss on 6 800 flights is as miscalibrated as
any other fitted quantile, and an 80 % interval that actually covers 62 % is worse than no
interval at all. Conformalized quantile regression (CQR) fixes that with one number per
stratum: the conformity score ``max(q_lo − T, T − q_hi)`` is how far outside the interval
the truth fell (negative when it fell inside), and its ``1 − α`` quantile ``δ_α`` on a
CALIBRATION set is exactly the amount ``[q_lo − δ, q_hi + δ]`` has to grow to cover ``1 − α``
of exchangeable flights.

**Four rules this module exists to hold** (design §六 4 and the repo's experiment
principles):

1. **The calibration set is never the test split and never the training split.** It is the
   VALIDATION split, halved: half A gives δ, half B measures what that δ actually covered,
   then the halves swap. Both coverages are published beside the mean δ — the artifact's own
   honesty check, not a claim.
2. **Calibration is PER STRATUM.** Measured on KRDU val (B0, §〇.2), the vectored |Δt| p80
   is 65.8–72.5 s against straight-in's 12.0–20.3 s: one pooled δ hands every straight-in
   flight an absurd interval. The strata are `approach_difficulty.strata_masks` — the same
   cut every other readout uses — computed once at the L−1 anchor.
3. **A thin stratum refuses.** Below `MIN_CALIBRATION_FLIGHTS` in either half, the stratum
   gets no δ; it is recorded in ``refused_strata`` with its count and a flight in it falls
   through `INTERVAL_STRATUM_PRECEDENCE` to the pooled one. A δ read off 11 flights is a
   number, not a calibration.
4. **No table, no claim.** A checkpoint without one predicts uncalibrated quantiles and the
   record says ``calibrated: false``. The table is bound to `checkpoint_sha256`, so a table
   written for another checkpoint is refused rather than applied.

The interval arithmetic lives here and only here: the calibration runner, `forecast` (which
stamps the interval on a record) and the fan readout all call these functions.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from approach_difficulty import STRATA_COVARIATES, STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, strata_masks
from config import DURATION_QUANTILES

#: The table's own version. Bump it when the meaning of a stored δ changes; `predict`
#: refuses a table it does not recognise rather than applying it under new rules.
CONFORMAL_SCHEMA = "ts-conformal-cqr-v1"

#: The miscoverage levels calibrated, and the quantile PAIR each is built from. α = 0.2 is
#: the design's headline 80 % interval (q10, q90); α = 0.5 is the 50 % one (q25, q75). The
#: pair is part of the definition — δ recentres a specific pair, and applying it to another
#: would be calibrating one interval and publishing a different one.
ALPHA_QUANTILE_PAIRS: dict[float, tuple[float, float]] = {
    0.2: (0.1, 0.9),
    0.5: (0.25, 0.75),
}
CONFORMAL_ALPHAS: tuple[float, ...] = tuple(sorted(ALPHA_QUANTILE_PAIRS))

#: Below this, in EITHER half, a stratum is refused (rule 3 above). 30 is the smallest count
#: at which the empirical 0.9 quantile of the scores is not simply "the largest one".
MIN_CALIBRATION_FLIGHTS = 30

#: The alpha whose CALIBRATED endpoints B3 decodes beside the five levels — the design's
#: headline 80 % interval. Decoding every alpha's endpoints would multiply the fan for a
#: reading nothing asks for.
FAN_INTERVAL_ALPHA = min(CONFORMAL_ALPHAS)

#: Where `predict --cta-from-quantiles` writes the fan, and how one leaf is named. The
#: directory grammar lives HERE rather than in `cli.predict` so `run_ts_quantile_fan_readout`
#: can read a fan without importing the CLI (and torch with it); a readout that guessed the
#: spelling would report an empty fan as a missing one.
QUANTILE_DIR_NAME = "quantiles"


def quantile_directory_name(tau: float) -> str:
    """``q10`` … ``q90`` for one duration quantile."""
    return f"q{round(tau * 100):02d}"


def interval_directory_name(alpha: float, end: str) -> str:
    """``a20lo`` / ``a20hi`` for a calibrated interval endpoint."""
    return f"a{round(alpha * 100):02d}{end}"


#: Which stratum's δ a flight's interval comes from: the FIRST of these it belongs to and
#: which HAS a δ. A precedence, not a partition — `strata_masks`' strata overlap (an
#: established flight can be straight-in) — and it names the two the design separates,
#: falling through to the pooled one for the remainder and for a refused stratum.
INTERVAL_STRATUM_PRECEDENCE = (STRATUM_STRAIGHT_IN, STRATUM_VECTORED, STRATUM_ALL)


def quantile_pair_indices(alpha: float) -> tuple[int, int]:
    """The (lo, hi) column indices of ``alpha``'s pair in `DURATION_QUANTILES`."""
    low, high = ALPHA_QUANTILE_PAIRS[alpha]
    return DURATION_QUANTILES.index(low), DURATION_QUANTILES.index(high)


@dataclass(frozen=True)
class CalibrationSample:
    """One flight's contribution: its predicted quantiles, its truth and its covariates."""

    key: str
    quantiles_s: np.ndarray                 # [Q], seconds, DURATION_QUANTILES order
    truth_final_time_s: float
    covariates: dict[str, Any]              # at least STRATA_COVARIATES


def conformity_scores(
    lo_s: np.ndarray, hi_s: np.ndarray, truth_s: np.ndarray
) -> np.ndarray:
    """CQR's score ``max(q_lo − T, T − q_hi)``: how far OUTSIDE the interval the truth fell.

    Negative inside (the distance to the nearer edge, as a slack), positive outside. One
    definition, so the runner, the readout and the record cannot each grow the interval by
    a differently signed quantity.
    """
    return np.maximum(lo_s - truth_s, truth_s - hi_s)


def conformal_delta(scores: np.ndarray, alpha: float) -> float:
    """The split-conformal ``δ_α``: the ``⌈(n+1)(1−α)⌉ / n`` empirical quantile of ``scores``.

    The finite-sample level, not the plain ``1 − α`` one. The design says "the ``1 − α``
    quantile"; the ``(n + 1)`` correction is what makes the coverage guarantee hold at finite
    n, and without it the interval under-covers by ``≈ 1/n`` on purpose-built data. At the
    KRDU half-split (n ≈ 700 per half) it moves δ by well under a second, so it changes the
    published number hardly at all and the guarantee entirely.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    count = len(scores)
    if count < 1:
        raise ValueError("a conformal quantile needs at least one calibration score")
    level = min(1.0, math.ceil((count + 1) * (1.0 - alpha)) / count)
    return float(np.quantile(np.asarray(scores, dtype=np.float64), level, method="higher"))


def calibrated_interval(lo_s: float, hi_s: float, delta_s: float) -> tuple[float, float]:
    """``[q_lo − δ, q_hi + δ]``, never inverted: δ can be negative when the head over-covers."""
    low, high = lo_s - delta_s, hi_s + delta_s
    return (low, high) if low <= high else (0.5 * (low + high),) * 2


def interval_coverage(
    lo_s: np.ndarray, hi_s: np.ndarray, delta_s: float, truth_s: np.ndarray
) -> float:
    """The share of flights whose truth falls in the δ-widened interval."""
    return float(np.mean((truth_s >= lo_s - delta_s) & (truth_s <= hi_s + delta_s)))


def calibration_halves(keys: Sequence[str], split_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two deterministic halves of the calibration cohort, as index arrays.

    Keyed on the run's own ``split_seed`` and the flight identity, hashed the way
    `splits.py` hashes the outer split — a DIFFERENT question (which half of val), so it
    carries its own salt and the two partitions are independent. Sorted-and-cut rather than
    thresholded, so the halves are exactly equal (± one flight) whatever the cohort size,
    and independent of the order the caller happened to build the flights in.
    """
    order = np.argsort([
        hashlib.sha256(f"conformal:{split_seed}:{key}".encode()).hexdigest()
        for key in keys
    ], kind="stable")
    cut = len(order) // 2
    return order[:cut], order[cut:]


def _stratum_block(
    samples: Sequence[CalibrationSample],
    mask: np.ndarray,
    halves: tuple[np.ndarray, np.ndarray],
    alpha: float,
) -> dict[str, Any]:
    """One (α, stratum) cell: δ from each half, the coverage the OTHER half realised."""
    low_index, high_index = quantile_pair_indices(alpha)
    quantiles = np.stack([sample.quantiles_s for sample in samples])
    truth = np.array([sample.truth_final_time_s for sample in samples], dtype=np.float64)
    parts = [np.intersect1d(half, np.flatnonzero(mask)) for half in halves]
    counts = [int(len(part)) for part in parts]
    if min(counts) < MIN_CALIBRATION_FLIGHTS:
        return {"refused": True, "calibration_flights": counts}

    def fit(part: np.ndarray) -> float:
        return conformal_delta(
            conformity_scores(
                quantiles[part, low_index], quantiles[part, high_index], truth[part]
            ),
            alpha,
        )

    deltas = [fit(part) for part in parts]
    # δ from one half, coverage on the OTHER: a δ scored on its own calibration set would
    # report the level it was constructed to hit, which is not a measurement.
    coverages = [
        interval_coverage(
            quantiles[other, low_index], quantiles[other, high_index], delta, truth[other]
        )
        for delta, other in zip(deltas, reversed(parts))
    ]
    widths = [
        float(np.median(
            quantiles[part, high_index] - quantiles[part, low_index] + 2.0 * delta
        ))
        for delta, part in zip(deltas, reversed(parts))
    ]
    return {
        "refused": False,
        "calibration_flights": counts,
        # The deployed δ is the mean of the two halves' — each is a valid split-conformal
        # δ for the other half, and the mean uses all the calibration data there is.
        "delta_s": float(np.mean(deltas)),
        "delta_halves_s": [float(value) for value in deltas],
        "coverage": float(np.mean(coverages)),
        "coverage_halves": [float(value) for value in coverages],
        "median_width_s": float(np.mean(widths)),
    }


def calibrate(
    samples: Sequence[CalibrationSample],
    *,
    split_seed: int,
    split: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """The conformal table: α → stratum → δ, the realised coverage, and what refused.

    Refuses outright only when the POOLED stratum is too thin — there is then nothing to
    calibrate and no fallback to fall through to.
    """
    if not samples:
        raise ValueError("calibration needs at least one flight")
    missing = [
        name for name in STRATA_COVARIATES if name not in samples[0].covariates
    ]
    if missing:
        raise ValueError(f"calibration samples are missing the covariates {missing}")
    keys = [sample.key for sample in samples]
    masks = strata_masks({sample.key: sample.covariates for sample in samples}, keys)
    halves = calibration_halves(keys, split_seed)
    alphas: dict[str, Any] = {}
    for alpha in CONFORMAL_ALPHAS:
        blocks = {
            stratum: _stratum_block(samples, mask, halves, alpha)
            for stratum, mask in masks.items()
        }
        if blocks[STRATUM_ALL]["refused"]:
            raise ValueError(
                f"{len(samples)} calibration flights cannot support the pooled stratum "
                f"(each half needs {MIN_CALIBRATION_FLIGHTS}); there is nothing to calibrate"
            )
        alphas[_alpha_key(alpha)] = {
            "quantile_pair": list(ALPHA_QUANTILE_PAIRS[alpha]),
            "strata": {
                stratum: block for stratum, block in blocks.items() if not block["refused"]
            },
            "refused_strata": {
                stratum: block["calibration_flights"]
                for stratum, block in blocks.items() if block["refused"]
            },
        }
    return {
        "schema": CONFORMAL_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "split": split,
        "split_seed": split_seed,
        "quantiles": list(DURATION_QUANTILES),
        "min_calibration_flights": MIN_CALIBRATION_FLIGHTS,
        "calibration_flights": len(samples),
        "half_flights": [int(len(halves[0])), int(len(halves[1]))],
        "interval_stratum_precedence": list(INTERVAL_STRATUM_PRECEDENCE),
        "alphas": alphas,
    }


def _alpha_key(alpha: float) -> str:
    """JSON object keys are strings; one spelling, so a lookup cannot miss by format."""
    return f"{alpha:g}"


def interval_stratum(table: dict[str, Any], covariates: dict[str, Any]) -> str:
    """The stratum ONE flight's interval is read from: the first it is in that HAS a δ.

    Goes through `strata_masks` on a one-row table rather than restating the cuts, so a
    record's stratum and the calibration table's are decided by the same code.
    """
    masks = strata_masks({"flight": covariates}, ["flight"])
    calibrated = {
        stratum
        for block in table["alphas"].values()
        for stratum in block["strata"]
    }
    for stratum in table["interval_stratum_precedence"]:
        if masks[stratum][0] and stratum in calibrated:
            return stratum
    return STRATUM_ALL


def conformal_intervals(
    quantiles_s: Sequence[float], table: dict[str, Any], stratum: str
) -> list[dict[str, float]]:
    """``[{"alpha", "lo", "hi"}]`` per α, widened by that (α, stratum)'s δ.

    The ONE place a published interval is computed — `predict` writes these into the record
    and the fan readout scores them, so a coverage number and the interval it was measured
    on can never be built by two different formulas.
    """
    values = np.asarray(quantiles_s, dtype=np.float64)
    intervals = []
    for alpha in CONFORMAL_ALPHAS:
        block = table["alphas"][_alpha_key(alpha)]["strata"][stratum]
        low_index, high_index = quantile_pair_indices(alpha)
        low, high = calibrated_interval(
            float(values[low_index]), float(values[high_index]), float(block["delta_s"])
        )
        intervals.append({"alpha": alpha, "lo": low, "hi": high})
    return intervals


# ── the sidecar ─────────────────────────────────────────────────────────────

#: Where the table lives inside `checkpoint_metadata.json`. A SIDECAR key, deliberately not
#: in the checkpoint payload and never inside `data_provenance` — `evaluate-fit` and
#: `freeze-test` compare that object for EQUALITY, and a calibration run would then make
#: every later replay report "the manifest changed".
CONFORMAL_METADATA_KEY = "conformal"


def write_conformal_table(metadata_path: Path, table: dict[str, Any]) -> None:
    """Merge the table into an existing ``checkpoint_metadata.json``, atomically."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("checkpoint_sha256") != table["checkpoint_sha256"]:
        raise ValueError(
            f"{metadata_path} belongs to checkpoint {metadata.get('checkpoint_sha256')!r}, "
            f"the table to {table['checkpoint_sha256']!r}"
        )
    metadata[CONFORMAL_METADATA_KEY] = table
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary.replace(metadata_path)


def load_conformal_table(
    checkpoint_path: str | Path, checkpoint_sha256: str
) -> dict[str, Any] | None:
    """The checkpoint's own calibration table, or None when it has never been calibrated.

    Bound to the checkpoint's digest: a table left behind by an earlier weights file is a
    calibration of a different model, and applying it silently is exactly the failure the
    ``calibrated`` flag exists to make visible. A mismatch RAISES rather than returning
    None — "no table" and "the wrong table" are different situations.
    """
    metadata_path = Path(checkpoint_path).parent / "checkpoint_metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    table = metadata.get(CONFORMAL_METADATA_KEY)
    if table is None:
        return None
    if table.get("schema") != CONFORMAL_SCHEMA:
        raise ValueError(
            f"{metadata_path} carries conformal schema {table.get('schema')!r}; this build "
            f"reads {CONFORMAL_SCHEMA!r} — recalibrate"
        )
    if table.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(
            f"{metadata_path}'s conformal table was fitted on checkpoint "
            f"{table.get('checkpoint_sha256')!r}, not on {checkpoint_sha256!r}; recalibrate"
        )
    return table


def render(table: dict[str, Any]) -> str:
    """The readout text: δ, both halves' realised coverage, and the median width."""
    lines = [
        "B2 split-conformal ETA calibration (CQR) — "
        f"{table['calibration_flights']} {table['split']} flights, halves "
        f"{table['half_flights'][0]}/{table['half_flights'][1]}, split_seed "
        f"{table['split_seed']}",
        "delta widens [q_lo, q_hi] on BOTH sides; coverage is measured on the half the "
        "delta was NOT fitted on (both are printed).",
    ]
    for alpha in CONFORMAL_ALPHAS:
        block = table["alphas"][_alpha_key(alpha)]
        low, high = block["quantile_pair"]
        lines += [
            "",
            f"── alpha {alpha:g} (nominal {(1 - alpha) * 100:.0f}% interval, "
            f"q{low:g}/q{high:g})",
            # The coverage columns are labelled by WHICH half's delta was measured on which
            # other half — "cov A" would read as "the coverage on half A", the one number a
            # split-conformal readout must never publish.
            f"   {'stratum':>46s} {'n A/B':>13s} {'delta s':>9s} {'dA->B':>7s} {'dB->A':>7s} "
            f"{'cov mean':>9s} {'width s':>9s}",
        ]
        for stratum, cell in block["strata"].items():
            first, second = cell["calibration_flights"]
            lines.append(
                f"   {stratum:>46s} {f'{first}/{second}':>13s} {cell['delta_s']:>9.1f} "
                f"{cell['coverage_halves'][0]:>7.3f} {cell['coverage_halves'][1]:>7.3f} "
                f"{cell['coverage']:>9.3f} {cell['median_width_s']:>9.1f}"
            )
        for stratum, counts in block["refused_strata"].items():
            lines.append(
                f"   {stratum:>46s} {f'{counts[0]}/{counts[1]}':>13s} "
                f"{'refused':>9s}  (below {MIN_CALIBRATION_FLIGHTS} in a half)"
            )
    return "\n".join(lines) + "\n"
