"""Split-conformal calibration of the predicted arrival-time interval (B2, design §三 3.2).

The quantile duration head (B1) publishes five levels of ``p(T | history)``. Nothing makes
them CALIBRATED: a head trained by the pinball loss on 6 800 flights is as miscalibrated as
any other fitted quantile, and an 80 % interval that actually covers 62 % is worse than no
interval at all. Conformalized quantile regression (CQR) addresses that with one number per
stratum: the conformity score ``max(q_lo − T, T − q_hi)`` is how far outside the interval
the truth fell (negative when it fell inside), and its ``1 − α`` quantile ``δ_α`` on a
CALIBRATION set is the amount ``[q_lo − δ, q_hi + δ]`` has to grow to cover ``1 − α`` of
EXCHANGEABLE flights.

**What this module measures, and what it does not claim.** The textbook split-conformal
argument gives a finite-sample coverage GUARANTEE when the calibration flights are
exchangeable with the ones the interval will be deployed on, and independent of the fitted
model. **That premise does not hold as constructed here, and the language everywhere in this
package is chosen to say so.** The calibration set is the VALIDATION split, which is also the
split the checkpoint was SELECTED on: the LR schedule, the best epoch and early stopping all
step on a validation metric. Halving val does not repair it — both halves fed the selection.
So what is published is an **empirically measured, cross-half coverage**, not a guarantee.
The coupling is weak and worth naming: selection reads a TRAJECTORY metric (common-grid ADE,
or the anchor-grid mean), never the duration residual these δ are quantiles of. Keeping the
construction is deliberate — the repo's experiment principle is that the outer test split
stays sealed until the ledger stage. **Pre-registered**: the guarantee-bearing number is a
SINGLE held-out read of the coverage on the test split at `freeze-test` time, once every
experiment decision is final; until then no surface in this package calls the coverage
guaranteed.

**Four rules this module exists to hold** (design §六 4 and the repo's experiment
principles):

1. **The calibration set is never the test split and never the training split.** It is the
   VALIDATION split, halved deterministically: half A fits δ, and δ_A **is the deployed
   number**; half B — which the deployed δ was not fitted on — measures what it actually
   covered, and that measurement is the published one. The mirror (δ_B applied to half A) is
   kept beside it as a STABILITY check, never averaged in: a mean of two δ is fitted on data
   that includes every flight it is then scored against, and nothing would measure it.
2. **Calibration is PER STRATUM.** Measured on KRDU val (B0, §〇.2), the vectored |Δt| p80
   is 65.8–72.5 s against straight-in's 12.0–20.3 s: one pooled δ hands every straight-in
   flight an absurd interval. The strata are `approach_difficulty.strata_masks` — the same
   cut every other readout uses — computed once at the L−1 anchor, and only the ones
   `INTERVAL_STRATUM_PRECEDENCE` can actually deploy are fitted at all.
3. **A thin stratum refuses.** Below `MIN_CALIBRATION_FLIGHTS` in either half, the stratum
   gets no δ; it is recorded in ``refused_strata`` with its count and a flight in it falls
   through `INTERVAL_STRATUM_PRECEDENCE` to the pooled one. A δ read off 11 flights is a
   number, not a calibration. **The coverage of that fall-through is measured** — see the
   ``deployed`` block, which is the number the design's gate reads.
4. **No table, no claim.** A checkpoint without one predicts uncalibrated quantiles and the
   record says ``calibrated: false``. The table is bound to `checkpoint_sha256` and to
   `DURATION_QUANTILES`, so a table written for another checkpoint — or under another set of
   levels — is refused rather than applied.

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

from ts_transformer.approach_difficulty import (
    STRATA_COVARIATES,
    STRATUM_ALL,
    STRATUM_STRAIGHT_IN,
    STRATUM_VECTORED,
    strata_masks,
)
from ts_transformer.config import DURATION_QUANTILES

#: The table's own version. Bump it when the meaning of a stored δ changes; `predict`
#: refuses a table it does not recognise rather than applying it under new rules. v2: δ is
#: half A's alone (v1 deployed the mean of both halves, which nothing measured), and the
#: table carries the ``deployed`` block and its own cohort. ``half_seed`` /
#: ``deployed_half_rule`` were ADDED to v2 without a bump (2026-09-08): they say which cut
#: produced the table, no stored δ changes meaning, and no reader of a stored table reads
#: them — bumping would have refused every deployed sidecar over a provenance field.
CONFORMAL_SCHEMA = "ts-conformal-cqr-v2"

#: The miscoverage levels calibrated, and the quantile PAIR each is built from. α = 0.2 is
#: the design's headline 80 % interval (q10, q90); α = 0.5 is the 50 % one (q25, q75). The
#: pair is part of the definition — δ recentres a specific pair, and applying it to another
#: would be calibrating one interval and publishing a different one.
ALPHA_QUANTILE_PAIRS: dict[float, tuple[float, float]] = {
    0.2: (0.1, 0.9),
    0.5: (0.25, 0.75),
}
CONFORMAL_ALPHAS: tuple[float, ...] = tuple(sorted(ALPHA_QUANTILE_PAIRS))

#: Below this, in EITHER half, a stratum is refused (rule 3 above). Both halves are checked
#: even though only half A fits: half B is what MEASURES the deployed δ, and a coverage read
#: off 11 flights is not a measurement either. 30 keeps the conformal level strictly inside
#: the sample at both α (0.8333 and 0.5167 of 30 scores — the 25th and 16th order
#: statistics) rather than running off the end of it.
MIN_CALIBRATION_FLIGHTS = 30

#: The alpha whose CALIBRATED endpoints B3 can decode beside the five levels — the design's
#: headline 80 % interval. Decoding every alpha's endpoints would multiply the fan for a
#: reading nothing asks for.
FAN_INTERVAL_ALPHA = min(CONFORMAL_ALPHAS)

#: Where `predict --cta-from-quantiles` writes the fan, and how one leaf is named. The
#: directory grammar lives HERE rather than in `cli.predict` so `experiments.quantile_fan_readout`
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
#: which HAS a δ. It is a PRECEDENCE rather than a lookup because of the FALL-THROUGH, not
#: because of overlap: straight-in and vectored are disjoint by construction (`strata_masks`
#: cuts them at the same tortuosity), but they do not cover the cohort — a vectored flight
#: that is already established is in neither — and a stratum refused for thinness has no δ of
#: its own. Both cases land on the pooled stratum, and the ``deployed`` block below is what
#: measures whether that landing was any good.
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
    quantile"; the ``(n + 1)`` correction is what the split-conformal construction asks for,
    and without it the interval under-covers by ``≈ 1/n`` on purpose-built data. At the KRDU
    half-split (n ≈ 700 per half) it moves δ by well under a second.

    The level EXCEEDS 1 when ``n`` is small enough that even the largest score is not high
    enough — the conformal answer is then ``+∞`` (no finite δ), and that is raised rather
    than silently clamped to the maximum. `MIN_CALIBRATION_FLIGHTS` = 30 keeps both α this
    module uses clear of it, so this fires only if a caller lowers the floor.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    count = len(scores)
    if count < 1:
        raise ValueError("a conformal quantile needs at least one calibration score")
    level = math.ceil((count + 1) * (1.0 - alpha)) / count
    if level > 1.0:
        raise ValueError(
            f"{count} scores cannot support alpha={alpha:g}: the split-conformal level is "
            f"{level:.4f} > 1, so the conformal delta is +inf. Calibrate on more flights"
        )
    return float(np.quantile(np.asarray(scores, dtype=np.float64), level, method="higher"))


def calibrated_interval(lo_s: float, hi_s: float, delta_s: float) -> tuple[float, float]:
    """``[q_lo − δ, q_hi + δ]``. Refuses an inverted result rather than repairing it.

    δ is negative when the head over-covers, and a δ so negative that it crosses the interval
    means the head's own ``q_lo``/``q_hi`` were further apart than the data ever needed — a
    calibration failure, not a value to publish at the midpoint. Collapsing it silently would
    also disagree with the width and coverage numbers, which are computed from the raw
    arithmetic.
    """
    low, high = lo_s - delta_s, hi_s + delta_s
    if low > high:
        raise ValueError(
            f"delta {delta_s:g} s inverts the interval [{lo_s:g}, {hi_s:g}] s: the fitted "
            "quantile pair is wider than the calibration data ever required. Recalibrate"
        )
    return low, high


def interval_coverage(
    lo_s: np.ndarray, hi_s: np.ndarray, delta_s: float, truth_s: np.ndarray
) -> float:
    """The share of flights whose truth falls in the δ-widened interval."""
    return float(np.mean((truth_s >= lo_s - delta_s) & (truth_s <= hi_s + delta_s)))


#: How the two halves are cut, recorded in the table so a reader can reproduce them. THE
#: deployed rule: the seed hashed is the checkpoint's own ``split_seed``, and only a table
#: cut this way may reach the sidecar (`write_conformal_table` refuses any other).
CALIBRATION_HALF_RULE = (
    "sort flights by sha256('conformal:{split_seed}:{flight_key}'); first half = A (fits "
    "the deployed delta), second half = B (measures its coverage)"
)

#: The SAME cut under a probe seed. A deployed coverage that sits well below its own mirror
#: is either a real property of the cohort or an artefact of the one half rule that was ever
#: tried, and the two cannot be told apart without re-cutting the halves — so
#: `run_ts.py eta_calibration --half-seed` exists, refuses to run without ``--readout-only``,
#: and the table it produces says here, in its own bytes, that it is not deployable.
PROBE_HALF_RULE = (
    "PROBE, NOT THE DEPLOYED RULE: sort flights by "
    "sha256('conformal:{half_seed}:{{flight_key}}') — half_seed {half_seed} instead of this "
    "checkpoint's split_seed {split_seed}; first half = A (fits the delta), second half = B "
    "(measures its coverage). A re-cut of the same cohort, for reading only"
)


def calibration_halves(keys: Sequence[str], half_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two deterministic halves of the calibration cohort, as index arrays.

    Keyed on ``half_seed`` — the run's own ``split_seed`` on every deployable table — and on
    the flight identity, hashed the way `splits.py` hashes the outer split: a DIFFERENT
    question (which half of val), so it carries its own salt and the two partitions are
    independent. Sorted-and-cut rather than thresholded, so the halves are exactly equal
    (± one flight) whatever the cohort size, and independent of the order the caller happened
    to build the flights in. Half A is the first: it fits the deployed δ, B is the held-out
    half (`CALIBRATION_HALF_RULE`). This is the ONE definition of the cut — a probe threads a
    different seed through it and never a second hash.
    """
    order = np.argsort([
        hashlib.sha256(f"conformal:{half_seed}:{key}".encode()).hexdigest()
        for key in keys
    ], kind="stable")
    cut = len(order) // 2
    return order[:cut], order[cut:]


def _stratum_block(
    quantiles: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    halves: tuple[np.ndarray, np.ndarray],
    alpha: float,
) -> dict[str, Any]:
    """One (α, stratum) cell: the DEPLOYED δ (half A's) and the coverage half B realised."""
    low_index, high_index = quantile_pair_indices(alpha)
    fit_part, held_out = (np.intersect1d(half, np.flatnonzero(mask)) for half in halves)
    counts = [int(len(fit_part)), int(len(held_out))]
    if min(counts) < MIN_CALIBRATION_FLIGHTS:
        return {"refused": True, "calibration_flights": counts}

    def fit(part: np.ndarray) -> float:
        return conformal_delta(
            conformity_scores(
                quantiles[part, low_index], quantiles[part, high_index], truth[part]
            ),
            alpha,
        )

    def measure(delta: float, part: np.ndarray) -> tuple[float, float]:
        low, high = quantiles[part, low_index], quantiles[part, high_index]
        return (
            interval_coverage(low, high, delta, truth[part]),
            float(np.median(high - low + 2.0 * delta)),
        )

    delta = fit(fit_part)
    coverage, width = measure(delta, held_out)
    # The mirror is a STABILITY check, not a second estimate to average in: if the two halves
    # disagree by more than the interval width cares about, the δ is not settled.
    stability_delta = fit(held_out)
    stability_coverage, _width = measure(stability_delta, fit_part)
    return {
        "refused": False,
        "calibration_flights": counts,
        # THE deployed number: fitted on half A alone, so half B is genuinely held out from
        # it and the coverage below is a measurement rather than a restatement of the level.
        "delta_s": float(delta),
        "coverage": float(coverage),
        "median_width_s": float(width),
        "stability_delta_s": float(stability_delta),
        "stability_coverage": float(stability_coverage),
    }


def _deployed_block(
    quantiles: np.ndarray,
    truth: np.ndarray,
    masks: dict[str, np.ndarray],
    strata: dict[str, dict[str, Any]],
    held_out: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    """What the flights actually GET, and whether it covers them.

    The per-stratum block above measures each δ on its own stratum's members. Deployment does
    not work that way: a flight takes the first stratum in `INTERVAL_STRATUM_PRECEDENCE` it
    belongs to AND that has a δ, so a flight whose own stratum refused is handed the pooled
    δ — and nothing in the per-stratum table says whether the pooled δ covers THOSE flights.
    It need not: the pooled δ is a mixture, and the fall-through group is by construction the
    part of the cohort the mixture is least like. This block assigns every held-out flight the
    δ deployment would give it and measures that assignment, pooled and broken out by
    ``natural -> assigned`` group. **The design's gate 3.4-2 reads this number**, not the
    per-stratum rows.
    """
    low_index, high_index = quantile_pair_indices(alpha)
    natural, assigned, deltas = [], [], []
    for row in held_out:
        first = next(
            (name for name in INTERVAL_STRATUM_PRECEDENCE if masks[name][row]), STRATUM_ALL
        )
        taken = next(
            (name for name in INTERVAL_STRATUM_PRECEDENCE
             if masks[name][row] and name in strata),
            STRATUM_ALL,
        )
        natural.append(first)
        assigned.append(taken)
        deltas.append(strata[taken]["delta_s"])
    delta = np.array(deltas, dtype=np.float64)
    low, high = quantiles[held_out, low_index], quantiles[held_out, high_index]
    covered = (truth[held_out] >= low - delta) & (truth[held_out] <= high + delta)
    widths = high - low + 2.0 * delta
    labels = np.array([
        first if first == taken else f"{first} -> {taken}"
        for first, taken in zip(natural, assigned)
    ])
    return {
        "measured_on": "the held-out half, under the delta each flight's own precedence gives it",
        "flights": int(len(held_out)),
        "coverage": float(np.mean(covered)),
        "median_width_s": float(np.median(widths)),
        "groups": {
            str(label): {
                "flights": int(np.count_nonzero(labels == label)),
                "coverage": float(np.mean(covered[labels == label])),
                "median_width_s": float(np.median(widths[labels == label])),
                "fell_through": " -> " in str(label),
            }
            for label in sorted(set(labels.tolist()))
        },
    }


def calibrate(
    samples: Sequence[CalibrationSample],
    *,
    split_seed: int,
    split: str,
    checkpoint_sha256: str,
    airports: Sequence[str] = (),
    limit: int = 0,
    half_seed: int | None = None,
) -> dict[str, Any]:
    """The conformal table: α → stratum → δ, the coverage it realised, and what refused.

    Only the strata `INTERVAL_STRATUM_PRECEDENCE` can deploy are fitted: a δ for a stratum no
    record will ever read is a number in the artifact that nothing measures. Refuses outright
    only when the POOLED stratum is too thin — there is then nothing to calibrate and nothing
    to fall through to.

    ``airports`` and ``limit`` are the table's own COHORT: a δ fitted at one airport and
    applied at another is a different calibration, and a ``--limit`` smoke table has to be
    visible as one everywhere it travels.

    ``half_seed`` defaults to ``split_seed``, which is `CALIBRATION_HALF_RULE` — the one rule
    a deployable table may be cut by. Anything else is a PROBE of the half split itself: the
    table says so in ``deployed_half_rule`` and ``half_rule``, and `write_conformal_table`
    refuses it. It exists so "the deployed coverage sits below its mirror" can be read as a
    property of the cohort or of the cut, which one fixed seed cannot answer.
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
    quantiles = np.stack([sample.quantiles_s for sample in samples])
    truth = np.array([sample.truth_final_time_s for sample in samples], dtype=np.float64)
    half_seed = split_seed if half_seed is None else int(half_seed)
    deployed_half_rule = half_seed == split_seed
    fit_half, held_out = calibration_halves(keys, half_seed)
    alphas: dict[str, Any] = {}
    for alpha in CONFORMAL_ALPHAS:
        blocks = {
            stratum: _stratum_block(
                quantiles, truth, masks[stratum], (fit_half, held_out), alpha
            )
            for stratum in INTERVAL_STRATUM_PRECEDENCE
        }
        if blocks[STRATUM_ALL]["refused"]:
            raise ValueError(
                f"{len(samples)} calibration flights cannot support the pooled stratum "
                f"(each half needs {MIN_CALIBRATION_FLIGHTS}); there is nothing to calibrate"
            )
        strata = {
            stratum: block for stratum, block in blocks.items() if not block["refused"]
        }
        alphas[_alpha_key(alpha)] = {
            "quantile_pair": list(ALPHA_QUANTILE_PAIRS[alpha]),
            "strata": strata,
            "refused_strata": {
                stratum: block["calibration_flights"]
                for stratum, block in blocks.items() if block["refused"]
            },
            "deployed": _deployed_block(
                quantiles, truth, masks, strata, held_out, alpha
            ),
        }
    return {
        "schema": CONFORMAL_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "split": split,
        "split_seed": split_seed,
        "quantiles": list(DURATION_QUANTILES),
        "min_calibration_flights": MIN_CALIBRATION_FLIGHTS,
        "calibration_flights": len(samples),
        "half_flights": [int(len(fit_half)), int(len(held_out))],
        # The seed the halves were actually cut with, and whether that is the deployed rule.
        # Every number below is a function of this cut, so it travels with them.
        "half_seed": half_seed,
        "deployed_half_rule": deployed_half_rule,
        "half_rule": CALIBRATION_HALF_RULE if deployed_half_rule else PROBE_HALF_RULE.format(
            half_seed=half_seed, split_seed=split_seed
        ),
        "interval_stratum_precedence": list(INTERVAL_STRATUM_PRECEDENCE),
        # The cohort this table belongs to. `limit` non-zero means it was fitted on a PREFIX
        # of the split, which is a smoke test and is marked as one everywhere it travels.
        "airports": [str(name) for name in airports],
        "limit": int(limit),
        "smoke_test": bool(limit),
        # Said in the artifact, not only in the module docstring: the calibration split is
        # also the model-selection split, so this is a measurement, not a guarantee.
        "coverage_claim": (
            "empirically measured cross-half coverage; the calibration split is also the "
            "model-selection split, so this is not a finite-sample guarantee. The coupling "
            "is through a trajectory metric (common-grid ADE), not the duration residual. "
            "The guarantee-bearing read is a single held-out test-split measurement at the "
            "freeze-test ledger stage."
        ),
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
    # Calibrated at EVERY alpha, not at any: `conformal_intervals` reads the stratum's δ
    # under each alpha in turn, and a stratum that calibrated at one level only would
    # raise there (review C-15). Today `_stratum_block` refuses on counts alone, so the
    # union and the intersection coincide; the intersection is the one that stays right.
    calibrated = set.intersection(*(
        set(block["strata"]) for block in table["alphas"].values()
    ))
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


def write_conformal_table(
    metadata_path: Path, table: dict[str, Any], *, allow_smoke: bool = False
) -> None:
    """Merge the table into an existing ``checkpoint_metadata.json``, atomically.

    A SMOKE table (``--limit``) is refused unless the caller says otherwise: the sidecar is
    what `predict` deploys, and a δ fitted on the first 200 flights of the split would
    otherwise widen every record of a full run with nothing at the write site saying so.

    A PROBE table (a ``half_seed`` other than the checkpoint's ``split_seed``) is refused
    outright, with no escape hatch: a deployed δ comes from `CALIBRATION_HALF_RULE` and from
    nothing else, so re-cutting the halves is a reading, never a deployment.
    """
    if not table["deployed_half_rule"]:
        raise ValueError(
            f"this table was cut with half_seed {table['half_seed']}, not the checkpoint's "
            f"split_seed {table['split_seed']}: it is a PROBE of the half rule, and a "
            "deployed delta comes from the documented rule alone. Re-run without "
            "--half-seed to deploy, or keep it as --readout-only"
        )
    if table["smoke_test"] and not allow_smoke:
        raise ValueError(
            f"this table was fitted on a --limit {table['limit']} PREFIX of the "
            f"{table['split']} split ({table['calibration_flights']} flights): it is a smoke "
            "test, and writing it into the checkpoint's sidecar would deploy it. Re-run "
            "without --limit, or pass --allow-smoke-table deliberately"
        )
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

    Bound to the checkpoint's digest AND to the quantile levels: a table left behind by an
    earlier weights file is a calibration of a different model, and one written under
    different `DURATION_QUANTILES` indexes different columns of the head's output. Applying
    either silently is exactly the failure the ``calibrated`` flag exists to make visible. A
    mismatch RAISES rather than returning None — "no table" and "the wrong table" are
    different situations.
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
    # The levels are a MIRROR of `DURATION_QUANTILES` inside the artifact, and a mirror that
    # is written and never checked is a version the artifact cannot verify.
    if list(table.get("quantiles", ())) != list(DURATION_QUANTILES):
        raise ValueError(
            f"{metadata_path}'s conformal table was fitted on quantile levels "
            f"{table.get('quantiles')}, this build's head emits {list(DURATION_QUANTILES)} — "
            "the deltas index different columns; recalibrate"
        )
    return table


def render(table: dict[str, Any]) -> str:
    """The readout text: the deployed δ, the coverage it realised, and what deployment gets."""
    lines = [
        "B2 split-conformal ETA calibration (CQR) — "
        f"{table['calibration_flights']} {table['split']} flights, halves "
        f"{table['half_flights'][0]}/{table['half_flights'][1]}, split_seed "
        f"{table['split_seed']}, half_seed {table['half_seed']}"
        + (f", airports {'/'.join(table['airports'])}" if table["airports"] else ""),
        f"half rule: {table['half_rule']}",
        "delta widens [q_lo, q_hi] on BOTH sides. The DEPLOYED delta is half A's; 'coverage' "
        "is measured on half B, which it was not fitted on. 'stab' repeats the fit on B and "
        "scores it on A — a stability check, never averaged in.",
        "NOT a finite-sample guarantee: the calibration split is also the split this "
        "checkpoint was selected on (see calibration.py). The guarantee-bearing read is a "
        "single held-out test-split measurement at the freeze-test stage.",
    ]
    if table["smoke_test"]:
        lines.insert(0, f"SMOKE TEST: --limit {table['limit']}, a PREFIX of the split — not "
                        "a calibration of the cohort")
    if not table["deployed_half_rule"]:
        lines.insert(0, f"PROBE HALF RULE: halves cut with half_seed {table['half_seed']}, "
                        f"NOT this checkpoint's split_seed {table['split_seed']} — a reading "
                        "of how much the numbers below depend on the cut. This table is not "
                        "deployable and the sidecar refuses it")
    for alpha in CONFORMAL_ALPHAS:
        block = table["alphas"][_alpha_key(alpha)]
        low, high = block["quantile_pair"]
        lines += [
            "",
            f"── alpha {alpha:g} (nominal {(1 - alpha) * 100:.0f}% interval, "
            f"q{low:g}/q{high:g})",
            f"   {'stratum':>46s} {'n A/B':>13s} {'delta s':>9s} {'coverage':>9s} "
            f"{'width s':>9s} {'stab d':>8s} {'stab cov':>9s}",
        ]
        for stratum, cell in block["strata"].items():
            first, second = cell["calibration_flights"]
            lines.append(
                f"   {stratum:>46s} {f'{first}/{second}':>13s} {cell['delta_s']:>9.1f} "
                f"{cell['coverage']:>9.3f} {cell['median_width_s']:>9.1f} "
                f"{cell['stability_delta_s']:>8.1f} {cell['stability_coverage']:>9.3f}"
            )
        for stratum, counts in block["refused_strata"].items():
            lines.append(
                f"   {stratum:>46s} {f'{counts[0]}/{counts[1]}':>13s} "
                f"{'refused':>9s}  (below {MIN_CALIBRATION_FLIGHTS} in a half)"
            )
        deployed = block["deployed"]
        lines += [
            "",
            f"   DEPLOYED — {deployed['measured_on']}.",
            "   GATE 3.4-2 READS THIS COVERAGE, not the per-stratum rows above.",
            f"   {'assignment':>46s} {'n':>13s} {'coverage':>9s} {'width s':>9s}",
            f"   {'(pooled over every held-out flight)':>46s} {deployed['flights']:>13d} "
            f"{deployed['coverage']:>9.3f} {deployed['median_width_s']:>9.1f}",
        ]
        for label, cell in deployed["groups"].items():
            mark = "  *fell through" if cell["fell_through"] else ""
            lines.append(
                f"   {label:>46s} {cell['flights']:>13d} "
                f"{cell['coverage']:>9.3f} {cell['median_width_s']:>9.1f}{mark}"
            )
    return "\n".join(lines) + "\n"
