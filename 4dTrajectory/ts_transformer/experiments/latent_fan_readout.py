#!/usr/bin/env python
"""4(a): read the latent model's SAMPLE fan as a distribution, by the B line's protocol.

`run_ts.py quantile_fan_readout` asks of a quantile fan: does the truth path lie closer to
the NEAREST leaf than to the single decode the arm would deliver? This runner asks exactly
that of a latent arm's samples, so the latent fan and the quantile fan become the same
deliverable measured the same way (programme report 2026-09-08, decision 4(a)). The
chamfer-to-nearest-leaf logic is imported, not restated: a fan read two ways is two
different measurements wearing one name.

A `predict --latent-samples K [--latent-random K]` directory nests the same shape as a
quantile fan::

    <arm>/                 the top-1 decode — the prior's most likely component MEAN, what
                           the arm delivers
    <arm>/modes/modeNN/    K draws from the same context-conditioned prior: the fan under test
    <arm>/random/modeNN/   K N(0, I) draws — latents the prior did NOT choose — decoded by
                           the same checkpoint: the CONTROL
    <arm>/shuffled/        the shuffled-z diagnostic (read by run_ts.py latent_readout)

Per `approach_difficulty.strata_masks` stratum this prints, for the prior fan and for the
random fan alike:

* the truth path's chamfer to the nearest leaf, against its chamfer to the top-1 decode,
  and the share of flights on which the nearest leaf wins;
* minADE_K over ``{top-1, modes}`` and the top-1 ADE, computed from the same records — a
  cross-check against the arm's `latent_readout_*.json`, which defines minADE_K that way;
* the fan's lateral spread: the median over flights of the LARGEST pairwise distance
  between two of the K decoded endpoints.

**The random fan is the whole point.** A fan of samples always contains something nearer
the truth than its own mean does, purely because it is a fan — so a nearest-leaf number
alone measures nothing. The reading is the comparison: if the prior fan does not beat K
draws the prior did not choose, its samples carry nothing about the flight in front of the
anchor, whatever the KL says. Read the control for what it is (``forecast.py``): the prior
is context-conditioned and, once trained, need not sit near N(0, I) at all, so the control
is "K chances at a latent the prior did not choose" and not a same-distribution shuffle.
This is a readout, not a coverage guarantee: K trajectories are not a distribution over
trajectories.

    conda activate aeroviz
    python run_ts.py latent_fan_readout --arm <pred_dir> --json <out>/latent_fan.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ts_transformer.experiments.support import REPO_ROOT

import ts_transformer.geometric_metrics as gm  # noqa: E402
from ts_transformer.approach_difficulty import strata_masks  # noqa: E402
# One implementation of "what a leaf contributes to a fan readout" and of the
# chamfer-to-nearest-leaf cell, shared with the B line's quantile fan.
from ts_transformer.experiments.quantile_fan_readout import (  # noqa: E402
    FAN_REQUIRED_FIELDS,
    cell,
    geometry_cell,
    leaf_geometry,
    leaf_rows,
    median,
)
# ...and one definition of where the samples live, shared with the latent readout whose
# minADE_K this cross-checks.
from ts_transformer.experiments.latent_readout import mode_dirs  # noqa: E402

RESULT_SCHEMA = "ts-latent-fan-readout-v1"

MODES_KIND = "modes"
RANDOM_KIND = "random"
# The latent arm's fan is read against its top-1 decode (the prior mean), not against a
# quantile; the cell's reference column says so in the JSON as well as in the table.
TOP1_REFERENCE_KEY = "chamfer_top1_p50_m"
# The chamfer needs the truth duration and the strata need the covariates (both in
# FAN_REQUIRED_FIELDS); the ADE cross-check needs the point metrics as well.
LATENT_REQUIRED_FIELDS = (*FAN_REQUIRED_FIELDS, "ade_m", "fde_m")


def _leaf_fan(
    leaves: list[Path], keys: list[str], *, geometry_truth: str
) -> dict[str, np.ndarray]:
    """``[K, F]`` chamfer and ADE, and ``[K, F, 2]`` endpoints, over a fan's leaves."""
    read = [
        leaf_geometry(leaf, keys, geometry_truth=geometry_truth,
                      required=LATENT_REQUIRED_FIELDS)
        for leaf in leaves
    ]
    return {
        "chamfer_m": np.stack([leaf["chamfer_m"] for leaf in read]),
        "ade_m": np.stack([
            np.array([row["ade_m"] for row in leaf["rows"]], dtype=np.float64)
            for leaf in read
        ]),
        "endpoint_en": np.stack([leaf["endpoint_en"] for leaf in read]),
    }


def endpoint_spread_m(endpoints: np.ndarray) -> np.ndarray:
    """Per flight, the largest distance between two of the fan's decoded endpoints.

    ``endpoints`` is ``[K, F, 2]`` in one chart per flight (each leaf decodes the same
    flight, so the threshold-anchored charts coincide). A single leaf has no spread.
    """
    if endpoints.shape[0] < 2:
        return np.zeros(endpoints.shape[1])
    pairwise = np.linalg.norm(endpoints[:, None] - endpoints[None, :], axis=-1)  # [K, K, F]
    return pairwise.reshape(-1, endpoints.shape[1]).max(axis=0)


def readout(arm: Path, *, geometry_truth: str) -> dict:
    rows, coverage, split = leaf_rows(arm, required=LATENT_REQUIRED_FIELDS)
    keys = sorted(rows)
    if not keys:
        raise SystemExit(
            f"{arm} has no rows carrying a truth duration, the strata covariates and ADE/FDE"
        )
    leaves = mode_dirs(arm, MODES_KIND)
    if not leaves:
        raise SystemExit(
            f"{arm} has no {MODES_KIND}/ leaves, so it holds no sample fan to read. "
            "Predict it with `predict --latent-samples K` (and `--latent-random K` for the "
            "control this readout compares against)"
        )
    control_leaves = mode_dirs(arm, RANDOM_KIND)

    top1 = leaf_geometry(arm, keys, geometry_truth=geometry_truth,
                         required=LATENT_REQUIRED_FIELDS)
    top1_chamfer = top1["chamfer_m"]
    top1_ade = np.array([row["ade_m"] for row in top1["rows"]], dtype=np.float64)

    prior = _leaf_fan(leaves, keys, geometry_truth=geometry_truth)
    control = (
        _leaf_fan(control_leaves, keys, geometry_truth=geometry_truth)
        if control_leaves else None
    )
    spread = endpoint_spread_m(prior["endpoint_en"])
    # minADE_K over {top-1, modes} — the definition run_ts.py latent_readout reports, so the
    # two artifacts are comparable row for row.
    min_ade = np.minimum(top1_ade, prior["ade_m"].min(axis=0))
    prior_nearest = prior["chamfer_m"].min(axis=0)
    control_nearest = None if control is None else control["chamfer_m"].min(axis=0)

    strata = {}
    for stratum, mask in strata_masks(rows, keys).items():
        idx = np.flatnonzero(mask)
        if not len(idx):
            continue
        strata[stratum] = {
            "n": int(len(idx)),
            # Both cells carry the same top-1 reference on purpose: each one is a complete
            # "fan against the decode it would replace" reading, and the comparison between
            # them is the measurement.
            "prior_fan": geometry_cell(top1_chamfer[idx], prior_nearest[idx],
                                       reference_key=TOP1_REFERENCE_KEY),
            "random_fan": (
                None if control_nearest is None
                else geometry_cell(top1_chamfer[idx], control_nearest[idx],
                                   reference_key=TOP1_REFERENCE_KEY)
            ),
            "top1_ade_mean_m": float(np.mean(top1_ade[idx])),
            "min_ade_mean_m": float(np.mean(min_ade[idx])),
            "endpoint_spread_p50_m": median(spread[idx]),
        }
    return {
        "arm": str(arm),
        "split": split,
        "geometry_truth": geometry_truth,
        "flights": len(keys),
        "coverage": coverage,
        "modes": len(leaves),
        "random_modes": len(control_leaves),
        "strata": strata,
    }


def verdict(payload: dict, stratum: str = "all") -> str:
    """Whether the prior fan beats its own N(0, I) control on the pooled stratum — the one
    sentence the readout exists to produce."""
    block = payload["strata"][stratum]
    prior, control = block["prior_fan"], block["random_fan"]
    if control is None:
        return (f"no {RANDOM_KIND}/ leaves: the prior fan has no control, so its "
                "nearest-leaf numbers say nothing about information in z")
    # A tie counts as not beating: the claim being tested is that the prior's own samples
    # carry MORE than latents it did not choose.
    nearer = prior["chamfer_nearest_p50_m"] < control["chamfer_nearest_p50_m"]
    wins_more = prior["nearest_better_share"] > control["nearest_better_share"]
    if nearer and wins_more:
        reading = "the prior fan BEATS the random control on both numbers"
    elif not nearer and not wins_more:
        reading = "the prior fan does NOT beat the random control on either number"
    else:
        reading = ("the prior fan beats the random control on "
                   f"{'the chamfer' if nearer else 'the win share'} only")
    return (
        f"{stratum}: nearest-leaf chamfer {prior['chamfer_nearest_p50_m']:.0f} m (prior) vs "
        f"{control['chamfer_nearest_p50_m']:.0f} m (random), better-than-top-1 share "
        f"{prior['nearest_better_share']:.3f} vs {control['nearest_better_share']:.3f} — "
        f"{reading}"
    )


def render(payload: dict) -> str:
    modes = payload["modes"]
    coverage = payload["coverage"]
    lines = [
        f"latent fan — {payload['arm']} ({payload['flights']} flights, {modes} prior modes, "
        f"{payload['random_modes']} random modes, split {payload['split']})",
        f"coverage: {coverage['scored_rows']} of {coverage['summary_rows']} summary rows "
        f"scored ({coverage['dropped_unscored_rows']} dropped for a missing field)",
        "the fan columns are a readout, not a coverage guarantee: "
        f"{modes} trajectories are not a distribution over trajectories",
        "",
        f"   {'stratum':>46s} {'n':>5s} {'cham top1':>10s} "
        f"{'prior min':>10s} {'p<top1':>7s} {'rand min':>9s} {'r<top1':>7s} "
        f"{'top1 ADE':>9s} {f'minADE_{modes}':>9s} {'spread':>8s}",
    ]
    for stratum, block in payload["strata"].items():
        prior, control = block["prior_fan"], block["random_fan"] or {}
        lines.append(
            f"   {stratum:>46s} {block['n']:>5d} "
            f"{cell(prior[TOP1_REFERENCE_KEY], '.0f'):>10s} "
            f"{cell(prior['chamfer_nearest_p50_m'], '.0f'):>10s} "
            f"{cell(prior['nearest_better_share'], '.3f'):>7s} "
            f"{cell(control.get('chamfer_nearest_p50_m'), '.0f'):>9s} "
            f"{cell(control.get('nearest_better_share'), '.3f'):>7s} "
            f"{block['top1_ade_mean_m']:>9.1f} {block['min_ade_mean_m']:>9.1f} "
            f"{cell(block['endpoint_spread_p50_m'], '.0f'):>8s}"
        )
    lines += [
        "",
        "cham top1 = truth's chamfer to the delivered decode (p50 m); prior/rand min = its "
        f"chamfer to the NEAREST of the {modes} leaves; p<top1 / r<top1 = share of flights "
        "the nearest leaf beats top-1 on; spread = p50 of the largest pairwise distance "
        "between two prior endpoints; ADE columns are means, minADE over {top-1, modes} — "
        "the same definition as run_ts.py latent_readout, and a cross-check against it.",
        verdict(payload),
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--arm", type=Path, required=True,
                        help="a latent arm's top-1 prediction directory; its modes/ leaves "
                             "are the fan and its random/ leaves the control")
    parser.add_argument("--geometry-truth", choices=gm.GEOMETRY_TRUTHS,
                        default=gm.GEOMETRY_TRUTH_CLOSED,
                        help="truth the time-free metrics are read against "
                             "(see geometric_metrics)")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the block here as well; must not exist (an immutable "
                             "artifact)")
    args = parser.parse_args(argv)
    arm = args.arm if args.arm.is_absolute() else REPO_ROOT / args.arm
    payload = {"schema": RESULT_SCHEMA, **readout(arm, geometry_truth=args.geometry_truth)}
    print(render(payload), end="")
    if args.json is not None:
        out = args.json if args.json.is_absolute() else REPO_ROOT / args.json
        if out.exists():
            raise FileExistsError(f"{out} exists; a readout is an immutable artifact")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
