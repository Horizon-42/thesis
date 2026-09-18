"""The tokenizer readouts (plan §3.2 "分词器" row, §3.3 gate T): what a code buys the executor,
and how the codes are used — measured, never guessed from a reconstruction error.

Everything here is a pure measurement over loaded models and built series; the runner
(`experiments/manoeuvre_readout.py`) does the loading, the cohort build and the writing.

* `fixed_anchor_readings`: every flight of the cohort at the arm's fixed anchor
  (`config.default_anchor`, 59 in P1.4), flown for the arm's horizon Δ under PROTOCOL C — the
  truth segment through the arm's own tokenizer (reads the future; that is the tokenizer's
  job) — and read as ADE[0, Δ] on the 1 s grid (`inference/receding.mean_displacement_to`,
  `lead_time_error`'s accounting), e(Δ), the code flown, and the difficulty covariates the
  strata are cut on (`data/approach_difficulty`). A flight whose OBSERVED track ends before
  Δ after the anchor reads None and is counted (``truth_shorter_than_horizon``): the cohort is
  admitted on the supervision rows, the reading is against the observed ones.
* `paired_gain`: an arm against its NO-TOKEN twin (same recipe, same seed, plan off), flight by
  flight over the flights both hold — the p50 of (twin − arm) is "the L1 gain", positive when
  the code helps — with the arm-better share beside it. Never a p value (CLAUDE.md: read
  magnitudes).
* `code_usage`: the histogram of the codes flown over the cohort — used / unused, the largest
  share, the entropy — gate T(iii)'s reading.
* `gate_t`: T(i) and T(iii) over the arms of one segment length, TWO seeds each, and the K
  rule (the smallest passing K whose mean gain is within `GATE_T_K_TOLERANCE_M` of the best).
  T(ii) (the prior's predictability) is P2's and is not judged here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from ts_transformer.config import PLAN_CONDITIONING_OFF, TSConfig, default_anchor
from ts_transformer.data.approach_difficulty import (
    STRATUM_ALL,
    STRATUM_ESTABLISHED,
    STRATUM_SHORT,
    STRATUM_STRAIGHT_IN,
    STRATUM_VECTORED,
    approach_difficulty,
    strata_masks,
)
from ts_transformer.data.dataset import FlightSeries, Normalizer
from ts_transformer.data.time_grids import ROW_TOLERANCE_S
from ts_transformer.inference.receding import displacement_at, mean_displacement_to
from ts_transformer.outputs.control.forecast import forecast_control_batch
from ts_transformer.outputs.control.plan_token import manoeuvre_code_count

READOUT_SCHEMA = "ts-manoeuvre-readout-v1"
#: Gate T (plan §3.3), the pre-registered numbers. (i): the truth-code arm's ADE[0, Δ] p50 sits
#: below the no-token twin's by at least this, on BOTH seeds — the v2 §10.8 line for "the
#: head used the token". (iii): the largest code share and the unused-code share.
GATE_T_GAIN_M = 30.0
GATE_T_MAX_CODE_SHARE = 0.25
GATE_T_UNUSED_CODE_SHARE = 0.10
#: The K rule: among the K that pass (i) and (iii), the smallest whose mean gain over the two
#: seeds is within this of the best — the smaller K is the more intent-like one (plan §2.4).
GATE_T_K_TOLERANCE_M = 10.0
#: The strata a table prints, in order.
STRATA = (STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, STRATUM_ESTABLISHED)


def _p50(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


# ── one arm, one cohort ──────────────────────────────────────────────────────

def fixed_anchor_readings(
    model: nn.Module, config: TSConfig, normalizer: Normalizer, series: Sequence[FlightSeries],
    device: torch.device, *, batch_size: int,
) -> dict[str, dict[str, Any]]:
    """Per flight (keyed by ``dataset_id``) at the fixed anchor: ``ade_m`` (ADE[0, Δ], None when
    the observed track is shorter than Δ), ``end_error_m`` (e(Δ), None likewise), ``code`` /
    ``code_source`` (None on a no-token arm) and the strata covariates. Every flight handed in
    must be admissible at the anchor (≥ Δ of supervision after it); the runner builds the
    cohort under the arm's own config, which is what guarantees it."""
    anchor = default_anchor(config)
    horizon = float(config.control_horizon_s)
    if not horizon:
        raise ValueError("the tokenizer readout reads a fixed-horizon executor (control_horizon_s > 0)")
    rows: dict[str, dict[str, Any]] = {}
    for start in range(0, len(series), batch_size):
        batch = list(series[start : start + batch_size])
        forecasts = forecast_control_batch(model, batch, config, normalizer, anchor, device)
        for item, forecast in zip(batch, forecasts, strict=True):
            anchor_time = float(item.times[anchor])
            observed_reaches = float(item.times[-1]) - anchor_time >= horizon - ROW_TOLERANCE_S
            difficulty = approach_difficulty(item, anchor)
            rows[item.dataset_id] = {
                "flight_id": item.flight_id,
                "anchor": anchor,
                "ade_m": mean_displacement_to(item, forecast, anchor, horizon) if observed_reaches else None,
                "end_error_m": displacement_at(item, forecast, anchor, anchor_time + horizon),
                "truth_shorter_than_horizon": not observed_reaches,
                "code": forecast.manoeuvre_code,
                "code_source": forecast.manoeuvre_code_source,
                "route_tortuosity": difficulty.route_tortuosity,
                "established_at_anchor": difficulty.established_at_anchor,
                "remaining_path_m": difficulty.remaining_path_m,
            }
    return rows


def code_usage(rows: dict[str, dict[str, Any]], code_count: int) -> dict[str, Any]:
    """The codes flown over the cohort: counts, used / unused, the largest share, the entropy,
    and whether gate T(iii) holds. ``code_count`` is K (`plan_token.manoeuvre_code_count`)."""
    codes = [row["code"] for row in rows.values() if row["code"] is not None]
    if not codes:
        raise ValueError("no flight carries a code: this is a no-token arm")
    counts = np.bincount(np.asarray(codes, dtype=np.int64), minlength=code_count)
    shares = counts[counts > 0] / counts.sum()
    used = int((counts > 0).sum())
    return {
        "count": int(code_count),
        "flights": int(counts.sum()),
        "used": used,
        "unused": int(code_count - used),
        "unused_share": float((code_count - used) / code_count),
        "max_share": float(shares.max()),
        "entropy_bits": float(-(shares * np.log2(shares)).sum()),
        "counts": counts.tolist(),
        "gate_t_iii": bool(shares.max() <= GATE_T_MAX_CODE_SHARE and (code_count - used) / code_count <= GATE_T_UNUSED_CODE_SHARE),
    }


def stratum_summary(rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """ADE[0, Δ] and e(Δ) per stratum over the flights that have a reading."""
    keys = [key for key, row in rows.items() if row["ade_m"] is not None]
    masks = strata_masks(rows, keys)
    out: dict[str, dict[str, Any]] = {}
    for stratum in STRATA:
        chosen = [key for key, keep in zip(keys, masks[stratum]) if keep]
        if not chosen:
            out[stratum] = {"n": 0}
            continue
        out[stratum] = {
            "n": len(chosen),
            "ade_p50_m": _p50([rows[key]["ade_m"] for key in chosen]),
            "ade_mean_m": float(np.mean([rows[key]["ade_m"] for key in chosen])),
            "end_error_p50_m": _p50([rows[key]["end_error_m"] for key in chosen if rows[key]["end_error_m"] is not None]),
        }
    out["truth_shorter_than_horizon"] = int(sum(row["truth_shorter_than_horizon"] for row in rows.values()))
    return out


def paired_gain(arm_rows: dict[str, dict[str, Any]], twin_rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The arm against its no-token twin, flight by flight, per stratum: ``gain_p50_m`` = the p50
    of (twin ADE − arm ADE) — positive when the code helps — and the arm-better share."""
    keys = [key for key, row in arm_rows.items()
            if row["ade_m"] is not None and key in twin_rows and twin_rows[key]["ade_m"] is not None]
    if not keys:
        raise ValueError("the arm and its twin share no flight with a reading")
    masks = strata_masks(arm_rows, keys)
    out: dict[str, dict[str, Any]] = {}
    for stratum in STRATA:
        chosen = [key for key, keep in zip(keys, masks[stratum]) if keep]
        if not chosen:
            out[stratum] = {"n": 0}
            continue
        deltas = np.array([twin_rows[key]["ade_m"] - arm_rows[key]["ade_m"] for key in chosen])
        out[stratum] = {
            "n": len(chosen),
            "gain_p50_m": float(np.median(deltas)),
            "gain_mean_m": float(deltas.mean()),
            "arm_better_share": float((deltas > 0.0).mean()),
        }
    return out


# ── gate T over one segment length ───────────────────────────────────────────

@dataclass(frozen=True)
class ArmReading:
    """One arm's readings: ``kind`` = ``no-token`` | ``learned`` | ``command-vocabulary``,
    ``k`` = the vocabulary size (None on a no-token arm), ``seed``."""

    key: str
    kind: str
    k: int | None
    seed: int
    rows: dict[str, dict[str, Any]]
    usage: dict[str, Any] | None
    summary: dict[str, dict[str, Any]]


def arm_reading(key: str, config: TSConfig, rows: dict[str, dict[str, Any]]) -> ArmReading:
    no_token = config.plan_conditioning == PLAN_CONDITIONING_OFF
    kind = "no-token" if no_token else config.manoeuvre_tokenizer
    return ArmReading(
        key=key, kind=kind, k=None if no_token else manoeuvre_code_count(config), seed=int(config.seed),
        rows=rows, usage=None if no_token else code_usage(rows, manoeuvre_code_count(config)),
        summary=stratum_summary(rows),
    )


def gate_t(arms: Sequence[ArmReading]) -> dict[str, Any]:
    """Gate T (i) and (iii) over the arms of ONE segment length, and the K rule.

    Every token arm is paired with the no-token twin of its OWN seed; an arm without a twin
    reads ``pending``. (i) holds when the all-stratum gain p50 ≥ `GATE_T_GAIN_M` on both seeds;
    (iii) when `code_usage` passes on both; a vocabulary (learned K, or the command vocabulary)
    passes T when both hold on both seeds. ``selected_k``: the smallest passing learned K whose
    mean gain is within `GATE_T_K_TOLERANCE_M` of the best passing K's.
    """
    twins = {arm.seed: arm for arm in arms if arm.kind == "no-token"}
    by_vocabulary: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in arms:
        if arm.kind == "no-token":
            continue
        name = f"K{arm.k}" if arm.kind == "learned" else arm.kind
        twin = twins.get(arm.seed)
        entry: dict[str, Any] = {"key": arm.key, "usage": arm.usage, "gate_t_iii": arm.usage["gate_t_iii"]}
        if twin is None:
            entry["gain"] = "pending"
        else:
            entry["twin"] = twin.key
            entry["gain"] = paired_gain(arm.rows, twin.rows)
            entry["gate_t_i"] = entry["gain"][STRATUM_ALL].get("gain_p50_m", -math.inf) >= GATE_T_GAIN_M
        by_vocabulary.setdefault(name, {})[arm.seed] = entry
    verdicts: dict[str, dict[str, Any]] = {}
    for name, seeds in by_vocabulary.items():
        complete = len(seeds) >= 2 and all(entry["gain"] != "pending" for entry in seeds.values())
        verdict: dict[str, Any] = {"seeds": sorted(seeds), "complete": complete}
        if complete:
            verdict["gate_t_i"] = all(entry["gate_t_i"] for entry in seeds.values())
            verdict["gate_t_iii"] = all(entry["gate_t_iii"] for entry in seeds.values())
            verdict["passes"] = verdict["gate_t_i"] and verdict["gate_t_iii"]
            verdict["mean_gain_p50_m"] = float(np.mean([entry["gain"][STRATUM_ALL]["gain_p50_m"] for entry in seeds.values()]))
        verdicts[name] = verdict
    passing = {name: v for name, v in verdicts.items() if name.startswith("K") and v.get("passes")}
    selected_k = None
    if passing:
        best = max(v["mean_gain_p50_m"] for v in passing.values())
        candidates = [int(name[1:]) for name, v in passing.items() if v["mean_gain_p50_m"] >= best - GATE_T_K_TOLERANCE_M]
        selected_k = min(candidates)
    return {
        "thresholds": {
            "gain_m": GATE_T_GAIN_M, "max_code_share": GATE_T_MAX_CODE_SHARE,
            "unused_code_share": GATE_T_UNUSED_CODE_SHARE, "k_tolerance_m": GATE_T_K_TOLERANCE_M,
        },
        "twins": {seed: twin.key for seed, twin in twins.items()},
        "arms": by_vocabulary,
        "verdicts": verdicts,
        "selected_k": selected_k,
        "note": "T(ii), the prior's predictability against a bigram Markov baseline, is P2.3's reading",
    }


# ── the table ────────────────────────────────────────────────────────────────

def render(payload: dict[str, Any]) -> str:
    """The segment's table: one row per arm, the strata ADE p50s, the gain against the twin,
    the code usage, and the gate verdicts."""
    lines = [
        f"manoeuvre readout · {payload['campaign']} · segment {payload['segment_s']:g} s · anchor {payload['anchor']} · "
        f"{payload['split']} · {payload['flights']} flights ({payload['truth_shorter_than_horizon']} observed shorter than Δ)",
        "",
        f"{'arm':<18}{'seed':>6}{'n':>6}" + "".join(f"{'ADE ' + STRATUM_SHORT[s]:>16}" for s in STRATA)
        + f"{'gain all':>10}{'better':>8}{'codes':>12}{'max%':>7}{'H bits':>8}",
    ]
    for key, arm in payload["arms"].items():
        summary = arm["summary"]
        cells = "".join(
            f"{summary[s]['ade_p50_m']:>16.0f}" if summary[s]["n"] else f"{'—':>16}" for s in STRATA
        )
        gain = arm.get("gain")
        gain_cell = (f"{gain[STRATUM_ALL]['gain_p50_m']:>+10.0f}{gain[STRATUM_ALL]['arm_better_share']:>8.2f}"
                     if isinstance(gain, dict) else f"{'—' if gain is None else gain:>10}{'':>8}")
        usage = arm.get("usage")
        usage_cell = (f"{usage['used']:>5}/{usage['count']:<6}{100 * usage['max_share']:>6.1f}{usage['entropy_bits']:>8.2f}"
                      if usage else f"{'':>12}{'':>7}{'':>8}")
        lines.append(f"{key:<18}{arm['seed']:>6}{summary[STRATUM_ALL]['n']:>6}{cells}{gain_cell}{usage_cell}")
    lines += ["", "gate T (both seeds):"]
    for name, verdict in payload["gate_t"]["verdicts"].items():
        if not verdict["complete"]:
            lines.append(f"  {name:<20} pending ({', '.join(f's{s}' for s in verdict['seeds'])})")
            continue
        lines.append(
            f"  {name:<20} (i) {'PASS' if verdict['gate_t_i'] else 'FAIL'}  (iii) {'PASS' if verdict['gate_t_iii'] else 'FAIL'}"
            f"  mean gain p50 {verdict['mean_gain_p50_m']:+.0f} m"
        )
    lines.append(f"  selected K: {payload['gate_t']['selected_k']}  (smallest passing K within {GATE_T_K_TOLERANCE_M:g} m of the best)")
    lines.append(f"  {payload['gate_t']['note']}")
    return "\n".join(lines) + "\n"


__all__ = [
    "GATE_T_GAIN_M", "GATE_T_K_TOLERANCE_M", "GATE_T_MAX_CODE_SHARE", "GATE_T_UNUSED_CODE_SHARE",
    "READOUT_SCHEMA", "STRATA", "ArmReading", "arm_reading", "code_usage", "fixed_anchor_readings",
    "gate_t", "paired_gain", "render", "stratum_summary",
]
