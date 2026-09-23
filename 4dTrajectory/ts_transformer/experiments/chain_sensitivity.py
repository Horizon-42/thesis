#!/usr/bin/env python
"""Chain sensitivity (T0(b)): a control checkpoint re-asked on its OWN rollout every Δ seconds.

Two-tier feasibility (`docs/2026-09-16_two_tier_transformer_feasibility.zh.md` §4.2, §10.1):
a short-horizon layer chained without new observations obeys ``e_k <= ε + L·e_(k-1)``, and
whether the chain beats the one-shot forecast depends on L — how the model's error answers
an error already in its history. This runner measures it on a stored checkpoint, with no
training::

    python run_ts.py chain_sensitivity \\
        --checkpoint native32=4dTrajectory/outputs/KRDU/experiments/l1_lowdim_20260907/L1_native32/checkpoint.pt \\
        --out 4dTrajectory/outputs/KRDU/experiments/two_tier_t0b_20260916/chain_60s --step-s 60 --links 5

Per flight of the checkpoint's own split, at the fixed anchor ``a0`` (`default_anchor`), three
forecasts from ONE forward code path (`forecast_approaches`):

* **one-shot** — the ordinary prediction at ``a0``;
* **chain** — link 0 is the one-shot cut at Δ; link k is the checkpoint asked at
  ``a_k = a0 + k·Δ/dt`` on a ROLLED series: the observed track to ``a0`` continued by the
  chain's own flown rows on the series' 2 s grid (`rolled_series`). The window, the anchor
  state and the lagged actuators' initial condition all come from the model's ordinary
  predict path on that series — the actuator state is inverted from the rolled lookback by
  the same rule training inverts it from the observed one. Every link but the last is cut at
  Δ; the last is kept whole, and a link whose own forecast ends inside Δ (it predicts the
  landing) ends the chain there;
* **re-anchored on the truth** (protocol B) — the checkpoint at ``a_(k-1)`` on the OBSERVED
  series, read at lead Δ: ``ε_k``, the one-step error with a TRUE history at that anchor.

At each lead ``kΔ`` the displacement of each forecast from the observed track (the flight's
own chart channels e/n/u, both interpolated onto the lead; a forecast or a truth that ends
before it is ABSENT there, never 0 or held; a non-finite one refuses the run) gives, per
stratum (`strata_fixed_at_anchor` at ``a0``), for ``k = 1 … links + 1`` (the last link's first Δ
is scored too): ``e_one(kΔ)``, ``e_chain(kΔ)``, ``ε_k``, and two estimates of the chain's L over the flights
that have all three at ``k`` and ``e_chain((k-1)Δ)`` (``k >= 2``): the paired median of
``(e_chain(kΔ) − ε_k) / e_chain((k−1)Δ)`` and the least-squares slope through the origin of
``e_chain(kΔ) − ε_k`` on ``e_chain((k−1)Δ)``. L near 1 is the linear accumulation §4.2's bound
reads; below 1 the model pulls an offset history back toward the runway it is anchored on.

A checkpoint trained only at L−1 is re-asked at anchors it never saw, so its ``ε_k`` carries the
out-of-distribution cost as well; a random-anchor checkpoint beside it separates the two. Both
are the arm's own properties, which is why the artifact names each checkpoint's anchor policy.

``--write-records`` also writes the one-shot and the chain as predict-shaped directories
(``records/<label>/oneshot/`` and ``records/<label>/chain_<Δ>s/``) — `python -m evaluation`,
`run_ts.py lead_time_error` and the publisher read them unchanged; each summary carries a
``chain`` block naming what it is.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

from ts_transformer.config import CONTROL_HOOK_OFF, DURATION_HEAD_POINT, PREDICTION_CONTROL, default_anchor
from ts_transformer.data.anchor_grid import strata_fixed_at_anchor
from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED
from ts_transformer.data.dataset import FlightSeries, window_anchors
from ts_transformer.backbone.adapters import resolve_device
from ts_transformer.experiments.anytime_curve import FORBIDDEN_SPLIT, Arm, Grid, cohort_series, load_arm, parse_arms
from ts_transformer.experiments.support import REPO_ROOT
from ts_transformer.inference.export import build_prediction_record, observed_series_metrics, write_batch
from ts_transformer.inference.forecast import Forecast, concatenate, forecast_approaches
from ts_transformer.data.time_grids import ROW_TOLERANCE_S
from ts_transformer.inference.receding import cut_at_lead, displacement_at, rolled_series
from ts_transformer.io_utils import file_sha256

RESULT_SCHEMA = "ts-chain-sensitivity-v1"
RECORDS_SCHEMA = "ts-chain-records-v1"
#: The summary block this runner writes beside the records — mirrored by the publisher
#: (`publish_ts_experiment_trajectories.VARIANT_RECORD_BLOCKS`, which cannot import torch).
RECORDS_BLOCK = "chain"
RECORDS_DIR = "records"
INSTRUMENT = "the chain-sensitivity readout"
DEFAULT_STEP_S = 60.0
DEFAULT_LINKS = 5


@dataclass(frozen=True)
class ChainPlan:
    """What the chain is measured on: parsed once, then read."""

    split: str
    step_s: float
    links: int
    limit: int
    batch_size: int | None
    write_records: bool


# ── the forecasts ──────────────────────────────────────────────────────────────

def batched_forecasts(arm: Arm, series: list[FlightSeries], anchor: int, batch_size: int,
                      device: torch.device) -> list[Forecast]:
    out: list[Forecast] = []
    for start in range(0, len(series), batch_size):
        out.extend(forecast_approaches(
            arm.model, series[start : start + batch_size], arm.config, arm.normalizer,
            anchor=anchor, device=device,
        ))
    return out


def chain_forecasts(arm: Arm, series: list[FlightSeries], one_shot: list[Forecast], plan: ChainPlan,
                    device: torch.device, batch_size: int) -> tuple[list[Forecast], list[int]]:
    """Every flight's chain as one forecast from ``a0``, and the number of links it holds."""
    config = arm.config
    a0 = default_anchor(config)
    step_rows = int(round(plan.step_s / config.dt_s))
    kept: list[list[Forecast]] = []
    active: list[int] = []
    for index, forecast in enumerate(one_shot):
        if forecast.final_time_s <= plan.step_s + ROW_TOLERANCE_S:
            kept.append([forecast])
        else:
            kept.append([cut_at_lead(forecast, plan.step_s)])
            active.append(index)
    for link in range(1, plan.links + 1):
        if not active:
            break
        anchor = a0 + link * step_rows
        rolled = [
            rolled_series(series[i], a0, concatenate(kept[i], a0, 0.0), anchor, config.dt_s) for i in active
        ]
        answers = batched_forecasts(arm, rolled, anchor, batch_size, device)
        still: list[int] = []
        for index, forecast in zip(active, answers, strict=True):
            if link == plan.links or forecast.final_time_s <= plan.step_s + ROW_TOLERANCE_S:
                kept[index].append(forecast)
            else:
                kept[index].append(cut_at_lead(forecast, plan.step_s))
                still.append(index)
        active = still
        print(f"    link {link}: asked {len(answers)} at anchor {anchor}, {len(active)} continue", flush=True)
    chains = []
    for legs in kept:
        start = float(np.sum([np.sum(leg.sample_durations_s) for leg in legs[:-1]]))
        chains.append(concatenate(legs, a0, start + float(legs[-1].predicted_final_time_s)))
    return chains, [len(legs) for legs in kept]


def reanchored_errors(arm: Arm, series: list[FlightSeries], one_shot: list[Forecast], plan: ChainPlan,
                      device: torch.device, batch_size: int) -> list[dict[int, float]]:
    """``ε_k`` per flight for ``k = 1 … links + 1``: the forecast at ``a_(k-1)`` on the OBSERVED
    series, at lead Δ, for every ``k`` whose anchor the observed series admits
    (`window_anchors`). At ``k = 1`` that forecast IS the one-shot, which is reused."""
    config = arm.config
    a0 = default_anchor(config)
    step_rows = int(round(plan.step_s / config.dt_s))
    errors: list[dict[int, float]] = [{} for _ in series]
    for k in range(1, plan.links + 2):
        anchor = a0 + (k - 1) * step_rows
        if k == 1:
            members, answers = list(range(len(series))), one_shot
        else:
            members = [i for i, item in enumerate(series) if anchor in window_anchors(item, config)]
            answers = batched_forecasts(arm, [series[i] for i in members], anchor, batch_size, device)
        for index, forecast in zip(members, answers, strict=True):
            item = series[index]
            value = displacement_at(item, forecast, anchor, float(item.times[anchor]) + plan.step_s)
            if value is not None:
                errors[index][k] = value
        print(f"    re-anchored k={k}: {len(members)} flights at anchor {anchor}", flush=True)
    return errors


# ── the readout ────────────────────────────────────────────────────────────────

def flight_rows(series: list[FlightSeries], one_shot: list[Forecast], chains: list[Forecast],
                links_held: list[int], epsilon: list[dict[int, float]], plan: ChainPlan, a0: int) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for item, one, chain, held, eps in zip(series, one_shot, chains, links_held, epsilon, strict=True):
        origin = float(item.times[a0])
        at = {}
        for k in range(1, plan.links + 2):
            time_s = origin + k * plan.step_s
            at[str(k)] = {
                "e_one_m": displacement_at(item, one, a0, time_s),
                "e_chain_m": displacement_at(item, chain, a0, time_s),
                "epsilon_m": eps.get(k),
            }
        rows[item.dataset_id] = {"links": held, "at": at}
    return rows


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "p50": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {"n": int(array.size), "p50": float(np.median(array)), "mean": float(array.mean())}


def stratum_summary(rows: dict[str, dict], keys: list[str], plan: ChainPlan) -> dict[str, dict]:
    """Per lead: each displacement's p50/mean over the flights that have it, the chain/one-shot
    ratio paired, and L's two estimates paired over the flights that have every term."""
    out: dict[str, dict] = {}
    for k in range(1, plan.links + 2):
        cells = [rows[key]["at"][str(k)] for key in keys]
        block: dict[str, object] = {
            "lead_s": k * plan.step_s,
            "e_one": _stats([c["e_one_m"] for c in cells if c["e_one_m"] is not None]),
            "e_chain": _stats([c["e_chain_m"] for c in cells if c["e_chain_m"] is not None]),
            "epsilon": _stats([c["epsilon_m"] for c in cells if c["epsilon_m"] is not None]),
        }
        both = [c for c in cells if c["e_one_m"] is not None and c["e_chain_m"] is not None and c["e_one_m"] > 0.0]
        block["chain_over_one_p50"] = (
            float(np.median([c["e_chain_m"] / c["e_one_m"] for c in both])) if both else None
        )
        block["chain_over_one_n"] = len(both)
        if k >= 2:
            x, y = [], []
            for key in keys:
                now, before = rows[key]["at"][str(k)], rows[key]["at"][str(k - 1)]
                if None in (now["e_chain_m"], now["epsilon_m"], before["e_chain_m"]) or before["e_chain_m"] <= 0.0:
                    continue
                x.append(before["e_chain_m"])
                y.append(now["e_chain_m"] - now["epsilon_m"])
            if x:
                xs, ys = np.asarray(x), np.asarray(y)
                block["L"] = {
                    "n": len(x),
                    "paired_median": float(np.median(ys / xs)),
                    "least_squares_through_origin": float(np.sum(xs * ys) / np.sum(xs * xs)),
                }
            else:
                block["L"] = {"n": 0, "paired_median": None, "least_squares_through_origin": None}
        out[str(k)] = block
    return out


def _fmt(value: float | None, digits: int = 0) -> str:
    return "—" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.{digits}f}"


def render(payload: dict) -> str:
    plan = payload["plan"]
    lines = [
        f"Chain sensitivity: re-asked every {plan['step_s']:g} s on its own rollout, {plan['links']} links "
        f"(split {plan['split']}{', limit ' + str(plan['limit']) if plan['limit'] else ''})",
        "e_one = one-shot from a0, e_chain = the chain, eps = the forecast at a_(k-1) on the OBSERVED series "
        "at lead step; displacement p50 / mean (m) over the flights that reach the lead, n per column.",
        "L = (e_chain(k) - eps_k) / e_chain(k-1), paired: median and least squares through the origin.",
    ]
    for label, arm in payload["checkpoints"].items():
        lines.append("")
        lines.append(f"── {label} [{arm['anchor_policy']}] — {arm['checkpoint']} ({arm['flights']} flights, a0={arm['anchor']})")
        for stratum, block in arm["strata"].items():
            lines.append(f"   {stratum}  [n={block['n']}]")
            lines.append("    lead s |  e_one p50   mean     n | e_chain p50   mean     n |  eps p50   mean     n | chain/one p50 |  L med  L lsq     n")
            for k, cell in block["leads"].items():
                l_block = cell.get("L")
                lines.append(
                    f"    {cell['lead_s']:>6.0f} | {_fmt(cell['e_one']['p50']):>10} {_fmt(cell['e_one']['mean']):>6} {cell['e_one']['n']:>5} |"
                    f" {_fmt(cell['e_chain']['p50']):>11} {_fmt(cell['e_chain']['mean']):>6} {cell['e_chain']['n']:>5} |"
                    f" {_fmt(cell['epsilon']['p50']):>8} {_fmt(cell['epsilon']['mean']):>6} {cell['epsilon']['n']:>5} |"
                    f" {_fmt(cell['chain_over_one_p50'], 2):>13} |"
                    + (f" {_fmt(l_block['paired_median'], 2):>6} {_fmt(l_block['least_squares_through_origin'], 2):>6} {l_block['n']:>5}"
                       if l_block else "      —      —     —")
                )
    return "\n".join(lines) + "\n"


# ── records ────────────────────────────────────────────────────────────────────

def write_records(arm: Arm, series: list[FlightSeries], forecasts: list[Forecast], directory: Path, *,
                  plan: ChainPlan, variant: str, campaign: str) -> None:
    records, metrics = [], []
    for index, (item, forecast) in enumerate(zip(series, forecasts, strict=True)):
        metrics.append(observed_series_metrics(item, forecast, points=arm.config.validation_common_grid_points))
        records.append(build_prediction_record(
            item, forecast, index=index, model_name=arm.config.model, horizon_mode=arm.config.horizon_mode,
            split=plan.split,
        ))
    write_batch(
        records, output_dir=directory, config_dict=arm.config.to_dict(), flight_metrics=metrics,
        checkpoint=str(arm.path), split=plan.split,
        extra_summary={RECORDS_BLOCK: {
            "schema": RECORDS_SCHEMA, "campaign": campaign, "label": arm.label, "variant": variant,
            "step_s": plan.step_s, "links": plan.links, "anchor": default_anchor(arm.config),
            "split": plan.split, "limit": plan.limit,
            "split_flights": len(arm.payload["split"][plan.split]), "records": len(records),
        }},
    )


# ── the run ────────────────────────────────────────────────────────────────────

def check_arm(arm: Arm) -> None:
    """The chain is a control-path instrument without a command hook: the rolled series has no
    truth, and a hook reads the rollout's own state, not the history."""
    if arm.config.prediction_output != PREDICTION_CONTROL:
        raise SystemExit(f"{arm.label} ({arm.path}): {INSTRUMENT} re-asks a CONTROL checkpoint on its own "
                         f"rollout; this one predicts {arm.config.prediction_output!r}")
    if arm.config.control_command_hook != CONTROL_HOOK_OFF:
        raise SystemExit(f"{arm.label} ({arm.path}): a hooked checkpoint rewrites the schedule it flies; "
                         f"{INSTRUMENT} measures the network's own chain")
    if arm.config.latent_dim:
        raise SystemExit(f"{arm.label} ({arm.path}): a latent checkpoint decodes a sample; {INSTRUMENT} is "
                         "defined on the deterministic top-1 forecast")
    if arm.config.duration_head != DURATION_HEAD_POINT:
        raise SystemExit(f"{arm.label} ({arm.path}): duration_head={arm.config.duration_head!r} — "
                         f"{INSTRUMENT} is defined on the point head")


def measure_arm(arm: Arm, series: list[FlightSeries], plan: ChainPlan, device: torch.device,
                records_root: Path | None, campaign: str) -> dict:
    started = time.time()
    config = arm.config
    a0 = default_anchor(config)
    batch_size = plan.batch_size or config.batch_size
    keys = [item.dataset_id for item in series]
    print(f"{arm.label}: {len(series)} flights, a0 {a0}, step {plan.step_s:g} s, {plan.links} links", flush=True)
    one_shot = batched_forecasts(arm, series, a0, batch_size, device)
    print("  chain", flush=True)
    chains, held = chain_forecasts(arm, series, one_shot, plan, device, batch_size)
    print("  re-anchored on the truth", flush=True)
    epsilon = reanchored_errors(arm, series, one_shot, plan, device, batch_size)
    rows = flight_rows(series, one_shot, chains, held, epsilon, plan, a0)
    masks = strata_fixed_at_anchor(series, keys, anchor=a0)
    strata = {}
    for stratum in (STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED):
        members = [key for key, keep in zip(keys, masks[stratum], strict=True) if keep]
        strata[stratum] = {"n": len(members), "leads": stratum_summary(rows, members, plan)}
    record_dirs = {}
    if records_root is not None:
        for variant, forecasts in (("oneshot", one_shot), (f"chain_{plan.step_s:g}s", chains)):
            directory = records_root / arm.label / variant
            write_records(arm, series, forecasts, directory, plan=plan, variant=variant, campaign=campaign)
            record_dirs[variant] = str(directory.relative_to(records_root.parent))
    return {
        "checkpoint": str(arm.path),
        "checkpoint_sha256": file_sha256(arm.path),
        "anchor_policy": arm.arm,
        "anchor": a0,
        "flights": len(series),
        "split_flights": len(arm.payload["split"][plan.split]),
        "links_held": {str(n): int(sum(1 for h in held if h == n)) for n in sorted(set(held))},
        "strata": strata,
        "flights_rows": rows,
        "record_dirs": record_dirs,
        "seconds": time.time() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("--checkpoint", action="append", default=None, metavar="LABEL=PATH",
                        help="a control checkpoint and its label; repeatable")
    parser.add_argument("--out", type=Path, required=True, help="must not exist (immutable artifact)")
    parser.add_argument("--split", choices=("val", "train", FORBIDDEN_SPLIT), default="val",
                        help=f"the checkpoint split to replay ({FORBIDDEN_SPLIT!r} is sealed and refused)")
    parser.add_argument("--step-s", type=float, default=DEFAULT_STEP_S,
                        help=f"the re-ask period Δ (default {DEFAULT_STEP_S:g}); a whole multiple of the "
                             "series dt and of the rollout's integrator step")
    parser.add_argument("--links", type=int, default=DEFAULT_LINKS,
                        help=f"re-asks after the first forecast (default {DEFAULT_LINKS}); leads are scored "
                             "at k·Δ for k = 1 … links + 1, the last link's first Δ included")
    parser.add_argument("--limit", type=int, default=0, help="first N flights of the split (a smoke test)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--write-records", action="store_true",
                        help="write the one-shot and the chain as predict-shaped record directories")
    parser.add_argument("--device", default="auto")
    return parser


def parse_plan(parser: argparse.ArgumentParser, args: argparse.Namespace) -> ChainPlan:
    if args.split == FORBIDDEN_SPLIT:
        parser.error(f"the {FORBIDDEN_SPLIT} split is sealed")
    if not math.isfinite(args.step_s) or args.step_s <= 0.0:
        parser.error("--step-s must be positive")
    if args.links < 1:
        parser.error("--links must be at least 1")
    if args.limit < 0:
        parser.error("--limit must be non-negative (0 = the whole split)")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return ChainPlan(split=args.split, step_s=float(args.step_s), links=int(args.links), limit=int(args.limit),
                     batch_size=args.batch_size, write_records=bool(args.write_records))


def check_step(parser: argparse.ArgumentParser, arm: Arm, plan: ChainPlan) -> None:
    for name, unit in (("dt_s", arm.config.dt_s), ("control_rollout_integrator_dt_s", arm.config.control_rollout_integrator_dt_s)):
        steps = plan.step_s / unit
        if abs(steps - round(steps)) > 1e-9:
            parser.error(f"{arm.label}: --step-s {plan.step_s:g} is not a whole multiple of the checkpoint's "
                         f"{name} {unit:g}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    arms = parse_arms(parser, args.checkpoint)
    plan = parse_plan(parser, args)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        raise FileExistsError(f"{out} exists; the chain readout is an immutable artifact")
    device = resolve_device(args.device)
    grid = Grid(split=plan.split, bins_m=(), min_future_s=0.0, batch_size=plan.batch_size, limit=plan.limit)
    loaded = [load_arm(label, path, grid, device, instrument=INSTRUMENT) for label, path in arms.items()]
    for arm in loaded:
        check_arm(arm)
        check_step(parser, arm, plan)
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(dir=out.parent, prefix=f"{out.name}.partial-"))
    try:
        records_root = staged / RECORDS_DIR if plan.write_records else None
        payload = {
            "schema": RESULT_SCHEMA,
            "plan": {"split": plan.split, "step_s": plan.step_s, "links": plan.links, "limit": plan.limit,
                     "write_records": plan.write_records},
            "displacement": "3D chart (e, n, u) distance to the observed track at the lead; absent when either ends first",
            "strata_anchor": "a0 = default_anchor, computed once (strata_fixed_at_anchor)",
            "device": str(device),
            "checkpoints": {
                arm.label: measure_arm(arm, cohort_series(arm, grid), plan, device, records_root, out.name)
                for arm in loaded
            },
        }
        (staged / "chain_sensitivity.json").write_text(json.dumps(payload, indent=1))
        text = render(payload)
        (staged / "chain_sensitivity.txt").write_text(text)
        staged.chmod(0o755)
        staged.rename(out)
    except BaseException:
        print(f"\nremoving the staged artifact {staged} — the run did not complete", file=sys.stderr, flush=True)
        shutil.rmtree(staged, ignore_errors=True)
        raise
    print()
    print(text, end="")
    print(f"wrote {out / 'chain_sensitivity.txt'} and {out / 'chain_sensitivity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
