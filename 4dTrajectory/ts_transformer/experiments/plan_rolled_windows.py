"""The rolled-window table (design v5.2, §9 step 3(e)): every flight of a split flown in lockstep from L-1, the head's input window at every step with the truth's targets there.

Why: the plan head is re-asked every 30 s on the window of its OWN flown rows, and trained
on observed windows only it jittered from step to step (§12.5). This runner writes the
closed-loop training set: under ``--policy truth`` the flown states are the lockstep
oracle's (the ceiling's own windows, along the truth's route); under ``--policy model`` a
plan checkpoint's — the head's own states, each labelled by the truth's queue at that
state (the expert at the learner's state, one DAgger round; ``--extend PATH`` carries a
previous table into the new one). One ``.npz`` per table, the provenance inside, bound to
the checkpoint's window contract and to its split's flights; ``train
--plan-rolled-windows-path <table> --plan-rolled-share <p>`` draws from it.

    python run_ts.py plan_rolled_windows --checkpoint head=<run>/checkpoint.pt \\
        --split train --split val --policy truth --out <dir>/rolled_truth.npz
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ts_transformer.config import PREDICTION_PLAN, default_anchor
from ts_transformer.data.approach_difficulty import (
    STRATUM_STRAIGHT_IN,
    STRATUM_VECTORED,
    approach_difficulty,
    strata_masks,
)
from ts_transformer.experiments.anytime_curve import Grid, cohort_series, load_arm, parse_arms
from ts_transformer.inference.export import observed_series_metrics
from ts_transformer.inference.forecast import cut_at_threshold_crossing
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.outputs.plan.extractors import extract_plan
from ts_transformer.outputs.plan.forecast import (
    LOCKSTEP_S,
    ORDER_HOLD_ASKS,
    closing_budget_s,
    lockstep_states,
    truth_lockstep_policy,
)
from ts_transformer.outputs.plan.labels import TARGETS
from ts_transformer.outputs.plan.rolled import (
    POLICIES,
    POLICY_MODEL,
    POLICY_TRUTH,
    load_rolled_windows,
    record_lockstep,
    rolled_table_header,
    write_rolled_windows,
)
from ts_transformer.outputs.plan.skeleton import runway_skeleton
from ts_transformer.outputs.plan.strategy import lockstep_model_policy

INSTRUMENT = "the rolled-window table"
SPLITS = ("train", "val")


def _p(values, quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile)) if len(values) else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH",
                        help="the plan checkpoint whose split and window contract the table is bound to "
                             "(under --policy model, also the head that flies)")
    parser.add_argument("--split", action="append", choices=SPLITS, default=None,
                        help="which split(s) to fly (repeatable; default: train and val — a run needs both)")
    parser.add_argument("--limit", type=int, default=0, help="a PREFIX of each split (a smoke test)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--policy", choices=POLICIES, default=POLICY_TRUTH,
                        help="whose flown states: the truth's instructions (the lockstep oracle) or the "
                             "checkpoint's orders (the head's own states, labelled by the truth's queue)")
    parser.add_argument("--lockstep-s", type=float, default=LOCKSTEP_S, help="the re-ask period")
    parser.add_argument("--hold-asks", type=int, default=ORDER_HOLD_ASKS,
                        help="a material change of order is adopted only once given on this many consecutive asks "
                             "(v5.3; a no-op under --policy truth, whose orders change only at execution)")
    parser.add_argument("--hold-flips-only", action="store_true",
                        help="hold only a fix <-> none flip; a moved fix is adopted at once (needs --hold-asks > 1)")
    parser.add_argument("--extend", type=Path, default=None,
                        help="a previous table whose samples are carried into this one (DAgger's aggregation)")
    parser.add_argument("--device", default="cpu", help="where the head and the guidance rollout run (cpu, cuda)")
    parser.add_argument("--out", type=Path, required=True, help="the table to write (.npz)")
    args = parser.parse_args(argv)
    arms = parse_arms(parser, args.checkpoint)
    if len(arms) != 1:
        parser.error("exactly one --checkpoint names the cohort")
    (label, path), = arms.items()
    splits = tuple(dict.fromkeys(args.split or SPLITS))
    device = torch.device(args.device)
    grid = Grid(split=splits[0], bins_m=(), min_future_s=0.0, batch_size=None, limit=args.limit)
    arm = load_arm(label, path, grid, device, instrument=INSTRUMENT)
    if arm.config.prediction_output != PREDICTION_PLAN:
        parser.error(f"{label} predicts {arm.config.prediction_output!r}; the table is bound to a plan checkpoint's window contract")
    absent = [split for split in splits if not arm.payload["split"].get(split)]
    if absent:
        parser.error(f"{label} stores no {', '.join(absent)} split")
    extend = load_rolled_windows(args.extend) if args.extend is not None else None
    anchor = default_anchor(arm.config)
    skeletons: dict[tuple[str, str], object] = {}
    samples, summary, airports = [], {}, set()
    started = time.perf_counter()
    for split in splits:
        series = cohort_series(arm, replace(grid, split=split))
        split_started = time.perf_counter()
        split_samples, rows = [], []
        for start in range(0, len(series), args.batch_size):
            chunk = series[start:start + args.batch_size]
            chunk_skeletons = []
            for item in chunk:
                key = (item.airport, str(item.scenario.source.get("runway")))
                if key not in skeletons:
                    skeletons[key] = runway_skeleton(item)
                chunk_skeletons.append(skeletons[key])
                airports.add(item.airport)
            anchors = [anchor] * len(chunk)
            labels = [extract_plan(item, anchor, sk) for item, sk in zip(chunk, chunk_skeletons, strict=True)]
            if args.policy == POLICY_TRUTH:
                # the oracle's own horizon: the truth's T plus the closing slack, so a late
                # arrival is flown out rather than cut unestablished
                horizons = [closing_budget_s(lab.T_s) for lab in labels]
                states = lockstep_states(chunk, anchors, chunk_skeletons, [lab.remaining_path_at_anchor_m for lab in labels])
                policy = truth_lockstep_policy(states, labels, horizons)
                caps = horizons
            else:
                states, policy = lockstep_model_policy(
                    arm.model, chunk, arm.config, arm.normalizer, anchors, device, chunk_skeletons,
                )
                caps = None
            flights, chunk_samples = record_lockstep(
                states, labels, arm.config, policy=policy, step_s=args.lockstep_s, time_caps_s=caps,
                hold_asks=args.hold_asks, hold_flips_only=args.hold_flips_only, device=device,
            )
            split_samples.extend(chunk_samples)
            for item, flight in zip(chunk, flights, strict=True):
                cut = cut_at_threshold_crossing(flight.forecast, item)
                metrics = observed_series_metrics(item, replace(cut, predicted_final_time_s=cut.final_time_s))
                rows.append({
                    "dataset_id": item.dataset_id, "difficulty": approach_difficulty(item, anchor).to_dict(),
                    "steps": len(flight.orders), "capped_by": flight.capped_by, "ade_m": metrics["ade_m"],
                    "final_time_error_s": metrics["final_time_error_s"],
                })
            print(f"  {split}: flown {len(rows)}/{len(series)}", flush=True)
        samples.extend(split_samples)
        steps = np.array([row["steps"] for row in rows], dtype=np.float64)
        ade = np.array([row["ade_m"] for row in rows], dtype=np.float64)
        masks = strata_masks({row["dataset_id"]: row["difficulty"] for row in rows}, [row["dataset_id"] for row in rows])
        summary[split] = {
            "flights": len(rows), "samples": len(split_samples),
            "steps_p50": _p(steps, 50), "steps_max": float(steps.max()) if len(steps) else 0.0,
            "on_final_share": float(np.mean([s.on_final for s in split_samples])) if split_samples else 0.0,
            "next_is_join_share": float(np.mean([s.targets.next_is_join for s in split_samples])) if split_samples else 0.0,
            "arrival_time_undefined_share": float(np.mean([s.targets.valid[TARGETS.index("T_s")] == 0.0 for s in split_samples])) if split_samples else 0.0,
            "capped_by": {name: sum(1 for row in rows if row["capped_by"] == name) for name in ("time", "legs")},
            "flown_ade_m": {
                stratum: {"n": int(masks[stratum].sum()),
                          "mean": float(ade[masks[stratum]].mean()) if masks[stratum].any() else float("nan")}
                for stratum in (STRATUM_STRAIGHT_IN, STRATUM_VECTORED)
            },
            "wall_s": time.perf_counter() - split_started,
        }
    wall_s = time.perf_counter() - started
    header = rolled_table_header(
        arm.config, policy=args.policy, lockstep_s=args.lockstep_s, hold_asks=args.hold_asks,
        hold_flips_only=args.hold_flips_only, airports=sorted(airports),
        splits={split: block["flights"] for split, block in summary.items()},
        checkpoint={"label": label, "path": str(path)}, generated_at=utc_now(), wall_s=wall_s,
    )
    header["limit"] = args.limit
    table = write_rolled_windows(args.out, samples, header, extend=extend)
    for split, block in summary.items():
        print(
            f"{split}: {block['flights']} flights, {block['samples']} samples (steps p50 {block['steps_p50']:.0f}, "
            f"max {block['steps_max']:.0f}); on the final {block['on_final_share']:.1%}, no fix ahead "
            f"{block['next_is_join_share']:.1%}, arrival time undefined {block['arrival_time_undefined_share']:.1%}; "
            f"capped {block['capped_by']}; flown ADE straight-in {block['flown_ade_m'][STRATUM_STRAIGHT_IN]['mean']:.0f} m "
            f"(n {block['flown_ade_m'][STRATUM_STRAIGHT_IN]['n']}), vectored {block['flown_ade_m'][STRATUM_VECTORED]['mean']:.0f} m "
            f"(n {block['flown_ade_m'][STRATUM_VECTORED]['n']}); {block['wall_s']:.0f} s"
        )
    print(f"wrote {table.path}: {table.samples} samples over {table.flights} flights"
          + (f" (extended from {extend.path}: {extend.samples} samples)" if extend is not None else "")
          + f", sha256 {table.sha256[:12]}, {wall_s:.0f} s", flush=True)
    write_json_atomic(args.out.with_suffix(".json"), {
        "schema_version": "ts-plan-rolled-windows-summary-v1", "generated_at": header["generated_at"],
        "instrument": INSTRUMENT, "table": table.provenance, "header": header, "summary": summary,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
