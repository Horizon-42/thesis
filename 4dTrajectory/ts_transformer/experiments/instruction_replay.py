"""Fly a vocabulary artefact's own sentences and report whether they reach the runway.

    python run_ts.py instruction_replay --vocabulary <…/instruction_vocabulary.json> \\
        (--executor <ckpt> | --airports KRDU KSJC …) --cohort <development_cohort.json> \\
        --out <dir> [--split train|val] [--limit N]

**The acceptance test for a vocabulary** (the user's rule, 2026-09-21): before anything is trained
on a sentence, that sentence has to be able to land. A word sequence read from a track is not
self-evidently flyable — a heading word names a direction and cannot say which way round to turn,
nor hold a line — and the two are exactly what the first v12 artefacts got wrong.

It flies what the FILE says (`Reading.from_dict`), never a re-reading: a gate that re-derives its
own input cannot see the artefact drift from the code that wrote it. The flying model, its
assumptions and their measured values are `manoeuvre.instruction_kinematics`; the same model draws
the frontend's Training view, so this number and that picture cannot come apart.

Written under ``--out`` (refused if it exists):

    replay.json   per flight: the end reason, the gap to the observed track, the fit residual
    summary.txt   the landing rate and the gap distribution, one screen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from ts_transformer.config import TSConfig, default_anchor
from ts_transformer.data.approach_difficulty import STRAIGHT_TORTUOSITY, approach_difficulty
from ts_transformer.data.development_cohorts import load_development_cohort
from ts_transformer.experiments.support import REPO_ROOT, cohort_from_manifests, cohort_splits, rebuild_cohort
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.manoeuvre import instruction_kinematics as kinematics
from ts_transformer.manoeuvre.instructions import Reading, course_frame, load_vocabulary
from ts_transformer.training.train import load_checkpoint_payload

REPLAY_SCHEMA = "ts-instruction-replay-v1"


def render(rows: list[dict[str, Any]], vocabulary_sha: str, split: str,
           refused_starts: list[dict[str, str]] | None = None) -> str:
    landed = [r for r in rows if r["landed"]]
    straight = [r for r in rows if r["stratum"] == "straight-in"]
    vectored = [r for r in rows if r["stratum"] == "vectored"]
    lines = [f"instruction replay · {split} · {len(rows)} flights · vocabulary {vocabulary_sha[:12]}…", ""]
    lines.append(f"  LANDED (crossed the threshold ON the final): {len(landed)} / {len(rows)} = {len(landed) / len(rows):.1%}")
    for name, group in (("straight-in", straight), ("vectored", vectored)):
        if group:
            lines.append(f"    {name:11s} {sum(r['landed'] for r in group)} / {len(group)} = "
                         f"{sum(r['landed'] for r in group) / len(group):.1%}")
    for label, key in (("gap p95 (m)", "gap_p95_m"), ("mean gap (m)", "gap_mean_m")):
        values = np.array([r[key] for r in rows])
        lines.append(f"  {label:14s} p50 {np.median(values):8.0f}   p90 {np.percentile(values, 90):8.0f}   "
                     f"worst {values.max():9.0f}")
    fit = np.array([r["vertical_fit_rms_m"] for r in rows])
    lines.append(f"  vertical fit RMS (m): p50 {np.median(fit):.0f}   p95 {np.percentile(fit, 95):.0f}")
    events = np.array([r["events"] for r in rows])
    lines.append(f"  events per flight: p50 {np.median(events):.0f}   p95 {np.percentile(events, 95):.0f}")
    reasons: dict[str, int] = {}
    for row in rows:
        reasons[row["end_reason"]] = reasons.get(row["end_reason"], 0) + 1
    lines.append(f"  end reasons: {reasons}")
    if refused_starts:
        # never a silent exclusion: the flights are named and the reason is theirs, not the
        # vocabulary's (a corrupt observed first row the model cannot start from)
        lines.append(f"  NOT FLOWN — the observed start is unreachable: {len(refused_starts)} flight(s)")
        for item in refused_starts:
            lines.append(f"    {item['dataset_id']}: {item['why'].split(': ', 1)[-1]}")
    lines.append("")
    lines.append("  the model these were flown under:")
    for name, value in kinematics.assumptions().items():
        lines.append(f"    {name}: {value}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--vocabulary", type=Path, required=True, help="the artefact whose sentences are flown")
    parser.add_argument("--executor", type=Path, help="a checkpoint.pt: the door to the cohort's data")
    parser.add_argument("--airports", nargs="+", metavar="ICAO", help="or read the cohort straight from these manifests")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--limit", type=int, default=0, help="a PREFIX of the split (a smoke test)")
    args = parser.parse_args(argv)
    if bool(args.executor) == bool(args.airports):
        parser.error("give exactly one of --executor (a checkpoint's provenance) or --airports (the manifests')")
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a replay is never overwritten")
    started = time.perf_counter()

    vocabulary_path = args.vocabulary if args.vocabulary.is_absolute() else REPO_ROOT / args.vocabulary
    vocabulary, _runways, _payload = load_vocabulary(vocabulary_path)
    sentences = json.loads((vocabulary_path.parent / f"sentences_{args.split}.json").read_text(encoding="utf-8"))
    if sentences["vocabulary_sha256"] != vocabulary.sha256:
        raise SystemExit(f"{vocabulary_path.parent}: the sentences were read under "
                         f"{sentences['vocabulary_sha256'][:12]}…, the spec beside them is {vocabulary.sha256[:12]}…")
    readings = {item["dataset_id"]: Reading.from_dict(item) for item in sentences["flights"]}

    cohort_path = args.cohort if args.cohort.is_absolute() else REPO_ROOT / args.cohort
    cohort = load_development_cohort(cohort_path)
    if args.airports:
        config = TSConfig()
        series, _splits, provenance = cohort_from_manifests(args.airports, cohort, config, args.limit)
    else:
        executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
        payload = load_checkpoint_payload(executor)
        config = TSConfig.from_dict(payload["config"])
        provenance = {"read_from": "executor checkpoint", "path": str(executor), "executor_sha256": file_sha256(executor)}
        splits = cohort_splits(payload, cohort, args.limit)
        series = rebuild_cohort(payload, config, [*splits["train"], *splits["val"]])
    series = [item for item in series if item.dataset_id in readings]
    if not series:
        raise SystemExit(f"none of the cohort's flights are in sentences_{args.split}.json")
    print(f"  {len(series)} flights of the {args.split} split", flush=True)

    anchor = default_anchor(config)
    rows: list[dict[str, Any]] = []
    refused_starts: list[dict[str, str]] = []
    for item in series:
        reading = readings[item.dataset_id]
        frame = course_frame(item)
        start = kinematics.Start(
            to_go_m=float(frame["to_go_m"][0]), cross_m=float(frame["cross_m"][0]),
            height_m=float(frame["height_m"][0]), ground_speed_mps=float(frame["ground_speed_mps"][0]),
            relative_course_deg=float(frame["relative_course_deg"][0]),
        )
        try:
            track = kinematics.fly(reading, vocabulary, start, reading.duration_s)
        except kinematics.UnreachableStart as refused:
            # a corrupt observed first row, not a sentence the vocabulary got wrong — counted and
            # NAMED here rather than flown into a number that would swamp every distribution
            refused_starts.append({"dataset_id": item.dataset_id, "why": str(refused)})
            continue
        gap = kinematics.gap_to_observed(track, frame)
        difficulty = approach_difficulty(item, anchor)
        rows.append({
            "dataset_id": item.dataset_id, "flight_id": reading.flight_id, "runway": reading.runway,
            "landed": track.end_reason != kinematics.END_TIME_CAP, "end_reason": track.end_reason,
            "gap_p95_m": gap["gapP95M"], "gap_mean_m": gap["meanGapM"],
            "final_gap_m": round(float(track.final_gap_m), 1),
            "events": len(reading.event_times_s), "vertical_fit_rms_m": reading.vertical_fit_rms_m,
            "stratum": "straight-in" if difficulty.route_tortuosity < STRAIGHT_TORTUOSITY else "vectored",
        })

    table = render(rows, vocabulary.sha256, args.split, refused_starts)
    out.mkdir(parents=True)
    write_json_atomic(out / "replay.json", {
        "schema": REPLAY_SCHEMA, "written_utc": utc_now(), "split": args.split,
        "vocabulary": str(vocabulary_path), "vocabulary_sha256": vocabulary.sha256,
        "reading_rule": vocabulary.reading_rule, "assumptions": kinematics.assumptions(),
        "provenance": provenance, "flights": rows, "refused_starts": refused_starts,
        "landed": sum(r["landed"] for r in rows), "elapsed_s": time.perf_counter() - started,
    })
    (out / "summary.txt").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
