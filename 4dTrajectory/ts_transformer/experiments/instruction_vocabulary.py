"""Read the instruction words of a cohort's tracks and write the vocabulary artefact, the sentences and the hand-check sample (two-tier v3 stage B, step B0′; plan §5.2.1, §5.2.5).

    python run_ts.py instruction_vocabulary --executor <ckpt> --cohort <development_cohort.json> --out <dir> \\
        [--token-step-s 10] [--set FIELD=VALUE ...] [--hand-check 300] [--seed 1337] [--limit N]

The cohort is a development cohort file (D33 / D47: the grid's L60_D60 cohort); its flights are
rebuilt through the checkpoint's data provenance (`support.rebuild_cohort`, C25) under the
checkpoint's config — the checkpoint is the door to the data, nothing of it is read. Every flight
of both splits is read (`instructions.read_instructions`) under the module's starting bins and
thresholds (D51 / D53) at ``--token-step-s`` (D54). Written under ``--out`` (refused if it exists):

    instruction_vocabulary.json    the spec under its sha, the cohort identity, per-word counts over
                                   the TRAIN split (the val counts beside them)
    sentences_train.json / sentences_val.json   every flight's instructions and positions
    summary.json / summary.txt     instructions per flight, word shares, plateau residuals,
                                   clamps (stated, never silent), intercept coverage
    hand_check/                    ``--hand-check`` train flights drawn at random (half straight-in,
                                   half vectored by tortuosity): one PNG each (plan view; relative
                                   course, height and ground speed against time with the words in
                                   force drawn as steps) and ``index.csv`` with a column for the
                                   human's verdict (一致 / 漏读 / 误读) — the hand check of B0′.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import fields
from pathlib import Path
import time
from typing import Any

import numpy as np

from ts_transformer.config import TSConfig, default_anchor
from ts_transformer.data.approach_difficulty import (
    STRAIGHT_TORTUOSITY, STRATUM_SHORT, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, approach_difficulty,
)
from ts_transformer.data.data_provenance import provenance_eligible_set_digests
from ts_transformer.data.development_cohorts import development_cohort_audit, load_development_cohort
from ts_transformer.experiments.support import REPO_ROOT, rebuild_cohort
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.manoeuvre.instructions import (
    INSTRUCTION_KINDS, MANDATORY_KINDS, NO_INTERCEPT, Reading, Vocabulary, course_frame, read_instructions, wrap_deg,
    write_vocabulary,
)
from ts_transformer.training.train import load_checkpoint_payload

SUMMARY_SCHEMA = "ts-instruction-vocabulary-summary-v1"


def _p50(values) -> float | None:
    values = list(values)
    return float(np.median(values)) if values else None


def _p95(values) -> float | None:
    values = list(values)
    return float(np.percentile(values, 95)) if values else None


def summarise(readings: list[Reading], vocabulary: Vocabulary) -> dict[str, Any]:
    """What the hand check and the bin decision read: how many words a flight carries, how the
    words are used, how far the read targets sit from their bin centres, how many were clamped."""
    per_kind = {kind: [sum(1 for i in r.instructions if i.kind == kind) for r in readings] for kind in INSTRUCTION_KINDS}
    counts = {kind: Counter(i.word for r in readings for i in r.instructions if i.kind == kind) for kind in INSTRUCTION_KINDS}
    residuals = {
        "heading_deg": [abs(wrap_deg(i.target - vocabulary.heading_centre_deg(i.word))) for r in readings for i in r.instructions if i.kind == "heading"],
        "altitude_m": [abs(i.target - vocabulary.altitude_centre_m(i.word)) for r in readings for i in r.instructions if i.kind == "altitude"],
        "speed_mps": [abs(i.target - vocabulary.speed_centre_mps(i.word)) for r in readings for i in r.instructions if i.kind == "speed"],
    }
    positions = sum(len(r.positions_s) for r in readings)
    changes = sum(int((np.diff(r.words[:, : len(MANDATORY_KINDS)], axis=0) != 0).any(axis=1).sum()) for r in readings)
    absorbed = {kind: sum(1 for r in readings for a in r.absorbed if a.kind == kind) for kind in MANDATORY_KINDS}
    orbits = sum(1 for r in readings for a in r.absorbed if a.kind == "heading" and abs(a.change) >= 180.0)
    return {
        "flights": len(readings),
        "instructions_per_flight_p50": {kind: _p50(per_kind[kind]) for kind in INSTRUCTION_KINDS},
        "instructions_per_flight_p95": {kind: _p95(per_kind[kind]) for kind in INSTRUCTION_KINDS},
        "word_counts": {kind: {str(w): int(c) for w, c in sorted(counts[kind].items())} for kind in INSTRUCTION_KINDS},
        "words_used": {kind: len(counts[kind]) for kind in INSTRUCTION_KINDS},
        "clamped": {kind: sum(1 for r in readings for i in r.instructions if i.kind == kind and i.clamped) for kind in ("altitude", "speed")},
        "target_to_bin_centre_p50": {k: _p50(v) for k, v in residuals.items()},
        "target_to_bin_centre_p95": {k: _p95(v) for k, v in residuals.items()},
        "absorbed": absorbed, "absorbed_heading_orbits": orbits,
        "intercept_share": float(np.mean([any(i.kind == "intercept" for i in r.instructions) for r in readings])) if readings else 0.0,
        "established_from_start_share": float(np.mean([r.established_from_start for r in readings])) if readings else 0.0,
        "positions": positions, "positions_with_a_change": changes,
        "duration_s_p50": _p50([r.duration_s for r in readings]),
    }


def render(summary: dict[str, dict[str, Any]], vocabulary: Vocabulary) -> str:
    lines = [f"instruction vocabulary · words {vocabulary.words} · τ {vocabulary.token_step_s:g} s · sha {vocabulary.sha256[:12]}…", ""]
    for split, s in summary.items():
        lines.append(f"{split}: {s['flights']} flights, duration p50 {s['duration_s_p50']:.0f} s, positions {s['positions']} "
                     f"({s['positions_with_a_change']} with a change), intercept on {s['intercept_share']:.2f}, "
                     f"established from the start {s['established_from_start_share']:.2f}")
        lines.append("  instructions / flight p50 (p95): " + ", ".join(
            f"{k} {s['instructions_per_flight_p50'][k]:.0f} ({s['instructions_per_flight_p95'][k]:.0f})" for k in INSTRUCTION_KINDS))
        lines.append("  words used: " + ", ".join(f"{k} {s['words_used'][k]}/{vocabulary.words[k]}" for k in INSTRUCTION_KINDS)
                     + f" · clamped altitude {s['clamped']['altitude']}, speed {s['clamped']['speed']}")
        lines.append("  target − bin centre p50 (p95): " + ", ".join(
            f"{k} {s['target_to_bin_centre_p50'][k]:.1f} ({s['target_to_bin_centre_p95'][k]:.1f})"
            for k in s["target_to_bin_centre_p50"] if s["target_to_bin_centre_p50"][k] is not None))
        lines.append("  absorbed manoeuvres (plateau = the word in force): " + ", ".join(f"{k} {v}" for k, v in s["absorbed"].items())
                     + f" · heading orbits (≥ 180° swept) {s['absorbed_heading_orbits']}")
        for kind in INSTRUCTION_KINDS:
            counts = s["word_counts"][kind]
            total = sum(counts.values()) or 1
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
            lines.append(f"  {kind:<10}" + "  ".join(f"w{w}:{c / total:.2f}" for w, c in top))
    return "\n".join(lines) + "\n"


def hand_check_figure(series, reading: Reading, vocabulary: Vocabulary, path: Path) -> None:
    """One flight's page: the plan view with the course, and the three signals against time with
    the words in force drawn as steps (the human marks 一致 / 漏读 / 误读 in index.csv)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = course_frame(series)
    t = frame["t"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    ax.plot(-frame["to_go_m"] / 1000.0, frame["cross_m"] / 1000.0, lw=1.2)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axvline(0.0, color="r", lw=0.8)
    for item in reading.instructions:
        k = int(np.searchsorted(t, item.issued_s))
        if k < len(t):
            ax.plot(-frame["to_go_m"][k] / 1000.0, frame["cross_m"][k] / 1000.0, "o", ms=4,
                    color={"heading": "C1", "altitude": "C2", "speed": "C3", "intercept": "C4"}[item.kind])
    ax.set_xlabel("along the final approach course, km (threshold at 0)")
    ax.set_ylabel("cross-track, km (right +)")
    ax.set_title(f"{reading.flight_id} · plan view")
    ax.set_aspect("equal", adjustable="datalim")

    def steps(ax, signal, kind, centre, unit, period=None):
        ax.plot(t, signal, lw=1.0, color="0.3")
        issued = sorted((i for i in reading.instructions if i.kind == kind), key=lambda i: i.issued_s)
        for j, item in enumerate(issued):
            end = issued[j + 1].issued_s if j + 1 < len(issued) else t[-1]
            level = centre(item.word)
            if period is not None:      # an unwrapped trace: draw the centre on the turn of the circle the plateau sits on
                k = min(int(np.searchsorted(t, t[-1] if item.settled_s is None else item.settled_s)), len(t) - 1)
                level += period * round((signal[k] - level) / period)
            ax.hlines(level, item.issued_s, end, colors="C1", lw=2.0)
            ax.axvline(item.issued_s, color="C1", lw=0.6, ls=":")
        absorbed = [a for a in reading.absorbed if a.kind == kind]
        for item in absorbed:
            ax.axvspan(item.start_s, item.end_s, color="0.85")
        ax.set_ylabel(f"{kind} ({unit})" + ("; grey = absorbed" if absorbed else ""))

    steps(axes[0, 1], frame["course_unwrapped_deg"], "heading", vocabulary.heading_centre_deg, "deg rel. course, unwrapped", period=360.0)
    steps(axes[1, 0], frame["height_m"], "altitude", vocabulary.altitude_centre_m, "m above threshold")
    steps(axes[1, 1], frame["ground_speed_mps"], "speed", vocabulary.speed_centre_mps, "m/s ground")
    for item in reading.instructions:
        if item.kind == "intercept":
            axes[0, 1].axvline(item.issued_s, color="C4", lw=1.5, ls="--", label=f"intercept w{item.word} ({item.target:.0f}°)")
            axes[0, 1].legend(loc="best", fontsize=8)
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_xlabel("s from the record's start")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--executor", type=Path, required=True, help="a checkpoint.pt: the door to the cohort's data (provenance, config)")
    parser.add_argument("--cohort", type=Path, required=True, help="the development cohort file whose train and val flights are read")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--token-step-s", type=float, default=Vocabulary().token_step_s)
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                        help="override one Vocabulary field (a bin or threshold) for a re-read; repeatable")
    parser.add_argument("--hand-check", type=int, default=300, help="train flights drawn for the human check (0: none)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int, default=0, help="a PREFIX of each split (a smoke test)")
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a vocabulary readout is never overwritten")
    started = time.perf_counter()
    overrides: dict[str, Any] = {"token_step_s": args.token_step_s}
    names = [f.name for f in fields(Vocabulary)]
    for item in args.set:
        name, _, value = item.partition("=")
        if name not in names or not value:
            parser.error(f"--set {item!r}: not a Vocabulary field (one of {names})")
        overrides[name] = tuple(float(v) for v in value.split(",")) if name == "intercept_angle_bins_deg" else float(value)
    vocabulary = Vocabulary(**overrides)

    executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    payload = load_checkpoint_payload(executor)
    config = TSConfig.from_dict(payload["config"])
    cohort_path = args.cohort if args.cohort.is_absolute() else REPO_ROOT / args.cohort
    cohort = load_development_cohort(cohort_path)
    splits = {"train": list(cohort.train_flight_ids), "val": list(cohort.val_flight_ids)}
    if args.limit:
        splits = {name: keys[: args.limit] for name, keys in splits.items()}
    for name, keys in splits.items():
        held = set(payload["split"][name])
        missing = [key for key in keys if key not in held]
        if missing:
            parser.error(f"{len(missing)} {name} flight(s) of the cohort are not in the executor's {name} split (first {missing[0]!r})")
    series = rebuild_cohort(payload, config, [*splits["train"], *splits["val"]])
    by_split = {"train": series[: len(splits["train"])], "val": series[len(splits["train"]) :]}
    print(f"  {len(series)} flights rebuilt; reading instructions at τ = {vocabulary.token_step_s:g} s", flush=True)

    readings = {name: [read_instructions(item, vocabulary) for item in items] for name, items in by_split.items()}
    summary = {name: summarise(items, vocabulary) for name, items in readings.items()}
    out.mkdir(parents=True)
    write_vocabulary(
        out, vocabulary,
        cohort_identity={**development_cohort_audit(cohort_path, cohort), "path": str(cohort_path),
                         "eligible_set_sha256": provenance_eligible_set_digests(payload["data_provenance"])},
        counts={name: summary[name]["word_counts"] for name in summary},
        source={"executor": str(executor), "executor_sha256": file_sha256(executor), "limit": args.limit or None},
    )
    for name, items in readings.items():
        write_json_atomic(out / f"sentences_{name}.json", {"schema": SUMMARY_SCHEMA, "split": name, "vocabulary_sha256": vocabulary.sha256,
                                                          "token_step_s": vocabulary.token_step_s, "flights": [r.to_dict() for r in items]})
    table = render(summary, vocabulary)
    write_json_atomic(out / "summary.json", {"schema": SUMMARY_SCHEMA, "written_utc": utc_now(), "vocabulary_sha256": vocabulary.sha256,
                                            "spec": vocabulary.to_dict(), "splits": summary, "elapsed_s": time.perf_counter() - started})
    (out / "summary.txt").write_text(table, encoding="utf-8")
    print(table)

    if args.hand_check:
        # the draw: seeded; half straight-in, half vectored — the package's strata at the executor's
        # anchor (`approach_difficulty`: straight-in = tortuosity below the fleet's cut; vectored =
        # above it and NOT established at the anchor); a flight in neither (tortuous but already
        # established) is not drawn. A short pool is refused, never refilled silently.
        anchor = default_anchor(config)
        train = by_split["train"]
        difficulty = [approach_difficulty(item, anchor) for item in train]
        pools = {
            STRATUM_STRAIGHT_IN: [i for i, d in enumerate(difficulty) if d.route_tortuosity < STRAIGHT_TORTUOSITY],
            STRATUM_VECTORED: [i for i, d in enumerate(difficulty) if d.route_tortuosity >= STRAIGHT_TORTUOSITY and not d.established_at_anchor],
        }
        half = args.hand_check // 2
        short = {STRATUM_SHORT[name]: len(pool) for name, pool in pools.items() if len(pool) < half}
        if short:
            raise ValueError(f"the hand check draws {half} per stratum; the train split holds {short}")
        rng = np.random.default_rng(args.seed)
        chosen = [(name, int(i)) for name, pool in pools.items() for i in rng.permutation(pool)[:half]]
        folder = out / "hand_check"
        folder.mkdir()
        with (folder / "index.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "flight_id", "stratum", "tortuosity", "established_at_anchor",
                             *(f"n_{kind}" for kind in MANDATORY_KINDS), "intercept_word", "n_absorbed",
                             "sentence (heading/altitude/speed/intercept per position)", "verdict (一致 / 漏读 / 误读)", "note"])
            for k, (stratum, index) in enumerate(chosen):
                item, reading, d = train[index], readings["train"][index], difficulty[index]
                name = f"{k:03d}_{reading.flight_id}.png"
                hand_check_figure(item, reading, vocabulary, folder / name)
                intercept = next((i.word for i in reading.instructions if i.kind == "intercept"), NO_INTERCEPT)
                writer.writerow([name, reading.flight_id, STRATUM_SHORT[stratum], f"{d.route_tortuosity:.2f}", d.established_at_anchor,
                                 *(sum(1 for i in reading.instructions if i.kind == kind) for kind in MANDATORY_KINDS),
                                 intercept, len(reading.absorbed), " ".join("/".join(str(w) for w in row) for row in reading.words), "", ""])
        print(f"  hand check: {len(chosen)} flights ({half} per stratum; pools straight-in {len(pools[STRATUM_STRAIGHT_IN])}, "
              f"vectored {len(pools[STRATUM_VECTORED])} of {len(train)} train flights) in {folder}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
