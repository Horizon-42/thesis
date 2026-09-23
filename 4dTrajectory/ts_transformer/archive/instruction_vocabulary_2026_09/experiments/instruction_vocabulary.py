"""Read the instruction words of a cohort's tracks and write the vocabulary artefact, the sentences and the hand-check sample (two-tier v3 stage B, step B0′; plan §5.2.1, §5.2.5).

    python run_ts.py instruction_vocabulary --executor <ckpt> --cohort <development_cohort.json> --out <dir> \\
        [--token-step-s 10] [--set FIELD=VALUE ...] [--hand-check 300] [--seed 1337]

The cohort is a development cohort file (D33 / D47: the grid's L60_D60 cohort); its flights are
rebuilt through the checkpoint's data provenance (`support.rebuild_cohort`, C25) under the
checkpoint's config — the checkpoint is the door to the data, nothing of it is read. Every flight
of both splits is read (`instructions.read_instructions`) under the module's starting bins and
thresholds (D51 / D53) at ``--token-step-s`` (D54). Written under ``--out`` (refused if it exists):

    instruction_vocabulary.json    the spec under its sha, the cohort's runway classes BESIDE it
                                   (`runway_idents`, D62: per airport, so never in the sha), the
                                   cohort identity, per-word counts over the TRAIN split (the val
                                   counts beside them)
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
import math
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
from ts_transformer.experiments.support import REPO_ROOT, cohort_from_manifests, cohort_splits, rebuild_cohort
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.data.runway_context import wrap_deg
from ts_transformer.manoeuvre.instructions import (
    ABSORBED_SAME_WORD, ABSORBED_SHORT_SEGMENT, ABSORBED_SHORT_TAIL, ABSORBED_SMALL_CHANGE, INSTRUCTION_KINDS,
    MANDATORY_KINDS, NO_INTERCEPT,
    Reading, RunwayVocabulary, Vocabulary, course_frame, flight_runway, read_instructions, word_counts, write_vocabulary,
)
from ts_transformer.training.train import load_checkpoint_payload

SUMMARY_SCHEMA = "ts-instruction-vocabulary-summary-v1"


def _p50(values) -> float:
    return float(np.median(list(values)))


def _p95(values) -> float:
    return float(np.percentile(list(values), 95))


def summarise(readings: list[Reading], vocabulary: Vocabulary) -> dict[str, Any]:
    """What the hand check and the bin decision read: how many words a flight carries, how the
    words are used, how far the UNCLAMPED targets sit from their bin centres (a clamped one
    measures the range, not the bin), how many were clamped, what was absorbed and why."""
    if not readings:
        raise ValueError("a split with no flights has nothing to summarise")
    # The duration and terminal words are COLUMNS computed by `sentence()`, not `Instruction`
    # objects, so counting them off the instruction list reads 0 for both — which is what the
    # first six-kind run printed ("duration 0/151, terminal 0/3") although every event carries
    # one. Both counts come off the word matrix, which is the sentence itself.
    columns = {kind: INSTRUCTION_KINDS.index(kind) for kind in INSTRUCTION_KINDS}
    spoken = ("heading", "vertical", "speed", "runway")     # the kinds an Instruction is issued for
    per_kind = {kind: ([sum(1 for i in r.instructions if i.kind == kind) for r in readings] if kind in spoken
                       else [len(r.words) for r in readings])
                for kind in INSTRUCTION_KINDS}
    counts = {kind: Counter(int(w) for r in readings for w in r.words[:, columns[kind]])
              for kind in INSTRUCTION_KINDS}
    unclamped = [i for r in readings for i in r.instructions if not i.clamped]
    residuals = {
        # the established word names a line, not a bearing, so it has no distance-to-centre to
        # report — like the runway word, it is excluded rather than given a fake 0
        "heading_deg": [abs(wrap_deg(i.target - vocabulary.heading_centre_deg(i.word)))
                        for i in unclamped if i.kind == "heading"
                        and not (vocabulary.use_established_word and i.word == vocabulary.heading_direction_words)],
        "vertical_deg": [abs(i.target - vocabulary.vertical_centre_deg(i.word)) for i in unclamped if i.kind == "vertical"],
        "speed_mps": [abs(i.target - vocabulary.speed_centre_mps(i.word)) for i in unclamped if i.kind == "speed"],
    }
    events = sum(len(r.event_times_s) for r in readings)
    # every event IS a change now (D52) — what is worth counting is how many, and the gaps
    gaps = [float(g) for r in readings for g in np.diff(r.event_times_s)]
    duration_clamped = sum(r.duration_clamped for r in readings)
    reasons = (ABSORBED_SAME_WORD, ABSORBED_SMALL_CHANGE, ABSORBED_SHORT_TAIL, ABSORBED_SHORT_SEGMENT)
    absorbed = {kind: {reason: sum(1 for r in readings for a in r.absorbed if a.kind == kind and a.reason == reason) for reason in reasons}
                for kind in MANDATORY_KINDS}
    orbits = sum(1 for r in readings for a in r.absorbed if a.kind == "heading" and abs(a.change) >= 180.0)
    return {
        "flights": len(readings),
        "instructions_per_flight_p50": {kind: _p50(per_kind[kind]) for kind in INSTRUCTION_KINDS},
        "instructions_per_flight_p95": {kind: _p95(per_kind[kind]) for kind in INSTRUCTION_KINDS},
        "word_counts": {kind: {str(w): int(c) for w, c in sorted(counts[kind].items())} for kind in INSTRUCTION_KINDS},
        "words_used": {kind: len(counts[kind]) for kind in INSTRUCTION_KINDS},
        "outside": {kind: sum(1 for r in readings for i in r.instructions if i.kind == kind and i.clamped) for kind in ("vertical", "speed")},
        "target_to_bin_centre_p50": {k: _p50(v) for k, v in residuals.items()},
        "target_to_bin_centre_p95": {k: _p95(v) for k, v in residuals.items()},
        "absorbed": absorbed, "absorbed_heading_orbits": orbits,
        "established_from_start_share": float(np.mean([r.established_from_start for r in readings])),
        # what the vertical fit's segment CEILING costs, stated rather than implied: the RMS
        # height error of the words' own profile against the track, and how many fitted pieces
        # read as one word and were folded
        "vertical_fit_rms_m_p50": _p50([r.vertical_fit_rms_m for r in readings]),
        "vertical_fit_rms_m_p95": _p95([r.vertical_fit_rms_m for r in readings]),
        "vertical_pieces_merged": sum(r.vertical_pieces_merged for r in readings),
        "events": events, "events_per_flight_p50": _p50([len(r.event_times_s) for r in readings]),
        "gap_s_p50": _p50(gaps), "gap_s_p95": _p95(gaps), "duration_clamped": duration_clamped,
        "duration_s_p50": _p50([r.duration_s for r in readings]),
    }


def render(summary: dict[str, dict[str, Any]], vocabulary: Vocabulary, runway_vocabulary: RunwayVocabulary) -> str:
    # the runway's classes are the cohort's, not the spec's (D62): its count comes from the
    # runway vocabulary, everything else from the hashed spec
    words = word_counts(vocabulary, runway_vocabulary)
    lines = [f"instruction vocabulary · words {words} · event sequence (D52) · sha {vocabulary.sha256[:12]}… "
             f"· runways {', '.join(runway_vocabulary.idents)}", ""]
    for split, s in summary.items():
        lines.append(f"{split}: {s['flights']} flights, duration p50 {s['duration_s_p50']:.0f} s, "
                     f"{s['events']} events (p50 {s['events_per_flight_p50']:.0f} a flight; gap p50 "
                     f"{s['gap_s_p50']:.0f} s, p95 {s['gap_s_p95']:.0f} s, {s['duration_clamped']} gaps clamped "
                     f"at the {vocabulary.duration_max_s:.0f} s ceiling), "
                     f"established from the start {s['established_from_start_share']:.2f}")
        lines.append("  instructions / flight p50 (p95): " + ", ".join(
            f"{k} {s['instructions_per_flight_p50'][k]:.0f} ({s['instructions_per_flight_p95'][k]:.0f})" for k in INSTRUCTION_KINDS))
        lines.append("  words used: " + ", ".join(f"{k} {s['words_used'][k]}/{words[k]}" for k in INSTRUCTION_KINDS)
                     + f" · outside the ends: vertical {s['outside']['vertical']}, speed {s['outside']['speed']}")
        lines.append("  unclamped target − bin centre p50 (p95): " + ", ".join(
            f"{k} {s['target_to_bin_centre_p50'][k]:.1f} ({s['target_to_bin_centre_p95'][k]:.1f})" for k in s["target_to_bin_centre_p50"]))
        lines.append("  absorbed manoeuvres (same word / small change / short tail / short segment): "
                     + ", ".join(f"{k} " + "/".join(str(v[reason]) for reason in
                                                    (ABSORBED_SAME_WORD, ABSORBED_SMALL_CHANGE, ABSORBED_SHORT_TAIL, ABSORBED_SHORT_SEGMENT))
                                 for k, v in s["absorbed"].items())
                     + f" · heading orbits (≥ 180°) {s['absorbed_heading_orbits']}")
        # what the vertical fit's ceiling of `vertical_segments` costs — never silent
        lines.append(f"  vertical fit against the track: RMS p50 {s['vertical_fit_rms_m_p50']:.0f} m, "
                     f"p95 {s['vertical_fit_rms_m_p95']:.0f} m; {s['vertical_pieces_merged']} fitted pieces "
                     f"read as the word beside them and were folded")
        for kind in INSTRUCTION_KINDS:
            counts = s["word_counts"][kind]
            total = sum(counts.values()) or 1
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
            lines.append(f"  {kind:<10}" + "  ".join(f"w{w}:{c / total:.2f}" for w, c in top)
                         + (f"   (top 8 of {len(counts)} words)" if len(counts) > 8 else ""))
    return "\n".join(lines) + "\n"


def _vertical_panel(ax, t, frame, reading, vocabulary) -> None:
    """The vertical panel plots HEIGHT, and the words on it are the PROFILE each angle draws.

    Drawing `hlines` at the word's centre — which is what every other kind's panel does, and what
    this one did until 2026-09-21 — puts lines at −3…+4.4 on an axis that runs 0–3000 m, so every
    word collapses onto y ≈ 0 and the page a human marks 一致/漏读/误读 on cannot show whether the
    vertical words are right. The word is a RATE, so what it says about this panel is a slope:
    each instruction's angle held from the height where it was issued.
    """
    ax.plot(t, frame["height_m"], lw=1.0, color="0.3")
    issued = sorted((i for i in reading.instructions if i.kind == "vertical"), key=lambda i: i.issued_s)
    speed = frame["ground_speed_mps"]
    for j, item in enumerate(issued):
        end = issued[j + 1].issued_s if j + 1 < len(issued) else t[-1]
        a, b = int(np.searchsorted(t, item.issued_s)), int(np.searchsorted(t, end))
        b = min(b, len(t) - 1)
        if b <= a:
            continue
        distance = float(np.trapezoid(speed[a:b + 1], t[a:b + 1]))
        height = float(frame["height_m"][a])
        drop = math.tan(math.radians(vocabulary.vertical_centre_deg(item.word))) * distance
        ax.plot([t[a], t[b]], [height, height - drop], color="C1", lw=2.0)
        ax.axvline(item.issued_s, color="C1", lw=0.6, ls=":")
        if item.settled_s is not None:
            ax.axvline(item.settled_s, color="C2", lw=0.6, ls=":")
    absorbed = [a for a in reading.absorbed if a.kind == "vertical"]
    for item in absorbed:
        ax.axvspan(item.start_s - 1.0, item.end_s + 1.0, color="0.85")
    ax.set_ylabel("height above the threshold (m); C1 = the angle words"
                  + ("; grey = absorbed" if absorbed else ""))


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
        if item.kind == "runway":
            continue                    # not a point on the track: it is the frame every other word is read in
        k = int(np.searchsorted(t, item.issued_s))
        ax.plot(-frame["to_go_m"][k] / 1000.0, frame["cross_m"][k] / 1000.0, "o", ms=4,
                color={"heading": "C1", "vertical": "C2", "speed": "C3", "runway": "C4"}[item.kind])
    ax.set_xlabel("along the final approach course, km (threshold at 0)")
    ax.set_ylabel("cross-track, km (right +)")
    ax.set_title(f"{reading.flight_id} · runway {reading.runway} · plan view")
    ax.set_aspect("equal", adjustable="datalim")

    def steps(ax, signal, kind, centre, unit, period=None):
        ax.plot(t, signal, lw=1.0, color="0.3")
        issued = sorted((i for i in reading.instructions if i.kind == kind), key=lambda i: i.issued_s)
        for j, item in enumerate(issued):
            end = issued[j + 1].issued_s if j + 1 < len(issued) else t[-1]
            if kind == "heading" and vocabulary.use_established_word and item.word == vocabulary.heading_direction_words:
                ax.axvspan(item.issued_s, end, color="C2", alpha=0.12)    # holding the line, not a bearing
                continue
            level = centre(item.word)
            if period is not None:      # an unwrapped trace: draw the centre on the turn of the circle the plateau sits on
                k = int(np.searchsorted(t, t[-1] if item.settled_s is None else item.settled_s))
                level += period * round((signal[k] - level) / period)
            ax.hlines(level, item.issued_s, end, colors="C1", lw=2.0)
            ax.axvline(item.issued_s, color="C1", lw=0.6, ls=":")                    # issued (orange)
            if item.settled_s is not None:
                ax.axvline(item.settled_s, color="C2", lw=0.6, ls=":")               # settled (green)
        absorbed = [a for a in reading.absorbed if a.kind == kind]
        for item in absorbed:           # at least a row wide, so a zero-length one still shows
            ax.axvspan(item.start_s - 1.0, item.end_s + 1.0, color="0.85")
        ax.set_ylabel(f"{kind} ({unit})" + ("; grey = absorbed" if absorbed else ""))

    steps(axes[0, 1], frame["course_unwrapped_deg"], "heading", vocabulary.heading_centre_deg, "deg rel. course, unwrapped", period=360.0)
    _vertical_panel(axes[1, 0], t, frame, reading, vocabulary)
    steps(axes[1, 1], frame["ground_speed_mps"], "speed", vocabulary.speed_centre_mps, "m/s ground")
    # D72: mark where the sentence says it lands — the terminal word, on every panel's clock
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.axvline(reading.event_times_s[-1], color="C4", lw=1.2, ls="--", label="landed")
    axes[0, 1].legend(loc="best", fontsize=8)
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_xlabel("s from the record's start")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--executor", type=Path, help="a checkpoint.pt: the door to the cohort's data (provenance, config)")
    parser.add_argument("--airports", nargs="+", metavar="ICAO",
                        help="read the cohort straight from these airports' arrival manifests instead. "
                             "The vocabulary is read from TRACKS, not from a model, so a trained executor is "
                             "not needed to read it — but the executor is also what carries the data provenance, "
                             "so this path records the manifests' own digests in its place. Use it when no "
                             "executor exists for the cohort (a pooled cohort has none).")
    parser.add_argument("--cohort", type=Path, required=True, help="the development cohort file whose train and val flights are read")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--token-step-s", type=float, default=Vocabulary().token_step_s)
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                        help="override one Vocabulary field (a bin or threshold; not the reading rule or the established rule) for a re-read; repeatable")
    parser.add_argument("--hand-check", type=int, default=300, help="train flights drawn for the human check (0: none)")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a vocabulary readout is never overwritten")
    started = time.perf_counter()
    overrides: dict[str, Any] = {"token_step_s": args.token_step_s}
    settable = [f.name for f in fields(Vocabulary) if f.name != "reading_rule"]
    for item in args.set:
        name, _, value = item.partition("=")
        if name not in settable or not value:
            parser.error(f"--set {item!r}: not a settable Vocabulary field (one of {settable})")
        # the type belongs at the boundary: `vertical_segments` and `vertical_fit_points` are
        # COUNTS, and a float there passes `__post_init__` and dies inside the fit with a
        # TypeError about integers
        wanted = {f.name: f.type for f in fields(Vocabulary)}[name]
        try:
            overrides[name] = (value not in ("0", "false", "False") if wanted in ("bool", bool)
                               else int(value) if wanted in ("int", int) else float(value))
        except ValueError:
            parser.error(f"--set {item!r}: {value!r} is not a {wanted}")
    vocabulary = Vocabulary(**overrides)
    if args.hand_check % 2:
        parser.error(f"--hand-check draws half per stratum: {args.hand_check} is odd")

    if bool(args.executor) == bool(args.airports):
        parser.error("give exactly one of --executor (a checkpoint's provenance) or --airports (the manifests')")
    cohort_path = args.cohort if args.cohort.is_absolute() else REPO_ROOT / args.cohort
    cohort = load_development_cohort(cohort_path)
    # NO --limit here, deliberately (removed 2026-09-20 with D62): the runway CLASSES are read off
    # these flights and they sit OUTSIDE the sha, so a limited run would write a fully valid,
    # sha-identical artefact whose class set is only a prefix's, and nothing downstream could tell
    # the difference. This runner reads the whole cohort or it does not write.
    if args.airports:
        config = TSConfig()
        series, splits, provenance = cohort_from_manifests(args.airports, cohort, config)
        # WHICH config: `build_series` skips a track shorter than one window, and the window is
        # this config's, so the population read depends on it. A default nobody wrote down is not
        # provenance.
        provenance["config"] = {name: getattr(config, name) for name in ("seq_len", "pred_len", "dt_s", "aircraft_type")}
    else:
        executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
        payload = load_checkpoint_payload(executor)
        config = TSConfig.from_dict(payload["config"])
        provenance = {"read_from": "executor checkpoint", "path": str(executor),
                      "executor_sha256": file_sha256(executor),
                      "eligible_set_sha256": provenance_eligible_set_digests(payload["data_provenance"])}
        splits = cohort_splits(payload, cohort, 0)
        series = rebuild_cohort(payload, config, [*splits["train"], *splits["val"]])
    by_split = {"train": series[: len(splits["train"])], "val": series[len(splits["train"]) :]}
    print(f"  {len(series)} flights rebuilt; reading instructions at τ = {vocabulary.token_step_s:g} s", flush=True)

    # the runway word's classes: the thresholds BOTH splits land on, sorted (D62) — a per-airport
    # set, carried beside the spec so the sha stays one vocabulary's across airports
    runway_vocabulary = RunwayVocabulary.from_idents(flight_runway(item) for item in series)
    readings = {name: [read_instructions(item, vocabulary, runway_vocabulary) for item in items] for name, items in by_split.items()}
    summary = {name: summarise(items, vocabulary) for name, items in readings.items()}
    out.mkdir(parents=True)
    write_vocabulary(
        out, vocabulary, runway_vocabulary=runway_vocabulary,
        cohort_identity={**development_cohort_audit(cohort_path, cohort), "path": str(cohort_path),
                         "provenance": provenance},
        counts={name: summary[name]["word_counts"] for name in summary},
        # ONE door decided this run (the parser refuses both and neither), and `provenance` is
        # already that door's own record — built by `cohort_from_manifests` or beside the
        # checkpoint read. Deriving the source from it keeps the two from disagreeing; deriving it
        # from a variable only one branch binds is what made this raise after reading 26,382
        # flights (2026-09-21).
        source=provenance,
    )
    for name, items in readings.items():
        write_json_atomic(out / f"sentences_{name}.json", {"schema": SUMMARY_SCHEMA, "split": name, "vocabulary_sha256": vocabulary.sha256,
                                                          "runway_idents": list(runway_vocabulary.idents),
                                                          "token_step_s": vocabulary.token_step_s, "flights": [r.to_dict() for r in items]})
    table = render(summary, vocabulary, runway_vocabulary)
    (out / "summary.txt").write_text(table, encoding="utf-8")
    print(table)

    draw: dict[str, Any] | None = None
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
                             *(f"n_{kind}" for kind in MANDATORY_KINDS), "runway", "n_events", "n_absorbed",
                             f"sentence ({'/'.join(INSTRUCTION_KINDS)} per event)",
                             "verdict (一致 / 漏读 / 误读)", "note"])
            for k, (stratum, index) in enumerate(chosen):
                item, reading, d = train[index], readings["train"][index], difficulty[index]
                name = f"{k:03d}_{reading.flight_id}.png"
                hand_check_figure(item, reading, vocabulary, folder / name)
                writer.writerow([name, reading.flight_id, STRATUM_SHORT[stratum], f"{d.route_tortuosity:.2f}", d.established_at_anchor,
                                 *(sum(1 for i in reading.instructions if i.kind == kind) for kind in MANDATORY_KINDS),
                                 reading.runway, len(reading.event_times_s), len(reading.absorbed),
                                 " ".join("/".join(str(w) for w in row) for row in reading.words), "", ""])
        draw = {"pages": len(chosen), "per_stratum": half, "seed": args.seed, "anchor": anchor, "split": "train",
                "pools": {STRATUM_SHORT[name]: len(pool) for name, pool in pools.items()},
                "in_neither_stratum": len(train) - sum(len(pool) for pool in pools.values())}
        print(f"  hand check: {len(chosen)} flights ({half} per stratum; pools straight-in {len(pools[STRATUM_STRAIGHT_IN])}, "
              f"vectored {len(pools[STRATUM_VECTORED])} of {len(train)} train flights, {draw['in_neither_stratum']} in neither) in {folder}", flush=True)
    write_json_atomic(out / "summary.json", {"schema": SUMMARY_SCHEMA, "written_utc": utc_now(), "vocabulary_sha256": vocabulary.sha256,
                                            "spec": vocabulary.to_dict(), "runway_idents": list(runway_vocabulary.idents),
                                            "splits": summary, "hand_check": draw,
                                            "elapsed_s": time.perf_counter() - started})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
