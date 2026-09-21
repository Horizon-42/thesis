"""Read the BOX vocabulary's word sequences for a cohort and write the artefact.

    python run_ts.py box_vocabulary --airports KRDU KSJC KSTL KSMF KMSY \\
        --cohort <development_cohort.json> --out <dir> [--set FIELD=VALUE ...]

A word is an INTERVAL the track must stay inside (`manoeuvre.box_vocabulary`), so this runner's
own contract is CONTAINMENT: every sentence it writes is checked against the track it was read
from, and a flight whose signals leave the vocabulary's bounds is REFUSED and named rather than
clamped into an edge word. Written under ``--out`` (refused if it exists):

    instruction_vocabulary.json   the spec under its sha, the boxes themselves, the cohort's
                                  runway classes beside it (per airport, so never in the sha),
                                  the cohort identity and every refusal with its reason
    sentences_train.json / sentences_val.json   every flight's events, words and holds
    summary.json / summary.txt    words a flight per kind, classes used, holds, refusals
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
from pathlib import Path
import time
from typing import Any

import numpy as np

from ts_transformer.config import TSConfig
from ts_transformer.data.development_cohorts import development_cohort_audit, load_development_cohort
from ts_transformer.experiments.support import REPO_ROOT, cohort_from_manifests
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.manoeuvre.box_vocabulary import (
    KINDS, SCHEMA, BoxVocabulary, contains, read_boxes,
)
from ts_transformer.manoeuvre.instructions import flight_runway


def summarise(rows: list[dict[str, Any]], vocabulary: BoxVocabulary) -> dict[str, Any]:
    if not rows:
        raise ValueError("a split with no sentences has nothing to summarise")
    events = [len(r["words"]) for r in rows]
    holds = np.concatenate([np.asarray(r["hold_s"]) for r in rows])
    per_kind = {}
    for index, kind in enumerate(KINDS):
        column = [np.asarray(r["words"], dtype=np.int64)[:, index] for r in rows]
        changes = [1 + int((np.diff(c) != 0).sum()) for c in column]
        per_kind[kind] = {
            "classes_used": int(len({int(w) for c in column for w in c})),
            "words_per_flight_p50": float(np.median(changes)),
            "words_per_flight_mean": float(np.mean(changes)),
            "words_per_flight_p95": float(np.percentile(changes, 95)),
        }
    return {
        "flights": len(rows), "events": int(sum(events)),
        "events_per_flight_p50": float(np.median(events)),
        "events_per_flight_mean": float(np.mean(events)),
        "events_per_flight_p95": float(np.percentile(events, 95)),
        "hold_s_p50": float(np.median(holds)), "hold_s_p95": float(np.percentile(holds, 95)),
        "hold_under_4s_share": float(np.mean(holds < 4.0)),
        "per_kind": per_kind,
        "runway_counts": dict(Counter(r["runway"] for r in rows)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--airports", nargs="+", required=True, metavar="ICAO",
                        help="read the cohort straight from these airports' arrival manifests")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                        help="override one BoxVocabulary field (not the reading rule); repeatable")
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a vocabulary readout is never overwritten")
    started = time.perf_counter()

    settable = {f.name: f.type for f in fields(BoxVocabulary) if f.name != "reading_rule"}
    overrides: dict[str, Any] = {}
    for item in args.set:
        name, _, value = item.partition("=")
        if name not in settable or not value:
            parser.error(f"--set {item!r}: not a settable field (one of {sorted(settable)})")
        try:
            overrides[name] = float(value)
        except ValueError:
            parser.error(f"--set {item!r}: {value!r} is not a number")
    vocabulary = BoxVocabulary(**overrides)

    cohort_path = args.cohort if args.cohort.is_absolute() else REPO_ROOT / args.cohort
    cohort = load_development_cohort(cohort_path)
    config = TSConfig()
    series, splits, provenance = cohort_from_manifests(args.airports, cohort, config)
    provenance["config"] = {name: getattr(config, name) for name in ("seq_len", "pred_len", "dt_s", "aircraft_type")}
    print(f"  {len(series)} flights rebuilt; reading boxes under {vocabulary.sha256[:12]}", flush=True)

    idents = sorted({flight_runway(item) for item in series})
    runway_word = {name: index for index, name in enumerate(idents)}
    by_split = {"train": series[: len(splits["train"])], "val": series[len(splits["train"]):]}
    readings: dict[str, list[dict[str, Any]]] = {}
    refused: list[dict[str, str]] = []
    for name, items in by_split.items():
        rows = []
        for item in items:
            try:
                moments, words, holds = read_boxes(item, vocabulary, runway_word[flight_runway(item)])
            except ValueError as error:
                refused.append({"dataset_id": item.dataset_id, "split": name, "why": str(error)})
                continue
            # the contract, checked on the way out: a sentence that does not contain its own
            # track is a bug in the reading, not a flight to record
            if not contains(item, vocabulary, moments, words):
                raise SystemExit(f"{item.dataset_id}: the sentence does not contain its own track")
            rows.append({"dataset_id": item.dataset_id, "flight_id": item.flight_id,
                         "runway": flight_runway(item), "event_times_s": moments.tolist(),
                         "hold_s": holds.tolist(), "words": words.tolist()})
        readings[name] = rows
        print(f"  {name}: {len(rows)} sentences, {len(items) - len(rows)} refused", flush=True)

    summary = {name: summarise(rows, vocabulary) for name, rows in readings.items()}
    out.mkdir(parents=True)
    write_json_atomic(out / "instruction_vocabulary.json", {
        "schema": SCHEMA, "written_utc": utc_now(), "spec": vocabulary.spec,
        "sha256": vocabulary.sha256, "words": {**vocabulary.words, "runway": len(idents)},
        "runway_idents": idents,
        "boxes": {"heading_edges_deg": vocabulary.heading_edges.tolist(),
                  "speed_edges_mps": vocabulary.speed_edges.tolist(),
                  "altitude_targets_m": vocabulary.altitude_targets.tolist()},
        "cohort_identity": {**development_cohort_audit(cohort_path, cohort),
                            "path": str(cohort_path), "provenance": provenance},
        "refused": refused,
    })
    for name, rows in readings.items():
        write_json_atomic(out / f"sentences_{name}.json",
                          {"schema": SCHEMA, "split": name, "vocabulary_sha256": vocabulary.sha256,
                           "runway_idents": idents, "flights": rows})
    write_json_atomic(out / "summary.json", {"schema": SCHEMA, "vocabulary_sha256": vocabulary.sha256,
                                             "spec": vocabulary.spec, "splits": summary,
                                             "refused": len(refused),
                                             "elapsed_s": time.perf_counter() - started})
    lines = [f"box vocabulary {vocabulary.sha256[:12]}  ({vocabulary.reading_rule})",
             "words: " + "  ".join(f"{k} {v}" for k, v in {**vocabulary.words, "runway": len(idents)}.items()), ""]
    for name, block in summary.items():
        lines.append(f"[{name}] {block['flights']} flights, {block['events']} events, "
                     f"{block['events_per_flight_p50']:.0f}/{block['events_per_flight_mean']:.1f} a flight "
                     f"(p95 {block['events_per_flight_p95']:.0f})")
        for kind in ("heading", "altitude", "speed"):
            k = block["per_kind"][kind]
            lines.append(f"    {kind:9s} {k['words_per_flight_p50']:5.1f}/{k['words_per_flight_mean']:5.1f} a flight   "
                         f"classes used {k['classes_used']}")
        lines.append(f"    hold s p50 {block['hold_s_p50']:.0f}  p95 {block['hold_s_p95']:.0f}  "
                     f"under 4 s {block['hold_under_4s_share']:.1%}")
    lines.append(f"refused {len(refused)}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nwrote {out} in {time.perf_counter() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
