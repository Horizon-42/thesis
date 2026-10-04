"""The labeller's readout (vocabulary §7.1): completeness, envelopes with their widths, sentence length, go-arounds and
class usage — per split, per airport, per stratum.

The stratum is read off the reading: VECTORED when the track's turns before the capture — each run of rows turning
one way, its net turn (`LateralReading.turning_deg`) — add up to at least `VECTORED_TURN_DEG`, STRAIGHT-IN otherwise.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

from ts_transformer.instructions.labeller.interval import in_force
from ts_transformer.instructions.labeller.read import Reading
from ts_transformer.instructions.words import (
    ALTITUDE, ANGLE, COLUMNS, HEADING, SPEED, UNCHANGED, Words,
)

VECTORED_TURN_DEG = 90.0
STRATA = ("straight-in", "vectored")


def stratum(reading: Reading) -> str:
    """The flight's stratum (module docstring)."""
    return "vectored" if reading.checks["turning_deg"] >= VECTORED_TURN_DEG else "straight-in"


def go_around_in_force(reading: Reading, words: Words) -> list[dict[str, bool]]:
    """At each go-around row (D26, vocabulary §9.4): whether "no level-off" and "unspecified" are in force (said before the
    row), whether the approach it ends had a capture row, whether the next approach is to another runway, and whether
    the go-around's lowest point lies past the threshold (where the capture corridor has ended)."""
    held = in_force(reading.words)
    out = []
    for position, row in enumerate(reading.go_around_rows):
        ending, following = reading.approaches[position], reading.approaches[position + 1]
        out.append({"no_level_off": bool(held[row - 1, ALTITUDE] == words.altitude_no_level_off),
                    "unspecified": bool(held[row - 1, SPEED] == words.speed_unspecified),
                    "captured": ending.capture_row is not None,
                    "other_runway": following.runway_index != ending.runway_index,
                    "past_threshold": reading.checks["go_arounds"][position]["along_m"] > 0.0})
    return out


def flight_record(reading: Reading, words: Words) -> dict[str, Any]:
    """The compact per-flight record the summary pools (and `labels.json` keeps)."""
    after = reading.words[1:] != UNCHANGED
    checks = reading.checks
    heading, vertical, speed = checks["heading"], checks["vertical"], checks["speed"]
    return {
        "dataset_id": reading.dataset_id, "airport": reading.airport, "status": "labelled",
        "rows": int(len(reading.words)),
        "stratum": stratum(reading),
        "turning_deg": checks["turning_deg"],
        "words_after_step0": {COLUMNS[c]: int(after[:, c].sum()) for c in range(len(COLUMNS))},
        "silent_steps": int((~after.any(axis=1)).sum()),
        "heading_words_judged": sum(1 for h in heading if h["rows"]),
        "heading_contained": sum(1 for h in heading if h["rows"] and h["inside"] == h["rows"]),
        "heading_rows": sum(h["rows"] for h in heading), "heading_rows_inside": sum(h["inside"] for h in heading),
        "capture_before_threshold_m": checks["capture_before_threshold_m"],
        "go_arounds": len(reading.go_around_rows),
        "go_around_in_force": go_around_in_force(reading, words),
        "altitude_words": len(vertical), "altitude_contained": sum(1 for v in vertical if v["contained"]),
        "altitude_rows": sum(v["rows"] for v in vertical), "altitude_rows_inside": sum(v["inside"] for v in vertical),
        "tube_width_end_m": [v["tube_width_end_m"] for v in vertical],
        "no_level_off_tube_width_end_m": [v["tube_width_end_m"] for v in vertical if v["level_m"] is None],
        "speed_words": len(speed), "speed_contained": sum(1 for v in speed if v["contained"]),
        "speed_transition_ok": sum(1 for v in speed if v["transition_ok"]),
        "speed_cut_before_arrival": sum(1 for v in speed if v["cut_before_arrival"]),
        "speed_accel_ok": sum(1 for v in speed if v["accel_ok"]),
        "speed_band_rows": sum(v["band_rows"] for v in speed), "speed_band_inside": sum(v["band_inside"] for v in speed),
        "capture_row": reading.capture_row, "unspecified_row": reading.unspecified_row,
        "go_around_rows": list(reading.go_around_rows), "runway_again_rows": list(reading.runway_again_rows),
        "cut_at_crossing": reading.cut_at_crossing,
    }


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return {"n": 0}
    return {"n": int(len(array)), "mean": float(array.mean()), "p5": float(np.percentile(array, 5)),
            "p50": float(np.percentile(array, 50)),
            "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def _share(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def summarise_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Every readout number over one group of labelled flights."""
    total = lambda key: sum(r[key] for r in records)  # noqa: E731
    return {
        "flights": len(records),
        "words_per_flight": {column: _quantiles(r["words_after_step0"][column] for r in records) for column in COLUMNS},
        "instructions_per_flight": _quantiles(sum(r["words_after_step0"].values()) for r in records),
        "non_silent_step_share": _share(sum(r["rows"] - 1 - r["silent_steps"] for r in records),
                                        sum(r["rows"] - 1 for r in records)),
        "cut_at_crossing_share": _share(sum(r["cut_at_crossing"] for r in records), len(records)),
        "turning_deg": _quantiles(r["turning_deg"] for r in records),
        "heading": {"judged": total("heading_words_judged"),
                    "contained_share": _share(total("heading_contained"), total("heading_words_judged")),
                    "row_share": _share(total("heading_rows_inside"), total("heading_rows"))},
        "capture_before_threshold_m": _quantiles(r["capture_before_threshold_m"] for r in records),
        "go_arounds": {"flights": sum(r["go_arounds"] > 0 for r in records), "words": total("go_arounds"),
                       **{key: sum(item[key] for r in records for item in r["go_around_in_force"])
                          for key in ("no_level_off", "unspecified", "captured", "other_runway", "past_threshold")}},
        "altitude": {"words": total("altitude_words"),
                     "contained_share": _share(total("altitude_contained"), total("altitude_words")),
                     "row_share": _share(total("altitude_rows_inside"), total("altitude_rows")),
                     "tube_width_end_m": _quantiles(w for r in records for w in r["tube_width_end_m"]),
                     "no_level_off_tube_width_end_m": _quantiles(w for r in records
                                                                  for w in r["no_level_off_tube_width_end_m"])},
        "speed": {"words": total("speed_words"),
                  "contained_share": _share(total("speed_contained"), total("speed_words")),
                  "transition_ok_share": _share(total("speed_transition_ok"), total("speed_words")),
                  "cut_before_arrival_share": _share(total("speed_cut_before_arrival"), total("speed_words")),
                  "accel_ok_share": _share(total("speed_accel_ok"), total("speed_words")),
                  "band_row_share": _share(total("speed_band_inside"), total("speed_band_rows"))},
    }


def class_usage(word_rows: np.ndarray, words: Words) -> dict[str, Any]:
    """How often each class is written (step 0 included), per column; empty and rare classes."""
    counts = words.class_counts()
    usage: dict[str, Any] = {}
    for column in (HEADING, ALTITUDE, ANGLE, SPEED):
        name = COLUMNS[column]
        values = word_rows[:, column]
        tally = np.bincount(values[values != UNCHANGED].astype(np.int64), minlength=counts[name])
        used = tally[tally > 0]
        usage[name] = {"classes": counts[name], "used": int((tally > 0).sum()),
                       "rare_under_10": int(((tally > 0) & (tally < 10)).sum()),
                       "counts": tally.tolist(), "min_used": int(used.min()) if len(used) else 0}
    return usage


def summarise(records: list[dict[str, Any]], refusals: list[dict[str, Any]]) -> dict[str, Any]:
    by_airport: dict[str, list] = defaultdict(list)
    by_stratum: dict[str, list] = defaultdict(list)
    for record in records:
        by_airport[record["airport"]].append(record)
        by_stratum[record["stratum"]].append(record)
    refused_by_airport: dict[str, Counter] = defaultdict(Counter)
    for item in refusals:
        refused_by_airport[item["airport"]][item["reason"]] += 1
    return {
        "labelled": len(records), "refused": len(refusals),
        "refusal_reasons": dict(Counter(item["reason"] for item in refusals).most_common()),
        "refused_by_airport": {a: dict(c.most_common()) for a, c in sorted(refused_by_airport.items())},
        "all": summarise_group(records),
        "by_airport": {a: summarise_group(r) for a, r in sorted(by_airport.items())},
        "by_stratum": {s: summarise_group(by_stratum[s]) for s in STRATA if by_stratum[s]},
    }
