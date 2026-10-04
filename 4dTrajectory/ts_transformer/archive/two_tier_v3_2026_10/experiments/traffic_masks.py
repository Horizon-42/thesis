"""Multi-aircraft M0 step 6 (design §6.4): the closed loop's two separation masks (`inference.separation_masks`) on the
labelled words of the training days — how many labelled speed words and approach clearances they would take away (design
§3.4 step 0, the first pass line: at most `PASS_SHARE` of the two together).

The scene is the recorded one on the loop's steps, as the closed loop will see it (`traffic_loop`): every flight with a
sentence along its own rows (`traffic_loop.recorded`: established from the artefact's capture row, each flight re-read by
the labeller and refused unless the reading is its stored sentence), the background arrivals replayed as the census
replays them. For every flight with a sentence, from its first predicted step (row `prior.scene.N_LOOK`, where the prior
says every column) on, at each of its steps — counted as the procedure's check counts (`prior_procedure_check`):

- **speed words**: the word in force at the first predicted step and every speed word said after it; the mask applies
  where both the flight and the aircraft next ahead of it are established and the flight is more than
  `instructions.spec.ATC_NO_SPEED_ASSIGNMENT_DISTANCE_M` from its threshold. A later step that says no speed word while the
  word in force is masked is **forced** (the mask would make the prior say a new word) — reported apart, like the
  procedure check's rule 5. The checks where every word fell short (the mask's fallback, which masks nothing:
  `separation_masks`) are counted;
- **clearances**: "cleared" in force at the first predicted step, and every "cleared" said after it.

What the masks read that is not geometry, read here from the words: each aircraft's speed target in force — its speed
word's, "unspecified" its type's published approach speed (`autopilot.speed.approach_speed_ias_mps`, as published: at the
maximum landing weight, indicated, taken as the speed along the course), its present speed along the course for a flight
whose type publishes none and for a background aircraft (no words); which aircraft are cleared — the approach word in
force, a background aircraft once established; the pace, the executor's own (`autopilot.speed.speed_change_mps2`).

Beside the counts, the speed prediction against the record: at every speed check, the gap it predicts for the word in
force against the gap the pair actually had when the one ahead crossed its threshold (read off its rows), and for every
masked word whether the pair then was under the required distance.

Only training days are read. Writes ``masks.json`` into a NEW directory; ``--airports`` limits it to a smoke test and says
so.

    python run_ts.py traffic_masks --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --out 4dTrajectory/outputs/POOLED/traffic/masks_<date>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from geokit import NM_M

from ts_transformer.autopilot.replay import observed_landing_s
from ts_transformer.autopilot.speed import approach_speed_ias_mps, speed_change_mps2
from ts_transformer.experiments.prior_free_generation import in_force
from ts_transformer.experiments.traffic_census import APPROACH_SPEED_MPS, Track, quantiles, track, traffic_at
from ts_transformer.experiments.traffic_loop import Controlled, at_steps, join, recorded, segments
from ts_transformer.inference.runway_schedule import Separation, faa_separation, read_cwt_tables, wake_category
from ts_transformer.inference.separation_masks import clearance_check, speed_check
from ts_transformer.instructions.artefact import (
    arrival_manifest_sha256s, load_candidates, load_sentences, load_signals, load_spec,
)
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.spec import ATC_NO_SPEED_ASSIGNMENT_DISTANCE_M
from ts_transformer.instructions.words import APPROACH, APPROACH_CLEARED, SPEED, UNCHANGED, Words
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.scene import N_LOOK, presence, scene_steps
from ts_transformer.repo_layout import REPO_ROOT, arrival_manifest_path, git_state, repo_relative

#: v2 (2026-09-28): a fallback masks nothing (design §9 item 20); v1 left the slowest words.
SCHEMA = "ts-traffic-masks-v2"
SPLIT = "train"
#: Design §3.4 step 0, the first pass line (user 2026-09-27, §9 item 5): at most this share of the labelled speed words and
#: clearances together masked.
PASS_SHARE = 0.01
#: What `measure_airport` counts, each over every airport.
COUNTS = ("speed_words", "speed_words_checked", "speed_words_masked", "silent_steps_checked", "forced_steps",
          "speed_checks", "fallbacks", "clearances", "clearances_behind_a_cleared_aircraft", "clearances_masked")


class Spoken:
    """A flight with a sentence in the scene: its controlled rows on the loop's steps, its words in force per row, which
    rows say a word, and its speed candidates, indexed as the speed words: each level's target, then at
    `Words.speed_unspecified` its own approach speed."""

    def __init__(self, aircraft: Controlled, grid: np.ndarray, words: Words, typecode: str | None) -> None:
        self.aircraft, self.grid, self.force = aircraft, np.asarray(grid), in_force(grid)
        own = approach_speed_ias_mps(typecode, None)
        self.own_mps = None if math.isnan(own) else own
        self.levels = np.array([words.speed_mps(i) for i in range(words.speed_unspecified)])

    def candidates(self, row: int) -> np.ndarray:
        own = self.own_mps if self.own_mps is not None else float(self.aircraft.along_speed_mps[row])
        return np.append(self.levels, own)

    def target(self, row: int) -> float:
        return float(self.candidates(row)[self.force[row, SPEED]])

    def cleared(self, row: int) -> bool:
        return bool(self.force[row, APPROACH] == APPROACH_CLEARED)


def measure_airport(spoken: list[Spoken], replayed: list[Track], separation: Separation, step_s: float,
                    accel_mps2: float) -> dict[str, Any]:
    """Every step of every segment: the masks on each flight with a sentence from its first predicted step (module
    docstring)."""
    by_key = {s.aircraft.key: s for s in spoken}
    counts = dict.fromkeys(COUNTS, 0)
    errors, masked_actual, forbidden_events = [], [], []
    for segment in segments([*(s.aircraft for s in spoken), *replayed]):
        flown_here = [a for a in segment if isinstance(a, Controlled)]
        replayed_here = [a for a in segment if not isinstance(a, Controlled)]
        start, end = min(a.first_step_s for a in segment), max(a.last_step_s for a in segment)
        for t_s in scene_steps(start, end, step_s):
            t_s = float(t_s)
            here_c = [a for a in flown_here if a.on_step(t_s)]
            here_r = [a for a in replayed_here if a.on_step(t_s)]
            rows = [int(round((t_s - a.first_step_s) / step_s)) for a in here_c]
            if not any(row >= N_LOOK for row in rows):
                continue
            traffic = join(at_steps(here_c, t_s, step_s), traffic_at(here_r, t_s))
            speeds = np.array([*(a.along_speed_mps[row] for a, row in zip(here_c, rows)),
                               *(float(np.interp(t_s, a.presence.times_s, a.along_rate_mps)) for a in here_r)])
            targets = np.array([*(by_key[a.key].target(row) for a, row in zip(here_c, rows)), *speeds[len(here_c):]])
            cleared = np.array([*(by_key[a.key].cleared(row) for a, row in zip(here_c, rows)),
                                *np.asarray(traffic.established[len(here_c):], dtype=bool)])
            here = [*here_c, *here_r]
            for i, (aircraft, row) in enumerate(zip(here_c, rows)):
                if row < N_LOOK:
                    continue
                flight = by_key[aircraft.key]
                word = int(flight.force[row, SPEED])
                said = row == N_LOOK or flight.grid[row, SPEED] != UNCHANGED
                counts["speed_words"] += said
                check = speed_check(traffic, i, separation, speeds, targets, flight.candidates(row), word, accel_mps2,
                                    ATC_NO_SPEED_ASSIGNMENT_DISTANCE_M)
                if check is not None:
                    counts["speed_checks"] += 1
                    counts["fallbacks"] += check.fallback
                    actual = _actual_gap_m(here[check.leader], aircraft, separation)
                    if actual is not None:
                        errors.append(float(check.gaps_m[word]) - actual)
                    if said:
                        counts["speed_words_checked"] += 1
                        if not check.allowed[word]:
                            counts["speed_words_masked"] += 1
                            masked_actual.append(actual is not None and actual < check.required_m)
                            forbidden_events.append({"key": aircraft.key, "row": row, "column": "speed", "word": word,
                                                     "leader": here[check.leader].key,
                                                     "gap_m": float(check.gaps_m[word]),
                                                     "required_m": check.required_m, "actual_gap_m": actual})
                    else:
                        counts["silent_steps_checked"] += 1
                        counts["forced_steps"] += not check.unchanged_allowed
                clear = (row == N_LOOK and flight.cleared(row)) or flight.grid[row, APPROACH] == APPROACH_CLEARED
                if clear:
                    counts["clearances"] += 1
                    gate = clearance_check(traffic, i, separation, cleared)
                    if gate is not None:
                        counts["clearances_behind_a_cleared_aircraft"] += 1
                        if not gate.allowed:
                            counts["clearances_masked"] += 1
                            forbidden_events.append({"key": aircraft.key, "row": row, "column": "approach",
                                                     "leader": here[gate.leader].key, "gap_m": gate.gap_m,
                                                     "required_m": gate.required_m})
    words = counts["speed_words"] + counts["clearances"]
    masked = counts["speed_words_masked"] + counts["clearances_masked"]
    return {
        **{name: int(value) for name, value in counts.items()},
        "masked_share": masked / words if words else None,
        "speed_prediction_minus_recorded_m": quantiles(errors),
        "masked_speed_words_the_record_then_broke": int(sum(masked_actual)),
        "masked": forbidden_events,
    }


def _actual_gap_m(leader: Controlled | Track, follower: Controlled, separation: Separation) -> float | None:
    """The gap the recorded pair had on the approach clock when the one ahead crossed its threshold — a flight with a
    sentence at its crossing read off its rows, a background one at its roster landing time — the follower interpolated
    at that instant; None when the follower is not in the scene then."""
    crossing = leader.landing_s if isinstance(leader, Controlled) else leader.presence.landing_s
    if not follower.first_step_s <= crossing <= follower.last_step_s:
        return None
    runway = leader.runway[-1] if isinstance(leader, Controlled) else leader.presence.runway
    return separation.along_nm[runway] * NM_M - float(np.interp(crossing, follower.times_s, follower.along_m))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True,
                        help="an instruction artefact split by operating day")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--airports", nargs="+", help="only these airports (a smoke test; the readout says so)")
    args = parser.parse_args(argv)
    directory = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    started = time.perf_counter()
    spec = load_spec(directory)
    words = Words(spec)
    accel = speed_change_mps2(spec)
    candidates = load_candidates(directory)
    signals = load_signals(directory, SPLIT)
    sentences = load_sentences(directory, SPLIT, spec)
    offsets = sentences["offsets"]
    spoken_rows = {int(signal): (int(offsets[k + 1] - offsets[k]), int(sentences["capture_row"][k]), k)
                   for k, signal in enumerate(sentences["signal_index"])}
    wanted = [i for i, f in enumerate(signals) if not args.airports or f.airport in args.airports]
    missing = sorted({signals[i].typecode for i in wanted if signals[i].typecode is not None} - set(read_cwt_tables()))
    if missing:
        raise SystemExit(f"{len(missing)} types are in neither CWT table: {missing} (add each to the supplement "
                         "with its source first; `traffic_census` lists them)")
    recorded_sha = arrival_manifest_sha256s(directory)
    by_airport: dict[str, list[int]] = defaultdict(list)
    for i in wanted:
        by_airport[signals[i].airport].append(i)

    report: dict[str, Any] = {}
    for airport in sorted(by_airport):
        manifest = arrival_manifest_path(airport)
        if file_sha256(manifest) != recorded_sha[airport]:
            raise ValueError(f"{manifest} is not the manifest the artefact's signals were read from "
                             f"(sha256 {recorded_sha[airport][:12]} recorded)")
        separation = faa_separation(json.loads(manifest.read_text(encoding="utf-8"))["runway_targets"],
                                    speed_mps=APPROACH_SPEED_MPS)
        geometry = candidates[airport]
        spoken, replayed = [], []
        for i in by_airport[airport]:
            flight = signals[i]
            category = None if flight.typecode is None else wake_category(flight.typecode)
            if i not in spoken_rows:
                replayed.append(track(flight, None, None, geometry, separation.along_nm[flight.runway] * NM_M, spec,
                                      spec.step_s))
                continue
            rows, capture, k = spoken_rows[i]
            grid = sentences["words"][offsets[k]: offsets[k + 1]]
            reading = read_flight(flight, geometry, spec, words)
            if not np.array_equal(reading.words, grid) or reading.runway_index != int(sentences["runway_index"][k]):
                raise ValueError(f"{flight.dataset_id}: the re-read sentence differs from the stored one")
            aircraft = recorded(presence(flight, rows, geometry, spec.step_s), flight, capture,
                                observed_landing_s(flight, reading, geometry), geometry, separation, category,
                                spec.step_s)
            spoken.append(Spoken(aircraft, grid, words, flight.typecode))
        report[airport] = {"flights_with_a_sentence": len(spoken), "background": len(replayed),
                           **measure_airport(spoken, replayed, separation, spec.step_s, accel)}
        a = report[airport]
        print(f"{airport}: {len(spoken)} flights; speed words masked {a['speed_words_masked']}/{a['speed_words']} "
              f"(checked {a['speed_words_checked']}), clearances masked {a['clearances_masked']}/{a['clearances']}, "
              f"forced steps {a['forced_steps']}/{a['silent_steps_checked']}, fallbacks {a['fallbacks']}/"
              f"{a['speed_checks']}, {time.perf_counter() - started:.0f}s", flush=True)

    totals = {name: sum(a[name] for a in report.values())
              for name in (*COUNTS, "masked_speed_words_the_record_then_broke")}
    masked = totals["speed_words_masked"] + totals["clearances_masked"]
    share = masked / (totals["speed_words"] + totals["clearances"])
    payload = {
        "schema": SCHEMA, "written_utc": utc_now(), "split": SPLIT, "only_airports": args.airports,
        "instructions": repo_relative(directory), "spec_sha256": spec.sha256,
        "arrival_manifest_sha256": {airport: recorded_sha[airport] for airport in sorted(by_airport)},
        "accel_mps2": accel, "no_speed_within_m": ATC_NO_SPEED_ASSIGNMENT_DISTANCE_M,
        "speed_prediction": "each aircraft along its own course from its present speed toward its target at a constant "
                            "pace, no turn, descent or wind (inference.separation_masks)",
        "pass_share": PASS_SHARE, "pass_line": {"masked_share": share, "passes": share <= PASS_SHARE,
                                                 "by_airport": {a: r["masked_share"] for a, r in sorted(report.items())}},
        "totals": totals, "airports": report, "git": git_state(), "seconds": time.perf_counter() - started,
    }
    out.mkdir(parents=True)
    write_json_atomic(out / "masks.json", payload)
    print(f"pooled: {share:.2%} of the labelled speed words and clearances masked (pass line {PASS_SHARE:.0%}: "
          f"{'passes' if share <= PASS_SHARE else 'fails'}); wrote {out / 'masks.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
