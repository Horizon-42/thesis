"""Prior step 0 (prior design §9): measure the data the scene prior will read, on the TRAINING days only.

From one instruction artefact split by operating day (`instruction_signals`, then `instruction_spec` and
`instruction_labels`) and the harvest's tracks rosters:

- the split: days and flights of each part, and how many flights on the sealed test days the per-flight split
  also holds out (the flights a cross-model comparison may use) — read from ``signals.json``, no test flight opened;
- the sentences: how long they are, how many have no predicted step (``N_LOOK`` rows or fewer), and how many are
  already on the final (captured, or cleared to join) at row ``N_LOOK``;
- the landing context: at each predicted step, whether the airport had a landing on a candidate runway in the
  ``CONTEXT_WINDOW_S`` before it (test-day landings are never counted), and how often that window reaches back
  into a sealed test day;
- the scene: at each predicted step, how many other aircraft with a sentence (spoken to) and without one
  (background: the labeller refused them) are in the scene, and whether one of them is a LEADER (§8): landing earlier
  on the same (observed) runway and at most `LEADER_RANGE_M` closer to its threshold (straight-line distances); and
  the segments the flights chain into by overlapping time.

Only train-day flights are in the scene index, so a step near a day boundary misses its neighbours from the
adjacent day (at 09Z, the overnight minimum). Departures and overflights are not in the scenes (§3.2), nor are
arrivals the data plane could not use (too short, no TCH). Writes ``census.json`` into a NEW directory.

    python run_ts.py prior_scene_census --instructions 4dTrajectory/outputs/POOLED/instruction_language/<name> \\
        --out 4dTrajectory/outputs/POOLED/prior/<name>
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ts_transformer.data.day_split import operating_day_span_s
from ts_transformer.instructions.artefact import load_candidates, load_day_split, load_sentences, load_signals, load_spec
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.scene import (
    CONTEXT_WINDOW_S, LEADER_RANGE_M, N_LOOK, Landings, Presence, SceneIndex, context_landings, presence,
)
from ts_transformer.repo_layout import REPO_ROOT, git_state, tracks_manifest_path

SPLIT = "train"
#: Scene sizes above this are counted together.
MOST_COUNTED = 5


def _quantiles(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0}
    return {"n": int(len(array)), "mean": float(array.mean()), "p50": float(np.percentile(array, 50)),
            "p90": float(np.percentile(array, 90)), "p99": float(np.percentile(array, 99)), "max": float(array.max())}


def _histogram(counts: np.ndarray) -> dict[str, int]:
    clipped = np.minimum(counts, MOST_COUNTED)
    return {(f"{k}+" if k == MOST_COUNTED else str(k)): int((clipped == k).sum()) for k in range(MOST_COUNTED + 1)}


def census_airport(speaking: list[Presence], background: list[Presence], landings: Landings,
                   test_spans: np.ndarray) -> dict[str, Any]:
    """One airport. ``speaking``: the flights with a sentence (their rows from `N_LOOK` on are the predicted steps);
    ``test_spans``: ``[M, 2]`` epoch seconds ``[start, end)`` of the sealed test days."""
    index = SceneIndex(speaking + background)
    others_speaking, others_background, leader, gaps, with_landing, into_test = [], [], [], [], [], []
    for ego in speaking:
        rows_s, ego_to_go = ego.times_s[N_LOOK:], ego.to_threshold_m[N_LOOK:]
        if not len(rows_s):
            continue
        near = [p for p in index.overlapping(float(rows_s[0]), float(rows_s[-1])) if p.dataset_id != ego.dataset_id]
        present = np.array([(rows_s >= p.start_s) & (rows_s <= p.end_s) for p in near]).reshape(len(near), len(rows_s))
        is_speaking = np.array([p.speaking for p in near], dtype=bool)
        others_speaking.append(present[is_speaking].sum(axis=0))
        others_background.append(present[~is_speaking].sum(axis=0))
        # the leaders: earlier on the same runway, present, and 0 < (ego's distance − theirs) ≤ the range
        gap = np.full(len(rows_s), np.inf)
        for p, here in zip(near, present):
            if p.runway == ego.runway and p.landing_s < ego.landing_s and here.any():
                ahead = np.where(here, ego_to_go - p.to_threshold_at(rows_s), np.inf)
                gap = np.minimum(gap, np.where((ahead > 0.0) & (ahead <= LEADER_RANGE_M), ahead, np.inf))
        leader.append(np.isfinite(gap))
        gaps.append(gap[np.isfinite(gap)])
        with_landing.append(landings.count_before(rows_s, CONTEXT_WINDOW_S) > 0)
        # the window [t − W, t) overlaps a sealed test day
        into_test.append(((test_spans[None, :, 0] < rows_s[:, None])
                          & (test_spans[None, :, 1] > rows_s[:, None] - CONTEXT_WINDOW_S)).any(axis=1))
    if not others_speaking:
        raise ValueError("no flight with a sentence longer than N_LOOK rows at this airport")
    spoken, silent = np.concatenate(others_speaking), np.concatenate(others_background)
    segments = index.segments()
    return {
        "predicted_steps": int(len(spoken)),
        "other_speaking": {**_quantiles(spoken), "histogram": _histogram(spoken)},
        "other_background": {**_quantiles(silent), "histogram": _histogram(silent)},
        "others_total": {**_quantiles(spoken + silent), "histogram": _histogram(spoken + silent)},
        "leader_share": float(np.concatenate(leader).mean()),
        "leader_gap_m": _quantiles(np.concatenate(gaps)),
        "landing_in_window_share": float(np.concatenate(with_landing).mean()),
        "window_reaches_a_test_day_share": float(np.concatenate(into_test).mean()),
        "segments": {"count": len(segments),
                     "flights": _quantiles([len(s) for s in segments]),
                     "minutes": _quantiles([(max(p.end_s for p in s) - s[0].start_s) / 60.0 for s in segments]),
                     "single_flight_share": float(np.mean([len(s) == 1 for s in segments]))},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True, help="an instruction artefact split by operating day")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)
    directory = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a census is never overwritten")
    started = time.perf_counter()
    record = json.loads((directory / "signals.json").read_text(encoding="utf-8"))
    days = load_day_split(directory)
    test_spans = np.array([operating_day_span_s(day) for day in days.days["test"]], dtype=np.float64).reshape(-1, 2)
    spec = load_spec(directory)
    candidates = load_candidates(directory)
    flights = load_signals(directory, SPLIT)
    sentences = load_sentences(directory, SPLIT, spec)
    offsets = sentences["offsets"]
    rows_of = {int(signal): int(offsets[i + 1] - offsets[i]) for i, signal in enumerate(sentences["signal_index"])}
    print(f"{len(flights)} {SPLIT} flights, {len(rows_of)} with a sentence, {time.perf_counter() - started:.0f}s",
          flush=True)

    by_airport: dict[str, dict[str, list]] = defaultdict(lambda: {"speaking": [], "background": []})
    lengths, established = [], defaultdict(Counter)
    for i, flight in enumerate(flights):
        kind = "speaking" if i in rows_of else "background"
        by_airport[flight.airport][kind].append(presence(flight, rows_of.get(i), candidates[flight.airport]))
    for i, signal in enumerate(sentences["signal_index"]):
        rows = int(sentences["offsets"][i + 1] - sentences["offsets"][i])
        lengths.append(rows)
        if rows <= N_LOOK:          # no predicted step: not in the shares below
            continue
        for group in ("all", flights[int(signal)].airport):
            established[group]["flights"] += 1
            # in the final corridor, to the end, from row N_LOOK on
            established[group]["captured"] += int(sentences["capture_row"][i] <= N_LOOK)
            # cleared before the prior's first predicted step (the clearance is not a word it says)
            established[group]["cleared_before"] += int(sentences["join_row"][i] < N_LOOK)

    airports: dict[str, Any] = {}
    rosters: dict[str, Any] = {}
    for code in sorted(by_airport):
        roster = tracks_manifest_path(code)
        pool = context_landings(roster, [c.ident for c in candidates[code].candidates], days)
        landings = pool.landings()
        rosters[code] = {"tracks_manifest": str(roster), "sha256": file_sha256(roster), "landings_kept": len(landings.times_s),
                         "sealed_test_day_landings_left_out": pool.sealed}
        airports[code] = census_airport(by_airport[code]["speaking"], by_airport[code]["background"], landings, test_spans)
        airports[code]["flights"] = {"speaking": len(by_airport[code]["speaking"]),
                                     "background": len(by_airport[code]["background"])}
        print(f"  {code}: {airports[code]['predicted_steps']} predicted steps, {time.perf_counter() - started:.0f}s",
              flush=True)

    census = {
        "written_utc": utc_now(), "git": git_state(), "instructions": str(directory), "spec_sha256": spec.sha256,
        "split": SPLIT, "n_look": N_LOOK, "context_window_s": CONTEXT_WINDOW_S,
        "day_split": {"counts": {name: len(d) for name, d in days.days.items()}, "seed": days.seed},
        "flights_by_split": record["counts"],
        "test_days": record["test_days"],
        "sentences": {
            "rows": _quantiles(lengths),
            "without_a_predicted_step": int(sum(n <= N_LOOK for n in lengths)),
            "on_the_final_at_n_look": {group: {"flights": c["flights"], "captured_share": c["captured"] / c["flights"],
                                               "cleared_before_share": c["cleared_before"] / c["flights"]}
                                       for group, c in sorted(established.items())},
        },
        "leader_range_m": LEADER_RANGE_M,
        "landing_context": rosters,
        "airports": airports,
        "notes": ["leader: another aircraft in the scene landing earlier on the same observed runway, at most "
                  "leader_range_m closer to its threshold (straight-line distances)",
                  "only train-day flights are in the scene index: steps near a day boundary miss the adjacent day",
                  "departures and overflights are not in the scenes; nor are arrivals without usable signals "
                  "(too short, no TCH)"],
    }
    out.mkdir(parents=True)
    write_json_atomic(out / "census.json", census)
    print(json.dumps({k: census[k] for k in ("day_split", "test_days", "sentences")}, indent=1))
    for code, a in airports.items():
        print(f"{code}: leader {a['leader_share']:.1%}, landing in window {a['landing_in_window_share']:.1%}, "
              f"others p50/p90 {a['others_total']['p50']:.0f}/{a['others_total']['p90']:.0f}, "
              f"segments {a['segments']['count']} (minutes p50 {a['segments']['minutes']['p50']:.0f}, "
              f"max {a['segments']['minutes']['max']:.0f})")
    print(f"wrote {out / 'census.json'} in {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
