"""The glidepath lower edge checked on labelled data before any model trains through it (post-training design §3.6).

The second stage's step 0. On every labelled flight of the split:

- **words** — the labelled altitude words the mask would forbid where the prior would say them: the word in force at
  the first predicted step (row `N_LOOK`, where every column is said) and every word said after it (rules 1 and 2, a
  level and "descend to land" counted apart);
- **forced** — the later steps where nothing is said but the word in force fails here, so the mask would make the prior
  say a new one (rule 3);
- **observed** — the tracks with a row more than the track tolerance below its floor;

and on a seeded per-airport sample flown on its own dynamics (``--replay-per-airport``), the labelled words flown by the
executor from the first predicted step as free generation flies its reference (`prior_free_generation.fly_reference`):
**stopped** — a flown step (every ``step_s``) more than the track tolerance below its floor (§3.5), found by the check
free generation and training use (`prior_free_generation.glidepath_stops`).

The pass lines, written before the run (user 2026-09-25, design §3.6): at most `MAX_FORBIDDEN_WORDS` of the labelled
altitude words forbidden and at most `MAX_STOPPED_REPLAYS` of the replayed flights stopped; failing either, the numbers go
back to the user.

Writes ``--out`` (a new directory, from a clean tree): ``check.json``.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.judge import flown_track
from ts_transformer.experiments.prior_free_generation import (
    flight_rows, fly_reference, glidepath_stops, in_force, reference_grid,
)
from ts_transformer.instructions.artefact import SPLITS, load_candidates, load_sentences, load_signals
from ts_transformer.instructions.words import ALTITUDE, RUNWAY, UNCHANGED, Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.prior.procedure import (
    GLIDEPATH_BELOW_M, RunwayProcedure, altitude_word_allowed, below_floor, published_procedures, track_tolerance_m,
    word_tolerance_m,
)
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.repo_layout import REPO_ROOT, git_state

CHECK_SCHEMA = "ts-prior-procedure-check-v1"
#: The pass lines of design §3.6 (user 2026-09-25).
MAX_FORBIDDEN_WORDS = 0.01
MAX_STOPPED_REPLAYS = 0.03


def labelled_rows(split: str, instructions: Path, words: Words) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """Every labelled flight's sentence rows, pooled by ``(airport, runway index)``: position, height, the altitude word
    in force and the one said, the row and the flight."""
    signals = load_signals(instructions, split)
    sentences = load_sentences(instructions, split, words.spec)
    groups: dict[tuple[str, int], list[dict[str, np.ndarray]]] = defaultdict(list)
    for k, index in enumerate(sentences["signal_index"]):
        flight = signals[int(index)]
        grid = sentences["words"][sentences["offsets"][k]: sentences["offsets"][k + 1]].astype(np.int64)
        count = len(grid)
        groups[(flight.airport, int(sentences["runway_index"][k]))].append({
            "e": flight.e_m[:count], "n": flight.n_m[:count], "h": flight.altitude_m[:count],
            "in_force": in_force(grid)[:, ALTITUDE], "said": grid[:, ALTITUDE], "row": np.arange(count),
            "flight": np.full(count, k)})
    return {key: {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
            for key, parts in groups.items()}


def check_labelled(rows: dict[str, np.ndarray], procedure: RunwayProcedure, words: Words) -> dict[str, Any]:
    """One runway's flights: the words forbidden (rules 1–2), the steps forced (rule 3), the observed tracks below their
    floor (and how far below, at each one's deepest row)."""
    allowed = altitude_word_allowed(procedure, rows["in_force"], rows["e"], rows["n"], rows["h"], words)
    first, later = rows["row"] == N_LOOK, rows["row"] > N_LOOK
    said = first | (later & (rows["said"] != UNCHANGED))
    land = rows["in_force"] == words.altitude_land
    forced = later & (rows["said"] == UNCHANGED) & ~allowed
    below, floor = below_floor(procedure, rows["e"], rows["n"], rows["h"], words.spec)
    flights, index = np.unique(rows["flight"][below], return_inverse=True)
    deepest = np.full(len(flights), -np.inf)
    np.maximum.at(deepest, index, (floor - rows["h"])[below])
    return {
        "flights": int(len(np.unique(rows["flight"]))),
        "words": {"level": int((said & ~land).sum()), "land": int((said & land).sum()),
                  "level_forbidden": int((said & ~land & ~allowed).sum()),
                  "land_forbidden": int((said & land & ~allowed).sum())},
        "forced_steps": int(forced.sum()), "steps": int(later.sum()),
        "flights_forced": int(len(np.unique(rows["flight"][forced]))),
        "observed_below": len(flights), "observed_depth_m": sorted(deepest.tolist()),
    }


def check_replays(batch: replay.Batch, procedures: dict[str, tuple[RunwayProcedure, ...]], words: Words, params: Any,
                  chunk: int) -> list[dict[str, Any]]:
    """Each flight of ``batch`` flown on its labelled words from the first predicted step: its outcome and stratum, and
    whether a flown step sank below its floor (`glidepath_stops`; at the stop: its distance to go, depth, altitude word).
    The outcome is the executor's judge's — the one the flight would have had without the stop."""
    cpu, spec = torch.device("cpu"), words.spec
    out = []
    for start in range(0, len(batch.readings), chunk):
        part = replay.subset(batch, list(range(start, min(start + chunk, len(batch.readings)))))
        flown = fly_reference(part, words, params, device=cpu)
        grids = [reference_grid(r.words) for r in part.readings]
        rows = flight_rows(part, flown, grids, words, "labelled", [None] * len(grids), None)
        finals = [procedures[g.code] for g in part.geometries]
        stops = glidepath_stops(flown, grids, part.geometries, finals, words)
        step_rows = round(spec.step_s / flown.cycle_s)
        for j, (reading, grid, row) in enumerate(zip(part.readings, grids, rows)):
            record = {"dataset_id": row["dataset_id"], "airport": row["airport"],
                      "runway": f"{row['airport']} {finals[j][reading.runway_index].ident}", "stratum": row["stratum"],
                      "outcome": row["outcome"], "stopped": bool(stops.step[j] >= 0)}
            if stops.step[j] >= 0:
                state = (int(stops.step[j]) + 1) * step_rows            # the state after the stopping step
                track = flown_track(flown.states[j, [state]].cpu().numpy(), part.geometries[j])
                force = in_force(grid)[stops.row[j]]                  # the words that step flew
                procedure = finals[j][force[RUNWAY]]
                d, _ = procedure.axes(track["e"], track["n"])
                record |= {"step": int(stops.step[j]), "row": int(stops.row[j]), "d_m": float(d[0]),
                           "depth_m": float(procedure.floor_m(track["e"], track["n"])[0] - track["height"][0]),
                           "word": "land" if force[ALTITUDE] == words.altitude_land else "level"}
            out.append(record)
        print(f"  replayed {min(start + chunk, len(batch.readings))}/{len(batch.readings)}", flush=True)
    return out


def _percentiles(values: list[float]) -> dict[str, float] | None:
    return ({f"p{q}": float(np.percentile(values, q)) for q in (50, 90)} | {"max": float(max(values))}) if values else None


def summarise(labelled: dict[str, dict[str, Any]], replays: list[dict[str, Any]]) -> dict[str, Any]:
    """The totals: the words forbidden, the steps forced, the observed tracks below, the replays stopped."""
    words = Counter()
    for part in labelled.values():
        words.update(part["words"])
    checked, forbidden = words["level"] + words["land"], words["level_forbidden"] + words["land_forbidden"]
    flights = sum(part["flights"] for part in labelled.values())
    below = sum(part["observed_below"] for part in labelled.values())
    stopped = [r for r in replays if r["stopped"]]
    return {
        "flights": flights, "words": dict(words), "forbidden_share": forbidden / checked,
        "forced_steps": sum(p["forced_steps"] for p in labelled.values()),
        "steps": sum(p["steps"] for p in labelled.values()),
        "flights_forced": sum(p["flights_forced"] for p in labelled.values()),
        "observed_below": below, "observed_below_share": below / flights,
        "observed_below_by_runway": {key: part["observed_below"] for key, part in labelled.items()
                                     if part["observed_below"]},
        "observed_depth_m": _percentiles([d for part in labelled.values() for d in part["observed_depth_m"]]),
        "replays": len(replays), "stopped": len(stopped), "stopped_share": len(stopped) / len(replays),
        "stopped_landed": sum(r["outcome"] == "landed" for r in stopped),
        "stopped_by": {key: dict(Counter(r[key] for r in stopped)) for key in ("airport", "runway", "stratum", "word")},
        "stopped_depth_m": _percentiles([r["depth_m"] for r in stopped]),
        "stopped_d_m": _percentiles([r["d_m"] for r in stopped]),
        "replays_by": {key: dict(Counter(r[key] for r in replays)) for key in ("airport", "stratum", "outcome")},
    }


def passes(summary: dict[str, Any]) -> bool:
    """The pass lines of design §3.6: both at most their line."""
    return summary["forbidden_share"] <= MAX_FORBIDDEN_WORDS and summary["stopped_share"] <= MAX_STOPPED_REPLAYS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--replay-per-airport", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    instructions, executor_dir, out = map(resolved, (args.instructions, args.executor, args.out))
    if out.exists():
        parser.error(f"{out} exists; a check is never overwritten")
    git = git_state()
    if git["dirty"]:
        parser.error("the check runs from a clean tree")
    started = time.perf_counter()
    params, record, words = replay.open_executor(executor_dir, instructions)
    procedures = published_procedures(load_candidates(instructions))
    labelled = {f"{airport} {procedures[airport][runway].ident}": check_labelled(rows, procedures[airport][runway], words)
                for (airport, runway), rows in sorted(labelled_rows(args.split, instructions, words).items())}
    print(f"labelled {args.split} flights read, {time.perf_counter() - started:.0f}s", flush=True)
    batch = replay.draw(instructions, args.split, words.spec, words, per_airport=args.replay_per_airport,
                        seed=args.seed)
    replays = check_replays(batch, procedures, words, params, args.chunk)
    summary = summarise(labelled, replays)
    passed = passes(summary)
    out.mkdir(parents=True)
    write_json_atomic(out / "check.json", {
        "schema": CHECK_SCHEMA, "written_utc": utc_now(), "git": git, "split": args.split,
        "instructions": str(instructions), "executor": {"directory": str(executor_dir), "sha256": record["sha256"]},
        "tolerances_m": {"word": word_tolerance_m(words.spec), "track": track_tolerance_m(words.spec),
                         "glidepath_below": GLIDEPATH_BELOW_M},
        "faf_d_m": {code: {p.ident: p.faf_d_m for p in runways} for code, runways in procedures.items()},
        "pass_lines": {"forbidden_words": MAX_FORBIDDEN_WORDS, "stopped_replays": MAX_STOPPED_REPLAYS},
        "replay_sample": batch.drawn, "summary": summary, "passed": passed, "by_runway": labelled,
        "replays": replays, "elapsed_s": time.perf_counter() - started})
    w = summary["words"]
    print(f"words forbidden {summary['forbidden_share']:.2%} (level {w['level_forbidden']}/{w['level']}, "
          f"descend to land {w['land_forbidden']}/{w['land']}); forced steps {summary['forced_steps']}/"
          f"{summary['steps']} in {summary['flights_forced']} flights; observed tracks below "
          f"{summary['observed_below']}/{summary['flights']} ({summary['observed_below_share']:.2%}); replays stopped "
          f"{summary['stopped']}/{summary['replays']} ({summary['stopped_share']:.2%}, landed {summary['stopped_landed']})")
    print(f"passed: {passed}  (≤ {MAX_FORBIDDEN_WORDS:.0%} words forbidden, ≤ {MAX_STOPPED_REPLAYS:.0%} replays "
          f"stopped)  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
