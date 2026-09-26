"""The procedure's altitude masks checked on labelled data before any model trains through them (post-training design
§3.6).

The second stage's step 0. On every labelled flight of the split, its sentence's runway throughout (the labels never
change it after the first step):

- **words** — the labelled altitude and descent-angle words the masks would forbid where the prior would say them: the
  word in force at the first predicted step (row `N_LOOK`, where every column is said) and every word said after it,
  rule by rule (`procedure`: 1–2 the glidepath lower edge, a level and "descend to land" apart; 3 the decision
  altitude; 4 no climbing back — a level above the aircraft and the climb class apart);
- **forced** — the later steps where no altitude word is said but the word in force fails here, so the mask would make
  the prior say a new one (rule 5);
- **observed** — the tracks with a row more than the track tolerance below the glidepath lower edge, and the tracks'
  readouts before the join (`procedure.pre_join_readout`: under the DA, climbing back after the dip, under the MVA);

and on a seeded per-airport sample flown on its own dynamics (``--replay-per-airport``), the labelled words flown by the
executor from the first predicted step as free generation flies its reference (`prior_free_generation.fly_reference`):
**stopped** — a flown step (every ``step_s``) more than the track tolerance below its floor (§3.5), found by the check
free generation and training use (`prior_free_generation.glidepath_stops`).

The pass lines, written before the run (user 2026-09-25, design §3.6): at most `MAX_FORBIDDEN_WORDS` of the labelled
altitude and descent-angle words forbidden (every rule together) and at most `MAX_STOPPED_REPLAYS` of the replayed
flights stopped; failing either, the numbers go back to the user.

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
    PRE_JOIN_LINES, ProcedureMasks, _mva, flight_rows, fly_reference, glidepath_stops, in_force, procedure_masks,
    reference_grid,
)
from ts_transformer.instructions.artefact import SPLITS, load_candidates, load_sentences, load_signals
from ts_transformer.instructions.words import ALTITUDE, ANGLE, APPROACH, RUNWAY, UNCHANGED, Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.prior import mva
from ts_transformer.prior.procedure import (
    GLIDEPATH_BELOW_M, RunwayProcedure, angle_word_allowed, below_floor, climb_allowed, climb_barred,
    decision_allowed, glidepath_allowed, pre_join, pre_join_readout, track_tolerance_m, word_tolerance_m,
)
from ts_transformer.prior.scene import N_LOOK
from ts_transformer.repo_layout import REPO_ROOT, git_state

CHECK_SCHEMA = "ts-prior-procedure-check-v2"
#: The pass lines of design §3.6 (user 2026-09-25).
MAX_FORBIDDEN_WORDS = 0.01
MAX_STOPPED_REPLAYS = 0.03


def labelled_rows(split: str, instructions: Path, words: Words, masks: ProcedureMasks
                  ) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """Every labelled flight's sentence rows, pooled by ``(airport, runway index)``: position, height, the altitude and
    descent-angle words in force and the ones said, the approach word in force, where the flight had joined its
    runway's final and dipped under its entry height (`procedure.pre_join`), the row and the flight — and, one per
    flight, its observed readouts before the join (`procedure.pre_join_readout`, under ``"pre_join"``)."""
    signals = load_signals(instructions, split)
    sentences = load_sentences(instructions, split, words.spec)
    geometries = load_candidates(instructions)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for k, index in enumerate(sentences["signal_index"]):
        flight = signals[int(index)]
        grid = sentences["words"][sentences["offsets"][k]: sentences["offsets"][k + 1]].astype(np.int64)
        count, runway = len(grid), int(sentences["runway_index"][k])
        finals = masks.finals[flight.airport]
        e, n, h = flight.e_m[:count], flight.n_m[:count], flight.altitude_m[:count]
        force = in_force(grid)
        joined, dipped = pre_join(finals[runway], e, n, h, words.spec)
        readout = pre_join_readout(finals, force[:, RUNWAY], force[:, APPROACH], e, n, h,
                                   _mva(masks.charts[flight.airport], geometries[flight.airport], e, n), N_LOOK,
                                   words.spec)
        groups[(flight.airport, runway)].append({
            "e": e, "n": n, "h": h, "in_force": force[:, ALTITUDE], "said": grid[:, ALTITUDE],
            "angle_in_force": force[:, ANGLE], "angle_said": grid[:, ANGLE], "approach": force[:, APPROACH],
            "joined": joined, "dipped": dipped, "row": np.arange(count), "flight": np.full(count, k),
            "pre_join": np.array([readout], dtype=object)})
    return {key: {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
            for key, parts in groups.items()}


def check_labelled(rows: dict[str, np.ndarray], procedure: RunwayProcedure, words: Words) -> dict[str, Any]:
    """One runway's flights: the words forbidden, rule by rule (1–2 the glidepath lower edge, 3 the decision altitude,
    4 no climbing back — the altitude column's and the descent-angle column's), the steps forced (rule 5), the observed
    tracks below the glidepath lower edge (and how far below, at each one's deepest row), and the observed readouts
    before the join."""
    barred = climb_barred(rows["joined"], rows["dipped"], rows["approach"])
    word, h = rows["in_force"], rows["h"]
    edge = glidepath_allowed(procedure, word, rows["e"], rows["n"], h, words)
    decision = decision_allowed(procedure, word, rows["joined"], words)
    climb = climb_allowed(word, h, barred, words)
    allowed = edge & decision & climb
    angle = angle_word_allowed(rows["angle_in_force"], barred, words)
    first, later = rows["row"] == N_LOOK, rows["row"] > N_LOOK
    said = first | (later & (rows["said"] != UNCHANGED))
    angle_said = first | (later & (rows["angle_said"] != UNCHANGED))
    land = word == words.altitude_land
    forced = later & (rows["said"] == UNCHANGED) & ~allowed
    below, floor = below_floor(procedure, rows["e"], rows["n"], h, words.spec)
    flights, index = np.unique(rows["flight"][below], return_inverse=True)
    deepest = np.full(len(flights), -np.inf)
    np.maximum.at(deepest, index, (floor - h)[below])
    readouts = rows["pre_join"]
    return {
        "flights": int(len(np.unique(rows["flight"]))),
        "words": {"level": int((said & ~land).sum()), "land": int((said & land).sum()),
                  "angle": int(angle_said.sum()),
                  "level_forbidden_edge": int((said & ~land & ~edge).sum()),
                  "land_forbidden_edge": int((said & land & ~edge).sum()),
                  "level_forbidden_decision": int((said & ~decision).sum()),
                  "level_forbidden_climb": int((said & ~climb).sum()),
                  "angle_forbidden_climb": int((angle_said & ~angle).sum()),
                  "forbidden": int((said & ~allowed).sum() + (angle_said & ~angle).sum())},
        "forced_steps": int(forced.sum()), "steps": int(later.sum()),
        "flights_forced": int(len(np.unique(rows["flight"][forced]))),
        "observed_below": len(flights), "observed_depth_m": sorted(deepest.tolist()),
        "observed_pre_join": {key: int(sum(r[key] for r in readouts)) for key in PRE_JOIN_LINES},
    }


def check_replays(batch: replay.Batch, procedures: ProcedureMasks, words: Words, params: Any, chunk: int
                  ) -> list[dict[str, Any]]:
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
        finals = [procedures.finals[g.code] for g in part.geometries]
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
    words, pre_join = Counter(), Counter()
    for part in labelled.values():
        words.update(part["words"])
        pre_join.update(part["observed_pre_join"])
    checked = words["level"] + words["land"] + words["angle"]
    flights = sum(part["flights"] for part in labelled.values())
    below = sum(part["observed_below"] for part in labelled.values())
    stopped = [r for r in replays if r["stopped"]]
    return {
        "flights": flights, "words": dict(words), "forbidden_share": words["forbidden"] / checked,
        "observed_pre_join": dict(pre_join),
        "observed_pre_join_share": {key: pre_join[key] / flights for key in PRE_JOIN_LINES},
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
    procedures = procedure_masks(load_candidates(instructions))
    finals = procedures.finals
    labelled = {f"{airport} {finals[airport][runway].ident}": check_labelled(rows, finals[airport][runway], words)
                for (airport, runway), rows in sorted(labelled_rows(args.split, instructions, words,
                                                                    procedures).items())}
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
        "faf_d_m": {code: {p.ident: p.faf_d_m for p in runways} for code, runways in finals.items()},
        "entry_m": {code: {p.ident: p.entry_m for p in runways} for code, runways in finals.items()},
        "decision_m": {code: {p.ident: p.decision_m for p in runways} for code, runways in finals.items()},
        "mva": {"charts_date": mva.CHARTS_DATE, "chart": mva.CHART},
        "pass_lines": {"forbidden_words": MAX_FORBIDDEN_WORDS, "stopped_replays": MAX_STOPPED_REPLAYS},
        "replay_sample": batch.drawn, "summary": summary, "passed": passed, "by_runway": labelled,
        "replays": replays, "elapsed_s": time.perf_counter() - started})
    w = summary["words"]
    print(f"words forbidden {summary['forbidden_share']:.2%} of {w['level'] + w['land'] + w['angle']} (edge: level "
          f"{w['level_forbidden_edge']}/{w['level']}, descend to land {w['land_forbidden_edge']}/{w['land']}; decision "
          f"altitude {w['level_forbidden_decision']}; no climbing back: level {w['level_forbidden_climb']}, climb class "
          f"{w['angle_forbidden_climb']}/{w['angle']}); forced steps {summary['forced_steps']}/{summary['steps']} in "
          f"{summary['flights_forced']} flights; observed tracks below the edge {summary['observed_below']}/"
          f"{summary['flights']} ({summary['observed_below_share']:.2%}), before the join "
          f"{summary['observed_pre_join_share']}; replays stopped {summary['stopped']}/{summary['replays']} "
          f"({summary['stopped_share']:.2%}, landed {summary['stopped_landed']})")
    print(f"passed: {passed}  (≤ {MAX_FORBIDDEN_WORDS:.0%} words forbidden, ≤ {MAX_STOPPED_REPLAYS:.0%} replays "
          f"stopped)  → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
