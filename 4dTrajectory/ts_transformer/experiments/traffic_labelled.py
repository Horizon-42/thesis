"""Multi-aircraft M0 step 5 (design §6.4; the "labelled words" setting, §2.2): every flight with a sentence flies its
labelled words on its own executor, all together on the training days' scenes, judged every step as the closed loop
judges (`traffic_loop`) — design §3.4 step 0, the second pass line: what the executor adds to the flights ended is at
most `PASS_SHARE`.

Who flies — the replay gate's rule (`autopilot.replay.group_of`): every flight with a sentence whose type the executor
can fly, on its own dynamics or a stand-in's (reported apart), from its row 0, its stored sentence said on the executor
spec's clock over its observed rows (`replay.fly_sentences`; each flight re-read by the labeller first and refused unless
the reading is its stored sentence), in chunks of ``--chunk`` flights, until the executor is done with it. Its own end is
the executor judge's outcome (`autopilot.judge.outcome_of`), or the first step whose end state sinks below the glidepath
lower edge (`prior_free_generation.glidepath_stops`; the closed loop keeps the procedure's altitudes on, design §9 item
11) when that comes first (`own_end`: a stop after the outcome's row never counts — the flight has ended by then, where
`prior_free_generation.flight_rows` lets any stop win). Everyone else follows the record (`traffic_census.track`): the
flights with a sentence the executor cannot fly — contract C31, counted by reason — and the background arrivals.

The loop then runs twice under each reading: with the flown flights (``executor``) and, as the control, with the same
flights along their own recorded rows on the same steps (``recorded``, `traffic_loop.recorded`) — the same scenes, the
same steps, the same check and the same replayed aircraft, so what differs is the executor. Each reading ends the
aircraft that answer for its losses: `VISUAL`, the check and the reward, and `IFR`, reported beside it (design §3.2).
Beside each: pairs and ended flights in both runs or in one only.

The pass line (user 2026-09-28, design §9 item 17) is what the executor adds: under `VISUAL`, over the flights on
their own dynamics (the replay gate's rule: a stand-in's errors are its aerodynamics, reported, never gated,
`autopilot.replay`), the executor run's ended share less the recorded control's — the recorded traffic already breaks
the check where its controllers kept visual separation, which the model cannot say (design §3.2) — per airport and
pooled. Beside it: both shares, and the flights ended in either run alone.

Only training days are read. Writes ``labelled.json`` into a NEW directory; ``--airports`` limits it to a smoke test and
says so.

    python run_ts.py traffic_labelled --instructions 4dTrajectory/outputs/POOLED/instruction_language/v5_20260926 \\
        --executor 4dTrajectory/outputs/POOLED/executor/v11_20260927 \\
        --out 4dTrajectory/outputs/POOLED/traffic/labelled_<date>
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from geokit import NM_M

from ts_transformer.autopilot import replay
from ts_transformer.autopilot.flights import rebuild_series
from ts_transformer.autopilot.executor import Flown
from ts_transformer.autopilot.judge import Outcome, flown_track, outcome_of
from ts_transformer.autopilot.runway_data import published_vertical_paths
from ts_transformer.experiments.prior_free_generation import BELOW_GLIDEPATH, glidepath_stops
from ts_transformer.experiments.traffic_census import APPROACH_SPEED_MPS, Track, quantiles, track
from ts_transformer.experiments.traffic_loop import LOST_SEPARATION, Controlled, Loop, Run, flown, recorded
from ts_transformer.inference.runway_schedule import Separation, faa_separation, read_cwt_tables, wake_category
from ts_transformer.inference.separation import READINGS, VISUAL
from ts_transformer.instructions.airport import AirportGeometry
from ts_transformer.instructions.artefact import arrival_manifest_sha256s, load_candidates, load_sentences, load_signals
from ts_transformer.instructions.labeller.read import read_flight
from ts_transformer.instructions.readout import flight_record
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.prior.masks import SETS, ProcedureMasks
from ts_transformer.prior.scene import Presence, presence
from ts_transformer.repo_layout import REPO_ROOT, arrival_manifest_path, git_state, repo_relative

#: v2 (2026-09-28): the pass line gates what the executor adds (`pass_line`), not its ended share.
SCHEMA = "ts-traffic-labelled-v2"
SPLIT = "train"
#: Design §3.4 step 0, the second pass line (user 2026-09-27, §9 item 5; read as what the executor adds, 2026-09-28,
#: §9 item 17): the flights flying the labelled words ended by the check, less the same flights along their records.
PASS_SHARE = 0.03
EXECUTOR, RECORDED = "executor", "recorded"
MODES = (EXECUTOR, RECORDED)
FLOWN_GROUPS = (replay.OWN, replay.STAND_IN)


def own_end(outcome: str, end_row: int, stop_step: int, step_rows: int) -> tuple[str, int]:
    """How a flown flight ends on its own, and the last state row it is in the scene at: the glidepath lower edge where
    the state row it sank at (``stop_step``'s end; −1: never) comes before the judge's outcome row, the outcome
    otherwise. A crossing, a ground contact or a dynamics failure is read at the first state row past it, so the last
    row in the scene is the one before; a timeout's is its last."""
    stop_row = (stop_step + 1) * step_rows
    if stop_step >= 0 and stop_row < end_row:
        return BELOW_GLIDEPATH, stop_row
    return outcome, end_row if outcome == "timeout" else end_row - 1


def flown_aircraft(states: Flown, j: int, ended: Outcome, stop_step: int, seen: Presence, runway: str,
                   geometry: AirportGeometry, separation: Separation, category: str | None, step_s: float
                   ) -> Controlled:
    """Flight ``j`` of ``states`` on the loop's steps (`traffic_loop.flown`): its state every step to its own end
    (`own_end`), the executor's capture after the cycle before each state (none at the first: the executor starts
    uncaptured), and its interpolated crossing when it landed."""
    step_rows = int(round(step_s / states.cycle_s))
    outcome, last_row = own_end(ended.outcome, ended.end_row, stop_step, step_rows)
    at_rows = np.arange(0, last_row + 1, step_rows)
    track_at = flown_track(states.states[j, at_rows].cpu().numpy(), geometry)
    captured = np.concatenate(([False], states.modes["captured"][j, at_rows[1:] - 1].cpu().numpy()))
    landing = ended.crossing["at_row"] * states.cycle_s if outcome == "landed" else None
    return flown(seen, step_s, track_at["e"], track_at["n"], track_at["height"], track_at["track"],
                 track_at["ground_speed"], captured, runway, geometry, separation, category, outcome, landing)


def fly_airport(directory: Path, airport: str, members: list[int], signals: list,
                spoken: dict[int, tuple[int, int, int]],
                sentences: dict[str, Any], geometry: Any, separation: Any, params: Any, words: Any, finals: Any, *,
                chunk: int, device: torch.device, started: float
                ) -> tuple[list[Controlled], list[Controlled], list[Track], list[dict[str, Any]], Counter]:
    """One airport's flights with a sentence: the flown ones on the steps (``executor``) and along their records
    (``recorded``), the ones the executor cannot fly as replayed tracks, a row per flown flight, and the reasons the
    others were not flown."""
    spec, step_s = words.spec, words.spec.step_s
    paths = published_vertical_paths(geometry)
    executor, control, replayed, rows = [], [], [], []
    not_flown: Counter = Counter()
    order = sorted(members, key=lambda i: (spoken[i][0], signals[i].dataset_id))   # like lengths fly together
    for start in range(0, len(order), chunk):
        part = order[start: start + chunk]
        series = rebuild_series(directory, [signals[i] for i in part])
        fly, batch_series, readings, groups = [], [], [], []
        for i, built in zip(part, series):
            flight, (rows_i, capture, k) = signals[i], spoken[i]
            group = replay.group_of(built)
            if group not in FLOWN_GROUPS:
                not_flown[group] += 1
                replayed.append(track(flight, rows_i, capture, geometry, separation.along_nm[flight.runway] * NM_M,
                                      spec, step_s))
                continue
            reading = read_flight(flight, geometry, spec, words)
            grid = sentences["words"][sentences["offsets"][k]: sentences["offsets"][k + 1]]
            if not np.array_equal(reading.words, grid) or reading.runway_index != int(sentences["runway_index"][k]):
                raise ValueError(f"{flight.dataset_id}: the re-read sentence differs from the stored one")
            fly.append(i)
            batch_series.append(built)
            readings.append(reading)
            groups.append(group)
        if not fly:
            continue
        batch = replay.Batch(signals=[signals[i] for i in fly], series=batch_series, readings=readings,
                             geometries=[geometry] * len(fly), vertical_paths=[paths] * len(fly),
                             approach_ias_mps=[replay.flight_approach_ias_mps(s, g)
                                               for s, g in zip(batch_series, groups)],
                             groups=groups, drawn={"split": SPLIT})
        states = replay.fly_sentences(batch, params, words, device=device)
        stops = glidepath_stops(states, [r.words for r in readings], batch.geometries, [finals] * len(fly), words)
        for j, i in enumerate(fly):
            flight, reading = signals[i], readings[j]
            seen = presence(flight, spoken[i][0], geometry)
            category = None if flight.typecode is None else wake_category(flight.typecode)
            aircraft = flown_aircraft(states, j, outcome_of(states, j, geometry, reading.runway_index, spec),
                                      int(stops.step[j]), seen, geometry.candidates[reading.runway_index].ident,
                                      geometry, separation, category, step_s)
            along_rows = recorded(seen, flight, spoken[i][1], replay.observed_landing_s(flight, reading, geometry),
                                  geometry, separation, category, step_s)
            executor.append(aircraft)
            control.append(along_rows)
            first = aircraft.first_step_s
            rows.append({"key": flight.dataset_id, "airport": airport, "group": groups[j],
                         "stratum": flight_record(reading)["stratum"], "outcome": aircraft.outcome,
                         "flown_s": aircraft.last_step_s - first,
                         "recorded_s": float(seen.times_s[-1] - seen.times_s[0]),
                         "landing_after_first_step_s": {
                             EXECUTOR: None if aircraft.landing_s is None else aircraft.landing_s - first,
                             RECORDED: along_rows.landing_s - first}})
        del states
        print(f"  {airport}: {min(start + chunk, len(order))}/{len(order)} with a sentence, "
              f"{time.perf_counter() - started:.0f}s", flush=True)
    return executor, control, replayed, rows, not_flown


def merged(runs: Sequence[Run]) -> Run:
    """Several airports' runs of one reading as one (their keys never meet)."""
    out = Run(runs[0].reading)
    for run in runs:
        out.ended.update(run.ended)
        out.episodes += run.episodes
        out.at_threshold += run.at_threshold
        out.landings_checked += run.landings_checked
        out.scene_seconds += run.scene_seconds
        out.steps_judged += run.steps_judged
    return out


def summary(run: Run, flights: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """A run read (``flights``: each controlled flight's row — its group and its own end — by key): its outcomes in the
    loop (`LOST_SEPARATION` where the run ended it, its own end otherwise), who was ended and why, and the episodes."""
    hours = run.scene_seconds / 3600.0
    ended = run.ended
    pairs = {tuple(e["pair"]) for e in run.episodes}
    by_group = Counter(row["group"] for row in flights.values())
    ended_by_group = Counter(flights[key]["group"] for key in ended)
    outcomes = Counter(LOST_SEPARATION if key in ended else row["outcome"] for key, row in flights.items())
    return {
        "flights": len(flights), "ended": len(ended), "ended_share": len(ended) / len(flights),
        "outcomes": dict(outcomes.most_common()), "landed_share": outcomes["landed"] / len(flights),
        "ended_by_group": {g: {"flights": n, "ended": ended_by_group[g], "share": ended_by_group[g] / n}
                           for g, n in sorted(by_group.items())},
        "ended_by_kind": dict(Counter(e["kind"] for e in ended.values()).most_common()),
        "ended_by_relation": dict(Counter(e["relation"] for e in ended.values()).most_common()),
        "ended_beside_a_replayed_aircraft": sum(not e["with_controlled"] for e in ended.values()),
        "scene_hours": hours, "steps_judged": run.steps_judged,
        "episodes": len(run.episodes), "episodes_per_scene_hour": len(run.episodes) / hours,
        "pairs_with_a_loss": len(pairs), "pairs_with_a_loss_per_scene_hour": len(pairs) / hours,
        "episodes_by_relation": dict(Counter(e["relation"] for e in run.episodes).most_common()),
        "episodes_by_kind": dict(Counter(kind for e in run.episodes for kind in e["kinds"]).most_common()),
        "episodes_only_a_replayed_aircraft_answers_for": sum(
            not any(r["controlled"] for r in e["responsible"]) for e in run.episodes),
        "episodes_between_two_replayed_aircraft": sum(not (set(e["pair"]) & set(flights)) for e in run.episodes),
        "episode_steps": quantiles([e["steps"] for e in run.episodes]),
        "min_distance_over_required": quantiles([e["min_ratio"] for e in run.episodes]),
        "at_threshold": {"landings_checked": run.landings_checked, "losses": len(run.at_threshold),
                         "followers_ended": sum(a["follower_controlled"] for a in run.at_threshold)},
    }


def compared(executor: Run, control: Run) -> dict[str, Any]:
    """The executor's run against the recorded control under one reading: the pairs with a loss and the flights ended,
    in both, in the executor's only, in the control's only."""
    def split(a: set, b: set) -> dict[str, int]:
        return {"both": len(a & b), "executor_only": len(a - b), "recorded_only": len(b - a)}

    return {"pairs_with_a_loss": split({tuple(e["pair"]) for e in executor.episodes},
                                       {tuple(e["pair"]) for e in control.episodes}),
            "ended": split(set(executor.ended), set(control.ended))}


def pass_line(runs: dict[tuple[str, str], Run], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The second pass line over the flights on their own dynamics (module docstring): the executor run's ended share
    under `VISUAL` less the recorded control's, both shares, and how many were ended in either run alone."""
    gated = {row["key"] for row in rows if row["group"] == replay.OWN}
    executor = gated & set(runs[(EXECUTOR, VISUAL)].ended)
    control = gated & set(runs[(RECORDED, VISUAL)].ended)
    share = len(executor) / len(gated)
    recorded_share = len(control) / len(gated)
    return {"flights": len(gated), "added_share": share - recorded_share, "passes": share - recorded_share <= PASS_SHARE,
            "ended_share": share, "recorded_ended_share": recorded_share,
            "ended_in_the_executor_run_alone": len(executor - control),
            "ended_in_the_recorded_run_alone": len(control - executor)}


def readout(runs: dict[tuple[str, str], Run], rows: list[dict[str, Any]], not_flown: Counter,
            background: int) -> dict[str, Any]:
    """One airport's (or the pool's) numbers; ``own_outcomes`` are the flown flights' own ends, before any check."""
    flights = {row["key"]: row for row in rows}
    return {
        "flights": {"with_a_sentence": len(rows) + sum(not_flown.values()), "flown": len(rows),
                    "flown_by_group": dict(Counter(row["group"] for row in rows)),
                    "not_flown_replayed": dict(not_flown.most_common()), "background_replayed": background},
        "own_outcomes": dict(Counter(row["outcome"] for row in rows).most_common()),
        "landing_minus_recorded_s": quantiles([landing[EXECUTOR] - landing[RECORDED] for landing in (
            row["landing_after_first_step_s"] for row in rows) if landing[EXECUTOR] is not None]),
        "pass_line": pass_line(runs, rows),
        "runs": {mode: {reading: summary(runs[(mode, reading)], flights) for reading in READINGS} for mode in MODES},
        "executor_against_recorded": {reading: compared(runs[(EXECUTOR, reading)], runs[(RECORDED, reading)])
                                      for reading in READINGS},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True,
                        help="an instruction artefact split by operating day")
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    parser.add_argument("--airports", nargs="+", help="only these airports (a smoke test; the readout says so)")
    parser.add_argument("--chunk", type=int, default=300, help="flights flown in one batch")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    directory = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    executor_dir = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    started = time.perf_counter()
    params, record, words = replay.open_executor(executor_dir, directory)
    spec = words.spec
    candidates = load_candidates(directory)
    signals = load_signals(directory, SPLIT)
    sentences = load_sentences(directory, SPLIT, spec)
    offsets = sentences["offsets"]
    spoken = {int(signal): (int(offsets[k + 1] - offsets[k]), int(sentences["capture_row"][k]), k)
              for k, signal in enumerate(sentences["signal_index"])}
    wanted = [i for i, f in enumerate(signals) if not args.airports or f.airport in args.airports]
    missing = sorted({signals[i].typecode for i in wanted if signals[i].typecode is not None} - set(read_cwt_tables()))
    if missing:
        raise SystemExit(f"{len(missing)} types are in neither CWT table: {missing} (add each to the supplement "
                         "with its source first; `traffic_census` lists them)")
    masks = ProcedureMasks.build(SETS, candidates)
    recorded_sha = arrival_manifest_sha256s(directory)
    by_airport: dict[str, list[int]] = defaultdict(list)
    for i in wanted:
        by_airport[signals[i].airport].append(i)

    report: dict[str, Any] = {}
    flights: list[dict[str, Any]] = []
    pooled: dict[tuple[str, str], list[Run]] = defaultdict(list)
    pooled_not_flown: Counter = Counter()
    background_total = 0
    for airport in sorted(by_airport):
        manifest = arrival_manifest_path(airport)
        if file_sha256(manifest) != recorded_sha[airport]:
            raise ValueError(f"{manifest} is not the manifest the artefact's signals were read from "
                             f"(sha256 {recorded_sha[airport][:12]} recorded)")
        separation = faa_separation(json.loads(manifest.read_text(encoding="utf-8"))["runway_targets"],
                                    speed_mps=APPROACH_SPEED_MPS)
        geometry = candidates[airport]
        members = [i for i in by_airport[airport] if i in spoken]
        executor, control, replayed, rows, not_flown = fly_airport(
            directory, airport, members, signals, spoken, sentences, geometry, separation, params, words,
            masks.finals[airport], chunk=args.chunk, device=torch.device(args.device), started=started)
        background = [i for i in by_airport[airport] if i not in spoken]
        replayed += [track(signals[i], None, None, geometry, separation.along_nm[signals[i].runway] * NM_M, spec,
                           spec.step_s) for i in background]
        runs: dict[tuple[str, str], Run] = {}
        for reading in READINGS:
            loop = Loop(separation, reading, spec.step_s)
            runs[(EXECUTOR, reading)] = loop.run(executor, replayed)
            runs[(RECORDED, reading)] = loop.run(control, replayed)
        for row in rows:
            row["ended"] = {mode: {reading: runs[(mode, reading)].ended.get(row["key"]) for reading in READINGS}
                            for mode in MODES}
        report[airport] = readout(runs, rows, not_flown, len(background))
        report[airport]["episodes"] = {f"{mode}/{reading}": runs[(mode, reading)].episodes
                                       for mode in MODES for reading in READINGS}
        for key, run in runs.items():
            pooled[key].append(run)
        pooled_not_flown.update(not_flown)
        background_total += len(background)
        flights += rows
        line = report[airport]["pass_line"]
        print(f"{airport}: {len(rows)} flown; own dynamics ended under the check {line['ended_share']:.2%}, the "
              f"recorded control {line['recorded_ended_share']:.2%}, {line['ended_in_the_executor_run_alone']} in the "
              f"executor's run alone, {time.perf_counter() - started:.0f}s", flush=True)

    pool = readout({key: merged(runs) for key, runs in pooled.items()}, flights, pooled_not_flown, background_total)
    line = pool["pass_line"]
    payload = {
        "schema": SCHEMA, "written_utc": utc_now(), "split": SPLIT, "only_airports": args.airports,
        "instructions": repo_relative(directory), "spec_sha256": spec.sha256,
        "executor": repo_relative(executor_dir), "executor_spec_sha256": record["sha256"], "params": asdict(params),
        "procedure_masks": {"names": list(masks.names), "data_sha256": masks.data_sha256()},
        "arrival_manifest_sha256": {airport: recorded_sha[airport] for airport in sorted(by_airport)},
        "check_reading": VISUAL, "pass_share": PASS_SHARE,
        "pass_line": {**line, "by_airport": {airport: report[airport]["pass_line"] for airport in sorted(report)}},
        "ended_outcome": LOST_SEPARATION, "pooled": pool, "airports": report, "flights": flights,
        "git": git_state(), "seconds": time.perf_counter() - started,
    }
    out.mkdir(parents=True)
    write_json_atomic(out / "labelled.json", payload)
    print(f"pooled: own dynamics ended under the check {line['ended_share']:.2%}, the recorded control "
          f"{line['recorded_ended_share']:.2%}: the executor adds {line['added_share']:.2%} (pass line {PASS_SHARE:.0%}: "
          f"{'passes' if line['passes'] else 'fails'}); wrote {out / 'labelled.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
