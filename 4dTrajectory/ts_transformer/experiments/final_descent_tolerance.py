"""Vocabulary A24 (D66): the closed-loop reading's vertical tolerance in the final descent, H_final, measured — for the
user's choice (D55). NO CRITERION IS READ (D7).

One readout reads one instruction artefact, built as the formal artefact in everything but H_final (the spec's
`closed_loop_final_vertical_m`), with its executor spec: the flights of ``--split`` (train: ``--per-airport`` of each
airport, the replay's seeded sample of D34; select: every labelled flight; on their own dynamics or a stand-in's,
`replay.draw_readings`), each read in closed loop at every ``--row-interval-s`` (`closed_loop.read_chunked`, as
`instruction_closed_loop.read_interval`, in memory: nothing is written into the artefact) and its closed-loop sentence
flown again as the replay flies it (`closed_loop.replay_batch`, `replay.fly_sentences`; every flight must fly its stored
states again, `executor_replay.closed_loop_columns`) and judged (`judge.outcome_of`: the outcome and the decision-altitude
check, D38). Per flight (``readout.json`` ``flights``): the outcome; the replay's DA check (its height above the
published glidepath at the DA point); the DA check of the OBSERVED track (`observed_decision`); the final descents
(runs of rows with "no level-off" in force) and the angle correction words said in them, beside all its angle
correction words.

Per row interval: the closed-loop reading's split numbers (`instruction_closed_loop.summarise`: D34 readings 1–3, the
tolerance in force at each row) and, per airport and pooled (`table`): the outcomes and the landed share; the
``unstable_at_minimums`` flights by why (`UNSTABLE_KINDS`: no DA point, high, low, lateral only); the height above the
glidepath at the DA point, p10 / p50 / p90; the DA check of the observed track against the replay's (`PAIRS`), with the
flights whose observed track passes and whose replay does not; the angle correction words per final descent and per
flight.

Each row interval is written as it ends (``interval_<Δ>s.json``), ``readout.json`` with all of them at the end.

``--table DIR ...`` puts readouts side by side (`side_by_side`): one per H_final and split, refused unless their specs
differ only in H_final, their executor parameters are the same and their drawn flights are the same; writes
``table.json`` and ``table.md`` into ``--out``.

    python run_ts.py final_descent_tolerance --instructions <artefact> --executor <spec> --split train \\
        --per-airport 400 --row-interval-s 2 4 8 --out <new dir>
    python run_ts.py final_descent_tolerance --table <readout dir> ... --out <new dir>
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_transformer.autopilot import closed_loop, replay
from ts_transformer.autopilot.judge import decision_check, outcome_of
from ts_transformer.autopilot.spec import params_sha256
from ts_transformer.experiments import instruction_closed_loop
from ts_transformer.experiments.executor_replay import closed_loop_columns
from ts_transformer.instructions.airport import AirportGeometry, relative_to_runway
from ts_transformer.instructions.artefact import ClosedLoopSentence
from ts_transformer.instructions.labeller.interval import in_force
from ts_transformer.instructions.labeller.read import Reading
from ts_transformer.instructions.readout import stratum
from ts_transformer.instructions.signals import FlightSignals
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import ALTITUDE, ANGLE, RUNWAY, Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state

READOUT_SCHEMA = "ts-final-descent-tolerance-readout-v1"
TABLE_SCHEMA = "ts-final-descent-tolerance-table-v1"
#: Why an ``unstable_at_minimums`` replay failed its DA check (`unstable_kind`).
UNSTABLE_KINDS = ("no DA point", "high", "low", "lateral only")
#: The observed track's DA check against the replay's: (observed passes, replay passes).
PAIRS = {"both pass": (True, True), "observed passes, replay does not": (True, False),
         "replay passes, observed does not": (False, True), "neither": (False, False)}
PERCENTILES = (10, 50, 90)
#: The outcomes whose crossing carries a decision-altitude check (`judge`: an approach crossing inside the runway limit).
DECIDED = ("landed", "unstable_at_minimums")


def observed_decision(signals: FlightSignals, reading: Reading, geometry: AirportGeometry,
                      spec: VocabularySpec) -> dict[str, Any] | None:
    """The judge's decision-altitude check (`judge.decision_check`, D38) on the observed track of the flight's landing
    approach: the rows from its last runway word (after a go-around, the approach after it) to the end of its reading,
    on its labelled runway, as observed — raw positions, track and MSL height (the reading of check A15–A22 §7.3: over
    every labelled select flight of `v9_20261004` it gives that check's numbers again, 97.97 %, −6.4 / +1.0 / +8.7 m; the
    readout reads it over the replayed flights only). None without a DA point."""
    span = len(reading.words)
    start = int(np.flatnonzero(reading.words[:, RUNWAY] >= 0)[-1])
    index = reading.runway_index
    relative = relative_to_runway(signals.e_m[:span], signals.n_m[:span], signals.track_deg[:span],
                                  signals.altitude_m[:span], geometry.candidates[index])
    return decision_check(relative, range(start + 1, span), geometry, index, spec)


def final_descents(sentence: ClosedLoopSentence, words: Words) -> dict[str, int]:
    """A closed-loop sentence's final descents (runs of said rows with "no level-off" in force, D66), their rows, and
    the angle correction words said in them and in the whole sentence."""
    final = in_force(sentence.grid)[:, ALTITUDE] == words.altitude_no_level_off
    starts = final & ~np.concatenate(([False], final[:-1]))
    angle = sentence.correction[:, ANGLE]
    return {"final_descents": int(starts.sum()), "final_descent_rows": int(final.sum()),
            "final_descent_angle_corrections": int((angle & final).sum()), "angle_corrections": int(angle.sum())}


def unstable_kind(decision: dict[str, Any] | None) -> str:
    """Why a DA check failed (`UNSTABLE_KINDS`): no DA point, above or below the ±22 m, or the FAS cone alone."""
    if decision is None:
        return "no DA point"
    if not decision["vertical_ok"]:
        return "high" if decision["above_glidepath_m"] > 0.0 else "low"
    return "lateral only"


def _passed(decision: dict[str, Any] | None) -> bool:
    return decision is not None and decision["passed"]


def _percentiles(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {f"p{q}": float(np.percentile(values, q)) for q in PERCENTILES} | {"n": len(values)}


def table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per airport and pooled (``all``): the readout's cells (module docstring); no cell without a flight."""
    out: dict[str, Any] = {}
    if not rows:
        return out
    for airport in (*sorted({r["airport"] for r in rows}), "all"):
        members = [r for r in rows if airport in (r["airport"], "all")]
        unstable = [r for r in members if r["outcome"] == "unstable_at_minimums"]
        pairs = Counter(next(name for name, key in PAIRS.items()
                             if key == (_passed(r["observed_decision"]), _passed(r["decision"]))) for r in members)
        descents = sum(r["final_descents"] for r in members)
        out[airport] = {
            "flights": len(members), "outcomes": dict(Counter(r["outcome"] for r in members).most_common()),
            "landed": sum(r["outcome"] == "landed" for r in members) / len(members),
            "unstable_at_minimums": {kind: sum(unstable_kind(r["decision"]) == kind for r in unstable)
                                     for kind in UNSTABLE_KINDS},
            "above_glidepath_at_da_m": _percentiles([r["decision"]["above_glidepath_m"] for r in members
                                                     if r["decision"] is not None]),
            "observed_above_glidepath_at_da_m": _percentiles([r["observed_decision"]["above_glidepath_m"]
                                                              for r in members if r["observed_decision"] is not None]),
            "da_check_observed_vs_replay": {name: pairs[name] for name in PAIRS},
            "final_descents": descents,
            "angle_corrections_per_final_descent": (sum(r["final_descent_angle_corrections"] for r in members)
                                                    / descents if descents else None),
            "final_descent_angle_corrections_per_flight": _percentiles(
                [float(r["final_descent_angle_corrections"]) for r in members]),
            "angle_corrections_per_flight": sum(r["angle_corrections"] for r in members) / len(members)}
    return out


def read_interval(drawn: replay.Drawn, readings: list[Reading], observed: dict[int, dict[str, Any] | None],
                  interval_s: float, params: Any, words: Words, *, chunk: int,
                  device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One row interval: the closed-loop reading of every drawn flight, its split numbers, and each sentence flown again
    and judged (module docstring); ``observed`` each flight's observed DA check by its artefact index."""
    spec = words.spec
    batch, results, numbers = instruction_closed_loop.read_interval(drawn, readings, interval_s, params, words,
                                                                    chunk=chunk, device=device)
    stored = {batch.indices[j]: r for j, r in enumerate(results) if isinstance(r, ClosedLoopSentence)}
    flown_batch, _ = closed_loop.replay_batch(batch, stored, words)
    check = closed_loop_columns(stored, spec.step_s)
    rows = []
    for first in range(0, len(flown_batch.sentences), chunk):
        part = replay.subset(flown_batch, list(range(first, min(first + chunk, len(flown_batch.sentences)))))
        flown = replay.fly_sentences(part, params, words, device=device)
        check(part, flown, [])                       # refused unless every flight flies its stored states again
        for j, index in enumerate(part.indices):
            ended = outcome_of(flown, j, part.geometries[j], spec)
            crossing = ended.crossing
            rows.append({"dataset_id": part.signals[j].dataset_id, "airport": part.signals[j].airport,
                         "stratum": stratum(part.readings[j]), "group": part.groups[j],
                         "go_arounds": part.sentences[j].go_arounds, "outcome": ended.outcome,
                         "decision": crossing["decision"] if ended.outcome in DECIDED else None,
                         "observed_decision": observed[index], **final_descents(stored[index], words)})
    return numbers, rows


def readout(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    instructions = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    git = git_state()
    if args.split != "train" and git["dirty"]:
        parser.error(f"the {args.split} readout runs from a clean tree")
    if args.split == "select" and args.per_airport:
        parser.error("the select readout reads every labelled select flight (D66): no --per-airport")
    params, record, words = replay.open_executor(executor, instructions)
    spec = words.spec
    started = time.perf_counter()
    drawn, readings = replay.draw_readings(instructions, args.split, spec, words, per_airport=args.per_airport,
                                           seed=args.seed, groups=(replay.OWN, replay.STAND_IN))
    observed = {drawn.indices[i]: observed_decision(drawn.signals[i], readings[i],
                                                    drawn.geometries[drawn.signals[i].airport], spec)
                for i in range(len(readings))}
    print(f"H_final {spec.closed_loop_final_vertical_m:g} m: {len(readings)} {args.split} flights, "
          f"{time.perf_counter() - started:.0f}s", flush=True)
    intervals, flights = {}, {}
    device = torch.device(args.device)
    out.mkdir(parents=True)
    for interval in args.row_interval_s:
        numbers, rows = read_interval(drawn, readings, observed, interval, params, words, chunk=args.chunk,
                                      device=device)
        intervals[f"{interval:g}"] = {"closed_loop": numbers, "replayed": len(rows), "table": table(rows)}
        flights[f"{interval:g}"] = rows
        write_json_atomic(out / f"interval_{interval:g}s.json", {"schema": READOUT_SCHEMA, "row_interval_s": interval,
                                                                 **intervals[f"{interval:g}"], "flights": rows})
        cells = intervals[f"{interval:g}"]["table"]
        landed = f"landed {cells['all']['landed']:.4f}, unstable {cells['all']['unstable_at_minimums']}" if cells else ""
        print(f"  Δ {interval:g} s: {len(rows)} flown, {landed}, {time.perf_counter() - started:.0f}s", flush=True)
    write_json_atomic(out / "readout.json", {
        "schema": READOUT_SCHEMA, "written_utc": utc_now(), "git": git,
        "instructions": str(instructions), "executor": str(executor), "executor_spec_sha256": record["sha256"],
        "executor_params_sha256": params_sha256(params),
        "vocabulary_spec_sha256": spec.sha256, "spec": spec.to_dict(),
        "closed_loop_final_vertical_m": spec.closed_loop_final_vertical_m, "split": args.split,
        "per_airport": args.per_airport, "seed": args.seed, "drawn": drawn.description,
        "row_intervals_s": args.row_interval_s, "intervals": intervals, "flights": flights,
        "elapsed_s": time.perf_counter() - started})
    print(f"→ {out / 'readout.json'}")
    return 0


def side_by_side(readouts: list[dict[str, Any]]) -> dict[str, Any]:
    """Readouts (one per H_final and split) side by side: per split, row interval and H_final, each readout's cells.
    Refused unless the specs differ only in H_final, the executor parameters are the same and a split's readouts drew
    the same flights at the same row intervals."""
    base = {k: v for k, v in readouts[0]["spec"].items() if k != "closed_loop_final_vertical_m"}
    for one in readouts:
        if one["schema"] != READOUT_SCHEMA:
            raise ValueError(f"not a {READOUT_SCHEMA} readout: {one['schema']}")
        if {k: v for k, v in one["spec"].items() if k != "closed_loop_final_vertical_m"} != base:
            raise ValueError("the readouts' specs differ in more than H_final")
        if one["executor_params_sha256"] != readouts[0]["executor_params_sha256"]:
            raise ValueError("the readouts were flown with other executor parameters")
    out: dict[str, Any] = {}
    for split in sorted({r["split"] for r in readouts}):
        group = sorted((r for r in readouts if r["split"] == split), key=lambda r: -r["closed_loop_final_vertical_m"])
        values = [r["closed_loop_final_vertical_m"] for r in group]
        if len(set(values)) != len(values):
            raise ValueError(f"two {split} readouts at one H_final")
        if any((r["drawn"], r["row_intervals_s"]) != (group[0]["drawn"], group[0]["row_intervals_s"]) for r in group):
            raise ValueError(f"the {split} readouts drew other flights or read other row intervals")
        out[split] = {"drawn": group[0]["drawn"], "intervals": {
            interval: {f"{r['closed_loop_final_vertical_m']:g}": {"closed_loop": r["intervals"][interval]["closed_loop"],
                                                                 "table": r["intervals"][interval]["table"]}
                       for r in group}
            for interval in group[0]["intervals"]}}
    return out


def _cell(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def markdown(tables: dict[str, Any]) -> str:
    """``table.md``: per split and row interval, one row per H_final (pooled), then the landed share per airport."""
    lines = []
    for split, part in tables.items():
        for interval, by_value in part["intervals"].items():
            lines += [f"## {split}, Δ = {interval} s", "",
                      "| H_final (m) | flights | landed % | unstable: none / high / low / lateral | above glidepath at DA "
                      "p10 / p50 / p90 (m) | observed passes, replay not | angle corrections per final descent | "
                      "vertical rows outside (share) |", "|---|---|---|---|---|---|---|---|"]
            for value, cell in by_value.items():
                pooled = cell["table"]["all"]
                above = pooled["above_glidepath_at_da_m"]
                outside = cell["closed_loop"]["outside_the_tolerance"]["vertical"]
                lines.append(
                    f"| {value} | {pooled['flights']} | {100 * pooled['landed']:.1f} | "
                    + " / ".join(str(pooled["unstable_at_minimums"][k]) for k in UNSTABLE_KINDS) + " | "
                    + (" / ".join(_cell(above[f"p{q}"]) for q in PERCENTILES) if above else "—") + " | "
                    + f"{pooled['da_check_observed_vs_replay']['observed passes, replay does not']} | "
                    + f"{_cell(pooled['angle_corrections_per_final_descent'], 2)} | "
                    + f"{_cell(None if outside['share'] is None else 100 * outside['share'])} % |")
            airports = [a for a in next(iter(by_value.values()))["table"] if a != "all"]
            lines += ["", "| H_final (m) | " + " | ".join(f"{a} landed %" for a in airports) + " |",
                      "|---|" + "---|" * len(airports)]
            for value, cell in by_value.items():
                lines.append(f"| {value} | " + " | ".join(f"{100 * cell['table'][a]['landed']:.1f}" for a in airports)
                             + " |")
            lines.append("")
    return "\n".join(lines)


def argv_has_reading_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> bool:
    """Whether a reading's own options (`--per-airport`, `--seed`, `--chunk`, `--device`) were moved off their defaults."""
    return any(getattr(args, name) != parser.get_default(name) for name in ("per_airport", "seed", "chunk", "device"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, default=None)
    parser.add_argument("--executor", type=Path, default=None, help="the executor spec directory")
    parser.add_argument("--split", choices=("train", "select"), default=None)
    parser.add_argument("--per-airport", type=int, default=0, help="0: every labelled flight of the split")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--row-interval-s", type=float, nargs="+", default=None,
                        help="the row intervals of the ablation (vocabulary §4.8, D25: 2, 4, 8 s)")
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--table", type=Path, nargs="+", default=None, help="readout directories to put side by side")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)
    reading = (args.instructions, args.executor, args.split, args.row_interval_s)
    if args.table is not None:
        if any(value is not None for value in reading) or argv_has_reading_options(parser, args):
            parser.error("--table puts readouts side by side and reads no artefact")
        out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
        if out.exists():
            parser.error(f"{out} exists; a table is never overwritten")
        tables = side_by_side([json.loads((d / "readout.json").read_text(encoding="utf-8")) for d in args.table])
        out.mkdir(parents=True)
        write_json_atomic(out / "table.json", {"schema": TABLE_SCHEMA, "written_utc": utc_now(),
                                               "readouts": [str(d) for d in args.table], "tables": tables})
        (out / "table.md").write_text(markdown(tables), encoding="utf-8")
        print(f"→ {out / 'table.md'}")
        return 0
    if any(value is None for value in reading):
        parser.error("a readout needs --instructions, --executor, --split and --row-interval-s")
    return readout(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
