"""Executor design V12 (§17, D7, D8): the words' leads measured, one column at a time — the heading words' L, the angle
words' L_γ, the speed words' L_v — for the user's choice of each (Q2). NO CRITERION IS READ (D7).

One readout reads one instruction artefact and its executor spec, and changes ONE lead (``--column``) to each value of
``--leads-s``, every other value of the artefact's spec kept: the vocabulary spec with that lead (`spec_with_lead`; the
spec's own rules refuse a lead that is no whole number of steps). For each value, the flights of the train split
(``--per-airport`` of each airport, the replay's seeded sample, on their own dynamics or a stand-in's, `replay.draw_flights`)
are READ AGAIN by the labeller under that spec, in memory (`read_flight`; the flights it refuses counted by reason) — no
stored sentence is read, as the stored sentences are the artefact's lead's — and each read flight is read in closed loop
at every ``--row-interval-s`` (`closed_loop.read_chunked`, as `instruction_closed_loop.read_interval`, in memory: nothing
is written into the artefact), flown by the executor of this code with the spec's parameters: the executor reads the
leads from the vocabulary (the heading law, R5.2; the angle ramp, R5.3; the speed law, R5.5), so each value is flown as
it means.

Per value and row interval (``readout.json`` ``cells``):

- ``labeller``: the flights read and refused (by reason), and the column's envelopes on the OBSERVED tracks as the
  labeller checks them (`read_flight`'s ``checks``: the heading rows inside their band, the altitude tubes and the speed
  spans contained) — a lead moves where a word is said, so it moves how well the observed track keeps it;
- ``closed_loop``: the closed-loop reading's split numbers (`instruction_closed_loop.summarise`: the correction words by
  column, the rows outside the tolerance, the judge's outcomes — D74, for readouts) and the column's correction words per
  sentence;
- ``after_words``: the error of the flown path against the observed one around each OBSERVED word of the column that
  changes its value (not a correction, not the first step's), at the said rows ``OFFSETS`` from the word (§7.8's table,
  one column at a time): the lateral offset e_y for a heading word, the height e_h for an angle word (both the closed
  loop's own, at its matched point), and for a speed word the flown ground speed less the observed one at the matched
  point — p10 / p50 / p90 and the share beyond the column's tolerance (Y, H, the speed band δv).

Each value is written as it ends (``lead_<value>s.json``), ``readout.json`` with all of them and ``table.md`` at the end.

    python run_ts.py lead_ablation --instructions <artefact> --executor <spec> --column angle \\
        --leads-s 0 2 4 6 8 --per-airport 200 --row-interval-s 2 4 --out <new dir>
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_transformer.autopilot import closed_loop, replay
from ts_transformer.autopilot.spec import params_sha256
from ts_transformer.experiments import instruction_closed_loop
from ts_transformer.instructions.artefact import ClosedLoopSentence, STATE_COLUMNS
from ts_transformer.instructions.labeller.read import Reading, read_flight
from ts_transformer.instructions.labeller.records import Refused
from ts_transformer.instructions.spec import VocabularySpec
from ts_transformer.instructions.words import ANGLE, HEADING, SPEED, UNCHANGED, Words
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state

READOUT_SCHEMA = "ts-lead-ablation-readout-v1"
#: Each column the readout measures: its lead's spec field and its word column.
COLUMNS = {"heading": ("heading_lead_s", HEADING), "angle": ("angle_lead_s", ANGLE), "speed": ("speed_lead_s", SPEED)}
#: The said rows, from an observed word of the column, the error is read at (§7.8's table: one row before to four after).
OFFSETS = (-1, 0, 1, 2, 3, 4)
PERCENTILES = (10, 50, 90)
GROUND_SPEED = STATE_COLUMNS.index("ground_speed_mps")


def spec_with_lead(base: VocabularySpec, column: str, seconds: float) -> VocabularySpec:
    """``base`` with the lead of ``column`` (`COLUMNS`) at ``seconds``, every other value kept (refused by the spec's own
    rules where it cannot be one)."""
    return replace(base, **{COLUMNS[column][0]: float(seconds)})


def read_again(drawn: replay.Drawn, spec: VocabularySpec, words: Words) -> tuple[list[int], list[Reading], Counter]:
    """Every drawn flight read again by the labeller under ``spec``: the drawn positions read, their readings, and the
    flights refused by reason."""
    keep, readings, refused = [], [], Counter()
    for k, flight in enumerate(drawn.signals):
        try:
            readings.append(read_flight(flight, drawn.geometries[flight.airport], spec, words))
        except Refused as refusal:
            refused[refusal.reason] += 1
            continue
        keep.append(k)
    return keep, readings, refused


def envelopes(readings: list[Reading], column: str) -> dict[str, Any]:
    """The column's envelopes on the observed tracks, as the labeller checks them (module docstring)."""
    if column == "heading":
        bands = [band for r in readings for band in r.checks["heading"] if band["rows"]]
        rows = sum(band["rows"] for band in bands)
        return {"words": len(bands), "words_inside": sum(band["inside"] == band["rows"] for band in bands),
                "rows": rows, "rows_inside": sum(band["inside"] for band in bands),
                "row_share": sum(band["inside"] for band in bands) / rows if rows else None}
    checks = [check for r in readings for check in r.checks["vertical" if column == "angle" else "speed"]]
    contained = sum(bool(check["contained"]) for check in checks)
    return {"words": len(checks), "words_contained": contained, "share": contained / len(checks) if checks else None}


def word_errors(sentence: ClosedLoopSentence, observed_speed_mps: np.ndarray, column: str) -> dict[int, list[float]]:
    """``OFFSETS`` → the errors at the said rows that far from each observed word of ``column`` changing its value
    (module docstring); ``observed_speed_mps`` the observed flight's ground speed on its 2 s rows (from row 0)."""
    rows, withheld = sentence.rows, sentence.withheld
    index = COLUMNS[column][1]
    said = rows.grid[:, index]
    if column == "heading":
        error = withheld.lateral_m
    elif column == "angle":
        error = withheld.vertical_m
    else:
        state_rows = np.flatnonzero(rows.on_interval)[rows.start: rows.start + len(said)]
        flown = rows.states[state_rows, GROUND_SPEED]
        at = rows.first_row + withheld.matched_row
        error = flown - np.interp(at, np.arange(len(observed_speed_mps)), observed_speed_mps)
    out: dict[int, list[float]] = {offset: [] for offset in OFFSETS}
    in_force = None
    for m, value in enumerate(said):
        if value == UNCHANGED:
            continue
        changed = m > 0 and not rows.correction[m, index] and in_force is not None and value != in_force
        in_force = int(value)
        if not changed:
            continue
        for offset in OFFSETS:
            if 0 <= m + offset < len(error) and np.isfinite(error[m + offset]):
                out[offset].append(float(error[m + offset]))
    return out


def error_table(errors: dict[int, list[float]], tolerance: float) -> dict[str, Any]:
    """Per offset: how many, p10 / p50 / p90, and the share beyond ``tolerance``."""
    out = {}
    for offset, values in errors.items():
        a = np.asarray(values, dtype=np.float64)
        out[str(offset)] = None if not len(a) else {
            "n": int(len(a)), **{f"p{q}": float(np.percentile(a, q)) for q in PERCENTILES},
            "beyond_tolerance": float((np.abs(a) > tolerance).mean())}
    return out


def tolerance_of(spec: VocabularySpec, column: str) -> float:
    """The column's tolerance (the closed loop's Y and H, the speed band δv)."""
    return {"heading": spec.closed_loop_lateral_m, "angle": spec.closed_loop_vertical_m,
            "speed": spec.speed_tolerance_mps}[column]


def read_value(drawn: replay.Drawn, base: VocabularySpec, column: str, seconds: float, intervals: list[float],
               params: Any, *, chunk: int, device: torch.device) -> dict[str, Any]:
    """One value of the lead: the labeller's reading of every drawn flight, then each row interval's closed loop and the
    errors around the column's words (module docstring)."""
    spec = spec_with_lead(base, column, seconds)
    words = Words(spec)
    keep, readings, refused = read_again(drawn, spec, words)
    cell: dict[str, Any] = {"lead_s": float(seconds), "spec_sha256": spec.sha256,
                            "labeller": {"flights": len(drawn.signals), "read": len(readings),
                                         "refused": dict(refused.most_common()), "envelopes": envelopes(readings, column)},
                            "intervals": {}}
    index = COLUMNS[column][1]
    for interval in intervals:
        batch = replay.batch_of(drawn, keep, readings, interval, words)
        results = closed_loop.read_chunked(batch, params, words, chunk=chunk, device=device)
        counted = instruction_closed_loop.tally(batch, results, words)
        errors: dict[int, list[float]] = {offset: [] for offset in OFFSETS}
        for j, result in enumerate(results):
            if isinstance(result, ClosedLoopSentence):
                for offset, values in word_errors(result, batch.observed[j].ground_speed_mps, column).items():
                    errors[offset] += values
        sentences = len(counted.corrections)
        outcomes = Counter(counted.outcomes)
        cell["intervals"][f"{interval:g}"] = {
            "closed_loop": instruction_closed_loop.summarise(counted, drawn.description["excluded"]),
            "corrections_per_sentence": float(counted.corrections[:, index].mean()) if sentences else None,
            "landed": outcomes["landed"] / sentences if sentences else None,
            "after_words": error_table(errors, tolerance_of(spec, column))}
    return cell


def _cell(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def markdown(column: str, cells: list[dict[str, Any]], intervals: list[float]) -> str:
    """``table.md``: per row interval one row per value of the lead."""
    lines = [f"# {column} lead ({COLUMNS[column][0]})", ""]
    for interval in intervals:
        key = f"{interval:g}"
        lines += [f"## Δ = {key} s", "",
                  "| lead (s) | read / refused | envelope (observed) | corrections per sentence | landed % | "
                  "error p50 at −1 / 0 / +1 / +2 / +3 / +4 rows | beyond tolerance at +1 / +2 / +3 |",
                  "|---|---|---|---|---|---|---|"]
        for cell in cells:
            labeller, part = cell["labeller"], cell["intervals"][key]
            envelope = labeller["envelopes"]
            share = envelope.get("row_share", envelope.get("share"))
            after = part["after_words"]
            p50 = " / ".join(_cell(None if after[str(o)] is None else after[str(o)]["p50"]) for o in OFFSETS)
            beyond = " / ".join(_cell(None if after[str(o)] is None else 100 * after[str(o)]["beyond_tolerance"], 0)
                                for o in (1, 2, 3))
            lines.append(f"| {cell['lead_s']:g} | {labeller['read']} / {sum(labeller['refused'].values())} | "
                         f"{_cell(None if share is None else 100 * share)} % | "
                         f"{_cell(part['corrections_per_sentence'], 2)} | "
                         f"{_cell(None if part['landed'] is None else 100 * part['landed'])} | {p50} | {beyond} % |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--instructions", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True, help="the executor spec directory")
    parser.add_argument("--column", choices=tuple(COLUMNS), required=True, help="the one lead the readout changes")
    parser.add_argument("--leads-s", type=float, nargs="+", required=True, help="the values of that lead")
    parser.add_argument("--per-airport", type=int, required=True, help="train flights of each airport (D34's sample)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--row-interval-s", type=float, nargs="+", required=True,
                        help="the row intervals of the closed loop (vocabulary §4.8: 2, 4, 8 s)")
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)
    if len(set(args.leads_s)) != len(args.leads_s):
        parser.error("--leads-s holds a value twice")
    if args.per_airport <= 0:
        parser.error("--per-airport must be positive (the train split is sampled, D34)")
    instructions = args.instructions if args.instructions.is_absolute() else REPO_ROOT / args.instructions
    executor = args.executor if args.executor.is_absolute() else REPO_ROOT / args.executor
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    params, record, base_words = replay.open_executor(executor, instructions)
    base = base_words.spec
    for seconds in args.leads_s:                                   # refused by the spec's own rules before any flight
        try:
            spec_with_lead(base, args.column, seconds)
        except ValueError as error:
            parser.error(f"--leads-s {seconds:g}: {error}")
    started = time.perf_counter()
    drawn = replay.draw_flights(instructions, "train", list(range(len(replay.load_signals(instructions, "train")))),
                                per_airport=args.per_airport, seed=args.seed, groups=(replay.OWN, replay.STAND_IN))
    print(f"{args.column} lead: {len(drawn.signals)} train flights drawn, {time.perf_counter() - started:.0f}s",
          flush=True)
    device = torch.device(args.device)
    out.mkdir(parents=True)
    cells = []
    for seconds in args.leads_s:
        cell = read_value(drawn, base, args.column, seconds, args.row_interval_s, params, chunk=args.chunk,
                          device=device)
        cells.append(cell)
        write_json_atomic(out / f"lead_{seconds:g}s.json", {"schema": READOUT_SCHEMA, "column": args.column, **cell})
        landed = ", ".join(f"Δ {k}: landed {_cell(None if v['landed'] is None else 100 * v['landed'])} %"
                           for k, v in cell["intervals"].items())
        print(f"  {seconds:g} s: read {cell['labeller']['read']}, {landed}, {time.perf_counter() - started:.0f}s",
              flush=True)
    write_json_atomic(out / "readout.json", {
        "schema": READOUT_SCHEMA, "written_utc": utc_now(), "git": git_state(), "instructions": str(instructions),
        "executor": str(executor), "executor_spec_sha256": record["sha256"],
        "executor_params_sha256": params_sha256(params), "checks": record["checks"],
        "base_spec_sha256": base.sha256, "base_spec": base.to_dict(), "column": args.column,
        "lead_field": COLUMNS[args.column][0], "per_airport": args.per_airport, "seed": args.seed,
        "drawn": drawn.description, "row_intervals_s": args.row_interval_s, "offsets_rows": list(OFFSETS),
        "cells": cells, "elapsed_s": time.perf_counter() - started})
    (out / "table.md").write_text(markdown(args.column, cells, args.row_interval_s), encoding="utf-8")
    print(f"→ {out / 'readout.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
