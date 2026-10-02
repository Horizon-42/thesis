"""Does a window readout read the same under today's code? (multi-aircraft design §6.6 step 9.9.2; the user 2026-10-02:
code that read two readouts is shown to behave the same by what it reads, never by an equal commit.)

Reads a `traffic_window_generation` (R34) readout's ``config.json`` with the readout's own loader and builds what it
read with today's code and the readout's own functions (`traffic_window_generation.prepare`: the prior, the executor, the
scenes, the draw, the augmentation, the batches — the checksums of the prior's checkpoint and the executor spec must be
the disk's, or the data changed, not the code). Then, against the readout's ``aircraft.jsonl``:

1. **the whole draw** (no model, quick): the same batches, and in each the same (window, flight) pairs as its rows —
   every window's draw, augmentation and batch;
2. **``--batches`` batches read again** (24 by default: `numpy.linspace` over the batch numbers, rounded, each once; the
   batches run from the smallest windows to the largest, so the ones read span the sizes), in forked processes as the
   readout reads (`traffic_window_generation.read_batches`), as the configuration says: row by row — the same rows by
   (window, flight, sample, source), each with the same fields of the same values (both written as JSON with sorted
   keys: a float bit for bit, NaN for NaN).

Not read: the summary, the counts, the run record — the configuration and the code make them; the same rows, the same
numbers. A check passes or prints the first difference (which batch, which row, which fields, both values) and exits
non-zero. Passed on a clean checkout, it writes ``<readout>.conformance/passed-<commit 12>.json`` beside the readout
(read-only): today's code version (`code_version.code_version`, the keys of the readout's ``code.json``), the readout,
the batches and the ones read, the rows compared, the time — the record `traffic_window_compare` accepts as the code
version's evidence. A dirty checkout's check is read and reported, never written; a commit's record is never written
over.

What it cannot see (the design's limits): a change that alters only batches not read; a row field added or dropped
fails it ("other fields"); another torch / CUDA / GPU may move a float — reported as it is.

    python run_ts.py traffic_window_conformance --readout 4dTrajectory/outputs/POOLED/traffic/<readout> \\
        [--batches 24] [--workers 6] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ts_transformer.experiments.code_version import code_version
from ts_transformer.experiments.traffic_window_generation import (
    READING_CONSTANTS, WORKERS, Prepared, prepare, read_aircraft, read_batches, readout_config,
)
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, git_state, repo_relative

#: Batches read again by default (the user, 2026-10-03).
BATCHES = 24
#: Beside a readout ``<readout>``: ``<readout>.conformance/passed-<commit 12>.json``.
RECORDS_SUFFIX = ".conformance"


class Differs(Exception):
    """The readout does not read the same under today's code: where first."""


def records_dir(readout: Path) -> Path:
    return readout.with_name(readout.name + RECORDS_SUFFIX)


def record_path(readout: Path, commit: str) -> Path:
    return records_dir(readout) / f"passed-{commit[:12]}.json"


def passed_records(readout: Path) -> list[tuple[Path, dict[str, Any]]]:
    """The readout's passed records (`record_path`), in name order."""
    directory = records_dir(readout)
    if not directory.is_dir():
        return []
    return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in sorted(directory.glob("passed-*.json"))]


def chosen_batches(count: int, batches: int) -> list[int]:
    """``batches`` of ``count`` batch numbers, evenly from the first to the last (all of them when there are fewer)."""
    return sorted({int(round(x)) for x in np.linspace(0, count - 1, min(batches, count))})


def row_key(row: Mapping[str, Any]) -> tuple[int, str, Any, str]:
    return row["window"], row["dataset_id"], row["sample"], row["source"]


def check_draw(prepared: Prepared, rows: Sequence[Mapping[str, Any]]) -> None:
    """The readout's rows hold today's batches, and each batch today's (window, flight) pairs (module docstring 1)."""
    drawn, batches = prepared.drawn, prepared.batches
    stored: dict[int, set[tuple[int, str]]] = {}
    for row in rows:
        stored.setdefault(row["batch"], set()).add((row["window"], row["dataset_id"]))
    if sorted(stored) != list(range(len(batches))):
        raise Differs(f"the readout's rows are in batches {sorted(stored)[:5]}… ({len(stored)}), today's draw makes "
                      f"{len(batches)}")
    for number, windows in enumerate(batches):
        today = {(w, drawn.batch.signals[j].dataset_id) for w in windows for j in drawn.members[w]}
        if stored[number] != today:
            raise Differs(f"batch {number} holds other (window, flight) pairs: {sorted(stored[number] - today)[:3]} "
                          f"only in the readout, {sorted(today - stored[number])[:3]} only today")


def _written(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def check_rows(number: int, stored: Sequence[Mapping[str, Any]], read: Sequence[Mapping[str, Any]]) -> int:
    """Batch ``number``'s rows read today against the readout's: the same rows (`row_key`), each the same fields of the
    same values as written (`_written`); the rows compared."""
    if len(stored) != len(read):
        raise Differs(f"batch {number}: {len(stored)} rows in the readout, {len(read)} read today")
    by_key = {row_key(row): row for row in stored}
    today = {row_key(row): row for row in read}
    if len(by_key) != len(stored) or len(today) != len(read):
        raise Differs(f"batch {number} holds a (window, flight, sample, source) twice")
    if by_key.keys() != today.keys():
        raise Differs(f"batch {number}: rows {sorted(by_key.keys() - today.keys())[:3]} only in the readout, "
                      f"{sorted(today.keys() - by_key.keys())[:3]} only today")
    for key, row in by_key.items():
        new = today[key]
        if _written(row) == _written(new):
            continue
        if row.keys() != new.keys():
            raise Differs(f"batch {number}, row {key}: other fields — {sorted(row.keys() - new.keys())} only in the "
                          f"readout, {sorted(new.keys() - row.keys())} only today")
        fields = [name for name in row if _written(row[name]) != _written(new[name])]
        raise Differs(f"batch {number}, row {key}: fields {fields} differ — "
                      + "; ".join(f"{name}: readout {_written(row[name])[:200]}, today {_written(new[name])[:200]}"
                                  for name in fields[:3]))
    return len(stored)


def check(readout: Path, batches: int, workers: int, device: torch.device) -> dict[str, Any]:
    """The readout checked under today's code (module docstring): what was checked, or `Differs` at the first
    difference."""
    prepared = prepare(readout_config(readout))
    rows = read_aircraft(readout)
    check_draw(prepared, rows)
    by_batch: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_batch.setdefault(row["batch"], []).append(row)
    numbers = chosen_batches(len(prepared.batches), batches)
    print(f"the draw is the readout's ({len(prepared.batches)} batches, {len(rows)} rows); reading batches {numbers} "
          f"again", flush=True)
    compared = 0
    for number, read, _ in read_batches(prepared, numbers, workers, device):
        compared += check_rows(number, by_batch[number], read)
        print(f"  batch {number}: {len(read)} rows the same", flush=True)
    return {"batches": len(prepared.batches), "checked_batches": numbers, "rows_compared": compared,
            "rows": len(rows)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--readout", type=Path, required=True, help="a window readout (R34) directory")
    parser.add_argument("--batches", type=int, default=BATCHES, help="batches read again")
    parser.add_argument("--workers", type=int, default=WORKERS, help="reading processes (what is read does not depend "
                        "on it)")
    parser.add_argument("--device", default="cuda", help="the prior's; the executors fly on CPU")
    args = parser.parse_args(argv)
    if args.batches < 1 or args.workers < 1:
        parser.error("at least one batch and one reading process")
    readout = args.readout if args.readout.is_absolute() else REPO_ROOT / args.readout
    git = git_state()
    record = record_path(readout, git["head"])
    if record.exists():
        parser.error(f"{record} exists: this commit's check is written once")
    device = torch.device(args.device)
    started = time.perf_counter()
    try:
        checked = check(readout, args.batches, args.workers, device)
    except Differs as difference:
        print(f"NOT THE SAME: {difference}", flush=True)
        return 1
    elapsed = time.perf_counter() - started
    print(f"the same: {checked['rows_compared']} rows of {len(checked['checked_batches'])} batches, {elapsed:.0f}s",
          flush=True)
    if git["dirty"]:
        print("the checkout is dirty: nothing written", flush=True)
        return 0
    if record.exists():
        parser.error(f"{record} was written while this check ran: this commit's check is written once")
    write_json_atomic(record, {"readout": repo_relative(readout), "code": code_version(git, device, READING_CONSTANTS),
                               **checked, "elapsed_s": elapsed, "written_utc": utc_now()}, allow_nan=False)
    print(f"→ {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
