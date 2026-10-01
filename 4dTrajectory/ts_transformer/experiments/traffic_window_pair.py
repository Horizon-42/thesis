"""Two window readouts of the same windows, aircraft by aircraft (multi-aircraft design §6.6 step 7; the user 2026-10-01:
the kept M4-in-windows round against its start on the val windows).

Reads two `traffic_window_generation` (R34) outputs — ``--first`` and ``--second``, typically two priors — and refuses
them unless they read the SAME windows the SAME way: every field of their ``window_generation.json`` that decides what
is drawn and how it is read (`SAME`: the split, the draw, the windows an airport, the samples, the temperature, the seed,
the augmentation, the executor spec, the sentence artefact, the scenes, the history, the readings, the batches and their
size) equal; the same rows (window, aircraft, sample, source) in both; every row's ``starts_in_a_loss`` equal (it reads
only the observed rows); and the sources no model reads — the labelled words and the record — equal row for row. Each
batch draws from its own streams, so the two read every window with the same random numbers: their rows pair.

For each model source (`MODEL_SOURCES`) and each group (pooled, airport, window size and — augmented windows — kind and
part in the augmentation), over the rows the readout counts (not starting in a loss): the share of each `MEASURES` in
both, and second − first. The difference is taken per aircraft — its samples averaged — and its standard error is
clustered by window (one window's aircraft fly together); beside it, the reward's error read per sentence as the M4
round choice reads its pairs (`traffic_reward.paired_difference`). Counted per sentence: the losses of separation the
second avoids and the ones it adds.

    python run_ts.py traffic_window_pair --first 4dTrajectory/outputs/POOLED/traffic/<readout> \\
        --second 4dTrajectory/outputs/POOLED/traffic/<readout> [--out <new directory>]

``--out`` writes ``traffic_window_pair.json`` (`SCHEMA`).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ts_transformer.experiments.traffic_loop import LOST_SEPARATION
from ts_transformer.experiments.traffic_reward import paired_difference
from ts_transformer.experiments.traffic_window_augment import KINDS, ROLES
from ts_transformer.experiments.traffic_window_generation import SCHEMA as READOUT_SCHEMA, SIZES, size_of
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, repo_relative

SCHEMA = "ts-traffic-window-pair-v1"
#: What decides which windows are read and how: equal in both readouts, or they do not pair.
SAME = ("schema", "split", "drawn", "windows_per_airport", "samples", "temperature", "seed", "augment_seed", "augmenting",
        "executor", "instructions", "scenes", "history_s", "readings", "aircraft_steps", "batches")
MODEL_SOURCES = ("scene", "alone")
FIXED_SOURCES = ("labelled", "recorded")
#: Each sentence's 0 / 1 readings (R34's own: `traffic_window_generation.summary`, M4's reward).
MEASURES: dict[str, Callable[[Mapping[str, Any]], float]] = {
    "reward": lambda r: float(r["reward"]),
    "lost_separation": lambda r: float(r["outcome"] == LOST_SEPARATION),
    "lost_separation_ifr": lambda r: float(r["ifr_outcome"] == LOST_SEPARATION),
    "landed": lambda r: float(r["outcome"] == "landed"),
    "landed_on_observed_runway": lambda r: float(r["outcome"] == "landed" and r["runway"] == r["observed_runway"]),
}


def read(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header = json.loads((directory / "window_generation.json").read_text(encoding="utf-8"))
    if header["schema"] != READOUT_SCHEMA:
        raise ValueError(f"{directory} is {header['schema']!r}, not a {READOUT_SCHEMA} window readout")
    with (directory / header["aircraft_file"]).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return header, rows


def key_of(row: Mapping[str, Any]) -> tuple[int, str, Any, str]:
    return row["window"], row["dataset_id"], row["sample"], row["source"]


def require_pairs(first: tuple[dict[str, Any], list[dict[str, Any]]],
                  second: tuple[dict[str, Any], list[dict[str, Any]]]) -> dict[tuple, tuple[dict, dict]]:
    """The two readouts' rows paired by key, refused unless they read the same windows the same way (module docstring)."""
    (head_a, rows_a), (head_b, rows_b) = first, second
    differing = [name for name in SAME if head_a[name] != head_b[name]]
    if differing:
        raise ValueError(f"the two readouts did not read the same windows the same way: {differing} differ")
    by_a, by_b = {key_of(r): r for r in rows_a}, {key_of(r): r for r in rows_b}
    if len(by_a) != len(rows_a) or len(by_b) != len(rows_b):
        raise ValueError("a readout holds a (window, aircraft, sample, source) twice")
    if by_a.keys() != by_b.keys():
        raise ValueError(f"the readouts hold other rows: {len(by_a.keys() - by_b.keys())} only in the first, "
                         f"{len(by_b.keys() - by_a.keys())} only in the second")
    moved = [k for k in by_a if by_a[k]["starts_in_a_loss"] != by_b[k]["starts_in_a_loss"]]
    if moved:
        raise ValueError(f"{len(moved)} rows start in a loss in one readout and not the other (e.g. {moved[0]})")
    fixed = [k for k in by_a if k[3] in FIXED_SOURCES and by_a[k] != by_b[k]]
    if fixed:
        raise ValueError(f"{len(fixed)} rows no model reads differ between the readouts (e.g. {fixed[0]}): not the same "
                         f"draws")
    counted = [k for k in by_a if k[3] in MODEL_SOURCES and not by_a[k]["starts_in_a_loss"]]
    return {k: (by_a[k], by_b[k]) for k in sorted(counted)}


def clustered(per_aircraft: Mapping[tuple[int, str], float]) -> tuple[float, float, int]:
    """The mean of the per-aircraft differences and its standard error clustered by window, with the aircraft counted."""
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for (window, _), value in per_aircraft.items():
        sums[window] += value
        counts[window] += 1
    n = sum(counts.values())
    mean = sum(sums.values()) / n
    k = len(sums)
    if k < 2:
        return mean, math.nan, n
    spread = sum((sums[w] - mean * counts[w]) ** 2 for w in sums)
    return mean, math.sqrt(k / (k - 1) * spread) / n, n


def compare(pairs: Mapping[tuple, tuple[dict, dict]]) -> dict[str, Any]:
    """Each measure in both readouts and second − first over ``pairs`` (one source, one group)."""
    out: dict[str, Any] = {"sentences": len(pairs)}
    for name, value in MEASURES.items():
        aircraft: dict[tuple[int, str], list[float]] = defaultdict(list)
        first, second = [], []
        for key, (a, b) in pairs.items():
            first.append(value(a))
            second.append(value(b))
            aircraft[(key[0], key[1])].append(value(b) - value(a))
        mean, error, n = clustered({k: float(np.mean(v)) for k, v in aircraft.items()})
        out[name] = {"first": float(np.mean(first)), "second": float(np.mean(second)), "difference": mean,
                     "standard_error": error}
        out["aircraft"] = n
    reward_first = {key: MEASURES["reward"](a) for key, (a, _) in pairs.items()}
    reward_second = {key: MEASURES["reward"](b) for key, (_, b) in pairs.items()}
    out["reward"]["per_sentence"] = dict(zip(("difference", "standard_error"),
                                             paired_difference(reward_first, reward_second)))
    lost = MEASURES["lost_separation"]
    out["lost_separation"]["avoided"] = sum(lost(a) == 1.0 and lost(b) == 0.0 for a, b in pairs.values())
    out["lost_separation"]["added"] = sum(lost(a) == 0.0 and lost(b) == 1.0 for a, b in pairs.values())
    return out


def groups(pairs: Mapping[tuple, tuple[dict, dict]], augmented: bool) -> dict[str, dict[str, dict]]:
    """``pairs`` split as R34's readout splits its rows (`traffic_window_generation.summaries`)."""
    out: dict[str, dict[str, dict]] = {"pooled": {"all": dict(pairs)}, "airports": defaultdict(dict),
                                       "window_sizes": defaultdict(dict)}
    for key, (a, b) in pairs.items():
        out["airports"][a["airport"]][key] = (a, b)
        out["window_sizes"][size_of(a["commanded"])][key] = (a, b)
    out["window_sizes"] = {size: out["window_sizes"][size] for size in SIZES if out["window_sizes"][size]}
    out["airports"] = dict(sorted(out["airports"].items()))
    if augmented:
        out["kinds"] = {kind: {k: p for k, p in pairs.items() if p[0]["augmented"]["kind"] == kind} for kind in KINDS}
        out["roles"] = {role or "as drawn": {k: p for k, p in pairs.items() if p[0]["role"] == role}
                        for role in (*ROLES, None)}
        out["kinds"] = {k: v for k, v in out["kinds"].items() if v}
        out["roles"] = {k: v for k, v in out["roles"].items() if v}
    return out


def pair(first_dir: Path, second_dir: Path) -> dict[str, Any]:
    first, second = read(first_dir), read(second_dir)
    pairs = require_pairs(first, second)
    augmented = first[0]["augment_seed"] is not None
    report: dict[str, Any] = {}
    for source in MODEL_SOURCES:
        mine = {k: p for k, p in pairs.items() if k[3] == source}
        if mine:
            report[source] = {group: {name: compare(part) for name, part in parts.items()}
                              for group, parts in groups(mine, augmented).items()}
    return {"schema": SCHEMA, "written_utc": utc_now(),
            "first": {"directory": repo_relative(first_dir), "prior": first[0]["prior"]},
            "second": {"directory": repo_relative(second_dir), "prior": second[0]["prior"]},
            "same": {name: first[0][name] for name in ("split", "windows_per_airport", "samples", "seed",
                                                        "augment_seed")},
            "measures": list(MEASURES), "report": report}


def _line(name: str, entry: Mapping[str, Any]) -> str:
    def cell(m: str, scale: float, unit: str) -> str:
        e = entry[m]
        return (f"{m} {scale * e['first']:.2f}{unit} → {scale * e['second']:.2f}{unit} "
                f"({scale * e['difference']:+.2f} ± {scale * e['standard_error']:.2f})")
    reward = entry["reward"]
    return (f"  {name:10s} {entry['aircraft']:5d} aircraft {entry['sentences']:6d} sentences | reward "
            f"{reward['first']:.4f} → {reward['second']:.4f} ({reward['difference']:+.4f} ± {reward['standard_error']:.4f}; "
            f"per sentence ± {reward['per_sentence']['standard_error']:.4f}) | {cell('lost_separation', 100, ' %')} "
            f"avoided {entry['lost_separation']['avoided']} added {entry['lost_separation']['added']} | "
            f"{cell('lost_separation_ifr', 100, ' %')} | {cell('landed', 100, ' %')} | "
            f"{cell('landed_on_observed_runway', 100, ' %')}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--first", type=Path, required=True, help="a window readout (R34) directory")
    parser.add_argument("--second", type=Path, required=True, help="another, of the same windows read the same way")
    parser.add_argument("--out", type=Path, default=None, help="a new directory for traffic_window_pair.json")
    args = parser.parse_args(argv)
    first, second = ((p if p.is_absolute() else REPO_ROOT / p) for p in (args.first, args.second))
    out = None if args.out is None else (args.out if args.out.is_absolute() else REPO_ROOT / args.out)
    if out is not None and out.exists():
        parser.error(f"{out} exists: the pair is written into a new directory")
    result = pair(first, second)
    print(f"second − first: {result['second']['prior']['directory']} − {result['first']['prior']['directory']} "
          f"({result['same']})")
    for source, parts in result["report"].items():
        print(f"{source}:")
        for group, entries in parts.items():
            for name, entry in entries.items():
                print(_line(name if group == "pooled" else f"{group[:4]} {name}", entry))
    if out is not None:
        out.mkdir(parents=True)
        write_json_atomic(out / "traffic_window_pair.json", result)
        print(f"→ {out / 'traffic_window_pair.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
