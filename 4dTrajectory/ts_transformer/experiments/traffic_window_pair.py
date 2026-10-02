"""Two window readouts of the same windows, aircraft by aircraft (multi-aircraft design §6.6 step 7; the user 2026-10-01:
the kept M4-in-windows round against its start on the val windows).

Reads two `traffic_window_generation` (R34) outputs — ``--first`` and ``--second``, typically two priors — and refuses
them unless they read the SAME windows the SAME way with the SAME code: every field of their ``window_generation.json``
that decides what is drawn and how it is read (`SAME`: the code's commit, the split, the draw, the windows an airport, the
samples, the temperature, the seed, the augmentation, the executor spec, the sentence artefact, the scenes, the history,
the readings, the batches and their size, the model's sources read) equal; the same rows (window, aircraft, sample, source) in both, whose fields no
model decides (`MODEL_FREE`) are equal; and the sources no model reads — the labelled words and the record — equal row for
row (they also read the prior's procedure masks: two priors under other masks differ there). Each batch draws from its
own streams, so the two read every window from the same random numbers as far as their words agree.

A row STARTING IN A LOSS depends on the model too: an aircraft that enters later is judged through its observed rows
against the commanded aircraft ahead of it, flown by the model. As the M4 round choice pairs its rounds
(`traffic_window_reward.select_counted`, `traffic_rounds.paired_difference`), a sentence is counted only where it starts
in a loss in neither readout; how many start in a loss in each is reported beside.

For each model source (`MODEL_SOURCES`) and each group (pooled, airport, window size and — augmented windows — kind and
part in the augmentation), over the sentences counted in both: the share of each `MEASURES` in both and second − first,
with its standard error CLUSTERED BY AIRPORT AND OPERATING DAY (`cluster_of`): windows overlap (one opens every 10
minutes, 20 minutes long), share flights and replayed traffic, and a day's runway configuration and weather reach them
all; beside it, the reward's error read per sentence as the M4 round choice reads its pairs. Counted per sentence: the
losses of separation the second avoids and the ones it adds.

    python run_ts.py traffic_window_pair --first 4dTrajectory/outputs/POOLED/traffic/<readout> \\
        --second 4dTrajectory/outputs/POOLED/traffic/<readout> [--out <new directory>]

``--out`` writes ``traffic_window_pair.json`` (`SCHEMA`).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ts_transformer.data.day_split import operational_day
from ts_transformer.experiments.traffic_loop import LOST_SEPARATION
from ts_transformer.experiments.traffic_rounds import paired_difference
from ts_transformer.experiments.traffic_speaking import INSERTED
from ts_transformer.experiments.traffic_window_augment import KINDS, ROLES
from ts_transformer.experiments.traffic_window_generation import (
    MODEL_SOURCES, SCHEMA as READOUT_SCHEMA, SIZES, size_of,
)
from ts_transformer.io_utils import utc_now, write_json_atomic
from ts_transformer.repo_layout import REPO_ROOT, repo_relative

SCHEMA = "ts-traffic-window-pair-v1"
#: What decides which windows are read and how: equal in both readouts, or they do not pair.
SAME = ("schema", "git", "split", "drawn", "windows_per_airport", "samples", "temperature", "seed", "augment_seed",
        "augmenting", "executor", "instructions", "scenes", "history_s", "readings", "aircraft_steps", "batches",
        "model_sources", "probes")
#: A model row's fields no model decides: equal in both readouts.
MODEL_FREE = ("airport", "commanded", "observed_runway", "augmented", "role", "batch")
#: A flight's identity ends in its landing time (`flight_scenarios.identity.flight_key`): what its operating day is read from.
LANDING_STAMP = re.compile(r"(\d{8}T\d{6}Z)$")
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
    model = [k for k in by_a if k[3] in MODEL_SOURCES]
    moved = [k for k in model if any(by_a[k][f] != by_b[k][f] for f in MODEL_FREE)]
    if moved:
        raise ValueError(f"{len(moved)} model rows differ in what no model decides ({MODEL_FREE}; e.g. {moved[0]}): not "
                         f"the same windows")
    fixed = [k for k in by_a if k[3] in FIXED_SOURCES and by_a[k] != by_b[k]]
    if fixed:
        raise ValueError(f"{len(fixed)} rows no model reads differ between the readouts (e.g. {fixed[0]}): not the same "
                         f"draws, or the two priors' procedure masks differ")
    return {k: (by_a[k], by_b[k]) for k in sorted(model)}


def operating_day_of(dataset_id: str) -> str:
    """The operating day a flight landed on, read from its identity's landing stamp."""
    found = LANDING_STAMP.search(dataset_id)
    if found is None:
        raise ValueError(f"{dataset_id!r} does not end in a landing stamp")
    return operational_day(datetime.strptime(found.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc))


def cluster_of(pairs: Mapping[tuple, tuple[dict, dict]]) -> dict[int, tuple[str, str]]:
    """Each window's cluster: its airport and the operating day of its earliest-landing commanded aircraft of its own —
    an inserted one (`traffic_speaking.INSERTED`, augmentation A) is another day's flight put into it, its stamp not
    the window's."""
    first_landing: dict[int, str] = {}
    airport: dict[int, str] = {}
    for (window, key, _, _), (a, _) in pairs.items():
        airport[window] = a["airport"]
        if key.endswith(INSERTED):
            continue
        stamp = LANDING_STAMP.search(key)
        if stamp is None:
            raise ValueError(f"{key!r} does not end in a landing stamp")
        if window not in first_landing or stamp.group(1) < LANDING_STAMP.search(first_landing[window]).group(1):
            first_landing[window] = key
    return {w: (airport[w], operating_day_of(key)) for w, key in first_landing.items()}


def clustered(differences: Sequence[tuple[Any, float]]) -> tuple[float, float | None]:
    """The mean of sentence differences ``(cluster, value)`` and its cluster-robust standard error (None under two
    clusters): each cluster's sum against its sentences times the mean."""
    sums: dict[Any, float] = defaultdict(float)
    counts: dict[Any, int] = defaultdict(int)
    for cluster, value in differences:
        sums[cluster] += value
        counts[cluster] += 1
    n = sum(counts.values())
    mean = sum(sums.values()) / n
    k = len(sums)
    if k < 2:
        return mean, None
    spread = sum((sums[c] - mean * counts[c]) ** 2 for c in sums)
    return mean, math.sqrt(k / (k - 1) * spread) / n


def compare(rows: Mapping[tuple, tuple[dict, dict]], clusters: Mapping[int, tuple[str, str]]) -> dict[str, Any]:
    """Each measure in both readouts and second − first over the sentences of ``rows`` counted in both (one source, one
    group), with how many start in a loss in each."""
    pairs = {k: p for k, p in rows.items() if not (p[0]["starts_in_a_loss"] or p[1]["starts_in_a_loss"])}
    if not pairs:
        raise ValueError("a group whose every sentence starts in a loss in one readout or the other")
    out: dict[str, Any] = {
        "sentences": len(pairs), "aircraft": len({(k[0], k[1]) for k in pairs}),
        "clusters": len({clusters[k[0]] for k in pairs}),
        "starting_in_a_loss": {"first": sum(a["starts_in_a_loss"] for a, _ in rows.values()),
                               "second": sum(b["starts_in_a_loss"] for _, b in rows.values()),
                               "both": sum(a["starts_in_a_loss"] and b["starts_in_a_loss"] for a, b in rows.values())}}
    for name, value in MEASURES.items():
        first = [value(a) for a, _ in pairs.values()]
        second = [value(b) for _, b in pairs.values()]
        mean, error = clustered([(clusters[k[0]], value(b) - value(a)) for k, (a, b) in pairs.items()])
        out[name] = {"first": float(np.mean(first)), "second": float(np.mean(second)), "difference": mean,
                     "standard_error": error}
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
    clusters = cluster_of(pairs)
    augmented = first[0]["augment_seed"] is not None
    report: dict[str, Any] = {}
    for source in MODEL_SOURCES:
        mine = {k: p for k, p in pairs.items() if k[3] == source}
        if mine:
            report[source] = {group: {name: compare(part, clusters) for name, part in parts.items()}
                              for group, parts in groups(mine, augmented).items()}
    return {"schema": SCHEMA, "written_utc": utc_now(),
            "first": {"directory": repo_relative(first_dir), "prior": first[0]["prior"]},
            "second": {"directory": repo_relative(second_dir), "prior": second[0]["prior"]},
            "same": {name: first[0][name] for name in ("split", "windows_per_airport", "samples", "seed",
                                                        "augment_seed")},
            "measures": list(MEASURES), "report": report}


def _line(name: str, entry: Mapping[str, Any]) -> str:
    def error(value: float | None, scale: float, digits: int) -> str:
        return "n/a" if value is None else f"{scale * value:.{digits}f}"

    def cell(m: str, scale: float, unit: str) -> str:
        e = entry[m]
        return (f"{m} {scale * e['first']:.2f}{unit} → {scale * e['second']:.2f}{unit} "
                f"({scale * e['difference']:+.2f} ± {error(e['standard_error'], scale, 2)})")
    reward, loss = entry["reward"], entry["starting_in_a_loss"]
    return (f"  {name:10s} {entry['aircraft']:5d} aircraft {entry['sentences']:6d} sentences {entry['clusters']:3d} days "
            f"(start in a loss {loss['first']} / {loss['second']}) | reward "
            f"{reward['first']:.4f} → {reward['second']:.4f} ({reward['difference']:+.4f} ± "
            f"{error(reward['standard_error'], 1, 4)}; per sentence ± {reward['per_sentence']['standard_error']:.4f}) | "
            f"{cell('lost_separation', 100, ' %')} "
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
        write_json_atomic(out / "traffic_window_pair.json", result, allow_nan=False)
        print(f"→ {out / 'traffic_window_pair.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
