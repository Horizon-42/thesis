"""D176 (2): a window list from per-window readouts by one rule (post-training C25) — the select windows whose outcome
is one of ``--outcomes`` in any of the reads given, or with ``--all`` in every one.

THE READS: ``--diagnose DIR READ`` (a D175 readout, `post_diagnose`: its lines of read ``on`` or ``off`` in
``per_window.jsonl``) and ``--ceiling DIR MODEL`` (a D161 readout, `post_ceiling`: draw 0 of ``model_<MODEL>.json``, the
readout's own numbers), each any number of times, at least one in all. Every read must index the same selection
windows — the same windows in the same places (flight, start time, kind) under the same select seed and windows an
airport — refused by name otherwise. A diagnose readout's selection is its ``config.json``'s settings; a ceiling's is
its campaign's (``campaign.json`` of the campaign its ``config.json`` names, as this checkout reads that path,
`training_export.this_checkout`: a ceiling may predate the select seed as a setting). A smoke readout is refused, and
an outcome that no read gives (a mistyped name would choose nothing).

WRITES, into a new directory, ``list.json`` (`post.window_lists`, ``ts-window-list-v1``, stage C: each window's place,
airport and identity; its information fields, which no reader acts on, are each read's outcome and, for a diagnose read
that lost separation, D175's fields) and ``SHA256SUMS``; the directory read-only. The readouts are named by path and
sha256 (the files read). Stage D's readouts join once they write a line per window (D176 (2)).

    python run_ts.py window_list --diagnose 4dTrajectory/outputs/POOLED/post/<diagnose id> on \\
        --ceiling 4dTrajectory/outputs/POOLED/post/<ceiling id> start --outcomes lost_separation \\
        --out 4dTrajectory/outputs/POOLED/post/<list id> [--all]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ts_transformer.experiments.post_ceiling import CEILING_SCHEMA
from ts_transformer.experiments.post_diagnose import DIAGNOSE_SCHEMA, OFF, ON, seal
from ts_transformer.experiments.training_export import this_checkout
from ts_transformer.io_utils import file_sha256
from ts_transformer.post.window_lists import SPLIT, ListedWindow, WindowList, write_window_list
from ts_transformer.repo_layout import REPO_ROOT, repo_relative

#: The stage whose readouts this runner reads (D176 (2): stage D's join once they write a line per window).
STAGE = "C"


@dataclass(frozen=True)
class Read:
    """One read of the select windows: its name, the selection it indexes (select seed, windows an airport), the
    windows' identities in place order, each window's outcome and fields, and the file read."""

    name: str
    selection: Mapping[str, int]
    identities: tuple[tuple[str, float, str], ...]
    outcomes: tuple[str, ...]
    fields: Mapping[int, Mapping[str, Any]]
    path: Path


def _identities(config: Mapping[str, Any]) -> tuple[tuple[str, float, str], ...]:
    return tuple((w["flight"], w["row0_s"], w["kind"]) for w in config["windows"])


def diagnose_read(directory: Path, read: str) -> Read:
    """Read ``read`` (on or off) of the D175 readout ``directory``."""
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config["schema"] != DIAGNOSE_SCHEMA:
        raise SystemExit(f"{directory} is a {config['schema']} readout, not {DIAGNOSE_SCHEMA}")
    if config["smoke"]:
        raise SystemExit(f"{directory} is a smoke readout: no list is written from it")
    if read not in (ON, OFF):
        raise SystemExit(f"a diagnose readout's reads are {ON} and {OFF}, not {read!r}")
    path = directory / "per_window.jsonl"
    lines = [line for line in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if line["read"] == read]
    identities = _identities(config)
    if [line["place"] for line in lines] != list(range(len(identities))):
        raise SystemExit(f"{path}: the lines of read {read} are not one a window, in place order")
    settings = config["settings"]
    return Read(f"{directory.name}:{read}", {"select_seed": settings["select_seed"],
                                             "per_airport": settings["select_per_airport"]},
                identities, tuple(line["outcome"] for line in lines),
                {line["place"]: line["fields"] for line in lines if "fields" in line}, path)


def ceiling_read(directory: Path, model: str) -> Read:
    """Draw 0 of model ``model`` in the D161 readout ``directory`` (the readout's own numbers)."""
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config["schema"] != CEILING_SCHEMA:
        raise SystemExit(f"{directory} is a {config['schema']} readout, not {CEILING_SCHEMA}")
    if config["smoke"]:
        raise SystemExit(f"{directory} is a smoke readout: no list is written from it")
    if model not in config["models"]:
        raise SystemExit(f"{directory} read the models {config['models']}, not {model!r}")
    campaign = json.loads((this_checkout(config["campaign"]) / "campaign.json").read_text(encoding="utf-8"))
    settings = campaign["inputs"]["settings"]
    if settings["select_per_airport"] != config["settings"]["select_per_airport"]:
        raise SystemExit(f"{directory}: its campaign's select windows an airport differ from the ceiling's record")
    path = directory / f"model_{model}.json"
    held = json.loads(path.read_text(encoding="utf-8"))
    identities = _identities(config)
    if len(held["outcomes"]) != len(identities):
        raise SystemExit(f"{path}: {len(held['outcomes'])} windows' draws for {len(identities)} windows")
    return Read(f"{directory.name}:{model}", {"select_seed": settings["select_seed"],
                                              "per_airport": settings["select_per_airport"]},
                identities, tuple(row[0] for row in held["outcomes"]), {}, path)


def chosen(reads: Sequence[Read], outcomes: Sequence[str], every: bool) -> list[int]:
    """The places whose outcome is one of ``outcomes`` in any of ``reads`` (in every one with ``every``); refused by
    name unless every read indexes the same selection windows."""
    first = reads[0]
    for read in reads[1:]:
        if read.selection != first.selection or read.identities != first.identities:
            raise SystemExit(f"{read.name} indexes other selection windows than {first.name}")
    given = {outcome for read in reads for outcome in read.outcomes}
    if not set(outcomes) <= given:
        raise SystemExit(f"--outcomes {sorted(set(outcomes) - given)}: no read gives it (the reads give {sorted(given)})")
    hits = [[read.outcomes[place] in outcomes for read in reads] for place in range(len(first.identities))]
    return [place for place, hit in enumerate(hits) if (all(hit) if every else any(hit))]


def window_list(reads: Sequence[Read], outcomes: Sequence[str], every: bool) -> WindowList:
    """The list of the chosen windows (`chosen`), each with its reads' outcomes and fields as information."""
    first = reads[0]
    windows = []
    for place in chosen(reads, outcomes, every):
        flight, row0_s, kind = first.identities[place]
        info: dict[str, Any] = {"outcomes": {read.name: read.outcomes[place] for read in reads}}
        fields = {read.name: read.fields[place] for read in reads if place in read.fields}
        if fields:
            info["fields"] = fields
        windows.append(ListedWindow(place, flight.partition(":")[0], {"flight": flight, "row0_s": row0_s, "kind": kind},
                                    info))
    rule = f"whose outcome is one of {list(outcomes)} in {'every one' if every else 'any'} of the reads " \
           f"{[read.name for read in reads]}"
    return WindowList(STAGE, SPLIT, dict(first.selection), f"the {SPLIT} windows {rule}",
                      tuple({"path": repo_relative(this_checkout(read.path)), "sha256": file_sha256(read.path)}
                            for read in reads), tuple(windows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--diagnose", nargs=2, action="append", default=[], metavar=("DIR", "READ"),
                        help="a D175 readout and its read (on, off)")
    parser.add_argument("--ceiling", nargs=2, action="append", default=[], metavar=("DIR", "MODEL"),
                        help="a D161 readout and a model it read (draw 0)")
    parser.add_argument("--outcomes", nargs="+", required=True, help="the outcomes that choose a window")
    parser.add_argument("--all", action="store_true", help="chosen in every read (default: in any)")
    parser.add_argument("--out", type=Path, required=True, help="a new directory")
    args = parser.parse_args(argv)

    def here(path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else REPO_ROOT / p

    if not args.diagnose and not args.ceiling:
        parser.error("give at least one read (--diagnose or --ceiling)")
    out = here(str(args.out))
    if out.exists():
        parser.error(f"{out} exists: the list is written into a new directory")
    reads = [diagnose_read(here(d), r) for d, r in args.diagnose] + [ceiling_read(here(d), m) for d, m in args.ceiling]
    if len({read.name for read in reads}) != len(reads):
        parser.error("each read once")
    listed = window_list(reads, args.outcomes, args.all)
    out.mkdir(parents=True)
    payload = write_window_list(out / "list.json", listed)
    seal(out)
    print(json.dumps({"windows": payload["count"], "by_airport": payload["by_airport"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
