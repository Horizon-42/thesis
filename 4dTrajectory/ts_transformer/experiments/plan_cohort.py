"""The development cohort a random-anchor run trains on — the locked by-flight split over the given rosters minus the train flights the random-anchor future contract does not cover.

The train CLI refuses a random-anchor run whose contract covers fewer train flights than the
split holds ("adjust the train roster explicitly instead of silently changing experiment
membership"). The cohort `step3c_plan_head_full_cohort.json` was written by exactly this rule by
hand on 2026-09-11; pooling five airports needs it again, so it is a runner: the same data,
config and roster flags as `train`, the same series build, the same split and training-cohort
filter, the same window set — every flight the window set gives no admissible anchor is dropped
from train, and the selection names them with the airports, the seed and the contract.

    python run_ts.py plan_cohort --data <harvest/ICAO> ... --eligibility-roster <json> ... \\
        --prediction-output control --config-overrides <json> --name NAME --output-dir <dir>

The cohort is ``<output-dir>/development_cohort.json`` (`--output-dir` is the shared training
flag), beside it ``data_selection.json`` — the build report with the REASON every flight of the
locked split is not in the cohort (a later `train` loads only the cohort's keys, so its own
report rejects nothing; the cohort file names the excluded flights, this file says why). The
selection records `dropped_train` (the contract), `excluded_train` (the fixed-anchor cohort
floor, `training_cohort_min_future_s`, empty at the default 0) and `not_usable` — the locked
split's flights on BOTH sides the series build did not produce (the aircraft filter, an
unusable track, or a record too short for one window: the lookback and, under a fixed
horizon, the segment after it); the hand-written 2026-09-11 cohort recorded the train side of
that list only.

    python run_ts.py plan_cohort --data … --eligibility-roster … --airport KRDU \\
        --arms docs/experiments/<declaration>.json --name NAME --output-dir <dir>

``--arms`` (two-tier v3, 2026-09-18) writes the cohorts an ARM DECLARATION names instead: one per
distinct per-arm ``"development_cohort"`` path (``{airport}`` substituted), each from that arm's
own config — a grid whose cells keep different flights (the (L, Δ) grid: a cell's usable set is
the flights long enough for its lookback AND its segment). The declaration is ONE recipe on the
grid's axes (`GRID_AXES`: the lookback, the segment and what follows from it, the seed): every
arm must agree on every other field, so one load serves every cell — under the config with the
shortest lookback, whose build keeps every flight any cell's build keeps (the build's own length
floor is the lookback plus one row, `dataset.minimum_build_samples`, applied per cell here too).
The split seed must be pinned (the cohort is shared across seeds), and no config flag may be
given (every cell's config is the declaration's). ``--output-dir`` then holds
``data_selection.json`` and ``cohorts.json`` (every cell's counts).
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

from ts_transformer.cli.common import (
    CLI_CONFIG_FIELDS, TrainingRun, add_training_run_arguments, config_from_args, prepare_training_run,
)
from ts_transformer.config import TSConfig
from ts_transformer.data.dataset import FlightSeries, Normalizer, minimum_build_samples, training_window_class
from ts_transformer.data.development_cohorts import DevelopmentCohort, write_development_cohort
from ts_transformer.data.splits import split_by_flight
from ts_transformer.experiments.support import REPO_ROOT, arm_config, declaration_base
from ts_transformer.io_utils import write_json_atomic
from ts_transformer.outputs import strategy
from ts_transformer.training.train import filter_training_cohort, usable_series

#: The one file this runner writes under `--output-dir` (or at every declared path under --arms).
COHORT_FILE = "development_cohort.json"
#: Under --arms: every cell's counts, beside `data_selection.json`.
CELLS_FILE = "cohorts.json"
#: The fields a grid declaration may vary between its arms: the lookback, the segment and what
#: follows from it, and the seed. Every other field is the one recipe every cell shares — the
#: series are built and split once, under one contract.
GRID_AXES = ("seq_len", "control_horizon_s", "n_segments", "random_train_anchor_min_future_s", "seed")


def cohort_selection(series: Sequence[FlightSeries], outer_split_keys: dict[str, Sequence[str]],
                     config: TSConfig) -> tuple[list[str], list[str], dict[str, Any]]:
    """``(train keys, val keys, selection)`` for ``config`` over the locked split's built
    ``series``: the flights usable under the config (one window at L−1 with the horizon of truth
    after it), split by flight, the train side minus the flights the random-anchor window set
    gives no admissible anchor — the flight `prepare_session` would refuse the run over."""
    # the window rule AND the build's own length gate: a series built under a shorter lookback
    # may hold a flight this config's build would drop (`minimum_build_samples`)
    floor = minimum_build_samples(config)
    usable = [item for item in usable_series(series, config, verbose=False) if item.n_samples >= floor]
    train_series, val_series, _test = split_by_flight(usable, config)
    train_series, cohort_filter = filter_training_cohort(train_series, config, verbose=False)
    normalizer = Normalizer.fit(train_series, balance_airports_and_flights=True)
    # the window set training would build: a flight with no admissible anchor is the one
    # `prepare_session` refuses the run over
    train_set = training_window_class(config)(
        train_series, config, normalizer, minimum_anchor_index=None,
        training_input=strategy(config).training_input(config),
    )
    dropped = sorted(
        item.dataset_id for index, item in enumerate(train_series) if train_set.series_ranges[index][1] == 0
    )
    dropped_set = set(dropped)
    keep = [item.dataset_id for item in train_series if item.dataset_id not in dropped_set]
    val = [item.dataset_id for item in val_series]
    locked = set(outer_split_keys["train"]) | set(outer_split_keys["val"])
    not_usable = sorted(locked - {item.dataset_id for item in usable})
    airports = sorted({item.airport for item in usable})
    excluded_train = sorted(cohort_filter.get("excluded", []))
    lookback_s = config.seq_len * config.dt_s
    selection = {
        "rule": (
            f"the locked split (split_seed {config.split_seed}, {config.aircraft_filter}) over "
            f"{', '.join(airports)}, the flights with one window at L-1 ({lookback_s:g} s of lookback"
            + (f" and {config.control_horizon_s:g} s of truth after it" if config.control_horizon_s else "")
            + f"), minus the train flights the {config.random_train_anchor_min_future_s:g} s "
            f"random-anchor future contract ({config.random_train_anchor_sampling}) does not cover"
            + (f", and the train flights under the {config.training_cohort_min_future_s:g} s cohort floor"
               if excluded_train else "")
        ),
        "airports": airports, "split_seed": config.split_seed, "aircraft_filter": config.aircraft_filter,
        "lookback_s": lookback_s, "control_horizon_s": config.control_horizon_s,
        "random_train_anchor_min_future_s": config.random_train_anchor_min_future_s,
        "random_train_anchor_sampling": config.random_train_anchor_sampling,
        "training_cohort_min_future_s": config.training_cohort_min_future_s,
        "dropped_train": dropped, "excluded_train": excluded_train,
        "not_usable": not_usable,        # both sides of the locked split (see the module docstring)
    }
    return keep, val, selection


def check_cohort_config(config: TSConfig, parser: argparse.ArgumentParser) -> None:
    """The config's own refusals BEFORE the 36k-track load: the contract the rule reads, a
    config training would refuse, and a rolled table — which cannot change an anchor count and
    would only demand coverage of the cohort it precedes."""
    if not config.random_train_anchor:
        parser.error("the cohort rule is the random-anchor future contract; the config has random_train_anchor=False")
    strategy(config).check_trainable(config)


def declared_cohorts(declaration: dict[str, Any], airport: str) -> dict[Path, TSConfig]:
    """``{cohort path: the config it is written from}`` for every distinct per-arm
    ``development_cohort`` of ``declaration``. Arms sharing a path must agree on every field
    but ``seed``; every arm must agree on every field outside `GRID_AXES`; the split seed must
    be pinned (the cohort is shared across seeds); two paths may not share a directory name
    (`cohorts.json` is keyed by it)."""
    base = declaration_base(declaration)
    cohorts: dict[Path, tuple[str, TSConfig, dict[str, Any]]] = {}
    recipe: tuple[str, dict[str, Any]] | None = None
    for arm in declaration["arms"]:
        if "checkpoint" in arm:
            continue
        declared = arm.get("development_cohort")
        if not declared:
            raise ValueError(f"arm {arm['key']} declares no development_cohort; --arms writes the paths the arms name")
        path = REPO_ROOT / str(declared).format(airport=airport)
        config, settings = arm_config(base, arm.get("overrides", {}))
        if config.split_seed is None:
            raise ValueError(f"arm {arm['key']}: split_seed must be pinned in the declaration — the cohort is shared "
                             "by arms of different seeds, and an unpinned split follows the model seed")
        shared = {key: value for key, value in settings.items() if key not in GRID_AXES}
        if recipe is None:
            recipe = (arm["key"], shared)
        elif shared != recipe[1]:
            differing = sorted(key for key in set(shared) | set(recipe[1]) if shared.get(key) != recipe[1].get(key))
            raise ValueError(f"arms {recipe[0]} and {arm['key']} differ on {', '.join(differing)}: a grid declaration varies "
                             f"only {', '.join(GRID_AXES)}, and the series are built and split once")
        identity = {key: value for key, value in settings.items() if key != "seed"}
        if path in cohorts and cohorts[path][2] != identity:
            raise ValueError(f"arms {cohorts[path][0]} and {arm['key']} share {path} but differ beyond the seed")
        cohorts.setdefault(path, (arm["key"], config, identity))
    if not cohorts:
        raise ValueError("the declaration has no training arm")
    leaves = [path.parent.name for path in cohorts]
    if len(set(leaves)) != len(leaves):
        raise ValueError("two cohort paths share a directory name; `cohorts.json` keys cells by it")
    return {path: config for path, (_key, config, _identity) in cohorts.items()}


def write_cohort(out: Path, name: str, run: TrainingRun, config: TSConfig) -> DevelopmentCohort:
    keep, val, selection = cohort_selection(run.series, run.outer_split_keys, config)
    cohort = write_development_cohort(out, name=name, train_flight_ids=keep, val_flight_ids=val, selection=selection)
    print(
        f"wrote {out}: {len(keep)} train + {len(val)} val flights; dropped {len(selection['dropped_train'])} train "
        f"flight(s) the contract does not cover; {len(selection['not_usable'])} of the locked split not usable", flush=True,
    )
    return cohort


def explicit_config_flags(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    """The config flags given a value other than their default -- what ``--arms`` refuses.

    Compared with the value an absent flag parses to, not with None: ``--model`` defaults to
    the first model, so an ``is not None`` test refused every ``--arms`` run whatever was
    passed. That value is the FIRST action's default for a dest, as ``parse_args`` sets it --
    not ``parser.get_default``, which skips a None default and so reads ``--no-instance-norm``'s
    True where an absent ``--instance-norm`` parses to None."""
    absent: dict[str, object] = {}
    for action in parser._actions:
        absent.setdefault(action.dest, action.default)
    return [
        name for name in ("config_overrides", "batch_size", "instance_norm", *CLI_CONFIG_FIELDS)
        if getattr(args, name) != absent[name]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    add_training_run_arguments(parser, cohort_help="not accepted here: this runner WRITES the cohort")
    parser.add_argument("--name", required=True, help="the cohort's name (recorded in every checkpoint trained on it; "
                                                      "under --arms the cell directory's name is appended)")
    parser.add_argument("--arms", type=Path, default=None,
                        help="an arm declaration: write every distinct per-arm development_cohort it names")
    args = parser.parse_args(argv)
    if args.development_cohort:
        parser.error("--development-cohort is what this runner writes; do not give one")
    if args.campaign_id or args.experiment_id:
        parser.error("a cohort is data selection, not an experiment arm: no --campaign-id / --experiment-id")
    if args.arms is None:
        config, _batch_auto = config_from_args(args, parser)
        check_cohort_config(config, parser)
        run = prepare_training_run(args, parser, argv)
        cohort = write_cohort(Path(args.output_dir) / COHORT_FILE, args.name, run, run.config)
        write_json_atomic(Path(args.output_dir) / "data_selection.json", run.data_selection)
        _print_per_airport(cohort)
        return 0

    given = explicit_config_flags(args, parser)
    if given:
        parser.error(f"--arms takes every cell's config from the declaration; drop --{given[0].replace('_', '-')}")
    if not args.airport:
        parser.error("--arms needs --airport (the declaration's {airport})")
    declaration = json.loads(args.arms.read_text(encoding="utf-8"))
    try:
        cohorts = declared_cohorts(declaration, args.airport.upper())
    except ValueError as exc:
        parser.error(str(exc))
    for config in cohorts.values():
        check_cohort_config(config, parser)
    # one load, under the shortest lookback: that build keeps every flight a longer lookback's
    # build keeps, and `cohort_selection` applies each cell's own lookback, segment and length gate
    build_config = min(cohorts.values(), key=lambda config: config.seq_len)
    run = prepare_training_run(args, parser, argv, config=build_config)
    cells: dict[str, dict[str, Any]] = {}
    for path, config in cohorts.items():
        cohort = write_cohort(path, f"{args.name}/{path.parent.name}", run, config)
        cells[path.parent.name] = {
            "cohort": str(path), "lookback_s": config.seq_len * config.dt_s, "control_horizon_s": config.control_horizon_s,
            "train": len(cohort.train_flight_ids), "val": len(cohort.val_flight_ids),
            "dropped_train": len(cohort.selection["dropped_train"]), "not_usable": len(cohort.selection["not_usable"]),
        }
    write_json_atomic(Path(args.output_dir) / "data_selection.json", run.data_selection)
    write_json_atomic(Path(args.output_dir) / CELLS_FILE, {"name": args.name, "arms": str(args.arms), "cells": cells})
    print(f"{'cell':<12}{'L s':>6}{'Δ s':>6}{'train':>7}{'val':>6}{'dropped':>9}{'not usable':>12}")
    for cell, row in cells.items():
        print(f"{cell:<12}{row['lookback_s']:>6g}{row['control_horizon_s']:>6g}{row['train']:>7}{row['val']:>6}"
              f"{row['dropped_train']:>9}{row['not_usable']:>12}")
    return 0


def _print_per_airport(cohort: DevelopmentCohort) -> None:
    per_airport = Counter(item.split(":")[0] for item in cohort.train_flight_ids)
    per_airport_val = Counter(item.split(":")[0] for item in cohort.val_flight_ids)
    for airport in sorted(per_airport | per_airport_val):
        print(f"  {airport}: train {per_airport[airport]}, val {per_airport_val[airport]}")


if __name__ == "__main__":
    raise SystemExit(main())
