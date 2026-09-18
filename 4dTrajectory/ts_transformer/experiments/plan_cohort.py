"""Plan-and-guidance step 5: the development cohort a plan run trains on — the locked by-flight split over the given rosters minus the train flights the random-anchor future contract does not cover.

The train CLI refuses a random-anchor run whose contract covers fewer train flights than the
split holds ("adjust the train roster explicitly instead of silently changing experiment
membership"). The cohort `step3c_plan_head_full_cohort.json` was written by exactly this rule by
hand on 2026-09-11; pooling five airports needs it again, so it is a runner: the same data,
config and roster flags as `train`, the same series build, the same split and training-cohort
filter, the same window set — every flight the window set gives no admissible anchor is dropped
from train, and the selection names them with the airports, the seed and the contract.

    python run_ts.py plan_cohort --data <harvest/ICAO> ... --eligibility-roster <json> ... \\
        --prediction-output plan --config-overrides <json> --name NAME --output-dir <dir>

The cohort is ``<output-dir>/development_cohort.json`` (`--output-dir` is the shared training
flag), beside it ``data_selection.json`` — the build report with the REASON every flight of the
locked split is not in the cohort (a later `train` loads only the cohort's keys, so its own
report rejects nothing; the cohort file names the excluded flights, this file says why). The
selection records `dropped_train` (the contract), `excluded_train` (the fixed-anchor cohort
floor, `training_cohort_min_future_s`, empty at the default 0) and `not_usable` — the locked
split's flights on BOTH sides the series build did not produce (the aircraft filter, an
unusable track); the hand-written 2026-09-11 cohort recorded the train side of that list only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ts_transformer.cli.common import add_training_run_arguments, config_from_args, prepare_training_run
from ts_transformer.data.dataset import Normalizer, training_window_class
from ts_transformer.io_utils import write_json_atomic
from ts_transformer.data.development_cohorts import write_development_cohort
from ts_transformer.data.splits import split_by_flight
from ts_transformer.outputs import strategy
from ts_transformer.training.train import filter_training_cohort, usable_series

#: The one file this runner writes under `--output-dir`.
COHORT_FILE = "development_cohort.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    add_training_run_arguments(parser, cohort_help="not accepted here: this runner WRITES the cohort")
    parser.add_argument("--name", required=True, help="the cohort's name (recorded in every checkpoint trained on it)")
    args = parser.parse_args(argv)
    out = Path(args.output_dir) / COHORT_FILE
    if args.development_cohort:
        parser.error("--development-cohort is what this runner writes; do not give one")
    if args.campaign_id or args.experiment_id:
        parser.error("a cohort is data selection, not an experiment arm: no --campaign-id / --experiment-id")
    # the config's own refusals BEFORE the 36k-track load: the contract the rule reads, a
    # config training would refuse, and a rolled table — which cannot change an anchor count
    # and would only demand coverage of the cohort it precedes
    config, _batch_auto = config_from_args(args, parser)
    if not config.random_train_anchor:
        parser.error("the cohort rule is the random-anchor future contract; the config has random_train_anchor=False")
    strategy(config).check_trainable(config)
    run = prepare_training_run(args, parser, argv)
    config = run.config
    series = usable_series(run.series, config, verbose=False)
    train_series, val_series, _test = split_by_flight(series, config)
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
    locked = set(run.outer_split_keys["train"] + run.outer_split_keys["val"])
    not_usable = sorted(locked - {item.dataset_id for item in series})
    airports = sorted({item.airport for item in series})
    excluded_train = sorted(cohort_filter.get("excluded", []))
    cohort = write_development_cohort(
        out, name=args.name, train_flight_ids=keep, val_flight_ids=val,
        selection={
            "rule": (
                f"the locked split (split_seed {config.split_seed}, {config.aircraft_filter}) over "
                f"{', '.join(airports)} minus the train flights the {config.random_train_anchor_min_future_s:g} s "
                f"random-anchor future contract ({config.random_train_anchor_sampling}) does not cover"
                + (f", and the train flights under the {config.training_cohort_min_future_s:g} s cohort floor"
                   if excluded_train else "")
            ),
            "airports": airports, "split_seed": config.split_seed, "aircraft_filter": config.aircraft_filter,
            "random_train_anchor_min_future_s": config.random_train_anchor_min_future_s,
            "random_train_anchor_sampling": config.random_train_anchor_sampling,
            "training_cohort_min_future_s": config.training_cohort_min_future_s,
            "dropped_train": dropped, "excluded_train": excluded_train,
            "not_usable": not_usable,        # both sides of the locked split (see the module docstring)
        },
    )
    write_json_atomic(out.with_name("data_selection.json"), run.data_selection)
    per_airport = Counter(item.split(":")[0] for item in cohort.train_flight_ids)
    per_airport_val = Counter(item.split(":")[0] for item in cohort.val_flight_ids)
    for airport in airports:
        print(f"  {airport}: train {per_airport[airport]}, val {per_airport_val[airport]}")
    print(
        f"wrote {out}: {len(cohort.train_flight_ids)} train + {len(cohort.val_flight_ids)} val flights; "
        f"dropped {len(dropped)} train flight(s) the contract does not cover; {len(not_usable)} of the locked "
        "split not usable", flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
