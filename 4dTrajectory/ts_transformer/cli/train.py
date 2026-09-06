"""``train``: fit one predictor on the development cohort and write a checkpoint.

The outer-test source tracks are never opened here: the split is resolved from manifest
metadata, and only train + validation flights are loaded.
"""

from __future__ import annotations

import argparse

from train import train as run_training

from .common import add_training_run_arguments, prepare_training_run

HELP = "train a predictor and write a checkpoint"


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    add_training_run_arguments(
        parser,
        cohort_help=(
            "explicit train/validation flight roster; the resulting checkpoint is "
            "development-only and cannot release outer-test"
        ),
    )


def run_cli(
    args: argparse.Namespace, parser: argparse.ArgumentParser, argv: list[str] | None
) -> int:
    run = prepare_training_run(args, parser, argv)
    try:
        run_training(
            run.series,
            run.config,
            output_dir=args.output_dir,
            data_provenance=run.data_provenance,
            reserved_test_keys=(
                [] if run.development_cohort is not None
                else run.outer_split_keys["test"]
            ),
            data_selection=run.data_selection,
            auto_batch_size=run.batch_auto,
        )
    except Exception as exc:
        run.finish(exc)
        raise
    run.finish()
    return 0
