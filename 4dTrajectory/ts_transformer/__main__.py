"""CLI: train a trajectory predictor, or predict a batch of approaches with a trained one.

The default ``--prediction-output state`` remains the original purely kinematic baseline.
The opt-in ``--prediction-output control`` predicts bounded per-flight controls and rolls
them through a differentiable Torch twin of ``CasadiSimulator``; it currently uses the
normalized complete-remainder horizon. The output strategy is checkpointed and cannot be
changed at inference without retraining.

    TS=4dTrajectory/ts_transformer/__main__.py

    # train iTransformer on N normalized progress segments plus final_time_s
    python $TS train \
        --data trajectory_data_process/outputs/harvest/KRDU/arrivals/manifest.json \
        --airport KRDU --model itransformer --n-segments 128 \
        --output-dir 4dTrajectory/outputs/KRDU/ts_itransformer_normalized_time

    # the same normalized-time target with PatchTST
    python $TS train --data ... --model patchtst --n-segments 128 ...

    # development prediction (validation is the safe default)
    python $TS predict \
        --checkpoint 4dTrajectory/outputs/KRDU/ts_itransformer_normalized_time/checkpoint.pt \
        --data ... --airport KRDU --output-dir 4dTrajectory/outputs/KRDU/ts_pred

    # outer-test is a separate, irreversible release after every decision is frozen
    python $TS freeze-test --checkpoint ... --data ...
    python $TS predict --checkpoint ... --data ... --output-dir ... \
        --split test --test-release
    python -m evaluation --input 4dTrajectory/outputs/KRDU/ts_pred

**Every flag that sets a ``TSConfig`` field is named after that field** (``--dt-s`` for
``dt_s``, ``--control-rollout-integrator-dt-s`` for
``control_rollout_integrator_dt_s``, …). ``cli/common.py`` asserts it at import, and every
parser here passes ``allow_abbrev=False`` so an old spelling is refused rather than
prefix-matched. The exceptions are declared where they occur: ``--batch-size`` (also takes
``"auto"``), ``--instance-norm`` (one flag, two backbone fields), ``--control-recipe-name``
(resolved first), and on ``predict`` — whose config comes from the checkpoint, so its flags
are OVERRIDES — ``--command-hook`` / ``--hook-saturation``, kept short because they are the
adopted delivery form and two arm files spell them.

One module per subcommand under ``cli/``; this file is the parser table and the bootstrap.

Invoked by script path, like ``4dTrajectory/optimization/scenario_optimization.py`` — the
bootstrap below makes it work from any working directory. (``python -m ts_transformer``
also works, but only from inside ``4dTrajectory/``, so the path form is what the docs use.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

# Same bootstrap as 4dTrajectory/optimization/*.py: the repo root for the shared packages
# (flight_scenarios, aerodynamic_model, geokit, evaluation), and this directory so sibling
# modules import flat.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TS_DIR = Path(__file__).resolve().parent
if str(_TS_DIR) not in sys.path:
    sys.path.insert(0, str(_TS_DIR))

import approach_clustering.cli as approach_cohorts_cli  # noqa: E402
import batch_benchmark  # noqa: E402
from cli import (  # noqa: E402
    cross_validate as cross_validate_cli,
    evaluate_fit as evaluate_fit_cli,
    freeze_test as freeze_test_cli,
    predict as predict_cli,
    train as train_cli,
)

AddArguments = Callable[[argparse.ArgumentParser], None]
RunCommand = Callable[[argparse.Namespace, argparse.ArgumentParser, "list[str] | None"], int]


def _drop_argv(run: Callable[[argparse.Namespace, argparse.ArgumentParser], int]) -> RunCommand:
    """Adapt a two-argument ``run_cli``: only train/cross-validate record their argv."""

    def run_cli(
        args: argparse.Namespace, parser: argparse.ArgumentParser, argv: list[str] | None
    ) -> int:
        del argv
        return run(args, parser)

    return run_cli


#: command name -> (help, add_cli_arguments, run_cli). Order is the order ``--help`` lists.
COMMANDS: dict[str, tuple[str, AddArguments, RunCommand]] = {
    "train": (train_cli.HELP, train_cli.add_cli_arguments, train_cli.run_cli),
    "cross-validate": (
        cross_validate_cli.HELP,
        cross_validate_cli.add_cli_arguments,
        cross_validate_cli.run_cli,
    ),
    "approach-cohorts": (
        "build approach clusters or compare checkpoints on a frozen cohort",
        approach_cohorts_cli.add_cli_arguments,
        _drop_argv(approach_cohorts_cli.run_cli),
    ),
    "benchmark-batch": (
        "benchmark outer-train CUDA batch throughput",
        batch_benchmark.add_cli_arguments,
        _drop_argv(batch_benchmark.run_cli),
    ),
    "evaluate-fit": (
        evaluate_fit_cli.HELP,
        evaluate_fit_cli.add_cli_arguments,
        evaluate_fit_cli.run_cli,
    ),
    "freeze-test": (
        freeze_test_cli.HELP,
        freeze_test_cli.add_cli_arguments,
        freeze_test_cli.run_cli,
    ),
    "predict": (predict_cli.HELP, predict_cli.add_cli_arguments, predict_cli.run_cli),
}


def main(argv: list[str] | None = None) -> int:
    # allow_abbrev=False everywhere, top parser and every subparser: with it on, argparse
    # accepts any unambiguous PREFIX, so the fifteen flags renamed in T3-19 kept working
    # under their old spellings (`--dt` is a prefix of `--dt-s`) and a stale command line
    # would have set the field it looks like it sets while reading as up to date.
    parser = argparse.ArgumentParser(
        prog="ts_transformer",
        description="Learned 4D trajectory prediction (iTransformer / PatchTST)",
        allow_abbrev=False,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, add_cli_arguments, _run) in COMMANDS.items():
        add_cli_arguments(sub.add_parser(name, help=help_text, allow_abbrev=False))

    args = parser.parse_args(argv)
    return COMMANDS[args.command][2](args, parser, argv)


if __name__ == "__main__":
    raise SystemExit(main())
