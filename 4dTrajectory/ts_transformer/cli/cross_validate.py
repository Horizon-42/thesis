"""``cross-validate``: select hyperparameters on outer-train folds only.

Same cohort assembly as ``train`` (`common.prepare_training_run`); what differs is that the
folds are cut inside the outer-train split, so neither outer-validation nor outer-test ever
reaches a fitted model.
"""

from __future__ import annotations

import argparse
import json

from ts_transformer.cross_validation import (
    CV_PARAMETER_GRIDS,
    DEFAULT_CV_EPOCHS,
    DEFAULT_CV_PARAMETERS,
    DEFAULT_CV_PATIENCE,
    cross_validate,
)

from .common import (
    add_training_run_arguments,
    cv_parameters_type,
    prepare_training_run,
)

HELP = "select hyperparameters using outer-train folds only"


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    add_training_run_arguments(
        parser,
        cohort_help=(
            "explicit train/validation flight roster; the run is development-only and "
            "defines no outer-test population"
        ),
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--cv-parameters",
        type=cv_parameters_type,
        default=DEFAULT_CV_PARAMETERS,
        metavar=",".join(DEFAULT_CV_PARAMETERS),
        help=(
            "comma-separated parameters to search exhaustively; default: "
            f"{','.join(DEFAULT_CV_PARAMETERS)}; available: {','.join(CV_PARAMETER_GRIDS)}"
        ),
    )
    parser.add_argument("--cv-epochs", type=int, default=DEFAULT_CV_EPOCHS)
    parser.add_argument("--cv-patience", type=int, default=DEFAULT_CV_PATIENCE)


def run_cli(
    args: argparse.Namespace, parser: argparse.ArgumentParser, argv: list[str] | None
) -> int:
    run = prepare_training_run(args, parser, argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "data_selection.json"
    selection_tmp = args.output_dir / "data_selection.json.tmp"
    selection_tmp.write_text(json.dumps(run.data_selection, indent=2), encoding="utf-8")
    selection_tmp.replace(selection_path)
    try:
        cross_validate(
            run.series,
            run.config,
            output_dir=args.output_dir,
            data_provenance=run.data_provenance,
            n_splits=args.folds,
            cv_parameters=args.cv_parameters,
            cv_epochs=args.cv_epochs,
            cv_patience=args.cv_patience,
            auto_batch_size=run.batch_auto,
        )
    except Exception as exc:
        run.finish(exc)
        raise
    run.finish()
    return 0
