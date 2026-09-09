"""``freeze-test``: bind the one-shot outer-test ledger to a checkpoint and roster.

Irreversible by design. Everything about the experiment must be decided before this runs;
afterwards the test split can be predicted exactly once.
"""

from __future__ import annotations

import argparse

from ts_transformer.evaluation_protocol import TestReleaseError, create_test_release
from ts_transformer.train import load_checkpoint

from .common import add_eligibility_arg, provenance_from_args

HELP = "bind the one-shot outer-test ledger to a frozen checkpoint and data roster"


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data", required=True, action="append",
        help="the complete training arrival-manifest roster; repeat for pooled training",
    )
    add_eligibility_arg(parser)


def run_cli(
    args: argparse.Namespace, parser: argparse.ArgumentParser, argv: list[str] | None
) -> int:
    del argv
    _model, _config, _normalizer, payload = load_checkpoint(args.checkpoint)
    current_provenance = provenance_from_args(args)
    try:
        release_path = create_test_release(
            args.checkpoint, payload, current_provenance
        )
    except TestReleaseError as exc:
        parser.error(str(exc))
    print(f"✓ outer-test frozen for one-shot evaluation: {release_path}")
    return 0

