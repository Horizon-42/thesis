"""``evaluate-fit``: replay a checkpoint deterministically on its own train + validation.

The same fixed ``L-1`` anchor, ``eval()`` and sequential batches on both splits, so the
generalization gap is a comparison and not an artifact of two different evaluators.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_transformer.data.data_provenance import require_matching_data_provenance
from ts_transformer.data.dataset import build_series, dataset_flight_key, load_flight_dicts
from ts_transformer.backbone.adapters import resolve_device
from ts_transformer.training.train import (
    HISTORY_NAME, evaluate_fit_splits, load_checkpoint, write_fit_evaluation,
)

from .common import add_eligibility_arg, provenance_from_args

HELP = "replay a best checkpoint deterministically on fixed-anchor train + validation"


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data", required=True, action="append",
        help="the complete training arrival-manifest roster; repeat for pooled training",
    )
    add_eligibility_arg(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="destination for fit_evaluation.json (default: checkpoint directory)",
    )
    parser.add_argument("--device", default="auto", help='"auto", "cpu" or "cuda"')


def run_cli(
    args: argparse.Namespace, parser: argparse.ArgumentParser, argv: list[str] | None
) -> int:
    del argv
    checkpoint_path = Path(args.checkpoint).resolve()
    model, config, normalizer, payload = load_checkpoint(checkpoint_path)
    current_provenance = provenance_from_args(args)
    require_matching_data_provenance(payload, current_provenance)

    wanted = payload["split"]["train"] + payload["split"]["val"]
    flights = load_flight_dicts(
        args.data,
        include_flight_keys=set(wanted),
    )
    indexed_flights = {
        dataset_flight_key(flight, index): flight
        for index, flight in enumerate(flights)
    }
    missing = [key for key in wanted if key not in indexed_flights]
    if missing:
        parser.error(
            f"{len(missing)} train/validation flight(s) from the checkpoint are absent; "
            f"first missing key: {missing[0]!r}"
        )
    selected_flights = [indexed_flights[key] for key in wanted]
    series, report = build_series(selected_flights, config)
    print(f"  aircraft   filter {config.aircraft_filter}")
    print(report.format())
    indexed_series = {item.dataset_id: item for item in series}
    missing = [key for key in wanted if key not in indexed_series]
    if missing:
        parser.error(
            f"{len(missing)} checkpoint train/validation flight(s) could not be rebuilt; "
            f"first missing key: {missing[0]!r}"
        )
    train_series = [indexed_series[key] for key in payload["split"]["train"]]
    val_series = [indexed_series[key] for key in payload["split"]["val"]]

    history_path = checkpoint_path.parent / HISTORY_NAME
    history = []
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
    evaluation = evaluate_fit_splits(
        model,
        train_series,
        val_series,
        normalizer,
        config,
        resolve_device(args.device),
        history=history,
    )
    output_dir = args.output_dir or checkpoint_path.parent
    document = write_fit_evaluation(
        evaluation,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )
    for split in ("train", "val"):
        metrics = document["splits"][split]["metrics"]
        print(
            f"  {split:5s}  ADE {metrics['ade_m']:7.1f} m   "
            f"FDE {metrics['fde_m']:7.1f} m   "
            f"time MAE {metrics['final_time_s']['mae']:5.1f} s"
        )
    gap = document["diagnostics"]["generalization"]
    print(
        "  gap    "
        f"ADE {gap['ade_m']['absolute_gap']:+.1f} m "
        f"({gap['ade_m']['ratio']:.2f}×)   "
        f"FDE {gap['fde_m']['absolute_gap']:+.1f} m "
        f"({gap['fde_m']['ratio']:.2f}×)"
    )
    print(f"✓ wrote {output_dir / 'fit_evaluation.json'}")
    return 0

