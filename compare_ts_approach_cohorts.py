"""Compare A/B/C checkpoints on one frozen validation approach cohort."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
TS_ROOT = REPO_ROOT / "4dTrajectory" / "ts_transformer"
sys.path[:0] = [str(TS_ROOT), str(REPO_ROOT)]

from approach_clustering.evaluation import compare_checkpoints  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.checkpoint) != len(args.label):
        parser.error("repeat --checkpoint and --label the same number of times")
    document = compare_checkpoints(
        data=args.data,
        cohort_path=args.cohort,
        checkpoints=args.checkpoint,
        labels=args.label,
        output_path=args.output,
        device_name=args.device,
    )
    for experiment in document["experiments"]:
        metrics = experiment["shared_validation"]["common_grid_metrics"]
        print(
            f"{experiment['label']}: ADE={metrics['ade_m']:.1f} m "
            f"FDE={metrics['fde_m']:.1f} m "
            f"terminal velocity={metrics['terminal_velocity_error_mps']:.2f} m/s"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
