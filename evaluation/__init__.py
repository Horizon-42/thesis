"""Trajectory evaluation — judge optimized trajectories against their targets.

File-based seam at the END of the optimization pipeline: inputs are per-trajectory
record files (initial state + target state + a state list with a 1:1 aligned
control list — see ``records.py`` for the contract), the output is one JSON report
with per-trajectory verdicts and batch metrics (solve rate, success rate, lateral /
vertical deviation spreads, flight times). Pass/fail gates are regulation-derived
(``thresholds.py``, FAA Order 8260.58D). Depends only on ``geokit`` + stdlib —
never on the optimizer.

CLI: ``python -m evaluation --input <dir> --output report.json``
"""

from evaluation.metrics import (
    FinalStateDeviation,
    TrajectoryEvaluation,
    evaluate_batch,
    evaluate_record,
    final_state_deviation,
)
from evaluation.records import (
    STATE_KEYS,
    TrajectoryRecord,
    load_record,
    load_records,
    record_from_dict,
)
from evaluation.reference import (
    ReferenceComparison,
    compare_to_reference,
    horizontal_arc_length_m,
    load_reference,
    resample_by_arc_length,
)
from evaluation.stats import percentile
from evaluation.thresholds import DeviationThresholds

__all__ = [
    "STATE_KEYS",
    "DeviationThresholds",
    "FinalStateDeviation",
    "ReferenceComparison",
    "TrajectoryEvaluation",
    "TrajectoryRecord",
    "compare_to_reference",
    "evaluate_batch",
    "evaluate_record",
    "final_state_deviation",
    "horizontal_arc_length_m",
    "load_record",
    "load_records",
    "load_reference",
    "percentile",
    "record_from_dict",
    "resample_by_arc_length",
]
