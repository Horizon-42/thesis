"""Trajectory evaluation — judge trajectories against their targets.

File-based seam at the END of the modeling pipeline: inputs are per-trajectory
record files (initial state + target state + a state list with a 1:1 aligned
control list — see ``records.py`` for the contract), the output is one JSON report
with per-trajectory verdicts and batch metrics. Pass/fail gates are regulation-
derived (``thresholds.py``, FAA Order 8260.58D). Depends only on ``geokit`` +
stdlib — never on the optimizer.

The arrival event is subject-aware (``arrival.py``): optimized and predicted
records are measured at ``states[-1]`` vs ``target_state``; observed records
(``source.subject == "observed"``) at their fitted final approach extrapolated
to the threshold, with an established-on-final precondition (``not_established``
is a counted outcome, never a drop) and a per-flight ``marginal`` flag where the
95 % interval straddles a gate.

CLI: ``python -m evaluation --input <dir> --output report.json``
"""

from evaluation.arrival import (
    ArrivalDeviation,
    ArrivalOutcome,
    EstablishedCriteria,
    Subject,
    arrival_deviation,
    final_state_deviation,
    subject_of,
)
from evaluation.metrics import (
    TrajectoryEvaluation,
    evaluate_batch,
    evaluate_record,
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
    "ArrivalDeviation",
    "ArrivalOutcome",
    "DeviationThresholds",
    "EstablishedCriteria",
    "ReferenceComparison",
    "Subject",
    "TrajectoryEvaluation",
    "TrajectoryRecord",
    "arrival_deviation",
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
    "subject_of",
]
