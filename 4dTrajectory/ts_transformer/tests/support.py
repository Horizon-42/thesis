"""The fixtures the test files share (review §4.6, T4-24).

Until 2026-09-10 nine files carried a byte-identical fake data provenance, three the same
control dynamics context and two the same terminal assessment context. `tests/` is a
namespace package under `ts_transformer`, so a test imports these as
``from ts_transformer.tests.support import …`` — the same way `test_two_head_duration`
already borrowed `test_duration_quantiles`'s config.
"""

from __future__ import annotations

import torch

from evaluation.thresholds import AssessmentContext
from ts_transformer.data.data_provenance import ARRIVAL_DATA_PROVENANCE_SCHEMA
from ts_transformer.outputs.control.conditioning import DYNAMICS_CONDITION_NAMES
from ts_transformer.outputs.control.envelope import CONTROL_LOWER, CONTROL_UPPER

AIRPORT, RUNWAY = "KRDU", "05L"


def fake_data_provenance(*airports: str) -> dict:
    """The arrival-data provenance a training fixture stamps: one manifest per airport
    (KRDU when none is named), with a placeholder digest and no source records."""
    return {
        "schema_version": ARRIVAL_DATA_PROVENANCE_SCHEMA,
        "manifests": [
            {"airport": airport, "arrival_manifest_sha256": "a" * 64, "source_records": []}
            for airport in (airports or (AIRPORT,))
        ],
    }


def dynamics_context(batch: int, cta_s: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """A control model's per-flight context for ``batch`` synthetic flights: a random
    condition vector and the shared dimensionless control box, plus the CTA when given."""
    rows = {
        "condition": torch.randn(batch, len(DYNAMICS_CONDITION_NAMES)),
        "control_lower": torch.tensor(CONTROL_LOWER, dtype=torch.float32).expand(batch, -1).clone(),
        "control_upper": torch.tensor(CONTROL_UPPER, dtype=torch.float32).expand(batch, -1).clone(),
    }
    if cta_s is not None:
        rows["cta_s"] = cta_s
    return rows


def terminal_contexts() -> dict[tuple[str, str], AssessmentContext]:
    """The one terminal assessment context the synthetic KRDU 05L fixtures are graded under.

    The synthetic fixtures build their approaches on the runway_thresholds.json point
    (flight_scenarios.runway_target.find_threshold), which sits 6.7 m from the CIFP Path
    Point LTP a real KRDU context would carry; the synthetic one is pinned so this context
    describes the data it is grading.
    """
    return {(AIRPORT, RUNWAY): AssessmentContext(
        benchmark="lpv", airport=AIRPORT, runway=RUNWAY,
        threshold_lat=35.8745003, threshold_lon=-78.802002,
        runway_course_deg=45.0, runway_width_m=45.72,
        runway_source="faa_nasr_apt_rwy", runway_source_cycle="2026-08-06",
        procedure_source="faa_cifp_path_point", procedure_source_cycle="2026-08-06",
        threshold_elevation_hae_m=141.86, threshold_elevation_msl_m=111.86,
        threshold_crossing_height_m=15.0, lpv_course_width_m=106.75,
    )}
