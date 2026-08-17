"""Assessment context and standards-backed threshold resolution.

The evaluator does not own runway or procedure data.  A producer supplies an
explicit :class:`AssessmentContext`; this module only turns that context into
the LPV or RNP APCH LNAV/VNAV limits documented in
``FINAL_APPROACH_VERDICT_STANDARD.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, get_args

Benchmark = Literal["lpv", "rnp_apch_lnav_vnav_baro"]
BENCHMARKS = get_args(Benchmark)
ComponentResult = Literal["pass", "fail", "indeterminate"]
Verdict = Literal["pass", "fail", "indeterminate"]

# THE LATERAL CRITERION IS HALF THE PUBLISHED RUNWAY WIDTH -- and that is the whole
# rule, deliberately.
#
# The obvious alternative is the procedure's own lateral containment: the LPV course
# width at threshold (106.75 m = 350 ft, FAA Formula 3-1-1) or the RNP APCH LNAV
# 0.15 NM (277.8 m) cross-track allowance. Measured over this project's entire fleet
# -- 26 thresholds at five airports -- those are 2.3x to 18x wider than the runway,
# so the previous ``min(guidance, runway_width / 2)`` rule selected the runway
# half-width at 26 of 26 runways and the guidance term never once bound.
#
# A term that cannot change an answer is worse than absent: it invites the reader to
# believe the verdict was a navigation-containment judgement, and it hides its own
# mistakes. The version that carried it also divided the LPV course width by two --
# that value is ALREADY a semi-width (centreline to full-scale deflection), so the
# ``guidance_lateral_m`` every report published was wrong by a factor of two, and
# nobody noticed precisely because it was inert.
#
# So the criterion is stated as what it is: did the extrapolated crossing land over
# the pavement. That is a LANDING-GEOMETRY claim, not a navigation-containment one,
# and ``evaluation.metrics.METHODOLOGY`` says so inside every report.
LATERAL_CRITERION_ID = "runway_half_width_at_threshold"

# Hashed into every ``evaluation_context_fingerprint``. Bump whenever the set of fields
# below changes: the hash already moves on its own, but the label is what tells a reader
# of an old report WHICH schema produced the digest they are holding.
CONTEXT_SCHEMA_VERSION = "terminal-assessment-context-v2"

# One common terminal-geometry acceptance bound for the vertically guided RNAV
# benchmarks supported by this project.  ICAO Doc 9613, Volume II, Part C,
# Chapter 5, Section A, §5.3.4.4.7 specifies +22 m/-22 m for Baro-VNAV
# deviations during the RNP APCH final approach segment.  The evaluator uses
# that value as its project-wide RNAV terminal bound; it does not reinterpret
# LPV display full-scale deflection as a landing-success threshold.
RNAV_TERMINAL_VERTICAL_BOUND_M = 22.0
RNAV_TERMINAL_VERTICAL_STANDARD_ID = "icao_doc_9613_rnp_apch_fas_22m"


# Every numeric field that can silently decide a verdict. These come from parsed FAA
# files rather than from Python literals, and a NaN reaching ``metrics._component``
# makes every ordered comparison False -- which reads out as a clean ``fail`` rather
# than as the data error it is. This is the one shape check worth keeping.
_FINITE_FIELDS = (
    "threshold_lat",
    "threshold_lon",
    "runway_course_deg",
    "runway_width_m",
    "threshold_elevation_hae_m",
    "threshold_elevation_msl_m",
    "threshold_crossing_height_m",
    "lpv_course_width_m",
)


@dataclass(frozen=True)
class AssessmentContext:
    """Runway/procedure facts required to resolve one terminal verdict.

    No trajectory fit or verdict belongs here.  Published threshold elevations
    and TCH define the authoritative desired crossing altitude; vertical
    acceptance policy remains evaluation-owned and is not supplied by a trajectory.

    ``threshold_lat``/``threshold_lon`` are the authoritative LANDING threshold, and
    they are what every deviation is measured from -- not the position a record
    happens to carry in its own ``target_state``. Without them here the evaluator had
    to take the artifact's word for where the runway is, which is the shape of the
    displaced-threshold bug: a target 775 m from the published point produces clean
    near-zero deviations and no symptom anywhere downstream.
    """

    benchmark: Benchmark
    airport: str
    runway: str
    threshold_lat: float
    threshold_lon: float
    runway_course_deg: float
    runway_width_m: float
    runway_source: str
    runway_source_cycle: str
    procedure_source: str
    procedure_source_cycle: str
    threshold_elevation_hae_m: float | None = None
    threshold_elevation_msl_m: float | None = None
    threshold_crossing_height_m: float | None = None
    # The published LPV course width at threshold. It selects the benchmark upstream
    # (``assessment_for_runway``) and is recorded as procedure provenance; it does NOT
    # bound the lateral result -- see LATERAL_CRITERION_ID above.
    lpv_course_width_m: float | None = None
    baro_vnav_approved: bool = False
    threshold_frame_fingerprint: str | None = None

    def __post_init__(self) -> None:
        # Only what can silently change a verdict is checked. The field types are the
        # dataclass's job, and every caller builds this from a typed ``Runway``.
        if self.benchmark not in BENCHMARKS:
            # ``limits()`` dispatches on this, and its second branch is a fallthrough:
            # an unrecognised benchmark would quietly be graded as LNAV/VNAV.
            raise ValueError(f"unsupported benchmark {self.benchmark!r}")
        for name in _FINITE_FIELDS:
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.runway_width_m <= 0.0:
            # The effective lateral bound is half of this; zero would fail everything.
            raise ValueError(f"runway_width_m must be positive, got {self.runway_width_m!r}")
        if (
            self.threshold_crossing_height_m is not None
            and self.threshold_crossing_height_m <= 0.0
        ):
            # The vertical REFERENCE PLANE is LTP elevation + this. Published values run
            # 15.27-18.11 m across this fleet, so a zero or negative TCH would shift the
            # plane by most of the +/-22 m window while every verdict still looked clean.
            raise ValueError(
                "threshold_crossing_height_m must be positive, got "
                f"{self.threshold_crossing_height_m!r}"
            )
        if self.benchmark == "lpv":
            for name in (
                "threshold_elevation_hae_m",
                "threshold_elevation_msl_m",
                "threshold_crossing_height_m",
            ):
                if getattr(self, name) is None:
                    raise ValueError(f"{name} is required for a vertically guided benchmark")

    @property
    def desired_threshold_altitude_msl_m(self) -> float | None:
        if (
            self.threshold_elevation_msl_m is None
            or self.threshold_crossing_height_m is None
        ):
            return None
        return self.threshold_elevation_msl_m + self.threshold_crossing_height_m

    @property
    def hae_minus_msl_m(self) -> float | None:
        if (
            self.threshold_elevation_hae_m is None
            or self.threshold_elevation_msl_m is None
        ):
            return None
        return self.threshold_elevation_hae_m - self.threshold_elevation_msl_m

    @property
    def evaluation_context_fingerprint(self) -> str:
        """Hash every physical and policy fact that can change a verdict."""
        encoded = json.dumps(
            {
                # v2: gained threshold_lat/threshold_lon (the frame origin became
                # authoritative) and renamed lpv_lateral_fsd_m -> lpv_course_width_m.
                "schema_version": CONTEXT_SCHEMA_VERSION,
                **asdict(self),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "desired_threshold_altitude_msl_m": self.desired_threshold_altitude_msl_m,
            "evaluation_context_fingerprint": self.evaluation_context_fingerprint,
        }

    def limits(self) -> "ResolvedLimits":
        """The two component bounds this context resolves to.

        Lateral is benchmark-independent: it is the runway, and the runway does not
        change width because the procedure did. Only the vertical bound depends on
        which benchmark applies, and only because LNAV/VNAV needs approved Baro-VNAV
        plus a published path reference before ±22 m means anything.
        """
        lateral_m = self.runway_width_m / 2.0
        if self.benchmark == "lpv":
            return ResolvedLimits(
                benchmark=self.benchmark,
                lateral_m=lateral_m,
                vertical_lower_m=-RNAV_TERMINAL_VERTICAL_BOUND_M,
                vertical_upper_m=RNAV_TERMINAL_VERTICAL_BOUND_M,
            )

        vertical_available = (
            self.baro_vnav_approved
            and self.desired_threshold_altitude_msl_m is not None
        )
        if not self.baro_vnav_approved:
            vertical_reason = (
                "LNAV/VNAV benchmark requires explicit approved Baro-VNAV context"
            )
        elif not vertical_available:
            vertical_reason = (
                "authoritative Baro-VNAV threshold path reference unavailable"
            )
        else:
            vertical_reason = None
        return ResolvedLimits(
            benchmark=self.benchmark,
            lateral_m=lateral_m,
            vertical_lower_m=(
                -RNAV_TERMINAL_VERTICAL_BOUND_M if vertical_available else None
            ),
            vertical_upper_m=(
                RNAV_TERMINAL_VERTICAL_BOUND_M if vertical_available else None
            ),
            vertical_reason=vertical_reason,
        )


@dataclass(frozen=True)
class ResolvedLimits:
    """Numeric component limits after resolving one assessment context.

    ``lateral_m`` is always available (a runway always has a width), so a lateral
    component is never indeterminate once a crossing was measured. The vertical
    bound can be absent, and ``vertical_reason`` then says why.
    """

    benchmark: Benchmark
    lateral_m: float
    vertical_lower_m: float | None
    vertical_upper_m: float | None
    vertical_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"lateral_criterion": LATERAL_CRITERION_ID, **asdict(self)}
