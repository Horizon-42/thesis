"""Assessment context and standards-backed threshold resolution.

The evaluator does not own runway or procedure data.  A producer supplies an
explicit :class:`AssessmentContext`; this module only turns that context into
the LPV or RNP APCH LNAV/VNAV limits documented in
``FINAL_APPROACH_VERDICT_STANDARD.md``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

Benchmark = Literal["lpv", "rnp_apch_lnav_vnav_baro"]
ComponentResult = Literal["pass", "fail", "indeterminate", "not_applicable"]
Verdict = Literal["pass", "fail", "indeterminate"]

LNAV_FINAL_XTK_M = 0.15 * 1852.0
BARO_VNAV_VERTICAL_M = 22.0
NORMAL_95_MULTIPLIER = 1.96
LPV_VERTICAL_FSD_MIN_M = 15.0
ICAO_NORMAL_FSD_FRACTION = 0.5
LPV_VERTICAL_BOUND_M = LPV_VERTICAL_FSD_MIN_M * ICAO_NORMAL_FSD_FRACTION
LPV_VERTICAL_SCALE_MODEL = "do229_lpv_angular_min_clamped"


def _positive(name: str, value: float | None, *, required: bool = True) -> float | None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required")
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return numeric


def _finite(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return numeric


@dataclass(frozen=True)
class AssessmentContext:
    """Runway/procedure facts required to resolve one terminal verdict.

    No trajectory fit or verdict belongs here.  Published threshold elevations
    and TCH define the authoritative desired crossing altitude; LPV vertical
    scale policy remains evaluation-owned and is not supplied by a trajectory.
    """

    benchmark: Benchmark
    airport: str
    runway: str
    runway_course_deg: float
    runway_width_m: float
    runway_source: str
    runway_source_cycle: str
    procedure_source: str
    procedure_source_cycle: str
    threshold_elevation_hae_m: float | None = None
    threshold_elevation_msl_m: float | None = None
    threshold_crossing_height_m: float | None = None
    lpv_lateral_fsd_m: float | None = None
    baro_vnav_approved: bool = False

    def __post_init__(self) -> None:
        if self.benchmark not in ("lpv", "rnp_apch_lnav_vnav_baro"):
            raise ValueError(f"unsupported benchmark {self.benchmark!r}")
        for name in (
            "airport", "runway", "runway_source", "runway_source_cycle",
            "procedure_source", "procedure_source_cycle",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        _finite("runway_course_deg", self.runway_course_deg)
        _positive("runway_width_m", self.runway_width_m)
        for name in ("threshold_elevation_hae_m", "threshold_elevation_msl_m"):
            value = getattr(self, name)
            if value is not None:
                _finite(name, value)
        _positive(
            "threshold_crossing_height_m",
            self.threshold_crossing_height_m,
            required=False,
        )
        _positive("lpv_lateral_fsd_m", self.lpv_lateral_fsd_m, required=False)
        if not isinstance(self.baro_vnav_approved, bool):
            raise ValueError("baro_vnav_approved must be boolean")
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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AssessmentContext":
        return cls(
            benchmark=data["benchmark"],
            airport=str(data["airport"]),
            runway=str(data["runway"]),
            runway_course_deg=float(data["runway_course_deg"]),
            runway_width_m=float(data["runway_width_m"]),
            runway_source=str(data["runway_source"]),
            runway_source_cycle=str(data["runway_source_cycle"]),
            procedure_source=str(data["procedure_source"]),
            procedure_source_cycle=str(data["procedure_source_cycle"]),
            threshold_elevation_hae_m=(
                None if data.get("threshold_elevation_hae_m") is None
                else float(data["threshold_elevation_hae_m"])
            ),
            threshold_elevation_msl_m=(
                None if data.get("threshold_elevation_msl_m") is None
                else float(data["threshold_elevation_msl_m"])
            ),
            threshold_crossing_height_m=(
                None if data.get("threshold_crossing_height_m") is None
                else float(data["threshold_crossing_height_m"])
            ),
            lpv_lateral_fsd_m=(
                None if data.get("lpv_lateral_fsd_m") is None
                else float(data["lpv_lateral_fsd_m"])
            ),
            baro_vnav_approved=data.get("baro_vnav_approved", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "desired_threshold_altitude_msl_m": self.desired_threshold_altitude_msl_m,
        }

    def limits(self) -> "ResolvedLimits":
        runway_half = self.runway_width_m / 2.0
        if self.benchmark == "lpv":
            guidance = (
                None if self.lpv_lateral_fsd_m is None
                else self.lpv_lateral_fsd_m / 2.0
            )
            lateral = None if guidance is None else min(guidance, runway_half)
            return ResolvedLimits(
                benchmark=self.benchmark,
                guidance_lateral_m=guidance,
                runway_lateral_m=runway_half,
                effective_lateral_m=lateral,
                vertical_lower_m=-LPV_VERTICAL_BOUND_M,
                vertical_upper_m=LPV_VERTICAL_BOUND_M,
                vertical_scale_model=LPV_VERTICAL_SCALE_MODEL,
                vertical_fsd_m=LPV_VERTICAL_FSD_MIN_M,
                vertical_fsd_fraction=ICAO_NORMAL_FSD_FRACTION,
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
            guidance_lateral_m=LNAV_FINAL_XTK_M,
            runway_lateral_m=runway_half,
            effective_lateral_m=min(LNAV_FINAL_XTK_M, runway_half),
            vertical_lower_m=-BARO_VNAV_VERTICAL_M if vertical_available else None,
            vertical_upper_m=BARO_VNAV_VERTICAL_M if vertical_available else None,
            vertical_reason=vertical_reason,
        )


@dataclass(frozen=True)
class ResolvedLimits:
    """Numeric component limits after resolving one assessment context."""

    benchmark: Benchmark
    guidance_lateral_m: float | None
    runway_lateral_m: float
    effective_lateral_m: float | None
    vertical_lower_m: float | None
    vertical_upper_m: float | None
    vertical_reason: str | None = None
    vertical_scale_model: str | None = None
    vertical_fsd_m: float | None = None
    vertical_fsd_fraction: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
