"""Which runway did this track land on? A relative comparison, never a quality gate.

ONE TRACK IN, ONE RUNWAY OUT -- BY CONSTRUCTION
-----------------------------------------------
The predecessor classified each track independently PER THRESHOLD and then tried to
undo the resulting double-assignment with a pairwise "does this runway beat its
parallel neighbour" guard. The guard's logic was right, but a guard can be bypassed:
measured on the artifacts that shipped with it, 232 of KSJC's 319 unique landings
(72.7%) still sat in two runways' files at once -- 169 in both 30L and 30R, 63 in both
12L and 12R, plus 32 at KSTL.

Here the shape of the computation forbids it: the track is fitted against every
candidate ONCE and the arg-min is taken. There is no code path that emits two runways,
so the bug is not prevented -- it is unrepresentable.

WHAT DISCRIMINATES, AND WHAT CANNOT
-----------------------------------
* Distance to the threshold point CANNOT separate parallels (measured: 763 m for the
  correct runway vs 791 m for the wrong one). It is even less useful where the runway
  has a displaced threshold, because ADS-B coverage stops before the landing point --
  at KSJC 30L every track ends ~775 m from it.
* Runway heading CANNOT separate parallels either: they share a landing direction.
* Median absolute CROSS-TRACK offset does. Parallels are separated LATERALLY, so a
  misassigned track scores about the runway separation while a correct one scores
  metres (measured at KSJC 30L/30R, 228.4 m apart: correct 13.9 m, misassigned
  230.5 m -- the wrong file's median offset IS the separation).
* Cross-track cannot separate the two ENDS of one runway -- they share the extended
  centreline. Direction of travel does, via ``SegmentFit.approaching``.

NO QUALITY FILTERING HERE
-------------------------
Assignment answers "which runway", never "flown well enough". If it rejected tracks on
the criterion ``evaluation`` later reports, every surviving track would pass by
construction and the established rate would be 1.0 -- manufactured by the selection
rather than measured. The only rejections are structural: not a landing, no fittable
segment, or a genuinely ambiguous one. All three are returned as outcomes and counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from final_approach.fit import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_MIN_SPAN_M,
    DEFAULT_WINDOW_M,
    SegmentFit,
    fit_final_segment,
)
from final_approach.frame import RunwayFrame, TrackPoint

# How much closer to one runway's centreline than to the runner-up before the
# assignment is called. Sits an order of magnitude below the tightest parallel
# separation in this fleet (228.4 m, KSJC 30L/30R) and well above the scatter of a
# correct assignment (median 5-35 m across all five airports' clean runways), so it
# only ever fires on a genuine tie.
AMBIGUITY_MARGIN_M = 50.0

Outcome = Literal["assigned", "ambiguous", "unassignable", "not_landing"]


@dataclass(frozen=True)
class LandingScreen:
    """Did this track land at this airport at all -- direction and runway aside.

    Deliberately generous, because crowd-sourced ADS-B rarely reaches the ground near
    a runway: this keys off an ALIGNED DESCENT that gets low near a threshold, not off
    a touchdown altitude.

    ``max_height_m`` compares against the frame's threshold elevation, so the package
    datum contract applies -- track altitudes and threshold elevation must share a
    datum. A caller that mixes HAE and MSL shifts this test by the geoid undulation
    (~33 m over the US), which is 2.2% of the default and silent.
    """

    threshold_radius_m: float = 1000.0
    max_height_m: float = 1500.0
    descent_margin_m: float = 300.0


@dataclass(frozen=True)
class Assignment:
    """The verdict, plus every number that produced it.

    ``scores`` maps runway ident -> median absolute cross-track offset (metres) for
    each candidate that yielded a fit. It is kept on every outcome, including the
    rejections, so a disputed assignment can be audited without re-running anything --
    and so ``ambiguous`` cases carry the evidence of what they were torn between.
    """

    outcome: Outcome
    runway: str | None
    fit: SegmentFit | None
    scores: Mapping[str, float]
    margin_m: float | None
    reason: str | None = None


def _closest(points: Sequence[TrackPoint], frames: Sequence[RunwayFrame]) -> tuple[RunwayFrame, int, float]:
    """The (frame, index, distance) of the track's closest approach to any threshold."""
    best = min(
        (
            (frame.distance_m(point), frame_index, point_index)
            for frame_index, frame in enumerate(frames)
            for point_index, point in enumerate(points)
        ),
    )
    distance, frame_index, point_index = best
    return frames[frame_index], point_index, distance


def _is_landing(
    points: Sequence[TrackPoint],
    frame: RunwayFrame,
    anchor_index: int,
    distance_m: float,
    screen: LandingScreen,
) -> str | None:
    """None if this looks like a landing; otherwise why it does not."""
    if distance_m > screen.threshold_radius_m:
        return f"closest approach {distance_m:.0f} m > {screen.threshold_radius_m:.0f} m"
    anchor_height = frame.project(points[anchor_index]).height_m
    if anchor_height > screen.max_height_m:
        return f"height at closest approach {anchor_height:.0f} m > {screen.max_height_m:.0f} m"
    if anchor_index == 0:
        return "closest approach is the first sample (no descent observable)"
    descent = max(p.alt_m for p in points[:anchor_index]) - points[anchor_index].alt_m
    if descent < screen.descent_margin_m:
        return f"descended {descent:.0f} m < {screen.descent_margin_m:.0f} m"
    return None


def assign_runway(
    points: Sequence[TrackPoint],
    frames: Sequence[RunwayFrame],
    *,
    screen: LandingScreen = LandingScreen(),
    window_m: tuple[float, float] = DEFAULT_WINDOW_M,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_span_m: float = DEFAULT_MIN_SPAN_M,
) -> Assignment:
    """Assign ``points`` (time-ordered) to at most one of ``frames``.

    ``frames`` is the airport's FULL threshold list. Passing a subset re-opens the
    double-assignment failure: the arg-min can only exclude a competitor it was shown.

    The four outcomes are exhaustive and each is a first-class result:

    ``not_landing``    the track never descended to near a threshold -- an overflight,
                       a departure, or a landing at a different airport.
    ``unassignable``   it landed, but no runway yielded a fittable final segment. This
                       is a COVERAGE statement (the track was truncated before the
                       window filled), not a judgement of the approach, and it must be
                       counted separately from a badly flown one.
    ``ambiguous``      two runways are within ``AMBIGUITY_MARGIN_M`` of each other.
                       Returned rather than resolved: guessing here is precisely how
                       one landing ends up in two runways' statistics.
    ``assigned``       one runway wins outright.
    """
    if len(points) < 2:
        return Assignment("not_landing", None, None, {}, None, "fewer than 2 samples")

    frame, anchor_index, distance_m = _closest(points, frames)
    not_landing = _is_landing(points, frame, anchor_index, distance_m, screen)
    if not_landing is not None:
        return Assignment("not_landing", None, None, {}, None, not_landing)

    fits: dict[str, SegmentFit] = {}
    for candidate in frames:
        fit = fit_final_segment(
            points,
            candidate,
            window_m=window_m,
            min_samples=min_samples,
            min_span_m=min_span_m,
        )
        # A fit against the opposite end of the same runway is geometrically perfect
        # and completely wrong; direction of travel is what rules it out.
        if fit is not None and fit.approaching:
            fits[candidate.ident] = fit

    if not fits:
        return Assignment(
            "unassignable", None, None, {}, None,
            f"no runway yielded a final segment in [{window_m[0]:.0f}, {window_m[1]:.0f}] m",
        )

    scores = {ident: fit.median_abs_cross_m for ident, fit in fits.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    best_ident, best_score = ranked[0]
    margin = ranked[1][1] - best_score if len(ranked) > 1 else None

    if margin is not None and margin < AMBIGUITY_MARGIN_M:
        return Assignment(
            "ambiguous", None, None, scores, margin,
            f"{best_ident} ({best_score:.0f} m) and {ranked[1][0]} ({ranked[1][1]:.0f} m) "
            f"differ by {margin:.0f} m < {AMBIGUITY_MARGIN_M:.0f} m",
        )

    return Assignment("assigned", best_ident, fits[best_ident], scores, margin)
