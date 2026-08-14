"""Sort reconstructed tracks into the four harvest buckets.

Thin by design: the geometry is ``final_approach``'s, the datum is
``harvest.airports``'s, and this module only decides where a track goes and gives it its
identity. Nothing here judges how well an approach was flown -- see the
``final_approach`` package docstring for why that separation is load-bearing.

IDENTITY
--------
A flight is ``(icao24, landing time)``. The raw harvest carries no unique id at all:
``callsign`` is not unique (measured on 996 KRDU arrivals: 552 distinct callsigns, 717
distinct icao24, but 996 distinct icao24+landing-time), and OpenSky stores state vectors
by aircraft and time, so an "arrival" is a segment this project derives. Every layer
that keyed on the callsign has been bitten -- the ts train/val/test split leaked, the
comparison CZML dropped 22% of a batch, the frontend table swapped verdicts between
namesakes, and Cesium merged two flights into one entity.

The landing time here is the time of the sample closest to the assigned threshold ON THE
FINAL INBOUND PASS selected by ``final_approach``.  Contiguity alone is not enough: one
flight can overfly a threshold, go around, and later land without any time discontinuity.
Searching the whole track would let the earlier pass steal both the landing timestamp and
the arrival crop merely because its discrete ADS-B sample happened to lie closer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from final_approach import Assignment, LandingScreen, SegmentFit, assign_runway

from flight_scenarios.identity import flight_key

from trajectory_data_process.harvest.airports import Airport
from trajectory_data_process.harvest.threshold_event import (
    StateMetadataLookup,
    build_observed_threshold_event,
)
from trajectory_data_process.harvest.tracks import Track


@dataclass(frozen=True)
class ClassifiedTrack:
    """A track, its bucket, and the evidence that put it there."""

    track: Track
    assignment: Assignment
    landing_time_utc: str | None
    landing_sample_index: int | None
    observed_threshold_event: dict

    @property
    def outcome(self) -> str:
        return self.assignment.outcome

    @property
    def runway(self) -> str | None:
        return self.assignment.runway

    @property
    def fit(self) -> SegmentFit | None:
        return self.assignment.fit

    @property
    def flight_key(self) -> str:
        """``<callsign>_<runway>_<icao24>_<landingtime>`` -- unique by the last two.

        Delegates to ``flight_scenarios.identity.flight_key`` rather than formatting the
        parts here. That function is the canonical identity: the observed CZML's entity
        ids, the optimizer's record filenames, the ts split keys and the comparison
        builder's reference lookup all derive from it, and it is pinned to a shared test
        vector precisely so no second implementation can drift from it. A hand-rolled
        copy here would be the third, and the join it feeds (verdict -> painted track) is
        exactly the kind that fails silently -- every flight simply comes out unmatched.

        Unassigned tracks substitute their bucket for the runway and their track end for
        the landing time, so every bucket writes into one flat namespace without
        collisions.
        """
        return flight_key(
            {
                "id": (self.track.callsign or self.track.icao24).replace(" ", "")[:16],
                "runway": self.runway or self.outcome,
                "icao24": self.track.icao24,
                "landing_time_utc": self.landing_time_utc or _iso(self.track.end_s),
            },
            index=0,
        )


def classify_track(
    track: Track,
    airport: Airport,
    *,
    screen: LandingScreen = LandingScreen(),
    metadata_lookup: StateMetadataLookup | None = None,
) -> ClassifiedTrack:
    """Assign one track to at most one runway.

    Frames are built in **HAE** because harvested altitudes are ellipsoidal, as
    broadcast. Passing MSL frames here would shift the landing screen's height test by
    the geoid undulation (~33 m) with no symptom -- the mistake the predecessor made.
    """
    points = [_track_point(s) for s in track.samples]
    assignment = assign_runway(points, airport.frames("hae"), screen=screen)
    landing_sample_index = _landing_sample_index(track, airport, assignment)
    return ClassifiedTrack(
        track=track,
        assignment=assignment,
        landing_time_utc=(
            _iso(track.samples[landing_sample_index].time_s)
            if landing_sample_index is not None
            else None
        ),
        landing_sample_index=landing_sample_index,
        observed_threshold_event=build_observed_threshold_event(
            track,
            airport.runway(assignment.runway) if assignment.runway is not None else None,
            assignment,
            metadata_lookup=metadata_lookup,
        ),
    )


def classify_tracks(
    tracks: list[Track],
    airport: Airport,
    *,
    screen: LandingScreen = LandingScreen(),
    metadata_lookup: StateMetadataLookup | None = None,
) -> list[ClassifiedTrack]:
    return [
        classify_track(
            track,
            airport,
            screen=screen,
            metadata_lookup=metadata_lookup,
        )
        for track in tracks
    ]


def _track_point(sample):
    from final_approach import TrackPoint

    return TrackPoint(lat=sample.lat, lon=sample.lon, alt_m=sample.alt_hae_m)


def _landing_sample_index(
    track: Track, airport: Airport, assignment: Assignment
) -> int | None:
    """Sample closest to the threshold on the assigned fit's final inbound pass.

    ``fit_final_segment`` records the exact source indices of the pass it fitted. Start the
    endpoint search at that fit's last sample, so an earlier overflight/go-around cannot
    become the landing identity and this code cannot drift from the assignment geometry.
    """
    if assignment.runway is None:
        return None
    if assignment.fit is None:
        raise ValueError(
            f"assigned runway {assignment.runway!r} has no final-approach fit"
        )
    runway = airport.runway(assignment.runway)
    frame = runway.frame("hae")
    final_pass_index = assignment.fit.last_sample_index
    if not 0 <= final_pass_index < len(track.samples):
        raise ValueError(
            f"assigned runway {assignment.runway!r} fit ends at invalid sample index "
            f"{final_pass_index} for a {len(track.samples)}-sample track"
        )
    return min(
        range(final_pass_index, len(track.samples)),
        key=lambda index: frame.distance_m(_track_point(track.samples[index])),
    )


def _iso(time_s: float) -> str:
    return datetime.fromtimestamp(time_s, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
