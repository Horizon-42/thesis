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

The landing time here is the time of the sample closest to the assigned threshold, which
is well defined precisely because ``tracks.reconstruct_tracks`` guarantees the track is
one contiguous flight -- under the old segmentation this timestamp could come from a
different pass entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from final_approach import Assignment, LandingScreen, SegmentFit, assign_runway

from flight_scenarios.identity import flight_key

from trajectory_data_process.harvest.airports import Airport
from trajectory_data_process.harvest.tracks import Track


@dataclass(frozen=True)
class ClassifiedTrack:
    """A track, its bucket, and the evidence that put it there."""

    track: Track
    assignment: Assignment
    landing_time_utc: str | None

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
) -> ClassifiedTrack:
    """Assign one track to at most one runway.

    Frames are built in **HAE** because harvested altitudes are ellipsoidal, as
    broadcast. Passing MSL frames here would shift the landing screen's height test by
    the geoid undulation (~33 m) with no symptom -- the mistake the predecessor made.
    """
    points = [_track_point(s) for s in track.samples]
    assignment = assign_runway(points, airport.frames("hae"), screen=screen)
    return ClassifiedTrack(
        track=track,
        assignment=assignment,
        landing_time_utc=_landing_time(track, airport, assignment),
    )


def classify_tracks(
    tracks: list[Track], airport: Airport, *, screen: LandingScreen = LandingScreen()
) -> list[ClassifiedTrack]:
    return [classify_track(t, airport, screen=screen) for t in tracks]


def _track_point(sample):
    from final_approach import TrackPoint

    return TrackPoint(lat=sample.lat, lon=sample.lon, alt_m=sample.alt_hae_m)


def _landing_time(track: Track, airport: Airport, assignment: Assignment) -> str | None:
    """When the aircraft passed closest to the runway it was assigned to."""
    if assignment.runway is None:
        return None
    frame = airport.runway(assignment.runway).frame("hae")
    closest = min(track.samples, key=lambda s: frame.distance_m(_track_point(s)))
    return _iso(closest.time_s)


def _iso(time_s: float) -> str:
    return datetime.fromtimestamp(time_s, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


