"""The published approach minima a threshold is judged against.

WHY THIS IS NOT READ FROM THE CIFP
----------------------------------
The CIFP carries no minima. Its own readme lists every record type it publishes and
there is no minima record among them; the entire approach section has exactly one kind
of continuation record, application type ``W`` (Level of Service), and that record says
WHICH service is authorised -- LNAV, LP, LNAV/VNAV, LPV -- never at what height. The
Path Point record (``harvest/cifp.py``) is geometry only. A decision altitude is a
charting product: it is published on the approach plate and nowhere else.

So it is read once off the plates in ``data/RNAV_CHARTS/`` by
``trajectory_data_process/extract_approach_minima.py`` and stored in
``config/runway_thresholds.json``. Nothing parses a PDF at harvest or training time.

WHAT A PLATE PRINTS, AND WHAT IT DOES NOT
-----------------------------------------
A plate prints the decision altitude in feet MSL and, beside it, the height that
altitude sits above the TOUCHDOWN ZONE elevation -- not above the landing threshold.
The two differ by the runway's first 3000 ft of slope: across this fleet the landing
threshold is 0.0-23.9 ft BELOW the touchdown zone (largest KSTL 29), so a consumer that
reaches for the plate's height where it wants the threshold's is wrong by up to 7.3 m.
The field is named ``decision_height_above_touchdown_ft`` for that reason, and all three
printed figures are stored unconverted, exactly as printed.

The height above the LANDING threshold -- the quantity a trajectory is actually measured
in -- is NOT stored. It is derived at read time by
``Runway.decision_height_above_threshold_m``, because it depends on which threshold
elevation the reader holds, and for an LPV runway ``load_airport`` replaces the
configured elevation with the Path Point's LTP.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: ``runway_thresholds.json`` carries ``published_minima`` on every threshold from v3
#: on. The label lives here, with the schema it labels, and both the extractor that
#: writes the file and ``load_airport`` that reads it import it -- a version restated in
#: a second place is a version neither place can check.
RUNWAY_THRESHOLDS_SCHEMA = "runway-thresholds-v3"

#: Vertically guided services, in the order the extractor prefers them -- which is the
#: order they are flown, NOT the order of their heights (KMSY 02 publishes an LNAV/VNAV
#: DA of 379 ft, LOWER than its LPV DA of 398 ft). AIM 5-4-5 f 5 puts the missed approach
#: point of ANY vertically guided approach at the decision altitude, so Baro-VNAV counts:
#: two runways in this fleet (KRDU 32, KSMF 35R) publish LNAV/VNAV minima and no LPV.
VERTICALLY_GUIDED = ("lpv", "lnav_vnav")
#: No vertically guided line of minima. Either the plate publishes an LNAV **MDA** only
#: (a fix is the missed approach point, not a height), or there is no instrument
#: approach to that end at all (KRDU 14). ``note`` says which.
NO_VERTICAL_MINIMA = "none"
SERVICES = VERTICALLY_GUIDED + (NO_VERTICAL_MINIMA,)


@dataclass(frozen=True)
class PublishedMinima:
    """One threshold's published vertically guided minima, as the plate prints them."""

    service: str
    decision_altitude_ft_msl: float | None
    decision_height_above_touchdown_ft: float | None
    touchdown_zone_elevation_ft: float | None
    chart: str | None
    procedure: str | None
    amendment: str | None
    #: The plate's own validity band ("16 APR 2026 to 14 MAY 2026"). A decision altitude
    #: is amended on a 28-day cycle, so a verdict graded against one has to name which
    #: plate it was graded against -- the file name alone does not move when the FAA
    #: reissues it.
    chart_effective: str | None
    note: str | None

    def __post_init__(self) -> None:
        if self.service not in SERVICES:
            raise ValueError(f"unknown minima service {self.service!r}, expected one of {SERVICES}")
        published = (
            self.decision_altitude_ft_msl,
            self.decision_height_above_touchdown_ft,
            self.touchdown_zone_elevation_ft,
            self.chart,
            self.procedure,
            self.amendment,
            self.chart_effective,
        )
        if self.service == NO_VERTICAL_MINIMA:
            if any(field is not None for field in published):
                raise ValueError("a threshold without vertical minima carries no plate figures")
            if not (self.note or "").strip():
                raise ValueError("a threshold without vertical minima must say why")
            return
        if any(field is None for field in published):
            raise ValueError(f"{self.service} minima are incomplete: {published}")
        # The three figures come off three different parts of the sheet and must agree.
        # This is a CLASS INVARIANT, not the pipeline's check: the extractor already
        # refuses a plate whose figures disagree (``read_plate``), so what this catches
        # is a hand-built object -- a fixture, or a config edited by hand.
        residual = self.decision_altitude_ft_msl - self.decision_height_above_touchdown_ft
        if abs(residual - self.touchdown_zone_elevation_ft) > 0.5:
            raise ValueError(
                f"{self.procedure}: DA {self.decision_altitude_ft_msl} - height above "
                f"touchdown {self.decision_height_above_touchdown_ft} = {residual}, which "
                f"is not the plate's TDZE {self.touchdown_zone_elevation_ft}"
            )

    @property
    def vertically_guided(self) -> bool:
        return self.service in VERTICALLY_GUIDED

    @classmethod
    def from_config(cls, payload: dict) -> PublishedMinima:
        """Build from a ``runway_thresholds.json`` block. Every key is required."""
        return cls(
            service=payload["service"],
            decision_altitude_ft_msl=payload["decision_altitude_ft_msl"],
            decision_height_above_touchdown_ft=payload["decision_height_above_touchdown_ft"],
            touchdown_zone_elevation_ft=payload["touchdown_zone_elevation_ft"],
            chart=payload["chart"],
            procedure=payload["procedure"],
            amendment=payload["amendment"],
            chart_effective=payload["chart_effective"],
            note=payload["note"],
        )

    def to_config(self) -> dict:
        return asdict(self)


def no_vertical_minima(note: str) -> PublishedMinima:
    """The explicit "this threshold publishes no decision altitude, and here is why"."""
    return PublishedMinima(
        service=NO_VERTICAL_MINIMA,
        decision_altitude_ft_msl=None,
        decision_height_above_touchdown_ft=None,
        touchdown_zone_elevation_ft=None,
        chart=None,
        procedure=None,
        amendment=None,
        chart_effective=None,
        note=note,
    )


__all__ = [
    "NO_VERTICAL_MINIMA",
    "RUNWAY_THRESHOLDS_SCHEMA",
    "PublishedMinima",
    "SERVICES",
    "VERTICALLY_GUIDED",
    "no_vertical_minima",
]
