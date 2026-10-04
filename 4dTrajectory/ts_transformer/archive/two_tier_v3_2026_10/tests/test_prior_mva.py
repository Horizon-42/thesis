"""The FAA minimum vectoring altitude charts (`prior.mva`, post-training design §3.7): the AIXM sectors read, the floor at
a point (the higher of two, none outside and in a hole), the units refused. Synthetic files only: the downloaded charts
are git-ignored."""

from __future__ import annotations

import numpy as np
import pytest

from geokit import FT_M
from ts_transformer.prior.mva import FACILITY, load_chart

_HEAD = ('<?xml version="1.0"?><message:AIXMBasicMessage xmlns:message="http://www.aixm.aero/schema/5.1/message" '
         'xmlns:aixm="http://www.aixm.aero/schema/5.1" xmlns:gml="http://www.opengis.net/gml/3.2">')


def _sector(floor: str, exterior: list[tuple[float, float]], holes: list[list[tuple[float, float]]] = (),
            uom: str = "FT", reference: str = "MSL") -> str:
    def ring(points):
        return " ".join(f"{lon} {lat}" for lon, lat in points)
    interiors = "".join(f"<gml:interior><gml:LinearRing><gml:posList>{ring(h)}</gml:posList></gml:LinearRing>"
                        f"</gml:interior>" for h in holes)
    return ('<message:hasMember><aixm:Airspace><aixm:timeSlice><aixm:AirspaceTimeSlice><aixm:geometryComponent>'
            '<aixm:AirspaceGeometryComponent><aixm:theAirspaceVolume><aixm:AirspaceVolume>'
            f'<aixm:minimumLimit uom="{uom}">{floor}</aixm:minimumLimit>'
            f'<aixm:minimumLimitReference>{reference}</aixm:minimumLimitReference>'
            '<aixm:horizontalProjection><aixm:Surface><gml:patches><gml:PolygonPatch>'
            f'<gml:exterior><gml:LinearRing><gml:posList>{ring(exterior)}</gml:posList></gml:LinearRing></gml:exterior>'
            f'{interiors}</gml:PolygonPatch></gml:patches></aixm:Surface></aixm:horizontalProjection>'
            '</aixm:AirspaceVolume></aixm:theAirspaceVolume></aixm:AirspaceGeometryComponent></aixm:geometryComponent>'
            '</aixm:AirspaceTimeSlice></aixm:timeSlice></aixm:Airspace></message:hasMember>')


def _square(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def test_a_point_takes_the_highest_sector_holding_it_none_outside_or_in_a_hole(tmp_path):
    path = tmp_path / "ABC_MVA_FUS3.xml"
    path.write_text(_HEAD + _sector("3000", _square(0.0, 0.0, 2.0, 2.0), [_square(0.2, 0.2, 0.4, 0.4)])
                    + _sector("4000", _square(1.5, 1.5, 3.0, 3.0)) + "</message:AIXMBasicMessage>")
    chart = load_chart(path)
    assert chart.facility == "ABC" and len(chart.sectors) == 2
    lon = np.array([1.0, 1.8, 2.5, 0.3, 5.0])
    lat = np.array([1.0, 1.8, 2.5, 0.3, 5.0])
    got = chart.at(lon, lat)
    assert got[:3] == pytest.approx([3000 * FT_M, 4000 * FT_M, 4000 * FT_M])     # one, both (the higher), the other
    assert np.isnan(got[3]) and np.isnan(got[4])                                # in the hole; off the chart


def test_a_sector_not_in_feet_above_msl_is_refused(tmp_path):
    for uom, reference in (("M", "MSL"), ("FT", "SFC")):
        path = tmp_path / f"{uom}{reference}_MVA_FUS3.xml"
        path.write_text(_HEAD + _sector("3000", _square(0.0, 0.0, 1.0, 1.0), uom=uom, reference=reference)
                        + "</message:AIXMBasicMessage>")
        with pytest.raises(ValueError, match="not FT above MSL"):
            load_chart(path)


def test_every_thesis_airport_has_a_facility():
    assert set(FACILITY) == {"KMSY", "KRDU", "KSJC", "KSMF", "KSTL"}
