"""The FAA's minimum vectoring altitudes (MVA), a READOUT of post-training stage 2 (post-training design §3.7).

Not a mask and not a reward (user 2026-09-26): 28.5 % of the recorded tracks go more than 40 m under the chart before
they join the final, 8.4 % even before the labelled approach clearance (readouts §14) — the MVA binds radar vectors, and
a track cannot tell a vectored aircraft from one cleared for a visual approach or established on a published procedure.

The charts are the TRACONs' AIXM 5.1 files (`docs/literature/minimum_vectoring_altitude/`: where they come from, their
layout), kept as served under `repo_layout.MVA_ROOT` / `CHARTS_DATE` (git-ignored; that folder's ``download.sh`` fetches
them). The normal-operations chart (`CHART`, FUS3) is read. One ``aixm:Airspace`` per sector, one polygon each (an
exterior ring and holes, CRS84 lon/lat), its floor in ``aixm:minimumLimit`` (feet above MSL — refused otherwise),
converted once here (`geokit.FT_M`). A point in two sectors takes the higher floor; a point in none has no MVA (NaN): on
the train days every observed row before the join lies in a sector. Torch-free, numpy only.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geokit import FT_M
from ts_transformer.repo_layout import MVA_ROOT

#: The download the readouts use (`docs/literature/minimum_vectoring_altitude/README.md` §3).
CHARTS_DATE = "2026-09-26"
#: The normal-operations chart (3-mile obstacle buffer, FUSION radar); FUS5 is the degraded-mode one.
CHART = "FUS3"
#: The TRACON whose chart covers each airport (the README's "At a glance"; NCT covers KSJC and KSMF).
FACILITY = {"KMSY": "MSY", "KRDU": "RDU", "KSJC": "NCT", "KSMF": "NCT", "KSTL": "T75"}
_NS = {"aixm": "http://www.aixm.aero/schema/5.1", "gml": "http://www.opengis.net/gml/3.2"}
#: Points tested against one ring at a time (the crossing test holds points × edges booleans).
_POINTS_PER_PASS = 20_000


@dataclass(frozen=True)
class Sector:
    """One sector: its floor (m MSL), its rings (lon/lat, closed; the first the exterior), and their bounding box."""

    floor_m: float
    rings: tuple[np.ndarray, ...]
    box: tuple[float, float, float, float]             # lon min, lat min, lon max, lat max


@dataclass(frozen=True)
class MvaChart:
    facility: str
    sectors: tuple[Sector, ...]

    def at(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """The MVA at each point, m MSL: the highest floor of the sectors holding it, NaN in none."""
        lon, lat = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
        out = np.full(lon.shape, -np.inf)
        for sector in self.sectors:
            x0, y0, x1, y1 = sector.box
            near = np.flatnonzero((lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1))
            if len(near) == 0:
                continue
            exterior, *holes = sector.rings
            inside = _in_ring(exterior, lon[near], lat[near])
            for hole in holes:
                inside &= ~_in_ring(hole, lon[near], lat[near])
            hit = near[inside]
            out[hit] = np.maximum(out[hit], sector.floor_m)
        return np.where(np.isfinite(out), out, np.nan)


def _in_ring(ring: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Even-odd crossing test of points against one closed ring."""
    x0, y0, x1, y1 = ring[:-1, 0], ring[:-1, 1], ring[1:, 0], ring[1:, 1]
    out = np.zeros(len(lon), dtype=bool)
    for start in range(0, len(lon), _POINTS_PER_PASS):
        x, y = lon[start: start + _POINTS_PER_PASS, None], lat[start: start + _POINTS_PER_PASS, None]
        straddles = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        out[start: start + len(x)] = ((straddles & (x < crossing)).sum(axis=1) % 2) == 1
    return out


def _ring(element: ET.Element) -> np.ndarray:
    values = np.array(element.find(".//gml:posList", _NS).text.split(), dtype=np.float64).reshape(-1, 2)
    return values if np.array_equal(values[0], values[-1]) else np.vstack([values, values[:1]])


def load_chart(path: Path) -> MvaChart:
    """One facility's chart. Refused: a sector not in feet above MSL, or not one polygon."""
    sectors = []
    for airspace in ET.parse(path).getroot().iter(f"{{{_NS['aixm']}}}Airspace"):
        volume = airspace.find(".//aixm:AirspaceVolume", _NS)
        limit = volume.find("aixm:minimumLimit", _NS)
        reference = volume.find("aixm:minimumLimitReference", _NS).text
        if limit.get("uom") != "FT" or reference != "MSL":
            raise ValueError(f"{path.name}: a sector's floor is {limit.get('uom')} above {reference}, not FT above MSL")
        patches = volume.findall(".//gml:PolygonPatch", _NS)
        if len(patches) != 1:
            raise ValueError(f"{path.name}: a sector of {len(patches)} polygons")
        rings = (_ring(patches[0].find("gml:exterior", _NS)),
                 *(_ring(hole) for hole in patches[0].findall("gml:interior", _NS)))
        exterior = rings[0]
        sectors.append(Sector(float(limit.text) * FT_M, rings, (float(exterior[:, 0].min()), float(exterior[:, 1].min()),
                                                                 float(exterior[:, 0].max()), float(exterior[:, 1].max()))))
    if not sectors:
        raise ValueError(f"{path.name}: no sector")
    return MvaChart(path.name.split("_")[0], tuple(sectors))


def airport_charts(codes: list[str] | tuple[str, ...], root: Path = MVA_ROOT) -> dict[str, MvaChart]:
    """Each airport's chart (`FACILITY`, `CHART`), a facility's file read once."""
    read: dict[str, MvaChart] = {}
    for facility in sorted({FACILITY[code] for code in codes}):
        read[facility] = load_chart(root / CHARTS_DATE / f"{facility}_MVA_{CHART}.xml")
    return {code: read[FACILITY[code]] for code in codes}
