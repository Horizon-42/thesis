# Minimum Vectoring Altitude (MVA) charts for KMSY, KRDU, KSJC, KSMF and KSTL (downloaded 2026-09-26)

This folder records where the FAA's published minimum vectoring altitudes for the five thesis airports
came from, what each file says about its own date, and how the AIXM 5.1 XML is laid out, so that a parser
can be written against it.

How to read it:
- **Quotes are copied** from the saved FAA pages and files, each under about 30 words; "…" marks a cut.
- **"(reading)"** marks a step the source does not state, such as a conclusion drawn from file contents
  or from matching a chart name to a regulation paragraph.
- The downloaded files are **not tracked in git**. They live in `data/MVA/2026-09-26/` (the `data/` tree
  is git-ignored). `./download.sh [DATE]` fetches the same 16 files into `data/MVA/<DATE>/`, refuses to
  write into a date directory that already exists, and prints the SHA-256 of every file.
- The files are kept **byte for byte as served**. The FAA says its certification lapses once a file is
  modified (§1).

## At a glance

| item | state (2026-09-26) |
|---|---|
| Facilities | MSY New Orleans (KMSY), RDU Raleigh (KRDU), NCT Northern California (KSJC **and** KSMF, both inside one file per chart), T75 St. Louis (KSTL) |
| Files | 8 AIXM 5.1 XML + 8 PDF: two charts per facility, **FUS3** and **FUS5**, both listed as current on the FAA page (§2) |
| Which chart to use | **FUS3** is the normal-operations chart (3-mile obstacle buffer, FUSION radar mode); **FUS5** is the degraded-mode / multi-sensor chart (5-mile buffer). Both were downloaded (§2, reading) |
| Effective date | **No file states one.** Each carries only a project name with a year (e.g. `RDU_MVA_FUS3_2025`); every AIXM sector's `validTime` starts at `1980-01-01T14:00:00Z`, a placeholder (§3) |
| Altitudes | Feet above **MSL** (`uom="FT"`, `minimumLimitReference` = `MSL`). The thesis's ADS-B heights are ellipsoidal, so they must be converted before being compared with these values (reading, see the root `CLAUDE.md` "Vertical datum") |
| PDF vs XML | Every sector label printed on the 8 PDFs matches the XML's name and altitude. **Exception: the NCT FUS3 PDF is the 2023 project and its XML the 2025 one; 9 XML sectors carry no label on the PDF**, 3 of them within 12 km of KSJC (§5) |
| Parser notes | One `aixm:Airspace` per sector, one polygon each (exterior + holes), lon/lat order (CRS84), altitude in `aixm:minimumLimit`. Namespace prefixes differ between files, so match on namespace URIs (§4) |

## 1. What an MVA chart is, and the FAA's usage notes

The MVA page itself
(<https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/mva_mia/mva/>, "Page last
modified: October 30, 2025") is only an alphabetical list of facilities and links. It contains no
definition. The definition below comes from the FAA Pilot/Controller Glossary (P/CG, effective 7/9/26,
Change 3), <https://www.faa.gov/air_traffic/publications/atpubs/pcg_html/glossary-m.html>:

> MINIMUM VECTORING ALTITUDE (MVA)- The lowest MSL altitude at which an IFR aircraft will be vectored by a
> radar controller, …

The same entry goes on: "The altitude meets IFR obstacle clearance criteria." Each XML sector repeats the
term in its `gml:description`: `MINIMUM VECTORING ALTITUDE (MVA)`.

Usage notes, verbatim, from the parent page
<https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/mva_mia/> ("Page last modified:
September 05, 2024"):

> These charts are geo-referenced but are not to be used for navigation.

> FAA Disclaimer These MVA/MIA map files are to be used as a visual reference; they are not to be used as a
> navigational tool.

> MVA/MIA charts are not updated on a regularly scheduled cycle. Any person using these files is
> responsible for checking currency dates on these individual MVA/MIA map files.

> Once the files are retrieved, if they are modified, the certification of these map files is no longer
> valid.

Page 1 of every PDF is a scanned FAA memorandum dated **December 8, 2014** ("MVA/MIA Release To Public
Disclaimer for MVA/MIA Data"), with the same five points. The scan is identical in all 8 PDFs (the
embedded JPEG has the same MD5, `cbd376d2…`). Page 2 of every chart is stamped "NOT FOR NAVIGATION".

## 2. Which files, and why both FUS3 and FUS5

The MVA page lists **exactly two entries for each of the four facilities**, FUS3 and FUS5, each offered
as PDF, XML and KML:

| facility (as the page names it) | airports | FUS3 files | FUS5 files |
|---|---|---|---|
| MSY, New Orleans Moissant ATCT/TRACON | KMSY | `MSY_MVA_FUS3.xml`, `.pdf` | `MSY_MVA_FUS5.xml`, `.pdf` |
| RDU, Raleigh ATCT/TRACON | KRDU | `RDU_MVA_FUS3.xml`, `.pdf` | `RDU_MVA_FUS5.xml`, `.pdf` |
| NCT, Northern California TRACON | KSJC, KSMF | `NCT_MVA_FUS3.xml`, `.pdf` | `NCT_MVA_FUS5.xml`, `.pdf` |
| T75, St. Louis TRACON | KSTL | `T75_MVA_FUS3.xml`, `.pdf` | `T75_MVA_FUS5.xml`, `.pdf` |

- **No other versions exist for these facilities.** The server directory listings
  (<https://aeronav.faa.gov/MVA_Charts/aixm/>, `…/pdf/`, `…/kmz/`) hold only these names for MSY, RDU, NCT
  and T75. The year and `_v1` suffixes appear only *inside* the files, as the project name (§3). Other
  facilities do have year-suffixed files, e.g. `SCT_MVA_FUS3_2025.pdf`.
- **FUS3 and FUS5 are two charts in force at the same time, not two versions of one chart.** FAA Order
  JO 7210.3EE (Change 3, effective 7/9/2026), para 3-8-1,
  <https://www.faa.gov/air_traffic/publications/atpubs/foa_html/chap3_section_8.html>:

  > Where the system is utilizing FUSION mode, develop an MVAC that provides: 3‐mile separation minima from
  > obstacles, …

  > … 5‐mile separation minima from obstacles for use whenever the FUSION system cannot provide 3‐mile
  > separation due to degraded status or system limitations, or when using Multi‐Sensor Mode.

  Reading: FUS3 = "FUSION, 3-mile" and FUS5 = "FUSION, 5-mile". The FAA page does not spell out the
  abbreviation. So FUS3 is the chart controllers use in normal operation, and FUS5 is the fallback. Both
  are downloaded; a consumer should state which one it uses.
- **NCT: one file per chart covers both airports.** A point-in-polygon test of the airport reference points
  (`aeroviz-4d/public/data/airports/<ICAO>/airport.json`) finds both KSJC and KSMF inside `NCT_MVA_FUS3.xml`
  and inside `NCT_MVA_FUS5.xml`. FUS3 sector names carry an area prefix (`BAB`, `MCC`, `MCE`, `MRY`, `OAK`,
  `RDD`, `RNO`, `SCK`); FUS5 names do not (`A1` … `Z2`).
- The KML files were not downloaded (the XML carries the same data in a documented schema).

Sector containing each airport reference point (from the XML):

| airport | FUS3 sector, MVA | FUS5 sector, MVA |
|---|---|---|
| KMSY | `F`, 1,500 ft | `F`, 1,600 ft |
| KRDU | `A`, 2,000 ft | `A`, 2,000 ft |
| KSJC | `OAK_N`, 2,000 ft | `M1`, **4,500 ft** |
| KSMF | `MCC_B`, 1,700 ft | `G1`, 1,900 ft |
| KSTL | `A`, 2,000 ft | `A`, 2,000 ft |

## 3. Files, dates and checksums

Downloaded 2026-09-26 (about 15:36 UTC) with `curl -L -A "Mozilla/5.0"` from `https://aeronav.faa.gov/MVA_Charts/aixm/<file>` (XML)
and `https://aeronav.faa.gov/MVA_Charts/pdf/<file>` (PDF). Sizes equal the server's `Content-Length`.

**No file states an effective or currency date.** Three date-like markers exist, none of them an
effective date:
1. The **project name**, which carries a year. In the XML it is the message-level `gml:description`
   (e.g. `<ns1:description ns2:type="simple">RDU_MVA_FUS3_2025</ns1:description>`). On the PDF it is the
   box on page 2 headed "SDAT MVA PROJECT NAME:".
2. The PDF's embedded creation date (`pdfinfo`), and the server's `Last-Modified` header.
3. Every AIXM sector's `gml:validTime` begins `1980-01-01T14:00:00Z` and ends `indeterminatePosition="unknown"`.
   This is the same value in all 8 files, so it is a placeholder (reading).

| file | bytes | SHA-256 | project name stated in the file | PDF created (UTC) | server Last-Modified (UTC) |
|---|---:|---|---|---|---|
| `MSY_MVA_FUS3.xml` | 94,230 | `3c625cb04eb22afd5d316b34510120261924090280f0a7b2771a3c76df1af9fb` | `MSY_MVA_FUS3_2024` | — | 2024-10-31 17:41 |
| `MSY_MVA_FUS3.pdf` | 609,902 | `b5d8a21e67e9c8ee9d06384fe9c25fe67335f440f61a6d9de88f4878eaba6258` | `MSY_MVA_FUS3_2024` | 2024-10-17 | 2024-12-09 14:22 |
| `MSY_MVA_FUS5.xml` | 81,354 | `f79d229ba7a3510e224d1787b995ee3ee2821aa171543b860e612fb3c10f5081` | `MSY_MVA_FUS5_2024` | — | 2024-11-02 04:02 |
| `MSY_MVA_FUS5.pdf` | 611,407 | `292fb21fa86ca69408ccc1ce357e5c5d29a2edd4258a307ad1f608c24b243d83` | `MSY_MVA_FUS5_2024_SHAPE` | 2024-10-17 | 2024-12-09 14:22 |
| `RDU_MVA_FUS3.xml` | 108,960 | `67f0cd540c814c21cdd29a4f775f8abff62f79bcdf358bdb0d534d7ca348dc95` | `RDU_MVA_FUS3_2025` | — | 2025-06-04 14:23 |
| `RDU_MVA_FUS3.pdf` | 619,289 | `c3f04e08b2a2130b37160403b2f64fd242e6355208380659057b452c69863e3c` | `RDU_MVA_FUS3_2025` | 2025-05-09 | 2025-06-04 14:31 |
| `RDU_MVA_FUS5.xml` | 101,364 | `1533855e2145b1017d59da30bb1b01be90e4dadd38224506bed917a5c697e8dc` | `RDU_MVA_FUS5_2025` | — | 2025-06-04 14:23 |
| `RDU_MVA_FUS5.pdf` | 620,029 | `72a38436b87f631b73d5b8c72e7e041715191331723f3d9e0e610459cc9851ba` | `RDU_MVA_FUS5_2025` | 2025-05-09 | 2025-06-04 14:31 |
| `NCT_MVA_FUS3.xml` | 733,195 | `3777ce91224417b589959b25b4cd3d7c89eed7718283104c52176984fb7f65bf` | `NCT_MVA_FUS3_2025` | — | 2025-10-20 18:57 |
| `NCT_MVA_FUS3.pdf` | 725,992 | `e8f3b89a3bbf27771115736ebe5474dd8d11003454823897bbd796f3c1367631` | **`NCT_MVA_FUS3_2023`** | 2023-10-03 | 2023-11-30 19:13 |
| `NCT_MVA_FUS5.xml` | 256,819 | `4294a8d881040c0175ce78a8edef70b477b798acf331aa8cc838555f0c08286b` | `NCT_MVA_FUS5_2025` | — | 2025-10-20 18:57 |
| `NCT_MVA_FUS5.pdf` | 523,326 | `42e32bcbc2e173f4b38326918c847bc68be9bbe611dcd04c9cd121335940b2b7` | **`NCT_MVA_FUS5_2024`** | 2024-08-28 | 2024-12-09 14:22 |
| `T75_MVA_FUS3.xml` | 207,970 | `23ecf30c00e5f377e09bba232c1b6dceae1010612ecd385e6938135a2c9075ef` | `T75_MVA_FUS3_2024_v1` | — | 2024-11-04 20:07 |
| `T75_MVA_FUS3.pdf` | 636,791 | `e1a61d95ea64f3966feb9ce463875a2381d3c82107b5ce5cea970ebfc6e09179` | `T75_MVA_FUS3_2024_v1` | 2024-10-10 | 2024-12-09 14:23 |
| `T75_MVA_FUS5.xml` | 216,200 | `29af11792b6ccc8c31498fa9cdd1beb7be442f8c76e2d3623bdc0c1967373e70` | `T75_MVA_FUS5_2024_v1` | — | 2024-11-04 20:07 |
| `T75_MVA_FUS5.pdf` | 646,293 | `9cb8b3c33183da2a84aec51b2964ffbbc28aa268965dbaaa763bbc50e1b4d8f5` | `T75_MVA_FUS5_2024_v1` | 2024-10-10 | 2024-12-09 14:23 |

The FAA re-uses a file name when it revises a chart. To see whether a chart has changed since
2026-09-26, run `./download.sh <new date>` and compare the SHA-256 values with this table.

## 4. AIXM 5.1 layout (for a parser)

All 8 XML files have the same structure. The only difference is which prefix is bound to which namespace.

**Namespaces.** Prefixes are generated (`ns1` … `ns8`), and the RDU files swap `ns5`/`ns6` relative to
the other six files. **A parser must match on URIs, never on prefixes.**

| URI | prefix in MSY/NCT/T75 | in RDU | used for |
|---|---|---|---|
| `http://www.aixm.aero/schema/5.1/message` | `ns8` | `ns8` | `AIXMBasicMessage`, `hasMember` |
| `http://www.aixm.aero/schema/5.1` | `ns3` | `ns3` | `Airspace`, `AirspaceTimeSlice`, `AirspaceVolume`, `minimumLimit`, `name`, **`Surface`** |
| `http://www.opengis.net/gml/3.2` | `ns1` | `ns1` | `description`, `id`, `validTime`, `patches`, `PolygonPatch`, `exterior`/`interior`, `LinearRing`, `posList` |
| `http://www.w3.org/1999/xlink` | `ns2` | `ns2` | `type="simple"` attributes |
| `http://www.isotc211.org/2005/gco` | `ns4` | `ns4` | declared, unused |
| `http://www.isotc211.org/2005/gmd` | `ns5` | `ns6` | declared, unused |
| `http://www.isotc211.org/2005/gmx` | `ns6` | `ns5` | declared, unused |
| `http://www.isotc211.org/2005/gts` | `ns7` | `ns7` | declared, unused |
| `http://www.w3.org/2001/XMLSchema-instance` | `xsi` | `xsi` | declared inline on each `xsi:nil="true"` element |

**Path to a sector.** The root is `message:AIXMBasicMessage`. Its `gml:description` holds the project name,
and `aixm:messageMetadata` is empty. Under it, one `message:hasMember` per sector, each holding exactly:

```
aixm:Airspace  @gml:id (numeric string, unique in the file)
 └ gml:description = "MINIMUM VECTORING ALTITUDE (MVA)"
 └ aixm:timeSlice/aixm:AirspaceTimeSlice            (exactly 1)
    ├ gml:validTime/gml:TimePeriod                   begin 1980-01-01T14:00:00Z, end unknown (placeholder)
    ├ aixm:interpretation = BASELINE, aixm:type = OTHER, aixm:designator = nil
    ├ aixm:name                                      sector label, unique in the file, = the label on the PDF
    └ aixm:geometryComponent/aixm:AirspaceGeometryComponent/aixm:theAirspaceVolume/aixm:AirspaceVolume   (exactly 1)
       ├ aixm:minimumLimit  @uom="FT"                the MVA, integer feet, always a multiple of 100
       ├ aixm:minimumLimitReference = MSL
       └ aixm:horizontalProjection/aixm:Surface  @srsName="urn:ogc:def:crs:OGC:1.3:CRS84"
          └ gml:patches/gml:PolygonPatch @interpolation="planar"   (exactly 1)
             ├ gml:exterior/gml:LinearRing/gml:posList            (exactly 1)
             └ gml:interior/gml:LinearRing/gml:posList            (0 or more holes)
```

Things to note:
- **`Surface` is in the AIXM namespace** (`aixm:Surface`), not `gml:Surface`. Everything inside it is GML.
- **Coordinate order is longitude, latitude**, because CRS84 is longitude-first (checked: the values fall
  at the facilities' longitudes and latitudes). `posList` has no `srsDimension` attribute; it is a flat list
  of 2-D pairs, at most 5 decimals (about 1 m). Every ring is closed (first point = last point). There
  are no arc elements; curved boundaries are densified into straight segments (many rings have 130–131
  points).
- **No upper limit** in any file (`maximumLimit`, `upperLimit` absent).
- **Ring orientation is mixed** (clockwise and counter-clockwise exteriors in the same file). Whether a ring is
  an exterior or a hole comes from the element (`exterior`/`interior`), never from the winding.
- **Holes are real and nested.** Every hole of every sector is completely covered by other sectors, e.g.
  RDU FUS3 sector `A` (2,000 ft) has three holes that are sectors with a different MVA.
- **Sectors tile the chart, but neighbouring boundaries do not share vertices exactly.** Along the
  boundaries there are thin overlaps and gaps. Per file, the overlaps add up to 0.03–1.7 km² and the gaps
  number 213–1,228, the largest 0.001–0.1 km². Against a covered area of 39,000–356,000 km², that is
  about 1e-5 of the area or less. A point lookup near a boundary can therefore hit zero or two sectors,
  and the parser has to choose a rule for both cases.
- Two files (NCT FUS3, RDU FUS3) each contain 2 consecutive duplicate vertices.

Per file (all polygons valid under shapely 2.1.2):

| file | sectors | rings (holes) | MVA min – max (ft MSL) | lon range | lat range | covered area (km²) |
|---|---:|---|---|---|---|---:|
| `MSY_MVA_FUS3.xml` | 11 | 15 (4) | 1,500 – 2,800 | −92.18 … −88.68 | 28.33 … 31.00 | 67,218 |
| `MSY_MVA_FUS5.xml` | 11 | 13 (2) | 1,600 – 3,000 | −92.96 … −84.95 | 26.16 … 32.40 | 355,546 |
| `RDU_MVA_FUS3.xml` | 14 | 21 (7) | 2,000 – 3,600 | −80.01 … −77.54 | 34.88 … 36.88 | 38,821 |
| `RDU_MVA_FUS5.xml` | 14 | 17 (3) | 2,000 – 3,600 | −80.01 … −77.54 | 34.88 … 36.88 | 38,822 |
| `NCT_MVA_FUS3.xml` | 150 | 162 (12) | 1,500 – 14,000 | −123.50 … −118.49 | 35.70 … 41.03 | 163,194 |
| `NCT_MVA_FUS5.xml` | 59 | 60 (1) | 1,800 – 16,900 | −124.12 … −117.99 | 35.33 … 40.86 | 315,081 |
| `T75_MVA_FUS3.xml` | 27 | 39 (12) | 2,000 – 3,100 | −91.69 … −88.37 | 37.55 … 41.64 | 97,210 |
| `T75_MVA_FUS5.xml` | 29 | 38 (9) | 2,000 – 3,100 | −91.70 … −88.37 | 37.55 … 41.65 | 97,267 |

One complete sector, RDU FUS3 sector `A` (the one containing KRDU), with each `posList` cut to its first two
points and its last point:

```xml
<ns8:hasMember ns2:type="simple">
    <ns3:Airspace ns1:id="19627567">
        <ns1:description ns2:type="simple">MINIMUM VECTORING ALTITUDE (MVA)</ns1:description>
        <ns1:boundedBy xsi:nil="true" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"/>
        <ns3:timeSlice>
            <ns3:AirspaceTimeSlice>
                <ns1:validTime ns2:type="simple">
                    <ns1:TimePeriod>
                        <ns1:beginPosition>1980-01-01T14:00:00Z</ns1:beginPosition>
                        <ns1:endPosition indeterminatePosition="unknown"/>
                    </ns1:TimePeriod>
                </ns1:validTime>
                <ns3:interpretation>BASELINE</ns3:interpretation>
                <ns3:sequenceNumber>1</ns3:sequenceNumber>
                <ns3:type>OTHER</ns3:type>
                <ns3:designator xsi:nil="true" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"/>
                <ns3:localType xsi:nil="true" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"/>
                <ns3:name>A</ns3:name>
                <ns3:designatorICAO xsi:nil="true" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"/>
                <ns3:controlType xsi:nil="true" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"/>
                <ns3:upperLowerSeparation xsi:nil="true" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"/>
                <ns3:protectedRoute xsi:nil="true" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"/>
                <ns3:geometryComponent>
                    <ns3:AirspaceGeometryComponent>
                        <ns3:operation xsi:nil="true" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"/>
                        <ns3:operationSequence xsi:nil="true" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"/>
                        <ns3:theAirspaceVolume>
                            <ns3:AirspaceVolume>
                                <ns3:minimumLimit uom="FT">2000</ns3:minimumLimit>
                                <ns3:minimumLimitReference>MSL</ns3:minimumLimitReference>
                                <ns3:horizontalProjection>
                                    <ns3:Surface srsName="urn:ogc:def:crs:OGC:1.3:CRS84" ns1:id="53e228a7-9be4-46f3-87a1-e254be090ba0">
                                        <ns1:patches>
                                            <ns1:PolygonPatch interpolation="planar">
                                                <ns1:exterior>
                                                    <ns1:LinearRing>
                                                        <ns1:posList>-79.16349 35.9898 -79.16913 35.97598 ... -79.16349 35.9898</ns1:posList>  <!-- 389 points -->
                                                    </ns1:LinearRing>
                                                </ns1:exterior>
                                                <ns1:interior>
                                                    <ns1:LinearRing>
                                                        <ns1:posList>-79.02738 36.10379 -79.02731 36.10625 ... -79.02738 36.10379</ns1:posList>  <!-- 131 points -->
                                                    </ns1:LinearRing>
                                                </ns1:interior>
                                                <ns1:interior>
                                                    <ns1:LinearRing>
                                                        <ns1:posList>-78.47889 35.89032 -78.48193 35.89039 ... -78.47889 35.89032</ns1:posList>  <!-- 131 points -->
                                                    </ns1:LinearRing>
                                                </ns1:interior>
                                                <ns1:interior>
                                                    <ns1:LinearRing>
                                                        <ns1:posList>-78.7569 35.72725 -78.75435 35.72866 ... -78.7569 35.72725</ns1:posList>  <!-- 210 points -->
                                                    </ns1:LinearRing>
                                                </ns1:interior>
                                            </ns1:PolygonPatch>
                                        </ns1:patches>
                                    </ns3:Surface>
                                </ns3:horizontalProjection>
                            </ns3:AirspaceVolume>
                        </ns3:theAirspaceVolume>
                    </ns3:AirspaceGeometryComponent>
                </ns3:geometryComponent>
            </ns3:AirspaceTimeSlice>
        </ns3:timeSlice>
    </ns3:Airspace>
</ns8:hasMember>
```

## 5. PDF charts, and how they compare with the XML

Each PDF has two pages. Page 1 (letter size) is the 2014 disclaimer memorandum (§1). Page 2 (a 24 × 24 inch
sheet) is the chart, drawn as vectors: sector outlines, each sector's name, and its MVA **in hundreds of feet** (`20` =
2,000 ft). Around them are the "SDAT MVA PROJECT NAME:" box, "NOT FOR NAVIGATION" and "Plot Generated by
Radar Video Mapping Team Federal Aviation Administration". There is no base map, scale or date.

Check done: each name on page 2 was paired with the number printed just below it (word positions from
`pdftotext -bbox`), and the pair was compared with the XML's `aixm:name` and `aixm:minimumLimit`.

| chart | labels on the PDF | agree with the XML | XML sectors with no PDF label |
|---|---:|---:|---|
| MSY FUS3 / FUS5 | 11 / 11 | 11 / 11 | none |
| RDU FUS3 / FUS5 | 14 / 14 | 14 / 14 | none |
| NCT FUS3 | 141 | 141 | **9**: `MRY_BB` 3,200, `MRY_Y` 4,700, `OAK_D` 2,700, `OAK_G` 1,900, `OAK_II` 3,700, `OAK_P` 2,500, `OAK_S` 2,300, `OAK_V` 4,000, `OAK_Z` 3,100 ft |
| NCT FUS5 | 59 | 59 | none |
| T75 FUS3 / FUS5 | 27 / 29 | 27 / 29 | none |

- The NCT FUS3 PDF is the **2023** project and was last uploaded 2023-11-30; its XML is the **2025** project
  (uploaded 2025-10-20). `OAK_P` (11 km²), `OAK_V` and `OAK_S` lie 4.7, 8.4 and 11.2 km from KSJC. Whether
  the 9 sectors are new since 2023 or were simply left unlabelled on the plot cannot be told from the files.
  **Treat the XML as the current data and the NCT FUS3 PDF as an older picture** (reading: the XML is two
  years newer).
- The NCT FUS5 PDF is the 2024 project and its XML the 2025 one. All 59 names and altitudes agree; the
  outlines were not compared.
- The MSY FUS5 PDF names its project `MSY_MVA_FUS5_2024_SHAPE`, the XML `MSY_MVA_FUS5_2024`. All 11 labels
  agree.

## 6. Not verified

- **The effective date of any chart.** None is published in the files or on the page (§3). The FAA puts the
  duty to check currency on the user (§1).
- **That FUS3/FUS5 stands for "FUSION 3-mile / 5-mile".** This is the reading in §2. JO 7210.3EE 3-8-1
  describes the two charts but does not use the file suffixes.
- **Which radar mode each facility runs day to day**, i.e. how often the FUS5 chart is the one in force.
- The polygon outlines on the PDFs were not compared with the XML geometry; only names and altitudes were.
