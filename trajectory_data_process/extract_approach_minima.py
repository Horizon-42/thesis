#!/usr/bin/env python3
"""Read the published approach minima off the FAA plates into runway_thresholds.json.

The CIFP publishes no minima at all (the reasoning is in
``harvest/approach_minima.py``), so the decision altitude has to come from the approach
plate. The plates are already on disk -- ``data/RNAV_CHARTS/<ICAO>/*.PDF``, the d-TPP
RNAV (GPS) and RNAV (RNP) approaches for the five airports. This script reads them ONCE
and writes what they print into the airport configuration, so that no part of the
harvest, the optimizer or the training pipeline ever parses a PDF.

Only the RNAV (GPS) plates are read. The RNAV (RNP) Z plates publish RNP AR minima, a
different service on a different authorisation, which this fleet is not flying.

    python trajectory_data_process/extract_approach_minima.py [--dry-run]

Re-run it when the chart cycle changes -- and only then.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

if __package__ is None or __package__ == "":  # pragma: no cover - direct execution.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory_data_process.harvest.approach_minima import (
    PublishedMinima,
    RUNWAY_THRESHOLDS_SCHEMA,
    VERTICALLY_GUIDED,
    no_vertical_minima,
)
DEFAULT_CHART_ROOT = Path(__file__).resolve().parents[1] / "data" / "RNAV_CHARTS"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "runway_thresholds.json"

# A plate's title names its own service and runway: "RNAV (GPS) Y RWY 30L". The
# designator letter only separates two procedures to the same runway; it is not part of
# the runway's identity. A plate prints its title more than once and they must agree --
# a title read out of a NOTE would name a procedure this sheet is not.
_TITLE = re.compile(r"RNAV \((?P<kind>GPS|RNP)\)(?: [A-Z])? RWY (?P<runway>\d{1,2}[LRC]?)")
# "Amdt 2A 26DEC24", without the airport coordinate stamp some plates print after it.
_AMENDMENT = re.compile(r"^(?:Amdt|Orig)[-A-Z0-9 ]*\d{2}[A-Z]{3}\d{2}")
# The 28-day validity band printed down both margins: "SE-2, 16 APR 2026 to 14 MAY 2026".
_EFFECTIVE = re.compile(r"\b[A-Z]{2}-\d, (\d{1,2} [A-Z]{3} \d{4} to \d{1,2} [A-Z]{3} \d{4})")
# "TDZE 384", and the sidestep form "TDZE 30L 57" where one plate prints two. Matched
# per LINE: across a line break "\s+" swallows the newline and pairs the word with a
# digit from the next line of the airport diagram (KRDU 32 reads a spurious TDZE 0).
_TDZE = re.compile(r"\bTDZE\s+(-?\d+)(?![LRC\d])")
_TDZE_FOR_RUNWAY = re.compile(r"\bTDZE\s+(\d{1,2}[LRC])\s+(-?\d+)\b")
# The minima table starts here. Anchoring on it keeps the plate's NOTES out of the
# parse -- they are full of the phrase "LNAV/VNAV NA below -12 C", which is not a row.
_TABLE_START = re.compile(r"\bCATEGORY\b")
_LPV = re.compile(r"\bLPV\b")
_LNAV_VNAV_LABEL = re.compile(r"LNAV/")
_ROW_VALUES = re.compile(r"\bDA\b\s*\*?\s*(?P<figures>[^(]*)\(")
_LNAV_MDA = re.compile(r"\bLNAV\s+MDA\b")
# A label and its figures are printed stacked, the row between "LNAV/" and "VNAV"; on
# KRDU 23L they are seven lines apart because the table is interleaved with the margin.
_LABEL_WINDOW = 8


@dataclass(frozen=True)
class Plate:
    """One approach plate, reduced to the facts the configuration needs."""

    path: Path
    runway: str
    procedure: str
    amendment: str
    effective: str
    touchdown_zone_elevation_ft: int
    decision_altitude_ft_msl: int | None
    decision_height_above_touchdown_ft: int | None
    service: str | None

    def minima(self) -> PublishedMinima:
        if self.service is None:
            return no_vertical_minima(
                f"{self.procedure} ({self.path.name}) publishes an LNAV MDA and no "
                "vertically guided line, so its missed approach point is a fix, not a height"
            )
        return PublishedMinima(
            service=self.service,
            decision_altitude_ft_msl=float(self.decision_altitude_ft_msl),
            decision_height_above_touchdown_ft=float(self.decision_height_above_touchdown_ft),
            touchdown_zone_elevation_ft=float(self.touchdown_zone_elevation_ft),
            chart=self.path.name,
            procedure=self.procedure,
            amendment=self.amendment,
            chart_effective=self.effective,
            note=None,
        )


def plate_lines(pdf: Path) -> list[str]:
    """The plate's text, one whitespace-collapsed line per printed line."""
    done = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True
    )
    return [re.sub(r"[^\S\n]+", " ", line).strip() for line in done.stdout.splitlines()]


def _row(lines: list[str]) -> tuple[int, int] | None:
    """The first ``DA`` row in these lines, as (decision altitude MSL, height above TDZ).

    A minima row prints the altitude first and the height last before the parenthesised
    ceiling/visibility, with the visibility between them in either of two spellings
    (``598/18`` RVR or ``820-1 1/4`` statute miles). Reading the ends rather than the
    middle is what makes one rule cover both.
    """
    for line in lines:
        found = _ROW_VALUES.search(line)
        if not found:
            continue
        figures = re.findall(r"-?\d+", found.group("figures"))
        if len(figures) >= 2:
            return int(figures[0]), int(figures[-1])
    return None


def _labelled_row(table: list[str], label: re.Pattern[str]) -> tuple[int, int] | None:
    """The figures belonging to ``label``, whether printed on its line or stacked under it."""
    for index, line in enumerate(table):
        if label.search(line) and not _LNAV_MDA.search(line):
            values = _row(table[index : index + _LABEL_WINDOW])
            if values is not None:
                return values
    return None


def _touchdown_zone_elevation(pdf: Path, lines: list[str], runway: str) -> int:
    """The TDZE this plate's own runway is published against.

    A sidestep plate prints two ("TDZE 30L 57" and "TDZE 30R 55"); when one is labelled
    with this runway it is the answer, and the unlabelled figures are not consulted at
    all. Only a plate that labels none of them leaves a set to choose from, and then the
    set must hold exactly one value -- two unlabelled candidates would make the
    cross-check below a choice, and a check that gets to choose is not a check.
    """
    for line in lines:
        for ident, value in _TDZE_FOR_RUNWAY.findall(line):
            if ident == runway or ident.zfill(3)[-len(runway):] == runway:
                return int(value)
    unlabelled = {int(v) for line in lines for v in _TDZE.findall(line)}
    if len(unlabelled) != 1:
        raise ValueError(f"{pdf.name}: cannot tell which TDZE is RWY {runway}'s ({sorted(unlabelled)})")
    return unlabelled.pop()


def read_plate(pdf: Path) -> Plate | None:
    """Read one plate. Returns None for a plate this fleet is not judged against."""
    lines = plate_lines(pdf)
    titles = {(m.group("kind"), m.group("runway")) for line in lines for m in _TITLE.finditer(line)}
    if not titles:
        raise ValueError(f"{pdf.name}: no RNAV approach title on the plate")
    if len(titles) > 1:
        raise ValueError(f"{pdf.name}: the page names more than one procedure {sorted(titles)}")
    kind, runway = titles.pop()
    if kind != "GPS":
        return None

    digits = runway[:-1] if runway[-1] in "LRC" else runway
    runway = f"{int(digits):02d}" + (runway[-1] if runway[-1] in "LRC" else "")
    procedure = next(m.group(0) for line in lines for m in _TITLE.finditer(line))
    amendment = next((_AMENDMENT.match(line).group(0) for line in lines if _AMENDMENT.match(line)), "")
    if not amendment:
        raise ValueError(f"{pdf.name}: no amendment line")
    effective = next((m.group(1) for line in lines for m in [_EFFECTIVE.search(line)] if m), "")
    if not effective:
        raise ValueError(f"{pdf.name}: no validity band")
    tdze = _touchdown_zone_elevation(pdf, lines, runway)

    starts = [i for i, line in enumerate(lines) if _TABLE_START.search(line)]
    if not starts:
        raise ValueError(f"{pdf.name}: no minima table")
    table = lines[starts[0] + 1 :]

    service: str | None = None
    values = _labelled_row(table, _LPV)
    if values is not None:
        service = "lpv"
    elif any(_LPV.search(line) for line in table):
        # The table says LPV and the row did not parse. Falling through to the LNAV/VNAV
        # row would store a decision altitude 150 ft too high on KRDU 05L, and the
        # cross-check below would pass, because every row on a plate satisfies it.
        raise ValueError(f"{pdf.name}: the minima table names LPV but no LPV row parsed")
    else:
        values = _labelled_row(table, _LNAV_VNAV_LABEL)
        if values is not None:
            service = "lnav_vnav"
    if values is None:
        # "No vertically guided minima" is an assertion about the plate, so the plate has
        # to carry it: an LNAV MDA row is what makes this runway non-precision rather
        # than a row the parser failed to find.
        if not any(_LNAV_MDA.search(line) for line in table):
            raise ValueError(f"{pdf.name}: no vertically guided row AND no LNAV MDA row")
        return Plate(pdf, runway, procedure, amendment, effective, tdze, None, None, None)

    altitude, height = values
    if altitude - height != tdze:
        raise ValueError(
            f"{pdf.name}: {service} DA {altitude} - height above touchdown {height} = "
            f"{altitude - height}, which is not RWY {runway}'s printed TDZE {tdze}"
        )
    return Plate(pdf, runway, procedure, amendment, effective, tdze, altitude, height, service)


def read_airport_plates(chart_dir: Path) -> dict[str, Plate]:
    """Every RNAV (GPS) plate in one airport's directory, keyed by runway."""
    plates: dict[str, Plate] = {}
    for pdf in sorted(chart_dir.iterdir()):
        if pdf.suffix.upper() != ".PDF":
            continue
        if "#" in pdf.name:
            # A browser's "save as" of a named destination: the same sheet again under a
            # mangled name. Named, not inferred, so a real second plate still collides.
            continue
        plate = read_plate(pdf)
        if plate is None:
            continue
        if plate.runway in plates:
            raise ValueError(
                f"{chart_dir.name} RWY {plate.runway}: two RNAV (GPS) plates "
                f"({plates[plate.runway].path.name}, {pdf.name}) -- which one is flown "
                "is a decision, not a default"
            )
        plates[plate.runway] = plate
    return plates


def minima_for_airport(code: str, chart_root: Path, idents: list[str]) -> dict[str, PublishedMinima]:
    chart_dir = chart_root / code
    if not chart_dir.is_dir():
        raise FileNotFoundError(f"{code}: no plates at {chart_dir}")
    plates = read_airport_plates(chart_dir)
    unknown = sorted(set(plates) - set(idents))
    if unknown:
        raise ValueError(f"{code}: plates for thresholds the configuration does not list: {unknown}")
    return {
        ident: plates[ident].minima()
        if ident in plates
        else no_vertical_minima(
            f"{code} publishes no RNAV (GPS) approach to RWY {ident}, so there is no "
            "decision altitude to be late for"
        )
        for ident in idents
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--charts", type=Path, default=DEFAULT_CHART_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="print the table, write nothing")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    guided = 0
    for code, entry in config["airports"].items():
        thresholds = [t for runway in entry["runways"] for t in runway["thresholds"]]
        minima = minima_for_airport(code, args.charts, [t["ident"] for t in thresholds])
        for threshold in thresholds:
            published = minima[threshold["ident"]]
            threshold["published_minima"] = published.to_config()
            guided += published.vertically_guided
            if published.vertically_guided:
                print(
                    f"  {code} {threshold['ident']:<4s} {published.service:<9s} "
                    f"DA {published.decision_altitude_ft_msl:6.0f} ft MSL   "
                    f"{published.decision_height_above_touchdown_ft:5.0f} ft above touchdown   "
                    f"{published.procedure}"
                )
            else:
                print(f"  {code} {threshold['ident']:<4s} {published.service:<9s} {published.note}")

    config["schema_version"] = RUNWAY_THRESHOLDS_SCHEMA
    total = sum(len(rw["thresholds"]) for a in config["airports"].values() for rw in a["runways"])
    print(f"[minima] {guided} of {total} thresholds publish vertically guided minima "
          f"({'/'.join(VERTICALLY_GUIDED)})")
    if args.dry_run:
        print("[minima] --dry-run: configuration not written")
        return
    args.config.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[minima] wrote {args.config} ({RUNWAY_THRESHOLDS_SCHEMA})")


if __name__ == "__main__":
    main()
