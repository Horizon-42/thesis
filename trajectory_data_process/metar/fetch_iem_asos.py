"""Fetch an airport's ASOS surface-wind observations from the IEM archive.

The observed baseline's speed gate judges a GROUND speed against an airspeed window;
an ordinary 10 kt headwind is half that window. OpenSky state vectors carry no
airspeed and no true heading, so the wind has to come from the field: the Iowa
Environmental Mesonet republishes every ASOS/METAR report (routine hourly + specials)
with the wind direction in degrees TRUE, the speed and gust in knots, timestamped UTC.

    python -m trajectory_data_process.metar.fetch_iem_asos --airport KRDU
    python -m trajectory_data_process.metar.fetch_iem_asos --airport KRDU \\
        --start 2026-05-01 --end 2026-07-23

Without ``--start/--end`` the span is the airport's assigned landings in
``tracks/manifest.json`` padded by a day on each side. Output:
``data/metar/<ICAO>/asos_<start>_<end>.csv`` (the archive's own CSV, unmodified) plus a
``.provenance.json`` beside it (URL, fetch time, row count) — the same static-data
discipline as the CIFP cycle under ``data/``. ``evaluation.wind`` reads the CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "metar"
DEFAULT_HARVEST_ROOT = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest"
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
# The columns evaluation.wind reads, in the archive's own naming: direction (deg true),
# speed and gust (knots). ``valid`` is the observation time (UTC, minute resolution).
EXPECTED_HEADER = "station,valid,drct,sknt,gust"
SOURCE = "IEM ASOS archive (routine hourly + special reports)"


def station_for(airport: str) -> str:
    """IEM keys the contiguous-US ASOS network by the FAA identifier: KRDU -> RDU."""
    code = airport.upper()
    if len(code) == 4 and code.startswith("K"):
        return code[1:]
    raise ValueError(f"{airport}: only K-prefixed FAA identifiers map to an IEM ASOS station")


def landing_span(airport: str, harvest_root: Path) -> tuple[date, date]:
    """First and last assigned landing dates in the airport's tracks manifest."""
    manifest = harvest_root / airport / "tracks" / "manifest.json"
    rows = json.loads(manifest.read_text(encoding="utf-8"))["records"]
    times = sorted(
        datetime.fromisoformat(row["landing_time_utc"].replace("Z", "+00:00"))
        for row in rows
        if row.get("outcome") == "assigned"
    )
    if not times:
        raise ValueError(f"{manifest} lists no assigned landings")
    return times[0].date(), times[-1].date()


def request_url(station: str, start: date, end_exclusive: date) -> str:
    """The archive request: ``end`` is exclusive (the day after the last wanted day)."""
    query = [
        ("station", station),
        ("data", "drct"), ("data", "sknt"), ("data", "gust"),
        ("year1", start.year), ("month1", start.month), ("day1", start.day),
        ("year2", end_exclusive.year), ("month2", end_exclusive.month), ("day2", end_exclusive.day),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("elev", "no"),
        ("missing", "M"), ("trace", "T"), ("direct", "no"),
        ("report_type", "3"), ("report_type", "4"),
    ]
    return IEM_ASOS_URL + "?" + urllib.parse.urlencode(query)


def fetch(airport: str, start: date, end: date, *, out_root: Path) -> Path:
    """Download ``[start, end]`` inclusive for ``airport`` and write CSV + provenance."""
    station = station_for(airport)
    url = request_url(station, start, end + timedelta(days=1))
    with urllib.request.urlopen(url, timeout=120) as response:
        text = response.read().decode("utf-8")
    lines = text.strip().splitlines()
    if not lines or lines[0].strip() != EXPECTED_HEADER:
        raise ValueError(
            f"{url}: unexpected response header {lines[0] if lines else '<empty>'!r}; "
            f"expected {EXPECTED_HEADER!r}"
        )
    rows = lines[1:]
    if not rows:
        raise ValueError(f"{url}: the archive returned no observations for {station}")
    target = out_root / airport.upper()
    target.mkdir(parents=True, exist_ok=True)
    stem = f"asos_{start.isoformat()}_{end.isoformat()}"
    csv_path = target / f"{stem}.csv"
    csv_path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    (target / f"{stem}.provenance.json").write_text(
        json.dumps(
            {
                "source": SOURCE,
                "url": url,
                "station": station,
                "airport": airport.upper(),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "rows": len(rows),
                "columns": EXPECTED_HEADER.split(","),
                "units": {"drct": "deg true", "sknt": "kt", "gust": "kt", "valid": "UTC"},
            },
            indent=1,
        ) + "\n",
        encoding="utf-8",
    )
    return csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--airport", required=True, nargs="+", metavar="ICAO")
    parser.add_argument("--start", type=date.fromisoformat, help="first day (UTC), inclusive")
    parser.add_argument("--end", type=date.fromisoformat, help="last day (UTC), inclusive")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--harvest-root", type=Path, default=DEFAULT_HARVEST_ROOT,
                        help="where tracks/manifest.json supplies the default span")
    args = parser.parse_args(argv)
    if (args.start is None) != (args.end is None):
        parser.error("--start and --end go together")
    for airport in args.airport:
        if args.start is None:
            first, last = landing_span(airport, args.harvest_root)
            start, end = first - timedelta(days=1), last + timedelta(days=1)
        else:
            start, end = args.start, args.end
        path = fetch(airport, start, end, out_root=args.out)
        rows = sum(1 for _ in path.open(encoding="utf-8")) - 1
        print(f"{airport}: {rows} observations {start} .. {end} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
