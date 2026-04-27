"""
faa_fix_lookup.py
=================
Resolve missing fix/waypoint coordinates for CIFP-derived procedure data.

The resolver is intentionally conservative:
- first search local FAA CIFP text for coordinate-bearing fix records;
- then query the FAA Location Identifier Fixes/Waypoints page and parse any
  coordinate text returned by that site.

The FAA web page is JavaScript-driven and may change its transport details, so
network lookups are best-effort. Returned coordinates should be treated as
research/display data until the preprocessing pipeline records provenance.

Usage:
  python aeroviz-4d/python/faa_fix_lookup.py DUHAM
  python aeroviz-4d/python/faa_fix_lookup.py DUHAM KASLE --json
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


DEFAULT_CIFP_ROOT = Path(__file__).parents[2] / "data" / "CIFP" / "CIFP_260319"
DEFAULT_FAACIFP_NAME = "FAACIFP18"
DEFAULT_FAA_FIX_URL = (
    "https://www.faa.gov/air_traffic/flight_info/aeronav/aero_data/"
    "loc_id_search/fixes_waypoints/"
)
DEFAULT_TIMEOUT_SECONDS = 20

COORD_PAIR_RE = re.compile(r"([NS]\d{8,10})([EW]\d{9,11})")
DECIMAL_PAIR_RE = re.compile(
    r"(?P<lat>[+-]?\d{1,2}\.\d+)\s*[,;/ ]+\s*(?P<lon>[+-]?\d{1,3}\.\d+)"
)
DMS_TEXT_PAIR_RE = re.compile(
    r"(?P<lat_deg>\d{1,2})\D+"
    r"(?P<lat_min>\d{1,2})\D+"
    r"(?P<lat_sec>\d{1,2}(?:\.\d+)?)\D*"
    r"(?P<lat_hem>[NS])\D+"
    r"(?P<lon_deg>\d{1,3})\D+"
    r"(?P<lon_min>\d{1,2})\D+"
    r"(?P<lon_sec>\d{1,2}(?:\.\d+)?)\D*"
    r"(?P<lon_hem>[EW])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FixCoordinate:
    """Resolved fix/waypoint coordinate."""

    ident: str
    lon: float
    lat: float
    source: str
    source_detail: str | None = None


def normalize_ident(ident: str) -> str:
    """Normalize a fix identifier for lookup."""
    return ident.strip().upper()


def decode_cifp_coordinate(token: str) -> float:
    """Decode FAA CIFP compact DMS coordinates to decimal degrees."""
    value = token.strip().upper()
    if len(value) < 9:
        raise ValueError(f"Coordinate token is too short: {token!r}")

    hemisphere = value[0]
    if hemisphere not in {"N", "S", "E", "W"}:
        raise ValueError(f"Missing coordinate hemisphere: {token!r}")

    deg_width = 2 if hemisphere in {"N", "S"} else 3
    digits = value[1:]
    if len(digits) < deg_width + 4 or not digits.isdigit():
        raise ValueError(f"Invalid coordinate digits: {token!r}")

    degrees = int(digits[:deg_width])
    minutes = int(digits[deg_width : deg_width + 2])
    seconds_digits = digits[deg_width + 2 :]
    seconds_scale = 10 ** max(len(seconds_digits) - 2, 0)
    seconds = int(seconds_digits) / seconds_scale

    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if hemisphere in {"S", "W"}:
        decimal *= -1.0
    return decimal


def dms_to_decimal(degrees: str, minutes: str, seconds: str, hemisphere: str) -> float:
    """Convert DMS coordinate parts to signed decimal degrees."""
    decimal = float(degrees) + float(minutes) / 60.0 + float(seconds) / 3600.0
    if hemisphere.upper() in {"S", "W"}:
        decimal *= -1.0
    return decimal


def faacifp_path(cifp_root: Path) -> Path:
    """Return the FAACIFP18 file path under a cycle root."""
    return cifp_root / DEFAULT_FAACIFP_NAME


def lookup_local_cifp_fix(ident: str, faacifp_file: Path) -> FixCoordinate | None:
    """Find a coordinate-bearing fix record in local FAACIFP18 text."""
    normalized_ident = normalize_ident(ident)
    if not faacifp_file.exists():
        return None

    with faacifp_file.open(encoding="ascii", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if normalized_ident not in line:
                continue

            match = COORD_PAIR_RE.search(line)
            if not match:
                continue

            try:
                lat = decode_cifp_coordinate(match.group(1))
                lon = decode_cifp_coordinate(match.group(2))
            except ValueError:
                continue

            return FixCoordinate(
                ident=normalized_ident,
                lon=lon,
                lat=lat,
                source="local-cifp",
                source_detail=f"{faacifp_file}:{line_number}",
            )

    return None


def parse_coordinate_from_text(ident: str, text: str, source_detail: str) -> FixCoordinate | None:
    """Parse the first plausible coordinate pair near a fix ident."""
    normalized_ident = normalize_ident(ident)
    clean_text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    upper_text = clean_text.upper()
    ident_index = upper_text.find(normalized_ident)
    if ident_index < 0:
        return None

    window_start = max(0, ident_index - 500)
    window_end = min(len(clean_text), ident_index + 1200)
    window = clean_text[window_start:window_end]

    compact_match = COORD_PAIR_RE.search(window)
    if compact_match:
        try:
            return FixCoordinate(
                ident=normalized_ident,
                lon=decode_cifp_coordinate(compact_match.group(2)),
                lat=decode_cifp_coordinate(compact_match.group(1)),
                source="faa-fixes-waypoints",
                source_detail=source_detail,
            )
        except ValueError:
            pass

    dms_match = DMS_TEXT_PAIR_RE.search(window)
    if dms_match:
        return FixCoordinate(
            ident=normalized_ident,
            lon=dms_to_decimal(
                dms_match.group("lon_deg"),
                dms_match.group("lon_min"),
                dms_match.group("lon_sec"),
                dms_match.group("lon_hem"),
            ),
            lat=dms_to_decimal(
                dms_match.group("lat_deg"),
                dms_match.group("lat_min"),
                dms_match.group("lat_sec"),
                dms_match.group("lat_hem"),
            ),
            source="faa-fixes-waypoints",
            source_detail=source_detail,
        )

    decimal_match = DECIMAL_PAIR_RE.search(window)
    if decimal_match:
        lat = float(decimal_match.group("lat"))
        lon = float(decimal_match.group("lon"))
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            return FixCoordinate(
                ident=normalized_ident,
                lon=lon,
                lat=lat,
                source="faa-fixes-waypoints",
                source_detail=source_detail,
            )

    return None


def _read_url(url: str, timeout_seconds: int) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "aeroviz-4d-fix-coordinate-resolver/0.1",
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8", errors="replace")


def _candidate_faa_search_urls(base_url: str, ident: str) -> list[str]:
    normalized_ident = normalize_ident(ident)
    query_variants = [
        {"search": normalized_ident},
        {"Search": normalized_ident},
        {"combine": normalized_ident},
        {"keys": normalized_ident},
        {"id": normalized_ident},
        {"fix": normalized_ident},
    ]
    return [f"{base_url}?{urlencode(query)}" for query in query_variants]


def _extract_links(page_html: str, base_url: str, ident: str) -> list[str]:
    normalized_ident = normalize_ident(ident)
    links: list[str] = []
    for match in re.finditer(r'href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', page_html, re.I | re.S):
        href = html.unescape(match.group(1))
        label = html.unescape(re.sub(r"<[^>]+>", " ", match.group(2))).strip().upper()
        if normalized_ident not in href.upper() and normalized_ident not in label:
            continue
        links.append(urljoin(base_url, href))
    return list(dict.fromkeys(links))


def lookup_faa_fix_page(
    ident: str,
    base_url: str = DEFAULT_FAA_FIX_URL,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> FixCoordinate | None:
    """Best-effort lookup through the FAA Fixes/Waypoints search page."""
    normalized_ident = normalize_ident(ident)
    checked_pages: list[tuple[str, str]] = []

    for url in _candidate_faa_search_urls(base_url, normalized_ident):
        try:
            page_html = _read_url(url, timeout_seconds)
        except (HTTPError, URLError, TimeoutError, OSError):
            continue

        checked_pages.append((url, page_html))
        coordinate = parse_coordinate_from_text(normalized_ident, page_html, url)
        if coordinate:
            return coordinate

        for link in _extract_links(page_html, base_url, normalized_ident):
            try:
                detail_html = _read_url(link, timeout_seconds)
            except (HTTPError, URLError, TimeoutError, OSError):
                continue

            coordinate = parse_coordinate_from_text(normalized_ident, detail_html, link)
            if coordinate:
                return coordinate

    return None


def resolve_fix_coordinate(
    ident: str,
    *,
    cifp_root: Path = DEFAULT_CIFP_ROOT,
    faa_fix_url: str = DEFAULT_FAA_FIX_URL,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    allow_network: bool = True,
) -> FixCoordinate | None:
    """Resolve a fix coordinate from local CIFP, then FAA Fixes/Waypoints."""
    normalized_ident = normalize_ident(ident)

    local = lookup_local_cifp_fix(normalized_ident, faacifp_path(cifp_root))
    if local:
        return local

    if not allow_network:
        return None

    return lookup_faa_fix_page(
        normalized_ident,
        base_url=faa_fix_url,
        timeout_seconds=timeout_seconds,
    )


def resolve_fix_coordinates(
    idents: Iterable[str],
    *,
    cifp_root: Path = DEFAULT_CIFP_ROOT,
    faa_fix_url: str = DEFAULT_FAA_FIX_URL,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    allow_network: bool = True,
) -> dict[str, FixCoordinate | None]:
    """Resolve multiple fix identifiers."""
    return {
        normalize_ident(ident): resolve_fix_coordinate(
            ident,
            cifp_root=cifp_root,
            faa_fix_url=faa_fix_url,
            timeout_seconds=timeout_seconds,
            allow_network=allow_network,
        )
        for ident in idents
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve missing CIFP fix coordinates from local CIFP and FAA Fixes/Waypoints."
    )
    parser.add_argument("idents", nargs="+", help="Fix identifiers, e.g. DUHAM KASLE")
    parser.add_argument(
        "--cifp-root",
        type=Path,
        default=DEFAULT_CIFP_ROOT,
        help="FAA CIFP cycle directory containing FAACIFP18",
    )
    parser.add_argument(
        "--faa-fix-url",
        default=DEFAULT_FAA_FIX_URL,
        help="FAA Fixes/Waypoints lookup page URL",
    )
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Only search local CIFP; do not query FAA web pages",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    results = resolve_fix_coordinates(
        args.idents,
        cifp_root=args.cifp_root,
        faa_fix_url=args.faa_fix_url,
        timeout_seconds=args.timeout_seconds,
        allow_network=not args.no_network,
    )

    if args.json:
        payload = {
            ident: asdict(result) if result else None
            for ident, result in results.items()
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for ident, result in results.items():
            if result is None:
                print(f"{ident}: unresolved")
                continue
            detail = f" ({result.source_detail})" if result.source_detail else ""
            print(f"{ident}: lon={result.lon:.8f} lat={result.lat:.8f} source={result.source}{detail}")

    return 0 if all(result is not None for result in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
