from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from data_layout import AEROVIZ_ROOT, airport_charts_dir


FAA_DTPP_SEARCH_URL = "https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/dtpp/search/"
FAA_DTPP_BASE_URL = "https://aeronav.faa.gov/d-tpp"
DEFAULT_TIMEOUT_SECONDS = 45
DEFAULT_CHART_ROOT = AEROVIZ_ROOT.parent / "data" / "RNAV_CHARTS"
SUPPORTED_CHART_PATTERNS = {
    "RNAV_GPS": re.compile(r"\bRNAV\s*\(\s*GPS\s*\)", re.IGNORECASE),
    "RNAV_RNP": re.compile(r"\bRNAV\s*\(\s*RNP\s*\)", re.IGNORECASE),
}


@dataclass(frozen=True)
class ChartRecord:
    airport_icao: str
    airport_ident: str
    chart_name: str
    chart_code: str
    pdf_name: str
    cycle: str

    @property
    def pdf_url(self) -> str:
        return f"{FAA_DTPP_BASE_URL}/{self.cycle}/{self.pdf_name}"


def normalize_airport_ident(value: str) -> tuple[str, str]:
    ident = value.strip().upper()
    if not ident:
        raise ValueError("Airport identifier is required")
    icao = ident if len(ident) == 4 else f"K{ident}"
    faa = ident[1:] if len(ident) == 4 and ident.startswith("K") else ident
    return icao, faa


def fetch_text(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "AeroViz-4D chart downloader"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def resolve_dtpp_cycle(requested_cycle: str | None) -> str:
    if requested_cycle and requested_cycle.lower() not in {"current", "latest"}:
        return requested_cycle

    html = fetch_text(FAA_DTPP_SEARCH_URL)
    current_match = re.search(
        r"Current Edition:.*?/d-tpp/(\d{4})/xml_data/d-TPP_Metafile\.xml",
        html,
        re.IGNORECASE | re.DOTALL,
    )
    if current_match:
        return current_match.group(1)

    any_match = re.search(r"/d-tpp/(\d{4})/xml_data/d-TPP_Metafile\.xml", html)
    if any_match:
        return any_match.group(1)

    raise RuntimeError("Could not discover current FAA d-TPP cycle from FAA search page")


def chart_mode_filter(modes: set[str]) -> re.Pattern[str]:
    patterns = [SUPPORTED_CHART_PATTERNS[mode].pattern for mode in sorted(modes)]
    return re.compile("|".join(patterns), re.IGNORECASE)


def iter_airport_chart_records(
    *,
    cycle: str,
    airport: str,
    modes: set[str],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[ChartRecord]:
    icao, faa = normalize_airport_ident(airport)
    xml_url = f"{FAA_DTPP_BASE_URL}/{cycle}/xml_data/d-TPP_Metafile.xml"
    request = urllib.request.Request(xml_url, headers={"User-Agent": "AeroViz-4D chart downloader"})
    mode_pattern = chart_mode_filter(modes)
    records: list[ChartRecord] = []

    with urllib.request.urlopen(request, timeout=timeout) as response:
        for _event, elem in ET.iterparse(response, events=("end",)):
            if elem.tag != "airport_name":
                continue

            airport_icao = elem.attrib.get("icao_ident", "").upper()
            airport_ident = elem.attrib.get("apt_ident", "").upper()
            if airport_icao == icao or airport_ident == faa:
                for record in elem.findall("record"):
                    chart_code = (record.findtext("chart_code") or "").strip().upper()
                    chart_name = (record.findtext("chart_name") or "").strip()
                    pdf_name = (record.findtext("pdf_name") or "").strip()
                    if chart_code != "IAP":
                        continue
                    if not pdf_name.upper().endswith(".PDF"):
                        continue
                    if not mode_pattern.search(chart_name):
                        continue
                    records.append(
                        ChartRecord(
                            airport_icao=airport_icao or icao,
                            airport_ident=airport_ident or faa,
                            chart_name=chart_name,
                            chart_code=chart_code,
                            pdf_name=pdf_name,
                            cycle=cycle,
                        )
                    )
                elem.clear()
                break

            elem.clear()

    return records


def download_file(url: str, target_path: Path, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "AeroViz-4D chart downloader"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        target_path.write_bytes(response.read())


def parse_modes(value: str) -> set[str]:
    requested = {item.strip().upper().replace("-", "_") for item in value.split(",") if item.strip()}
    if not requested:
        raise ValueError("At least one mode must be requested")
    unknown = requested - set(SUPPORTED_CHART_PATTERNS)
    if unknown:
        raise ValueError(f"Unsupported chart mode(s): {', '.join(sorted(unknown))}")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download FAA RNAV(GPS) and RNAV(RNP) approach chart PDFs for one airport."
    )
    parser.add_argument("airport", help="Airport ICAO or FAA identifier, e.g. KRDU or RDU")
    parser.add_argument(
        "--cycle",
        default="current",
        help="FAA d-TPP cycle, e.g. 2604. Default discovers the current cycle.",
    )
    parser.add_argument(
        "--modes",
        default="RNAV_GPS,RNAV_RNP",
        help="Comma-separated modes: RNAV_GPS,RNAV_RNP",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_CHART_ROOT),
        help="Root directory for downloads. Files go under <root>/<ICAO>/ by default.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Write directly to aeroviz-4d/public/data/airports/<ICAO>/charts instead of <output-root>/<ICAO>.",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--dry-run", action="store_true", help="List matching charts without downloading PDFs")
    args = parser.parse_args()

    icao, _faa = normalize_airport_ident(args.airport)
    cycle = resolve_dtpp_cycle(args.cycle)
    modes = parse_modes(args.modes)
    target_dir = airport_charts_dir(icao) if args.public else Path(args.output_root) / icao

    try:
        records = iter_airport_chart_records(
            cycle=cycle,
            airport=icao,
            modes=modes,
            timeout=args.timeout,
        )
    except urllib.error.URLError as error:
        raise SystemExit(f"Failed to fetch FAA d-TPP metafile: {error}") from error

    if not records:
        raise SystemExit(f"No RNAV(GPS)/RNAV(RNP) IAP charts found for {icao} in cycle {cycle}")

    print(f"FAA d-TPP cycle: {cycle}")
    print(f"Airport: {icao}")
    print(f"Target: {target_dir}")
    for record in records:
        target_path = target_dir / record.pdf_name
        print(f"{record.chart_name}: {record.pdf_url}")
        if not args.dry_run:
            download_file(record.pdf_url, target_path, timeout=args.timeout)
            print(f"  saved {target_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
