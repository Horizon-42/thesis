#!/usr/bin/env python3
"""Merge downloaded landing files into one CZML the frontend can load.

``download_landings.py`` writes one CZML-input file per runway threshold, but the
frontend loads a single ``trajectories.czml`` per airport. This merges an airport's
threshold files (all of them, or a chosen subset) and runs ``generate_czml.py``.

    # all runways of KRDU -> public/data/airports/KRDU/trajectories.czml
    python trajectory_data_process/landings_to_czml.py --airport KRDU

    # only specific runway ends, to a custom path
    python trajectory_data_process/landings_to_czml.py --airport KRDU --runway 23R 23L \\
      --output /tmp/krdu_23.czml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def landing_files(airport_dir: Path, code: str, runways: list[str] | None) -> list[Path]:
    """Return the landing files for an airport (all, or the requested runways)."""
    if runways:
        paths = [airport_dir / f"{code}_{r.upper()}_landings.json" for r in runways]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"No landing files for: {', '.join(missing)} (run download_landings.py first)")
        return paths
    return sorted(airport_dir.glob(f"{code}_*_landings.json"))


def merge_landing_flights(paths: list[Path]) -> list[dict[str, Any]]:
    """Concatenate landing flights across files, de-duplicating and re-uniquing ids."""
    flights: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None]] = set()
    used_ids: set[str] = set()
    for path in paths:
        for flight in json.loads(path.read_text(encoding="utf-8")):
            key = (flight.get("icao24"), flight.get("landing_time_utc"))
            if key in seen:
                continue
            seen.add(key)
            flight = dict(flight)
            flight["id"] = _unique_id(str(flight["id"]), used_ids)
            used_ids.add(flight["id"])
            flights.append(flight)
    return flights


def _unique_id(base: str, used_ids: set[str]) -> str:
    if base not in used_ids:
        return base
    suffix = 2
    while f"{base}_{suffix}" in used_ids:
        suffix += 1
    return f"{base}_{suffix}"


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Merge landing files into one CZML for the frontend")
    p.add_argument("--airport", required=True, help="Airport ICAO code, e.g. KRDU")
    p.add_argument("--runway", nargs="+", default=None, help="Runway ends to include (default: all)")
    p.add_argument("--output", default=None, help="Output CZML (default: <aeroviz>/public/data/airports/<ICAO>/trajectories.czml)")
    p.add_argument("--multiplier", type=int, default=None, help="Optional CZML clock multiplier")
    p.add_argument("--output-root", default=str(here / "outputs" / "landings"))
    p.add_argument("--aeroviz-root", default=str(here.parents[0] / "aeroviz-4d"))
    args = p.parse_args()

    code = args.airport.upper()
    airport_dir = Path(args.output_root) / code
    aeroviz_root = Path(args.aeroviz_root)

    paths = landing_files(airport_dir, code, args.runway)
    flights = merge_landing_flights(paths)
    if not flights:
        raise SystemExit(f"No landings found under {airport_dir}")

    combined = airport_dir / f"{code}_combined_czml_input.json"
    combined.write_text(json.dumps(flights, indent=2), encoding="utf-8")

    output = Path(args.output) if args.output else aeroviz_root / "public" / "data" / "airports" / code / "trajectories.czml"
    output.parent.mkdir(parents=True, exist_ok=True)

    generator = aeroviz_root / "python" / "generate_czml.py"
    if not generator.exists():
        raise SystemExit(f"generate_czml.py not found: {generator}")

    cmd = [sys.executable, str(generator), "--airport", code, "--input", str(combined), "--output", str(output)]
    if args.multiplier is not None:
        cmd += ["--multiplier", str(args.multiplier)]
    print(f"[landings->czml] {len(flights)} landings from {len(paths)} file(s) -> {output}", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
