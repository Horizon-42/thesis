#!/usr/bin/env python3
"""Render downloaded landings into per-runway + combined CZML for the frontend.

``download_landings.py`` writes one CZML-input file per runway threshold. The
frontend can load a single runway or all of them, so this writes, into the
airport's frontend folder:

  public/data/airports/<ICAO>/landings/<ICAO>_<RWY>.czml   one CZML per runway
  public/data/airports/<ICAO>/landings/index.json          manifest of runways
  public/data/airports/<ICAO>/trajectories.czml            all runways combined

    python trajectory_data_process/landings_to_czml.py --airport KRDU
    python trajectory_data_process/landings_to_czml.py --airport KRDU --runway 23R 23L
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


def runway_ident_from_path(path: Path, code: str) -> str:
    """KRDU_23R_landings.json -> 23R."""
    return path.stem[len(f"{code}_"):-len("_landings")]


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


def _generate_czml(generator: Path, code: str, input_path: Path, output_path: Path, multiplier: int | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(generator), "--airport", code, "--input", str(input_path), "--output", str(output_path)]
    if multiplier is not None:
        cmd += ["--multiplier", str(multiplier)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Render landings into per-runway + combined CZML for the frontend")
    p.add_argument("--airport", required=True, help="Airport ICAO code, e.g. KRDU")
    p.add_argument("--runway", nargs="+", default=None, help="Runway ends to include (default: all)")
    p.add_argument("--multiplier", type=int, default=None, help="Optional CZML clock multiplier")
    p.add_argument("--output-root", default=str(here / "outputs" / "landings"), help="Where the *_landings.json live")
    p.add_argument("--aeroviz-root", default=str(here.parents[0] / "aeroviz-4d"), help="Frontend root to write CZML into")
    args = p.parse_args()

    code = args.airport.upper()
    source_dir = Path(args.output_root) / code
    aeroviz_root = Path(args.aeroviz_root)
    airport_dir = aeroviz_root / "public" / "data" / "airports" / code
    landings_dir = airport_dir / "landings"

    generator = aeroviz_root / "python" / "generate_czml.py"
    if not generator.exists():
        raise SystemExit(f"generate_czml.py not found: {generator}")

    paths = [p for p in landing_files(source_dir, code, args.runway)]
    runway_manifest: list[dict[str, Any]] = []
    for path in paths:
        flights = json.loads(path.read_text(encoding="utf-8"))
        if not flights:
            continue  # an idle runway end with no landings: skip it
        ident = runway_ident_from_path(path, code)
        czml_path = landings_dir / f"{code}_{ident}.czml"
        _generate_czml(generator, code, path, czml_path, args.multiplier)
        runway_manifest.append({"runway": ident, "file": f"landings/{czml_path.name}", "count": len(flights)})
        print(f"[landings->czml] {code} {ident}: {len(flights)} -> {czml_path}")

    if not runway_manifest:
        raise SystemExit(f"No landings found under {source_dir}")

    # Combined (all runways) -> the airport's default trajectories.czml.
    combined_flights = merge_landing_flights(paths)
    combined_input = source_dir / f"{code}_combined_czml_input.json"
    combined_input.write_text(json.dumps(combined_flights, indent=2), encoding="utf-8")
    combined_czml = airport_dir / "trajectories.czml"
    _generate_czml(generator, code, combined_input, combined_czml, args.multiplier)
    print(f"[landings->czml] {code} combined: {len(combined_flights)} -> {combined_czml}")

    manifest = {
        "airport": code,
        "combined": "trajectories.czml",
        "runways": sorted(runway_manifest, key=lambda r: r["runway"]),
    }
    index_path = landings_dir / "index.json"
    index_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[landings->czml] manifest: {index_path}")


if __name__ == "__main__":
    main()
