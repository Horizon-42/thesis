#!/usr/bin/env python3
"""Render downloaded landings into per-runway + combined CZML for the frontend.

``download_landings.py`` writes one CZML-input file per runway threshold. Each
raw track is first cut to its ARRIVAL SEGMENT (final entry into the 25 km ring
→ touchdown; see ``arrival_segment.py``) — pure local circuits that never left
the ring are excluded into ``<ICAO>_local_rejected.json`` for review. The raw
``*_landings.json`` stay untouched; the truncated arrivals are written as
derived ``*_arrivals.json`` next to them, and everything downstream (the CZMLs,
the combined czml-input that scenario building and reference records consume)
is built from those. The frontend can load a single runway or all of them, so
this writes, into the airport's frontend folder:

  public/data/airports/<ICAO>/landings/<ICAO>_<RWY>.czml   one CZML per runway
  public/data/airports/<ICAO>/landings/index.json          manifest of runways
  public/data/airports/<ICAO>/trajectories.czml            all runways combined

    # all downloaded airports (default)
    python trajectory_data_process/landings_to_czml.py

    # one airport, or a subset of its runways
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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory_data_process.arrival_segment import truncate_flights


def _airport_reference(code: str) -> tuple[float, float]:
    """The airport's (lat, lon) from the runway-thresholds config — the entry ring's centre."""
    config_path = Path(__file__).resolve().parent / "config" / "runway_thresholds.json"
    airports = json.loads(config_path.read_text(encoding="utf-8"))["airports"]
    record = airports.get(code.upper())
    if record is None:
        raise SystemExit(f"airport {code} not in {config_path} — cannot place the entry ring")
    return float(record["lat"]), float(record["lon"])


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


def discover_airports(output_root: Path) -> list[str]:
    """Return airport codes that have downloaded landing files under output_root."""
    if not output_root.exists():
        return []
    codes = [d.name.upper() for d in output_root.iterdir() if d.is_dir() and any(d.glob("*_landings.json"))]
    return sorted(codes)


def render_airport(
    code: str,
    *,
    runways: list[str] | None,
    source_root: Path,
    aeroviz_root: Path,
    generator: Path,
    multiplier: int | None,
) -> int:
    """Render one airport's landings into per-runway + combined CZML + manifest.

    Returns the number of runways rendered (0 if the airport has no landings).
    """
    source_dir = source_root / code
    airport_dir = aeroviz_root / "public" / "data" / "airports" / code
    landings_dir = airport_dir / "landings"
    airport_lat, airport_lon = _airport_reference(code)

    paths = landing_files(source_dir, code, runways)
    runway_manifest: list[dict[str, Any]] = []
    arrival_paths: list[Path] = []
    local_flights: list[dict[str, Any]] = []
    for path in paths:
        flights = json.loads(path.read_text(encoding="utf-8"))
        if not flights:
            continue  # an idle runway end with no landings: skip it
        # Cut each raw track to its arrival segment (final 25 km ring entry ->
        # touchdown); pure local circuits are set aside, never silently dropped.
        arrivals, locals_ = truncate_flights(flights, airport_lat, airport_lon)
        local_flights.extend(locals_)
        if not arrivals:
            continue
        ident = runway_ident_from_path(path, code)
        arrivals_path = path.with_name(path.name.replace("_landings.json", "_arrivals.json"))
        arrivals_path.write_text(json.dumps(arrivals, indent=2), encoding="utf-8")
        arrival_paths.append(arrivals_path)
        czml_path = landings_dir / f"{code}_{ident}.czml"
        _generate_czml(generator, code, arrivals_path, czml_path, multiplier)
        truncated = sum(1 for f in arrivals if f.get("arrival_truncated"))
        runway_manifest.append({"runway": ident, "file": f"landings/{czml_path.name}", "count": len(arrivals)})
        print(f"[landings->czml] {code} {ident}: {len(arrivals)} arrival(s) "
              f"({truncated} truncated, {len(locals_)} local excluded) -> {czml_path}")

    if local_flights:
        local_path = source_dir / f"{code}_local_rejected.json"
        local_path.write_text(json.dumps(local_flights, indent=2), encoding="utf-8")
        print(f"[landings->czml] {code}: {len(local_flights)} local circuit(s) "
              f"(never left the entry ring) -> {local_path}")

    if not runway_manifest:
        print(f"[landings->czml] {code}: no landings, skipped")
        return 0

    # Combined (all runways) -> the airport's default trajectories.czml.
    combined_flights = merge_landing_flights(arrival_paths)
    combined_input = source_dir / f"{code}_combined_czml_input.json"
    combined_input.write_text(json.dumps(combined_flights, indent=2), encoding="utf-8")
    _generate_czml(generator, code, combined_input, airport_dir / "trajectories.czml", multiplier)
    print(f"[landings->czml] {code} combined: {len(combined_flights)} -> {airport_dir / 'trajectories.czml'}")

    manifest = {
        "airport": code,
        "combined": "trajectories.czml",
        "runways": sorted(runway_manifest, key=lambda r: r["runway"]),
    }
    (landings_dir / "index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[landings->czml] {code} manifest: {landings_dir / 'index.json'}")
    return len(runway_manifest)


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Render landings into per-runway + combined CZML for the frontend")
    p.add_argument("--airport", default=None, help="Airport ICAO code (default: all downloaded airports)")
    p.add_argument("--runway", nargs="+", default=None, help="Runway ends to include (requires --airport)")
    p.add_argument("--multiplier", type=int, default=None, help="Optional CZML clock multiplier")
    p.add_argument("--output-root", default=str(here / "outputs" / "landings"), help="Where the *_landings.json live")
    p.add_argument("--aeroviz-root", default=str(here.parents[0] / "aeroviz-4d"), help="Frontend root to write CZML into")
    args = p.parse_args()

    if args.runway and not args.airport:
        raise SystemExit("--runway requires --airport")

    source_root = Path(args.output_root)
    aeroviz_root = Path(args.aeroviz_root)
    generator = aeroviz_root / "python" / "generate_czml.py"
    if not generator.exists():
        raise SystemExit(f"generate_czml.py not found: {generator}")

    codes = [args.airport.upper()] if args.airport else discover_airports(source_root)
    if not codes:
        raise SystemExit(f"No downloaded landings found under {source_root} (run download_landings.py first)")

    total = 0
    for code in codes:
        total += render_airport(
            code,
            runways=args.runway,
            source_root=source_root,
            aeroviz_root=aeroviz_root,
            generator=generator,
            multiplier=args.multiplier,
        )
    print(f"[landings->czml] done: {len(codes)} airport(s), {total} runway file(s)")


if __name__ == "__main__":
    main()
