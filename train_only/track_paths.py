"""The harvest track files the training chain reads: one for each flight of the sentence artefact's splits.

A flight that a runner starts is rebuilt from its airport's arrival manifest and its track file
(`autopilot.flights.rebuild_series`, keyed by the artefact's flights). The artefact's flights are those of
``signals.json`` (train, select, val); the test days are not in it (C32), so their tracks are not listed.

    python train_only/track_paths.py <artefact dir>      one path a line, relative to the repository root
"""

import json
import sys
from pathlib import Path

HARVEST = Path("trajectory_data_process/outputs/harvest")


def track_paths(artefact: Path) -> list[str]:
    signals = json.loads((artefact / "signals.json").read_text(encoding="utf-8"))
    keys = {flight["dataset_id"] for split in signals["splits"].values() for flight in split["flights"]}
    airports = sorted({key.split(":", 1)[0] for key in keys})
    paths = []
    for airport in airports:
        manifest = json.loads((HARVEST / airport / "arrivals" / "manifest.json").read_text(encoding="utf-8"))
        paths += [(HARVEST / airport / "tracks" / record["source_file"]).as_posix()
                  for record in manifest["records"] if f"{airport}:{record['flight_key']}" in keys]
    if len(paths) != len(keys):
        raise SystemExit(f"{len(keys)} flights in {artefact}, {len(paths)} found in the arrival manifests")
    return paths


if __name__ == "__main__":
    print("\n".join(track_paths(Path(sys.argv[1]))))
