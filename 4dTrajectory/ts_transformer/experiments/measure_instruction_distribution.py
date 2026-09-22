"""Read the existing development roster; measure instruction signals without model inference.

Run with conda run -n aeroviz python. See the accompanying distribution report for commands.
Source tracks, manifests, published vocabulary artefacts and test partitions are never written.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "4dTrajectory")]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def extract(artefact: Path, out: Path) -> None:
    from ts_transformer.config import TSConfig
    from ts_transformer.data.dataset import build_series, load_flight_dicts
    from ts_transformer.data.development_cohorts import load_development_cohort
    from ts_transformer.manoeuvre.instructions import course_frame, min_rows, smooth

    source = json.loads(artefact.read_text())
    identity = source["cohort_identity"]
    cohort_path = Path(identity["path"])
    assert sha(cohort_path) == identity["artifact_sha256"]
    cohort = load_development_cohort(cohort_path)
    keys = {"train": set(cohort.train_flight_ids), "val": set(cohort.val_flight_ids)}
    assert not keys["train"] & keys["val"]
    saved_config = identity["provenance"]["config"]
    config = TSConfig(**{name: saved_config[name] for name in ("seq_len", "dt_s", "aircraft_type")})
    assert all(getattr(config, name) == value for name, value in saved_config.items())
    out.mkdir(parents=True, exist_ok=False)
    metadata = {
        "vocabulary_artefact": str(artefact.relative_to(REPO)),
        "vocabulary_artefact_sha256": sha(artefact),
        "cohort_path": str(cohort_path.relative_to(REPO)), "cohort_sha256": sha(cohort_path),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "config": asdict(config), "airports": {}, "split_names": ["train", "val"],
        "columns": ["heading_deg", "speed_mps", "height_m", "cross_m"],
        "cross_sign": "positive right of inbound final approach course; negative left; signed distance to its extended centreline",
        "smoothing_seconds": {"heading": 6, "speed": 10, "height": 10},
        "script_sha256_at_extraction": sha(Path(__file__)),
        "source_hashes": {str(p.relative_to(REPO)): sha(p) for p in [
            REPO / "4dTrajectory/ts_transformer/data/dataset.py",
            REPO / "4dTrajectory/ts_transformer/manoeuvre/instructions.py",
            REPO / "4dTrajectory/ts_transformer/data/approach_difficulty.py",
        ]},
    }
    started = time.monotonic()
    for manifest_name, expected_sha in zip(identity["provenance"]["manifests"],
                                           identity["provenance"]["manifest_sha256"]):
        manifest = Path(manifest_name)
        assert sha(manifest) == expected_sha, f"manifest moved: {manifest}"
        airport = manifest.parent.parent.name
        selected = sorted(k for group in keys.values() for k in group if k.startswith(airport + ":"))
        raw_rows, read_rows, offsets, flight_ids, splits, times = [], [], [0], [], [], []
        missing, refused_frame, reports = [], [], []
        print(f"{airport}: rebuilding {len(selected)} development flights", flush=True)
        for first in range(0, len(selected), 400):
            batch_keys = selected[first:first + 400]
            flights = load_flight_dicts(manifest, include_flight_keys=set(batch_keys), verbose=False)
            built, report = build_series(flights, config, aircraft_type=config.aircraft_type)
            reports.append(report.format())
            by_id = {item.dataset_id: item for item in built}
            missing.extend(k for k in batch_keys if k not in by_id)
            for key in batch_keys:
                if key not in by_id:
                    continue
                item = by_id[key]
                try:
                    frame = course_frame(item)
                except ValueError as error:
                    refused_frame.append({"id": key, "reason": str(error)})
                    continue
                dt = float(np.median(np.diff(frame["t"])))
                assert np.allclose(np.diff(frame["t"]), config.dt_s)
                course = (smooth(frame["course_unwrapped_deg"], min_rows(6., dt)) + 180.) % 360. - 180.
                speed = smooth(frame["ground_speed_mps"], min_rows(10., dt))
                height = smooth(frame["height_m"], min_rows(10., dt))
                raw_rows.append(np.column_stack((frame["relative_course_deg"], frame["ground_speed_mps"],
                                                 frame["height_m"], frame["cross_m"])).astype(np.float32))
                read_rows.append(np.column_stack((course, speed, height, frame["cross_m"])).astype(np.float32))
                times.append(np.asarray(frame["t"], dtype=np.float32))
                offsets.append(offsets[-1] + len(course))
                flight_ids.append(key)
                splits.append(0 if key in keys["train"] else 1)
            print(f"  {min(first + 400, len(selected))}/{len(selected)}, {offsets[-1]} rows, "
                  f"elapsed {time.monotonic()-started:.0f}s", flush=True)
            del flights, built, by_id
        np.savez_compressed(out / f"{airport}.npz", raw=np.concatenate(raw_rows), read=np.concatenate(read_rows),
                            time=np.concatenate(times), offsets=np.array(offsets), flight_ids=np.array(flight_ids),
                            split=np.array(splits, dtype=np.int8))
        metadata["airports"][airport] = {
            "manifest": str(manifest.relative_to(REPO)), "manifest_sha256": expected_sha,
            "requested": len(selected), "built": len(flight_ids), "rows": offsets[-1],
            "missing": missing, "frame_refused": refused_frame, "build_reports": reports,
            "cache_sha256": sha(out / f"{airport}.npz"),
        }
        write_json(out / "extraction.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artefact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    extract(args.artefact.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
