"""B0′′: does a flight change runway, and does it go around? (plan v3 §5.2.1, D63 / D64)

Both questions decide a WORD in the instruction vocabulary, and neither can be answered from
the harvest's own products: it assigns ONE runway per track (KRDU: 1 ambiguous row in 16 056)
and its outcomes carry no go-around. An aborted approach also falls outside the 25 km arrival
slice, so this reads the FULL stored tracks, rostered by the manifest, never globbed.

    python run_ts.py approach_events --airport KRDU --out <dir>

What it measures, per landed flight, in the runway frame of EVERY threshold at the airport
(`Airport.frames`, datum `hae` — the track store's datum):

  runway change   the arg-min threshold by |cross-track| among the thresholds the aircraft is
                  still short of, over the rows inside RANGE_M and below CEILING_M. The flight
                  "changed" when that arg-min was some other threshold and then became the one
                  it landed on. Reported with the along-track distance at the switch and
                  whether the other threshold was the landed one's PARALLEL partner — a
                  side-step (JO 7110.65BB 4-8-7) is exactly that case. The parallel pairs'
                  centreline separation is printed too, because a separation under 2x the
                  established rule's 500 m means cross-track alone cannot tell them apart.

  go-around       a row before the landing that is low (height <= LOW_M) and near
                  (along <= RANGE_M), followed by a later row above LOW_M + CLIMB_M, the
                  flight landing after that. "Descended to the runway, climbed away, came back."

Approximation, stated because the repo forbids silent ones: heights are HAE against an HAE
threshold elevation, so the datum is consistent and no geoid term enters. Tracks are read at
every STRIDE-th sample (~4 s at the store's 2 s spacing), which cannot move a 150 m/900 m test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from final_approach.frame import TrackPoint
from trajectory_data_process.harvest.airports import load_airport
from ts_transformer.repo_layout import REPO_ROOT

#: Rows this far along the approach, and no higher, are the ones a runway choice is read from.
RANGE_M = 30_000.0
CEILING_M = 3_000.0
STRIDE = 2
#: Inside this the thresholds and the crossing runways converge and "the nearest centreline" stops
#: meaning anything — at KRDU 14/32 cross 05/23 near the airport, and every flight flickers there.
INNER_M = 3_000.0
#: A different runway must hold the choice over this much along-track before it counts as a change.
MIN_RUN_M = 2_000.0
#: A row only expresses a runway CHOICE when the aircraft is on that centreline, not merely nearest
#: to it — the same half-width the intercept word's established rule uses (`approach_difficulty`).
#: Without it a vectored aircraft 20 km out flips between two parallels as it crosses their midline.
ESTABLISHED_CROSS_M = 500.0
#: Two parallels are only TELLABLE APART this way when their corridors do not overlap.
SEPARABLE_M = 2 * ESTABLISHED_CROSS_M
#: AIM 5-4-19 a: a side-step is authorised for parallels "separated by 1,200 feet or less".
SIDE_STEP_MAX_M = 1200.0 * 0.3048
#: A go-around is low AND NEAR, then a climb. 460 m at 11 km is an ordinary approach, not an abort.
NEAR_M = 5_000.0
LOW_M = 300.0
CLIMB_M = 600.0


def parallel_partner(ident: str, idents: list[str]) -> str | None:
    """`05L` -> `05R` when the airport has it: the same number, the other side."""
    if not ident[-1] in "LRC":
        return None
    number, side = ident[:-1], ident[-1]
    for other in idents:
        if other != ident and other[:-1] == number and other[-1] in "LRC":
            return other
    return None


def read_track(path: Path) -> tuple[list[TrackPoint], int]:
    """The stored samples as ``[t, lon, lat, alt]`` and the landing row."""
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data["samples"]
    landing = data["landing_sample_index"]
    return [TrackPoint(lat=s[2], lon=s[1], alt_m=s[3]) for s in samples], int(landing)


def measure(airport_code: str, out: Path, config_file: Path, cifp_file: Path, limit: int) -> dict:
    airport = load_airport(airport_code, config_file=config_file, cifp_file=cifp_file)
    frames = airport.frames("hae")
    idents = [f.ident for f in frames]
    store = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest" / airport_code / "tracks"
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    records = [r for r in manifest["records"] if r["outcome"] == "assigned"]
    if limit:
        records = records[:limit]
    print(f"  {airport_code}: {len(records)} landed tracks, {len(idents)} thresholds {idents}", flush=True)

    partners = {i: parallel_partner(i, idents) for i in idents}
    separation = {}
    for i, frame in zip(idents, frames):
        other = partners[i]
        if other:
            j = frames[idents.index(other)]
            separation[f"{i}/{other}"] = round(abs(frame.project(TrackPoint(j.lat, j.lon, j.elevation_m)).cross_m), 1)
    print(f"  parallel centreline separation (m): {separation} · the established rule uses "
          f"+/-{ESTABLISHED_CROSS_M:.0f} m", flush=True)

    changed, changed_to_partner, switch_km, goarounds, goaround_km = [], [], [], [], []
    other_runway_seen = Counter()
    started = time.time()
    for n, record in enumerate(records, 1):
        if n % 1000 == 0:
            print(f"  {n}/{len(records)} ({time.time() - started:.0f} s)", flush=True)
        landed = record["runway"]
        points, landing_row = read_track(store / record["file"])
        landed_frame = frames[idents.index(landed)]

        # one pass per threshold, strided
        rows = list(range(0, len(points), STRIDE))
        projected = {i: [f.project(points[k]) for k in rows] for i, f in zip(idents, frames)}
        landed_proj = projected[landed]

        # --- runway choice over the approach
        choice: list[tuple[float, str]] = []
        for index, k in enumerate(rows):
            if k > landing_row:
                break
            here = landed_proj[index]
            if not (INNER_M < here.along_m <= RANGE_M and here.height_m <= CEILING_M):
                continue
            on = [(abs(projected[i][index].cross_m), i) for i in idents
                  if projected[i][index].along_m > INNER_M
                  and abs(projected[i][index].cross_m) <= ESTABLISHED_CROSS_M]
            if on:
                choice.append((here.along_m, min(on)[1]))
        # compress to runs of (runway, along at its start, along at its end); along DECREASES
        runs: list[list] = []
        for along, c in choice:
            if runs and runs[-1][0] == c:
                runs[-1][2] = along
            else:
                runs.append([c, along, along])
        runs = [r for r in runs if abs(r[1] - r[2]) >= MIN_RUN_M]
        if len(runs) > 1 and runs[-1][0] == landed:
            previous = runs[-2][0]
            other_runway_seen[previous] += 1
            changed.append(record["flight_key"])
            switch_km.append(round(runs[-2][2] / 1000.0, 2))       # where the other one was left
            if partners[landed] == previous:
                changed_to_partner.append(record["flight_key"])

        # --- go-around: low and near, then high again, then landing
        low_at = None
        for index, k in enumerate(rows):
            if k > landing_row:
                break
            here = landed_proj[index]
            if low_at is None:
                if here.height_m <= LOW_M and 0.0 < here.along_m <= NEAR_M:
                    low_at = here.along_m
            elif here.height_m >= LOW_M + CLIMB_M:
                goarounds.append(record["flight_key"])
                goaround_km.append(round(low_at / 1000.0, 2))
                break

    result = {
        "schema": "ts-approach-events-v1",
        "airport": airport_code,
        "landed_tracks": len(records),
        "thresholds": idents,
        "parallel_separation_m": separation,
        "established_cross_track_m": ESTABLISHED_CROSS_M,
        "parallels_separable": {k: v >= SEPARABLE_M for k, v in separation.items()},
        "side_step_eligible": {k: v <= SIDE_STEP_MAX_M for k, v in separation.items()},
        "settings": {"range_m": RANGE_M, "ceiling_m": CEILING_M, "inner_m": INNER_M,
                     "min_run_m": MIN_RUN_M, "stride": STRIDE,
                     "near_m": NEAR_M, "low_m": LOW_M, "climb_m": CLIMB_M},
        "runway_change": {
            "flights": len(changed), "share": round(len(changed) / len(records), 4),
            "to_parallel": len(changed_to_partner),
            "switch_km_p50": sorted(switch_km)[len(switch_km) // 2] if switch_km else None,
            "switch_km_min": min(switch_km) if switch_km else None,
            "previous_runway": dict(other_runway_seen),
            "examples": changed[:20],
        },
        "go_around": {
            "flights": len(goarounds), "share": round(len(goarounds) / len(records), 4),
            "low_km_p50": sorted(goaround_km)[len(goaround_km) // 2] if goaround_km else None,
            "examples": goarounds[:20],
        },
        "elapsed_s": round(time.time() - started, 1),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "approach_events.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nRUNWAY CHANGE {len(changed)}/{len(records)} ({result['runway_change']['share']:.3%}), "
          f"of which to the parallel {len(changed_to_partner)}; switch p50 "
          f"{result['runway_change']['switch_km_p50']} km", flush=True)
    print(f"GO-AROUND {len(goarounds)}/{len(records)} ({result['go_around']['share']:.3%}); "
          f"low at p50 {result['go_around']['low_km_p50']} km", flush=True)
    print(f"written {out / 'approach_events.json'} in {result['elapsed_s']:.0f} s", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--airport", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config-file", type=Path,
                        default=REPO_ROOT / "trajectory_data_process" / "config" / "runway_thresholds.json")
    parser.add_argument("--cifp-file", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="a PREFIX of the roster (a smoke test)")
    args = parser.parse_args(argv)
    measure(args.airport.upper(), args.out, args.config_file, args.cifp_file, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
