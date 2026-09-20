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

import math

from final_approach.frame import TrackPoint
from trajectory_data_process.harvest.airports import load_airport
from trajectory_data_process.harvest.altitude_filter import filter_altitude_outliers
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
#: What the PUBLISHED final approach course actually is, on the final — the 500 m above is ours,
#: an analysis threshold for "on the extended centreline versus on a downwind", and it is 5x too
#: wide to tell two parallels apart. AIM 1-1-18 d 4: "The width of the final approach course is
#: tailored so that the total width is usually 700 feet at the runway threshold" -> +/-350 ft.
#: (The same paragraph: lateral integrity is 0.3 NM = 556 m for LNAV, and 40 m for LPV.)
FINAL_HALF_M = 350.0 * 0.3048
FINAL_RANGE_M = 3_000.0
#: The established rule's THIRD condition, which the first reading left out: the track must lie
#: along the course. Without it a crossing runway (KRDU 14 against 05) wins the arg-min near the
#: airport although the aircraft is 90 degrees off it.
TRACK_TOLERANCE_DEG = 30.0
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


def read_track(path: Path) -> tuple[list[TrackPoint], int, int]:
    """The stored samples as ``[t, lon, lat, alt]``, the landing row, and how many altitudes
    the read-time repair fixed.

    The altitude-outlier repair is READ-TIME in this repository and `tracks/` is never edited,
    so every reader has to apply it. Reading the store raw put a 9125 m sample in the middle of
    a 3.9 km final and made this runner call an ordinary approach a go-around.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    repaired = filter_altitude_outliers(data["samples"])
    return ([TrackPoint(lat=s[2], lon=s[1], alt_m=s[3]) for s in repaired.samples],
            int(data["landing_sample_index"]), len(repaired.outliers))


def measure(airport_code: str, out: Path, config_file: Path, cifp_file: Path, limit: int,
            cohort: Path | None = None) -> dict:
    airport = load_airport(airport_code, config_file=config_file, cifp_file=cifp_file)
    frames = airport.frames("hae")
    idents = [f.ident for f in frames]
    store = REPO_ROOT / "trajectory_data_process" / "outputs" / "harvest" / airport_code / "tracks"
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    records = [r for r in manifest["records"] if r["outcome"] == "assigned"]
    if cohort is not None:
        # B0′′: the runway word's check runs on the MODELLING cohort, not the fleet — the word is
        # only ever read for these flights, and the fleet's rate says nothing about them.
        roster = json.loads(cohort.read_text(encoding="utf-8"))["splits"]
        keys = {k.split(":", 1)[1] for part in roster.values() for k in part}
        records = [r for r in records if r["flight_key"] in keys]
        print(f"  restricted to the cohort: {len(records)} of {len(keys)} rostered flights", flush=True)
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
    on_final, final_mismatch, no_final_course = Counter(), [], 0
    outliers_repaired = 0
    started = time.time()
    for n, record in enumerate(records, 1):
        if n % 1000 == 0:
            print(f"  {n}/{len(records)} ({time.time() - started:.0f} s)", flush=True)
        landed = record["runway"]
        points, landing_row, repaired = read_track(store / record["file"])
        outliers_repaired += repaired
        landed_frame = frames[idents.index(landed)]

        # one pass per threshold, strided.
        # SIGN: RunwayFrame's along axis points ALONG THE LANDING DIRECTION, so an aircraft short
        # of the threshold projects NEGATIVE and `-along_m` is its distance to go. Reading it the
        # other way silently measures the rows PAST the threshold (and the reciprocal end of the
        # same pavement, whose along is large and positive) — which is what the first two readings
        # of this runner did, and why they were thrown away.
        rows = list(range(0, len(points), STRIDE))
        projected = {i: [f.project(points[k]) for k in rows] for i, f in zip(idents, frames)}

        def to_go(ident: str, index: int) -> float:
            return -projected[ident][index].along_m

        def aligned(ident: str, index: int) -> bool:
            """The track lies along this course: the established rule's third condition."""
            j = min(index + 1, len(rows) - 1)
            here, nxt = projected[ident][index], projected[ident][j]
            closing, drift = nxt.along_m - here.along_m, nxt.cross_m - here.cross_m
            if closing <= 0.0 or math.hypot(closing, drift) < 1.0:
                return False                                    # going away, or not moving
            return abs(math.degrees(math.atan2(drift, closing))) <= TRACK_TOLERANCE_DEG

        def chosen(index: int, half_m: float, near_m: float, far_m: float) -> str | None:
            """The course this row expresses: nearest centreline the aircraft is ON and tracking."""
            on = [(abs(projected[i][index].cross_m), i) for i in idents
                  if near_m < to_go(i, index) <= far_m
                  and abs(projected[i][index].cross_m) <= half_m
                  and aligned(i, index)]
            return min(on)[1] if on else None

        usable = [index for index, k in enumerate(rows) if k <= landing_row]

        # --- which runway was it being vectored to (wide corridor, outside the crossing zone)
        choice: list[tuple[float, str]] = []
        for index in usable:
            d = to_go(landed, index)
            if not (INNER_M < d <= RANGE_M and projected[landed][index].height_m <= CEILING_M):
                continue
            c = chosen(index, ESTABLISHED_CROSS_M, INNER_M, RANGE_M)
            if c:
                choice.append((d, c))
        runs: list[list] = []
        for d, c in choice:
            if runs and runs[-1][0] == c:
                runs[-1][2] = d
            else:
                runs.append([c, d, d])
        runs = [r for r in runs if abs(r[1] - r[2]) >= MIN_RUN_M]
        if len(runs) > 1 and runs[-1][0] == landed:
            previous = runs[-2][0]
            other_runway_seen[previous] += 1
            changed.append(record["flight_key"])
            switch_km.append(round(runs[-2][2] / 1000.0, 2))
            if partners[landed] == previous:
                changed_to_partner.append(record["flight_key"])

        # --- which PUBLISHED course did it actually line up on, on the final
        final_course = None
        for index in usable:
            if 0.0 < to_go(landed, index) <= FINAL_RANGE_M:
                c = chosen(index, FINAL_HALF_M, 0.0, FINAL_RANGE_M)
                if c:
                    final_course = c
        if final_course is None:
            no_final_course += 1
        else:
            on_final[final_course == landed] += 1
            if final_course != landed:
                final_mismatch.append((record["flight_key"], final_course, landed))

        # --- go-around: low AND near, then a climb, then the landing
        low_at = None
        for index in usable:
            here = projected[landed][index]
            d = to_go(landed, index)
            if low_at is None:
                if here.height_m <= LOW_M and 0.0 < d <= NEAR_M:
                    low_at = d
            elif here.height_m >= LOW_M + CLIMB_M:
                goarounds.append(record["flight_key"])
                goaround_km.append(round(low_at / 1000.0, 2))
                break

    result = {
        "schema": "ts-approach-events-v1",
        "airport": airport_code,
        "landed_tracks": len(records), "cohort": str(cohort) if cohort else None,
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
        "final_course": {
            "matched_landed_runway": on_final[True],
            "other_course": on_final[False],
            "no_course_held": no_final_course,
            "half_width_m": round(FINAL_HALF_M, 1),
            "range_m": FINAL_RANGE_M,
            "examples": [{"flight": f, "lined_up_on": c, "landed_on": l} for f, c, l in final_mismatch[:20]],
        },
        "go_around": {
            "flights": len(goarounds), "share": round(len(goarounds) / len(records), 4),
            "low_km_p50": sorted(goaround_km)[len(goaround_km) // 2] if goaround_km else None,
            "examples": goarounds[:20],
        },
        "altitudes_repaired": outliers_repaired,
        "elapsed_s": round(time.time() - started, 1),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "approach_events.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nRUNWAY CHANGE {len(changed)}/{len(records)} ({result['runway_change']['share']:.3%}), "
          f"of which to the parallel {len(changed_to_partner)}; switch p50 "
          f"{result['runway_change']['switch_km_p50']} km", flush=True)
    print(f"FINAL COURSE (+/-{FINAL_HALF_M:.0f} m, the published 700 ft width) matched {on_final[True]}, "
          f"other {on_final[False]}, none held {no_final_course}", flush=True)
    print(f"GO-AROUND {len(goarounds)}/{len(records)} ({result['go_around']['share']:.3%}); "
          f"low at p50 {result['go_around']['low_km_p50']} km", flush=True)
    print(f"altitudes repaired at read: {outliers_repaired}", flush=True)
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
    parser.add_argument("--cohort", type=Path, default=None,
                        help="a development cohort: measure only its flights (B0′′ reads the runway "
                             "word's agreement on the flights the word is actually read for)")
    args = parser.parse_args(argv)
    measure(args.airport.upper(), args.out, args.config_file, args.cifp_file, args.limit, args.cohort)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
