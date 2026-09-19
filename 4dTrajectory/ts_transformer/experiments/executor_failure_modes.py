"""Two-tier v3 A2 (§5.2): how the flights that did not cross failed — read off one lockstep's records.

    python run_ts.py executor_failure_modes --lockstep <dir> --out <dir> [--stratum vectored] [--per-mode 3]

``--lockstep`` is a `manoeuvre_lockstep` output directory written WITH ``--write-records``
(``manoeuvre_lockstep.json`` beside ``records/``). Every flight of the stratum (the lockstep row's
own covariates, `approach_difficulty.strata_masks`; default the vectored group) that did NOT
cross the threshold on the final (the row's reference verdict) is read in the runway's course
frame (`manoeuvre/failure_modes.py`; "established" in the mode names is that module's
centreline rule, not the crossing verdict):
where it ended, whether it was ever established, when each side first aligned, the turn delay,
the speed and height error at the end — and one failure mode per flight. Writes
``failure_modes.json`` (the per-flight rows and the per-mode table), ``failure_modes.txt``, and
``records_<mode>/`` — the first ``--per-mode`` flights of each mode as a record directory the
publisher takes (`--category-variant`), with a ``failure_modes`` summary block. Refuses an
existing ``--out``. Not a gate: the table and at most three hypotheses go to the results doc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.records import load_record


from ts_transformer.data.approach_difficulty import STRATUM_ALL, STRATUM_STRAIGHT_IN, STRATUM_VECTORED, strata_masks
from ts_transformer.experiments.manoeuvre_lockstep import LOCKSTEP_SCHEMA
from ts_transformer.experiments.support import REPO_ROOT
from ts_transformer.inference.export import EVAL_SUFFIX, REFERENCE_EVAL_SUFFIX, REFERENCES_DIR, copy_record_subset
from ts_transformer.io_utils import file_sha256, utc_now, write_json_atomic
from ts_transformer.manoeuvre.failure_modes import FAILURE_MODES, flight_failure, summarise

FAILURE_MODES_SCHEMA = "ts-executor-failure-modes-v1"
#: The summary block a per-mode record directory carries (the publisher's variant block).
FAILURE_MODES_BLOCK = "failure_modes"
STRATA = {"vectored": STRATUM_VECTORED, "straight-in": STRATUM_STRAIGHT_IN, "all": STRATUM_ALL}


def not_crossed_flights(payload: dict[str, Any], stratum: str) -> tuple[list[str], int]:
    """``(the lockstep rows of ``stratum`` that did not cross the threshold on the final, in row
    order; the stratum's size)``."""
    rows = payload["rows"]
    keys = list(rows)
    mask = strata_masks(rows, keys)[STRATA[stratum]]
    return [key for key, keep in zip(keys, mask) if keep and not rows[key]["reference"]["established"]], int(mask.sum())


def render(payload: dict[str, Any]) -> str:
    table = payload["modes"]
    def num(value, width, digits=0):
        return f"{value:>{width}.{digits}f}" if value is not None else f"{'—':>{width}}"

    lines = [
        f"executor failure modes · {payload['stratum']} · {payload['flights']} of {payload['stratum_flights']} "
        f"{payload['stratum']} flights did not cross the threshold · {payload['lockstep']}",
        "",
        f"{'mode':<20}{'n':>5}{'share':>7}{'ever est':>10}{'to-go start':>13}{'to-go end':>11}{'|cross| p50':>13}{'right':>7}"
        f"{'turn delay':>12}{'ΔV p50':>8}{'Δh p50':>8}",
    ]
    for mode in FAILURE_MODES:
        cell = table[mode]
        if not cell["n"]:
            continue
        lines.append(f"{mode:<20}{cell['n']:>5}{cell['share']:>7.2f}{num(cell['ever_established_share'], 10, 2)}"
                     f"{num(cell['to_go_start_p50_m'], 13)}{num(cell['to_go_end_p50_m'], 11)}{num(cell['cross_end_abs_p50_m'], 13)}"
                     f"{num(cell['right_side_share'], 7, 2)}{num(cell['turn_delay_p50_s'], 12)}{num(cell['speed_error_p50_mps'], 8, 1)}"
                     f"{num(cell['altitude_error_p50_m'], 8)}")
    lines.append("")
    lines.append("ever est: share ever established under the centreline rule (cross-track < 500 m, heading within 30°, ahead of the "
                 "threshold) — not the crossing verdict; to-go: metres before the threshold along the final approach course at the "
                 "first prediction / the budget's end (negative = past its abeam line); |cross|: off the course at the end; right: share ending "
                 "right of the course; turn delay: flown minus truth onset of the final turn onto the course (s); ΔV / Δh: flown minus "
                 "truth at the end")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--lockstep", type=Path, required=True, help="a manoeuvre_lockstep output directory with records/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stratum", default="vectored", choices=sorted(STRATA))
    parser.add_argument("--per-mode", type=int, default=3, help="flights of each mode copied to records_<mode>/ for the picker")
    args = parser.parse_args(argv)
    lockstep = args.lockstep if args.lockstep.is_absolute() else REPO_ROOT / args.lockstep
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out.exists():
        parser.error(f"{out} exists; a readout is never overwritten")
    artefact, records = lockstep / "manoeuvre_lockstep.json", lockstep / "records"
    if not (records / "summary.json").is_file():
        parser.error(f"{records} holds no records; fly the lockstep with --write-records")
    payload = json.loads(artefact.read_text(encoding="utf-8"))
    if payload.get("schema") != LOCKSTEP_SCHEMA:
        # this runner reads what this code writes, nothing older (the repo rule of 2026-09-19: no compatibility)
        parser.error(f"{artefact}: lockstep schema {payload.get('schema')!r} is not {LOCKSTEP_SCHEMA!r}; fly the reading again with this code")
    keys, stratum_n = not_crossed_flights(payload, args.stratum)
    rows: dict[str, dict[str, Any]] = {}
    for key in keys:
        row = payload["rows"][key]
        stem = row["flight_id"]
        flown = load_record(records / f"{stem}{EVAL_SUFFIX}")
        truth = load_record(records / REFERENCES_DIR / f"{stem}{REFERENCE_EVAL_SUFFIX}")
        rows[key] = {"flight_id": stem, "ended": row["ended"], "predictions": row["predictions"],
                     **flight_failure(flown.states, truth.states, flown.target_state)}
    modes = summarise(rows)
    result = {
        "schema": FAILURE_MODES_SCHEMA, "written_utc": utc_now(),
        "lockstep": str(artefact), "lockstep_sha256": file_sha256(artefact), "protocol": payload["protocol"],
        "executor": payload["executor"], "first_prediction": payload["first_prediction"],
        "segment_s": payload["segment_s"], "executed_s": payload["executed_s"],
        "stratum": args.stratum, "stratum_flights": stratum_n, "flights": len(rows),
        "modes": modes, "rows": rows,
    }
    out.mkdir(parents=True)
    write_json_atomic(out / "failure_modes.json", result)
    table = render(result)
    (out / "failure_modes.txt").write_text(table, encoding="utf-8")
    print(table)
    for mode in FAILURE_MODES:
        chosen = modes[mode]["flights"][: args.per_mode]
        if not chosen:
            continue
        block = {FAILURE_MODES_BLOCK: {"mode": mode, "stratum": args.stratum, "lockstep": str(artefact), "lockstep_sha256": result["lockstep_sha256"],
                                       "mode_flights": modes[mode]["n"], "protocol": payload["protocol"], "first_prediction": payload["first_prediction"],
                                       "segment_s": result["segment_s"], "executed_s": result["executed_s"]}}
        copy_record_subset(records, out / f"records_{mode}", [rows[key]["flight_id"] for key in chosen], extra_summary=block)
        print(f"  records_{mode}: {len(chosen)} of {modes[mode]['n']} flights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
