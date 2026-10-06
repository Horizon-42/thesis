"""The batch driver behind every scenario batch: process pool, resume, stale sweep, summary.

:func:`run_batch` solves each scenario through a caller's process-pool ``worker`` and writes one
``*_states.json`` + ``*_eval.json`` pair per scenario (a failed one gets an eval record with empty
lists) and one ``summary.json``. The record filenames (:func:`scenario_filename` and its siblings)
are defined here, once.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from flight_scenarios import FlightScenario, flight_key
from evaluation_export import (
    EVAL_SUFFIX as _EVAL_SUFFIX,
    OBSERVED_TRACK_SUFFIX,
    REFERENCE_EVAL_SUFFIX as _REFERENCE_EVAL_SUFFIX,
    STATES_SUFFIX as _STATES_SUFFIX,
    failed_evaluation_record,
    state_dict,
    summary_row,
)


def resolve_jobs(jobs: int, n_tasks: int) -> int:
    """Resolve the requested worker count (``0`` ⇒ auto = half the CPU cores).

    Auto leaves cores free for other work; capped at the number of scenarios so we
    never spawn idle workers.
    """
    workers = jobs if jobs and jobs > 0 else max(1, (os.cpu_count() or 2) // 2)
    return max(1, min(workers, n_tasks)) if n_tasks else 1


#: The thread-pool variables of the numeric libraries a solve loads (BLAS, OpenMP, numexpr, Accelerate).
SOLVER_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def limit_solver_threads() -> None:
    """Pin each worker's BLAS/OpenMP pools to one thread to avoid oversubscription.

    With ``spawn`` (the macOS default) child processes inherit ``os.environ``, so
    setting these BEFORE the pool is created makes every worker import numpy / casadi
    single-threaded. Without it, N worker processes would each spin up a full BLAS
    thread pool and fight over the cores — *slowing* the batch down instead of
    speeding it up. ``setdefault`` respects any value the caller already exported.
    """
    for var in SOLVER_THREAD_ENV:
        os.environ.setdefault(var, "1")


# The record-filename suffixes are single-sourced in evaluation_export.py (imported above)
# — shared with ts_transformer/inference/export.py, which writes the same directory shape.


def sidecar_filename(states_name: str, suffix: str) -> str:
    """``<flight>_states.json`` → ``<flight><suffix>`` (a batch's per-record sidecar)."""
    return states_name.removesuffix(_STATES_SUFFIX) + suffix


def _clear_stale_records(out: Path, keep: set[str] | None = None, sidecar_suffix: str | None = None) -> None:
    """Delete leftover per-trajectory records from a previous batch in ``out``.

    A fresh batch writes one ``*_states.json`` + ``*_eval.json`` per CURRENT scenario;
    records from an earlier run over a DIFFERENT scenario set survive by filename and
    pollute everything that scans the directory (``python -m evaluation`` once counted
    27 orphans into a KRDU report). The ``references/`` subdirectory is untouched —
    the CLI writes the reference records immediately before the batch.

    ``keep`` (the resume path) spares the named files. Orphan removal is what makes the
    directory a faithful image of the roster, so it happens either way — resume narrows
    WHICH files survive, it never turns the sweep off.
    """
    keep = keep or set()
    suffixes = (_STATES_SUFFIX, _EVAL_SUFFIX) + ((sidecar_suffix,) if sidecar_suffix else ())
    stale = [path for suffix in suffixes for path in sorted(out.glob(f"*{suffix}")) if path.name not in keep]
    for path in stale:
        path.unlink()
    if stale:
        print(f"… cleared {len(stale)} record file(s) from a previous batch in {out}")


def _resumable_record(
    out: Path, scenario: FlightScenario, index: int,
    expected_config: dict[str, Any], sidecar_suffix: str | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """The summary row for one scenario's already-complete record pair, or ``None``.

    A 70k-solve batch runs for tens of hours and ``summary.json`` is only written at the
    end, so a crash at hour 25 used to discard every finished record with it. This reads
    one finished record back and rebuilds its roster row, so ``--resume`` re-runs only what
    is genuinely missing. It is deliberately strict — identity must match the scenario, a
    solved record must still have its states file, and the record's stamped
    ``optimization_config`` must equal THIS batch's — because a half-written record
    silently reused is worse than one re-solved. The config check is what keeps the
    ``--skip-optimize`` guarantee honest: without it, a resume across a changed
    ``--max-iterations``/``--fitting``/``--rollout-dt`` absorbed the old records and
    stamped the new config over the whole roster (records with no stamp — pre-guard
    batches — are re-solved for the same reason).
    """
    name = scenario_filename(scenario, index)
    eval_path = out / eval_filename(name)
    if not eval_path.is_file():
        return None
    try:
        record = json.loads(eval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("optimization_config") != expected_config:
        return None
    # A record solved against another target is not this scenario's result: a scenario file
    # regenerated under new target rules (the published approach speeds, 2026-09-24) keeps the
    # flight identity but moves the runway target's V, and the old record must be re-solved.
    if record.get("target_state") != state_dict(scenario.target):
        return None
    final_time = record.get("final_time_s")
    if final_time is None:
        return (name, _summary_record(
            scenario, status="failed", states_file=None,
            eval_file=eval_path.name, final_time_s=None,
            reason=record["reason"],
        ))
    if not (out / name).is_file():
        return None
    if sidecar_suffix and not (out / sidecar_filename(name, sidecar_suffix)).is_file():
        return None
    row = _summary_record(
        scenario, status="solved", states_file=name, eval_file=eval_path.name,
        final_time_s=float(final_time), reason=None,
    )
    chosen_iaf = record["source"].get("chosenIaf")
    if chosen_iaf is not None:
        row["chosenIaf"] = chosen_iaf
    return (name, row)


def write_failed_record(
    out: Path, scenario: FlightScenario, index: int, error: str, *,
    optimization_config: dict[str, Any], references_dir: str | None,
) -> dict[str, Any]:
    """An unsolved scenario's eval record (empty lists — how the evaluation computes the solve rate);
    returns its summary row."""
    eval_name = eval_filename(scenario_filename(scenario, index))
    failed_record = failed_evaluation_record(
        scenario.initial, scenario.target, scenario.source, error, subject="optimized",
    )
    failed_record["optimization_config"] = optimization_config
    if references_dir:
        failed_record["reference_file"] = f"{references_dir}/{reference_filename(scenario_filename(scenario, index))}"
    (out / eval_name).write_text(json.dumps(failed_record, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return _summary_record(scenario, status="failed", states_file=None, eval_file=eval_name,
                           final_time_s=None, reason=error)


def write_solved_record(
    out: Path, scenario: FlightScenario, index: int, result_dict: dict[str, Any], eval_dict: dict[str, Any], *,
    optimization_config: dict[str, Any], references_dir: str | None, sidecar_suffix: str | None,
) -> tuple[Path, dict[str, Any]]:
    """A solved scenario's ``*_states.json`` + ``*_eval.json`` (+ the sidecar ``result_dict["sidecar"]``
    when ``sidecar_suffix``); returns the states path and the summary row."""
    name = scenario_filename(scenario, index)
    eval_name = eval_filename(name)
    path = out / name
    if sidecar_suffix:
        result_dict = dict(result_dict)
        (out / sidecar_filename(name, sidecar_suffix)).write_text(
            json.dumps(result_dict.pop("sidecar"), separators=(",", ":"), allow_nan=False), encoding="utf-8")
    path.write_text(json.dumps(result_dict, separators=(",", ":")), encoding="utf-8")
    eval_dict = dict(eval_dict)
    eval_dict["states_ref"] = {"file": name, "key": "simulator_states"}
    eval_dict["states"] = []
    eval_dict["optimization_config"] = optimization_config
    if references_dir:
        eval_dict["reference_file"] = f"{references_dir}/{reference_filename(name)}"
    (out / eval_name).write_text(json.dumps(eval_dict, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    # The roster quotes the EVAL record's final_time_s — the replay's LAST sample. A
    # guard-truncated replay ends earlier than the planned NLP horizon, and resumed
    # rows are rebuilt from the eval record, so quoting the plan's horizon here made
    # a fresh row disagree with the same flight's resumed row.
    row = _summary_record(scenario, status="solved", states_file=name, eval_file=eval_name,
                          final_time_s=float(eval_dict["final_time_s"]), reason=None)
    chosen_iaf = result_dict["source"].get("chosenIaf")
    if chosen_iaf is not None:
        row["chosenIaf"] = chosen_iaf
    return path, row


def run_batch(
    scenarios: list[FlightScenario],
    *,
    output_dir: str | Path,
    worker: Any,
    params: dict[str, Any],
    optimization_config: dict[str, Any],
    mode: str | None,
    progress: str,
    jobs: int,
    scenarios_label: str | None,
    references_dir: str | None,
    resume: bool,
    sidecar_suffix: str | None = None,
) -> list[Path]:
    """The ONE batch driver behind both public entry points (unconstrained +
    constrained-IAF).

    ``worker`` is the process-pool function (payload → picklable record), ``params``
    its per-scenario keyword dict, ``optimization_config`` the persisted solver recipe —
    stamped into every eval record (what ``--resume`` verifies) and into
    ``summary.json`` (what ``--skip-optimize`` verifies). The two entry points used to
    be ~150-line near-duplicates, and several real bugs came from updating one and
    missing the other (the "batch edition" seam class in CLAUDE.md).

    Each scenario is an independent NLP solve, so they run across a process pool
    (``jobs`` workers; ``0`` ⇒ half the CPU cores). Processes — not threads — because
    the IPOPT solve is CPU-bound C++; a pool sidesteps the GIL entirely. All file IO
    and logging stay in the parent (collected as workers finish). Per-scenario ORDER
    varies with worker count, and a pooled run additionally pins each worker's BLAS
    pools to one thread (:func:`limit_solver_threads`) while a serial run does not —
    a borderline scenario can tip between solving and ``Maximum_Iterations_Exceeded``
    across that difference (a known open item; do not read bit-identical output into
    ``--jobs``).

    Infeasible / failed scenarios are **skipped and logged** (a real landings file mixes
    feasible approaches with too-slow or noisy ones), so one bad scenario never aborts
    the batch. A summary of failures is printed at the end.

    ``sidecar_suffix``: a worker's solved ``result_dict`` carries a ``"sidecar"`` payload, written
    beside the record as ``<flight><sidecar_suffix>`` (not into the states file); resume and the
    stale sweep treat it as part of the record.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Resume BEFORE the stale sweep: the rows we keep are exactly the files it must spare.
    records: dict[int, dict[str, Any]] = {}  # index -> summary record (parallel-safe ordering)
    resumed_files: set[str] = set()
    pending = list(range(len(scenarios)))
    if resume:
        pending = []
        for index, scenario in enumerate(scenarios):
            found = _resumable_record(out, scenario, index, optimization_config, sidecar_suffix)
            if found is None:
                pending.append(index)
                continue
            name, row = found
            records[index] = row
            resumed_files.update({name, eval_filename(name)})
            if sidecar_suffix:
                resumed_files.add(sidecar_filename(name, sidecar_suffix))
        if records:
            print(f"… resuming: {len(records)} record(s) already complete, "
                  f"{len(pending)} to solve")
    _clear_stale_records(out, keep=resumed_files, sidecar_suffix=sidecar_suffix)
    payloads = [(index, scenarios[index], params) for index in pending]
    workers = resolve_jobs(jobs, len(payloads))

    written: list[Path] = []
    failures: list[tuple[str, str]] = []

    def _handle(
        record: tuple[int, str, dict[str, Any] | None, dict[str, Any] | None, str | None],
    ) -> None:
        index, flight_id, result_dict, eval_dict, error = record
        scenario = scenarios[index]
        if error is not None:
            failures.append((flight_id, error))
            records[index] = write_failed_record(out, scenario, index, error, optimization_config=optimization_config,
                                                 references_dir=references_dir)
            print(f"✗ {flight_id}: skipped ({error.split(':', 1)[0]})")
            return
        path, row = write_solved_record(out, scenario, index, result_dict, eval_dict,
                                        optimization_config=optimization_config, references_dir=references_dir,
                                        sidecar_suffix=sidecar_suffix)
        written.append(path)
        chosen_iaf = row.get("chosenIaf")
        if chosen_iaf is not None:
            print(f"✓ {path.name}: IAF {chosen_iaf}, T={result_dict['final_time_s']:.1f}s")
        else:
            print(
                f"✓ {path.name}: optimizer {len(result_dict['optimizer_states'])} states, "
                f"simulator {len(result_dict['simulator_states'])} states, "
                f"T={result_dict['final_time_s']:.1f}s"
            )
        records[index] = row

    if not payloads:
        pass
    elif workers == 1:
        for payload in payloads:
            _handle(worker(payload))
    else:
        limit_solver_threads()
        print(f"… solving {len(payloads)} scenario(s){progress} "
              f"across {workers} worker process(es)")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, payload) for payload in payloads]
            for future in as_completed(futures):
                _handle(future.result())

    # One summary per run: the failure rate + which flights solved/failed, with the ids
    # needed to find each flight's reference (e.g. for the comparison CZML).
    total = len(scenarios)
    solved_rows = sum(1 for row in records.values() if row["status"] == "solved")
    summary: dict[str, Any] = {"scenarios": scenarios_label}
    if mode is not None:
        summary["mode"] = mode
    summary.update({
        "optimization_config": optimization_config,
        "total": total,
        "solved": solved_rows,
        "failed": total - solved_rows,
        "failure_rate": ((total - solved_rows) / total) if total else 0.0,
        "results": [records[i] for i in sorted(records)],
    })
    if len(records) != total:
        raise RuntimeError(
            f"roster is incomplete: {len(records)} row(s) for {total} scenario(s)"
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if failures:
        print(f"\n⚠ {len(failures)}/{total} scenario(s) skipped:")
        for flight_id, reason in failures[:15]:
            print(f"    {flight_id}: {reason}")
        if len(failures) > 15:
            print(f"    … and {len(failures) - 15} more")
    print(f"✓ solved {solved_rows}/{total} scenario(s){progress} "
          f"(failure rate {summary['failure_rate']:.1%}) -> {out}")
    print(f"  summary -> {out / 'summary.json'}")
    return written


def _summary_record(
    scenario: FlightScenario,
    *,
    status: str,
    states_file: str | None,
    eval_file: str | None,
    final_time_s: float | None,
    reason: str | None,
) -> dict[str, Any]:
    """One summary row: the flight's identity (so its reference can be found by id) + status.

    The row shape is single-sourced in ``evaluation_export.summary_row`` (shared with
    ts_transformer's batch writer).
    """
    return summary_row(
        scenario.source,
        status=status,
        states_file=states_file,
        eval_file=eval_file,
        final_time_s=final_time_s,
        reason=reason,
    )


def eval_filename(states_name: str) -> str:
    """``<flight>_states.json`` → ``<flight>_eval.json`` (same identity key)."""
    return states_name.removesuffix(_STATES_SUFFIX) + _EVAL_SUFFIX


def reference_filename(states_name: str) -> str:
    """``<flight>_states.json`` → ``<flight>_reference_eval.json`` (same identity key)."""
    return states_name.removesuffix(_STATES_SUFFIX) + _REFERENCE_EVAL_SUFFIX


def observed_track_filename(states_name: str) -> str:
    """``<flight>_states.json`` → ``<flight>_track.json`` (same identity key)."""
    return states_name.removesuffix(_STATES_SUFFIX) + OBSERVED_TRACK_SUFFIX


def scenario_filename(scenario: FlightScenario, index: int) -> str:
    """A unique, stable filename for one scenario's states JSON.

    A callsign + runway is NOT unique: the same aircraft can land more than once, and a
    callsign can recur across days — so ``id_runway`` silently overwrote sibling scenarios.
    The identity ``id_runway_icao24_landingTime`` is single-sourced in
    ``flight_scenarios.identity.flight_key`` — the SAME function keys ts_transformer's
    train/val/test split and record stems, so learned and optimized records for one flight
    always share a filename stem.
    """
    return f"{flight_key(scenario.source, index)}{_STATES_SUFFIX}"
