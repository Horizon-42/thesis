#!/usr/bin/env bash
#
# Run every Python test suite in the thesis project.
#
# Why three pytest invocations instead of one:
#   - Test files in different dirs share names (test_frame.py) and the test
#     dirs have no __init__.py, so pytest's default importer collides. The modeling +
#     backend group works around this with --import-mode=importlib.
#   - aeroviz-4d/python has its own pytest.ini (pythonpath/testpaths), so it runs
#     on its own.
#   - ts_transformer's suite is split off the modeling group only so each group can be given
#     its own number of workers.
#
# The three groups run AT THE SAME TIME, and each is spread over pytest-xdist workers
# (--dist worksteal: a free worker takes the next test, so one slow file does not set the
# time). Torch / MKL / OpenMP threads are pinned to 1 per worker — the tests are small
# tensors, and N workers x the default 20 torch threads would fight over the cores.
# casadi is not thread-safe, which is why this is processes, never threads.
# Each group's output goes to its own log (path printed at the start) and is replayed in
# order at the end; follow one live with `tail -f`.
#
# Usage:
#   ./run_all_tests.sh            # run everything
#   ./run_all_tests.sh -x         # extra args are forwarded to every pytest (stop on first fail)
#   ./run_all_tests.sh -k rollout # ...or filter by keyword
#   TS_TEST_WORKERS=0 MODELING_TEST_WORKERS=0 FRONTEND_TEST_WORKERS=0 ./run_all_tests.sh   # all serial
# Workers per group (defaults 8 / 6 / 2; 0 = that group serial, still beside the others).
#
# Needs pytest-xdist in the thesis env (installed 2026-10-04) unless every group is 0.
# Tests that read the git-ignored `data/` tree (CIFP) need it present — in a worktree, symlink it.
#
# Environment: needs the thesis conda env — casadi (optimizer, backend, aero model) AND
# torch (the ts_transformer suite, collected under 4dTrajectory). Resolution is delegated
# to scripts/activate_aeroviz_env.sh; see CLAUDE.md "Environment" for why the env is
# probed for casadi rather than trusted by name.
#
# Expected result: every suite exits 0 (since 2026-09-25: the numpy-2 test and the reference-record fixture that
# failed before are fixed). Any failure is a real regression, and the exit code says so.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Activate the thesis conda env (casadi etc. aren't in system python). Resolution rules —
# including why candidate envs are probed for casadi instead of trusted by name — live in
# the shared helper, which the fullstack launcher uses too.
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/activate_aeroviz_env.sh"
if ! aeroviz_activate_env; then
  # Not fatal — a system python may still run some suites — but say what is coming, so a
  # wall of ImportErrors reads as "wrong env" rather than as broken code.
  echo "warning: continuing with the current python ($(command -v python))" >&2
  echo "         expect collection errors from every suite needing casadi or torch" >&2
else
  echo "env: $CONDA_DEFAULT_ENV ($(command -v python))"
fi

# ts_transformer: collected under 4dTrajectory, needs torch as well as casadi.
TS_SUITE=4dTrajectory/ts_transformer/tests
TS_WORKERS="${TS_TEST_WORKERS:-8}"
MODELING_WORKERS="${MODELING_TEST_WORKERS:-6}"
FRONTEND_WORKERS="${FRONTEND_TEST_WORKERS:-2}"

# The other modeling + backend suites (geokit, aircraft, flight_scenarios, optimizer, backend).
MODELING_SUITES=(
  aerodynamic_model/tests
  4dTrajectory
  aircraft
  evaluation/tests
  flight_scenarios
  geokit/tests
  trajectory_data_process
  aeroviz_backend/tests
)

if { [ "$TS_WORKERS" -gt 0 ] || [ "$MODELING_WORKERS" -gt 0 ] || [ "$FRONTEND_WORKERS" -gt 0 ]; } \
   && ! python -c "import xdist" 2>/dev/null; then
  echo "error: pytest-xdist is not installed in this env; install it or set every *_TEST_WORKERS=0" >&2
  exit 2
fi

LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/run_all_tests.XXXXXX")"
echo "logs: $LOG_DIR   (ts.log, modeling.log, frontend.log)"

# workers_args <n>: the xdist options for n workers, nothing for 0.
workers_args() {
  if [ "$1" -gt 0 ]; then echo "-n $1 --dist worksteal"; fi
}

(
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  # shellcheck disable=SC2046
  python -m pytest --import-mode=importlib $(workers_args "$TS_WORKERS") "$TS_SUITE" "$@"
) > "$LOG_DIR/ts.log" 2>&1 &
pid_ts=$!

(
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  # shellcheck disable=SC2046
  python -m pytest --import-mode=importlib $(workers_args "$MODELING_WORKERS") \
    --ignore="$TS_SUITE" "${MODELING_SUITES[@]}" "$@"
) > "$LOG_DIR/modeling.log" 2>&1 &
pid_modeling=$!

(
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  # shellcheck disable=SC2046
  python -m pytest --continue-on-collection-errors $(workers_args "$FRONTEND_WORKERS") aeroviz-4d/python/tests "$@"
) > "$LOG_DIR/frontend.log" 2>&1 &
pid_frontend=$!

wait "$pid_ts";       rc_ts=$?
wait "$pid_modeling"; rc_modeling=$?
wait "$pid_frontend"; rc_frontend=$?

show() {  # show <title> <log>
  echo
  echo "=================================================================="
  echo " $1"
  echo "=================================================================="
  cat "$2"
}
show "1/3  ts_transformer suite ($TS_WORKERS workers)" "$LOG_DIR/ts.log"
show "2/3  Other modeling + backend suites ($MODELING_WORKERS workers)" "$LOG_DIR/modeling.log"
show "3/3  aeroviz-4d/python suite ($FRONTEND_WORKERS workers)" "$LOG_DIR/frontend.log"

echo
echo "=================================================================="
echo " Summary"
echo "=================================================================="
echo "  ts_transformer     : exit $rc_ts"
echo "  modeling + backend : exit $rc_modeling"
echo "  aeroviz-4d/python  : exit $rc_frontend"
echo "  logs               : $LOG_DIR"

# Exit non-zero if any suite reported a problem.
if [ "$rc_ts" -ne 0 ] || [ "$rc_modeling" -ne 0 ] || [ "$rc_frontend" -ne 0 ]; then
  exit 1
fi
