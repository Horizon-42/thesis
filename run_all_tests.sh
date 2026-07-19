#!/usr/bin/env bash
#
# Run every Python test suite in the thesis project.
#
# Why two pytest invocations instead of one:
#   - Two suites share the filename test_single_shooting_optimizor.py and the test
#     dirs have no __init__.py, so pytest's default importer collides. The modeling +
#     backend group works around this with --import-mode=importlib.
#   - aeroviz-4d/python has its own pytest.ini (pythonpath/testpaths), so it runs
#     on its own.
#
# Usage:
#   ./run_all_tests.sh            # run everything
#   ./run_all_tests.sh -x         # extra args are forwarded to pytest (stop on first fail)
#   ./run_all_tests.sh -k rollout # ...or filter by keyword
#
# Environment: needs the thesis conda env — casadi (optimizer, backend, aero model) AND
# torch (the ts_transformer suite, collected under 4dTrajectory). Resolution is delegated
# to scripts/activate_aeroviz_env.sh; see CLAUDE.md "Environment" for why the env is
# probed for casadi rather than trusted by name.
#
# Expected result: aeroviz-4d/python exits 0. Modeling+backend exits 1 on ONE known,
# pre-existing, unrelated failure —
#   4dTrajectory/optimization/collocation/tests/test_optimizer.py
#     ::test_fixed_time_objective_weights_control_effort_at_one
#   TypeError: only 0-dimensional arrays can be converted to Python scalars
# (a numpy scalar-conversion deprecation). Anything beyond that one is a real regression.

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

# Modeling + backend suites (geokit, aircraft, flight_scenarios, optimizer, backend, and
# the learned-prediction suite — ts_transformer/tests is collected under 4dTrajectory,
# which is why this group needs torch as well as casadi).
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

echo "=================================================================="
echo " 1/2  Modeling + backend suites"
echo "=================================================================="
python -m pytest --import-mode=importlib "${MODELING_SUITES[@]}" "$@"
rc_modeling=$?

echo
echo "=================================================================="
echo " 2/2  aeroviz-4d/python suite"
echo "=================================================================="
python -m pytest --continue-on-collection-errors aeroviz-4d/python/tests "$@"
rc_frontend=$?

echo
echo "=================================================================="
echo " Summary"
echo "=================================================================="
echo "  modeling + backend : exit $rc_modeling"
echo "  aeroviz-4d/python  : exit $rc_frontend"

# Exit non-zero if either suite reported a problem.
if [ "$rc_modeling" -ne 0 ] || [ "$rc_frontend" -ne 0 ]; then
  exit 1
fi
