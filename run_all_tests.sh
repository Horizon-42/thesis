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
# All suites are expected to pass (modeling+backend and aeroviz-4d/python both exit 0).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Activate the thesis conda env (casadi etc. aren't in system python). Resolution rules —
# including why candidate envs are probed for casadi instead of trusted by name — live in
# the shared helper, which the fullstack launcher uses too.
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/activate_aeroviz_env.sh"
if ! aeroviz_activate_env; then
  echo "warning: continuing with the current python ($(command -v python))" >&2
fi

# Modeling + backend suites (geokit, aircraft, flight_scenarios, optimizer, backend, …).
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
