#!/bin/bash
# The tests of the training chain's code (stages A–D: instructions, autopilot, prior, post, multi, their runners), not
# the rest of the repository's suite: the Training exports (they need the frontend's fixtures), the one-tier paths, the
# optimizer, the frontend and the backend are left out — this branch does not run them.
#
#   train_only/run_tests.sh [pytest options]       e.g. -x, or -k <expr>; workers: TEST_WORKERS (default 8)
set -euo pipefail
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$ROOT"
T=4dTrajectory/ts_transformer/tests
FILES=($T/test_architecture.py $T/test_autopilot.py $T/test_closed_loop.py $T/test_day_split.py
       $T/test_executor_conformance.py $T/test_instruction_conformance.py $T/test_instruction_labeller.py
       $T/test_instruction_vocabulary.py $T/test_separation.py $T/test_start.py $T/test_window_behaviour.py
       $T/test_window_lists.py $T/test_windows.py)
for f in $T/test_prior_*.py $T/test_post_*.py $T/test_multi_*.py; do
    [[ $f == *training_export* ]] || FILES+=("$f")
done
OMP_NUM_THREADS=1 python -m pytest "${FILES[@]}" -q --import-mode=importlib -n "${TEST_WORKERS:-8}" --dist worksteal "$@"
