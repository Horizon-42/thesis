#!/usr/bin/env bash
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../../../../.."   # the repository (README §2)
SCR=4dTrajectory/outputs/POOLED/spai_copy_20261011/plant_comparison   # the results (was the session's scratch area)
run() { name=$1; shift; rm -rf $SCR/sweep_$name; python "$HERE/compare_plants.py" --per-airport 100 --out $SCR/sweep_$name "$@" > $SCR/sweep_$name.log 2>&1 && echo "$name ok" || echo "$name FAILED"; }
run all_0p25  --taus 0.25 0.25 0.25
run half      --taus 0.75 1.0 0.4
run bank_only --taus 0.25 2.0 0.25
run thrust_only --taus 1.5 0.25 0.25
run load_only --taus 0.25 0.25 0.8
echo sweep-done
