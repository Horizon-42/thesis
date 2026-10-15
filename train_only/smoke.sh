#!/bin/bash
# The training chain at a small size, before a formal chain on a new machine: every runner of B, C and D once, on the
# formal inputs, each told it is a smoke (no result; a tree with changes allowed). B: a fold's training run (2 epochs on
# 20 sentences of each airport and split) and its free generation, the base's training run, its validation and free
# generation readouts on the select days (a smoke never reads the val days, D85). C: two rounds of post_train from the
# start. D: two rounds of multi_train from the start, and two from C's last round. The start: B21's base when it is in
# place (train_only/data_paths.txt carries it), else this smoke's own base (it lands little, so C and D may update
# nothing; every step still runs).
#
#   train_only/smoke.sh [device]          cuda (default) or cpu; the output goes to 4dTrajectory/outputs/POOLED/smoke_<UTC>
#
# Measured on the RTX 4060 host with the CPU (2026-10-15): about 35 min.
set -u
DEVICE=${1:-cuda}
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$ROOT"
P=4dTrajectory/outputs/POOLED
O=$P/smoke_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p $O
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
I=$P/instruction_language/v19_20261014; E=$P/executor/v24_20261014; WIN=$P/post/windows_20261013
step() { echo "=== $1 $(date -u +%H:%M:%S)"; }
fail() { echo "=== STOP: $1 failed $(date -u +%H:%M:%S)"; exit 1; }
B=(--instructions $I --executor $E)
TRAIN=(--row-interval-s 4 --variant full --selection landed --device $DEVICE --sample 20 --max-epochs 2)
GEN=(--samples 2 --seed 1337 --chunk 400 --device $DEVICE --smoke)
step "B fold: prior_train"
python run_ts.py prior_train "${B[@]}" "${TRAIN[@]}" --held-out KMSY --seed 1337 --out $O/prior/A_full_s1337/KMSY || fail B1
step "B fold: prior_free_generation"
python run_ts.py prior_free_generation --prior $O/prior/A_full_s1337/KMSY "${B[@]}" --split select --airports KMSY \
    --per-airport 5 "${GEN[@]}" --out $O/prior/A_full_s1337/KMSY/free_generation || fail B2
step "B base: prior_train"
python run_ts.py prior_train "${B[@]}" "${TRAIN[@]}" --seed 1337 --out $O/prior/base/run || fail B3
step "B base: prior_validation"
python run_ts.py prior_validation --prior $O/prior/base/run "${B[@]}" --split select --device $DEVICE --smoke \
    --out $O/prior/base/validation || fail B4
step "B base: prior_free_generation"
python run_ts.py prior_free_generation --prior $O/prior/base/run "${B[@]}" --split select --per-airport 5 "${GEN[@]}" \
    --out $O/prior/base/free_generation || fail B5
BASE=$P/prior/prior_b21_20261014/base/run
[ -f $BASE/procedure_masks.json ] || BASE=$O/prior/base/run
echo "C and D start from $BASE"
COMMON=(--prior $BASE --instructions $I --executor $E --windows $WIN --device $DEVICE --smoke
        --prior-lr 1e-5 --traffic-lr 1e-4 --weight-decay 0.01 --update-groups 4 --data-sentences 4 --select-per-airport 2)
step "C: post_train"
python run_ts.py post_train --method branch --segment-only --branch-every-s 60 "${COMMON[@]}" --out $O/post \
    --rounds 2 --windows-real 2 --windows-a 2 --windows-d 2 --windows-b 2 --batch-windows 4 --continuations 2 \
    --seed 1337 --traffic-hidden 64 --traffic-heads 4 --select-seed 1337 || fail C
D=(--front-attention merged --rounds 2 --windows-real 2 --windows-compressed 2 --spans-s 300 600 1200 --c-min 0.8
   --batch-rows 64 32 24 --continuations 2 --seed 2035)
step "D from the start: multi_train"
python run_ts.py multi_train --branching grid --branch-every-s 60 --segment-only "${COMMON[@]}" "${D[@]}" \
    --traffic-hidden 64 --traffic-heads 4 --out $O/multi_base || fail D1
step "D from C: multi_train"
python run_ts.py multi_train --branching majority "${COMMON[@]}" "${D[@]}" --start-campaign $O/post --start-round 1 \
    --out $O/multi_c || fail D2
step "ALL DONE: $O"
