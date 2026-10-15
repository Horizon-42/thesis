#!/bin/bash
# The two-tier training chain on one machine: B (the prior's campaign: folds, choice, base, its one val read) -> C (stage
# C's post-training from B's base) -> D (stage D's multi-aircraft training from C's last round, or from B's base).
# Run from the repository root of a clone of the branch two-tier2-train-only, with the data of train_only/data_paths.txt
# in place (train_only/README.zh.md). A clean tree is required (the runners refuse a formal run from a tree with changes).
#
#   train_only/chain.sh <tag>          e.g. 20261020; the runs are named prior_<tag>, post_<tag>, multi_<tag>
#
# A resume after a stop or a crash: the same command again. Each runner resumes on its own (prior_campaign: the steps
# done are skipped; post_train / multi_train: the last checkpoint); a stage whose last round is there is skipped.
# The settings below are those of the last formal runs: B = B21's campaign; C = post_seg60's settings (the user's chosen
# stage C campaign) from the base; D = D200's arm (b), the grid every 60 s, credit to the next branch point. Change them
# to the method the user chooses after D200 (multi_control D200) before a formal run. New settings take a new tag: a
# stage whose last round is there is skipped whatever its settings, and the profiles of a tag are reused.
set -u
TAG=${1:?a tag for the run names, e.g. 20261020}
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$ROOT"
P=4dTrajectory/outputs/POOLED
I=$P/instruction_language/v19_20261014
E=$P/executor/v24_20261014
WIN=$P/post/windows_20261013
B=$P/prior/prior_$TAG
C=$P/post/post_$TAG
D=$P/multi/multi_$TAG
LOG=$P/train_only_$TAG.log
mkdir -p $P/prior $P/post $P/multi
# the speaking workers (2 or more each: multi_profile measures a worker beside another); the runners measure their
# memory and refuse a number that does not fit (O15, D191)
SPEAK_WORKERS=${SPEAK_WORKERS:-3}
CPU_WORKERS=${CPU_WORKERS:-4}
D_START=${D_START:-c}                       # c: D starts from C's last round; base: from B's base (D200)

# --- settings -----------------------------------------------------------------------------------------------------
C_ROUNDS=8
C_SETTINGS=(--method branch --segment-only --branch-every-s 60 --rounds $C_ROUNDS --windows-real 1000 --windows-a 1000
            --windows-d 1000 --windows-b 1000 --batch-windows 64 --continuations 8 --seed 2028 --prior-lr 1e-5
            --traffic-lr 1e-4 --weight-decay 0.01 --update-groups 4 --data-sentences 64 --select-per-airport 200
            --traffic-hidden 64 --traffic-heads 4 --select-seed 1337)
D_ROUNDS=4
D_SETTINGS=(--rounds $D_ROUNDS --windows-real 500 --windows-compressed 500 --spans-s 300 600 1200 --c-min 0.8
            --batch-rows 64 32 24 --continuations 8 --seed 2035 --prior-lr 1e-5 --traffic-lr 1e-4 --weight-decay 0.01
            --update-groups 4 --data-sentences 64 --select-per-airport 100 --front-attention merged
            --cpu-batches 8 --cpu-period 15 --gpu-squares 200 --branching grid --branch-every-s 60 --segment-only)
# -------------------------------------------------------------------------------------------------------------------

log() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
stop() { log "=== STOP: $*"; exit 1; }
test -z "$(git status --porcelain)" || stop "the tree has changes (git status); a formal run needs a clean tree"
export PYTHONUNBUFFERED=1
log "chain $TAG at $(git rev-parse HEAD)"

# B: the prior's campaign (R60): 15 folds, the choice, the base, its one val read
if grep -q "every step done" $B.log 2>/dev/null; then
    log "B: done before"
else
    log "B: prior_campaign -> $B"
    OMP_NUM_THREADS=1 python run_ts.py prior_campaign --instructions $I --executor $E --row-interval-s 4 --out $B \
        >> $B.log 2>&1 || stop "B failed (log $B.log)"
    grep -q "every step done" $B.log || stop "B ended without its last step (log $B.log)"
    log "B: done"
fi
BASE=$B/base/run
INPUTS=(--prior $BASE --instructions $I --executor $E --windows $WIN)

# C: stage C's post-training from the base (R65)
if [ -f $C/round_$((C_ROUNDS - 1))/checkpoint.pt ]; then
    log "C: done before"
else
    log "C: post_train -> $C"
    OMP_NUM_THREADS=1 python run_ts.py post_train "${INPUTS[@]}" "${C_SETTINGS[@]}" --speak-workers $SPEAK_WORKERS \
        --out $C >> $C.log 2>&1 || stop "C failed (log $C.log)"
    log "C: done"
fi

# D: the profiles (multi_profile, on the speak device and on the CPU), then stage D's training (multi_train)
if [ $D_START = c ]; then
    START=(--start-campaign $C --start-round $((C_ROUNDS - 1)))
else
    START=(--traffic-hidden 64 --traffic-heads 4)
fi
profile() {   # $1 name, $2 device of the speaking workers, $3 workers
    local out=$P/multi/$1 where=()
    [ $2 = cpu ] && where=(--speak-device cpu)
    if [ -f $out/profile.json ] && grep -q finished_utc $out/profile.json; then log "$1: done before"; return 0; fi
    [ -e $out ] && mv $out $out.aborted-$(date -u +%Y%m%dT%H%M%SZ)
    log "$1: multi_profile"
    OMP_NUM_THREADS=1 python run_ts.py multi_profile "${INPUTS[@]}" "${D_SETTINGS[@]}" "${START[@]}" \
        --speak-workers $3 "${where[@]}" --out $out >> $out.log 2>&1 || stop "$1 failed (log $out.log)"
    log "$1: done"
}
if [ -f $D/round_$((D_ROUNDS - 1))/checkpoint.pt ]; then
    log "D: done before"
else
    profile profile_gpu_$TAG cuda $SPEAK_WORKERS
    profile profile_cpu_$TAG cpu $CPU_WORKERS
    log "D: multi_train -> $D"
    OMP_NUM_THREADS=1 python run_ts.py multi_train "${INPUTS[@]}" "${D_SETTINGS[@]}" "${START[@]}" --out $D \
        --speak-workers $SPEAK_WORKERS --profile $P/multi/profile_gpu_$TAG --cpu-workers $CPU_WORKERS \
        --cpu-profile $P/multi/profile_cpu_$TAG >> $D.log 2>&1 || stop "D failed (log $D.log)"
    log "D: done"
fi
log "every stage done"
