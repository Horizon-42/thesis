#!/usr/bin/env bash
# Read-only pull of the closed-loop labelled data from SP-AI. Nothing is written on the remote:
# rsync only reads there, niced and idle-IO-class, bandwidth-capped.
# usage: pull.sh [--dry-run]
set -euo pipefail
DRY="${1:-}"
REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)"   # the repository this archive sits in (README §2)
R=supercomputing@192.168.1.103:studys/thesis
IL=4dTrajectory/outputs/POOLED/instruction_language/v12_20261005
EX=4dTrajectory/outputs/POOLED/executor/v17_20261005

COMMON=(-rt --no-perms --chmod=Du+rwx,Fu+rw --bwlimit=20000 --partial --itemize-changes --stats
        --rsync-path="nice -n 19 ionice -c3 rsync" -e "ssh -o BatchMode=yes")
[[ "$DRY" == "--dry-run" ]] && COMMON+=(-n)

mkdir -p "$REPO/$IL" "$REPO/$EX"

echo "== instruction artefact: train split + 2 s closed loop (no select, no val)"
rsync "${COMMON[@]}" \
  --include='/candidates.json' --include='/spec.json' --include='/measurements.json' \
  --include='/signals.json' --include='/labels.json' \
  --include='/signals_train.npz' --include='/sentences_train.npz' \
  --include='/conformance/' --include='/conformance/*' \
  --include='/closed_loop/' --include='/closed_loop/summary.json' --include='/closed_loop/train_2s.npz' \
  --include='/closed_loop/conformance/' --include='/closed_loop/conformance/*' \
  --exclude='*' "$R/$IL/" "$REPO/$IL/"

echo "== executor spec (no replay outputs)"
rsync "${COMMON[@]}" \
  --include='/spec.json' --include='/measurements.json' \
  --include='/conformance/' --include='/conformance/*' \
  --exclude='*' "$R/$EX/" "$REPO/$EX/"

for A in KMSY KRDU KSJC KSMF KSTL; do
  echo "== arrivals manifest $A"
  mkdir -p "$REPO/trajectory_data_process/outputs/harvest/$A/arrivals"
  rsync "${COMMON[@]}" "$R/trajectory_data_process/outputs/harvest/$A/arrivals/" \
        "$REPO/trajectory_data_process/outputs/harvest/$A/arrivals/"
done
