#!/usr/bin/env bash
# Re-fetch every PDF in papers/ (gitignored). Runs the three per-cluster scripts, each of
# which skips files that already exist. Three papers have no open copy — see README.md.
set -u
cd "$(dirname "$0")"
for s in download_A.sh download_B.sh download_C.sh; do bash "$s"; done
